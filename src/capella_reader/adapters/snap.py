"""SNAP conversion utilities for Capella SLC data.

This module writes a Capella SLC as a BEAM-DIMAP product (a `.dim` header and
a `.data` directory), the native format of the ESA Sentinel Application
Platform (SNAP). SNAP is not required to be installed: the product is written
with `numpy` and `tifffile` only.

SNAP has a Capella reader, but its metadata is not sufficient for
interferometry. The reader does not set the range sampling rate, the
bandwidths, or the Doppler centroid coefficients. It sets the radar PRF where
SNAP needs the image line rate, and it writes a constant incidence angle grid,
from which SNAP concludes that near range is on the right. The product written
here sets all of these from the Capella metadata.

Spotlight phase
---------------
A Capella stripmap SLC has the usual zero-Doppler phase: a target at slant
range `R0` has the phase ``-4 pi / lambda * R0``. A Capella spotlight SLC is
basebanded. Its phase is referenced to one antenna position `A` (the
`reference_antenna_position`) and one target `S` (the
`reference_target_position`):

    phi = -4 pi / lambda * (|A - T| - |A - S|)

where `T` is the target position. SNAP removes the flat-earth and topographic
phase of the zero-Doppler convention, so `write_dimap` multiplies spotlight
data by ``exp(1j * spotlight_phase(...))`` to convert it. The distance
``|A - T|`` is almost the same for all points on a range-Doppler circle, so
the conversion needs no DEM: `T` is taken on the raised WGS84 ellipsoid that
Capella focused the image on. For 100 m of relief the error is about 0.03 rad
at the edge of a 5 km scene.

After the conversion, the azimuth spectrum of a spotlight SLC is not at
baseband: its center changes linearly along azimuth, by about 0.2 of the line
rate across a 5 km scene.

Only SLCs with the `slant_plane` image geometry are supported. Spotlight SLCs
with the `pfa` geometry raise `CapellaImageGeometryError`.

Functions
---------
abstracted_metadata
    SNAP "Abstracted Metadata" values for a Capella SLC
geolocate
    ECEF positions of image pixels on the raised WGS84 ellipsoid
spotlight_phase
    Phase that converts spotlight data to the zero-Doppler convention
write_dimap
    Write the SLC as a BEAM-DIMAP product

"""

from __future__ import annotations

import datetime
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tifffile
from numpy.polynomial.polynomial import polygrid2d, polyvander2d
from numpy.typing import ArrayLike, NDArray

from capella_reader.enums import LookSide
from capella_reader.slc import CapellaSLC, _get_suffix, _open_file

__all__ = [
    "abstracted_metadata",
    "geolocate",
    "spotlight_phase",
    "write_dimap",
]

C_LIGHT = 299792458.0
WGS84_SEMI_MAJOR = 6378137.0
WGS84_E2 = 6.69437999014e-3

# The SNAP value for "no metadata" in a numeric attribute
_NO_METADATA = 99999
_NO_METADATA_UTC = "01-JAN-2000 00:00:00.000000"
_MONTHS = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)

_INFLATED_WGS84 = re.compile(r"ExplicitInflatedWGS84\[([^\]]+)\]")

# Degree of the 2D polynomial fitted to the spotlight range term, and the
# largest residual (meters) accepted. With `trajd` state vectors the residual
# is about 2e-6 m. `real_time` state vectors scatter by 1e-4 m, which the
# polynomial does not follow. 1e-3 m (0.4 rad at X band) only stops a fit
# that failed.
_SPOTLIGHT_POLY_DEGREE = 5
_SPOTLIGHT_POLY_TOLERANCE_M = 1e-3

# (name, type, unit) of the SNAP abstracted metadata attributes, in SNAP order
_ABSTRACTED_ATTRIBUTES: tuple[tuple[str, str, str], ...] = (
    ("PRODUCT", "ascii", ""),
    ("PRODUCT_TYPE", "ascii", ""),
    ("SPH_DESCRIPTOR", "ascii", ""),
    ("MISSION", "ascii", ""),
    ("ACQUISITION_MODE", "ascii", ""),
    ("antenna_pointing", "ascii", ""),
    ("BEAMS", "ascii", ""),
    ("SWATH", "ascii", ""),
    ("PROC_TIME", "utc", "utc"),
    ("Processing_system_identifier", "ascii", ""),
    ("orbit_cycle", "int32", ""),
    ("REL_ORBIT", "int32", ""),
    ("ABS_ORBIT", "int32", ""),
    ("STATE_VECTOR_TIME", "utc", "utc"),
    ("VECTOR_SOURCE", "ascii", ""),
    ("incidence_near", "float64", "deg"),
    ("incidence_far", "float64", "deg"),
    ("slice_num", "int32", ""),
    ("data_take_id", "int32", ""),
    ("first_line_time", "utc", "utc"),
    ("last_line_time", "utc", "utc"),
    ("first_near_lat", "float64", "deg"),
    ("first_near_long", "float64", "deg"),
    ("first_far_lat", "float64", "deg"),
    ("first_far_long", "float64", "deg"),
    ("last_near_lat", "float64", "deg"),
    ("last_near_long", "float64", "deg"),
    ("last_far_lat", "float64", "deg"),
    ("last_far_long", "float64", "deg"),
    ("PASS", "ascii", ""),
    ("SAMPLE_TYPE", "ascii", ""),
    ("mds1_tx_rx_polar", "ascii", ""),
    ("mds2_tx_rx_polar", "ascii", ""),
    ("mds3_tx_rx_polar", "ascii", ""),
    ("mds4_tx_rx_polar", "ascii", ""),
    ("polsar_data", "uint8", "flag"),
    ("algorithm", "ascii", ""),
    ("azimuth_looks", "float64", ""),
    ("range_looks", "float64", ""),
    ("range_spacing", "float64", "m"),
    ("azimuth_spacing", "float64", "m"),
    ("range_window_type", "ascii", ""),
    ("range_window_coefficient", "float64", ""),
    ("pulse_repetition_frequency", "float64", "Hz"),
    ("radar_frequency", "float64", "MHz"),
    ("line_time_interval", "float64", "s"),
    ("total_size", "uint32", "MB"),
    ("num_output_lines", "uint32", "lines"),
    ("num_samples_per_line", "uint32", "samples"),
    ("subset_offset_x", "uint32", "samples"),
    ("subset_offset_y", "uint32", "samples"),
    ("srgr_flag", "uint8", "flag"),
    ("avg_scene_height", "float64", "m"),
    ("map_projection", "ascii", ""),
    ("is_terrain_corrected", "uint8", "flag"),
    ("DEM", "ascii", ""),
    ("geo_ref_system", "ascii", ""),
    ("lat_pixel_res", "float64", "deg"),
    ("lon_pixel_res", "float64", "deg"),
    ("slant_range_to_first_pixel", "float64", "m"),
    ("ant_elev_corr_flag", "uint8", "flag"),
    ("range_spread_comp_flag", "uint8", "flag"),
    ("replica_power_corr_flag", "uint8", "flag"),
    ("abs_calibration_flag", "uint8", "flag"),
    ("calibration_factor", "float64", "dB"),
    ("chirp_power", "float64", ""),
    ("inc_angle_comp_flag", "uint8", "flag"),
    ("ref_inc_angle", "float64", ""),
    ("ref_slant_range", "float64", ""),
    ("ref_slant_range_exp", "float64", ""),
    ("rescaling_factor", "float64", ""),
    ("bistatic_correction_applied", "uint8", "flag"),
    ("range_sampling_rate", "float64", "MHz"),
    ("range_bandwidth", "float64", "MHz"),
    ("azimuth_bandwidth", "float64", "Hz"),
    ("multilook_flag", "uint8", "flag"),
    ("coregistered_stack", "uint8", "flag"),
    ("external_calibration_file", "ascii", ""),
    ("orbit_state_vector_file", "ascii", ""),
    ("metadata_version", "ascii", ""),
    ("centre_lat", "float64", "deg"),
    ("centre_lon", "float64", "deg"),
    ("centre_heading", "float64", "deg"),
    ("centre_heading2", "float64", "deg"),
)


class _TiePoints(NamedTuple):
    """Tie-point grids of a product, with their pixel coordinates."""

    rows: NDArray[np.floating]
    cols: NDArray[np.floating]
    latitude: NDArray[np.floating]
    longitude: NDArray[np.floating]
    incident_angle: NDArray[np.floating]
    slant_range_time: NDArray[np.floating]


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------


class _Orbit:
    """Cubic Hermite interpolation of the SLC state vectors."""

    def __init__(self, slc: CapellaSLC) -> None:
        times, self.positions, self.velocities = slc.collect.state.get_state(
            time_as_float=False
        )
        self.epoch: np.datetime64 = times[0]
        self.times = (times - self.epoch) / np.timedelta64(1, "s")

    def seconds(self, time: np.datetime64) -> float:
        """Convert a UTC time to seconds since the first state vector."""
        return float((time - self.epoch) / np.timedelta64(1, "ns")) * 1e-9

    def __call__(
        self, t: ArrayLike
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Interpolate position and velocity at `t` seconds since the epoch."""
        t = np.asarray(t, dtype=float)
        i = np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 2)
        h = (self.times[i + 1] - self.times[i])[..., None]
        s = ((t - self.times[i]) / h[..., 0])[..., None]
        p0, p1 = self.positions[i], self.positions[i + 1]
        v0, v1 = self.velocities[i] * h, self.velocities[i + 1] * h
        position = (
            (2 * s**3 - 3 * s**2 + 1) * p0
            + (s**3 - 2 * s**2 + s) * v0
            + (-2 * s**3 + 3 * s**2) * p1
            + (s**3 - s**2) * v1
        )
        velocity = (
            (6 * s**2 - 6 * s) * p0
            + (3 * s**2 - 4 * s + 1) * v0
            + (-6 * s**2 + 6 * s) * p1
            + (3 * s**2 - 2 * s) * v1
        ) / h
        return position, velocity


def _ecef_to_llh(
    xyz: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Convert ECEF positions to WGS84 latitude, longitude (radians), height."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        n = WGS84_SEMI_MAJOR / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        height = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + height)))
    n = WGS84_SEMI_MAJOR / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return lat, lon, p / np.cos(lat) - n


def _unit(v: NDArray[np.floating]) -> NDArray[np.floating]:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _reference_height(slc: CapellaSLC) -> float:
    """Return the WGS84 height of the surface that Capella focused the image on.

    This is the height of the `ExplicitInflatedWGS84[<height>]` focusing
    terrain model. For other terrain models, it is the height of the scene
    center target.
    """
    terrain_models = slc.collect.image.terrain_models
    if terrain_models is not None and terrain_models.focusing is not None:
        match = _INFLATED_WGS84.match(terrain_models.focusing.name)
        if match:
            return float(match.group(1))
    target = slc.collect.image.center_pixel.target_position.as_array()
    return float(_ecef_to_llh(target)[2])


def geolocate(
    slc: CapellaSLC,
    rows: ArrayLike,
    cols: ArrayLike,
    *,
    height: float | None = None,
) -> NDArray[np.floating]:
    """Find the ECEF positions of image pixels on the raised WGS84 ellipsoid.

    Solves the zero-Doppler range equations for a surface at a constant
    height above the WGS84 ellipsoid.

    Parameters
    ----------
    slc : CapellaSLC
        Capella SLC with slant-plane image geometry.
    rows, cols : ArrayLike
        Pixel coordinates (0 is the center of the first pixel). The two arrays
        are broadcast against each other.
    height : float, optional
        Height of the surface above the WGS84 ellipsoid in meters.
        Default is the height of the surface that Capella focused the image
        on (the `ExplicitInflatedWGS84` focusing terrain model).

    Returns
    -------
    NDArray[np.floating]
        ECEF positions in meters, with shape ``(..., 3)``.

    """
    if height is None:
        height = _reference_height(slc)
    rows, cols = np.broadcast_arrays(
        np.asarray(rows, dtype=float), np.asarray(cols, dtype=float)
    )
    orbit = _Orbit(slc)
    t = orbit.seconds(slc.first_line_time.as_numpy()) + rows * slc.delta_line_time
    slant_range = (slc.range_to_first_sample + cols * slc.delta_range_sample)[..., None]
    position, velocity = orbit(t)

    # Basis of the zero-Doppler plane: `down` points to the Earth, `side`
    # points to the look side.
    along = _unit(velocity)
    down = _unit(-position + np.sum(position * along, axis=-1, keepdims=True) * along)
    side = np.cross(down, along)
    if slc.collect.radar.pointing == LookSide.LEFT:
        side = -side

    # The height of the target increases with the look angle from `down`
    low = np.zeros(rows.shape)
    high = np.full(rows.shape, np.pi / 2)
    for _ in range(60):
        mid = 0.5 * (low + high)[..., None]
        target = position + slant_range * (np.cos(mid) * down + np.sin(mid) * side)
        below = _ecef_to_llh(target)[2] < height
        low = np.where(below, mid[..., 0], low)
        high = np.where(below, high, mid[..., 0])
    look = (0.5 * (low + high))[..., None]
    return position + slant_range * (np.cos(look) * down + np.sin(look) * side)


def _tie_points(slc: CapellaSLC, shape: tuple[int, int]) -> _TiePoints:
    """Compute the SNAP tie-point grids on the raised ellipsoid."""
    n_rows, n_cols = slc.shape
    rows = np.linspace(0, n_rows - 1, shape[0])
    cols = np.linspace(0, n_cols - 1, shape[1])
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    target = geolocate(slc, rr, cc)
    lat, lon, _ = _ecef_to_llh(target)

    orbit = _Orbit(slc)
    t = orbit.seconds(slc.first_line_time.as_numpy()) + rr * slc.delta_line_time
    to_sensor = _unit(orbit(t)[0] - target)
    up = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )
    incidence = np.degrees(np.arccos(np.sum(up * to_sensor, axis=-1)))

    slant_range = slc.range_to_first_sample + cc * slc.delta_range_sample
    return _TiePoints(
        rows=rows,
        cols=cols,
        latitude=np.degrees(lat),
        longitude=np.degrees(lon),
        incident_angle=incidence,
        # two-way time in nanoseconds
        slant_range_time=2 * slant_range / C_LIGHT * 1e9,
    )


# -----------------------------------------------------------------------------
# Spotlight phase
# -----------------------------------------------------------------------------


def _is_spotlight(slc: CapellaSLC) -> bool:
    return slc.collect.mode.lower() == "spotlight"


def _spotlight_range(
    slc: CapellaSLC, rows: ArrayLike, cols: ArrayLike, height: float | None
) -> NDArray[np.floating]:
    """Return ``|A - T| - R0`` in meters for each pixel."""
    antenna = slc.collect.image.reference_antenna_position
    if antenna is None:
        msg = "The SLC metadata has no reference_antenna_position"
        raise ValueError(msg)
    target = geolocate(slc, rows, cols, height=height)
    slant_range = (
        slc.range_to_first_sample
        + np.asarray(cols, dtype=float) * slc.delta_range_sample
    )
    return np.linalg.norm(target - antenna.as_array(), axis=-1) - slant_range


def _spotlight_reference_phase(slc: CapellaSLC) -> float:
    """Return the constant part of the spotlight phase, wrapped to one cycle."""
    image = slc.collect.image
    if image.reference_antenna_position is None:
        msg = "The SLC metadata has no reference_antenna_position"
        raise ValueError(msg)
    if image.reference_target_position is None:
        msg = "The SLC metadata has no reference_target_position"
        raise ValueError(msg)
    reference_range = (
        image.reference_antenna_position - image.reference_target_position
    ).norm()
    return float(-np.mod(4 * np.pi / slc.wavelength * reference_range, 2 * np.pi))


def spotlight_phase(
    slc: CapellaSLC,
    rows: ArrayLike,
    cols: ArrayLike,
    *,
    height: float | None = None,
) -> NDArray[np.floating]:
    """Compute the phase that converts spotlight data to zero-Doppler phase.

    Multiply the SLC data by ``exp(1j * phase)`` to get the phase convention
    of a stripmap SLC, ``-4 pi / lambda * R0``. See the module docstring.

    Parameters
    ----------
    slc : CapellaSLC
        Capella spotlight SLC.
    rows, cols : ArrayLike
        Pixel coordinates (0 is the center of the first pixel). The two arrays
        are broadcast against each other.
    height : float, optional
        Height above the WGS84 ellipsoid, in meters, at which the pixels are
        located. Default is the height of the surface that Capella focused
        the image on.

    Returns
    -------
    NDArray[np.floating]
        Phase in radians. The constant part is wrapped to one cycle.

    Raises
    ------
    ValueError
        If the metadata has no reference antenna or target position.

    """
    rows, cols = np.broadcast_arrays(
        np.asarray(rows, dtype=float), np.asarray(cols, dtype=float)
    )
    range_term = _spotlight_range(slc, rows, cols, height)
    return 4 * np.pi / slc.wavelength * range_term + _spotlight_reference_phase(slc)


class _SpotlightPhaseModel:
    """Polynomial model of `spotlight_phase`, fast to evaluate on all pixels."""

    def __init__(self, slc: CapellaSLC) -> None:
        self.shape = slc.shape
        degree = _SPOTLIGHT_POLY_DEGREE
        self.scale = 4 * np.pi / slc.wavelength
        self.constant = _spotlight_reference_phase(slc)

        # The fit and the check use different sample positions
        n_fit = 4 * degree + 1
        fit_values, fit_matrix = self._samples(slc, n_fit)
        solution = np.linalg.lstsq(fit_matrix, fit_values, rcond=None)[0]
        self.coefficients = solution.reshape(degree + 1, degree + 1)
        check_values, check_matrix = self._samples(slc, n_fit + 6)
        residual = check_matrix @ self.coefficients.ravel() - check_values
        self.max_residual = float(np.abs(residual).max())
        if self.max_residual > _SPOTLIGHT_POLY_TOLERANCE_M:
            msg = (
                "Polynomial model of the spotlight phase has a residual of "
                f"{self.max_residual:.2e} m"
            )
            raise ValueError(msg)

    def _normalize(
        self, rows: NDArray[np.floating], cols: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        return 2 * rows / (self.shape[0] - 1) - 1, 2 * cols / (self.shape[1] - 1) - 1

    def _samples(
        self, slc: CapellaSLC, n: int
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        rows = np.linspace(0, self.shape[0] - 1, n)
        cols = np.linspace(0, self.shape[1] - 1, n)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        values = _spotlight_range(slc, rr, cc, None).ravel()
        x, y = self._normalize(rr.ravel(), cc.ravel())
        degree = _SPOTLIGHT_POLY_DEGREE
        return values, polyvander2d(x, y, [degree, degree])

    def __call__(self, row_start: int, row_stop: int) -> NDArray[np.floating]:
        """Evaluate the phase for the full-width block of rows."""
        x, y = self._normalize(
            np.arange(row_start, row_stop, dtype=float),
            np.arange(self.shape[1], dtype=float),
        )
        return self.scale * polygrid2d(x, y, self.coefficients) + self.constant


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------


def _format_utc(time: np.datetime64) -> str:
    """Format a time as SNAP does: ``21-SEP-2025 04:33:59.151110``."""
    # SNAP keeps microseconds: round, do not truncate
    rounded = (time.astype("datetime64[ns]") + np.timedelta64(500, "ns")).astype(
        "datetime64[us]"
    )
    date, clock = str(rounded).split("T")
    year, month, day = date.split("-")
    return f"{day}-{_MONTHS[int(month) - 1]}-{year} {clock}"


def _to_datetime64(time: datetime.datetime) -> np.datetime64:
    """Convert a datetime (UTC if it has no time zone) to numpy."""
    if time.tzinfo is not None:
        time = time.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return np.datetime64(time, "ns")


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the initial bearing in degrees from point 1 to point 2."""
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float(np.degrees(np.arctan2(y, x)) % 360)


def abstracted_metadata(slc: CapellaSLC) -> dict[str, str | int | float]:
    """Compute the SNAP "Abstracted Metadata" values for a Capella SLC.

    The names and units are those of `AbstractMetadata` in SNAP. Attributes
    that have no Capella equivalent are left out; `write_dimap` writes the
    SNAP "no metadata" value for them.

    Parameters
    ----------
    slc : CapellaSLC
        Capella SLC with slant-plane image geometry.

    Returns
    -------
    dict[str, str | int | float]
        Attribute values by SNAP attribute name. Times are strings in the
        SNAP format, for example ``21-SEP-2025 04:33:59.151110``.

    """
    image = slc.collect.image
    radar = slc.collect.radar
    state = slc.collect.state
    n_rows, n_cols = slc.shape
    name = Path(slc.path).name
    for suffix in (".tif", ".tiff", "_extended.json", ".json"):
        name = name.removesuffix(suffix)

    first_line = slc.first_line_time.as_numpy()
    last_line_ns = round((n_rows - 1) * slc.delta_line_time * 1e9)
    last_line = first_line + np.timedelta64(last_line_ns, "ns")

    # Corner and center geolocation on the raised ellipsoid
    center_row, center_col = (n_rows - 1) / 2, (n_cols - 1) / 2
    points = _tie_points(slc, (3, 3))
    lat, lon = points.latitude, points.longitude
    ahead = geolocate(slc, min(center_row + 100, n_rows - 1), center_col)
    ahead_lat, ahead_lon, _ = _ecef_to_llh(ahead)
    heading = _bearing(
        lat[1, 1], lon[1, 1], float(np.degrees(ahead_lat)), float(np.degrees(ahead_lon))
    )

    return {
        "PRODUCT": name,
        "PRODUCT_TYPE": "SLC",
        "SPH_DESCRIPTOR": "Single Look Complex",
        "MISSION": "Capella",
        "ACQUISITION_MODE": slc.collect.mode,
        "antenna_pointing": radar.pointing.value,
        "PROC_TIME": _format_utc(_to_datetime64(slc.meta.processing_time)),
        "Processing_system_identifier": slc.meta.software_version,
        "STATE_VECTOR_TIME": _format_utc(state.state_vectors[0].time.as_numpy()),
        "VECTOR_SOURCE": state.source,
        "incidence_near": float(points.incident_angle[1, 0]),
        "incidence_far": float(points.incident_angle[1, -1]),
        "first_line_time": _format_utc(first_line),
        "last_line_time": _format_utc(last_line),
        "first_near_lat": float(lat[0, 0]),
        "first_near_long": float(lon[0, 0]),
        "first_far_lat": float(lat[0, -1]),
        "first_far_long": float(lon[0, -1]),
        "last_near_lat": float(lat[-1, 0]),
        "last_near_long": float(lon[-1, 0]),
        "last_far_lat": float(lat[-1, -1]),
        "last_far_long": float(lon[-1, -1]),
        "PASS": state.direction.upper(),
        "SAMPLE_TYPE": "COMPLEX",
        "mds1_tx_rx_polar": slc.polarization,
        "polsar_data": 0,
        "algorithm": image.algorithm,
        "azimuth_looks": image.azimuth_looks,
        "range_looks": image.range_looks,
        "range_spacing": slc.delta_range_sample,
        "azimuth_spacing": image.pixel_spacing_row,
        "range_window_type": image.range_window.name,
        # SNAP uses this value as the azimuth sampling rate of the image,
        # which for a Capella SLC is the line rate, not the radar PRF.
        "pulse_repetition_frequency": 1 / slc.delta_line_time,
        "radar_frequency": slc.center_frequency / 1e6,
        "line_time_interval": slc.delta_line_time,
        "total_size": round(n_rows * n_cols * 8 / 1e6),
        "num_output_lines": n_rows,
        "num_samples_per_line": n_cols,
        "subset_offset_x": 0,
        "subset_offset_y": 0,
        "srgr_flag": 0,
        "avg_scene_height": _reference_height(slc),
        "is_terrain_corrected": 0,
        "slant_range_to_first_pixel": slc.range_to_first_sample,
        "ant_elev_corr_flag": int(bool(image.elevation_beam_pattern_corrected)),
        "range_spread_comp_flag": 0,
        "replica_power_corr_flag": 0,
        "abs_calibration_flag": int(image.radiometry == "sigma_nought"),
        "calibration_factor": image.scale_factor,
        "inc_angle_comp_flag": 0,
        # Capella line times are zero-Doppler times
        "bistatic_correction_applied": 1,
        # Sampling rate of the slant range grid, not of the receiver
        "range_sampling_rate": C_LIGHT / (2 * slc.delta_range_sample) / 1e6,
        "range_bandwidth": (image.processed_range_bandwidth or 0.0) / 1e6,
        "azimuth_bandwidth": image.processed_azimuth_bandwidth,
        "multilook_flag": 0,
        "coregistered_stack": 0,
        "metadata_version": "7.0",
        "centre_lat": float(lat[1, 1]),
        "centre_lon": float(lon[1, 1]),
        "centre_heading": heading,
        "centre_heading2": (heading + 180) % 360,
    }


# -----------------------------------------------------------------------------
# DIMAP writing
# -----------------------------------------------------------------------------


def _attribute(
    parent: ET.Element, name: str, kind: str, value: str | float, unit: str = ""
) -> None:
    attrib = {"name": name, "type": kind, "mode": "rw"}
    if unit:
        attrib["unit"] = unit
    ET.SubElement(parent, "MDATTR", attrib).text = (
        value if isinstance(value, str) else repr(value)
    )


def _element(parent: ET.Element, name: str) -> ET.Element:
    return ET.SubElement(parent, "MDElem", {"name": name})


def _text(parent: ET.Element, tag: str, text: object = None) -> ET.Element:
    child = ET.SubElement(parent, tag)
    if text is not None:
        child.text = str(text)
    return child


def _abstracted_metadata_element(parent: ET.Element, slc: CapellaSLC) -> None:
    values = abstracted_metadata(slc)
    root = _element(parent, "Abstracted_Metadata")
    defaults: dict[str, str | int | float] = {
        "ascii": "-",
        "utc": _NO_METADATA_UTC,
        "int32": _NO_METADATA,
        "uint32": _NO_METADATA,
        "uint8": 0,
        "float64": float(_NO_METADATA),
    }
    for name, kind, unit in _ABSTRACTED_ATTRIBUTES:
        value = values.get(name, defaults[kind])
        if kind == "float64":
            value = float(value)
        _attribute(root, name, kind, value, unit)

    vectors = _element(root, "Orbit_State_Vectors")
    for i, sv in enumerate(slc.collect.state.state_vectors, start=1):
        vector = _element(vectors, f"orbit_vector{i}")
        _attribute(vector, "time", "utc", _format_utc(sv.time.as_numpy()))
        for axis, value in zip("xyz", sv.position.as_array(), strict=True):
            _attribute(vector, f"{axis}_pos", "float64", float(value))
        for axis, value in zip("xyz", sv.velocity.as_array(), strict=True):
            _attribute(vector, f"{axis}_vel", "float64", float(value))

    _element(root, "SRGR_Coefficients")

    # The image grid is zero Doppler: one list entry with zero coefficients.
    # SNAP stops with an error if there is no entry.
    coefficients = _element(root, "Doppler_Centroid_Coefficients")
    doppler = _element(coefficients, "dop_coef_list.1")
    _attribute(doppler, "zero_doppler_time", "utc", str(values["first_line_time"]))
    _attribute(
        doppler,
        "slant_range_time",
        "float64",
        2 * slc.range_to_first_sample / C_LIGHT * 1e9,
        "ns",
    )
    for i in range(1, 4):
        _attribute(_element(doppler, f"coefficient.{i}"), "dop_coef", "float64", 0.0)


def _envi_header(path: Path, name: str, shape: tuple[int, int], unit: str) -> None:
    path.write_text(
        "ENVI\n"
        f"description = {{{name} - Unit: {unit}}}\n"
        f"samples = {shape[1]}\n"
        f"lines = {shape[0]}\n"
        "bands = 1\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 4\n"
        "interleave = bsq\n"
        "byte order = 1\n"
        f"band names = {{ {name} }}\n"
    )


def _band_info(
    parent: ET.Element,
    index: int,
    name: str,
    shape: tuple[int, int],
    unit: str,
    expression: str | None = None,
) -> None:
    info = ET.SubElement(parent, "Spectral_Band_Info")
    _text(info, "BAND_INDEX", index)
    _text(info, "BAND_DESCRIPTION")
    _text(info, "BAND_NAME", name)
    _text(info, "BAND_RASTER_WIDTH", shape[1])
    _text(info, "BAND_RASTER_HEIGHT", shape[0])
    _text(info, "DATA_TYPE", "float32")
    _text(info, "PHYSICAL_UNIT", unit)
    _text(info, "SOLAR_FLUX", 0.0)
    _text(info, "BAND_WAVELEN", 0.0)
    _text(info, "BAND_ANGULAR_VALUE", -999.0)
    _text(info, "BANDWIDTH", 0.0)
    _text(info, "SCALING_FACTOR", 1.0)
    _text(info, "SCALING_OFFSET", 0.0)
    _text(info, "LOG10_SCALED", "false")
    _text(info, "NO_DATA_VALUE_USED", "true")
    _text(info, "NO_DATA_VALUE", 0.0)
    if expression is not None:
        _text(info, "VIRTUAL_BAND", "true")
        _text(info, "EXPRESSION", expression)


def _dimap_document(
    slc: CapellaSLC,
    product_name: str,
    points: _TiePoints,
    *,
    zero_doppler_phase: bool,
) -> ET.Element:
    shape = slc.shape
    pol = slc.polarization
    data_dir = f"{product_name}.data"
    grids = {
        "latitude": "deg",
        "longitude": "deg",
        "incident_angle": "deg",
        "slant_range_time": "ns",
    }
    metadata = abstracted_metadata(slc)

    root = ET.Element("Dimap_Document", {"name": f"{product_name}.dim"})
    ident = _text(root, "Metadata_Id")
    ET.SubElement(ident, "METADATA_FORMAT", {"version": "2.13.0"}).text = "DIMAP"
    _text(ident, "METADATA_PROFILE", "BEAM-DATAMODEL-V1")
    dataset = _text(root, "Dataset_Id")
    _text(dataset, "DATASET_SERIES", "BEAM-PRODUCT")
    _text(dataset, "DATASET_NAME", product_name)
    _text(_text(root, "Dataset_Use"), "DATASET_COMMENTS", "Single Look Complex")
    production = _text(root, "Production")
    _text(production, "DATASET_PRODUCER_NAME")
    _text(production, "PRODUCT_TYPE", "SLC")
    _text(production, "PRODUCT_SCENE_RASTER_START_TIME", metadata["first_line_time"])
    _text(production, "PRODUCT_SCENE_RASTER_STOP_TIME", metadata["last_line_time"])
    _text(production, "QUICKLOOK_BAND_NAME", f"Intensity_{pol}")

    crs = _text(_text(root, "Coordinate_Reference_System"), "Horizontal_CS")
    _text(crs, "HORIZONTAL_CS_TYPE", "GEOGRAPHIC")
    datum = _text(_text(crs, "Geographic_CS"), "Horizontal_Datum")
    _text(datum, "HORIZONTAL_DATUM_NAME", "WGS-84")
    ellipsoid = _text(datum, "Ellipsoid")
    _text(ellipsoid, "ELLIPSOID_NAME", "WGS-84")
    parameters = _text(ellipsoid, "Ellipsoid_Parameters")
    ET.SubElement(parameters, "ELLIPSOID_MAJ_AXIS", {"unit": "M"}).text = "6378137.0"
    ET.SubElement(parameters, "ELLIPSOID_MIN_AXIS", {"unit": "M"}).text = "6356752.3"

    geoposition = _text(_text(root, "Geoposition"), "Geoposition_Points")
    _text(geoposition, "TIE_POINT_GRID_NAME_LAT", "latitude")
    _text(geoposition, "TIE_POINT_GRID_NAME_LON", "longitude")

    dimensions = _text(root, "Raster_Dimensions")
    _text(dimensions, "NCOLS", shape[1])
    _text(dimensions, "NROWS", shape[0])
    _text(dimensions, "NBANDS", 3)

    access = _text(root, "Data_Access")
    _text(access, "DATA_FILE_FORMAT", "ENVI")
    _text(access, "DATA_FILE_FORMAT_DESC", "ENVI File Format")
    _text(access, "DATA_FILE_ORGANISATION", "BAND_SEPARATE")
    for index, band in enumerate((f"i_{pol}", f"q_{pol}")):
        data_file = _text(access, "Data_File")
        ET.SubElement(data_file, "DATA_FILE_PATH", {"href": f"{data_dir}/{band}.hdr"})
        _text(data_file, "BAND_INDEX", index)
    for index, grid in enumerate(grids):
        grid_file = _text(access, "Tie_Point_Grid_File")
        ET.SubElement(
            grid_file,
            "TIE_POINT_GRID_FILE_PATH",
            {"href": f"{data_dir}/tie_point_grids/{grid}.hdr"},
        )
        _text(grid_file, "TIE_POINT_GRID_INDEX", index)

    # A tie point is at the center of a pixel, which SNAP places at +0.5
    grid_shape = points.latitude.shape
    grid_infos = _text(root, "Tie_Point_Grids")
    _text(grid_infos, "NUM_TIE_POINT_GRIDS", len(grids))
    for index, (grid, unit) in enumerate(grids.items()):
        info = _text(grid_infos, "Tie_Point_Grid_Info")
        _text(info, "TIE_POINT_GRID_INDEX", index)
        _text(info, "TIE_POINT_DESCRIPTION")
        _text(info, "PHYSICAL_UNIT", unit)
        _text(info, "TIE_POINT_GRID_NAME", grid)
        _text(info, "DATA_TYPE", "float32")
        _text(info, "NCOLS", grid_shape[1])
        _text(info, "NROWS", grid_shape[0])
        _text(info, "OFFSET_X", 0.5)
        _text(info, "OFFSET_Y", 0.5)
        _text(info, "STEP_X", float(points.cols[1] - points.cols[0]))
        _text(info, "STEP_Y", float(points.rows[1] - points.rows[0]))
        ET.SubElement(info, "CYCLIC", {"discontinuity": "0"}).text = "false"

    interpretation = _text(root, "Image_Interpretation")
    _band_info(interpretation, 0, f"i_{pol}", shape, "real")
    _band_info(interpretation, 1, f"q_{pol}", shape, "imaginary")
    _band_info(
        interpretation,
        2,
        f"Intensity_{pol}",
        shape,
        "intensity",
        expression=f"i_{pol} == 0.0 ? 0.0 : i_{pol} * i_{pol} + q_{pol} * q_{pol}",
    )

    sources = _element(_text(root, "Dataset_Sources"), "metadata")
    _abstracted_metadata_element(sources, slc)
    original = _element(_element(sources, "Original_Product_Metadata"), "capella")
    image = slc.collect.image
    _attribute(original, "collect_id", "ascii", slc.collect.collect_id)
    _attribute(original, "platform", "ascii", slc.collect.platform)
    _attribute(original, "radar_prf", "float64", slc.prf_average, "Hz")
    _attribute(original, "zero_doppler_phase_applied", "uint8", int(zero_doppler_phase))
    for key, position in (
        ("reference_antenna_position", image.reference_antenna_position),
        ("reference_target_position", image.reference_target_position),
    ):
        if position is not None:
            text = ",".join(repr(float(v)) for v in position.as_array())
            _attribute(original, key, "ascii", text)
    return root


def _iter_blocks(
    path: str, block_rows: int
) -> Iterator[tuple[int, NDArray[np.complex64]]]:
    """Yield ``(first_row, data)`` blocks of full image width from the TIFF."""
    with _open_file(path) as f, tifffile.TiffFile(f) as tif:
        page = tif.pages[0]
        assert isinstance(page, tifffile.TiffPage)
        n_rows, n_cols = page.shape[:2]
        if page.is_tiled:
            seg_rows, seg_cols = page.tilelength, page.tilewidth
        else:
            seg_rows, seg_cols = page.rowsperstrip, n_cols
        n_across = -(-n_cols // seg_cols)
        step = max(1, block_rows // seg_rows) * seg_rows
        for row in range(0, n_rows, step):
            stop = min(row + step, n_rows)
            block = np.zeros((stop - row, n_cols), dtype=np.complex64)
            for seg_row in range(row, stop, seg_rows):
                for i_across in range(n_across):
                    index = (seg_row // seg_rows) * n_across + i_across
                    if page.databytecounts[index] == 0:
                        continue
                    tif.filehandle.seek(page.dataoffsets[index])
                    raw = tif.filehandle.read(page.databytecounts[index])
                    data, _, seg_shape = page.decode(raw, index)
                    assert data is not None
                    segment = data.reshape(seg_shape[1], seg_shape[2])
                    col = i_across * seg_cols
                    height = min(seg_rows, n_rows - seg_row)
                    width = min(seg_cols, n_cols - col)
                    block[seg_row - row : seg_row - row + height, col : col + width] = (
                        segment[:height, :width]
                    )
            yield row, block


def write_dimap(
    slc: CapellaSLC,
    output: str | Path,
    *,
    zero_doppler_phase: bool = True,
    tie_point_shape: tuple[int, int] = (21, 21),
    block_rows: int = 512,
) -> Path:
    """Write a Capella SLC as a SNAP BEAM-DIMAP product.

    The product has the bands ``i_<pol>`` and ``q_<pol>`` (float32, scaled by
    the Capella `scale_factor`), tie-point grids for latitude, longitude,
    incidence angle and slant range time, and the metadata that the SNAP
    InSAR operators need.

    Parameters
    ----------
    slc : CapellaSLC
        Capella SLC that was read from a GeoTIFF file.
    output : str or Path
        Path of the product header. The suffix is set to ``.dim``; the pixel
        data go to the directory with the same name and the suffix ``.data``.
    zero_doppler_phase : bool, optional
        For a spotlight SLC, convert the phase to the zero-Doppler convention
        with `spotlight_phase`. SNAP interferograms of spotlight data are
        wrong without this. No effect on other modes. Default is True.
    tie_point_shape : tuple[int, int], optional
        Number of tie points in (azimuth, range). Default is (21, 21).
    block_rows : int, optional
        Number of image rows to convert at a time. Default is 512.

    Returns
    -------
    Path
        Path of the `.dim` header.

    Raises
    ------
    ValueError
        If `slc` was read from a JSON file, which has no pixel data.

    Examples
    --------
    >>> from capella_reader import CapellaSLC
    >>> from capella_reader.adapters import snap
    >>> slc = CapellaSLC.from_file("CAPELLA_SP_SLC.tif")  # doctest: +SKIP
    >>> snap.write_dimap(slc, "snap/20250730.dim")  # doctest: +SKIP
    PosixPath('snap/20250730.dim')

    """
    if _get_suffix(slc.path) == ".json":
        msg = "write_dimap needs the SLC GeoTIFF, not the JSON metadata file"
        raise ValueError(msg)

    header = Path(output).with_suffix(".dim")
    product_name = header.stem
    data_dir = header.with_suffix(".data")
    grid_dir = data_dir / "tie_point_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    apply_phase = zero_doppler_phase and _is_spotlight(slc)
    phase_model = _SpotlightPhaseModel(slc) if apply_phase else None

    points = _tie_points(slc, tie_point_shape)
    for name, unit in (
        ("latitude", "deg"),
        ("longitude", "deg"),
        ("incident_angle", "deg"),
        ("slant_range_time", "ns"),
    ):
        grid = getattr(points, name)
        grid.astype(">f4").tofile(grid_dir / f"{name}.img")
        _envi_header(grid_dir / f"{name}.hdr", name, grid.shape, unit)

    pol = slc.polarization
    scale = slc.collect.image.scale_factor
    with (
        open(data_dir / f"i_{pol}.img", "wb") as f_real,
        open(data_dir / f"q_{pol}.img", "wb") as f_imag,
    ):
        for row, block in _iter_blocks(slc.path, block_rows):
            data = block
            if phase_model is not None:
                phase = phase_model(row, row + block.shape[0])
                data = block * np.exp(1j * phase).astype(np.complex64)
            (data.real * scale).astype(">f4").tofile(f_real)
            (data.imag * scale).astype(">f4").tofile(f_imag)
    _envi_header(data_dir / f"i_{pol}.hdr", f"i_{pol}", slc.shape, "real")
    _envi_header(data_dir / f"q_{pol}.hdr", f"q_{pol}", slc.shape, "imaginary")

    document = _dimap_document(
        slc, product_name, points, zero_doppler_phase=apply_phase
    )
    ET.indent(document, space="    ")
    ET.ElementTree(document).write(header, encoding="ISO-8859-1", xml_declaration=True)
    return header
