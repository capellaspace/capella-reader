"""ISCE3 conversion utilities for Capella SLC data.

This module provides functions to convert Capella SLC metadata and data
into ISCE3 data structures for use with the ISCE3 processing library.

Functions
---------
get_radar_grid
    Create ISCE3 `RadarGridParameters` from Capella SLC
get_orbit
    Create ISCE3 `Orbit` from Capella SLC state vectors
get_doppler_poly
    Create ISCE3 `Poly2d` for Doppler centroid frequency
get_doppler_lut2d
    Create ISCE3 `LUT2d` for Doppler centroid frequency
get_attitude
    Create ISCE3 `Attitude` from Capella pointing samples

"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from capella_reader._time import Time
from capella_reader.enums import LookSide
from capella_reader.orbit import PointingSample, StateVector

if TYPE_CHECKING:
    import isce3

    from capella_reader.slc import CapellaSLC


C_LIGHT = 299792458.0


def get_radar_grid(slc: CapellaSLC) -> isce3.product.RadarGridParameters:
    """Create ISCE3 RadarGridParameters from Capella SLC.

    Parameters
    ----------
    slc : CapellaSLC
        Capella SLC object with metadata

    Returns
    -------
    isce3.product.RadarGridParameters
        ISCE3 radar grid parameters

    """
    import isce3

    assert slc.sensing_start is not None
    # Using logic similar to s1-reader to create the radar grid
    sensing_start_seconds = (slc.sensing_start - slc.ref_epoch).total_seconds()
    ref_epoch_isce = isce3.core.DateTime(str(slc.ref_epoch.as_numpy()))

    return isce3.product.RadarGridParameters(
        sensing_start_seconds,
        slc.wavelength,
        # NB: "PRF" is used in isce to make the radar grid's azimuth time spacing.
        #  it is *not* the actual radar PRF.
        # Capella already specifies delta_line_time as the azimuth spacing
        1 / slc.delta_line_time,
        slc.starting_range,
        slc.delta_range_sample,
        isce3.core.LookSide(
            1 if slc.meta.collect.radar.pointing == LookSide.LEFT else -1
        ),
        slc.shape[0],
        slc.shape[1],
        ref_epoch_isce,
    )


def get_orbit(
    state_vectors: Sequence[StateVector], ref_epoch: Time | None = None
) -> isce3.core.Orbit:
    """Create ISCE3 orbit from Capella state vectors.

    Parameters
    ----------
    state_vectors : Sequence[StateVector]
        Sequence of StateVector objects from Capella metadata
    ref_epoch : Time, optional
        Reference epoch to use for the orbit.
        If None, uses the first state vector's time.

    Returns
    -------
    isce3.core.Orbit
        ISCE3 orbit object

    """
    import isce3

    from capella_reader.orbit import interpolate_orbit, is_uniformly_sampled

    if not state_vectors:
        msg = "No state vectors found in Capella metadata"
        raise ValueError(msg)

    # isce3 will throw the following for default collects:
    # ValueError: non-uniform spacing between state vectors encountered ...
    if not is_uniformly_sampled(state_vectors):
        sampled_dts = np.diff([sv.time.as_numpy() for sv in state_vectors])  # type: ignore[arg-type]
        dt_counter = Counter(sampled_dts)
        warnings.warn(
            f"State vectors are not uniformly sampled. Found {dt_counter}."
            " Interpolating to most common spacing.",
            stacklevel=2,
        )
        state_vectors = interpolate_orbit(state_vectors)

    # Use provided reference epoch or default to first state vector time
    if ref_epoch is None:
        ref_epoch = state_vectors[0].time
    ref_epoch_isce = isce3.core.DateTime(str(ref_epoch).strip("Z"))

    orbit_svs: list[isce3.core.StateVector] = [
        isce3.core.StateVector(
            isce3.core.DateTime(str(sv.time).strip("Z")),
            sv.position.as_array(),
            sv.velocity.as_array(),
        )
        for sv in state_vectors
    ]

    return isce3.core.Orbit(orbit_svs, ref_epoch_isce)


def get_attitude(
    pointing_samples: Sequence[PointingSample], ref_epoch: Time | None = None
) -> isce3.core.Attitude:
    """Create ISCE3 attitude from Capella pointing samples.

    Parameters
    ----------
    pointing_samples : Sequence[PointingSample]
        Sequence of PointingSample objects from Capella metadata
    ref_epoch : Time, optional
        Reference epoch to use for the attitude.
        If None, uses the first pointing sample's time.

    Returns
    -------
    isce3.core.Attitude
        ISCE3 attitude object

    Notes
    -----
    Capella quaternions are stored as (q0, q1, q2, q3) = (w, x, y, z) in the
    Hamilton convention (scalar-first), and represent a rotation from the
    coordinate_system frame to the Capella antenna frame.

    ISCE3 `Attitude` expects quaternions that rotate from the antenna frame
    to ECEF.

    """
    import isce3

    if not pointing_samples:
        msg = "No pointing samples found in Capella metadata"
        raise ValueError(msg)

    if ref_epoch is None:
        ref_epoch = pointing_samples[0].time

    # ISCE3 DateTime doesn't like trailing 'Z'
    ref_epoch_isce = isce3.core.DateTime(str(ref_epoch).strip("Z"))

    times = [
        (ps.time.as_numpy() - ref_epoch.as_numpy()) / np.timedelta64(1, "s")
        for ps in pointing_samples
    ]
    # Constant +90 deg rotation about +Z: RCS -> Capella antenna frame
    half_angle = 0.5 * np.deg2rad(90.0)
    q_R2A = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)], dtype=float)

    def quat_mul(q2: np.ndarray, q1: np.ndarray):
        """Hamilton product q = q2 ⊗ q1 (apply q1, then q2)."""
        w2, x2, y2, z2 = q2
        w1, x1, y1, z1 = q1
        return np.array(
            [
                w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
                w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
                w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
                w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
            ]
        )

    quaternions = []
    for ps in pointing_samples:
        # Capella quaternion: ECEF -> Antenna
        # Flip to be Antenna -> ECEF
        q_A2E = ps.attitude.as_array() * np.array([1, -1, -1, -1])

        # Radar coords -> ECEF = (Antenna -> ECEF) ⨂ (Radar coords -> Antenna)
        q_R2E = quat_mul(q_A2E, q_R2A)
        quaternions.append(isce3.core.Quaternion(*q_R2E))

    return isce3.core.Attitude(times, quaternions, ref_epoch_isce)


def get_doppler_poly(slc: CapellaSLC) -> isce3.core.Poly2d:
    """Build an ISCE3 Poly2D for Doppler centroid frequency (Hz).

    Capella polynomial convention:
        f_dc = poly(az_s, range_m)
    where az_s is seconds since first_line_time, and range_m is slant range (m).

    Parameters
    ----------
    slc : CapellaSLC
        Capella SLC object with metadata

    Returns
    -------
    isce3.core.Poly2d
        2D polynomial converted to isce3 object

    """
    import isce3

    poly = slc.frequency_doppler_centroid_polynomial
    return isce3.core.Poly2d(poly.coefficients)


def get_doppler_lut2d(
    slc: CapellaSLC,
    n_az: int = 10,
    method: str = "bilinear",
) -> isce3.core.LUT2d:
    """Build an ISCE3 LUT2d for Doppler centroid frequency (Hz).

    Capella polynomial convention:
        f_dc = poly(az_rel_s, range_m)
    where az_rel_s is seconds since first_line_time.

    ISCE3 LUT2d convention:
        f_dc = doppler_lut.eval(az_s_since_ref_epoch, range_m)

    This method constructs y-coordinates in "seconds since ref_epoch"
    (i.e., the same time basis used by RadarGridParameters.ref_epoch) and
    evaluates the Capella poly using az_rel = y - sensing_start_seconds.

    Parameters
    ----------
    slc : CapellaSLC
        Capella SLC object with metadata
    n_az : int
        Number of knots in azimuth (time) for the LUT.
        Default is 10.
    method : str
        Interpolation method for LUT2d (e.g. 'bilinear', 'bicubic').

    Returns
    -------
    isce3.core.LUT2d
        Doppler centroid LUT2d in Hz.

    """
    import isce3

    n_lines, n_samples = slc.shape

    # x axis: slant range (meters)
    r0 = slc.range_to_first_sample
    dr = slc.delta_range_sample

    # Pad one before and after to avoid OOB errors
    rg_idx = np.arange(-1, n_samples + 1, dtype=int)
    slant_ranges = r0 + rg_idx * dr
    slant_ranges = np.asarray(slant_ranges, dtype=np.float64)

    # y axis: az time in seconds since radar-grid reference epoch
    sensing_start_seconds = (slc.sensing_start - slc.ref_epoch).total_seconds()
    az_idx = np.linspace(-1, n_lines + 1, int(n_az))
    az_rel = az_idx * slc.delta_line_time  # seconds since first line time

    az_times = sensing_start_seconds + az_rel  # seconds since ref_epoch
    az_times = np.asarray(az_times, dtype=np.float64)

    # evaluate Capella poly: f_dc(az_rel, range)
    Y, X = np.meshgrid(az_times, slant_ranges, indexing="ij")
    data = slc.frequency_doppler_centroid_polynomial(Y, X)  # (len(az), len(rg))
    data = np.ascontiguousarray(data, dtype=np.float64)

    # LUT2d wants (xcoord, ycoord, data[y, x])
    return isce3.core.LUT2d(slant_ranges, az_times, data, method=method, b_error=True)
