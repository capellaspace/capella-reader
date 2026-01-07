"""Example of estimating Doppler from Capella attitude data.

Functions started from ISCE3 example:
https://github.com/isce-framework/isce3/blob/5b2d6f6057a706d3ff5cfe7342ced8a7933fb302/python/packages/isce3/geometry/doppler.py

Uses the `get_attitude` in `capella_reader.adapters.isce3`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import isce3
import matplotlib.pyplot as plt
import numpy as np
from isce3.geometry import DEMInterpolator
from numpy.typing import ArrayLike

from capella_reader import CapellaSLC
from capella_reader.adapters.isce3 import get_attitude, get_doppler_lut2d, get_orbit


def los2doppler(look, v, wvl):
    """
    Compute Doppler given line-of-sight vector

    Parameters
    ----------
    look : array_like
        ECEF line of sight vector in m
    v : array_like
        ECEF velocity vector in m/s
    wvl : float
        Radar wavelength in m

    Returns
    -------
    float
        Doppler frequency in Hz
    """
    return 2 / wvl * np.asarray(v).dot(look) / np.linalg.norm(look)


def make_doppler_lut_from_attitude(
    az_time: ArrayLike,
    slant_range: ArrayLike,
    orbit: isce3.core.Orbit,
    attitude: isce3.core.Attitude,
    wavelength: float,
    *,
    dem: DEMInterpolator | None = None,
    epoch: isce3.core.DateTime | None = None,
    az_angle: float = 0.0,
    interp_method: isce3.core.DataInterpMethod | str = "bilinear",
    bounds_error: bool = True,
) -> isce3.core.LUT2d:
    """
    Estimate a 2-D lookup table (LUT) of Doppler centroid values from antenna attitude.

    Parameters
    ----------
    az_time : array_like
        Azimuth time coordinates of the output LUT, in seconds. Values must be uniformly
        spaced. If `epoch` is not None, these coordinates should be specified relative
        to that datetime. Otherwise, they should be relative to the reference epoch of
        `orbit` and `attitude` (which must have the same reference epoch in this case).
    slant_range : array_like
        Slant range coordinates of the output LUT, in meters. Values must be uniformly
        spaced.
    orbit : isce3.core.Orbit
        The path of the antenna phase center over a period spanning the azimuth time
        bounds of the output LUT. If `epoch` is None, this must have the same reference
        epoch as `attitude`.
    attitude : isce3.core.Attitude
        The orientation of the antenna over a period spanning the azimuth time bounds of
        the output LUT. Represents the rotation from the radar antenna coordinate system
        to ECEF coordinates as a function of time. The antenna coordinate system is a
        Cartesian coordinate system with +Z axis pointing along the mechanical boresight
        of the antenna, +X axis pointing in the direction of increasing elevation angle,
        and +Y axis pointing in the direction of increasing azimuth angle. If `epoch` is
        None, this must have the same reference epoch as `orbit`.
    wavelength : float
        The radar central wavelength, in meters.
    dem : DEMInterpolator or None, optional
        Digital elevation model specifying the height of the scene in meters above the
        WGS 84 ellipsoid. If None, a zero-height DEM is used. Will calculate stats
        (modifying input object) if they haven't already been calculated. Defaults to
        None.
    epoch : isce3.core.DateTime or None, optional
        Reference epoch for the azimuth time coordinates of the output LUT. If None,
        defaults to the reference epoch of `orbit` and `attitude` (which must have the
        same reference epoch in this case). Defaults to None.
    az_angle : float, optional
        Complement of the angle between the along-track axis of the antenna and its
        electrical boresight, in radians.  Zero for non-scanned, flush-mounted antennas
        like ALOS-1. Defaults to 0.
    interp_method : isce3.core.DataInterpMethod or str, optional
        Interpolation method used by the output LUT. Defaults to 'bilinear'.
    bounds_error : bool, optional
        Whether to raise an exception when attempting to evaluate the output LUT outside
        of its valid domain. Defaults to True.

    Returns
    -------
    isce3.core.LUT2d
        A 2-D LUT that may be used to compute the estimated Doppler centroid of the SAR
        acquisition, in hertz, as a function of azimuth time and slant range.
    """
    log = logging.getLogger("isce3.geometry.doppler")

    if epoch is None:
        # If `orbit` and `attitude` have different reference epochs, it could cause
        # confusion about which epoch the output LUT is referencing. It seems best to
        # raise an exception in this case, rather than update one reference epoch to
        # match the other.
        if orbit.reference_epoch != attitude.reference_epoch:
            msg = (
                f"orbit reference epoch ({orbit.reference_epoch}) must match attitude"
                f" reference epoch ({attitude.reference_epoch})"
            )
            raise ValueError(msg)
    else:
        # If `epoch` was specified, ensure that `orbit` and `attitude` each have that
        # reference epoch. If necessary, create a temporary copy of the orbit and/or
        # attitude data and update its time tags.
        if orbit.reference_epoch != epoch:
            orbit = orbit.copy()
            orbit.update_reference_epoch(epoch)
        if attitude.reference_epoch != epoch:
            attitude = attitude.copy()
            attitude.update_reference_epoch(epoch)

    az_time = np.asarray(az_time)
    slant_range = np.asarray(slant_range)

    # If a DEM wasn't provided, default to a zero-height DEM.
    if dem is None:
        dem = DEMInterpolator()

    # Compute height statistics of the input DEM, including the min & max heights. These
    # are needed by `get_approx_el_bounds` below. Statistics are stored internally in
    # the `DEMInterpolator` object. This is a no-op if height statistics were previously
    # computed.
    dem.compute_min_max_mean_height()

    dop = np.zeros((len(az_time), len(slant_range)))

    # Using the default EL bounds [-45, 45] deg can cause trouble when looking
    # near nadir, as this large interval can span both sides of the left-right
    # ambiguity.  So solve the problem on the sphere a few times using bounding
    # cases.
    log.info("Attempting to find reasonable EL search bounds.")
    ti = az_time[len(az_time) // 2]
    rdr_xyz, _ = orbit.interpolate(ti)
    qi = attitude.interpolate(ti)
    rmin = slant_range[0]
    rmax = slant_range[-1]
    el0, el1 = isce3.antenna.get_approx_el_bounds(rmin, az_angle, rdr_xyz, qi, dem)
    el2, el3 = isce3.antenna.get_approx_el_bounds(rmax, az_angle, rdr_xyz, qi, dem)
    el_min, el_max = min(el0, el2), max(el1, el3)
    log.info(
        f"Preliminary EL bounds are [{np.rad2deg(el_min) :.3f}, "
        f"{np.rad2deg(el_max) :.3f}] deg"
    )

    for i, ti in enumerate(az_time):
        rdr_xyz, v = orbit.interpolate(ti)
        qi = attitude.interpolate(ti)
        for j, rj in enumerate(slant_range):
            # For very long observations the geometry may change enough that
            # the bounds become invalid.  If that happens, recalculate.
            try:
                tgt_xyz = isce3.antenna.range_az_to_xyz(
                    rj, az_angle, rdr_xyz, qi, dem, el_min=el_min, el_max=el_max
                )
            except RuntimeError:
                el0, el1 = isce3.antenna.get_approx_el_bounds(
                    rj, az_angle, rdr_xyz, qi, dem
                )
                el_min, el_max = min(el0, el_min), max(el1, el_max)
                log.info(
                    f"Updating EL bounds to [{np.rad2deg(el_min) :.3f}, "
                    f"{np.rad2deg(el_max) :.3f}] deg"
                )
                tgt_xyz = isce3.antenna.range_az_to_xyz(
                    rj, az_angle, rdr_xyz, qi, dem, el_min=el_min, el_max=el_max
                )
            dop[i, j] = los2doppler(tgt_xyz - rdr_xyz, v, wavelength)

    return isce3.core.LUT2d(slant_range, az_time, dop, interp_method, bounds_error)


def main():
    """Demonstrate using attitude to compute Doppler centroid for a Capella SLC."""
    slc_path = Path(
        "tests/data/CAPELLA_C14_SM_SLC_HH_20240626150051_20240626150055.tif"
    )
    slc = CapellaSLC.from_file(slc_path)

    print(f"Loaded SLC: {slc_path.name}")
    print(f"Shape: {slc.shape}")
    print(f"Wavelength: {slc.wavelength:.4f} m")
    print(f"Number of state vectors: {len(slc.collect.state.state_vectors)}")
    print(f"Number of pointing samples: {len(slc.collect.pointing)}")

    orbit = get_orbit(slc.collect.state.state_vectors, ref_epoch=slc.ref_epoch)
    attitude = get_attitude(slc.collect.pointing, ref_epoch=slc.ref_epoch)

    print("\nComputing Doppler LUT from attitude...")

    sensing_start = (slc.sensing_start - slc.ref_epoch).total_seconds()
    sensing_end = sensing_start + slc.shape[0] * slc.delta_line_time
    n_az = 20
    az_times = np.linspace(sensing_start, sensing_end, n_az)

    n_rg = 10
    slant_ranges = np.linspace(
        slc.starting_range,
        slc.starting_range + slc.shape[1] * slc.delta_range_sample,
        n_rg,
    )

    doppler_from_attitude = make_doppler_lut_from_attitude(
        az_time=az_times,
        slant_range=slant_ranges,
        orbit=orbit,
        attitude=attitude,
        wavelength=slc.wavelength,
        epoch=isce3.core.DateTime(str(slc.ref_epoch).strip("Z")),
        bounds_error=False,
    )

    doppler_from_metadata = get_doppler_lut2d(slc, n_az=n_az)

    print("Computing differences...")
    az_eval = az_times[1:-1]
    rg_eval = slant_ranges[1:-1]

    Y, X = np.meshgrid(az_eval, rg_eval, indexing="ij")
    dop_att = np.zeros_like(Y)
    dop_meta = np.zeros_like(Y)

    for i, t in enumerate(az_eval):
        for j, r in enumerate(rg_eval):
            dop_att[i, j] = doppler_from_attitude.eval(t, r)
            dop_meta[i, j] = doppler_from_metadata.eval(t, r)

    diff = dop_att - dop_meta

    print(
        f"\nDoppler from attitude - mean: {dop_att.mean():.2f} Hz, std:"
        f" {dop_att.std():.2f} Hz"
    )
    print(
        f"Doppler from metadata - mean: {dop_meta.mean():.2f} Hz, std:"
        f" {dop_meta.std():.2f} Hz"
    )
    print(
        f"Difference - mean: {diff.mean():.2f} Hz, std: {diff.std():.2f} Hz, max:"
        f" {np.abs(diff).max():.2f} Hz"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(dop_att, aspect="auto", cmap="RdBu_r")
    axes[0].set_title("Doppler from Attitude")
    axes[0].set_xlabel("Slant Range Index")
    axes[0].set_ylabel("Azimuth Time Index")
    plt.colorbar(im0, ax=axes[0], label="Doppler (Hz)")

    im1 = axes[1].imshow(dop_meta, aspect="auto", cmap="RdBu_r")
    axes[1].set_title("Doppler from Metadata")
    axes[1].set_xlabel("Slant Range Index")
    axes[1].set_ylabel("Azimuth Time Index")
    plt.colorbar(im1, ax=axes[1], label="Doppler (Hz)")

    im2 = axes[2].imshow(diff, aspect="auto", cmap="RdBu_r")
    axes[2].set_title("Difference (Attitude - Metadata)")
    axes[2].set_xlabel("Slant Range Index")
    axes[2].set_ylabel("Azimuth Time Index")
    plt.colorbar(im2, ax=axes[2], label="Doppler (Hz)")

    plt.tight_layout()
    plt.savefig("doppler_comparison.png", dpi=150)
    print("\nSaved comparison plot to doppler_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
