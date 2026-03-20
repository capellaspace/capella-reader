#!/usr/bin/env python
"""Self-contained Capella SLC coregistration using capella-reader + isce3.

Demonstrates the minimum code needed to coregister two Capella stripmap SLCs
via DEM-based geometry offsets + amplitude cross-correlation refinement.

Dependencies: isce3, capella-reader, numpy, scipy, gdal.
Optional: sardem (auto-downloads Copernicus DEM if --dem-file is not provided).
Simplest installation:

    conda install -c conda-forge isce3 capella-reader sardem

Uses code from `phase_cross_correlation` from scikit-image.

Usage
-----
python coregister_capella.py REFERENCE.tif SECONDARY.tif [--dem-file DEM.tif] [--output-dir ./coreg]

"""

import argparse
import time
import warnings
from os import fsdecode
from pathlib import Path

import isce3
import numpy as np
from numpy.polynomial.polynomial import polyval2d, polyvander2d
from osgeo import gdal
from scipy.fft import fftn, ifftn

import capella_reader.adapters.isce3
from capella_reader import CapellaSLC

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# 1. DEM creation
# ---------------------------------------------------------------------------


def create_dem(slc_file: Path, output_dir: Path, dem_file: Path | None) -> Path:
    """Return path to a DEM covering the SLC extent.

    If `dem_file` is provided, return it directly. Otherwise, auto-download
    a Copernicus DEM via sardem.
    """
    if dem_file is not None:
        return dem_file

    import sardem.dem

    out = output_dir / "dem.tif"
    if out.exists():
        print(f"  DEM already exists: {out}")
        return out

    slc = CapellaSLC.from_file(slc_file)
    w, s, e, n = slc.bounds
    pad = 0.3  # degrees
    bbox = (w - pad, s - pad, e + pad, n + pad)
    print(f"  Downloading Copernicus DEM for bbox {bbox} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    sardem.dem.main(
        output_name=str(out),
        bbox=bbox,
        data_source="COP",
        output_type="float32",
        output_format="GTiff",
    )
    print(f"  DEM saved to {out}")
    return out


# ---------------------------------------------------------------------------
# 2. Reference geometry (rdr2geo)
# ---------------------------------------------------------------------------

GEOMETRY_LAYERS = {
    "x_raster": ("x", gdal.GDT_Float64),
    "y_raster": ("y", gdal.GDT_Float64),
    "height_raster": ("z", gdal.GDT_Float64),
    "incidence_angle_raster": ("incidence_angle", gdal.GDT_Float32),
    "heading_angle_raster": ("heading_angle", gdal.GDT_Float32),
    "local_incidence_angle_raster": ("local_incidence_angle", gdal.GDT_Float32),
    "local_psi_raster": ("psi", gdal.GDT_Float32),
    "simulated_amplitude_raster": ("simulated_amplitude", gdal.GDT_Float32),
    "layover_shadow_raster": ("layover_shadow_mask", gdal.GDT_Byte),
    "ground_to_sat_east_raster": ("los_east", gdal.GDT_Float32),
    "ground_to_sat_north_raster": ("los_north", gdal.GDT_Float32),
}


def _open_slc_isce3(slc_file: Path):
    """Open a Capella SLC and return (radar_grid, orbit, ellipsoid)."""
    slc = CapellaSLC.from_file(slc_file)
    radar_grid = capella_reader.adapters.isce3.get_radar_grid(slc)
    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        orbit = capella_reader.adapters.isce3.get_orbit(slc)
    ellipsoid = isce3.core.make_projection(4326).ellipsoid
    return slc, radar_grid, orbit, ellipsoid


def run_geometry(ref_file: Path, dem_file: Path, output_dir: Path) -> Path:
    """Compute reference geometry layers via rdr2geo."""
    geom_dir = output_dir / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    out_vrt = geom_dir / "geometry.vrt"

    _, radar_grid, orbit, ellipsoid = _open_slc_isce3(ref_file)
    dem_raster = isce3.io.Raster(fsdecode(dem_file))

    rdr2geo = isce3.geometry.Rdr2Geo(
        radar_grid,
        orbit,
        ellipsoid,
        isce3.core.LUT2d(),
        threshold=1e-8,
        numiter=20,
        extraiter=10,
        lines_per_block=1024,
    )

    rasters = [
        isce3.io.Raster(
            fsdecode(geom_dir / f"{fname}.tif"),
            radar_grid.width,
            radar_grid.length,
            1,
            dtype,
            "GTiff",
        )
        for fname, dtype in GEOMETRY_LAYERS.values()
    ]

    t0 = time.time()
    rdr2geo.topo(dem_raster, *rasters)
    print(f"  rdr2geo took {time.time() - t0:.1f} s")

    # Build VRT stacking all layers
    out_stack = isce3.io.Raster(fsdecode(out_vrt), rasters)
    out_stack.set_epsg(rdr2geo.epsg_out)
    del out_stack, rasters
    return out_vrt


# ---------------------------------------------------------------------------
# 3. geo2rdr offsets
# ---------------------------------------------------------------------------


def run_geo2rdr(
    sec_file: Path, geometry_vrt: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Compute range/azimuth offsets mapping reference geometry to secondary grid."""
    g2r_dir = output_dir / "geo2rdr"
    g2r_dir.mkdir(parents=True, exist_ok=True)

    _, radar_grid, orbit, ellipsoid = _open_slc_isce3(sec_file)
    doppler = isce3.core.LUT2d()  # Zero-doppler grid

    geo2rdr = isce3.geometry.Geo2Rdr(
        radar_grid,
        orbit,
        ellipsoid,
        doppler,
        1e-8,
        20,
        1024,
    )

    geometry_raster = isce3.io.Raster(fsdecode(geometry_vrt))
    t0 = time.time()
    geo2rdr.geo2rdr(geometry_raster, fsdecode(g2r_dir))
    print(f"  geo2rdr took {time.time() - t0:.1f} s")

    rg_off = g2r_dir / "range.off"
    az_off = g2r_dir / "azimuth.off"
    return rg_off, az_off


# ---------------------------------------------------------------------------
# 4. SLC resampling
# ---------------------------------------------------------------------------


def resample_slc(
    ref_file: Path,
    sec_file: Path,
    rg_off_path: Path,
    az_off_path: Path,
    output_file: Path,
) -> Path:
    """Resample the secondary SLC onto the reference radar grid."""
    ref_slc = CapellaSLC.from_file(ref_file)
    sec_slc = CapellaSLC.from_file(sec_file)

    ref_grid = capella_reader.adapters.isce3.get_radar_grid(ref_slc)
    sec_grid = capella_reader.adapters.isce3.get_radar_grid(sec_slc)
    doppler_lut = capella_reader.adapters.isce3.get_doppler_lut2d(sec_slc)

    az_carrier = isce3.core.Poly2d(np.array([0.0]))
    rg_carrier = isce3.core.Poly2d(np.array([0.0]))

    resamp = isce3.image.ResampSlc(
        sec_grid,
        doppler_lut,
        az_carrier,
        rg_carrier,
        0.0j,
        ref_grid,
    )
    resamp.lines_per_tile = 1024

    rg_off_r = isce3.io.Raster(fsdecode(rg_off_path))
    az_off_r = isce3.io.Raster(fsdecode(az_off_path))
    in_raster = isce3.io.Raster(fsdecode(sec_file))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_raster = isce3.io.Raster(
        fsdecode(output_file),
        rg_off_r.width,
        rg_off_r.length,
        1,
        gdal.GDT_CFloat32,
        "GTiff",
    )

    t0 = time.time()
    resamp.resamp(in_raster, out_raster, rg_off_r, az_off_r, flatten=True)
    del in_raster, out_raster, rg_off_r, az_off_r
    print(f"  Resample took {time.time() - t0:.1f} s")

    # Copy Capella TIFF metadata tag to output
    ds_in = gdal.Open(str(sec_file))
    ds_out = gdal.Open(str(output_file), gdal.GA_Update)
    ds_out.SetMetadataItem(
        "TIFFTAG_IMAGEDESCRIPTION",
        ds_in.GetMetadataItem("TIFFTAG_IMAGEDESCRIPTION"),
    )
    ds_in = ds_out = None
    return output_file


# ---------------------------------------------------------------------------
# 5. Fine cross-correlation
# ---------------------------------------------------------------------------


def _upsampled_dft(data, upsampled_region_size, upsample_factor, axis_offsets):
    """Upsampled DFT by matrix multiplication (Guizar et al. 2008).

    Computes the cross-correlation in a small neighborhood around the peak
    at sub-pixel resolution without zero-padding the full array.
    """
    im2pi = 1j * 2 * np.pi
    n_rows, n_cols = data.shape
    row_offset, col_offset = axis_offsets

    row_kernel = (np.arange(upsampled_region_size) - row_offset)[
        :, None
    ] * np.fft.fftfreq(n_rows, upsample_factor)
    row_kernel = np.exp(-im2pi * row_kernel)

    col_kernel = (np.arange(upsampled_region_size) - col_offset)[
        :, None
    ] * np.fft.fftfreq(n_cols, upsample_factor)
    col_kernel = np.exp(-im2pi * col_kernel)

    return row_kernel @ data @ col_kernel.T


def correlate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    expected_shift: tuple[float, float] = (0.0, 0.0),
    search_radius: tuple[int, int] | None = None,
    upsample_factor: int = 1,
) -> tuple[np.ndarray, float, float]:
    """Estimate sub-pixel translation between two images via FFT cross-correlation.

    Returns (shift_yx, snr, peak_ncc).
    """
    assert reference.shape == moving.shape

    ref = (
        np.abs(reference)
        if np.iscomplexobj(reference)
        else reference.astype(np.float32, copy=False)
    )
    mov = (
        np.abs(moving)
        if np.iscomplexobj(moving)
        else moving.astype(np.float32, copy=False)
    )
    # Demean for robust SNR
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    H, W = ref.shape

    F = fftn(ref)
    G = fftn(mov)
    prod = F * np.conj(G)
    cc = ifftn(prod)
    cc_abs = np.abs(cc)
    src_energy = float(np.sum(ref**2))
    mov_energy = float(np.sum(mov**2))

    # Bounded search around expected shift
    if search_radius is None:
        peak_idx = int(np.argmax(cc_abs))
        peak = (peak_idx // W, peak_idx % W)
        peak_cc_abs = float(cc_abs.ravel()[peak_idx])
    else:
        ry, rx = search_radius
        ry = min(ry, H // 2 - 1)
        rx = min(rx, W // 2 - 1)
        cy = int(np.round(expected_shift[0])) % H
        cx = int(np.round(expected_shift[1])) % W
        rows = (np.arange(-ry, ry + 1) + cy) % H
        cols = (np.arange(-rx, rx + 1) + cx) % W
        window = cc_abs[np.ix_(rows, cols)]
        w_peak_idx = int(np.argmax(window))
        w_shape = (2 * ry + 1, 2 * rx + 1)
        pr = w_peak_idx // w_shape[1]
        pc = w_peak_idx % w_shape[1]
        peak = (int(rows[pr]), int(cols[pc]))
        peak_cc_abs = float(cc_abs[peak])

    # Signed shift
    shift = np.array([float(peak[0]), float(peak[1])], dtype=np.float64)
    midpoint = np.array([np.floor(H / 2), np.floor(W / 2)], dtype=np.float64)
    dims = np.array([H, W], dtype=np.float64)
    shift[shift > midpoint] -= dims[shift > midpoint]

    # Sub-pixel refinement via upsampled DFT
    if upsample_factor > 1:
        shift_rounded = np.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = int(np.ceil(upsample_factor * 1.5))
        dftshift = np.fix(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shift_rounded * upsample_factor

        up_cc = np.conj(
            _upsampled_dft(
                np.conj(prod),
                upsampled_region_size,
                upsample_factor,
                (sample_region_offset[0], sample_region_offset[1]),
            )
        )
        up_cc_abs = np.abs(up_cc)
        up_peak_idx = int(np.argmax(up_cc_abs))
        up_peak = (
            up_peak_idx // upsampled_region_size,
            up_peak_idx % upsampled_region_size,
        )
        shift = (
            shift_rounded
            + (np.array(up_peak, dtype=np.float64) - dftshift) / upsample_factor
        )

    # Metrics
    denom = np.sqrt(src_energy * mov_energy)
    peak_ncc = float(peak_cc_abs / max(denom, 1e-12))

    # SNR: peak / RMS of background (excluding guard region around peak)
    guard = max(3, int(min(cc_abs.shape) / 32))
    pr, pc = peak[0], peak[1]
    mask = np.ones(cc_abs.shape, dtype=bool)
    r0, r1 = max(0, pr - guard), min(H, pr + guard + 1)
    c0, c1 = max(0, pc - guard), min(W, pc + guard + 1)
    mask[r0:r1, c0:c1] = False
    bg = cc_abs[mask].astype(np.float64)
    rms = float(np.sqrt(np.mean(bg**2))) if bg.size >= 16 else 1e-12
    snr = float(peak_cc_abs / max(rms, 1e-12))

    return shift.astype(np.float32), snr, peak_ncc


def estimate_bulk_offset(
    ref_data: np.ndarray,
    sec_data: np.ndarray,
    chip_size: tuple[int, int] = (2048, 2048),
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate global (dy, dx) by median of large-chip correlations."""
    nrows, ncols = ref_data.shape
    ch, cw = chip_size
    shifts = []

    for r0 in range(0, nrows - ch + 1, ch):
        for c0 in range(0, ncols - cw + 1, cw):
            ref_chip = ref_data[r0 : r0 + ch, c0 : c0 + cw]
            sec_chip = sec_data[r0 : r0 + ch, c0 : c0 + cw]
            ref_mag = np.abs(ref_chip) if np.iscomplexobj(ref_chip) else ref_chip
            if np.var(ref_mag) < 1e-6:
                continue
            s, _, _ = correlate_translation(ref_chip, sec_chip)
            shifts.append(s.astype(np.float64))

    if len(shifts) == 0:
        print("  WARNING: no valid chips for bulk offset, returning zero")
        return np.zeros(2, dtype=np.float32), np.full(2, 32.0, dtype=np.float32)

    arr = np.vstack(shifts)
    med = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - med), axis=0) * 1.4826
    return med.astype(np.float32), mad.astype(np.float32)


def correlate_grid(
    ref_data: np.ndarray,
    sec_data: np.ndarray,
    *,
    chip_size: tuple[int, int] = (256, 256),
    upsample_factor: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Correlate at a grid of points across the image pair.

    Returns (row_centers, col_centers, az_offsets, rg_offsets, snr, peak_ncc).
    """
    nrows, ncols = ref_data.shape
    ch, cw = chip_size

    # Bulk offset first
    bulk_chip = (min(2048, nrows), min(2048, ncols))
    print("  Estimating bulk offset ...")
    bulk_shift, bulk_mad = estimate_bulk_offset(ref_data, sec_data, chip_size=bulk_chip)
    print(f"  Bulk shift (dy, dx): {bulk_shift}, MAD: {bulk_mad}")

    # Search radius from dispersion
    ry = int(min(max(32, np.ceil(4 * bulk_mad[0])), ch // 3))
    rx = int(min(max(32, np.ceil(4 * bulk_mad[1])), cw // 3))
    search_radius = (ry, rx)
    print(f"  Search radius: {search_radius}")

    idy, idx = int(np.round(bulk_shift[0])), int(np.round(bulk_shift[1]))
    fdy, fdx = float(bulk_shift[0]) - idy, float(bulk_shift[1]) - idx

    # Build grid
    row_centers = np.arange(ch // 2, nrows - ch // 2, ch)
    col_centers = np.arange(cw // 2, ncols - cw // 2, cw)
    rr, cc = np.meshgrid(row_centers, col_centers, indexing="ij")
    rr_flat = rr.ravel()
    cc_flat = cc.ravel()
    n_points = len(rr_flat)

    az_off = np.full(n_points, np.nan)
    rg_off = np.full(n_points, np.nan)
    snr_arr = np.full(n_points, np.nan)
    ncc_arr = np.full(n_points, np.nan)

    for i in range(n_points):
        rc, ccc = int(rr_flat[i]), int(cc_flat[i])
        r0, r1 = rc - ch // 2, rc + ch // 2
        c0, c1 = ccc - cw // 2, ccc + cw // 2
        if r0 < 0 or r1 > nrows or c0 < 0 or c1 > ncols:
            continue
        sr0, sr1 = r0 - idy, r1 - idy
        sc0, sc1 = c0 - idx, c1 - idx
        if sr0 < 0 or sr1 > nrows or sc0 < 0 or sc1 > ncols:
            continue

        chip_ref = ref_data[r0:r1, c0:c1]
        chip_sec = sec_data[sr0:sr1, sc0:sc1]
        s, snr, ncc = correlate_translation(
            chip_ref,
            chip_sec,
            expected_shift=(fdy, fdx),
            search_radius=search_radius,
            upsample_factor=upsample_factor,
        )
        az_off[i] = float(-idy) + float(-s[0])
        rg_off[i] = float(-idx) + float(-s[1])
        snr_arr[i] = snr
        ncc_arr[i] = ncc

        if (i + 1) % 50 == 0 or i == n_points - 1:
            print(f"  Correlated {i + 1}/{n_points} grid points", end="\r")

    print()
    return rr_flat, cc_flat, az_off, rg_off, snr_arr, ncc_arr


# ---------------------------------------------------------------------------
# 6. Polynomial fitting with outlier rejection
# ---------------------------------------------------------------------------


def fit_poly2d(x, y, data, degree=1):
    """Fit a 2D polynomial p(x, y) = sum c[i,j] x^i y^j via least squares."""
    deg = (degree, degree)
    vander = polyvander2d(x, y, deg)
    coeffs_flat = np.linalg.lstsq(vander, data, rcond=None)[0]
    return coeffs_flat.reshape(degree + 1, degree + 1)


def fit_polynomials_robust(
    rows,
    cols,
    az_off,
    rg_off,
    *,
    degree: int = 1,
    n_iterations: int = 5,
    outlier_threshold: float = 3.0,
    min_samples: int = 10,
):
    """Fit az/rg polynomials with joint MAD-based outlier rejection.

    Returns (az_coeffs, rg_coeffs, inlier_mask).
    """
    active = np.ones(len(rows), dtype=bool)

    for iteration in range(n_iterations):
        if np.sum(active) < min_samples:
            break

        r, c = rows[active], cols[active]
        az, rg = az_off[active], rg_off[active]

        az_c = fit_poly2d(r, c, az, degree)
        rg_c = fit_poly2d(r, c, rg, degree)

        az_resid = az - polyval2d(r, c, az_c)
        rg_resid = rg - polyval2d(r, c, rg_c)

        az_scale = 1.4826 * np.median(np.abs(az_resid - np.median(az_resid)))
        rg_scale = 1.4826 * np.median(np.abs(rg_resid - np.median(rg_resid)))
        if az_scale < 1e-10 or rg_scale < 1e-10:
            break

        joint = np.sqrt((az_resid / az_scale) ** 2 + (rg_resid / rg_scale) ** 2)
        outliers = joint > outlier_threshold * np.sqrt(2)
        n_out = int(np.sum(outliers))
        if n_out == 0:
            break

        # Map back to full-size mask
        idx = np.where(active)[0]
        keep = ~outliers
        if np.sum(keep) < min_samples:
            # Remove worst ones down to min
            n_to_remove = np.sum(active) - min_samples
            if n_to_remove <= 0:
                break
            worst = np.argsort(joint)[-n_to_remove:]
            keep = np.ones(len(joint), dtype=bool)
            keep[worst] = False
            n_out = int(n_to_remove)
        active[idx[~keep]] = False
        print(
            f"  Iteration {iteration + 1}: removed {n_out} outliers, {np.sum(active)}"
            " remain"
        )

    # Final fit
    r, c = rows[active], cols[active]
    az_coeffs = fit_poly2d(r, c, az_off[active], degree)
    rg_coeffs = fit_poly2d(r, c, rg_off[active], degree)
    return az_coeffs, rg_coeffs, active


def compute_fine_offsets(
    ref_file: Path,
    sec_file: Path,
    output_dir: Path,
    *,
    chip_size: tuple[int, int] = (256, 256),
    upsample_factor: int = 32,
    peak_ncc_threshold: float = 0.05,
) -> tuple[Path, Path]:
    """Compute fine offset rasters from cross-correlation."""
    fine_dir = output_dir / "fine_offsets"
    fine_dir.mkdir(parents=True, exist_ok=True)
    az_path = fine_dir / "azimuth.fine.off"
    rg_path = fine_dir / "range.fine.off"

    if az_path.exists() and rg_path.exists():
        print("  Fine offsets already exist, skipping.")
        return az_path, rg_path

    # Load both SLCs into memory
    print("  Loading SLCs ...")
    ref_ds = gdal.Open(str(ref_file))
    sec_ds = gdal.Open(str(sec_file))
    ref_data = ref_ds.GetRasterBand(1).ReadAsArray()
    sec_data = sec_ds.GetRasterBand(1).ReadAsArray()
    nrows, ncols = ref_data.shape
    assert sec_data.shape == ref_data.shape, "SLC shapes must match for fine offsets"

    # Correlate on grid
    row_c, col_c, az_off, rg_off, _snr, ncc = correlate_grid(
        ref_data,
        sec_data,
        chip_size=chip_size,
        upsample_factor=upsample_factor,
    )

    # Quality filter
    valid = ~np.isnan(az_off) & (ncc >= peak_ncc_threshold)
    n_valid = int(np.sum(valid))
    print(f"  Valid correlations: {n_valid} / {len(az_off)}")
    assert n_valid > 0, "No correlations passed quality filter"

    # Robust polynomial fit
    print("  Fitting polynomials ...")
    az_coeffs, rg_coeffs, inlier = fit_polynomials_robust(
        row_c[valid],
        col_c[valid],
        az_off[valid],
        rg_off[valid],
    )
    print(f"  Final inliers: {np.sum(inlier)}")

    # Evaluate on full pixel grid -> write flat binary + ENVI header
    az_centers = 0.5 + np.arange(nrows)
    rg_centers = 0.5 + np.arange(ncols)

    az_fine = np.memmap(az_path, mode="w+", dtype=np.float32, shape=(nrows, ncols))
    rg_fine = np.memmap(rg_path, mode="w+", dtype=np.float32, shape=(nrows, ncols))

    block = 1024
    for r0 in range(0, nrows, block):
        r1 = min(r0 + block, nrows)
        YY, XX = np.meshgrid(az_centers[r0:r1], rg_centers, indexing="ij")
        az_fine[r0:r1, :] = polyval2d(YY, XX, az_coeffs).astype(np.float32)
        rg_fine[r0:r1, :] = polyval2d(YY, XX, rg_coeffs).astype(np.float32)

    az_fine.flush()
    rg_fine.flush()

    # Write ENVI headers
    for path in (az_path, rg_path):
        _write_envi_header(path, nrows, ncols, np.dtype("float32"))

    del az_fine, rg_fine
    print(f"  Fine offsets written to {fine_dir}")
    return az_path, rg_path


def _write_envi_header(file_path: Path, lines: int, samples: int, dtype: np.dtype):
    """Write a minimal ENVI header for a flat binary file."""
    envi_dtypes = {
        np.dtype("float32"): 4,
        np.dtype("float64"): 5,
        np.dtype("complex64"): 6,
    }
    hdr = (
        "ENVI\n"
        "description = {Created by coregister_capella.py}\n"
        f"samples = {samples}\n"
        f"lines = {lines}\n"
        "bands = 1\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        f"data type = {envi_dtypes[dtype]}\n"
        "interleave = bsq\n"
    )
    Path(str(file_path) + ".hdr").write_text(hdr)


# ---------------------------------------------------------------------------
# 8. Main / CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Coregister two Capella SLCs (DEM-based + cross-correlation refinement)."
        ),
    )
    parser.add_argument("reference", type=Path, help="Reference SLC (GeoTIFF)")
    parser.add_argument("secondary", type=Path, help="Secondary SLC (GeoTIFF)")
    parser.add_argument(
        "--dem-file",
        type=Path,
        default=None,
        help="DEM in EPSG:4326 (auto-downloaded if omitted)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("coreg_output"), help="Output directory"
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Step 1: DEM
    print("[1/6] DEM")
    dem_file = create_dem(args.reference, output_dir, args.dem_file)

    # Step 2: Reference geometry
    print("[2/6] Reference geometry (rdr2geo)")
    geometry_vrt = run_geometry(args.reference, dem_file, output_dir)

    # Step 3: geo2rdr offsets
    print("[3/6] geo2rdr offsets")
    rg_off, az_off = run_geo2rdr(args.secondary, geometry_vrt, output_dir)

    # Step 4: Coarse resample
    print("[4/6] Coarse resample")
    coarse_file = output_dir / "coarse_resampled.tif"
    resample_slc(args.reference, args.secondary, rg_off, az_off, coarse_file)

    # Step 5: Fine cross-correlation offsets
    print("[5/6] Fine cross-correlation offsets")
    az_fine, rg_fine = compute_fine_offsets(args.reference, coarse_file, output_dir)

    # Step 6: Fine resample (coarse offsets + fine offsets combined)
    # The fine offsets are additive corrections to the coarse (geo2rdr) offsets.
    # We sum them and resample from the *original* secondary.
    print("[6/6] Fine resample")
    combined_dir = output_dir / "combined_offsets"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Read coarse offset dimensions via isce3 (geo2rdr outputs use XML metadata)
    rg_raster = isce3.io.Raster(fsdecode(rg_off))
    nrows, ncols = rg_raster.length, rg_raster.width
    del rg_raster
    rg_coarse = np.memmap(rg_off, dtype=np.float64, mode="r", shape=(nrows, ncols))
    az_coarse = np.memmap(az_off, dtype=np.float64, mode="r", shape=(nrows, ncols))

    # Read fine offsets
    rg_fine_data = np.memmap(rg_fine, dtype=np.float32, mode="r", shape=(nrows, ncols))
    az_fine_data = np.memmap(az_fine, dtype=np.float32, mode="r", shape=(nrows, ncols))

    # Sum and write combined offsets as float64 (isce3 resamp expects double)
    rg_combined_path = combined_dir / "range.off"
    az_combined_path = combined_dir / "azimuth.off"
    rg_combined = np.memmap(
        rg_combined_path, mode="w+", dtype=np.float64, shape=(nrows, ncols)
    )
    az_combined = np.memmap(
        az_combined_path, mode="w+", dtype=np.float64, shape=(nrows, ncols)
    )
    rg_combined[:] = rg_coarse + rg_fine_data
    az_combined[:] = az_coarse + az_fine_data
    rg_combined.flush()
    az_combined.flush()

    for path in (rg_combined_path, az_combined_path):
        _write_envi_header(path, nrows, ncols, np.dtype("float64"))
    del rg_coarse, az_coarse, rg_fine_data, az_fine_data, rg_combined, az_combined

    final_output = output_dir / "coregistered.tif"
    resample_slc(
        args.reference, args.secondary, rg_combined_path, az_combined_path, final_output
    )

    print(f"\nDone in {time.time() - t_start:.1f} s")
    print(f"Final coregistered SLC: {final_output}")


if __name__ == "__main__":
    main()
