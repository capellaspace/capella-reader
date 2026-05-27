"""Shared helpers for the Capella InSAR coregistration examples.

Two groups of helpers live here:

* **Geometry / DEM** (``create_dem``, ``open_slc_isce3``, ``run_geometry``,
  ``write_envi_header``): used by both the stripmap pipeline
  (``coregister_isce3.py``) and the spotlight phase restoration
  (``restore_spotlight_phase.py``).
* **Sub-pixel cross-correlation** (``correlate_grid``, ``bulk_offset``):
  fine offset estimation on a regular grid of chips, delegated to
  ``skimage.registration.phase_cross_correlation`` with
  ``normalization=None`` (unnormalized cross-correlation, recommended for
  noisy SAR amplitude imagery per Guizar et al. 2008).

References
----------
.. [1] Guizar-Sicairos, M., Thurman, S. T., & Fienup, J. R. (2008).
   Efficient subpixel image registration algorithms. Optics Letters, 33(2),
   156-158. https://doi.org/10.1364/OL.33.000156
"""

from __future__ import annotations

import time
import warnings
from os import fsdecode
from pathlib import Path

import isce3
import numpy as np
from osgeo import gdal
from skimage.registration import phase_cross_correlation

import capella_reader.adapters.isce3
from capella_reader import CapellaSLC


def create_dem(slc_file: Path, output_dir: Path, dem_file: Path | None) -> Path:
    """Return a DEM covering the SLC extent; auto-download Copernicus if None."""
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
    return out


def open_slc_isce3(slc_file: Path):
    """Open a Capella SLC and return ``(slc, radar_grid, orbit, ellipsoid)``."""
    slc = CapellaSLC.from_file(slc_file)
    radar_grid = capella_reader.adapters.isce3.get_radar_grid(slc)
    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        orbit = capella_reader.adapters.isce3.get_orbit(slc)
    ellipsoid = isce3.core.make_projection(4326).ellipsoid
    return slc, radar_grid, orbit, ellipsoid


def run_geometry(slc_file: Path, dem_file: Path, output_dir: Path) -> Path:
    """Compute lon / lat / height per pixel and return the 3-band geometry VRT."""
    geom_dir = output_dir / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    out_vrt = geom_dir / "geometry.vrt"
    if out_vrt.exists():
        return out_vrt

    _, radar_grid, orbit, ellipsoid = open_slc_isce3(slc_file)
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

    def _layer(name: str) -> isce3.io.Raster:
        return isce3.io.Raster(
            fsdecode(geom_dir / f"{name}.tif"),
            radar_grid.width,
            radar_grid.length,
            1,
            gdal.GDT_Float64,
            "GTiff",
        )

    x_raster = _layer("x")
    y_raster = _layer("y")
    z_raster = _layer("z")

    t0 = time.time()
    rdr2geo.topo(
        isce3.io.Raster(fsdecode(dem_file)),
        x_raster=x_raster,
        y_raster=y_raster,
        height_raster=z_raster,
    )
    print(f"  rdr2geo took {time.time() - t0:.1f} s")

    stack = isce3.io.Raster(fsdecode(out_vrt), [x_raster, y_raster, z_raster])
    stack.set_epsg(rdr2geo.epsg_out)
    del stack, x_raster, y_raster, z_raster
    return out_vrt


def write_envi_header(
    file_path: Path, lines: int, samples: int, dtype: np.dtype
) -> None:
    """Write a minimal ENVI .hdr sidecar so ISCE3 can mmap the offset rasters."""
    envi_dtypes = {
        np.dtype("float32"): 4,
        np.dtype("float64"): 5,
        np.dtype("complex64"): 6,
    }
    hdr = (
        "ENVI\n"
        f"samples = {samples}\n"
        f"lines = {lines}\n"
        "bands = 1\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        f"data type = {envi_dtypes[dtype]}\n"
        "interleave = bsq\n"
    )
    Path(str(file_path) + ".hdr").write_text(hdr)


def correlate_grid(
    ref_data: np.ndarray,
    sec_data: np.ndarray,
    *,
    chip_size: tuple[int, int] = (256, 256),
    upsample_factor: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-pixel offsets on a regular grid of chips.

    Parameters
    ----------
    ref_data : np.ndarray
        Reference image data.
    sec_data : np.ndarray
        Secondary image data (same shape as ``ref_data``, already coarsely
        resampled onto the reference grid).
    chip_size : tuple[int, int]
        Size of correlation chips (rows, cols).
    upsample_factor : int
        Sub-pixel refinement factor passed to ``phase_cross_correlation``.

    Returns
    -------
    az_off, rg_off : np.ndarray
        Azimuth (row) and range (col) offsets at each chip, in resample
        convention (value to ADD to a reference pixel index to find the
        corresponding secondary pixel). NaN where the chip was skipped.
    error : np.ndarray
        ``phase_cross_correlation`` normalized RMS error at each chip
        (0 = perfect, ~1 = no correlation).
    """
    assert ref_data.shape == sec_data.shape
    ref = np.abs(ref_data) if np.iscomplexobj(ref_data) else ref_data
    sec = np.abs(sec_data) if np.iscomplexobj(sec_data) else sec_data

    nrows, ncols = ref.shape
    ch, cw = chip_size

    row_centers = np.arange(ch // 2, nrows - ch // 2, ch)
    col_centers = np.arange(cw // 2, ncols - cw // 2, cw)
    n_points = len(row_centers) * len(col_centers)

    az_off = np.full(n_points, np.nan)
    rg_off = np.full(n_points, np.nan)
    err = np.full(n_points, np.nan)

    i = 0
    for rc in row_centers:
        for ccc in col_centers:
            r0, r1 = rc - ch // 2, rc + ch // 2
            c0, c1 = ccc - cw // 2, ccc + cw // 2
            shift, error, _ = phase_cross_correlation(
                ref[r0:r1, c0:c1],
                sec[r0:r1, c0:c1],
                upsample_factor=upsample_factor,
                normalization=None,  # type: ignore[arg-type]
            )
            # phase_cross_correlation returns the shift needed to register
            # `moving` (secondary) onto `reference`; resample wants the inverse.
            az_off[i] = -float(shift[0])
            rg_off[i] = -float(shift[1])
            err[i] = float(error)
            i += 1
            if i % 50 == 0 or i == n_points:
                print(f"  Correlated {i}/{n_points} chips", end="\r")

    print()
    return az_off, rg_off, err


def bulk_offset(
    az_off: np.ndarray,
    rg_off: np.ndarray,
    *,
    n_mad: float = 3.0,
) -> tuple[float, float, np.ndarray]:
    """Robust constant (az, rg) shift from per-chip offsets.

    Iteratively clips outliers more than ``n_mad`` MADs from the median in
    either axis, then returns the median of what remains.

    Returns
    -------
    az_median, rg_median : float
        Robust constant offsets.
    inliers : np.ndarray
        Boolean mask of which input chips were used.
    """
    inliers = np.isfinite(az_off) & np.isfinite(rg_off)
    assert inliers.any(), "No finite chip offsets to take a median over"
    for _ in range(3):
        az_med = np.median(az_off[inliers])
        rg_med = np.median(rg_off[inliers])
        az_mad = 1.4826 * np.median(np.abs(az_off[inliers] - az_med)) + 1e-6
        rg_mad = 1.4826 * np.median(np.abs(rg_off[inliers] - rg_med)) + 1e-6
        new_inliers = (
            np.isfinite(az_off)
            & np.isfinite(rg_off)
            & (np.abs(az_off - az_med) < n_mad * az_mad)
            & (np.abs(rg_off - rg_med) < n_mad * rg_mad)
        )
        if new_inliers.sum() == inliers.sum():
            break
        inliers = new_inliers
    return float(np.median(az_off[inliers])), float(np.median(rg_off[inliers])), inliers
