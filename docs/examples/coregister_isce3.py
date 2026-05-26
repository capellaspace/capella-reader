#!/usr/bin/env python
"""Capella SLC coregistration using capella-reader + isce3.

Demonstrates the minimum code needed to coregister two Capella stripmap SLCs
via DEM-based geometry offsets + amplitude cross-correlation refinement.

Cross-correlation and polynomial fitting utilities are shared with the
sarpy-based example (coregister_sarpy.py) via coreg_utils.py.

Dependencies: isce3, capella-reader, numpy, scipy, gdal.
Optional: sardem (auto-downloads Copernicus DEM if --dem-file is not provided).
Simplest installation:

    conda install -c conda-forge isce3 capella-reader sardem

Usage
-----
python coregister_isce3.py REFERENCE.tif SECONDARY.tif [--dem-file DEM.tif] [--output-dir ./coreg]

"""

import argparse
import time
import warnings
from os import fsdecode
from pathlib import Path

import isce3
import numpy as np
from coreg_utils import bulk_offset, correlate_grid
from osgeo import gdal

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
    *,
    baseband_input: bool = False,
) -> Path:
    """Resample the secondary SLC onto the reference radar grid.

    Set ``baseband_input=True`` when the input SLC carries no Doppler-centroid
    ramp — e.g. Capella spotlight after deramp-phase restoration — so the
    azimuth sinc kernel is not frequency-shifted.
    """
    ref_slc = CapellaSLC.from_file(ref_file)
    sec_slc = CapellaSLC.from_file(sec_file)

    ref_grid = capella_reader.adapters.isce3.get_radar_grid(ref_slc)
    sec_grid = capella_reader.adapters.isce3.get_radar_grid(sec_slc)
    doppler_lut = (
        isce3.core.LUT2d()
        if baseband_input
        else capella_reader.adapters.isce3.get_doppler_lut2d(sec_slc)
    )

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


def compute_fine_offsets(
    ref_file: Path,
    sec_file: Path,
    output_dir: Path,
    *,
    chip_size: tuple[int, int] = (256, 256),
    upsample_factor: int = 32,
) -> tuple[Path, Path]:
    """Constant-shift fine offsets from the median of per-chip correlations."""
    fine_dir = output_dir / "fine_offsets"
    fine_dir.mkdir(parents=True, exist_ok=True)
    az_path = fine_dir / "azimuth.fine.off"
    rg_path = fine_dir / "range.fine.off"

    if az_path.exists() and rg_path.exists():
        print("  Fine offsets already exist, skipping.")
        return az_path, rg_path

    print("  Loading SLCs ...")
    ref_ds = gdal.Open(str(ref_file))
    sec_ds = gdal.Open(str(sec_file))
    ref_data = ref_ds.GetRasterBand(1).ReadAsArray()
    sec_data = sec_ds.GetRasterBand(1).ReadAsArray()
    nrows, ncols = ref_data.shape
    assert sec_data.shape == ref_data.shape, "SLC shapes must match for fine offsets"

    az_off, rg_off, _err = correlate_grid(
        ref_data,
        sec_data,
        chip_size=chip_size,
        upsample_factor=upsample_factor,
    )
    az_med, rg_med, inliers = bulk_offset(az_off, rg_off)
    print(
        f"  Bulk fine offset: az={az_med:+.3f}, rg={rg_med:+.3f}"
        f" ({inliers.sum()}/{len(az_off)} inlier chips)"
    )

    az_fine = np.memmap(az_path, mode="w+", dtype=np.float32, shape=(nrows, ncols))
    rg_fine = np.memmap(rg_path, mode="w+", dtype=np.float32, shape=(nrows, ncols))
    az_fine[:] = np.float32(az_med)
    rg_fine[:] = np.float32(rg_med)
    az_fine.flush()
    rg_fine.flush()
    del az_fine, rg_fine

    for path in (az_path, rg_path):
        _write_envi_header(path, nrows, ncols, np.dtype("float32"))
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
