#!/usr/bin/env python
"""End-to-end coregistration of two Capella spotlight SLCs for InSAR.

Spotlight SLCs are deramped and basebanded by the on-ground processor: a
geometry-dependent phase has been removed so that the Doppler spectrum sits
at baseband. Before standard InSAR coregistration we must restore the
per-pixel slant-range phase (see ``restore_spotlight_phase.py`` and
``spotlight_phase_restoration.md``). After restoration a spotlight SLC
behaves like a normal zero-Doppler stripmap SLC and can be fed into the
usual rdr2geo / geo2rdr / resample pipeline.

This script chains the two stages:

  1. DEM (auto-downloaded if --dem-file is omitted)
  2. Reference phase restoration (rdr2geo + apply restoration)
  3. Secondary phase restoration (rdr2geo + apply restoration)
  4. geo2rdr offsets (reference geometry vs. secondary radar grid)
  5. Coarse resample of the (restored) secondary onto the reference grid
  6. Fine cross-correlation offsets
  7. Fine resample with combined coarse + fine offsets

The output is a phase-restored, coregistered secondary SLC ready to form
an interferogram against the (also phase-restored) reference SLC.

Dependencies: isce3, capella-reader, numpy, scipy, gdal.
Optional: sardem (auto-downloads a Copernicus DEM).

Usage
-----
python coregister_spotlight.py REFERENCE.tif SECONDARY.tif \\
    [--dem-file DEM.tif] [--output-dir ./coreg_spotlight]

"""

from __future__ import annotations

import argparse
import time
from os import fsdecode
from pathlib import Path

import isce3
import numpy as np
from coregister_isce3 import (
    _write_envi_header,
    compute_fine_offsets,
    resample_slc,
    run_geo2rdr,
)
from osgeo import gdal
from restore_spotlight_phase import (
    apply_spotlight_phase_restoration,
    create_dem,
    run_geometry,
)

gdal.UseExceptions()


def restore_phase(slc_file: Path, dem_file: Path, slc_dir: Path) -> tuple[Path, Path]:
    """Restore the deramping phase of ``slc_file`` into ``slc_dir``.

    Returns
    -------
    restored_slc
        Path to the restored complex64 GeoTIFF.
    geometry_vrt
        Path to the 3-band lon/lat/height VRT (reused for geo2rdr when this
        SLC is the reference).
    """
    slc_dir.mkdir(parents=True, exist_ok=True)
    geometry_vrt = run_geometry(slc_file, dem_file, slc_dir)
    restored = slc_dir / f"{slc_file.stem}.tif"
    apply_spotlight_phase_restoration(slc_file, geometry_vrt, restored)
    return restored, geometry_vrt


def combine_offsets(
    rg_coarse_path: Path,
    az_coarse_path: Path,
    rg_fine_path: Path,
    az_fine_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Sum coarse (geo2rdr, float64) and fine (cross-corr, float32) offsets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rg_raster = isce3.io.Raster(fsdecode(rg_coarse_path))
    nrows, ncols = rg_raster.length, rg_raster.width
    del rg_raster

    rg_coarse = np.memmap(
        rg_coarse_path, dtype=np.float64, mode="r", shape=(nrows, ncols)
    )
    az_coarse = np.memmap(
        az_coarse_path, dtype=np.float64, mode="r", shape=(nrows, ncols)
    )
    rg_fine = np.memmap(rg_fine_path, dtype=np.float32, mode="r", shape=(nrows, ncols))
    az_fine = np.memmap(az_fine_path, dtype=np.float32, mode="r", shape=(nrows, ncols))

    rg_combined_path = output_dir / "range.off"
    az_combined_path = output_dir / "azimuth.off"
    rg_combined = np.memmap(
        rg_combined_path, mode="w+", dtype=np.float64, shape=(nrows, ncols)
    )
    az_combined = np.memmap(
        az_combined_path, mode="w+", dtype=np.float64, shape=(nrows, ncols)
    )
    rg_combined[:] = rg_coarse + rg_fine
    az_combined[:] = az_coarse + az_fine
    rg_combined.flush()
    az_combined.flush()
    for path in (rg_combined_path, az_combined_path):
        _write_envi_header(path, nrows, ncols, np.dtype("float64"))
    del rg_coarse, az_coarse, rg_fine, az_fine, rg_combined, az_combined
    return rg_combined_path, az_combined_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Coregister two Capella spotlight SLCs (phase restoration + InSAR coreg)."
        ),
    )
    parser.add_argument("reference", type=Path, help="Reference spotlight SLC")
    parser.add_argument("secondary", type=Path, help="Secondary spotlight SLC")
    parser.add_argument(
        "--dem-file",
        type=Path,
        default=None,
        help="DEM in EPSG:4326 (auto-downloaded if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("coreg_spotlight"),
        help="Output directory (intermediates + final coregistered SLC)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("[1/7] DEM")
    dem_file = create_dem(args.reference, output_dir, args.dem_file)

    print("[2/7] Reference phase restoration")
    ref_restored, ref_geometry = restore_phase(
        args.reference, dem_file, output_dir / "reference"
    )

    print("[3/7] Secondary phase restoration")
    sec_restored, _ = restore_phase(args.secondary, dem_file, output_dir / "secondary")

    coreg_dir = output_dir / "coreg"
    coreg_dir.mkdir(parents=True, exist_ok=True)

    print("[4/7] geo2rdr offsets")
    rg_off, az_off = run_geo2rdr(sec_restored, ref_geometry, coreg_dir)

    print("[5/7] Coarse resample")
    coarse_file = coreg_dir / "coarse_resampled.tif"
    resample_slc(
        ref_restored, sec_restored, rg_off, az_off, coarse_file, baseband_input=True
    )

    print("[6/7] Fine cross-correlation offsets")
    az_fine, rg_fine = compute_fine_offsets(ref_restored, coarse_file, coreg_dir)

    print("[7/7] Fine resample")
    rg_combined, az_combined = combine_offsets(
        rg_off, az_off, rg_fine, az_fine, coreg_dir / "combined_offsets"
    )
    final_output = coreg_dir / "secondary.coregistered.tif"
    resample_slc(
        ref_restored,
        sec_restored,
        rg_combined,
        az_combined,
        final_output,
        baseband_input=True,
    )

    print(f"\nDone in {time.time() - t_start:.1f} s")
    print(f"Reference (restored)         : {ref_restored}")
    print(f"Secondary (restored + coreg) : {final_output}")


if __name__ == "__main__":
    main()
