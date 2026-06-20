#!/usr/bin/env python
"""End-to-end coregistration of two Capella spotlight SLCs for InSAR.

Spotlight SLCs are deramped and basebanded by the on-ground processor: a
geometry-dependent phase has been removed so that the Doppler spectrum sits
at baseband. We coregister and resample the *deramped* SLCs, then apply
phase restoration last (see ``restore_spotlight_phase.py``). Restoring after
resampling keeps the input to the sinc kernel at baseband and lets the
restoration phase be evaluated analytically on the reference grid for both
SLCs.

Pipeline:

  1. DEM (auto-downloaded if --dem-file is omitted)
  2. Reference geometry (rdr2geo on the reference)
  3. geo2rdr offsets (reference geometry vs. secondary radar grid)
  4. Coarse resample of the secondary onto the reference grid
  5. Fine cross-correlation offsets + fine resample
  6. Restore reference phase
  7. Restore coregistered secondary phase (using reference's geometry but the
     secondary's own ARP / SRP for the restoration term)

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
from coreg_utils import (
    create_dem,
    run_geometry,
    write_envi_header,
)
from coregister_isce3 import (
    compute_fine_offsets,
    resample_slc,
    run_geo2rdr,
)
from osgeo import gdal
from restore_spotlight_phase import apply_spotlight_phase_restoration

gdal.UseExceptions()


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
        write_envi_header(path, nrows, ncols, np.dtype("float64"))
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

    coreg_dir = output_dir / "coreg"
    coreg_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] DEM")
    dem_file = create_dem(args.reference, output_dir, args.dem_file)

    print("[2/7] Reference geometry (rdr2geo)")
    ref_geometry = run_geometry(args.reference, dem_file, output_dir / "reference")

    # Coregister on the *deramped* SLCs - their azimuth signal is at baseband,
    # which is what ISCE3's sinc kernel handles best. Restoration happens last.
    print("[3/7] geo2rdr offsets")
    rg_off, az_off = run_geo2rdr(args.secondary, ref_geometry, coreg_dir)

    print("[4/7] Coarse resample (deramped)")
    coarse_file = coreg_dir / "coarse_resampled.tif"
    resample_slc(
        args.reference, args.secondary, rg_off, az_off, coarse_file, flatten=False
    )

    print("[5/7] Fine cross-correlation offsets")
    az_fine, rg_fine = compute_fine_offsets(args.reference, coarse_file, coreg_dir)
    rg_combined, az_combined = combine_offsets(
        rg_off, az_off, rg_fine, az_fine, coreg_dir / "combined_offsets"
    )
    sec_coreg_deramped = coreg_dir / "secondary.coregistered.deramped.tif"
    resample_slc(
        args.reference,
        args.secondary,
        rg_combined,
        az_combined,
        sec_coreg_deramped,
        flatten=False,
    )

    print("[6/7] Restore reference phase")
    ref_restored = output_dir / f"{args.reference.stem}.restored.tif"
    apply_spotlight_phase_restoration(args.reference, ref_geometry, ref_restored)

    print("[7/7] Restore coregistered secondary phase")
    sec_restored = coreg_dir / "secondary.coregistered.tif"
    apply_spotlight_phase_restoration(
        args.secondary, ref_geometry, sec_restored, data_file=sec_coreg_deramped
    )

    print(f"\nDone in {time.time() - t_start:.1f} s")
    print(f"Reference (restored)         : {ref_restored}")
    print(f"Secondary (restored + coreg) : {sec_restored}")


if __name__ == "__main__":
    main()
