#!/usr/bin/env python
"""Restore the Doppler deramping phase in a Capella spotlight SLC.

Capella spotlight SLC products are deramped and basebanded by the on-ground
processor: a geometry-dependent phase is removed so that the Doppler spectrum
sits at baseband. This is efficient for storage but breaks interferometric
techniques like phase linking because the phase of a pixel no longer
corresponds to the two-way slant range between the antenna and the target.

The removed phase, for a target P, is

    phi_P = -4 * pi / lambda * ( |ARP - P| - |ARP - P0| )

where ARP is the annotated Antenna Reference Position, P0 is the Scene
Reference Point (both in ECEF, both from the SLC metadata), and P is the
ECEF position of the ground target underneath the pixel. Multiplying the
deramped SLC by ``exp(-1j * phi_P)`` returns it to zero-Doppler geometry.

This is a *preprocessing* step - feed the restored SLC into a normal
coregistration pipeline (e.g. ``coregister_isce3.py``) afterwards. See
``spotlight_phase_restoration.md`` for a longer explanation.

Dependencies: isce3, capella-reader, numpy, gdal
Optional: sardem (auto-downloads a Copernicus DEM if --dem-file is omitted)

Usage
-----
python restore_spotlight_phase.py SLC.tif [--dem-file DEM.tif] [--output-dir ./restore]

"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from os import fsdecode
from pathlib import Path

import numpy as np
from coreg_utils import create_dem, run_geometry
from osgeo import gdal
from pyproj import Transformer

from capella_reader import CapellaSLC

gdal.UseExceptions()


# ---------------------------------------------------------------------------
# 1. Phase restoration (the actual point of this example)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpotlightGeometry:
    """Reference geometry for spotlight phase restoration.

    Holds the Antenna Reference Position (ARP), the Scene Reference Point (P0,
    in the Capella spec "reference_target_position"), and the radar wavelength.
    Everything is in ECEF / meters.
    """

    reference_antenna_position: np.ndarray  # shape (3,)
    scene_reference_point: np.ndarray  # shape (3,)
    wavelength: float

    @classmethod
    def from_capella_slc(cls, slc: CapellaSLC) -> SpotlightGeometry:
        image = slc.meta.collect.image
        if image.reference_antenna_position is None:
            msg = "SLC metadata is missing reference_antenna_position"
            raise ValueError(msg)
        if image.reference_target_position is None:
            msg = "SLC metadata is missing reference_target_position"
            raise ValueError(msg)
        return cls(
            reference_antenna_position=image.reference_antenna_position.as_array(),
            scene_reference_point=image.reference_target_position.as_array(),
            wavelength=slc.wavelength,
        )

    @property
    def reference_range(self) -> float:
        """Slant range from the antenna to the scene reference point (meters)."""
        return float(
            np.linalg.norm(self.reference_antenna_position - self.scene_reference_point)
        )


def compute_restoration_phase(
    geometry: SpotlightGeometry,
    target_ecef: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """phi_P = -4 * pi / lambda * ( |ARP - P| - |ARP - P0| )."""
    x, y, z = target_ecef
    arp_x, arp_y, arp_z = geometry.reference_antenna_position
    r = np.sqrt((x - arp_x) ** 2 + (y - arp_y) ** 2 + (z - arp_z) ** 2)
    return (-4.0 * np.pi / geometry.wavelength) * (r - geometry.reference_range)


def apply_spotlight_phase_restoration(
    slc_file: Path,
    geometry_vrt: Path,
    output_file: Path,
    *,
    data_file: Path | None = None,
    lines_per_block: int = 1024,
) -> Path:
    """Restore the deramping phase, block by block, to a corrected GeoTIFF.

    Parameters
    ----------
    slc_file
        Capella spotlight SLC supplying metadata (ARP, SRP, wavelength).
    geometry_vrt
        3-band VRT with band 1 = lon, 2 = lat, 3 = height. Must have the
        same shape as ``data_file`` (i.e. the reference grid when correcting
        a coregistered secondary).
    output_file
        Destination GeoTIFF (complex64).
    data_file
        Pixel data to correct. Defaults to ``slc_file``. For a coregistered
        secondary, pass the resampled SLC here while keeping ``slc_file``
        pointing at the original secondary for its ARP/SRP.
    lines_per_block
        Row block height for processing.
    """
    slc = CapellaSLC.from_file(slc_file)
    geometry = SpotlightGeometry.from_capella_slc(slc)
    if data_file is None:
        data_file = slc_file

    geo_ds = gdal.Open(fsdecode(geometry_vrt))
    slc_ds = gdal.Open(fsdecode(data_file))
    rows, cols = slc_ds.RasterYSize, slc_ds.RasterXSize
    if (geo_ds.RasterYSize, geo_ds.RasterXSize) != (rows, cols):
        msg = (
            f"Geometry shape ({geo_ds.RasterYSize}, {geo_ds.RasterXSize}) "
            f"does not match SLC shape ({rows}, {cols})"
        )
        raise ValueError(msg)

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        fsdecode(output_file),
        cols,
        rows,
        1,
        gdal.GDT_CFloat32,
        options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"],
    )

    lon_band = geo_ds.GetRasterBand(1)
    lat_band = geo_ds.GetRasterBand(2)
    hgt_band = geo_ds.GetRasterBand(3)
    slc_band = slc_ds.GetRasterBand(1)
    out_band = out_ds.GetRasterBand(1)

    t0 = time.time()
    # Setup LLH to ECEF transformation. always_xy keeps lon/lat order.
    epsg_wgs84_ecef = 4978  # geocentric Cartesian XYZ on WGS84
    transformer_llh_ecef = Transformer.from_crs(4326, epsg_wgs84_ecef, always_xy=True)
    for r0 in range(0, rows, lines_per_block):
        nrow = min(lines_per_block, rows - r0)
        lon = lon_band.ReadAsArray(0, r0, cols, nrow)
        lat = lat_band.ReadAsArray(0, r0, cols, nrow)
        hgt = hgt_band.ReadAsArray(0, r0, cols, nrow)
        slc_data = slc_band.ReadAsArray(0, r0, cols, nrow)

        target_ecef = transformer_llh_ecef.transform(lon, lat, hgt, radians=False)

        phi = compute_restoration_phase(geometry, target_ecef)
        corrected = (slc_data * np.exp(-1j * phi)).astype(np.complex64, copy=False)

        out_band.WriteArray(corrected, 0, r0)
        print(f"  block {r0 + nrow}/{rows}", end="\r")
    print()
    print(f"  phase restoration took {time.time() - t0:.1f} s")

    # Preserve the Capella TIFF metadata tag so downstream readers still work.
    out_ds.SetMetadataItem(
        "TIFFTAG_IMAGEDESCRIPTION",
        slc_ds.GetMetadataItem("TIFFTAG_IMAGEDESCRIPTION"),
    )
    geo_ds = slc_ds = out_ds = None
    return output_file


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore the Doppler deramping phase in a Capella spotlight SLC.",
    )
    parser.add_argument("slc", type=Path, help="Capella spotlight SLC (GeoTIFF)")
    parser.add_argument(
        "--dem-file",
        type=Path,
        default=None,
        help="DEM in EPSG:4326 (auto-downloaded via sardem if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("spotlight_restore_output"),
        help="Directory for intermediate files (DEM, geometry layers)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SLC path (defaults to <output-dir>/<stem>.tif)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output or output_dir / f"{args.slc.stem}.tif"
    t_start = time.time()

    print("[1/3] DEM")
    dem_file = create_dem(args.slc, output_dir, args.dem_file)

    print("[2/3] Geometry (rdr2geo)")
    geometry_vrt = run_geometry(args.slc, dem_file, output_dir)

    print("[3/3] Phase restoration")
    apply_spotlight_phase_restoration(args.slc, geometry_vrt, output_file)

    print(f"\nDone in {time.time() - t_start:.1f} s")
    print(f"Restored SLC: {output_file}")


if __name__ == "__main__":
    main()
