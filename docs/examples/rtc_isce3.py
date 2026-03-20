#!/usr/bin/env python
"""Radiometric Terrain Correction (RTC) for Capella SLC data using ISCE3.

Geocodes a Capella SLC to map-projected gamma0 backscatter using ISCE3,
with automatic DEM generation and squinted-mode Doppler handling.

Dependencies: isce3, capella_reader, sardem, numpy, gdal

Simplest installation:

    conda install -c conda-forge isce3 capella-reader sardem

Usage
-----
    python capella_rtc_standalone.py CAPELLA_*.tif
    python capella_rtc_standalone.py CAPELLA_*.tif --dem-file dem.tif
    python capella_rtc_standalone.py CAPELLA_*.tif --output-dir rtc_out --resolution 5
"""

from __future__ import annotations

import argparse
import time
import warnings
from os import fsdecode
from pathlib import Path

import isce3
import numpy as np
from osgeo import gdal

import capella_reader.adapters.isce3 as cr_isce3
from capella_reader import CapellaSLC

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# UTM zone selection
# ---------------------------------------------------------------------------


def get_point_epsg(lat: float, lon: float) -> int:
    """Return the EPSG code for the UTM zone containing (lat, lon)."""
    if lat >= 75.0:
        return 3413  # Arctic polar stereographic
    if lat <= -75.0:
        return 3031  # Antarctic polar stereographic
    zone = round((lon + 177) / 6.0)
    zone = max(1, min(60, zone))
    return (32600 if lat > 0 else 32700) + zone


# ---------------------------------------------------------------------------
# Geogrid helpers
# ---------------------------------------------------------------------------


def snap_coord(val: float, snap: float, round_func) -> float:
    """Snap a coordinate value to a grid."""
    return round_func(float(val) / snap) * snap


def snap_geogrid(geogrid, x_snap: float, y_snap: float):
    """Snap geogrid corners to grid spacing."""
    xmax = geogrid.start_x + geogrid.width * geogrid.spacing_x
    ymin = geogrid.start_y + geogrid.length * geogrid.spacing_y

    geogrid.start_x = snap_coord(geogrid.start_x, x_snap, np.floor)
    end_x = snap_coord(xmax, x_snap, np.ceil)
    geogrid.width = int(np.round(np.abs((end_x - geogrid.start_x) / geogrid.spacing_x)))

    geogrid.start_y = snap_coord(geogrid.start_y, y_snap, np.ceil)
    end_y = snap_coord(ymin, y_snap, np.floor)
    geogrid.length = int(
        np.round(np.abs((end_y - geogrid.start_y) / geogrid.spacing_y))
    )
    return geogrid


def create_geogrid(radar_grid, orbit, doppler, spacing_meters, epsg=None):
    """Create a geogrid from radar grid parameters."""
    x_spacing = spacing_meters
    y_spacing = -1 * np.abs(spacing_meters)

    if epsg is None:
        mid_az = radar_grid.sensing_start + 0.5 * (radar_grid.length / radar_grid.prf)
        mid_rg = radar_grid.starting_range + 0.5 * (
            radar_grid.width * radar_grid.range_pixel_spacing
        )
        dem_interp = isce3.geometry.DEMInterpolator(0.0)
        mid_doppler = float(doppler.eval(mid_az, mid_rg))
        llh = isce3.geometry.rdr2geo(
            mid_az,
            mid_rg,
            orbit,
            radar_grid.lookside,
            doppler=mid_doppler,
            wavelength=radar_grid.wavelength,
            dem=dem_interp,
        )
        lat_deg = np.rad2deg(llh[1])
        lon_deg = np.rad2deg(llh[0])
        epsg = get_point_epsg(lat_deg, lon_deg)
        print(f"  Auto-selected EPSG:{epsg} for center ({lat_deg:.2f}, {lon_deg:.2f})")

    geogrid = isce3.product.bbox_to_geogrid(
        radar_grid, orbit, doppler, x_spacing, y_spacing, epsg
    )
    return snap_geogrid(geogrid, geogrid.spacing_x, geogrid.spacing_y)


# ---------------------------------------------------------------------------
# Doppler detection
# ---------------------------------------------------------------------------


def get_doppler(slc, radar_grid, squint_threshold_hz=50_000.0):
    """Return the appropriate Doppler LUT2d for geocoding.

    Standard Capella products (stripmap, spotlight) are focused in zero-Doppler
    geometry.  Heavily squinted products (parallel stripmap, ~200+ kHz) are
    focused at the Doppler centroid.

    Returns
    -------
    doppler : isce3.core.LUT2d
    is_squinted : bool
    center_doppler_hz : float
    """
    native_doppler = cr_isce3.get_doppler_lut2d(slc)
    native_doppler.bounds_error = False

    mid_az = radar_grid.sensing_start + 0.5 * (radar_grid.length / radar_grid.prf)
    mid_rg = radar_grid.starting_range + 0.5 * (
        radar_grid.width * radar_grid.range_pixel_spacing
    )
    center_doppler_hz = abs(float(native_doppler.eval(mid_az, mid_rg)))

    if center_doppler_hz > squint_threshold_hz:
        return native_doppler, True, center_doppler_hz

    doppler = isce3.core.LUT2d()
    doppler.bounds_error = False
    return doppler, False, center_doppler_hz


# ---------------------------------------------------------------------------
# Beta0 calibration
# ---------------------------------------------------------------------------


def create_beta0_raster(slc_file: Path, output_file: Path) -> Path:
    """Convert CInt16 SLC to calibrated complex beta0 (CFloat32).

    Uses the Capella scale_factor: beta0_complex = SF * DN.
    """
    slc = CapellaSLC.from_file(slc_file)
    scale_factor = slc.meta.collect.image.scale_factor

    src_ds = gdal.Open(str(slc_file), gdal.GA_ReadOnly)
    rows, cols = src_ds.RasterYSize, src_ds.RasterXSize

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        str(output_file),
        cols,
        rows,
        1,
        gdal.GDT_CFloat32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )

    block_size = 1024
    for r0 in range(0, rows, block_size):
        nrows = min(block_size, rows - r0)
        data = src_ds.GetRasterBand(1).ReadAsArray(0, r0, cols, nrows)
        calibrated = data.astype(np.complex64) * np.float32(scale_factor)
        out_ds.GetRasterBand(1).WriteArray(calibrated, 0, r0)

    out_ds.FlushCache()
    out_ds = None
    src_ds = None
    return output_file


# ---------------------------------------------------------------------------
# Layover / shadow mask
# ---------------------------------------------------------------------------


def compute_layover_shadow_mask(
    radar_grid,
    orbit,
    dem_raster,
    geogrid,
    output_file,
    doppler=None,
    threshold=1e-8,
    numiter=20,
    lines_per_block=1024,
):
    """Compute and geocode layover/shadow mask.  Returns radar-coordinate mask."""
    if doppler is None:
        doppler = isce3.core.LUT2d()

    ellipsoid = isce3.core.Ellipsoid()

    rdr2geo_obj = isce3.geometry.Rdr2Geo(
        radar_grid,
        orbit,
        ellipsoid,
        doppler,
        threshold=threshold,
        numiter=numiter,
        lines_per_block=lines_per_block,
    )

    slantrange_mask = isce3.io.Raster(
        "layover_shadow_mask",
        radar_grid.width,
        radar_grid.length,
        1,
        gdal.GDT_Byte,
        "MEM",
    )
    rdr2geo_obj.topo(dem_raster, layover_shadow_raster=slantrange_mask)

    # Geocode the mask
    geo = isce3.geocode.GeocodeFloat32()
    geo.orbit = orbit
    geo.ellipsoid = ellipsoid
    geo.doppler = doppler
    geo.threshold_geo2rdr = threshold
    geo.numiter_geo2rdr = numiter
    geo.data_interpolator = "NEAREST"
    geo.geogrid(
        float(geogrid.start_x),
        float(geogrid.start_y),
        float(geogrid.spacing_x),
        float(geogrid.spacing_y),
        int(geogrid.width),
        int(geogrid.length),
        int(geogrid.epsg),
    )

    geocoded_mask = isce3.io.Raster(
        fsdecode(output_file),
        geogrid.width,
        geogrid.length,
        1,
        gdal.GDT_Byte,
        "GTiff",
    )
    geo.geocode(
        radar_grid=radar_grid,
        input_raster=slantrange_mask,
        output_raster=geocoded_mask,
        dem_raster=dem_raster,
        output_mode=isce3.geocode.GeocodeOutputMode.INTERP,
    )
    geocoded_mask.close_dataset()
    del geocoded_mask

    return slantrange_mask


# ---------------------------------------------------------------------------
# DEM generation
# ---------------------------------------------------------------------------


def generate_dem(slc_file: Path, output_file: Path, buffer_degrees: float = 0.5):
    """Generate a DEM covering the SLC footprint using sardem (Copernicus GLO-30)."""
    try:
        import sardem.dem
    except ImportError as exc:
        msg = "sardem is required for DEM generation. Install with: pip install sardem"
        raise ImportError(msg) from exc

    if output_file.exists():
        print(f"  DEM already exists: {output_file}")
        return output_file

    slc = CapellaSLC.from_file(slc_file)
    west, south, east, north = slc.bounds
    padded = (
        west - buffer_degrees,
        south - buffer_degrees,
        east + buffer_degrees,
        north + buffer_degrees,
    )
    print(f"  SLC bounds: ({west:.3f}, {south:.3f}, {east:.3f}, {north:.3f})")
    print(f"  DEM bounds (padded {buffer_degrees} deg): {padded}")

    sardem.dem.main(
        output_name=str(output_file),
        bbox=padded,
        data_source="COP",
        output_type="float32",
        output_format="GTiff",
    )
    print(f"  DEM generated: {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# Main RTC
# ---------------------------------------------------------------------------


def run_rtc(
    slc_file: Path,
    dem_file: Path,
    output_dir: Path,
    output_resolution_m: float = 10.0,
    output_epsg: int | None = None,
):
    """Run Radiometric Terrain Correction on a Capella SLC.

    Parameters
    ----------
    slc_file
        Path to Capella SLC (GeoTIFF with embedded metadata).
    dem_file
        Path to DEM in EPSG:4326.
    output_dir
        Directory for output files.
    output_resolution_m
        Output pixel spacing in meters.
    output_epsg
        Output EPSG code.  If None, auto-selects UTM zone.

    Returns
    -------
    dict[str, Path]
        Mapping of output layer names to file paths.
    """
    t_start = time.perf_counter()

    slc = CapellaSLC.from_file(slc_file)
    product_id = slc.sensing_start.as_datetime().strftime("%Y%m%dT%H%M%S")
    print(f"SLC: {slc_file.name}  ({slc.shape[0]}x{slc.shape[1]})")
    print(f"  Product ID: {product_id}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # DEM
    dem_raster = isce3.io.Raster(fsdecode(dem_file))
    assert dem_raster.get_epsg() == 4326, "DEM must be EPSG:4326"
    ellipsoid = isce3.core.make_projection(dem_raster.get_epsg()).ellipsoid

    # Radar grid and orbit
    radar_grid = cr_isce3.get_radar_grid(slc)
    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        orbit = cr_isce3.get_orbit(slc)

    # Doppler detection
    doppler, is_squinted, center_dop = get_doppler(slc, radar_grid)
    if is_squinted:
        print(
            f"  Squinted mode detected (Doppler = {center_dop:.0f} Hz)."
            "  Using native Doppler."
        )
    else:
        print(f"  Standard mode (Doppler = {center_dop:.0f} Hz).  Using zero Doppler.")

    # Geogrid
    geogrid = create_geogrid(
        radar_grid, orbit, doppler, output_resolution_m, output_epsg
    )
    print(
        f"  Output geogrid: {geogrid.width}x{geogrid.length} px, "
        f"EPSG:{geogrid.epsg}, spacing={geogrid.spacing_x}m"
    )

    # Output paths
    outputs: dict[str, Path] = {}
    backscatter_file = output_dir / f"{product_id}.tif"
    nlooks_file = output_dir / f"{product_id}_number_of_looks.tif"
    mask_file = output_dir / f"{product_id}_mask.tif"
    beta0_file = output_dir / f"{product_id}_beta0.tif"
    rtc_anf_file = output_dir / f"{product_id}_rtc_anf_gamma0_to_beta0.tif"
    outputs["backscatter"] = backscatter_file
    outputs["number_of_looks"] = nlooks_file
    outputs["mask"] = mask_file
    outputs["beta0"] = beta0_file
    outputs["rtc_anf"] = rtc_anf_file

    # Layover / shadow mask
    print("  Computing layover/shadow mask...")
    t0 = time.perf_counter()
    slantrange_mask = compute_layover_shadow_mask(
        radar_grid,
        orbit,
        dem_raster,
        geogrid,
        mask_file,
        doppler,
    )
    print(f"  Mask computed in {time.perf_counter() - t0:.1f}s")

    # Beta0 conversion
    print("  Converting SLC to beta0...")
    t0 = time.perf_counter()
    create_beta0_raster(slc_file, beta0_file)
    print(f"  Beta0 in {time.perf_counter() - t0:.1f}s")

    slc_raster = isce3.io.Raster(fsdecode(beta0_file))

    # Create output rasters
    out_raster = isce3.io.Raster(
        fsdecode(backscatter_file),
        geogrid.width,
        geogrid.length,
        slc_raster.num_bands,
        gdal.GDT_Float32,
        "GTiff",
    )
    out_nlooks = isce3.io.Raster(
        fsdecode(nlooks_file),
        geogrid.width,
        geogrid.length,
        1,
        gdal.GDT_Float32,
        "GTiff",
    )
    out_rtc_anf = isce3.io.Raster(
        fsdecode(rtc_anf_file),
        geogrid.width,
        geogrid.length,
        1,
        gdal.GDT_Float32,
        "GTiff",
    )

    # Select geocode object by datatype
    dtype = slc_raster.datatype()
    geo_classes = {
        gdal.GDT_Float32: isce3.geocode.GeocodeFloat32,
        gdal.GDT_Float64: isce3.geocode.GeocodeFloat64,
        gdal.GDT_CFloat32: isce3.geocode.GeocodeCFloat32,
        gdal.GDT_CFloat64: isce3.geocode.GeocodeCFloat64,
    }
    assert dtype in geo_classes, f"Unsupported datatype: {dtype}"
    geo_obj = geo_classes[dtype]()

    geo_obj.orbit = orbit
    geo_obj.ellipsoid = ellipsoid
    geo_obj.doppler = doppler
    geo_obj.threshold_geo2rdr = 1e-8
    geo_obj.numiter_geo2rdr = 20

    geo_obj.geogrid(
        geogrid.start_x,
        geogrid.start_y,
        geogrid.spacing_x,
        geogrid.spacing_y,
        geogrid.width,
        geogrid.length,
        geogrid.epsg,
    )

    # Run geocode with RTC
    print("  Geocoding with RTC...")
    t0 = time.perf_counter()
    geo_obj.geocode(
        radar_grid=radar_grid,
        input_raster=slc_raster,
        output_raster=out_raster,
        dem_raster=dem_raster,
        output_mode=isce3.geocode.GeocodeOutputMode.AREA_PROJECTION,
        geogrid_upsampling=1,
        flag_apply_rtc=True,
        input_terrain_radiometry=isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT,
        output_terrain_radiometry=isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT,
        exponent=2,
        rtc_min_value_db=-30.0,
        rtc_upsampling=2,
        rtc_algorithm=isce3.geometry.RtcAlgorithm.RTC_AREA_PROJECTION,
        abs_cal_factor=1,
        flag_upsample_radar_grid=False,
        clip_min=np.nan,
        clip_max=np.nan,
        out_geo_nlooks=out_nlooks,
        out_geo_rtc=out_rtc_anf,
        rtc_area_beta_mode=isce3.geometry.RtcAreaBetaMode.AUTO,
        dem_interp_method=isce3.core.DataInterpMethod.BIQUINTIC,
        memory_mode=isce3.core.GeocodeMemoryMode.SingleBlock,
        input_layover_shadow_mask_raster=slantrange_mask,
    )
    print(f"  Geocoding completed in {time.perf_counter() - t0:.1f}s")

    # Clean up
    del out_raster
    out_nlooks.close_dataset()
    del out_nlooks
    out_rtc_anf.close_dataset()
    del out_rtc_anf

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.1f}s.  Outputs in {output_dir}/")
    for name, path in outputs.items():
        print(f"  {name}: {path.name}")

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Radiometric Terrain Correction for Capella SLC data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("slc_file", type=Path, help="Path to Capella SLC GeoTIFF")
    parser.add_argument(
        "--dem-file",
        type=Path,
        default=None,
        help="Path to DEM (EPSG:4326). Auto-generated from SLC bounds if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./rtc_<product_id>)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=10.0,
        help="Output pixel spacing in meters",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=None,
        help="Output EPSG code (default: auto UTM)",
    )
    parser.add_argument(
        "--dem-buffer",
        type=float,
        default=0.5,
        help="DEM buffer in degrees around SLC bounds (for auto-generated DEM)",
    )
    args = parser.parse_args()

    assert args.slc_file.exists(), f"SLC file not found: {args.slc_file}"

    # Auto-generate DEM if not provided
    dem_file = args.dem_file
    if dem_file is None:
        dem_file = args.slc_file.parent / "dem.tif"
        print("No DEM provided, generating from SLC bounds...")
        generate_dem(args.slc_file, dem_file, buffer_degrees=args.dem_buffer)

    assert dem_file.exists(), f"DEM file not found: {dem_file}"

    # Default output dir
    output_dir = args.output_dir
    if output_dir is None:
        slc = CapellaSLC.from_file(args.slc_file)
        pid = slc.sensing_start.as_datetime().strftime("%Y%m%dT%H%M%S")
        output_dir = args.slc_file.parent / f"rtc_{pid}"

    run_rtc(
        slc_file=args.slc_file,
        dem_file=dem_file,
        output_dir=output_dir,
        output_resolution_m=args.resolution,
        output_epsg=args.epsg,
    )


if __name__ == "__main__":
    main()
