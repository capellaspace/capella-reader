#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cartopy",
#     "numpy",
#     "pyproj",
#     "shapely",
# ]
# ///
"""Visualize satellite ground track and image footprint on a 2D map.

This script demonstrates how to:
- Load Capella metadata from JSON
- Extract orbit state vectors
- Convert ECEF coordinates to lat/lon
- Calculate image footprint corners
- Plot the ground track and footprint on a map
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from shapely.geometry import Polygon

from capella_reader.metadata import CapellaSLCMetadata
from capella_reader.slc import CapellaSLC


def ecef_to_latlon(positions):
    """Convert ECEF positions to latitude/longitude.

    Parameters
    ----------
    positions : np.ndarray
        Nx3 array of ECEF positions [x, y, z] in meters

    Returns
    -------
    lats : np.ndarray
        Latitudes in degrees
    lons : np.ndarray
        Longitudes in degrees
    alts : np.ndarray
        Altitudes in meters above WGS84 ellipsoid

    """
    ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
    lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
    transformer = pyproj.Transformer.from_proj(ecef, lla, always_xy=True)

    lons, lats, alts = transformer.transform(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )

    return np.array(lats), np.array(lons), np.array(alts)


def calculate_image_corners_ecef(meta):
    """Calculate the ECEF coordinates of the image corners.

    Parameters
    ----------
    meta : CapellaSLCMetadata
        Capella metadata object

    Returns
    -------
    corners : np.ndarray
        4x3 array of corner positions [x, y, z] in ECEF meters
        Order: top-left, top-right, bottom-right, bottom-left

    """

    def _get_pfa_corners(image, image_geometry):
        rows = image.rows
        cols = image.columns

        ref_row, ref_col = image_geometry.scene_reference_point_row_col
        ref_ecef = image_geometry.scene_reference_point_ecef.as_array()

        row_dir = np.array(image_geometry.row_direction)
        col_dir = np.array(image_geometry.col_direction)

        row_spacing = image.pixel_spacing_row
        col_spacing = image.pixel_spacing_column

        return np.array(
            [
                ref_ecef
                - (ref_row * row_spacing * row_dir)
                - (ref_col * col_spacing * col_dir),
                ref_ecef
                - (ref_row * row_spacing * row_dir)
                + ((cols - ref_col) * col_spacing * col_dir),
                ref_ecef
                + ((rows - ref_row) * row_spacing * row_dir)
                + ((cols - ref_col) * col_spacing * col_dir),
                ref_ecef
                + ((rows - ref_row) * row_spacing * row_dir)
                - (ref_col * col_spacing * col_dir),
            ]
        )

    def _get_slant_plane_corners(image, state_vectors, pointing):
        center_ecef = np.array(image.center_pixel.target_position.as_array())
        center_lat, center_lon, _ = ecef_to_latlon(center_ecef.reshape(1, -1))
        lat_rad = np.deg2rad(center_lat[0])
        lon_rad = np.deg2rad(center_lon[0])

        east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
        north = np.array(
            [
                -np.sin(lat_rad) * np.cos(lon_rad),
                -np.sin(lat_rad) * np.sin(lon_rad),
                np.cos(lat_rad),
            ]
        )

        center_time = image.center_pixel.center_time
        closest_sv = min(
            state_vectors,
            key=lambda sv: abs((sv.time - center_time).total_seconds()),
        )
        sat_vel = np.array(closest_sv.velocity.as_array())

        along_enu = (np.dot(sat_vel, east) * east) + (np.dot(sat_vel, north) * north)
        along_dir = along_enu / np.linalg.norm(along_enu)

        along_e = np.dot(along_dir, east)
        along_n = np.dot(along_dir, north)
        col_enu = (
            (along_n * east - along_e * north)
            if pointing == "right"
            else (-along_n * east + along_e * north)
        )
        col_dir = col_enu / np.linalg.norm(col_enu)

        row_extent = 0.5 * (image.rows - 1) * image.pixel_spacing_row
        col_extent = 0.5 * (image.columns - 1) * image.pixel_spacing_column
        row_dir = along_dir

        return np.array(
            [
                center_ecef - row_extent * row_dir - col_extent * col_dir,
                center_ecef - row_extent * row_dir + col_extent * col_dir,
                center_ecef + row_extent * row_dir + col_extent * col_dir,
                center_ecef + row_extent * row_dir - col_extent * col_dir,
            ]
        )

    image = meta.collect.image
    image_geometry = image.image_geometry

    if image_geometry.type == "pfa":
        return _get_pfa_corners(image, image_geometry)

    return _get_slant_plane_corners(
        image, meta.collect.state.state_vectors, meta.collect.radar.pointing
    )


def main(filename: Path | str, /):
    """Plot the ground track and image footprint for a Capella SLC file.

    Parameters
    ----------
    filename : Path | str
        Path to the Capella SLC file (JSON metadata or .slc file)

    """
    path = Path(filename)
    if path.suffix == ".json":
        print(f"Loading metadata from {path.name}...")
        json_str = path.read_text()
        meta = CapellaSLCMetadata.model_validate_json(json_str)
    else:
        print(f"Loading SLC file from {path.name}...")
        meta = CapellaSLC.from_file(path).meta

    state_vectors = meta.collect.state.state_vectors
    print(f"Found {len(state_vectors)} state vectors")

    positions = np.array([sv.position.as_array() for sv in state_vectors])
    sat_lats, sat_lons, sat_alts = ecef_to_latlon(positions)
    print(f"Satellite altitude: {sat_alts.mean() / 1000:.1f} km")
    print(f"Orbit direction: {meta.collect.state.direction}")

    geometry_type = meta.collect.image.image_geometry.type
    print(f"Geometry type: {geometry_type}")

    if geometry_type == "slant_plane":
        image_corners_ecef = calculate_image_corners_ecef(meta)
        corner_lats, corner_lons, _ = ecef_to_latlon(image_corners_ecef)
    else:
        print("Skipping footprint calculation for PFA geometry (not yet supported)")
        corner_lats = corner_lons = None

    center_ecef = np.array(meta.collect.image.center_pixel.target_position.as_array())
    center_lats, center_lons, _ = ecef_to_latlon(center_ecef.reshape(1, -1))

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lat_margin = 1.0
    lon_margin = 1.0

    if corner_lats is not None:
        min_lon = min(sat_lons.min(), corner_lons.min())
        max_lon = max(sat_lons.max(), corner_lons.max())
        min_lat = min(sat_lats.min(), corner_lats.min())
        max_lat = max(sat_lats.max(), corner_lats.max())
    else:
        min_lon = min(sat_lons.min(), center_lons[0])
        max_lon = max(sat_lons.max(), center_lons[0])
        min_lat = min(sat_lats.min(), center_lats[0])
        max_lat = max(sat_lats.max(), center_lats[0])

    ax.set_extent(
        [
            min_lon - lon_margin,
            max_lon + lon_margin,
            min_lat - lat_margin,
            max_lat + lat_margin,
        ],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    ax.plot(
        sat_lons,
        sat_lats,
        "r-",
        linewidth=2,
        label=f"Satellite Ground Track ({meta.collect.state.direction})",
        transform=ccrs.Geodetic(),
    )

    ax.scatter(
        sat_lons[0],
        sat_lats[0],
        c="green",
        s=100,
        marker="o",
        label="Start",
        transform=ccrs.Geodetic(),
        zorder=5,
    )
    ax.scatter(
        sat_lons[-1],
        sat_lats[-1],
        c="red",
        s=100,
        marker="s",
        label="End",
        transform=ccrs.Geodetic(),
        zorder=5,
    )

    if corner_lats is not None:
        footprint_polygon = Polygon(zip(corner_lons, corner_lats, strict=False))
        ax.add_geometries(
            [footprint_polygon],
            crs=ccrs.PlateCarree(),
            facecolor="blue",
            alpha=0.2,
            edgecolor="blue",
            linewidth=2,
            label="Image Footprint",
        )

    ax.scatter(
        center_lons[0],
        center_lats[0],
        c="blue",
        s=150,
        marker="*",
        label="Scene Center",
        transform=ccrs.Geodetic(),
        zorder=5,
    )

    ax.legend(loc="upper right")

    center_time_str = meta.collect.image.center_pixel.center_time.as_datetime()
    ax.set_title(
        "Capella SAR Ground Track and Image Footprint\n"
        f"{center_time_str.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    output_file = Path(__file__).parent / "ground_track.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {output_file}")

    plt.show()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
