"""Combined visualization of satellite orbit track and image footprint.

This script creates a comprehensive map showing:
- Satellite ground track with time annotations
- Image footprint polygon
- Scene center and imaging geometry
- Geographic context with coastlines and borders
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from shapely import Polygon
from shapely.plotting import plot_polygon

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


def main():
    metadata_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "data"
        / "CAPELLA_C13_SP_SLC_HH_20241126045307_20241126045346_extended.json"
        # / "CAPELLA_C14_SM_SLC_HH_20240626150051_20240626150055.tif"
    )

    assert metadata_file.exists(), f"Test data not found: {metadata_file}"

    print(f"Loading metadata from {metadata_file.name}...")
    slc = CapellaSLC.from_file(metadata_file)
    times, positions, velocities = slc.meta.collect.state.get_state()
    times_python = slc.meta.collect.state.get_state(time_as_float=False)[0]

    meta = slc.meta

    print(f"Found {len(times)} state vectors")
    print(f"Time span: {times[0]} to {times[-1]}")

    sat_lats, sat_lons, sat_alts = ecef_to_latlon(positions)

    slc_gcps = slc.gcps
    lon = np.array([gcp.x for gcp in slc_gcps])
    lat = np.array([gcp.y for gcp in slc_gcps])

    # image_corners_ecef = calculate_image_corners_ecef(meta)
    # corner_lats, corner_lons, _ = ecef_to_latlon(image_corners_ecef)

    center_ecef = np.array(meta.collect.image.center_pixel.target_position.as_array())
    center_lats, center_lons, _ = ecef_to_latlon(center_ecef.reshape(1, -1))

    image_center_time = meta.collect.image.center_pixel.center_time
    center_idx = len(times) // 2

    print("\nOrbit Information:")
    print(f"  Direction: {meta.collect.state.direction}")
    print(f"  Mean altitude: {sat_alts.mean() / 1000:.1f} km")
    print(f"  Mean velocity: {np.linalg.norm(velocities, axis=1).mean():.0f} m/s")
    print("\nImage Information:")
    print(f"  Size: {meta.collect.image.rows} x {meta.collect.image.columns} pixels")
    print(
        f"  Dimensions: {meta.collect.image.length:.1f} x"
        f" {meta.collect.image.width:.1f} m"
    )
    print(
        f"  Incidence angle: {meta.collect.image.center_pixel.incidence_angle:.1f} deg"
    )
    print(f"  Center time: {image_center_time}")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lat_margin = 1.0
    lon_margin = 1.0
    ax.set_extent(
        [
            min(sat_lons.min(), lon.min()) - lon_margin,
            max(sat_lons.max(), lon.max()) + lon_margin,
            min(sat_lats.min(), lat.min()) - lat_margin,
            max(sat_lats.max(), lat.max()) + lat_margin,
        ],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.LAND, facecolor="wheat", alpha=0.4)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.4)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", edgecolor="gray")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue", alpha=0.3)

    gl = ax.gridlines(
        draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.plot(
        sat_lons,
        sat_lats,
        "r-",
        linewidth=2.5,
        label=f"Ground Track ({meta.collect.state.direction})",
        transform=ccrs.Geodetic(),
        zorder=3,
    )

    annotation_stride = len(sat_lons) // 5
    for i in range(0, len(sat_lons), annotation_stride):
        time_str = str(times_python[i])
        ax.annotate(
            time_str,
            xy=(sat_lons[i], sat_lats[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    ax.scatter(
        sat_lons[0],
        sat_lats[0],
        c="green",
        s=150,
        marker="o",
        edgecolors="black",
        linewidths=1,
        label="Orbit Start",
        transform=ccrs.Geodetic(),
        zorder=5,
    )
    ax.scatter(
        sat_lons[-1],
        sat_lats[-1],
        c="darkred",
        s=150,
        marker="s",
        edgecolors="black",
        linewidths=1,
        label="Orbit End",
        transform=ccrs.Geodetic(),
        zorder=5,
    )

    ax.scatter(
        sat_lons[center_idx],
        sat_lats[center_idx],
        c="orange",
        s=200,
        marker="D",
        edgecolors="black",
        linewidths=1,
        label="Satellite at Image Center Time",
        transform=ccrs.Geodetic(),
        zorder=5,
    )

    polygon = Polygon(zip(lon, lat, strict=False))
    footprint = polygon.convex_hull
    plot_polygon(footprint, transform=ccrs.Geodetic())

    ax.scatter(
        center_lons[0],
        center_lats[0],
        c="blue",
        s=250,
        marker="*",
        edgecolors="black",
        linewidths=1,
        label="Scene Center",
        transform=ccrs.Geodetic(),
        zorder=5,
    )

    ax.plot(
        [sat_lons[center_idx], center_lons[0]],
        [sat_lats[center_idx], center_lats[0]],
        "k--",
        linewidth=1.5,
        alpha=0.6,
        label="Line-of-Sight",
        transform=ccrs.Geodetic(),
        zorder=2,
    )

    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    title_text = (
        "Capella SAR Orbit and Image Footprint\n"
        f"Platform: Capella-{meta.collect.platform} | "
        f"Mode: {meta.collect.mode} | "
        f"Polarization: {slc.polarization}\n"
        f"Collection: {image_center_time!s} UTC | "
        f"Altitude: {sat_alts.mean() / 1000:.1f} km"
    )
    ax.set_title(title_text, fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    output_file = Path(__file__).parent / "orbit_footprint_map.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
