# Spotlight Phase Restoration

This page explains what `restore_spotlight_phase.py` does and why it is a prerequisite for interferometric processing of Capella spotlight SLCs.

## Background: spotlight deramping

SAR acquisitions collected in spotlight mode steer the antenna beam in azimuth to track the scene. This increases the dwell time, and consequently the resolution, but the beam rotation produces a linear Doppler variation across the received signal: the instantaneous Doppler frequency sweeps from one end of the antenna bandwidth to the other as slow time progresses.

![Spotlight deramping in azimuth time-frequency](spotlight_deramp_diagram.png)

To keep the product compact, the Capella processor applies a **deramping and basebanding** step during focusing. The deramp multiplies the signal by a complex exponential whose instantaneous frequency cancels the linear Doppler ramp, so the delivered SLC has its azimuth spectrum centered at zero.

The figure below shows the actual azimuth Doppler spectrogram of a Capella C14 spotlight SLC over Mt Etna, before and after restoration. The as-delivered product (left) sits at baseband, while restoration (right) puts the Doppler centroid back on its original linear track. (The sign of $k_a$ depends on the orbit / look geometry; for this overpass it happens to be positive.)

![Mt Etna spectrogram before and after restoration](etna_spectrogram.png)

## Why the phase has to be restored

The deramping step is a *pixel-wise* multiplication by a geometry-dependent complex exponential, so it mutates the phase of every pixel. Interferometric techniques (InSAR, phase linking, persistent scatterers, ...) rely on the SLC phase being the two-way propagation phase, `-4π/λ · R`, between the antenna and the scatterer. The deramped SLC no longer carries that phase, so we must restore it before forming interferograms.

For a ground target `P`, the phase that was removed is

$$
\phi_P \;=\; -\frac{4\pi}{\lambda}\,\bigl(R - R_0\bigr)
\;=\; -\frac{4\pi}{\lambda}\,\Bigl(\bigl|\mathrm{ARP} - P\bigr| \;-\; \bigl|\mathrm{ARP} - P_0\bigr|\Bigr)
$$

where

- `ARP` is the **Antenna Reference Position** (ECEF, from the SLC metadata
  field `reference_antenna_position`),
- `P_0` is the **Scene Reference Point** (ECEF, metadata field
  `reference_target_position`),
- `P` is the ECEF position of each ground target under a pixel,
- `R_0 = |ARP - P_0|` is a single scalar — the slant range from the
  antenna to the scene reference point,
- `R = |ARP - P|` is the slant range to the pixel's target.

The restored SLC is obtained by multiplying the deramped data by `exp(-1j · φ_P)`, which undoes the deramping and returns it to zero-Doppler geometry:

```python
restored = deramped * np.exp(-1j * phi_P)
```

The two reference positions (`ARP` and `P_0`) are annotated directly in the SLC metadata by the Capella processor, so the only missing ingredient is `P` — the ground position of every pixel.

## Getting `P` for every pixel

We need the ECEF coordinates of the target point underneath each SLC pixel.  That is exactly what `isce3.geometry.Rdr2Geo.topo` ("rdr2geo") computes given the radar grid, orbit, and a DEM: it intersects each range / azimuth ray with the DEM surface and writes the resulting `lon / lat / height` rasters.

`restore_spotlight_phase.py` runs rdr2geo on the input SLC using the capella-reader → isce3 adapters (`get_radar_grid`, `get_orbit`), producing a 3-band geometry VRT (band 1 = longitude, band 2 = latitude, band 3 = height). The restoration step then reads those bands block by block, converts `(lon, lat, height)` to ECEF via the WGS84 ellipsoid, evaluates `φ_P`, and multiplies the SLC.

## Pipeline

`restore_spotlight_phase.py` executes three steps:

1. **DEM** — if `--dem-file` is omitted, a Copernicus DEM covering the SLC footprint (plus a small buffer) is downloaded via [`sardem`](https://github.com/scottstanie/sardem).
2. **Geometry** — run `isce3.geometry.Rdr2Geo` on the SLC's radar grid and orbit, producing per-pixel lon / lat / height rasters and stacking them into a 3-band `geometry.vrt`.
3. **Phase restoration** — open the SLC and the geometry VRT, and for
   each row block:
   - read `lon`, `lat`, `height`, and the complex SLC tile;
   - convert to ECEF target positions (`llh_to_ecef_wgs84`);
   - compute `φ_P` with `compute_restoration_phase`;
   - multiply by `exp(-1j · φ_P)` and write the corrected tile.

   The Capella `TIFFTAG_IMAGEDESCRIPTION` tag is copied across so the restored GeoTIFF stays readable by `CapellaSLC.from_file`.

## Usage

```bash
# Auto-downloads a Copernicus DEM over the SLC footprint
python docs/examples/restore_spotlight_phase.py SPOTLIGHT.tif

# Or provide your own DEM in EPSG:4326
python docs/examples/restore_spotlight_phase.py SPOTLIGHT.tif \
    --dem-file my_dem.tif \
    --output-dir restore_out \
    --output SPOTLIGHT_restored.tif
```

After restoration the SLC can be fed into a normal InSAR coregistration
pipeline such as `coregister_isce3.py`.

## Key functions

- `SpotlightGeometry.from_capella_slc` — pull `ARP`, `P_0`, and `λ` from the SLC metadata.
- `llh_to_ecef_wgs84` — geodetic → ECEF via the WGS84 ellipsoid.
- `compute_restoration_phase` — evaluate `-4π/λ · (|ARP - P| - R_0)` for an array of ECEF target points.
- `apply_spotlight_phase_restoration` — block-wise I/O loop that ties it all together and writes a corrected complex64 GeoTIFF.

## References

- Capella Space SAR Products Format Specification v1.8 — definitions of `reference_antenna_position` and `reference_target_position`.
- `isce3.geometry.Rdr2Geo` — per-pixel radar-to-ground coordinate computation used to obtain `P`.
