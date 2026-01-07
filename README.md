# capella-reader

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]

[![Conda-Forge][conda-badge]][conda-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

Package for creating typed Python models from Capella Single Look Complex (SLC) GeoTIFF images and JSON metadata.

Capella embeds product metadata in the TIFF tags of its SLC GeoTIFF files[^1]; this package provides utilities for reading and working with that metadata based on the [Capella Product Specification version 1.8](https://support.capellaspace.com/hubfs/Capella_Space_SAR_Products_Format_Specification_v1.8.pdf?hsLang=en). Note that the JSON can be parsed with Python's built-in `json` module; however, this package provides typed models with validation and descriptions for each field for easier integration with other SAR processing libraries.
Additionally, this package includes an ISCE3 adapter via the `adapters.isce3` subpackage for working with the [ISCE3](https://github.com/isce-framework/isce3) processing library.

Example datasets can be found on the [Capella Open Data catalog](https://www.capellaspace.com/earth-observation/gallery).

[^1]: To work with other formats, such as CPHD and SICD, the [sarpy](https://github.com/ngageoint/sarpy) library is recommended.

## Installation

```bash
pip install capella-reader
```

## Quick Start

```python
from capella_reader import CapellaSLC

# Load a Capella SLC GeoTIFF
slc = CapellaSLC.from_file("tests/data/CAPELLA_C14_SM_SLC_HH_20240626150051_20240626150055.tif")

# Access metadata
print(slc.shape)                              # (rows, cols)
print(slc.meta.collect.platform)              # 'capella-14'
print(slc.meta.collect.image.center_pixel.incidence_angle)
```

### Loading Metadata from JSON

You can also load metadata directly from extended metadata JSON files:

```python
from pathlib import Path
from capella_reader import CapellaSLC, CapellaSLCMetadata

md_file = "CAPELLA_C13_SP_SLC_HH_20241126045307_20241126045346_extended.json"
slc = CapellaSLC.from_file(md_file)

# Or from a Python dict
import json
data = json.loads(Path(md_file).read_text())
meta = CapellaSLCMetadata.model_validate(data)

# Round-trip: dump back to JSON
json_str = meta.model_dump_json(indent=2)
```

## Examples

The `docs/examples/` directory contains visualization scripts demonstrating orbit and image footprint analysis:

### Orbit Visualization

```bash
# Combined map with orbit track and image footprint
python examples/orbit_footprint_map.py
```

See [docs/examples/README.md](docs/examples/README.md) for installation requirements and details.

### ISCE3 Integration

For users working with ISCE3, conversion utilities are available in the `adapters.isce3` subpackage:

```python
from capella_reader import CapellaSLC
from capella_reader import adapters

slc = CapellaSLC.from_file("path/to/slc.tif")

# Convert to ISCE3 data structures
radar_grid = adapters.isce3.get_radar_grid(slc)
orbit = adapters.isce3.get_orbit(slc)
doppler_poly = adapters.isce3.get_doppler_poly(slc)
doppler_lut = adapters.isce3.get_doppler_lut2d(slc)
```

These utilities require [ISCE3 to be installed separately](https://isce-framework.github.io/isce3/buildinstall/). They are not required for basic metadata parsing.

## Example slc model

```python

CapellaSLC(
    path=PosixPath('tests/data/CAPELLA_C14_SM_SLC_HH_20240626150051_20240626150055.tif'),
    meta=CapellaSLCMetadata(
        software_version='3.2.1',
        software_revision='f8884c31ba9ba129f4d838d6d02dea2c45a114ba-dirty',
        processing_time=datetime.datetime(2025, 5, 8, 17, 38, 5, 669807, tzinfo=TzInfo(UTC)),
        processing_deployment='production',
        product_version='1.10',
        product_type='SLC',
        collect=Collect(
            start_timestamp=datetime.datetime(2024, 6, 26, 15, 0, 51, 295975, tzinfo=TzInfo(UTC)),
            stop_timestamp=datetime.datetime(2024, 6, 26, 15, 0, 55, 793597, tzinfo=TzInfo(UTC)),
            local_datetime=datetime.datetime(2024, 6, 26, 9, 0, 53, 544786, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=64800))),
            local_timezone='America/Mexico_City',
            platform='capella-14',
            mode='stripmap',
            collect_id='0141157d-3cef-4db0-9239-66c8478e09e6',
            image=ImageMetadata(
                data_type='CInt16',
                length=20395.09163067352,
                width=10071.72523110632,
                rows=19191,
                columns=10631,
                pixel_spacing_row=1.0627985769047086,
                pixel_spacing_column=0.9474459172784211,
                algorithm='backprojection',
                scale_factor=0.004915546693498318,
                range_autofocus=None,
                azimuth_autofocus=None,
                range_window=Window(name='rectangular', parameters={}, broadening_factor=0.8844848400382688),
                processed_range_bandwidth=200000000.0,
                azimuth_window=Window(name='antenna-taper', parameters={'proc_beamwidth': 0.011835009078692163}, broadening_factor=0.9743063402100489),
                processed_azimuth_bandwidth=5022.581708747391,
                image_geometry=ImageGeometry(
                    type='slant_plane',
                    doppler_centroid_polynomial=Poly2D(degree=(3, 3), coefficients=array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])),
                    first_line_time=datetime.datetime(2024, 6, 26, 15, 0, 51, 889124, tzinfo=TzInfo(UTC)),
                    delta_line_time=0.00016105733333333333,
                    range_to_first_sample=761275.5576171188,
                    delta_range_sample=0.6171875
                ),
                center_pixel=CenterPixel(
                    incidence_angle=40.93041272468091,
                    look_angle=36.797539101417286,
                    squint_angle=-0.05586304095603156,
                    layover_angle=0.05014323818914027,
                    target_position=ECEFPosition(x=-940324.8666604777, y=-5946543.280362384, z=2098893.4900432685),
                    center_time=datetime.datetime(2024, 6, 26, 15, 0, 53, 544786, tzinfo=TzInfo(UTC))
                ),
                range_resolution=0.6629047106470235,
                ground_range_resolution=1.011849005778887,
                azimuth_resolution=1.280051064876099,
                ground_azimuth_resolution=1.2800517167779046,
                azimuth_looks=1.0,
                range_looks=1.0,
                enl=1.0,
                reference_antenna_position=ECEFPosition(x=-684525.5142771972, y=-6655343.817559726, z=1969607.6195497077),
                reference_target_position=ECEFPosition(x=-942416.8362992472, y=-5947750.607837195, z=2101298.743469007),
                azimuth_beam_pattern_corrected=True,
                elevation_beam_pattern_corrected=True,
                radiometry='beta_nought',
                calibration='full',
                calibration_id='calibration_bundle/61b7b4ca-81a8-45e2-932e-11e1489c02c6',
                nesz_polynomial=Poly1D(degree=3, coefficients=array([ 2.10754181e+05, -5.51826980e-01,  3.61176889e-07,  0.00000000e+00])),
                nesz_peak=-24.100213130335916,
                terrain_models=TerrainModels(
                    focusing=TerrainModelRef(link='ExplicitInflatedWGS84[2234.0048828125]+https://en.wikipedia.org/wiki/World_Geodetic_System', name='ExplicitInflatedWGS84[2234.0048828125]')
                ),
                reference_doppler_centroid=0.0,
                frequency_doppler_centroid_polynomial=Poly2D(
                    degree=(3, 3),
                    coefficients=array([[-1.44553946e-20, -7.48708712e-15, -2.85415191e-09,
         2.73100984e-15],
       [-1.74404613e-18,  3.05520758e-17,  1.18415007e-11,
        -7.73815716e-18],
       [ 4.66622572e-23,  2.51364523e-17,  1.01345640e-11,
        -2.31147587e-17],
       [-3.83941864e-23, -1.36857307e-17, -2.95147334e-12,
         5.47845610e-18]])
                ),
                quantization=Quantization(type='block_adaptive_quantization', block_sample_size=64, mean_bits=0, std_bits=0, sample_bits=6)
            ),
            radar=Radar(
                rank=50,
                center_frequency=9649999872.0,
                pointing=<LookSide.LEFT: 'left'>,
                sampling_frequency=750000000.0,
                transmit_polarization='H',
                receive_polarization='H',
                time_varying_parameters=[
                    RadarTimeVaryingParams(
                        start_timestamps=[datetime.datetime(2024, 6, 26, 15, 0, 51, 301024, tzinfo=TzInfo(UTC))],
                        prf=9901.251518191899,
                        pulse_bandwidth=200000000.0,
                        pulse_duration=2.0197333333333334e-05,
                        rank=50
                    ),
                    RadarTimeVaryingParams(
                        start_timestamps=[
                            datetime.datetime(2024, 6, 26, 15, 0, 51, 801024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 52, 301024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 52, 801024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 53, 301024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 53, 801024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 54, 301024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 54, 801024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 55, 301024, tzinfo=TzInfo(UTC))
                        ],
                        prf=9900.72869363185,
                        pulse_bandwidth=200000000.0,
                        pulse_duration=2.0197333333333334e-05,
                        rank=50
                    )
                ],
                prf=[
                    PRFEntry(start_timestamps=[datetime.datetime(2024, 6, 26, 15, 0, 51, 301024, tzinfo=TzInfo(UTC))], prf=9901.251518191899),
                    PRFEntry(
                        start_timestamps=[
                            datetime.datetime(2024, 6, 26, 15, 0, 51, 801024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 52, 301024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 52, 801024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 53, 301024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 53, 801024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 54, 301024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 54, 801024, tzinfo=TzInfo(UTC)),
                            datetime.datetime(2024, 6, 26, 15, 0, 55, 301024, tzinfo=TzInfo(UTC))
                        ],
                        prf=9900.72869363185
                    )
                ]
            ),
            state=State(
                coordinate_system=CoordinateSystem(type='ecef'),
                direction='ascending',
                state_vectors=[
                    StateVector(
                        time=datetime.datetime(2024, 6, 26, 15, 0, 21, 199958, tzinfo=TzInfo(UTC)),
                        position=ECEFPosition(x=-850598.581307973, y=-6682238.839189964, z=1807837.287228866),
                        velocity=ECEFVelocity(vx=5117.104280428376, vy=717.1797056134662, vz=5036.528855949608)
                    ),
                    StateVector(
                        time=datetime.datetime(2024, 6, 26, 15, 0, 21, 799957, tzinfo=TzInfo(UTC)),
                        position=ECEFPosition(x=-847528.1256471712, y=-6681807.258344243, z=1810858.8158753526),
                        velocity=ECEFVelocity(vx=5117.763610991626, vy=721.4251640291432, vz=5035.248748046506)
                    ),
...
                    StateVector(
                        time=datetime.datetime(2024, 6, 26, 15, 1, 24, 799857, tzinfo=TzInfo(UTC)),
                        position=ECEFPosition(x=-523139.33015611547, y=-6622363.191185433, z=2123595.7839055755),
                        velocity=ECEFVelocity(vx=5177.000593219978, vy=1164.868056941853, vz=4889.0522055202055)
                    ),
                    StateVector(
                        time=datetime.datetime(2024, 6, 26, 15, 1, 25, 399856, tzinfo=TzInfo(UTC)),
                        position=ECEFPosition(x=-520032.9939516515, y=-6621663.012097265, z=2126528.759803662),
                        velocity=ECEFVelocity(vx=5177.469063184616, vy=1169.066041486259, vz=4887.548741599158)
                    )
                ],
                source='precise_determination'
            ),
            pointing=[
                PointingSample(
                    time=datetime.datetime(2024, 6, 26, 15, 0, 21, 199958, tzinfo=TzInfo(UTC)),
                    attitude=AttitudeQuaternion(q0=0.7514354558972666, q1=0.539089174592113, q2=0.33782447576219665, q3=0.17493464855872806)
                ),
                PointingSample(
                    time=datetime.datetime(2024, 6, 26, 15, 0, 21, 799957, tzinfo=TzInfo(UTC)),
                    attitude=AttitudeQuaternion(q0=0.7513778110356235, q1=0.5391998219192001, q2=0.3379087735187681, q3=0.1746782481659656)
                ),
                ...
                PointingSample(
                    time=datetime.datetime(2024, 6, 26, 15, 1, 25, 399856, tzinfo=TzInfo(UTC)),
                    attitude=AttitudeQuaternion(q0=0.7464363921729759, q1=0.549879633720447, q2=0.3455829675461274, q3=0.14504242371384654)
                )
            ],
            transmit_antenna=Antenna(
                azimuth_beamwidth=0.01211729820681252,
                elevation_beamwidth=0.01219859298030783,
                gain=49.58,
                beam_pattern=Poly2D(
                    degree=(7, 7),
                    coefficients=array([[ 4.96288054e+01,  3.54236916e-04, -8.23751862e+04,
        -5.46972016e+00, -3.85558272e+07,  3.86615092e+04,
         5.23604124e+11,  2.89923661e+08],
       [-3.48671932e-04,  2.20787614e+00, -5.08022213e+00,
        -9.12656092e+03,  6.91149686e+04,  5.63776331e+06,
        ...
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00],
       [-7.85159183e+07,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00]])
                )
            ),
            receive_antenna=Antenna(
                azimuth_beamwidth=0.01211729820681252,
                elevation_beamwidth=0.01219859298030783,
                gain=49.58,
                beam_pattern=Poly2D(
                    degree=(7, 7),
                    coefficients=array([[ 4.96288054e+01,  3.54236916e-04, -8.23751862e+04,
        -5.46972016e+00, -3.85558272e+07,  3.86615092e+04,
         5.23604124e+11,  2.89923661e+08],
       [-3.48671932e-04,  2.20787614e+00, -5.08022213e+00,
        -9.12656092e+03,  6.91149686e+04,  5.63776331e+06,
         ...
       [-7.85159183e+07,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00]])
                )
            )
        )
    )
)
```

## Development

Using [pixi](https://pixi.sh/) for pacakage management and task running:

```bash
# after installing pixi: curl -fsSL https://pixi.sh/install.sh | sh
pixi install
```

To create a shell for development
```bash
pixi shell -e test
pre-commit install
```

You can see all pixi tasks with:

```bash
pixi task list
```

Download test data (required for integration tests):
```bash
pixi run download-test-data        # Get the sample data images
pixi run test                      # Run all tests (requires test data)
```

Run linting and formatting:
```bash
pixi run --environment test check
pixi run --environment test test
```

## References

[Capella Product Specification version 1.8](https://support.capellaspace.com/hubfs/Capella_Space_SAR_Products_Format_Specification_v1.8.pdf?hsLang=en)

## License

See LICENSE.txt

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/scottstanie/capella-reader/workflows/CI/badge.svg
[actions-link]:             https://github.com/scottstanie/capella-reader/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/capella-reader
[conda-link]:               https://github.com/conda-forge/capella-reader-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/scottstanie/capella-reader/discussions
[pypi-link]:                https://pypi.org/project/capella-reader/
[pypi-version]:             https://img.shields.io/pypi/v/capella-reader

<!-- prettier-ignore-end -->
