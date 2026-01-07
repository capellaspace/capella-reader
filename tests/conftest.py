"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


# Make a parametrized fixture with all the metadata files
def pytest_generate_tests(metafunc):
    """Generate parametrized tests for metadata files."""
    if "metadata_file" in metafunc.fixturenames:
        test_data_dir = Path(__file__).parent / "data"
        files = [
            test_data_dir / f
            for f in [
                "CAPELLA_C11_SM_SLC_VV_20251031191104_20251031191109_extended.json",
                "CAPELLA_C13_SP_SLC_HH_20241126045307_20241126045346_extended.json",
                "CAPELLA_C13_SP_SLC_HH_20250826023518_20250826023527_extended.json",
                "CAPELLA_C13_SP_SLC_HH_20251102104909_20251102104943_extended.json",
                "CAPELLA_C17_SM_SLC_HH_20251103180619_20251103180628_extended.json",
            ]
        ]
        metafunc.parametrize("metadata_file", files, ids=lambda p: p.name)


@pytest.fixture(scope="session")
def capella_test_files():
    """Validate and return paths to Capella SLC test data files.

    Raises
    ------
    FileNotFoundError
        If test data files are missing.

    Returns
    -------
    tuple[Path, Path]
        Paths to the two Capella SLC test files.

    """
    test_data_dir = Path(__file__).parent / "data"
    file1 = test_data_dir / "CAPELLA_C14_SM_SLC_HH_20240626150051_20240626150055.tif"
    file2 = test_data_dir / "CAPELLA_C14_SM_SLC_HH_20240629134910_20240629134915.tif"

    missing_files = []
    if not file1.exists():
        missing_files.append(file1.name)
    if not file2.exists():
        missing_files.append(file2.name)

    if missing_files:
        msg = (
            f"Test data files not found: {', '.join(missing_files)}\n\n"
            "Download test data by running:\n"
            "    pixi run download-test-data\n\n"
            "Or download manually from:\n"
            "    https://capella-open-data.s3.amazonaws.com/data/2024/6/26/"
            "CAPELLA_C14_SM_SLC_HH_20240626150051_20240626150055/"
            "CAPELLA_C14_SM_SLC_HH_20240626150051_20240626150055.tif\n"
            "    https://capella-open-data.s3.amazonaws.com/data/2024/6/29/"
            "CAPELLA_C14_SM_SLC_HH_20240629134910_20240629134915/"
            "CAPELLA_C14_SM_SLC_HH_20240629134910_20240629134915.tif"
        )
        raise FileNotFoundError(msg)

    return file1, file2


@pytest.fixture
def sample_metadata_dict():
    """Sample metadata dictionary matching Capella SLC structure."""
    return {
        "software_version": "1.0.0",
        "software_revision": "abc123",
        "processing_time": "2024-07-09T04:30:00Z",
        "processing_deployment": "production",
        "copyright": "Copyright 2025 Capella Space. All Rights Reserved.",
        "license": "https://www.capellaspace.com/data-licensing/",
        "product_version": "1.0",
        "product_type": "SLC",
        "collect": {
            "start_timestamp": "2024-07-09T04:03:29Z",
            "stop_timestamp": "2024-07-09T04:03:58Z",
            "local_datetime": "2024-07-09T12:03:43+0800",
            "local_timezone": "Asia/Shanghai",
            "platform": "capella-14",
            "mode": "spotlight",
            "collect_id": "CAPELLA_C14_SP_SLC_HH_20240709040329_20240709040358",
            "image": {
                "data_type": "CInt16",
                "length": 5.0,
                "width": 5.0,
                "rows": 1000,
                "columns": 1000,
                "pixel_spacing_row": 0.5,
                "pixel_spacing_column": 0.5,
                "algorithm": "backprojection",
                "scale_factor": 1.0,
                "range_autofocus": True,
                "azimuth_autofocus": True,
                "range_window": {
                    "name": "rectangular",
                    "parameters": {},
                    "broadening_factor": 1.0,
                },
                "processed_range_bandwidth": 1e9,
                "azimuth_window": {
                    "name": "rectangular",
                    "parameters": {},
                    "broadening_factor": 1.0,
                },
                "processed_azimuth_bandwidth": 1e9,
                "image_geometry": {
                    "type": "slant_plane",
                    "doppler_centroid_polynomial": {
                        "degree": 1,
                        "coefficients": [[0.0, 0.1], [0.2, 0.3]],
                    },
                    "first_line_time": "2024-07-09T04:03:29Z",
                    "delta_line_time": 0.001,
                    "range_to_first_sample": 800000.0,
                    "delta_range_sample": 0.5,
                },
                "center_pixel": {
                    "incidence_angle": 30.0,
                    "look_angle": 45.0,
                    "squint_angle": 0.1,
                    "layover_angle": 10.0,
                    "target_position": [1000000.0, 2000000.0, 3000000.0],
                    "center_time": "2024-07-09T04:03:43Z",
                },
                "range_resolution": 0.5,
                "ground_range_resolution": 0.5,
                "azimuth_resolution": 0.5,
                "ground_azimuth_resolution": 0.5,
                "azimuth_looks": 1.0,
                "range_looks": 1.0,
                "enl": 1.0,
                "reference_antenna_position": [5000000.0, 6000000.0, 7000000.0],
                "reference_target_position": [1000000.0, 2000000.0, 3000000.0],
                "azimuth_beam_pattern_corrected": True,
                "elevation_beam_pattern_corrected": True,
                "radiometry": "beta_nought",
                "calibration": "full",
                "calibration_id": "CAL123",
                "nesz_polynomial": {
                    "degree": 2,
                    "coefficients": [-30.0, 0.1, 0.01],
                },
                "nesz_peak": -25.0,
                "terrain_models": {"focusing": None},
                "reference_doppler_centroid": 0.0,
                "frequency_doppler_centroid_polynomial": {
                    "degree": 1,
                    "coefficients": [[0.0, 0.1], [0.2, 0.3]],
                },
                "quantization": {
                    "type": "block_adaptive_quantization",
                    "block_sample_size": 256,
                    "mean_bits": 8,
                    "std_bits": 8,
                    "sample_bits": 4,
                },
            },
            "radar": {
                "rank": 1,
                "center_frequency": 9.65e9,
                "pointing": "left",
                "sampling_frequency": 1.2e9,
                "transmit_polarization": "H",
                "receive_polarization": "H",
                "time_varying_parameters": [
                    {
                        "start_timestamps": ["2024-07-09T04:03:29Z"],
                        "prf": 5000.0,
                        "pulse_bandwidth": 1e9,
                        "pulse_duration": 1e-6,
                        "rank": 1,
                    }
                ],
                "prf": [
                    {
                        "start_timestamps": ["2024-07-09T04:03:29Z"],
                        "prf": 5000.0,
                    }
                ],
            },
            "state": {
                "coordinate_system": {"type": "ecef"},
                "direction": "ascending",
                "state_vectors": [
                    {
                        "time": "2024-07-09T04:03:29Z",
                        "position": [5000000.0, 6000000.0, 7000000.0],
                        "velocity": [1000.0, 2000.0, 3000.0],
                    }
                ],
                "source": "precise_determination",
            },
            "pointing": [
                {
                    "time": "2024-07-09T04:03:29Z",
                    "attitude": [1.0, 0.0, 0.0, 0.0],
                }
            ],
            "transmit_antenna": {
                "azimuth_beamwidth": 0.01,
                "elevation_beamwidth": 0.01,
                "gain": 40.0,
                "beam_pattern": {
                    "degree": 1,
                    "coefficients": [[1.0, 0.0], [0.0, 0.0]],
                },
            },
            "receive_antenna": {
                "azimuth_beamwidth": 0.01,
                "elevation_beamwidth": 0.01,
                "gain": 40.0,
                "beam_pattern": {
                    "degree": 1,
                    "coefficients": [[1.0, 0.0], [0.0, 0.0]],
                },
            },
        },
    }
