"""Tests for CapellaSLC wrapper."""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from capella_reader.metadata import CapellaSLCMetadata
from capella_reader.slc import (
    C_LIGHT,
    CapellaImageGeometryError,
    CapellaParseError,
    CapellaSLC,
)


class TestCapellaSLC:
    """Tests for CapellaSLC wrapper."""

    def test_creation(self, sample_metadata_dict):
        """Test creating a CapellaSLC object."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)
        slc = CapellaSLC(path=Path("/fake/path.tif"), meta=meta)

        assert slc.path == Path("/fake/path.tif")
        assert isinstance(slc.meta, CapellaSLCMetadata)

    def test_shape_property(self, sample_metadata_dict):
        """Test the shape property."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)
        slc = CapellaSLC(path=Path("/fake/path.tif"), meta=meta)

        assert slc.shape == (1000, 1000)

    def test_dtype_property(self, sample_metadata_dict):
        """Test the dtype property."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)
        slc = CapellaSLC(path=Path("/fake/path.tif"), meta=meta)

        assert slc.dtype == np.dtype(np.complex64)

    def test_metadata_properties(self, sample_metadata_dict):
        """Test derived properties from metadata."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)
        slc = CapellaSLC(path=Path("/fake/path.tif"), meta=meta)

        assert slc.range_to_first_sample == 800000.0
        assert slc.delta_range_sample == 0.5
        assert str(slc.first_line_time) == "2024-07-09T04:03:29.000000000Z"
        assert str(slc.center_time) == "2024-07-09T04:03:43.000000000Z"
        assert slc.ref_epoch == slc.center_time
        assert slc.delta_line_time == 0.001
        assert slc.sensing_start == slc.first_line_time
        assert slc.starting_range == slc.range_to_first_sample

        assert slc.prf_average == 5000.0
        assert slc.center_frequency == 9.65e9
        assert slc.wavelength == pytest.approx(C_LIGHT / 9.65e9)
        assert slc.polarization == "HH"

        doppler_poly = slc.frequency_doppler_centroid_polynomial
        np.testing.assert_array_equal(
            doppler_poly.coefficients,
            [[0.0, 0.1], [0.2, 0.3]],
        )

    def test_from_file(self, tmp_path, sample_metadata_dict):
        """Test loading from a TIFF file."""
        # Create a test TIFF file with metadata
        test_file = tmp_path / "test.tif"
        test_data = np.random.randn(100, 100).astype(np.complex64)

        # Write TIFF with metadata in ImageDescription tag
        with tifffile.TiffWriter(test_file) as tif:
            tif.write(
                test_data,
                description=json.dumps(sample_metadata_dict),
            )

        slc = CapellaSLC.from_file(test_file)

        assert slc.path == test_file
        assert isinstance(slc.meta, CapellaSLCMetadata)
        assert slc.meta.software_version == sample_metadata_dict["software_version"]

    def test_from_file_json(self, tmp_path, sample_metadata_dict):
        """Test loading from a JSON metadata file."""
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(sample_metadata_dict))

        slc = CapellaSLC.from_file(test_file)

        assert slc.path == test_file
        assert slc.meta.software_version == sample_metadata_dict["software_version"]

    def test_from_real_metadata_files(self, metadata_file):
        slc = CapellaSLC.from_file(metadata_file)
        assert slc.path == metadata_file
        assert slc.delta_range_sample > 0.0
        if slc.meta.collect.image.is_slant_plane:
            # Does not exist in pfa
            assert slc.delta_line_time > 0.0
            assert slc.range_to_first_sample > 0.0
            assert slc.sensing_start == slc.first_line_time
            assert slc.starting_range == slc.range_to_first_sample

    def test_pfa_properties_raise(self, metadata_file):
        """Test slant-plane-only properties raise for PFA metadata."""
        slc = CapellaSLC.from_file(metadata_file)
        # Only test for pfa metadata files
        if not slc.meta.collect.image.is_pfa:
            return

        # These do not:
        with pytest.raises(CapellaImageGeometryError, match="slant_plane"):
            _ = slc.range_to_first_sample
        with pytest.raises(CapellaImageGeometryError, match="slant_plane"):
            _ = slc.first_line_time
        with pytest.raises(CapellaImageGeometryError, match="slant_plane"):
            _ = slc.delta_line_time

    def test_from_file_unsupported_extension(self, tmp_path, sample_metadata_dict):
        """Test unsupported file extension raises CapellaParseError."""
        test_file = tmp_path / "test.txt"
        test_file.write_text(json.dumps(sample_metadata_dict))

        with pytest.raises(CapellaParseError, match="Unsupported file type"):
            CapellaSLC.from_file(test_file)

    def test_gcps_and_bounds(self, tmp_path, sample_metadata_dict):
        """Test reading GCPs and computing bounds."""
        test_file = tmp_path / "test_gcps.tif"
        test_data = np.random.randn(10, 10).astype(np.complex64)
        gcp_values = np.array(
            [
                0.0,
                0.0,
                0.0,
                10.0,
                20.0,
                30.0,
                5.0,
                10.0,
                0.0,
                40.0,
                50.0,
                60.0,
            ],
            dtype=np.float64,
        )
        extratags = [(33922, "d", len(gcp_values), gcp_values, False)]

        with tifffile.TiffWriter(test_file) as tif:
            tif.write(
                test_data,
                description=json.dumps(sample_metadata_dict),
                extratags=extratags,
            )

        slc = CapellaSLC.from_file(test_file)
        gcps = slc.gcps

        assert len(gcps) == 2
        assert gcps[0].row == 0.0
        assert gcps[0].col == 0.0
        assert gcps[0].x == 10.0
        assert gcps[0].y == 20.0
        assert gcps[0].z == 30.0

        assert gcps[1].row == 5.0
        assert gcps[1].col == 10.0
        assert gcps[1].x == 40.0
        assert gcps[1].y == 50.0
        assert gcps[1].z == 60.0

        assert slc.bounds == (10.0, 20.0, 40.0, 50.0)

    def test_gcps_unavailable_for_json(self, sample_metadata_dict):
        """Test that JSON-backed SLCs do not expose GCPs."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)
        slc = CapellaSLC(path=Path("/fake/path.json"), meta=meta)

        with pytest.raises(ValueError, match="No GCPs available"):
            _ = slc.gcps
