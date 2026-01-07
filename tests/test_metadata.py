"""Tests for metadata parsing and models."""

from datetime import datetime
from pathlib import Path

import numpy as np

from capella_reader._time import Time
from capella_reader.collect import Collect
from capella_reader.geometry import AttitudeQuaternion, ECEFPosition, ECEFVelocity
from capella_reader.image import ImageMetadata
from capella_reader.metadata import CapellaSLCMetadata
from capella_reader.orbit import Antenna, State, StateVector
from capella_reader.polynomials import Poly1D, Poly2D
from capella_reader.radar import Radar


class TestCapellaSLCMetadata:
    """Tests for CapellaSLCMetadata parsing."""

    def test_parse_full_metadata(self, sample_metadata_dict):
        """Test parsing a complete metadata dictionary."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        assert meta.software_version == "1.0.0"
        assert meta.software_revision == "abc123"
        assert meta.product_type == "SLC"
        assert meta.processing_deployment == "production"

    def test_collect_parsing(self, sample_metadata_dict):
        """Test that collect metadata is parsed correctly."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        assert isinstance(meta.collect, Collect)
        assert meta.collect.platform == "capella-14"
        assert meta.collect.mode == "spotlight"

    def test_image_metadata_parsing(self, sample_metadata_dict):
        """Test that image metadata is parsed correctly."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        assert isinstance(meta.collect.image, ImageMetadata)
        assert meta.collect.image.rows == 1000
        assert meta.collect.image.columns == 1000
        assert meta.collect.image.data_type == "CInt16"

    def test_image_shape_property(self, sample_metadata_dict):
        """Test the shape property of ImageMetadata."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        assert meta.collect.image.shape == (1000, 1000)

    def test_image_dtype_property(self, sample_metadata_dict):
        """Test the numpy dtype property of ImageMetadata."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        assert meta.collect.image.dtype == np.dtype(np.complex64)

    def test_radar_parsing(self, sample_metadata_dict):
        """Test that radar metadata is parsed correctly."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        assert isinstance(meta.collect.radar, Radar)
        assert meta.collect.radar.center_frequency == 9.65e9
        assert meta.collect.radar.pointing == "left"

    def test_state_parsing(self, sample_metadata_dict):
        """Test that state metadata is parsed correctly."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        assert isinstance(meta.collect.state, State)
        assert meta.collect.state.direction == "ascending"
        assert len(meta.collect.state.state_vectors) == 1

    def test_state_vector_parsing(self, sample_metadata_dict):
        """Test that state vectors are parsed correctly."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        sv = meta.collect.state.state_vectors[0]
        assert isinstance(sv, StateVector)
        assert isinstance(sv.position, ECEFPosition)
        assert isinstance(sv.velocity, ECEFVelocity)
        assert sv.position.x == 5000000.0

    def test_pointing_parsing(self, sample_metadata_dict):
        """Test that pointing data is parsed correctly."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        assert len(meta.collect.pointing) == 1
        assert isinstance(meta.collect.pointing[0].attitude, AttitudeQuaternion)
        assert meta.collect.pointing[0].attitude.q0 == 1.0

    def test_antenna_parsing(self, sample_metadata_dict):
        """Test that antenna metadata is parsed correctly."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        assert isinstance(meta.collect.transmit_antenna, Antenna)
        assert isinstance(meta.collect.receive_antenna, Antenna)
        assert meta.collect.transmit_antenna.gain == 40.0

    def test_polynomial_parsing(self, sample_metadata_dict):
        """Test that polynomials are parsed correctly."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        doppler_poly = meta.collect.image.image_geometry.doppler_centroid_polynomial
        assert isinstance(doppler_poly, Poly2D)

        nesz_poly = meta.collect.image.nesz_polynomial
        assert isinstance(nesz_poly, Poly1D)

    def test_datetime_parsing(self, sample_metadata_dict):
        """Test that datetimes are parsed correctly."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        assert isinstance(meta.processing_time, datetime)
        assert isinstance(meta.collect.start_timestamp, Time)
        assert isinstance(meta.collect.local_datetime, str)

    def test_ecef_position_from_list(self, sample_metadata_dict):
        """Test that ECEF positions are converted from lists."""
        meta = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        target_pos = meta.collect.image.center_pixel.target_position
        assert isinstance(target_pos, ECEFPosition)
        assert target_pos.x == 1000000.0
        assert target_pos.y == 2000000.0
        assert target_pos.z == 3000000.0

    def test_extra_fields_ignored(self, sample_metadata_dict):
        """Test that extra unknown fields are ignored."""
        data = sample_metadata_dict.copy()
        data["unknown_field"] = "should be ignored"
        data["collect"]["unknown_collect_field"] = "also ignored"

        meta = CapellaSLCMetadata.model_validate(data)
        assert meta.software_version == "1.0.0"

    def test_load_from_json_file(self):
        """Test loading metadata from a JSON file."""
        json_file = (
            Path(__file__).parent
            / "data"
            / "CAPELLA_C13_SP_SLC_HH_20241126045307_20241126045346_extended.json"
        )

        with open(json_file) as f:
            json_str = f.read()

        meta = CapellaSLCMetadata.model_validate_json(json_str)

        assert meta.software_version == "2.64.3"
        assert meta.product_type == "SLC"
        assert meta.collect.platform == "capella-13"
        assert meta.collect.mode == "spotlight"

    def test_round_trip_json(self, sample_metadata_dict):
        """Test that metadata can be dumped to JSON and loaded back identically."""
        import json

        meta_original = CapellaSLCMetadata.model_validate(sample_metadata_dict)

        json_str = meta_original.model_dump_json()
        meta_reloaded = CapellaSLCMetadata.model_validate_json(json_str)

        assert meta_original.model_dump() == meta_reloaded.model_dump()

        json_dict = json.loads(json_str)
        meta_from_dict = CapellaSLCMetadata.model_validate(json_dict)
        assert meta_original.model_dump() == meta_from_dict.model_dump()

    def test_round_trip_real_json_file(self):
        """Test round-trip with the actual extended JSON file."""
        import json
        from pathlib import Path

        json_file = (
            Path(__file__).parent
            / "data"
            / "CAPELLA_C13_SP_SLC_HH_20241126045307_20241126045346_extended.json"
        )

        with open(json_file) as f:
            original_str = f.read()
        original_dict = json.loads(original_str)

        meta = CapellaSLCMetadata.model_validate_json(original_str)

        dumped_dict = meta.model_dump(mode="json")

        keys_to_check = [
            "software_version",
            "product_type",
            "processing_deployment",
        ]
        for key in keys_to_check:
            assert dumped_dict[key] == original_dict[key]

        assert (
            dumped_dict["collect"]["platform"] == original_dict["collect"]["platform"]
        )
        assert dumped_dict["collect"]["mode"] == original_dict["collect"]["mode"]

    def test_load_all_metadata_files(self, metadata_file):
        """Test loading all metadata files without errors.

        This test is parameterized to run separately for each metadata file
        in the test data directory, ensuring they can all be successfully
        loaded and parsed without errors.
        """
        with open(metadata_file) as f:
            json_str = f.read()

        meta = CapellaSLCMetadata.model_validate_json(json_str)

        assert meta.software_version is not None
        assert meta.product_type == "SLC"
        assert meta.collect.platform.startswith("capella-")
        assert meta.collect.mode in ["spotlight", "stripmap"]
