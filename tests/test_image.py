"""Tests for image module."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from capella_reader import Time
from capella_reader.geometry import ECEFPosition
from capella_reader.image import (
    CenterPixel,
    ImageMetadata,
    Quantization,
    SlantPlaneGeometry,
    TerrainModelRef,
    TerrainModels,
    Window,
)
from capella_reader.polynomials import Poly1D, Poly2D


@pytest.fixture
def minimal_slant_plane_geometry():
    """Create a minimal SlantPlaneGeometry for testing."""
    doppler_poly = Poly2D(degree=(1, 1), coefficients=[[100.0, 0.5], [0.1, 0.0]])
    return SlantPlaneGeometry(
        type="slant_plane",
        doppler_centroid_polynomial=doppler_poly,
        first_line_time=Time("2024-01-01T12:00:00.000000000"),
        delta_line_time=0.001,
        range_to_first_sample=800000.0,
        delta_range_sample=1.5,
    )


@pytest.fixture
def image_metadata_factory(minimal_slant_plane_geometry):
    """Create ImageMetadata objects with sensible defaults."""
    nesz = Poly1D(degree=2, coefficients=[1.0, 0.5, 0.1])
    doppler_poly = Poly2D(degree=(1, 1), coefficients=[[100.0, 0.5], [0.1, 0.0]])

    def create(**overrides):
        data = {
            "data_type": "CInt16",
            "length": 5000.0,
            "width": 5000.0,
            "rows": 1024,
            "columns": 2048,
            "pixel_spacing_row": 3.0,
            "pixel_spacing_column": 3.0,
            "algorithm": "backprojection",
            "scale_factor": 1.0,
            "range_window": Window(name="rectangular", broadening_factor=1.0),
            "processed_range_bandwidth": 300e6,
            "azimuth_window": Window(name="rectangular", broadening_factor=1.0),
            "processed_azimuth_bandwidth": 500.0,
            "image_geometry": minimal_slant_plane_geometry,
            "center_pixel": CenterPixel(
                incidence_angle=35.5,
                look_angle=30.0,
                squint_angle=2.5,
                target_position=[6378137.0, 0.0, 0.0],
                center_time=Time("2024-01-01T12:00:00.000000000"),
            ),
            "range_resolution": 1.0,
            "ground_range_resolution": 1.5,
            "azimuth_resolution": 1.0,
            "ground_azimuth_resolution": 1.5,
            "azimuth_looks": 1.0,
            "range_looks": 1.0,
            "enl": 1.0,
            "azimuth_beam_pattern_corrected": True,
            "elevation_beam_pattern_corrected": True,
            "radiometry": "beta_nought",
            "calibration": "full",
            "calibration_id": "cal_v1",
            "nesz_polynomial": nesz,
            "nesz_peak": -25.0,
            "reference_doppler_centroid": 100.0,
            "frequency_doppler_centroid_polynomial": doppler_poly,
            "quantization": Quantization(
                type="block_adaptive_quantization",
                block_sample_size=32,
                mean_bits=5,
                std_bits=3,
                sample_bits=4,
            ),
        }
        data.update(overrides)
        return ImageMetadata(**data)

    return create


class TestWindow:
    """Tests for Window."""

    def test_creation_basic(self):
        """Test creating a basic Window."""
        window = Window(
            name="rectangular",
            parameters={},
            broadening_factor=1.0,
        )

        assert window.name == "rectangular"
        assert window.parameters == {}
        assert window.broadening_factor == 1.0

    def test_creation_with_parameters(self):
        """Test creating Window with parameters."""
        window = Window(
            name="hamming",
            parameters={"alpha": 0.54, "beta": 0.46},
            broadening_factor=1.3,
        )

        assert window.name == "hamming"
        assert window.parameters["alpha"] == 0.54
        assert window.parameters["beta"] == 0.46
        assert window.broadening_factor == 1.3

    def test_validation_requires_fields(self):
        """Test that required fields are validated."""
        with pytest.raises(ValidationError):
            Window(name="rectangular")


class TestQuantization:
    """Tests for Quantization."""

    def test_creation(self):
        """Test creating a Quantization."""
        quant = Quantization(
            type="block_adaptive_quantization",
            block_sample_size=32,
            mean_bits=5,
            std_bits=3,
            sample_bits=4,
        )

        assert quant.type == "block_adaptive_quantization"
        assert quant.block_sample_size == 32
        assert quant.mean_bits == 5
        assert quant.std_bits == 3
        assert quant.sample_bits == 4

    def test_validation_requires_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            Quantization(type="block_adaptive_quantization")


class TestTerrainModelRef:
    """Tests for TerrainModelRef."""

    def test_creation(self):
        """Test creating a TerrainModelRef."""
        tm = TerrainModelRef(
            link="https://example.com/srtm",
            name="SRTM",
        )

        assert tm.link == "https://example.com/srtm"
        assert tm.name == "SRTM"


class TestTerrainModels:
    """Tests for TerrainModels."""

    def test_creation_empty(self):
        """Test creating empty TerrainModels."""
        tm = TerrainModels()

        assert tm.focusing is None

    def test_creation_with_focusing(self):
        """Test creating TerrainModels with focusing model."""
        ref = TerrainModelRef(link="https://example.com/srtm", name="SRTM")
        tm = TerrainModels(focusing=ref)

        assert tm.focusing is not None
        assert tm.focusing.name == "SRTM"


class TestImageGeometry:
    """Tests for ImageGeometry union types."""

    def test_creation_slant_plane(self):
        """Test creating SlantPlaneGeometry."""
        from capella_reader.image import SlantPlaneGeometry

        doppler_poly = Poly2D(
            degree=(1, 1),
            coefficients=[[100.0, 0.5], [0.1, 0.0]],
        )

        geom = SlantPlaneGeometry(
            type="slant_plane",
            doppler_centroid_polynomial=doppler_poly,
            first_line_time=Time("2024-01-01T12:00:00.000000000"),
            delta_line_time=0.001,
            range_to_first_sample=800000.0,
            delta_range_sample=1.5,
        )

        assert geom.type == "slant_plane"
        assert geom.doppler_centroid_polynomial is not None
        assert geom.first_line_time is not None
        assert geom.delta_line_time == 0.001
        assert geom.range_to_first_sample == 800000.0
        assert geom.delta_range_sample == 1.5

    def test_creation_pfa(self):
        """Test creating PFAGeometry with required and extra fields."""
        from capella_reader.image import PFAGeometry

        geom = PFAGeometry(
            type="pfa",
            scene_reference_point_row_col=(100.0, 200.0),
            scene_reference_point_ecef=[1000000.0, 2000000.0, 3000000.0],
            row_sample_spacing=1.0,
            col_sample_spacing=1.0,
            row_direction=(1.0, 0.0, 0.0),
            col_direction=(0.0, 1.0, 0.0),
            slant_plane_normal=(0.0, 0.0, 1.0),
            ground_plane_normal=(0.0, 0.0, 1.0),
            polar_angle_polynomial=Poly1D(degree=1, coefficients=[0.0, 1.0]),
            spatial_frequency_scale_factor_polynomial=Poly1D(
                degree=1, coefficients=[1.0, 0.0]
            ),
            custom_field="custom_value",
        )

        assert geom.type == "pfa"
        assert not hasattr(geom, "doppler_centroid_polynomial")
        assert hasattr(geom, "custom_field")

    def test_allows_extra_fields(self):
        """Test that extra fields are allowed for PFA geometry."""
        from capella_reader.image import PFAGeometry

        geom = PFAGeometry(
            type="pfa",
            scene_reference_point_row_col=(100.0, 200.0),
            scene_reference_point_ecef=[1000000.0, 2000000.0, 3000000.0],
            row_sample_spacing=1.0,
            col_sample_spacing=1.0,
            row_direction=(1.0, 0.0, 0.0),
            col_direction=(0.0, 1.0, 0.0),
            slant_plane_normal=(0.0, 0.0, 1.0),
            ground_plane_normal=(0.0, 0.0, 1.0),
            polar_angle_polynomial=Poly1D(degree=1, coefficients=[0.0, 1.0]),
            spatial_frequency_scale_factor_polynomial=Poly1D(
                degree=1, coefficients=[1.0, 0.0]
            ),
            pfa_specific_param=123.456,
            another_param="value",
        )

        assert geom.type == "pfa"
        assert geom.model_extra["pfa_specific_param"] == 123.456
        assert geom.model_extra["another_param"] == "value"


class TestCenterPixel:
    """Tests for CenterPixel."""

    def test_creation(self):
        """Test creating a CenterPixel."""
        cp = CenterPixel(
            incidence_angle=35.5,
            look_angle=30.0,
            squint_angle=2.5,
            layover_angle=45.0,
            target_position=ECEFPosition(x=6378137.0, y=0.0, z=0.0),
            center_time=Time("2024-01-01T12:00:00.000000000"),
        )

        assert cp.incidence_angle == 35.5
        assert cp.look_angle == 30.0
        assert cp.squint_angle == 2.5
        assert cp.layover_angle == 45.0
        assert cp.target_position.x == 6378137.0
        assert cp.center_time is not None

    def test_creation_with_list_position(self):
        """Test creating CenterPixel with position as list."""
        cp = CenterPixel(
            incidence_angle=35.5,
            look_angle=30.0,
            squint_angle=2.5,
            target_position=[6378137.0, 0.0, 0.0],
            center_time=Time("2024-01-01T12:00:00.000000000"),
        )

        assert cp.target_position.x == 6378137.0
        assert cp.target_position.y == 0.0
        assert cp.target_position.z == 0.0

    def test_layover_angle_optional(self):
        """Test that layover_angle can be None."""
        cp = CenterPixel(
            incidence_angle=35.5,
            look_angle=30.0,
            squint_angle=2.5,
            layover_angle=None,
            target_position=[6378137.0, 0.0, 0.0],
            center_time=Time("2024-01-01T12:00:00.000000000"),
        )

        assert cp.layover_angle is None


class TestImageMetadata:
    """Tests for ImageMetadata."""

    def test_shape_property(self, image_metadata_factory):
        """Test that shape property returns (rows, columns)."""
        img = image_metadata_factory()

        assert img.shape == (1024, 2048)

    def test_dtype_cint16(self, image_metadata_factory):
        """Test that dtype returns complex64 for CInt16."""
        img = image_metadata_factory()

        assert img.dtype == np.dtype("complex64")

    def test_dtype_uint16(self, image_metadata_factory):
        """Test that dtype returns uint16 for UInt16."""
        img = image_metadata_factory(data_type="UInt16")

        assert img.dtype == np.dtype("uint16")

    def test_dtype_fallback(self, image_metadata_factory):
        """Test dtype fallback for unexpected data types."""
        img = image_metadata_factory(data_type="float32")

        assert img.dtype == np.dtype("float32")

    def test_autofocus_bool(self, image_metadata_factory):
        """Test that autofocus fields accept boolean values."""
        img = image_metadata_factory(range_autofocus=True, azimuth_autofocus=False)

        assert img.range_autofocus is True
        assert img.azimuth_autofocus is False

    def test_autofocus_string(self, image_metadata_factory):
        """Test that autofocus fields accept string values."""
        img = image_metadata_factory(
            range_autofocus="global", azimuth_autofocus="local"
        )

        assert img.range_autofocus == "global"
        assert img.azimuth_autofocus == "local"

    def test_reference_positions_with_lists(self, image_metadata_factory):
        """Test that reference positions can be parsed from lists."""
        img = image_metadata_factory(
            reference_antenna_position=[6400000.0, 100.0, 200.0],
            reference_target_position=[6378137.0, 0.0, 0.0],
        )

        assert img.reference_antenna_position is not None
        assert img.reference_antenna_position.x == 6400000.0
        assert img.reference_target_position is not None
        assert img.reference_target_position.x == 6378137.0
