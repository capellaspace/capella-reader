"""Tests for collect module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from capella_reader import Time
from capella_reader.collect import Collect
from capella_reader.enums import LookSide
from capella_reader.image import (
    CenterPixel,
    ImageMetadata,
    Quantization,
    SlantPlaneGeometry,
    Window,
)
from capella_reader.orbit import (
    Antenna,
    CoordinateSystem,
    PointingSample,
    State,
    StateVector,
)
from capella_reader.polynomials import Poly1D, Poly2D
from capella_reader.radar import PRFEntry, Radar, RadarTimeVaryingParams


class TestCollect:
    """Tests for Collect."""

    @pytest.fixture
    def sample_image_metadata(self):
        """Create sample ImageMetadata for testing."""
        nesz = Poly1D(degree=2, coefficients=[1.0, 0.5, 0.1])
        doppler_poly = Poly2D(degree=(1, 1), coefficients=[[100.0, 0.5], [0.1, 0.0]])

        geometry = SlantPlaneGeometry(
            type="slant_plane",
            doppler_centroid_polynomial=doppler_poly,
            first_line_time=Time("2024-01-01T12:00:00.000000000"),
            delta_line_time=0.001,
            range_to_first_sample=800000.0,
            delta_range_sample=1.5,
        )

        return ImageMetadata(
            data_type="CInt16",
            length=5000.0,
            width=5000.0,
            rows=1024,
            columns=2048,
            pixel_spacing_row=3.0,
            pixel_spacing_column=3.0,
            algorithm="backprojection",
            scale_factor=1.0,
            range_window=Window(name="rectangular", broadening_factor=1.0),
            processed_range_bandwidth=300e6,
            azimuth_window=Window(name="rectangular", broadening_factor=1.0),
            processed_azimuth_bandwidth=500.0,
            image_geometry=geometry,
            center_pixel=CenterPixel(
                incidence_angle=35.5,
                look_angle=30.0,
                squint_angle=2.5,
                target_position=[6378137.0, 0.0, 0.0],
                center_time=Time("2024-01-01T12:00:00.000000000"),
            ),
            range_resolution=1.0,
            ground_range_resolution=1.5,
            azimuth_resolution=1.0,
            ground_azimuth_resolution=1.5,
            azimuth_looks=1.0,
            range_looks=1.0,
            enl=1.0,
            azimuth_beam_pattern_corrected=True,
            elevation_beam_pattern_corrected=True,
            radiometry="beta_nought",
            calibration="full",
            calibration_id="cal_v1",
            nesz_polynomial=nesz,
            nesz_peak=-25.0,
            reference_doppler_centroid=100.0,
            frequency_doppler_centroid_polynomial=doppler_poly,
            quantization=Quantization(
                type="block_adaptive_quantization",
                block_sample_size=32,
                mean_bits=5,
                std_bits=3,
                sample_bits=4,
            ),
        )

    @pytest.fixture
    def sample_radar(self):
        """Create sample Radar for testing."""
        return Radar(
            rank=1,
            center_frequency=9.65e9,
            pointing=LookSide.RIGHT,
            sampling_frequency=600e6,
            transmit_polarization="H",
            receive_polarization="H",
            time_varying_parameters=[
                RadarTimeVaryingParams(
                    start_timestamps=[Time("2024-01-01T12:00:00.000000000")],
                    prf=3000.0,
                    pulse_bandwidth=300e6,
                    pulse_duration=1e-6,
                    rank=1,
                )
            ],
            prf=[
                PRFEntry(
                    start_timestamps=[Time("2024-01-01T12:00:00.000000000")],
                    prf=3000.0,
                )
            ],
        )

    @pytest.fixture
    def sample_state(self):
        """Create sample State for testing."""
        return State(
            coordinate_system=CoordinateSystem(type="ecef"),
            direction="ascending",
            state_vectors=[
                StateVector(
                    time=Time("2024-01-01T12:00:00.000000000"),
                    position=[6378137.0, 0.0, 0.0],
                    velocity=[0.0, 7500.0, 0.0],
                ),
                StateVector(
                    time=Time("2024-01-01T12:00:10.000000000"),
                    position=[6378137.0, 75000.0, 0.0],
                    velocity=[0.0, 7500.0, 0.0],
                ),
            ],
            source="precise_determination",
        )

    @pytest.fixture
    def sample_antenna(self):
        """Create sample Antenna for testing."""
        beam_pattern = Poly2D(
            degree=(1, 1),
            coefficients=[[1.0, 0.0], [0.0, 0.0]],
        )
        return Antenna(
            azimuth_beamwidth=0.01,
            elevation_beamwidth=0.02,
            gain=30.0,
            beam_pattern=beam_pattern,
        )

    def test_creation(
        self, sample_image_metadata, sample_radar, sample_state, sample_antenna
    ):
        """Test creating a Collect."""
        collect = Collect(
            start_timestamp=Time("2024-01-01T12:00:00.000000000"),
            stop_timestamp=Time("2024-01-01T12:01:00.000000000"),
            local_datetime="2024-01-01 05:00:00",
            local_timezone="America/Los_Angeles",
            platform="capella-14",
            mode="spotlight",
            collect_id="CAPELLA_C14_SP_GEO_HH_20240101T120000_20240101T120100",
            image=sample_image_metadata,
            radar=sample_radar,
            state=sample_state,
            pointing=[
                PointingSample(
                    time=Time("2024-01-01T12:00:00.000000000"),
                    attitude=[1.0, 0.0, 0.0, 0.0],
                )
            ],
            transmit_antenna=sample_antenna,
            receive_antenna=sample_antenna,
        )

        assert collect.start_timestamp.as_numpy() < collect.stop_timestamp.as_numpy()
        assert collect.platform == "capella-14"
        assert collect.mode == "spotlight"
        assert collect.local_timezone == "America/Los_Angeles"
        assert len(collect.pointing) == 1
        assert collect.image.rows == 1024
        assert collect.radar.center_frequency == 9.65e9

    def test_time_ordering(
        self, sample_image_metadata, sample_radar, sample_state, sample_antenna
    ):
        """Test that start_timestamp is before stop_timestamp."""
        collect = Collect(
            start_timestamp=Time("2024-01-01T12:00:00.000000000"),
            stop_timestamp=Time("2024-01-01T12:01:00.000000000"),
            local_datetime="2024-01-01 05:00:00",
            local_timezone="America/Los_Angeles",
            platform="capella-14",
            mode="spotlight",
            collect_id="CAPELLA_C14_SP_GEO_HH_20240101T120000_20240101T120100",
            image=sample_image_metadata,
            radar=sample_radar,
            state=sample_state,
            pointing=[],
            transmit_antenna=sample_antenna,
            receive_antenna=sample_antenna,
        )

        delta = collect.stop_timestamp - collect.start_timestamp
        assert delta.total_seconds() == 60.0

    def test_multiple_pointing_samples(
        self, sample_image_metadata, sample_radar, sample_state, sample_antenna
    ):
        """Test Collect with multiple pointing samples."""
        collect = Collect(
            start_timestamp=Time("2024-01-01T12:00:00.000000000"),
            stop_timestamp=Time("2024-01-01T12:01:00.000000000"),
            local_datetime="2024-01-01 05:00:00",
            local_timezone="America/Los_Angeles",
            platform="capella-14",
            mode="spotlight",
            collect_id="CAPELLA_C14_SP_GEO_HH_20240101T120000_20240101T120100",
            image=sample_image_metadata,
            radar=sample_radar,
            state=sample_state,
            pointing=[
                PointingSample(
                    time=Time("2024-01-01T12:00:00.000000000"),
                    attitude=[1.0, 0.0, 0.0, 0.0],
                ),
                PointingSample(
                    time=Time("2024-01-01T12:00:30.000000000"),
                    attitude=[0.999, 0.01, 0.01, 0.01],
                ),
                PointingSample(
                    time=Time("2024-01-01T12:01:00.000000000"),
                    attitude=[0.998, 0.02, 0.02, 0.02],
                ),
            ],
            transmit_antenna=sample_antenna,
            receive_antenna=sample_antenna,
        )

        assert len(collect.pointing) == 3
        assert collect.pointing[0].time.as_numpy() < collect.pointing[1].time.as_numpy()
        assert collect.pointing[1].time.as_numpy() < collect.pointing[2].time.as_numpy()

    def test_different_transmit_receive_antennas(
        self, sample_image_metadata, sample_radar, sample_state
    ):
        """Test Collect with different transmit and receive antennas."""
        tx_antenna = Antenna(
            azimuth_beamwidth=0.01,
            elevation_beamwidth=0.02,
            gain=30.0,
            beam_pattern=Poly2D(degree=(1, 1), coefficients=[[1.0, 0.0], [0.0, 0.0]]),
        )
        rx_antenna = Antenna(
            azimuth_beamwidth=0.015,
            elevation_beamwidth=0.025,
            gain=32.0,
            beam_pattern=Poly2D(degree=(1, 1), coefficients=[[1.2, 0.0], [0.0, 0.0]]),
        )

        collect = Collect(
            start_timestamp=Time("2024-01-01T12:00:00.000000000"),
            stop_timestamp=Time("2024-01-01T12:01:00.000000000"),
            local_datetime="2024-01-01 05:00:00",
            local_timezone="America/Los_Angeles",
            platform="capella-14",
            mode="spotlight",
            collect_id="CAPELLA_C14_SP_GEO_HH_20240101T120000_20240101T120100",
            image=sample_image_metadata,
            radar=sample_radar,
            state=sample_state,
            pointing=[],
            transmit_antenna=tx_antenna,
            receive_antenna=rx_antenna,
        )

        assert collect.transmit_antenna.gain == 30.0
        assert collect.receive_antenna.gain == 32.0
        assert (
            collect.transmit_antenna.azimuth_beamwidth
            != collect.receive_antenna.azimuth_beamwidth
        )

    def test_validation_requires_fields(self):
        """Test that required fields are validated."""
        with pytest.raises(ValidationError):
            Collect(
                start_timestamp=Time("2024-01-01T12:00:00.000000000"),
                platform="capella-14",
            )
