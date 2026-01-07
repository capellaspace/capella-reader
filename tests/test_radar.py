"""Tests for radar module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from capella_reader import Time
from capella_reader.enums import LookSide
from capella_reader.radar import PRFEntry, Radar, RadarTimeVaryingParams


class TestRadarTimeVaryingParams:
    """Tests for RadarTimeVaryingParams."""

    def test_creation(self):
        """Test creating RadarTimeVaryingParams."""
        params = RadarTimeVaryingParams(
            start_timestamps=[Time("2024-01-01T12:00:00.000000000")],
            prf=3000.0,
            pulse_bandwidth=300e6,
            pulse_duration=1e-6,
            rank=1,
        )

        assert len(params.start_timestamps) == 1
        assert params.prf == 3000.0
        assert params.pulse_bandwidth == 300e6
        assert params.pulse_duration == 1e-6
        assert params.rank == 1

    def test_multiple_timestamps(self):
        """Test creating RadarTimeVaryingParams with multiple timestamps."""
        params = RadarTimeVaryingParams(
            start_timestamps=[
                Time("2024-01-01T12:00:00.000000000"),
                Time("2024-01-01T12:00:10.000000000"),
            ],
            prf=3000.0,
            pulse_bandwidth=300e6,
            pulse_duration=1e-6,
            rank=1,
        )

        assert len(params.start_timestamps) == 2
        assert (
            params.start_timestamps[0].as_numpy()
            < params.start_timestamps[1].as_numpy()
        )

    def test_validation_requires_fields(self):
        """Test that required fields are validated."""
        with pytest.raises(ValidationError):
            RadarTimeVaryingParams(prf=3000.0)


class TestPRFEntry:
    """Tests for PRFEntry."""

    def test_creation(self):
        """Test creating a PRFEntry."""
        entry = PRFEntry(
            start_timestamps=[Time("2024-01-01T12:00:00.000000000")],
            prf=3000.0,
        )

        assert len(entry.start_timestamps) == 1
        assert entry.prf == 3000.0

    def test_multiple_timestamps(self):
        """Test creating PRFEntry with multiple timestamps."""
        entry = PRFEntry(
            start_timestamps=[
                Time("2024-01-01T12:00:00.000000000"),
                Time("2024-01-01T12:00:10.000000000"),
                Time("2024-01-01T12:00:20.000000000"),
            ],
            prf=3000.0,
        )

        assert len(entry.start_timestamps) == 3


class TestRadar:
    """Tests for Radar."""

    def test_creation_basic(self):
        """Test creating a basic Radar configuration."""
        radar = Radar(
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

        assert radar.rank == 1
        assert radar.center_frequency == 9.65e9
        assert radar.pointing == LookSide.RIGHT
        assert radar.sampling_frequency == 600e6
        assert radar.transmit_polarization == "H"
        assert radar.receive_polarization == "H"
        assert len(radar.time_varying_parameters) == 1
        assert len(radar.prf) == 1

    def test_creation_vertical_polarization(self):
        """Test creating Radar with vertical polarization."""
        radar = Radar(
            rank=1,
            center_frequency=9.65e9,
            pointing=LookSide.LEFT,
            sampling_frequency=600e6,
            transmit_polarization="V",
            receive_polarization="V",
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

        assert radar.transmit_polarization == "V"
        assert radar.receive_polarization == "V"
        assert radar.pointing == LookSide.LEFT

    def test_creation_cross_polarization(self):
        """Test creating Radar with cross-polarization."""
        radar = Radar(
            rank=1,
            center_frequency=9.65e9,
            pointing=LookSide.RIGHT,
            sampling_frequency=600e6,
            transmit_polarization="H",
            receive_polarization="V",
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

        assert radar.transmit_polarization == "H"
        assert radar.receive_polarization == "V"

    def test_multiple_time_varying_params(self):
        """Test creating Radar with multiple time-varying parameter sets."""
        radar = Radar(
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
                ),
                RadarTimeVaryingParams(
                    start_timestamps=[Time("2024-01-01T12:00:10.000000000")],
                    prf=3500.0,
                    pulse_bandwidth=350e6,
                    pulse_duration=1.2e-6,
                    rank=2,
                ),
            ],
            prf=[
                PRFEntry(
                    start_timestamps=[Time("2024-01-01T12:00:00.000000000")],
                    prf=3000.0,
                ),
                PRFEntry(
                    start_timestamps=[Time("2024-01-01T12:00:10.000000000")],
                    prf=3500.0,
                ),
            ],
        )

        assert len(radar.time_varying_parameters) == 2
        assert radar.time_varying_parameters[0].prf == 3000.0
        assert radar.time_varying_parameters[1].prf == 3500.0
        assert len(radar.prf) == 2

    def test_validation_invalid_polarization(self):
        """Test that invalid polarization values are rejected."""
        with pytest.raises(ValidationError):
            Radar(
                rank=1,
                center_frequency=9.65e9,
                pointing=LookSide.RIGHT,
                sampling_frequency=600e6,
                transmit_polarization="X",
                receive_polarization="H",
                time_varying_parameters=[],
                prf=[],
            )
