"""Tests for orbit module."""

from __future__ import annotations

import numpy as np
import pytest

from capella_reader import Time
from capella_reader.adapters import isce3 as isce3_adapter
from capella_reader.orbit import StateVector
from capella_reader.slc import CapellaSLC

try:
    isce3 = pytest.importorskip("isce3", reason="isce3 not installed")
    from capella_reader.adapters.isce3 import get_orbit
except ImportError:
    isce3 = None


@pytest.mark.skipif(not isce3, reason="isce3 not installed")
class TestGetOrbit:
    """Tests for get_orbit function."""

    def test_basic_creation(self):
        """Test creating ISCE3 orbit from uniform state vectors."""
        svs = [
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
            StateVector(
                time=Time("2024-01-01T12:00:20.000000000"),
                position=[6378137.0, 150000.0, 0.0],
                velocity=[0.0, 7500.0, 0.0],
            ),
        ]

        orbit = get_orbit(svs)

        assert isinstance(orbit, isce3.core.Orbit)
        assert orbit.size == len(svs)

    def test_non_uniform_triggers_warning(self):
        """Test that non-uniform state vectors trigger interpolation warning."""
        svs = [
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
            StateVector(
                time=Time("2024-01-01T12:00:25.000000000"),
                position=[6378137.0, 187500.0, 0.0],
                velocity=[0.0, 7500.0, 0.0],
            ),
        ]

        with pytest.warns(UserWarning, match="not uniformly sampled"):
            orbit = get_orbit(svs)

        assert isinstance(orbit, isce3.core.Orbit)

    def test_empty_state_vectors(self):
        """Test that empty state vectors raise ValueError."""
        with pytest.raises(ValueError, match="No state vectors"):
            get_orbit([])

    def test_reference_epoch(self):
        """Test that reference epoch is set to first state vector time."""
        svs = [
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
        ]

        orbit = get_orbit(svs)

        ref_epoch_str = str(orbit.reference_epoch)
        assert "2024-01-01" in ref_epoch_str
        assert "12:00:00" in ref_epoch_str

    def test_non_uniform_spacing_with_fractional_seconds(self):
        """Test that fractional second spacing (like 0.6s) is handled correctly.

        This test reproduces the bug where floating point errors in nanosecond
        conversion caused ISCE3 to reject orbits with spacing like 0.6 seconds.
        The error was: "non-uniform spacing between state vectors encountered -
        interval between state vector at position 5 and state vector at position 6
        is 0.599999 s, expected 0.599999 s"

        The bug occurred because converting float seconds to nanoseconds accumulated
        floating point errors, creating variations at the nanosecond level that
        ISCE3's strict validation detected.
        """
        # Create state vectors with non-uniform spacing similar to real Capella data
        # The spacing is approximately 0.6 seconds but varies slightly
        base_time = Time("2024-01-01T12:00:00.000000000")
        svs = []
        for i in range(108):  # Similar to the error case which had 107 state vectors
            # Add slight variations at nanosecond level to mimic real data
            ns_offset = 599999455 if i % 2 == 0 else 599999454
            time_offset_s = i * ns_offset / 1e9

            t = Time(
                base_time.as_numpy() + np.timedelta64(int(time_offset_s * 1e9), "ns")
            )
            svs.append(
                StateVector(
                    time=t,
                    position=[6378137.0 + i * 100, i * 100, 0.0],
                    velocity=[0.0, 7500.0, 0.0],
                )
            )

        # This should not raise ValueError from ISCE3
        # The interpolate_orbit should fix the spacing to be truly uniform
        with pytest.warns(UserWarning, match="not uniformly sampled"):
            orbit = get_orbit(svs)

        # Verify the orbit was created successfully
        assert isinstance(orbit, isce3.core.Orbit)

        # Verify that ISCE3 accepts it (no ValueError about non-uniform spacing)
        # If the bug exists, ISCE3 constructor would raise:
        # ValueError: non-uniform spacing between state vectors encountered
        assert orbit.size > 0


def test_get_radar_grid(metadata_file) -> None:
    slc = CapellaSLC.from_file(metadata_file)
    if slc.meta.collect.image.is_pfa:
        pytest.skip("PFA mode not supported for isce")

    grid = isce3_adapter.get_radar_grid(slc)
    assert grid.length == slc.shape[0]
    assert grid.width == slc.shape[1]
    assert grid.wavelength == slc.wavelength
    assert grid.prf == (1 / slc.delta_line_time)
    assert grid.starting_range == slc.starting_range


def test_get_doppler_lut2d(metadata_file) -> None:
    slc = CapellaSLC.from_file(metadata_file)
    if slc.meta.collect.image.is_pfa:
        pytest.skip("PFA mode not supported for isce")
    lut = isce3_adapter.get_doppler_lut2d(slc)
    # Check that out sample files have a doppler lookup +/- 500 Hz
    assert -500.0 < lut.eval(0.0, slc.range_to_first_sample) < 500
    # TODO: better verification here by comparing with usage?


def test_get_doppler_poly(metadata_file) -> None:
    slc = CapellaSLC.from_file(metadata_file)
    if slc.meta.collect.image.is_pfa:
        pytest.skip("PFA mode not supported for isce")

    poly = isce3_adapter.get_doppler_poly(slc)
    np.testing.assert_array_equal(
        poly.coeffs, slc.frequency_doppler_centroid_polynomial.coefficients
    )

    lut = isce3_adapter.get_doppler_lut2d(slc, n_az=4, method="bilinear")
    assert lut.data.shape == (4, slc.shape[1] + 2)
    assert lut.interp_method.name == "BILINEAR"
    assert lut.bounds_error is True


def test_get_attitude(metadata_file) -> None:
    slc = CapellaSLC.from_file(metadata_file)
    pointing_samples = slc.collect.pointing
    attitude = isce3_adapter.get_attitude(pointing_samples=pointing_samples)
    assert attitude is not None
    # TODO: better verifying
