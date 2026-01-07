"""Tests for orbit module."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from capella_reader import Time
from capella_reader.geometry import AttitudeQuaternion, ECEFPosition, ECEFVelocity
from capella_reader.orbit import (
    Antenna,
    CoordinateSystem,
    PointingSample,
    State,
    StateVector,
    interpolate_orbit,
    is_uniformly_sampled,
    resample_orbit_data_linear,
)
from capella_reader.polynomials import Poly2D


class TestCoordinateSystem:
    """Tests for CoordinateSystem."""

    def test_creation_basic(self):
        """Test creating a basic CoordinateSystem."""
        cs = CoordinateSystem(type="ecef")

        assert cs.type == "ecef"
        assert cs.wkt is None

    def test_creation_with_wkt(self):
        """Test creating a CoordinateSystem with WKT."""
        wkt_str = 'GEOGCS["WGS 84",DATUM["WGS_1984"]]'
        cs = CoordinateSystem(type="geographic", wkt=wkt_str)

        assert cs.type == "geographic"
        assert cs.wkt == wkt_str

    def test_validation_requires_type(self):
        """Test that type field is required."""
        with pytest.raises(ValidationError):
            CoordinateSystem()


class TestStateVector:
    """Tests for StateVector."""

    def test_creation(self):
        """Test creating a StateVector."""
        t = Time("2024-01-01T12:00:00.000000000")
        pos = ECEFPosition(x=6378137.0, y=0.0, z=0.0)
        vel = ECEFVelocity(vx=0.0, vy=7500.0, vz=0.0)

        sv = StateVector(time=t, position=pos, velocity=vel)

        assert sv.time == t
        assert sv.position == pos
        assert sv.velocity == vel

    def test_creation_with_list_position(self):
        """Test creating StateVector with position as list."""
        sv = StateVector(
            time="2024-01-01T12:00:00.000000000",
            position=[6378137.0, 0.0, 0.0],
            velocity=ECEFVelocity(vx=0.0, vy=7500.0, vz=0.0),
        )

        assert sv.position.x == 6378137.0
        assert sv.position.y == 0.0
        assert sv.position.z == 0.0

    def test_creation_with_list_velocity(self):
        """Test creating StateVector with velocity as list."""
        sv = StateVector(
            time="2024-01-01T12:00:00.000000000",
            position=ECEFPosition(x=6378137.0, y=0.0, z=0.0),
            velocity=[0.0, 7500.0, 0.0],
        )

        assert sv.velocity.vx == 0.0
        assert sv.velocity.vy == 7500.0
        assert sv.velocity.vz == 0.0

    def test_creation_with_both_lists(self):
        """Test creating StateVector with both position and velocity as lists."""
        sv = StateVector(
            time="2024-01-01T12:00:00.000000000",
            position=[6378137.0, 0.0, 0.0],
            velocity=[0.0, 7500.0, 0.0],
        )

        assert sv.position.x == 6378137.0
        assert sv.velocity.vy == 7500.0


class TestState:
    """Tests for State."""

    def test_creation(self):
        """Test creating a State object."""
        cs = CoordinateSystem(type="ecef")
        sv1 = StateVector(
            time="2024-01-01T12:00:00.000000000",
            position=[6378137.0, 0.0, 0.0],
            velocity=[0.0, 7500.0, 0.0],
        )
        sv2 = StateVector(
            time="2024-01-01T12:00:10.000000000",
            position=[6378137.0, 75000.0, 0.0],
            velocity=[0.0, 7500.0, 0.0],
        )

        state = State(
            coordinate_system=cs,
            direction="ascending",
            state_vectors=[sv1, sv2],
            source="precise_determination",
        )

        assert state.coordinate_system == cs
        assert state.direction == "ascending"
        assert len(state.state_vectors) == 2
        assert state.source == "precise_determination"

    def test_get_state_outputs(self):
        """Test state arrays for both float and datetime time outputs."""
        cs = CoordinateSystem(type="ecef")
        sv1 = StateVector(
            time="2024-01-01T12:00:00.000000000",
            position=[6378137.0, 0.0, 0.0],
            velocity=[0.0, 7500.0, 0.0],
        )
        sv2 = StateVector(
            time="2024-01-01T12:00:10.000000000",
            position=[6378137.0, 75000.0, 0.0],
            velocity=[0.0, 7500.0, 0.0],
        )
        state = State(
            coordinate_system=cs,
            direction="ascending",
            state_vectors=[sv1, sv2],
            source="precise_determination",
        )

        times_float, positions, velocities = state.get_state(time_as_float=True)
        assert times_float.dtype.kind == "f"
        assert positions.shape == (2, 3)
        assert velocities.shape == (2, 3)

        times_dt, _positions, _velocities = state.get_state(time_as_float=False)
        assert np.issubdtype(times_dt.dtype, np.datetime64)


class TestPointingSample:
    """Tests for PointingSample."""

    def test_creation(self):
        """Test creating a PointingSample."""
        t = Time("2024-01-01T12:00:00.000000000")
        att = AttitudeQuaternion(q0=1.0, q1=0.0, q2=0.0, q3=0.0)

        ps = PointingSample(time=t, attitude=att)

        assert ps.time == t
        assert ps.attitude == att

    def test_creation_with_list_attitude(self):
        """Test creating PointingSample with attitude as list."""
        ps = PointingSample(
            time="2024-01-01T12:00:00.000000000",
            attitude=[1.0, 0.0, 0.0, 0.0],
        )

        assert ps.attitude.q0 == 1.0
        assert ps.attitude.q1 == 0.0
        assert ps.attitude.q2 == 0.0
        assert ps.attitude.q3 == 0.0


class TestAntenna:
    """Tests for Antenna."""

    def test_creation(self):
        """Test creating an Antenna."""
        beam_pattern = Poly2D(
            degree=(1, 1),
            coefficients=[[1.0, 0.0], [0.0, 0.0]],
        )

        ant = Antenna(
            azimuth_beamwidth=0.01,
            elevation_beamwidth=0.02,
            gain=30.0,
            beam_pattern=beam_pattern,
        )

        assert ant.azimuth_beamwidth == 0.01
        assert ant.elevation_beamwidth == 0.02
        assert ant.gain == 30.0
        assert ant.beam_pattern == beam_pattern


class TestIsUniformlySampled:
    """Tests for is_uniformly_sampled function."""

    def test_uniform_spacing(self):
        """Test that uniformly spaced state vectors return True."""
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

        assert is_uniformly_sampled(svs)

    def test_non_uniform_spacing(self):
        """Test that non-uniformly spaced state vectors return False."""
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

        assert not is_uniformly_sampled(svs)

    def test_single_state_vector(self):
        """Test that a single state vector is not considered uniformly sampled."""
        svs = [
            StateVector(
                time=Time("2024-01-01T12:00:00.000000000"),
                position=[6378137.0, 0.0, 0.0],
                velocity=[0.0, 7500.0, 0.0],
            ),
        ]

        assert not is_uniformly_sampled(svs)


class TestResampleOrbitDataLinear:
    """Tests for resample_orbit_data_linear function."""

    def test_basic_resampling(self):
        """Test basic linear resampling."""
        t = [
            np.datetime64("2024-01-01T12:00:00", "ns"),
            np.datetime64("2024-01-01T12:00:20", "ns"),
            np.datetime64("2024-01-01T12:00:40", "ns"),
        ]
        p = np.array([[1000.0, 0.0, 0.0], [2000.0, 0.0, 0.0], [3000.0, 0.0, 0.0]])
        v = np.array([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

        t_new, p_new, v_new = resample_orbit_data_linear(t, p, v, dt_seconds=10.0)

        assert len(t_new) == 5
        assert t_new[0] == t[0]
        assert t_new[-1] >= t[-1]

        np.testing.assert_array_almost_equal(p_new[0], p[0])
        np.testing.assert_array_almost_equal(v_new[0], v[0])

    def test_preserves_endpoints(self):
        """Test that linear resampling preserves start point."""
        t = [
            np.datetime64("2024-01-01T12:00:00", "ns"),
            np.datetime64("2024-01-01T12:00:10", "ns"),
        ]
        p = np.array([[1000.0, 2000.0, 3000.0], [1100.0, 2100.0, 3100.0]])
        v = np.array([[100.0, 200.0, 300.0], [100.0, 200.0, 300.0]])

        _t_new, p_new, v_new = resample_orbit_data_linear(t, p, v, dt_seconds=5.0)

        np.testing.assert_array_almost_equal(p_new[0], p[0])
        np.testing.assert_array_almost_equal(v_new[0], v[0])

    def test_uniform_spacing_at_nanosecond_level(self):
        """Test that resampled times are uniformly spaced at nanosecond precision.

        This test verifies the fix for a bug where floating point errors in
        nanosecond conversion caused non-uniform spacing. When using fractional
        seconds like 0.6, the old code would produce timestamps that varied by
        1-2 nanoseconds due to floating point precision issues.

        The fix converts dt_seconds to integer nanoseconds once, then uses
        integer arithmetic to avoid accumulation errors.
        """
        t = [
            np.datetime64("2024-01-01T12:00:00.000000000", "ns"),
            np.datetime64("2024-01-01T12:01:00.000000000", "ns"),
        ]
        p = np.array([[1000.0, 0.0, 0.0], [2000.0, 0.0, 0.0]])
        v = np.array([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

        # Use 0.6 seconds spacing - a case that triggered the bug
        t_new, _p_new, _v_new = resample_orbit_data_linear(t, p, v, dt_seconds=0.6)

        # Verify all time differences are exactly the same at nanosecond level
        time_diffs = np.diff(t_new)
        unique_diffs = np.unique(time_diffs)

        # Should have only ONE unique time difference (uniformly sampled)
        assert len(unique_diffs) == 1, (
            f"Expected uniform spacing, but got {len(unique_diffs)} different "
            f"intervals: {unique_diffs}"
        )

        # The difference should be exactly 600000000 nanoseconds (0.6 seconds)
        expected_diff = np.timedelta64(600000000, "ns")
        assert (
            unique_diffs[0] == expected_diff
        ), f"Expected {expected_diff} spacing, got {unique_diffs[0]}"


class TestInterpolateOrbit:
    """Tests for interpolate_orbit function."""

    def test_already_uniform(self):
        """Test that uniformly sampled orbits are returned unchanged."""
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

        result = interpolate_orbit(svs)

        assert len(result) == len(svs)
        assert result[0].time == svs[0].time
        assert result[1].time == svs[1].time

    def test_linear_interpolation(self):
        """Test linear interpolation of non-uniform orbit."""
        svs = [
            StateVector(
                time=Time("2024-01-01T12:00:00.000000000"),
                position=[1000.0, 0.0, 0.0],
                velocity=[100.0, 0.0, 0.0],
            ),
            StateVector(
                time=Time("2024-01-01T12:00:10.000000000"),
                position=[2000.0, 0.0, 0.0],
                velocity=[100.0, 0.0, 0.0],
            ),
            StateVector(
                time=Time("2024-01-01T12:00:25.000000000"),
                position=[3500.0, 0.0, 0.0],
                velocity=[100.0, 0.0, 0.0],
            ),
        ]

        result = interpolate_orbit(svs)

        assert len(result) >= len(svs)
        assert is_uniformly_sampled(result)
