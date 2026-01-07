"""Tests for geometry types."""

import numpy as np
import pytest

from capella_reader.geometry import AttitudeQuaternion, ECEFPosition, ECEFVelocity


class TestECEFPosition:
    """Tests for ECEFPosition."""

    def test_creation(self):
        """Test creating an ECEFPosition."""
        pos = ECEFPosition(x=1000.0, y=2000.0, z=3000.0)

        assert pos.x == 1000.0
        assert pos.y == 2000.0
        assert pos.z == 3000.0

    def test_from_list(self):
        """Test creating from a list."""
        coords = [1000.0, 2000.0, 3000.0]
        pos = ECEFPosition.from_list(coords)

        assert pos.x == 1000.0
        assert pos.y == 2000.0
        assert pos.z == 3000.0

    def test_from_list_wrong_length(self):
        """Test that wrong length list raises a ValueError."""
        with pytest.raises(ValueError, match="expects 3 elements"):
            ECEFPosition.from_list([1000.0, 2000.0])

    def test_as_array(self):
        """Test conversion to numpy array."""
        pos = ECEFPosition(x=1000.0, y=2000.0, z=3000.0)
        arr = pos.as_array()

        expected = np.array([1000.0, 2000.0, 3000.0])
        np.testing.assert_array_equal(arr, expected)

    def test_add_subtract_and_norm(self):
        """Test vector arithmetic and norms."""
        pos1 = ECEFPosition(x=1000.0, y=2000.0, z=3000.0)
        pos2 = ECEFPosition(x=100.0, y=200.0, z=300.0)

        added = pos1 + pos2
        subtracted = pos1 - pos2

        assert added == ECEFPosition(x=1100.0, y=2200.0, z=3300.0)
        assert subtracted == ECEFPosition(x=900.0, y=1800.0, z=2700.0)
        assert pos2.norm() == pytest.approx(np.linalg.norm([100.0, 200.0, 300.0]))


class TestECEFVelocity:
    """Tests for ECEFVelocity."""

    def test_creation(self):
        """Test creating an ECEFVelocity."""
        vel = ECEFVelocity(vx=100.0, vy=200.0, vz=300.0)

        assert vel.vx == 100.0
        assert vel.vy == 200.0
        assert vel.vz == 300.0

    def test_from_list(self):
        """Test creating from a list."""
        coords = [100.0, 200.0, 300.0]
        vel = ECEFVelocity.from_list(coords)

        assert vel.vx == 100.0
        assert vel.vy == 200.0
        assert vel.vz == 300.0

    def test_from_list_wrong_length(self):
        """Test that wrong length list raises a ValueError."""
        with pytest.raises(ValueError, match="expects 3 elements"):
            ECEFVelocity.from_list([100.0, 200.0])

    def test_as_array(self):
        """Test conversion to numpy array."""
        vel = ECEFVelocity(vx=100.0, vy=200.0, vz=300.0)
        arr = vel.as_array()

        expected = np.array([100.0, 200.0, 300.0])
        np.testing.assert_array_equal(arr, expected)

    def test_add_subtract_and_norm(self):
        """Test vector arithmetic and norms."""
        vel1 = ECEFVelocity(vx=100.0, vy=200.0, vz=300.0)
        vel2 = ECEFVelocity(vx=10.0, vy=20.0, vz=30.0)

        added = vel1 + vel2
        subtracted = vel1 - vel2

        assert added == ECEFVelocity(vx=110.0, vy=220.0, vz=330.0)
        assert subtracted == ECEFVelocity(vx=90.0, vy=180.0, vz=270.0)
        assert vel2.norm() == pytest.approx(np.linalg.norm([10.0, 20.0, 30.0]))


class TestAttitudeQuaternion:
    """Tests for AttitudeQuaternion."""

    def test_creation(self):
        """Test creating an AttitudeQuaternion."""
        quat = AttitudeQuaternion(q0=1.0, q1=0.0, q2=0.0, q3=0.0)

        assert quat.q0 == 1.0
        assert quat.q1 == 0.0
        assert quat.q2 == 0.0
        assert quat.q3 == 0.0

    def test_from_list(self):
        """Test creating from a list."""
        coords = [1.0, 0.0, 0.0, 0.0]
        quat = AttitudeQuaternion.from_list(coords)

        assert quat.q0 == 1.0
        assert quat.q1 == 0.0
        assert quat.q2 == 0.0
        assert quat.q3 == 0.0

    def test_from_list_wrong_length(self):
        """Test that wrong length list raises a ValueError."""
        with pytest.raises(ValueError, match="expects 4 elements"):
            AttitudeQuaternion.from_list([1.0, 0.0, 0.0])

    def test_as_array(self):
        """Test conversion to numpy array."""
        quat = AttitudeQuaternion(q0=0.707, q1=0.707, q2=0.0, q3=0.0)
        arr = quat.as_array()

        expected = np.array([0.707, 0.707, 0.0, 0.0])
        np.testing.assert_array_almost_equal(arr, expected)
