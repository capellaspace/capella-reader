"""Geometric types: ECEF position, velocity, and attitude quaternions."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class ECEFPosition(BaseModel):
    """Earth-Centered Earth-Fixed (ECEF) position in meters."""

    x: float = Field(..., description="ECEF X coordinate [m]")
    y: float = Field(..., description="ECEF Y coordinate [m]")
    z: float = Field(..., description="ECEF Z coordinate [m]")

    @classmethod
    def from_list(cls, v: list[float]) -> ECEFPosition:
        """Create from a list of [x, y, z]."""
        if len(v) != 3:
            msg = "ECEFPosition expects 3 elements [x, y, z]"
            raise ValueError(msg)
        return cls(x=v[0], y=v[1], z=v[2])

    def as_array(self) -> np.ndarray:
        """Return as numpy array."""
        return np.array([self.x, self.y, self.z], dtype=float)

    def __sub__(self, other: ECEFPosition) -> ECEFPosition:
        """Subtract another ECEFPosition from this one."""
        return ECEFPosition(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __add__(self, other: ECEFPosition) -> ECEFPosition:
        """Add another ECEFPosition to this one."""
        return ECEFPosition(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def norm(self) -> float:
        """Return the Euclidean norm of the position vector."""
        return float(np.linalg.norm(self.as_array()))


class ECEFVelocity(BaseModel):
    """ECEF velocity in meters per second."""

    vx: float = Field(..., description="ECEF X velocity [m/s]")
    vy: float = Field(..., description="ECEF Y velocity [m/s]")
    vz: float = Field(..., description="ECEF Z velocity [m/s]")

    @classmethod
    def from_list(cls, v: list[float]) -> ECEFVelocity:
        """Create from a list of [vx, vy, vz]."""
        if len(v) != 3:
            msg = "ECEFVelocity expects 3 elements [vx, vy, vz]"
            raise ValueError(msg)
        return cls(vx=v[0], vy=v[1], vz=v[2])

    def as_array(self) -> np.ndarray:
        """Return as numpy array."""
        return np.array([self.vx, self.vy, self.vz], dtype=float)

    def __add__(self, other: ECEFVelocity) -> ECEFVelocity:
        """Add another ECEFVelocity to this one."""
        return ECEFVelocity(
            vx=self.vx + other.vx, vy=self.vy + other.vy, vz=self.vz + other.vz
        )

    def __sub__(self, other: ECEFVelocity) -> ECEFVelocity:
        """Subtract another ECEFVelocity from this one."""
        return ECEFVelocity(
            vx=self.vx - other.vx, vy=self.vy - other.vy, vz=self.vz - other.vz
        )

    def norm(self) -> float:
        """Return the Euclidean norm of the velocity vector."""
        return float(np.linalg.norm(self.as_array()))


class AttitudeQuaternion(BaseModel):
    """Spacecraft attitude quaternion components (order as given in metadata)."""

    q0: float = Field(..., description="Quaternion component q0")
    q1: float = Field(..., description="Quaternion component q1")
    q2: float = Field(..., description="Quaternion component q2")
    q3: float = Field(..., description="Quaternion component q3")

    @classmethod
    def from_list(cls, v: list[float]) -> AttitudeQuaternion:
        """Create from a list of [q0, q1, q2, q3]."""
        if len(v) != 4:
            msg = "AttitudeQuaternion expects 4 elements"
            raise ValueError(msg)
        return cls(q0=v[0], q1=v[1], q2=v[2], q3=v[3])

    def as_array(self) -> np.ndarray:
        """Return as numpy array."""
        return np.array([self.q0, self.q1, self.q2, self.q3], dtype=float)
