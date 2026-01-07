"""State vectors, pointing, orbit, and antenna models."""

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

from capella_reader._time import Time
from capella_reader.geometry import AttitudeQuaternion, ECEFPosition, ECEFVelocity
from capella_reader.polynomials import Poly2D


class CoordinateSystem(BaseModel):
    """Coordinate system specification."""

    type: str = Field(..., description="Coordinate system name (e.g. 'ecef')")
    wkt: str | None = Field(
        None,
        description=(
            "Well-Known Text (WKT) representation of coordinate system, if using"
            " geographic projection"
        ),
    )
    # https://github.com/pydantic/pydantic/discussions/9800#discussioncomment-9916979
    # TODO: exclude serializing wkt if none


class StateVector(BaseModel):
    """Platform state vector at a specific time."""

    time: Time = Field(..., description="Time of state vector (UTC)")
    position: ECEFPosition = Field(..., description="Platform position in ECEF [m]")
    velocity: ECEFVelocity = Field(..., description="Platform velocity in ECEF [m/s]")

    @field_validator("position", mode="before")
    @classmethod
    def _parse_position(cls, v: Any) -> ECEFPosition:
        if isinstance(v, list):
            return ECEFPosition.from_list(v)
        return v

    @field_validator("velocity", mode="before")
    @classmethod
    def _parse_velocity(cls, v: Any) -> ECEFVelocity:
        if isinstance(v, list):
            return ECEFVelocity.from_list(v)
        return v


class State(BaseModel):
    """Platform state ephemeris."""

    coordinate_system: CoordinateSystem = Field(
        ...,
        description="Coordinate system for state vectors (ECEF, etc.)",
    )
    direction: str = Field(
        ..., description="Orbit direction (e.g. 'ascending', 'descending')"
    )
    state_vectors: list[StateVector] = Field(
        ...,
        description="Platform ephemeris sampled over the collect",
    )
    source: str = Field(
        ..., description="Source of orbit solution (e.g. 'precise_determination')"
    )

    def _to_seconds(
        self, t: NDArray[np.datetime64], epoch: np.datetime64
    ) -> np.ndarray:
        """Convert a time-like object to seconds since epoch."""
        return (t - epoch) / np.timedelta64(1, "s")

    def get_state(
        self, time_as_float: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the (time, position, velocity) for all StateVectors as arrays.

        Parameters
        ----------
        time_as_float : bool, optional
            If True, return times as floating-point seconds since first epoch.
            If False, return times as numpy.datetime64 objects.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of (times, positions, velocities).

        """
        times = np.array([sv.time.as_numpy() for sv in self.state_vectors])
        if time_as_float:
            times = self._to_seconds(times, times[0])
        positions = np.array([sv.position.as_array() for sv in self.state_vectors])
        velocities = np.array([sv.velocity.as_array() for sv in self.state_vectors])
        return times, positions, velocities


class PointingSample(BaseModel):
    """Platform attitude at a specific time."""

    time: Time = Field(..., description="Time of attitude sample (UTC)")
    attitude: AttitudeQuaternion = Field(
        ...,
        description="Platform attitude quaternion at this time",
    )

    @field_validator("attitude", mode="before")
    @classmethod
    def _parse_attitude(cls, v: Any) -> AttitudeQuaternion:
        if isinstance(v, list):
            return AttitudeQuaternion.from_list(v)
        return v


class Antenna(BaseModel):
    """Antenna parameters."""

    azimuth_beamwidth: float = Field(..., description="Azimuth 3-dB beamwidth [rad]")
    elevation_beamwidth: float = Field(
        ..., description="Elevation 3-dB beamwidth [rad]"
    )
    gain: float = Field(..., description="Antenna boresight gain [dB]")
    beam_pattern: Poly2D = Field(
        ...,
        description="2D polynomial approximation of antenna gain vs. angle",
    )


def is_uniformly_sampled(state_vectors: Sequence[StateVector]) -> bool:
    """Return True if the StateVectors are uniformly sampled."""
    times = [sv.time.as_numpy() for sv in state_vectors]
    dt_count = set(np.diff(times))  # type: ignore[arg-type]
    return len(dt_count) == 1


def interpolate_orbit(state_vectors: Sequence[StateVector]) -> list[StateVector]:
    """Ensure the `state_vectors` are uniformly sampled, interpolating if necessary.

    Parameters
    ----------
    state_vectors : Sequence[StateVector]
        Sequence of StateVector objects to interpolate.

    Returns
    -------
    list[StateVector]
        Uniformly sampled state vectors.

    """
    if is_uniformly_sampled(state_vectors):
        return list(state_vectors)
    times = [sv.time.as_numpy() for sv in state_vectors]
    positions = np.array([sv.position.as_array() for sv in state_vectors])
    velocities = np.array([sv.velocity.as_array() for sv in state_vectors])
    dt_count = Counter(np.diff(times))  # type: ignore[arg-type]
    # example: dt_count.most_common(1)
    #  [(datetime.timedelta(microseconds=599999), 102)]
    dt, _count = dt_count.most_common(1)[0]

    t2, p2, v2 = resample_orbit_data_linear(
        times, positions, velocities, dt_seconds=dt / np.timedelta64(1, "s")
    )
    return [
        StateVector(
            time=tt,
            position=ECEFPosition(x=pp[0], y=pp[1], z=pp[2]),
            velocity=ECEFVelocity(vx=vv[0], vy=vv[1], vz=vv[2]),
        )
        for tt, pp, vv in zip(t2, p2, v2, strict=True)
    ]


def resample_orbit_data_linear(
    t: Sequence[np.datetime64], p: NDArray, v: NDArray, dt_seconds: float = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample orbit data to have uniform time spacing.

    Parameters
    ----------
    t : Sequence[np.datetime64]
        Sequence of datetime objects.
    p : np.ndarray
        Array of position vectors, shape (n, 3).
    v : np.ndarray
        Array of velocity vectors, shape (n, 3).
    dt_seconds : float, optional
        Desired time step in seconds, by default 10.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Resampled t, p, and v arrays.

    Notes
    -----
    This function uses linear interpolation for resampling.

    """
    t_seconds = np.array([(ti - t[0]) / np.timedelta64(1, "s") for ti in t])

    t_new = np.arange(t_seconds[0], t_seconds[-1] + dt_seconds, dt_seconds)

    p_new = np.array([np.interp(t_new, t_seconds, p[:, i]) for i in range(3)]).T
    v_new = np.array([np.interp(t_new, t_seconds, v[:, i]) for i in range(3)]).T

    # Convert to integer nanoseconds to avoid floating point precision issues
    dt_ns = round(dt_seconds * 1e9)
    n_steps = len(t_new)
    t_new_datetime = np.array(
        [t[0] + np.timedelta64(i * dt_ns, "ns") for i in range(n_steps)]
    )

    return t_new_datetime, p_new, v_new
