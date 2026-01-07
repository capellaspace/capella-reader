from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pydantic_core import core_schema


class TimeDelta:
    """Wrapper around np.timedelta64 for nanosecond-precision time differences."""

    __slots__ = ("_td",)

    def __init__(self, td: np.timedelta64) -> None:
        self._td = np.timedelta64(td, "ns")

    def total_seconds(self) -> float:
        """Return total seconds as float."""
        return self._td / np.timedelta64(1, "s")

    def __repr__(self) -> str:
        return f"TimeDelta({self._td})"

    def as_numpy(self) -> np.timedelta64:
        """Return underlying numpy timedelta64."""
        return self._td


class Time:
    """Wrapper around np.datetime64 for nanosecond-precision timestamps.

    Provides proper Pydantic serialization and arithmetic with nanosecond precision.
    """

    __slots__ = ("_dt",)

    def __init__(self, dt: str | np.datetime64) -> None:
        if isinstance(dt, str):
            dt = dt.rstrip("Z")
            self._dt = np.datetime64(dt, "ns")
        else:
            self._dt = np.datetime64(dt, "ns")  # type: ignore[call-overload]

    def __sub__(self, other: Time) -> TimeDelta:
        return TimeDelta(self._dt - other._dt)

    def __add__(self, delta: np.timedelta64) -> Time:
        return Time(self._dt + np.timedelta64(delta, "ns"))

    def __repr__(self) -> str:
        return f"Time({self._dt})"

    def __eq__(self, other: Time) -> bool:  # type: ignore[override]
        return self._dt == other._dt

    def __hash__(self) -> int:
        return hash(self._dt)

    def __lt__(self, other: Time) -> bool:
        return bool(self._dt < other._dt)

    def __str__(self) -> str:
        return np.datetime_as_string(self._dt, timezone="UTC").item()

    def as_numpy(self) -> np.datetime64:
        """Return underlying numpy datetime64."""
        return self._dt

    def as_datetime(self):
        """Return as Python datetime object.

        Returns
        -------
        datetime.datetime
            Python datetime object in UTC timezone.

        """
        import datetime

        # Convert to UTC timestamp in seconds
        timestamp_ns = (self._dt - np.datetime64("1970-01-01T00:00:00", "ns")).astype(
            int
        )
        timestamp_s = timestamp_ns / 1e9
        return datetime.datetime.fromtimestamp(timestamp_s, tz=datetime.timezone.utc)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        from pydantic_core import core_schema

        def validate(value: str | np.datetime64 | Time) -> Time:
            if isinstance(value, Time):
                return value
            return Time(value)

        def serialize(value: Time) -> str:
            return np.datetime_as_string(value._dt, timezone="UTC").item()

        python_schema = core_schema.with_info_plain_validator_function(
            lambda v, _: validate(v)
        )

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                validate, core_schema.str_schema()
            ),
            python_schema=python_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize, return_schema=core_schema.str_schema()
            ),
        )
