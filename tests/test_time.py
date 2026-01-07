"""Tests for Time and TimeDelta classes."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from capella_reader import Time, TimeDelta


class TestTime:
    """Test Time class."""

    def test_creation_from_string(self):
        """Test creating Time from ISO string."""
        t = Time("2024-01-01T12:00:00.123456789")
        assert str(t) == "2024-01-01T12:00:00.123456789Z"

    def test_creation_from_string_with_timezone(self):
        """Test creating Time from ISO string with Z suffix."""
        t = Time("2024-01-01T12:00:00.123456789Z")
        assert str(t) == "2024-01-01T12:00:00.123456789Z"

    def test_creation_from_datetime64(self):
        """Test creating Time from numpy datetime64."""
        dt = np.datetime64("2024-01-01T12:00:00.123456789", "ns")
        t = Time(dt)
        assert t.as_numpy() == dt

    def test_nanosecond_precision(self):
        """Test that nanosecond precision is preserved."""
        t = Time("2024-01-01T12:00:00.123456789")
        assert str(t) == "2024-01-01T12:00:00.123456789Z"
        assert str(t.as_numpy()) == "2024-01-01T12:00:00.123456789"

    def test_subtraction(self):
        """Test Time subtraction returns TimeDelta."""
        t1 = Time("2024-01-01T12:00:00.000000000")
        t2 = Time("2024-01-01T12:00:01.000000000")

        delta = t2 - t1
        assert isinstance(delta, TimeDelta)
        assert delta.total_seconds() == 1.0

    def test_subtraction_with_nanoseconds(self):
        """Test Time subtraction preserves nanoseconds."""
        t1 = Time("2024-01-01T12:00:00.000000000")
        t2 = Time("2024-01-01T12:00:00.123456789")

        delta = t2 - t1
        assert delta.total_seconds() == pytest.approx(0.123456789, rel=1e-9)

    def test_addition(self):
        """Test Time addition with timedelta64."""
        t = Time("2024-01-01T12:00:00.000000000")
        t2 = t + np.timedelta64(10, "s")

        assert str(t2) == "2024-01-01T12:00:10.000000000Z"

    def test_addition_with_nanoseconds(self):
        """Test Time addition preserves nanoseconds."""
        t = Time("2024-01-01T12:00:00.123456789")
        t2 = t + np.timedelta64(1_000_000_000, "ns")

        assert str(t2) == "2024-01-01T12:00:01.123456789Z"

    def test_repr(self):
        """Test Time repr."""
        t = Time("2024-01-01T12:00:00.123456789")
        assert repr(t) == "Time(2024-01-01T12:00:00.123456789)"

    def test_as_datetime(self):
        """Test converting to Python datetime."""
        import datetime

        t = Time("2024-01-01T12:00:00.000000000")
        dt = t.as_datetime()

        assert isinstance(dt, datetime.datetime)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 12
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.tzinfo == datetime.timezone.utc

    def test_as_datetime_preserves_microseconds(self):
        """Test that as_datetime preserves microsecond precision."""

        t = Time("2024-01-01T12:00:00.123456")
        dt = t.as_datetime()

        assert dt.microsecond == 123456

    def test_hash_and_comparison(self):
        """Test hashing and ordering comparisons."""
        t1 = Time("2024-01-01T12:00:00.000000000")
        t2 = Time("2024-01-01T12:00:01.000000000")
        t1_dup = Time("2024-01-01T12:00:00.000000000")

        assert t1 == t1_dup
        assert t1 < t2
        assert len({t1, t1_dup, t2}) == 2


class TestTimeDelta:
    """Test TimeDelta class."""

    def test_total_seconds_integer(self):
        """Test total_seconds with integer seconds."""
        td = TimeDelta(np.timedelta64(10, "s"))
        assert td.total_seconds() == 10.0

    def test_total_seconds_fractional(self):
        """Test total_seconds with fractional seconds."""
        td = TimeDelta(np.timedelta64(1_500_000_000, "ns"))
        assert td.total_seconds() == 1.5

    def test_total_seconds_nanosecond_precision(self):
        """Test total_seconds preserves nanosecond precision."""
        td = TimeDelta(np.timedelta64(123_456_789, "ns"))
        assert td.total_seconds() == pytest.approx(0.123456789, rel=1e-9)

    def test_as_numpy(self):
        """Test accessing underlying numpy timedelta64."""
        td = TimeDelta(np.timedelta64(10, "s"))
        assert isinstance(td.as_numpy(), np.timedelta64)

    def test_repr(self):
        """Test TimeDelta repr."""
        td = TimeDelta(np.timedelta64(1, "s"))
        assert "TimeDelta" in repr(td)
        assert "nanoseconds" in repr(td)


class TestPydanticIntegration:
    """Test Pydantic integration with Time."""

    def test_model_with_time_field(self):
        """Test creating a Pydantic model with Time field."""

        class TestModel(BaseModel):
            timestamp: Time

        m = TestModel(timestamp="2024-01-01T12:00:00.123456789")
        assert isinstance(m.timestamp, Time)
        assert str(m.timestamp) == "2024-01-01T12:00:00.123456789Z"

    def test_model_with_time_from_datetime64(self):
        """Test Pydantic model accepts numpy datetime64."""

        class TestModel(BaseModel):
            timestamp: Time

        dt = np.datetime64("2024-01-01T12:00:00.123456789", "ns")
        m = TestModel(timestamp=dt)
        assert isinstance(m.timestamp, Time)

    def test_model_json_serialization(self):
        """Test JSON serialization of Time in Pydantic model."""

        class TestModel(BaseModel):
            timestamp: Time

        m = TestModel(timestamp="2024-01-01T12:00:00.123456789")
        json_str = m.model_dump_json()

        assert "2024-01-01T12:00:00.123456789Z" in json_str

    def test_model_json_deserialization(self):
        """Test JSON deserialization of Time in Pydantic model."""

        class TestModel(BaseModel):
            timestamp: Time

        m1 = TestModel(timestamp="2024-01-01T12:00:00.123456789")
        json_str = m1.model_dump_json()

        m2 = TestModel.model_validate_json(json_str)
        assert isinstance(m2.timestamp, Time)
        assert str(m2.timestamp) == str(m1.timestamp)

    def test_model_round_trip(self):
        """Test round-trip serialization preserves nanoseconds."""

        class TestModel(BaseModel):
            timestamp: Time

        m1 = TestModel(timestamp="2024-01-01T12:00:00.123456789")
        json_str = m1.model_dump_json()
        m2 = TestModel.model_validate_json(json_str)

        assert str(m1.timestamp) == str(m2.timestamp)
        assert m1.model_dump() == m2.model_dump()

    def test_model_with_timezone_suffix(self):
        """Test Pydantic model handles timezone suffix."""

        class TestModel(BaseModel):
            timestamp: Time

        m = TestModel(timestamp="2024-01-01T12:00:00.123456789Z")
        assert str(m.timestamp) == "2024-01-01T12:00:00.123456789Z"

    def test_model_optional_time(self):
        """Test Pydantic model with optional Time field."""

        class TestModel(BaseModel):
            timestamp: Time | None = None

        m1 = TestModel()
        assert m1.timestamp is None

        m2 = TestModel(timestamp="2024-01-01T12:00:00.123456789")
        assert isinstance(m2.timestamp, Time)

    def test_model_time_list(self):
        """Test Pydantic model with list of Time objects."""

        class TestModel(BaseModel):
            timestamps: list[Time]

        m = TestModel(
            timestamps=[
                "2024-01-01T12:00:00.000000000",
                "2024-01-01T12:00:01.000000000",
                "2024-01-01T12:00:02.000000000",
            ]
        )

        assert len(m.timestamps) == 3
        assert all(isinstance(t, Time) for t in m.timestamps)

    def test_model_validation_error(self):
        """Test that invalid time strings raise validation errors."""

        class TestModel(BaseModel):
            timestamp: Time

        with pytest.raises(ValidationError):
            TestModel(timestamp="not-a-datetime")


class TestEdgeCases:
    """Test edge cases and special values."""

    def test_negative_timedelta(self):
        """Test negative time differences."""
        t1 = Time("2024-01-01T12:00:01.000000000")
        t2 = Time("2024-01-01T12:00:00.000000000")

        delta = t2 - t1
        assert delta.total_seconds() == -1.0

    def test_very_small_timedelta(self):
        """Test very small nanosecond differences."""
        t1 = Time("2024-01-01T12:00:00.000000000")
        t2 = Time("2024-01-01T12:00:00.000000001")

        delta = t2 - t1
        assert delta.total_seconds() == pytest.approx(1e-9, rel=1e-9)

    def test_large_timedelta(self):
        """Test large time differences."""
        t1 = Time("2024-01-01T00:00:00.000000000")
        t2 = Time("2024-01-02T00:00:00.000000000")

        delta = t2 - t1
        assert delta.total_seconds() == 86400.0  # 24 hours

    def test_microsecond_precision(self):
        """Test that microsecond precision is preserved."""
        t = Time("2024-01-01T12:00:00.123456")
        assert "123456" in str(t)

    def test_millisecond_precision(self):
        """Test that millisecond precision is preserved."""
        t = Time("2024-01-01T12:00:00.123")
        assert "123" in str(t)
