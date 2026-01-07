"""Tests for small utility modules."""

from __future__ import annotations

import pytest

from capella_reader.enums import LookSide


def test_lookside_int_and_missing():
    """Test LookSide int conversion and missing value handling."""
    assert int(LookSide.LEFT) == 1
    assert int(LookSide.RIGHT) == -1
    assert LookSide("LEFT") is LookSide.LEFT
    assert LookSide("right") is LookSide.RIGHT
    assert LookSide(1) is LookSide.LEFT
    assert LookSide(-1) is LookSide.RIGHT
    with pytest.raises(ValueError, match="unknown"):
        LookSide("unknown")
