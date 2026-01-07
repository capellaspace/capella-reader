"""Tests for small utility modules."""

from __future__ import annotations

from types import EllipsisType
from typing import get_args

from capella_reader._types import Index


def test_index_type_alias():
    """Test Index type alias expands to the expected union."""
    args = get_args(Index)
    assert args[0] is int
    assert args[1] is slice
    assert args[2] is EllipsisType
