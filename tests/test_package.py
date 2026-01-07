from __future__ import annotations

import capella_reader


def test_version():
    """Test that the package version is accessible."""
    assert hasattr(capella_reader, "__version__")
