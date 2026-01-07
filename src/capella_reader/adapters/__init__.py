"""Adapters for Capella reader to other processing libraries.

This subpackage contains optional adapters for working with Capella SLC data
using external processing libraries. These adapters are not required for basic metadata
parsing and are only available if the corresponding libraries are installed.

Modules
-------
isce3
    ISCE3 conversion utilities (requires isce3 to be installed)

"""

__all__ = ["isce3"]
