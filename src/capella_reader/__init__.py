"""capella-reader: Basic models for parsing and working with Capella SLCs."""

from __future__ import annotations

from capella_reader._time import Time, TimeDelta

try:
    from capella_reader._version import version as __version__
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except Exception:
        __version__ = "0.0.0+unknown"

from capella_reader.collect import Collect
from capella_reader.geometry import (
    AttitudeQuaternion,
    ECEFPosition,
    ECEFVelocity,
)
from capella_reader.image import (
    CenterPixel,
    GeotransformGeometry,
    ImageGeometry,
    ImageMetadata,
    Quantization,
    SurfaceGeometry,
    TerrainModelRef,
    TerrainModels,
    Window,
)
from capella_reader.metadata import CapellaSLCMetadata
from capella_reader.orbit import (
    Antenna,
    CoordinateSystem,
    PointingSample,
    State,
    StateVector,
)
from capella_reader.polynomials import Poly1D, Poly2D
from capella_reader.radar import PRFEntry, Radar, RadarTimeVaryingParams
from capella_reader.slc import CapellaSLC

__all__ = [
    "Antenna",
    "AttitudeQuaternion",
    "CapellaSLC",
    "CapellaSLCMetadata",
    "CenterPixel",
    "Collect",
    "CoordinateSystem",
    "ECEFPosition",
    "ECEFVelocity",
    "GeotransformGeometry",
    "ImageGeometry",
    "ImageMetadata",
    "PRFEntry",
    "PointingSample",
    "Poly1D",
    "Poly2D",
    "Quantization",
    "Radar",
    "RadarTimeVaryingParams",
    "State",
    "StateVector",
    "SurfaceGeometry",
    "TerrainModelRef",
    "TerrainModels",
    "Time",
    "TimeDelta",
    "Window",
    "__version__",
]
