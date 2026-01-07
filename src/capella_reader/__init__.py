"""capella-reader: Basic models for parsing and working with Capella SLCs."""

from __future__ import annotations

from capella_reader._time import Time, TimeDelta
from capella_reader._version import version as __version__
from capella_reader.collect import Collect
from capella_reader.geometry import (
    AttitudeQuaternion,
    ECEFPosition,
    ECEFVelocity,
)
from capella_reader.image import (
    CenterPixel,
    ImageGeometry,
    ImageMetadata,
    Quantization,
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
    "TerrainModelRef",
    "TerrainModels",
    "Time",
    "TimeDelta",
    "Window",
    "__version__",
]
