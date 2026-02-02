"""Capella SLC wrapper with convenient data access."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import IO, NamedTuple, cast

import numpy as np
import pydantic
import tifffile
from pydantic import BaseModel, Field
from typing_extensions import Self

from capella_reader._time import Time
from capella_reader.collect import Collect
from capella_reader.image import PFAGeometry, SlantPlaneGeometry
from capella_reader.metadata import CapellaSLCMetadata
from capella_reader.polynomials import Poly2D

C_LIGHT = 299792458.0


def _strip_gdal_vsi_prefix(path: str) -> str:
    """Strip GDAL virtual file system prefixes (e.g., /vsicurl/, /vsis3/).

    These prefixes are used by GDAL for remote file access but are not understood
    by fsspec. This function converts them back to standard URLs.
    """
    if path.startswith("/vsicurl/"):
        return path[9:]  # len("/vsicurl/") == 9
    if path.startswith("/vsis3/"):
        # Convert /vsis3/bucket/key to s3://bucket/key
        return "s3://" + path[7:]  # len("/vsis3/") == 7
    return path


def _is_remote_path(path: str) -> bool:
    """Check if path is a remote URL (http, https, s3, etc.) or GDAL virtual path."""
    # Handle GDAL virtual paths
    if path.startswith(("/vsicurl/", "/vsis3/")):
        return True
    return "://" in path and not path.startswith("file://")


def _get_suffix(path: str) -> str:
    """Get file suffix from a path string, handling both local and remote paths."""
    # Remove query string if present (for URLs like s3://bucket/file.tif?param=value)
    path_without_query = path.split("?")[0]
    return Path(path_without_query).suffix


@contextmanager
def _open_file(path: str, mode: str = "rb") -> Iterator[IO]:
    """Open a file, using fsspec for remote paths or standard open for local paths."""
    if _is_remote_path(path):
        try:
            import fsspec
        except ImportError as e:
            msg = (
                "fsspec is required to read remote files. "
                "Install it with: pip install capella-reader[fsspec]"
            )
            raise ImportError(msg) from e
        # Strip GDAL virtual prefixes for fsspec compatibility
        fsspec_path = _strip_gdal_vsi_prefix(path)
        with fsspec.open(fsspec_path, mode=mode) as f:
            yield cast(IO, f)
    else:
        with open(path, mode=mode) as f:
            yield f


class CapellaParseError(ValueError):
    """Exception thrown from incorrectly parsing a Capella GeoTiff file."""


class CapellaImageGeometryError(ValueError):
    """Exception thrown when image geometry is not supported."""


class GroundControlPoint(NamedTuple):
    """A mapping of row, col image coordinates to x, y, z."""

    row: float
    col: float
    x: float
    y: float
    z: float


class CapellaSLC(BaseModel):
    """Convenience wrapper for Capella SLC GeoTIFF files.

    Holds the path to the TIFF and the parsed metadata,
    and exposes image-like helpers (shape, dtype, slicing with __getitem__).
    """

    path: str = Field(
        ..., description="Path or URL to Capella SLC GeoTIFF or JSON file"
    )
    meta: CapellaSLCMetadata = Field(..., description="Full parsed Capella metadata")

    @property
    def shape(self) -> tuple[int, int]:
        """Alias to the underlying image shape (rows, columns)."""
        return self.meta.collect.image.shape

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype of the pixel data."""
        return self.meta.collect.image.dtype

    @classmethod
    def from_file(cls, path: str | Path) -> CapellaSLC:
        """Read TIFFTAG_IMAGEDESCRIPTION and parse metadata.

        Supports both local files and remote URLs (http, https, s3, gs, etc.)
        via fsspec.

        Parameters
        ----------
        path
            Path or URL to Capella SLC GeoTIFF or JSON file

        Returns
        -------
        CapellaSLC
            Parsed SLC object with metadata

        """
        path_str = str(path)
        suffix = _get_suffix(path_str)
        if suffix in (".tiff", ".tif"):
            d = cls._read_tiff_metadata(path_str)
        elif suffix == ".json":
            d = cls._read_json_metadata(path_str)
        else:
            msg = f"Unsupported file type: {suffix}"
            raise CapellaParseError(msg)

        try:
            meta = CapellaSLCMetadata.model_validate(d)
        except pydantic.ValidationError as e:
            msg = "Failed to validate Capella metadata"
            raise CapellaParseError(msg) from e
        return cls(path=path_str, meta=meta)

    @staticmethod
    def _read_tiff_metadata(path: str) -> dict:
        """Read metadata from a TIFF file (local or remote)."""
        try:
            with _open_file(path) as f, tifffile.TiffFile(f) as tif:
                image_description_tag: str = (
                    tif.pages[0].tags["ImageDescription"].value  # type: ignore[union-attr]
                )
        except KeyError as e:
            msg = f"Failed to parse Capella ImageDescription tags in {path}"
            raise CapellaParseError(msg) from e
        return json.loads(image_description_tag)

    @staticmethod
    def _read_json_metadata(path: str) -> dict:
        """Read metadata from a JSON file (local or remote)."""
        with _open_file(path, mode="r") as f:
            return json.load(f)

    @property
    def collect(self) -> Collect:
        """Alias to the underlying collect metadata."""
        return self.meta.collect

    # Geometry attributes
    @property
    def range_to_first_sample(self: Self) -> float:
        geom = self.meta.collect.image.image_geometry
        if geom.type != "slant_plane":
            msg = "Only supported for slant_plane geometry"
            raise CapellaImageGeometryError(msg)
        return geom.range_to_first_sample

    @property
    def delta_range_sample(self: Self) -> float:
        """Alias to the underlying range pixel spacing."""
        image = self.meta.collect.image
        geom = image.image_geometry
        if image.is_pfa:
            # appease mypy
            assert isinstance(geom, PFAGeometry)
            return geom.row_sample_spacing
        else:
            assert isinstance(geom, SlantPlaneGeometry)
            return geom.delta_range_sample

    @property
    def first_line_time(self: Self) -> Time:
        geom = self.meta.collect.image.image_geometry
        if geom.type != "slant_plane":
            msg = "Only supported for slant_plane geometry"
            raise CapellaImageGeometryError(msg)
        return geom.first_line_time

    @property
    def center_time(self: Self) -> Time:
        return self.meta.collect.image.center_pixel.center_time

    @property
    def ref_epoch(self: Self) -> Time:
        """Azimuth Time of center pixel, aliased to be a scene reference epoch."""
        # Center time is available for both slant_plane/pfa geometry
        return self.center_time

    @property
    def delta_line_time(self: Self) -> float:
        """Alias to the underlying azimuth time interval."""
        geom = self.meta.collect.image.image_geometry
        if geom.type != "slant_plane":
            msg = "Only supported for slant_plane geometry"
            raise CapellaImageGeometryError(msg)
        return geom.delta_line_time

    @property
    def sensing_start(self: Self) -> Time:
        return self.first_line_time

    @property
    def starting_range(self: Self) -> float:
        return self.range_to_first_sample

    @property
    def prf_average(self: Self) -> float:
        prf_values = [entry.prf for entry in self.meta.collect.radar.prf]
        return float(np.mean(prf_values))

    @property
    def center_frequency(self: Self) -> float:
        return self.meta.collect.radar.center_frequency

    @property
    def wavelength(self: Self) -> float:
        return C_LIGHT / self.center_frequency

    @property
    def polarization(self: Self) -> str:
        pol_transmit = self.meta.collect.radar.transmit_polarization
        pol_receive = self.meta.collect.radar.receive_polarization
        return f"{pol_transmit}{pol_receive}"

    @cached_property
    def gcps(self: Self) -> list[GroundControlPoint]:
        """Get the Ground Control Points in the tiff file."""
        if _get_suffix(self.path) == ".json":
            msg = "No GCPs available in JSON metadata files"
            raise ValueError(msg)

        with _open_file(self.path) as f, tifffile.TiffFile(f) as tif:
            gcp_arr: np.ndarray = tif.pages[0].tags["ModelTiepointTag"].value  # type: ignore[union-attr]
        # Delete the 3rd column ("3rd" dim of the 2D image)
        out: list[GroundControlPoint] = []
        gcp_arr = np.asarray(gcp_arr, dtype=float)
        gcp_arr = np.delete(gcp_arr.reshape(-1, 6), 2, axis=1)
        for row in gcp_arr:
            out.append(GroundControlPoint(*row.tolist()))
        return out

    @property
    def bounds(self: Self) -> tuple[float, float, float, float]:
        """Return the GCP bounding box as (min_lon, min_lat, max_lon, max_lat)."""
        if not self.gcps:
            msg = "No GCPs available to compute bounds"
            raise ValueError(msg)

        x = [gcp.x for gcp in self.gcps]
        y = [gcp.y for gcp in self.gcps]
        return (min(x), min(y), max(x), max(y))

    @property
    def frequency_doppler_centroid_polynomial(self: Self) -> Poly2D:
        return self.meta.collect.image.frequency_doppler_centroid_polynomial
