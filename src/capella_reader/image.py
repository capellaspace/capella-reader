"""Image metadata: geometry, windows, quantization, terrain models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from capella_reader._time import Time
from capella_reader.geometry import ECEFPosition
from capella_reader.polynomials import Poly1D, Poly2D


class Window(BaseModel):
    """Window function parameters."""

    name: str = Field(..., description="Window function name (e.g. 'rectangular')")
    parameters: Mapping[str, float] = Field(
        default_factory=dict,
        description="Window parameters (implementation-specific numeric values)",
    )
    broadening_factor: float = Field(
        ...,
        description="Main-lobe broadening factor due to windowing",
    )


class Quantization(BaseModel):
    """Quantization scheme parameters."""

    type: str = Field(
        ..., description="Quantization scheme, e.g. 'block_adaptive_quantization'"
    )
    block_sample_size: int = Field(..., description="Samples per block")
    mean_bits: int = Field(..., description="Bits used for block mean")
    std_bits: int = Field(..., description="Bits used for block std-dev")
    sample_bits: int = Field(..., description="Bits per quantized sample")


class TerrainModelRef(BaseModel):
    """Reference to a terrain model."""

    link: str = Field(..., description="Reference or URL describing terrain model")
    name: str = Field(..., description="Terrain model name / identifier")


class TerrainModels(BaseModel):
    """Collection of terrain model references."""

    focusing: TerrainModelRef | None = Field(
        None,
        description="Terrain model used for focusing / geolocation",
    )


class SlantPlaneGeometry(BaseModel):
    """Geometry parameters for slant-plane focused products."""

    type: Literal["slant_plane"] = Field(
        ..., description="Image geometry type (slant_plane)"
    )
    doppler_centroid_polynomial: Poly2D = Field(
        ...,
        description=(
            "A 2D polynomial mapping range and azimuth time to doppler centroid "
            "frequency in Hz used to compute the image geometry. Notice that the "
            "range dependence of the DC polynomial uses range distance. The azimuth "
            "variable is seconds since first_line_time."
        ),
    )
    first_line_time: Time = Field(
        ...,
        description="The timestamp of the first line",
    )
    delta_line_time: float = Field(
        ...,
        description="The time difference between successive lines in seconds",
    )
    range_to_first_sample: float = Field(
        ...,
        description="The slant range distance to the first sample in meters",
    )
    delta_range_sample: float = Field(
        ...,
        description="The slant range delta distance between each sample in meters",
    )


class PFAGeometry(BaseModel):
    """Geometry parameters for Polar Format Algorithm (PFA) products.

    PFA products use a different coordinate system and have many additional
    fields that are preserved via extra='allow'.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["pfa"] = Field(..., description="Image geometry type (pfa)")
    scene_reference_point_row_col: tuple[float, float] = Field(
        ...,
        description=(
            "The row and column at the scene reference point in units of pixels"
        ),
    )
    scene_reference_point_ecef: ECEFPosition = Field(
        ..., description="The scene reference point in ECEF meters"
    )
    row_sample_spacing: float = Field(
        ..., description="The sample spacing in meters in the row direction"
    )
    col_sample_spacing: float = Field(
        ..., description="The samples spacing in meters in the column direction"
    )
    row_direction: tuple[float, float, float] = Field(
        ..., description="A unit vector in ECEF indicating the row direction"
    )
    col_direction: tuple[float, float, float] = Field(
        ..., description="A unit vector in ECEF indicating the column direction"
    )
    slant_plane_normal: tuple[float, float, float] = Field(
        ...,
        description=(
            "A 3D unit vector in ECEF describing the direction normal to the slant"
            " plane"
        ),
    )
    ground_plane_normal: tuple[float, float, float] = Field(
        ...,
        description=(
            "A 3D unit vector in ECEF describing the direction normal to the ground"
            " plane"
        ),
    )
    polar_angle_polynomial: Poly1D = Field(
        ...,
        description=(
            "A 1D polynomial mapping seconds since collect_start to polar angle in"
            " radians"
        ),
    )
    spatial_frequency_scale_factor_polynomial: Poly1D = Field(
        ...,
        description=(
            "A 1D polynomial mapping polar angle in radians to Spatial Frequency Scale"
            " Factor. Used to scale RF frequency to aperture spatial frequency."
        ),
    )

    @field_validator("scene_reference_point_ecef", mode="before")
    @classmethod
    def _parse_ecef(cls, v: Any) -> ECEFPosition:
        if isinstance(v, list):
            return ECEFPosition.from_list(v)
        return v


ImageGeometry = Annotated[
    SlantPlaneGeometry | PFAGeometry,
    Field(discriminator="type"),
]


class CenterPixel(BaseModel):
    """Scene center pixel metadata."""

    incidence_angle: float = Field(
        ..., description="Incidence angle at scene center [deg]"
    )
    look_angle: float = Field(..., description="Look angle at scene center [deg]")
    squint_angle: float = Field(..., description="Squint angle at scene center [deg]")
    layover_angle: float | None = Field(
        None,
        description="Layover angle at scene center [deg]; null if not provided",
    )
    target_position: ECEFPosition = Field(
        ...,
        description="ECEF coordinates of scene center target [m]",
    )
    center_time: Time = Field(
        ...,
        description="Acquisition time of scene center (UTC)",
    )

    @field_validator("target_position", mode="before")
    @classmethod
    def _parse_target_position(cls, v: Any) -> ECEFPosition:
        if isinstance(v, list):
            return ECEFPosition.from_list(v)
        return v


class ImageMetadata(BaseModel):
    """Complete image metadata."""

    data_type: str = Field(
        ..., description="Underlying sample data type (e.g. 'CInt16')"
    )
    length: float = Field(..., description="Approximate scene length on ground [m]")
    width: float = Field(..., description="Approximate scene width on ground [m]")
    rows: int = Field(..., description="Number of image lines (azimuth dimension)")
    columns: int = Field(
        ..., description="Number of samples per line (range dimension)"
    )
    pixel_spacing_row: float = Field(..., description="Pixel spacing in azimuth [m]")
    pixel_spacing_column: float = Field(
        ..., description="Pixel spacing in ground-range or slant-range [m]"
    )
    algorithm: str = Field(
        ..., description="Imaging algorithm used (e.g. 'backprojection')"
    )
    scale_factor: float = Field(
        ..., description="Radiometric scale factor to convert stored DN"
    )

    range_autofocus: bool | str | None = Field(
        None,
        description=(
            "Whether range autofocus was applied (if known); "
            "may be bool, string like 'global', or null if unknown"
        ),
    )
    azimuth_autofocus: bool | str | None = Field(
        None,
        description=(
            "Whether azimuth autofocus was applied (if known); "
            "may be bool, string like 'global', or null if unknown"
        ),
    )

    range_window: Window
    processed_range_bandwidth: float = Field(
        ..., description="Processed range bandwidth [Hz]"
    )
    azimuth_window: Window
    processed_azimuth_bandwidth: float = Field(
        ..., description="Processed azimuth bandwidth [Hz]"
    )

    image_geometry: ImageGeometry
    center_pixel: CenterPixel

    range_resolution: float = Field(
        ..., description="Nominal slant-range resolution [m]"
    )
    ground_range_resolution: float = Field(
        ..., description="Nominal ground-range resolution [m]"
    )
    azimuth_resolution: float = Field(..., description="Nominal azimuth resolution [m]")
    ground_azimuth_resolution: float = Field(
        ..., description="Nominal ground azimuth resolution [m]"
    )

    azimuth_looks: float = Field(..., description="Number of looks in azimuth")
    range_looks: float = Field(..., description="Number of looks in range")
    enl: float = Field(..., description="Equivalent number of looks (ENL)")

    reference_antenna_position: ECEFPosition | None = Field(
        None,
        description="Reference antenna phase center position in ECEF [m], if provided",
    )
    reference_target_position: ECEFPosition | None = Field(
        None,
        description="Reference target position in ECEF [m], if provided",
    )

    azimuth_beam_pattern_corrected: bool = Field(
        ..., description="True if azimuth beam pattern was corrected"
    )
    elevation_beam_pattern_corrected: bool = Field(
        ..., description="True if elevation beam pattern was corrected"
    )

    radiometry: Literal["beta_nought", "sigma_nought", "gamma_nought", "beta"] = Field(
        ..., description="Radiometric convention"
    )
    calibration: Literal["full", "partial", "limited", "none"] = Field(
        ..., description="Calibration level"
    )
    calibration_id: str = Field(
        ..., description="Identifier for calibration bundle used"
    )

    nesz_polynomial: Poly1D = Field(
        ..., description="Noise Equivalent Sigma Zero (NESZ) as 1D polynomial"
    )
    nesz_peak: float = Field(..., description="Peak NESZ [dB]")
    terrain_models: TerrainModels | None = Field(
        None,
        description="Terrain models used for focusing/geolocation, if provided",
    )

    reference_doppler_centroid: float = Field(
        ..., description="Reference Doppler centroid at scene center [Hz]"
    )
    frequency_doppler_centroid_polynomial: Poly2D = Field(
        ...,
        description=(
            "Doppler centroid frequency as 2D polynomial of image coordinates [Hz]"
        ),
    )

    quantization: Quantization

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape as (rows, columns)."""
        return self.rows, self.columns

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype corresponding to the on-disk pixel type."""
        if self.data_type == "CInt16":
            return np.dtype("complex64")  # decoded from int16 pairs
        if self.data_type == "UInt16":
            return np.dtype("uint16")
        return np.dtype(self.data_type)

    @property
    def is_slant_plane(self) -> bool:
        """Check if image uses slant-plane geometry."""
        return self.image_geometry.type == "slant_plane"

    @property
    def is_pfa(self) -> bool:
        """Check if image uses PFA geometry."""
        return self.image_geometry.type == "pfa"

    @field_validator(
        "reference_antenna_position", "reference_target_position", mode="before"
    )
    @classmethod
    def _parse_ecef(cls, v: Any) -> ECEFPosition:
        if isinstance(v, list):
            return ECEFPosition.from_list(v)
        return v
