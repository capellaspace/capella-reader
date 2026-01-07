"""Top-level metadata parsed from TIFFTAG_IMAGEDESCRIPTION."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from capella_reader.collect import Collect


class CapellaSLCMetadata(BaseModel):
    """Top-level metadata parsed from TIFFTAG_IMAGEDESCRIPTION."""

    software_version: str = Field(
        ..., description="Capella processing software version"
    )
    software_revision: str = Field(..., description="Git revision or internal build ID")
    processing_time: datetime = Field(
        ..., description="Time this product was processed (UTC)"
    )
    processing_deployment: str = Field(
        ..., description="Deployment environment (e.g. 'production')"
    )
    copyright: str = Field(..., description="Copyright notice")
    license: str = Field(..., description="License information")
    product_version: str = Field(..., description="Internal Capella product version")
    product_type: str = Field(..., description="Product type (e.g. 'SLC')")

    collect: Collect
