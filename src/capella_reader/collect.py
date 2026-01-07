"""Collect metadata block."""

from __future__ import annotations

from pydantic import BaseModel, Field

from capella_reader._time import Time
from capella_reader.image import ImageMetadata
from capella_reader.orbit import Antenna, PointingSample, State
from capella_reader.radar import Radar


class Collect(BaseModel):
    """Collect-level metadata."""

    start_timestamp: Time = Field(..., description="Collect start time (UTC)")
    stop_timestamp: Time = Field(..., description="Collect stop time (UTC)")

    local_datetime: str = Field(
        ...,
        description="String containing formatted local time at scene center",
    )
    local_timezone: str = Field(
        ..., description="IANA or offset representation of local timezone"
    )

    platform: str = Field(..., description="Satellite identifier (e.g. 'capella-14')")
    mode: str = Field(..., description="Imaging mode (e.g. 'spotlight')")
    collect_id: str = Field(..., description="Unique collect identifier")

    image: ImageMetadata
    radar: Radar
    state: State
    pointing: list[PointingSample]
    transmit_antenna: Antenna
    receive_antenna: Antenna
