"""Radar configuration and time-varying parameters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._time import Time
from .enums import LookSide


class RadarTimeVaryingParams(BaseModel):
    """Time-varying radar parameters."""

    start_timestamps: list[Time] = Field(
        ...,
        description="Start times when this PRF / waveform setting becomes active",
    )
    prf: float = Field(..., description="Pulse repetition frequency [Hz]")
    pulse_bandwidth: float = Field(..., description="Transmit pulse bandwidth [Hz]")
    pulse_duration: float = Field(..., description="Transmit pulse duration [s]")
    rank: int = Field(
        ..., description="Internal rank / priority for this configuration"
    )


class PRFEntry(BaseModel):
    """Pulse repetition frequency entry."""

    start_timestamps: list[Time] = Field(
        ...,
        description="Start times when this PRF is active",
    )
    prf: float = Field(..., description="Pulse repetition frequency [Hz]")


class Radar(BaseModel):
    """Radar configuration."""

    rank: int = Field(..., description="Internal radar configuration rank")
    center_frequency: float = Field(..., description="Radar carrier frequency [Hz]")
    pointing: LookSide = Field(..., description="Look side of platform")
    sampling_frequency: float = Field(
        ..., description="Receive sampling frequency [Hz]"
    )
    transmit_polarization: Literal["H", "V"] = Field(
        ..., description="Transmit polarization"
    )
    receive_polarization: Literal["H", "V"] = Field(
        ..., description="Receive polarization"
    )

    time_varying_parameters: list[RadarTimeVaryingParams] = Field(
        ...,
        description="Full time-varying radar configuration over the collect",
    )
    prf: list[PRFEntry] = Field(
        ...,
        description="Simplified time-varying PRF sequence",
    )
