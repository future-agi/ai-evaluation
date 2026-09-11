"""Canonical audio media frame (plan §5.1).

One media router will eventually own resampling and pacing; provider
adapters must not reintroduce ad-hoc resampling. This module gives that
future router — and any evidence source that wants to inspect frames —
a single frame type to depend on.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MediaDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class AudioProfile(BaseModel):
    encoding: str = "pcm_s16le"
    channels: int = Field(1, gt=0)
    sample_rate: int = Field(16_000, gt=0)
    frame_duration_ms: int = Field(20, gt=0)

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate * self.frame_duration_ms // 1000


DEFAULT_AUDIO_PROFILE = AudioProfile()


class AudioFrame(BaseModel):
    """Immutable audio frame carried by ``RealtimeEndpoint``.

    Provider adapters preserve ``provider_stream_id``, ``ssrc``, and
    original ``media_timestamp`` values instead of overwriting them so
    downstream evidence can correlate to provider-side artifacts.
    """

    event_id: str
    session_id: str
    leg_id: str
    sequence: int = Field(ge=0)
    timestamp_ns: int
    media_timestamp: int | None = None
    direction: MediaDirection
    encoding: str = "pcm_s16le"
    sample_rate: int = Field(gt=0)
    channels: int = Field(default=1, gt=0)
    samples_per_channel: int = Field(gt=0)
    payload_size_bytes: int = Field(ge=0)
    provider_stream_id: str | None = None
    provider_sequence: int | None = None
    ssrc: int | None = None
    discontinuity: bool = False


__all__ = ["AudioFrame", "AudioProfile", "DEFAULT_AUDIO_PROFILE", "MediaDirection"]
