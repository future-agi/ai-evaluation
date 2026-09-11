"""RealtimeEndpoint + RealtimeBridgeSession contracts (plan §5).

Only Protocol/data shapes here — the media router, backpressure, and
provider-pair bridge live in follow-up work. Adding these gives the
existing ``LiveKitEngine`` a stable seam it can be adapted onto and
gives future adapters (Pipecat, alternative LiveKit backends, direct
SIP shims) a target contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field, JsonValue

from .events import RealtimeEvent
from .media import AudioFrame
from fi.simulate.runtime.capabilities import EndpointCapabilities


class CloseReason(str, Enum):
    NORMAL = "normal"
    INTERRUPTED = "interrupted"
    CANCELED = "canceled"
    ERROR = "error"
    TIMEOUT = "timeout"


class EndpointSession(BaseModel):
    session_id: str
    leg_id: str
    capabilities: EndpointCapabilities
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class BridgeResult(BaseModel):
    session_id: str
    close_reason: CloseReason
    left_events: int = 0
    right_events: int = 0
    audio_frames: int = 0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RealtimeEndpoint(Protocol):
    """One side of a realtime bridge (customer agent OR simulator)."""

    capabilities: EndpointCapabilities

    async def connect(self) -> EndpointSession: ...

    async def receive(self) -> AsyncIterator[RealtimeEvent | AudioFrame]:  # noqa: D401
        """Yield events + audio frames until the endpoint closes."""
        ...

    async def send(self, event: RealtimeEvent | AudioFrame) -> None: ...

    async def close(self, reason: CloseReason) -> None: ...


@dataclass
class RealtimeBridgeSession:
    """Neutral bridge between two ``RealtimeEndpoint`` implementations.

    The real router (pacing, backpressure, playback-clear on
    interruption, evidence taps) lands under §5.3. This dataclass gives
    that router a stable public constructor today so the rest of the
    SDK can start referencing bridge shape without waiting for that
    implementation.
    """

    left: RealtimeEndpoint
    right: RealtimeEndpoint
    session_id: str

    async def run(self) -> BridgeResult:
        raise NotImplementedError(
            "RealtimeBridgeSession.run is a seam; implement per plan §5.3."
        )


__all__ = [
    "BridgeResult",
    "CloseReason",
    "EndpointSession",
    "RealtimeBridgeSession",
    "RealtimeEndpoint",
]
