"""Canonical realtime event vocabulary (plan §5.2).

Event *names* freeze here so any future router, provider adapter, or
evidence source can emit ``CanonicalEvent(type=...)`` with a name the
platform recognizes. The vocabulary is deliberately open: additional
provider-specific types stay under the ``provider.raw`` umbrella and
must carry a ``provider_raw_ref`` before they can be persisted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, JsonValue


CANONICAL_LIFECYCLE_EVENTS: tuple[str, ...] = (
    "session.started",
    "session.ready",
    "session.ended",
    "session.error",
    "endpoint.connected",
    "endpoint.ready",
    "endpoint.disconnected",
    "participant.joined",
    "participant.left",
    "track.published",
    "track.unpublished",
)

CANONICAL_MEDIA_EVENTS: tuple[str, ...] = (
    "audio.frame",
    "speech.started",
    "speech.stopped",
    "playback.started",
    "playback.ended",
    "playback.clear",
    "interruption",
    "transcript.partial",
    "transcript.final",
)

CANONICAL_TOOL_EVENTS: tuple[str, ...] = (
    "tool.started",
    "tool.completed",
    "tool.failed",
)

CANONICAL_TELEPHONY_EVENTS: tuple[str, ...] = (
    "dtmf.sent",
    "dtmf.received",
    "transfer.started",
    "transfer.completed",
)

CANONICAL_TELEMETRY_EVENTS: tuple[str, ...] = (
    "usage.updated",
    "metric.observed",
    "provider.raw",
)

CANONICAL_EVENT_TYPES: tuple[str, ...] = (
    *CANONICAL_LIFECYCLE_EVENTS,
    *CANONICAL_MEDIA_EVENTS,
    *CANONICAL_TOOL_EVENTS,
    *CANONICAL_TELEPHONY_EVENTS,
    *CANONICAL_TELEMETRY_EVENTS,
)


class RealtimeEvent(BaseModel):
    """Envelope emitted by a ``RealtimeEndpoint``.

    Adapters wrap provider payloads into this shape rather than exposing
    provider SDK objects across the SDK boundary.
    """

    event_id: str
    session_id: str
    leg_id: str | None = None
    type: str
    source: str
    wall_time: datetime
    monotonic_ns: int
    sequence: int = Field(ge=0)
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    provider: str | None = None
    provider_raw_ref: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None

    def with_payload(self, **overrides: Any) -> "RealtimeEvent":
        merged = dict(self.payload)
        merged.update(overrides)
        return self.model_copy(update={"payload": merged})


__all__ = [
    "CANONICAL_EVENT_TYPES",
    "CANONICAL_LIFECYCLE_EVENTS",
    "CANONICAL_MEDIA_EVENTS",
    "CANONICAL_TELEMETRY_EVENTS",
    "CANONICAL_TELEPHONY_EVENTS",
    "CANONICAL_TOOL_EVENTS",
    "RealtimeEvent",
]
