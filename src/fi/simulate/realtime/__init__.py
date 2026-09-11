"""Provider-neutral realtime session primitives (plan §5).

These are contracts, not yet a full router. ``LiveKitEngine`` keeps
owning the actual media path today; the classes here give Stage 6/7
callers a stable seam for adding a real router, evidence tap, and
alternate provider backends without rewriting engine internals.
"""

from __future__ import annotations

from .events import (
    CANONICAL_EVENT_TYPES,
    CANONICAL_MEDIA_EVENTS,
    CANONICAL_TOOL_EVENTS,
    RealtimeEvent,
)
from .media import DEFAULT_AUDIO_PROFILE, AudioFrame, AudioProfile, MediaDirection
from .session import (
    BridgeResult,
    CloseReason,
    EndpointSession,
    RealtimeBridgeSession,
    RealtimeEndpoint,
)

__all__ = [
    "AudioFrame",
    "AudioProfile",
    "BridgeResult",
    "CANONICAL_EVENT_TYPES",
    "CANONICAL_MEDIA_EVENTS",
    "CANONICAL_TOOL_EVENTS",
    "CloseReason",
    "DEFAULT_AUDIO_PROFILE",
    "EndpointSession",
    "MediaDirection",
    "RealtimeBridgeSession",
    "RealtimeEndpoint",
    "RealtimeEvent",
]
