from __future__ import annotations

import time
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, JsonValue

from .ids import derive_event_id

EVENT_SCHEMA_VERSION = "futureagi.simulation-event.v1"


class EventReliability(str, Enum):
    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"
    MEDIA = "media"


class CanonicalEvent(BaseModel):
    schema_version: str = EVENT_SCHEMA_VERSION
    event_id: str
    run_id: str
    test_case_id: str
    session_id: str | None = None
    type: str
    source: str
    provider: str | None = None
    wall_time: datetime
    monotonic_ns: int
    reliability: EventReliability = EventReliability.RELIABLE
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    trace_id: str | None = None
    correlation_id: str | None = None
    provider_raw_ref: str | None = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        test_case_id: str,
        event_type: str,
        source: str,
        sequence: int,
        provider: str | None = None,
        session_id: str | None = None,
        reliability: EventReliability = EventReliability.RELIABLE,
        payload: dict[str, JsonValue] | None = None,
    ) -> "CanonicalEvent":
        return cls(
            event_id=derive_event_id(test_case_id, source, sequence),
            run_id=run_id,
            test_case_id=test_case_id,
            session_id=session_id,
            type=event_type,
            source=source,
            provider=provider,
            wall_time=datetime.now(timezone.utc),
            monotonic_ns=time.monotonic_ns(),
            reliability=reliability,
            payload=payload or {},
        )
