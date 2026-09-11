"""FutureAGIObserver — LiveKit AgentSession event tap (plan §6.6 skeleton).

Attach an observer to a ``livekit.agents.AgentSession`` and it will
subscribe to a documented list of session events, translate each into
a ``CanonicalEvent`` from ``fi.simulate.runtime``, and hand it to a
pluggable sink (defaulting to an in-memory list so tests can assert
against emitted events). The real OTLP wiring lands with
``OpenTelemetryEvidenceSource``; this observer only owns the SDK-side
event capture.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fi.simulate.runtime import CanonicalEvent, EventReliability

_SESSION_EVENT_MAP: dict[str, str] = {
    "conversation_item_added": "transcript.final",
    "user_state_changed": "speech.started",
    "agent_state_changed": "session.ready",
    "function_tool_execution_started": "tool.started",
    "function_tool_execution_completed": "tool.completed",
    "function_tool_execution_failed": "tool.failed",
    "session_usage_updated": "usage.updated",
    "close": "session.ended",
    "error": "session.error",
}

logger = logging.getLogger(__name__)

EventSink = Callable[[CanonicalEvent], None]


class FutureAGIObserver:
    def __init__(
        self,
        *,
        run_id: str,
        test_case_id: str,
        sink: EventSink,
        source: str = "livekit-observer",
    ) -> None:
        self._run_id = run_id
        self._test_case_id = test_case_id
        self._sink = sink
        self._source = source
        self._sequence = 0
        self._attached = False

    def attach(self, session: Any) -> "FutureAGIObserver":
        if self._attached:
            raise RuntimeError("observer_already_attached")
        if not hasattr(session, "on"):
            raise TypeError("session_incompatible: object has no on(event, callback)")
        for session_event, canonical_type in _SESSION_EVENT_MAP.items():
            handler = self._handler_for(session_event, canonical_type)
            try:
                session.on(session_event, handler)
            except (AttributeError, ValueError):
                logger.debug(
                    "livekit observer: session does not expose event",
                    extra={"event": session_event},
                )
        self._attached = True
        return self

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        reliability: EventReliability = EventReliability.RELIABLE,
    ) -> CanonicalEvent:
        self._sequence += 1
        event = CanonicalEvent.create(
            run_id=self._run_id,
            test_case_id=self._test_case_id,
            event_type=event_type,
            source=self._source,
            sequence=self._sequence,
            reliability=reliability,
            payload=payload or {},
        )
        self._sink(event)
        return event

    def _handler_for(self, session_event: str, canonical_type: str) -> Callable[..., None]:
        def handler(*args: Any, **kwargs: Any) -> None:
            payload = _summarize_payload(session_event, args, kwargs)
            self.emit(canonical_type, payload)

        return handler


def _summarize_payload(
    session_event: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    def _describe(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump(mode="json", exclude_none=True)
            except Exception:  # noqa: BLE001
                pass
        if hasattr(value, "__dict__"):
            return {"repr": type(value).__name__}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return type(value).__name__

    return {
        "session_event": session_event,
        "args": [_describe(item) for item in args],
        "kwargs": {key: _describe(value) for key, value in kwargs.items()},
    }


__all__ = ["FutureAGIObserver"]
