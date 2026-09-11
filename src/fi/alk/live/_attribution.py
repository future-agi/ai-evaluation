"""Failure-layer classification for live lanes (R§1 #1, R§3.3).

Imports: stdlib only. Strictly ordered, first match wins, evaluated BEFORE
scoring: lane_infra → framework_runtime → provider → agent_behavior. Only
``agent_behavior`` scores the agent; ``lane_infra`` voids the row.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping, Sequence

from ._contract import FAILURE_LAYERS
from ._runner import LaneProcessResult, READY_EVENT_TYPE

VERIFICATION_EVENT_TYPE = "verification"
PROVIDER_ERROR_EVENT_TYPE = "provider_error"

_TRACEBACK_FILE = re.compile(r'File "([^"]+)"')


@dataclasses.dataclass
class FailureAttribution:
    layer: str                 # member of FAILURE_LAYERS
    detail: str
    scored: bool               # True ONLY for agent_behavior (PRD §4.1)

    def __post_init__(self) -> None:
        if self.layer not in FAILURE_LAYERS:
            raise ValueError(f"unknown failure layer: {self.layer!r}")


def _first_event(
    events: Sequence[Mapping[str, Any]], event_type: str
) -> Mapping[str, Any] | None:
    for event in events:
        if event.get("type") == event_type:
            return event
    return None


def _verification_passed(events: Sequence[Mapping[str, Any]]) -> bool | None:
    """Last worker-reported verification verdict; None = no verifier evidence
    at all (itself lane_infra — sampling without verification is the
    documented gap, R§1 #5)."""

    verdict: bool | None = None
    for event in events:
        if event.get("type") != VERIFICATION_EVENT_TYPE:
            continue
        payload = event.get("payload")
        if isinstance(payload, Mapping) and "passed" in payload:
            verdict = bool(payload["passed"])
    return verdict


def _deepest_traceback_file(stderr_tail: str) -> str | None:
    matches = _TRACEBACK_FILE.findall(stderr_tail or "")
    return matches[-1] if matches else None


def _framework_package_paths(
    events: Sequence[Mapping[str, Any]],
) -> list[str]:
    ready = _first_event(events, READY_EVENT_TYPE)
    if not ready:
        return []
    payload = ready.get("payload")
    if not isinstance(payload, Mapping):
        return []
    paths = payload.get("package_paths")
    if not isinstance(paths, (list, tuple)):
        return []
    return [str(path) for path in paths if path]


def attribute_failure(
    process: LaneProcessResult,
    transcript_events: Sequence[Mapping[str, Any]],
) -> FailureAttribution | None:
    """HTIR-style layer attribution BEFORE scoring (R§1 #1, R§3.3).

    Order of classification (first match wins):
      lane_infra        — spawn failure, timeout before the worker's
                          'framework_ready' event, transcript unreadable;
                          row is VOID and auto-quarantined, never scored.
      framework_runtime — nonzero exit with a traceback whose deepest frame
                          is inside the framework package (worker stamps
                          package paths into the 'lane' channel at boot);
                          reported as robustness evidence.
      provider          — worker-reported provider error event (HTTP 4xx/5xx,
                          auth, rate-limit markers from the provider client).
      agent_behavior    — process exited clean but verification failed;
                          the ONLY class that scores the agent.
    Returns None when the run passed verification.
    """

    events = list(transcript_events)
    ready = _first_event(events, READY_EVENT_TYPE)
    provider_error = _first_event(events, PROVIDER_ERROR_EVENT_TYPE)
    unreadable = _first_event(events, "transcript_unreadable_line")

    # --- 1. lane_infra ------------------------------------------------------
    if process.exit_code is None and not process.timed_out:
        return FailureAttribution(
            layer="lane_infra",
            detail=f"spawn failure: {process.stderr_tail or 'no process'}",
            scored=False,
        )
    if process.timed_out:
        phase = "before framework_ready" if ready is None else "after ready"
        return FailureAttribution(
            layer="lane_infra",
            detail=f"timeout {phase} (budget kill)",
            scored=False,
        )
    if unreadable is not None:
        return FailureAttribution(
            layer="lane_infra",
            detail="transcript unreadable",
            scored=False,
        )

    # --- 2./3. crashed worker: framework_runtime, else provider, else infra --
    if process.exit_code not in (0, None):
        deepest = _deepest_traceback_file(process.stderr_tail)
        package_paths = _framework_package_paths(events)
        if deepest and any(deepest.startswith(path) for path in package_paths):
            return FailureAttribution(
                layer="framework_runtime",
                detail=(
                    f"worker exit {process.exit_code}; deepest frame "
                    f"{deepest} is inside the framework package"
                ),
                scored=False,
            )
        if provider_error is not None:
            payload = provider_error.get("payload")
            return FailureAttribution(
                layer="provider",
                detail=f"provider error: {dict(payload) if isinstance(payload, Mapping) else payload}",
                scored=False,
            )
        return FailureAttribution(
            layer="lane_infra",
            detail=(
                f"worker exit {process.exit_code} outside the framework "
                "package (lane/worker fault)"
            ),
            scored=False,
        )

    # --- clean exit ----------------------------------------------------------
    if ready is None:
        return FailureAttribution(
            layer="lane_infra",
            detail="worker exited clean but never emitted framework_ready",
            scored=False,
        )
    verification = _verification_passed(events)
    if verification is True:
        return None
    if provider_error is not None:
        payload = provider_error.get("payload")
        return FailureAttribution(
            layer="provider",
            detail=f"provider error: {dict(payload) if isinstance(payload, Mapping) else payload}",
            scored=False,
        )
    if verification is None:
        return FailureAttribution(
            layer="lane_infra",
            detail=(
                "no verifier evidence: repeat carried no programmatic/judge/"
                "end-state verdict (R§1 #5)"
            ),
            scored=False,
        )
    return FailureAttribution(
        layer="agent_behavior",
        detail="verification failed on a clean run",
        scored=True,
    )
