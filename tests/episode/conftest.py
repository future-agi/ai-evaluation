"""Shared fixtures for the Episode tests: the documented example and a builder of small valid Episodes."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import pytest

import fi.alk
from fi.alk.episode import (
    EPISODE_SCHEMA_VERSION,
    EVENT_SCHEMA_VERSION,
    derive_episode_event_id,
    derive_episode_id,
)

EXAMPLE_TENANT_ID = "0b7e6a52-3c1d-4f6e-9a61-2f1c5d8e9a10"
EXAMPLE_EPISODE_ID = "episode_d8fae3c7fd6153fcb22587195d661c2d"


def _example_event(seq: int, event_id: str, type_: str, wall_time: str, payload: dict, ref: str | None) -> dict:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "episode_id": EXAMPLE_EPISODE_ID,
        "seq": seq,
        "type": type_,
        "source": "vapi",
        "unit_id": "main",
        "wall_time": wall_time,
        "payload": payload,
        "provider_raw_ref": ref,
    }


def _example_episode() -> dict[str, Any]:
    # The worked example of the Episode contract, written out key by key so
    # the JSON form is pinned exactly, time strings included.
    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "type": "voice",
        "sdk_version": fi.alk.__version__,
        "id": EXAMPLE_EPISODE_ID,
        "version": {"ended": True, "event_count": 4},
        "provider": "vapi",
        "provider_call_id": "call-0001",
        "provider_ref": {"provider": "vapi", "id": "asst-1", "version": None},
        "status": "ended",
        "started_at": "2026-10-12T05:00:00Z",
        "ended_at": "2026-10-12T05:00:42.500000Z",
        "direction": "unknown",
        "end_reason": "customer-ended-call",
        "cost": {
            "amount_micros": 18600,
            "currency": "USD",
            "breakdown": [
                {"component": "stt", "label": None, "amount_micros": 16500, "tokens_in": None, "tokens_out": None},
                {"component": "llm", "label": None, "amount_micros": 2100, "tokens_in": 1234, "tokens_out": 210},
            ],
        },
        "latency": [],
        "analysis": None,
        "recordings": [],
        "provider_log_url": None,
        "coverage": [
            {"capability": "transcript", "status": "observed", "reason": None},
            {"capability": "turn_timing", "status": "observed", "reason": None},
            {"capability": "tool_calls", "status": "observed", "reason": None},
            {"capability": "cost_breakdown", "status": "observed", "reason": None},
            {"capability": "direction", "status": "unavailable", "reason": "not_in_record"},
        ],
        "unmapped_entries": {"system": 1},
        "events": [
            _example_event(
                0, "event_a6830e999a7f5140b02683928040bdc7", "episode.started", "2026-10-12T05:00:00Z", {}, None
            ),
            _example_event(
                1,
                "event_bb9da3b7b9f45c3c93ee4d9897baeb98",
                "utterance",
                "2026-10-12T05:00:01Z",
                {"role": "caller", "text": "Hi", "duration_ms": 2000},
                "/messages/1",
            ),
            _example_event(
                2,
                "event_8e5eea35d4a45201960284090feeaa67",
                "utterance",
                "2026-10-12T05:00:04Z",
                {"role": "agent", "text": "Hello, how can I help?", "duration_ms": 2500},
                "/messages/2",
            ),
            _example_event(
                3, "event_2bbed8fb534858fba4ea31c5bee3bf08", "episode.ended", "2026-10-12T05:00:42.500000Z", {}, None
            ),
        ],
    }


@pytest.fixture
def example_episode_json() -> str:
    return json.dumps(_example_episode())


EventSpec = Sequence[Any]  # (type, payload) or (type, payload, wall_time)

# Lets a caller pass ended_at=None on purpose (an errored call with no end time)
# without getting the default end time that "not given" means.
_NOT_GIVEN: Any = object()


def _build_episode(
    *,
    body: Sequence[EventSpec] = (),
    status: str = "ended",
    with_ended: bool | None = None,
    coverage: Sequence[dict] = (),
    tenant_id: str = EXAMPLE_TENANT_ID,
    provider: str = "vapi",
    provider_call_id: str = "call-0001",
    started_at: str | None = "2026-10-12T05:00:00Z",
    ended_at: str | None = _NOT_GIVEN,
) -> dict[str, Any]:
    """A valid Episode in JSON form: episode.started, then ``body``, then episode.ended unless in progress.

    Ids, seq and the version are derived, so a test breaks exactly the one rule it edits afterwards.
    """
    if with_ended is None:
        with_ended = status != "in_progress"
    if ended_at is _NOT_GIVEN:
        ended_at = None if status == "in_progress" else "2026-10-12T05:00:42.500000Z"
    episode_id = derive_episode_id(tenant_id, provider, provider_call_id)
    specs: list[tuple[str, dict, str | None, str | None]] = [("episode.started", {}, started_at, None)]
    for index, spec in enumerate(body):
        wall_time = spec[2] if len(spec) > 2 else None
        specs.append((spec[0], spec[1], wall_time, f"/messages/{index}"))
    if with_ended:
        specs.append(("episode.ended", {}, ended_at, None))
    events = [
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": derive_episode_event_id(episode_id, provider, seq),
            "episode_id": episode_id,
            "seq": seq,
            "type": type_,
            "source": provider,
            "unit_id": "main",
            "wall_time": wall_time,
            "payload": dict(payload),
            "provider_raw_ref": ref,
        }
        for seq, (type_, payload, wall_time, ref) in enumerate(specs)
    ]
    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "type": "voice",
        "sdk_version": fi.alk.__version__,
        "id": episode_id,
        "version": {"ended": ended_at is not None, "event_count": len(events)},
        "provider": provider,
        "provider_call_id": provider_call_id,
        "provider_ref": None,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "direction": "unknown",
        "end_reason": None,
        "cost": None,
        "latency": [],
        "analysis": None,
        "recordings": [],
        "provider_log_url": None,
        "coverage": [dict(entry) for entry in coverage],
        "unmapped_entries": {},
        "events": events,
    }


@pytest.fixture
def build_episode() -> Callable[..., dict[str, Any]]:
    return _build_episode


def _when(text: str | None) -> datetime | None:
    return None if text is None else datetime.fromisoformat(text.replace("Z", "+00:00"))


def _python_form(episode: dict[str, Any]) -> dict[str, Any]:
    """The JSON form as the Python values a converter passes: tuples for arrays, aware datetimes for times.

    The models are strict, so the JSON form itself is only accepted in JSON mode.
    """
    form = {**episode, "started_at": _when(episode["started_at"]), "ended_at": _when(episode["ended_at"])}
    form["events"] = tuple({**event, "wall_time": _when(event["wall_time"])} for event in episode["events"])
    form["latency"] = tuple(episode["latency"])
    form["recordings"] = tuple({**r, "expires_at": _when(r["expires_at"])} for r in episode["recordings"])
    form["coverage"] = tuple(episode["coverage"])
    if episode["cost"] is not None:
        form["cost"] = {**episode["cost"], "breakdown": tuple(episode["cost"]["breakdown"])}
    return form


@pytest.fixture
def python_form() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return _python_form
