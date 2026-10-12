"""The rules the conformance suite checks on an Episode in JSON form.

No JSON Schema validation here: the suite runs that itself, and importing a
validator would pull a third-party library into every import of this package.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .coverage import CoverageFacts, build_coverage
from .events import EPISODE_EVENT_TYPES
from .ids import derive_episode_id
from .models import ProviderRef, VoiceEpisode

# Events a converter adds itself; every other event comes from an entry of the
# record's entry list, so its presence shows the list was found.
_SYNTHETIC_EVENT_TYPES = ("episode.started", "tool.unresolved", "episode.ended")
_BROKEN_SHAPE = (KeyError, IndexError, TypeError, AttributeError, ValueError)


def _invariants_hold(episode: Mapping[str, Any]) -> bool:
    # JSON mode: the models are strict, and strict Python mode refuses the JSON
    # form's time strings and arrays. allow_nan=False: a NaN or Infinity cannot
    # survive the JSON form, so the ValueError counts as the rule broken.
    try:
        VoiceEpisode.model_validate_json(json.dumps(dict(episode), allow_nan=False))
    except _BROKEN_SHAPE:
        return False
    return True


def _id_matches_context(episode: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    try:
        parts = (context["tenant_id"], episode["provider"], episode["provider_call_id"])
        if not all(isinstance(part, str) for part in parts):
            return False
        return episode["id"] == derive_episode_id(*parts)
    except _BROKEN_SHAPE:
        return False


def _coverage_facts(episode: Mapping[str, Any]) -> CoverageFacts:
    events = episode["events"]
    utterances = [event for event in events if event["type"] == "utterance"]
    cost = episode["cost"]
    analysis = episode["analysis"]
    provider_ref = episode["provider_ref"]
    return CoverageFacts(
        status=episode["status"],
        utterance_count=len(utterances),
        fully_timed_utterance_count=sum(
            1 for u in utterances if u["wall_time"] is not None and u["payload"]["duration_ms"] is not None
        ),
        utterances_with_wall_time=sum(1 for u in utterances if u["wall_time"] is not None),
        entry_list_found=bool(episode["unmapped_entries"])
        or any(event["type"] not in _SYNTHETIC_EVENT_TYPES for event in events),
        unresolved_tool_count=sum(1 for event in events if event["type"] == "tool.unresolved"),
        latency_count=len(episode["latency"]),
        cost_total_present=cost is not None,
        cost_line_count=len(cost["breakdown"]) if cost is not None else 0,
        recording_count=len(episode["recordings"]),
        provider_ref=ProviderRef.model_validate(provider_ref) if provider_ref is not None else None,
        end_reason=episode["end_reason"],
        direction=episode["direction"],
        analysis_field_set=analysis is not None and any(value is not None for value in analysis.values()),
        provider_log_url=episode["provider_log_url"],
    )


def _coverage_matches_declared(episode: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    # The whole coverage rule, not only the entry names: every fact the rule
    # table reads is on the Episode itself, so the expected list can be rebuilt.
    try:
        declared = context["declared_capabilities"]
        if isinstance(declared, (str, bytes)):
            return False
        expected = build_coverage(list(declared), _coverage_facts(episode))
        actual = list(episode["coverage"])
    except _BROKEN_SHAPE:
        return False
    return actual == [entry.model_dump(mode="json") for entry in expected]


def _closed_event_types(episode: Mapping[str, Any]) -> bool:
    try:
        return all(event["type"] in EPISODE_EVENT_TYPES for event in episode["events"])
    except _BROKEN_SHAPE:
        return False


def check_conformance(episode: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[str, ...]:
    """Names of the Episode rules ``episode`` breaks, in a fixed order; empty when it breaks none."""
    checks = (
        ("invariants", _invariants_hold(episode)),
        ("id_matches_context", _id_matches_context(episode, context)),
        ("coverage_matches_declared", _coverage_matches_declared(episode, context)),
        ("closed_event_types", _closed_event_types(episode)),
    )
    return tuple(name for name, holds in checks if not holds)
