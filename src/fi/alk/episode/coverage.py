"""The coverage list of an Episode: one entry per capability the provider declares."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .models import CoverageEntry, ProviderRef

_Cell = tuple[str, str | None]


@dataclass(frozen=True)
class CoverageFacts:
    """What the coverage rules read from one converted call.

    A frozen value rather than keyword arguments: a converter gathers these
    once while it walks the record, and a test changes one fact at a time
    with ``dataclasses.replace``.
    """

    status: Literal["in_progress", "ended", "errored"]
    utterance_count: int
    # Utterances that have both a wall_time and a duration_ms.
    fully_timed_utterance_count: int
    # Utterances that have a wall_time, whatever their duration.
    utterances_with_wall_time: int
    entry_list_found: bool
    unresolved_tool_count: int
    latency_count: int
    cost_total_present: bool
    cost_line_count: int
    recording_count: int
    provider_ref: ProviderRef | None
    end_reason: str | None
    direction: str
    analysis_field_set: bool
    provider_log_url: str | None


def _transcript(f: CoverageFacts) -> _Cell:
    return ("observed", None) if f.utterance_count else ("unavailable", "not_in_record")


def _turn_timing(f: CoverageFacts) -> _Cell:
    if f.utterance_count and f.fully_timed_utterance_count == f.utterance_count:
        return ("observed", None)
    if f.utterances_with_wall_time:
        return ("partial", "some_entries_untimed")
    return ("unavailable", "not_in_record")


def _tool_calls(f: CoverageFacts) -> _Cell:
    if f.entry_list_found and not f.unresolved_tool_count:
        return ("observed", None)
    if f.unresolved_tool_count:
        return ("partial", "result_missing")
    return ("unavailable", "not_in_record")


def _latency(f: CoverageFacts) -> _Cell:
    return ("observed", None) if f.latency_count else ("unavailable", "not_in_record")


def _cost_breakdown(f: CoverageFacts) -> _Cell:
    if f.cost_total_present and f.cost_line_count:
        return ("observed", None)
    if f.cost_total_present:
        return ("partial", "total_only")
    return ("unavailable", "not_in_record")


def _recordings(f: CoverageFacts) -> _Cell:
    return ("observed", None) if f.recording_count else ("unavailable", "not_in_record")


def _transfers(f: CoverageFacts) -> _Cell:
    # No transfer field of either provider is mapped yet.
    return ("unavailable", "not_mapped")


def _agent_version(f: CoverageFacts) -> _Cell:
    if f.provider_ref is not None and f.provider_ref.version is not None:
        return ("observed", None)
    if f.provider_ref is not None:
        return ("partial", "no_version_on_record")
    return ("unavailable", "not_in_record")


def _end_reason(f: CoverageFacts) -> _Cell:
    return ("observed", None) if f.end_reason is not None else ("unavailable", "not_in_record")


def _direction(f: CoverageFacts) -> _Cell:
    if f.direction in ("inbound", "outbound"):
        return ("observed", None)
    return ("unavailable", "not_in_record")


def _provider_analysis(f: CoverageFacts) -> _Cell:
    return ("observed", None) if f.analysis_field_set else ("unavailable", "not_in_record")


def _raw_log(f: CoverageFacts) -> _Cell:
    return ("observed", None) if f.provider_log_url is not None else ("unavailable", "not_in_record")


_RULES: dict[str, Callable[[CoverageFacts], _Cell]] = {
    "transcript": _transcript,
    "turn_timing": _turn_timing,
    "tool_calls": _tool_calls,
    "latency": _latency,
    "cost_breakdown": _cost_breakdown,
    "recordings": _recordings,
    "transfers": _transfers,
    "agent_version": _agent_version,
    "end_reason": _end_reason,
    "direction": _direction,
    "provider_analysis": _provider_analysis,
    "raw_log": _raw_log,
}


def build_coverage(declared: Sequence[str], facts: CoverageFacts) -> tuple[CoverageEntry, ...]:
    """One entry per declared capability, in the declared order.

    While a call is in progress anything missing may still arrive, so every
    entry that is not observed says so instead of blaming the record.
    """
    entries = []
    for capability in declared:
        rule = _RULES.get(capability)
        if rule is None:
            raise ValueError("a declared capability is not one of the voice capabilities")
        status, reason = rule(facts)
        if status != "observed" and facts.status == "in_progress":
            reason = "call_in_progress"
        entries.append(CoverageEntry(capability=capability, status=status, reason=reason))
    return tuple(entries)
