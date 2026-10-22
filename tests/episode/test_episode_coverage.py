from __future__ import annotations

from dataclasses import replace

import pytest

from fi.alk.episode import VOICE_CAPABILITIES, ProviderRef
from fi.alk.episode.coverage import CoverageFacts, build_coverage

# Every capability observed: each table cell below changes the facts it reads.
FULL = CoverageFacts(
    status="ended",
    utterance_count=2,
    fully_timed_utterance_count=2,
    utterances_with_wall_time=2,
    entry_list_found=True,
    unresolved_tool_count=0,
    latency_count=1,
    cost_total_present=True,
    cost_line_count=2,
    recording_count=1,
    provider_ref=ProviderRef(provider="retell", id="agent-1", version="3"),
    end_reason="user_hangup",
    direction="inbound",
    analysis_field_set=True,
    provider_log_url="https://example.invalid/log",
)
# Nothing in the record at all.
EMPTY = replace(
    FULL,
    utterance_count=0,
    fully_timed_utterance_count=0,
    utterances_with_wall_time=0,
    entry_list_found=False,
    latency_count=0,
    cost_total_present=False,
    cost_line_count=0,
    recording_count=0,
    provider_ref=None,
    end_reason=None,
    direction="unknown",
    analysis_field_set=False,
    provider_log_url=None,
)


def _entry(capability, facts):
    (entry,) = build_coverage([capability], facts)
    assert entry.capability == capability
    return (entry.status, entry.reason)


def test_one_entry_per_declared_in_order():
    declared = ["direction", "transcript", "raw_log", "tool_calls"]
    assert [e.capability for e in build_coverage(declared, FULL)] == declared
    assert [e.capability for e in build_coverage(list(reversed(VOICE_CAPABILITIES)), EMPTY)] == list(
        reversed(VOICE_CAPABILITIES)
    )
    assert build_coverage([], FULL) == ()
    assert build_coverage((), EMPTY) == ()


RULE_TABLE = [
    ("transcript", FULL, ("observed", None)),
    ("transcript", EMPTY, ("unavailable", "not_in_record")),
    ("turn_timing", FULL, ("observed", None)),
    ("turn_timing", replace(FULL, fully_timed_utterance_count=1), ("partial", "some_entries_untimed")),
    (
        "turn_timing",
        replace(FULL, fully_timed_utterance_count=0, utterances_with_wall_time=0),
        ("unavailable", "not_in_record"),
    ),
    ("turn_timing", EMPTY, ("unavailable", "not_in_record")),
    ("tool_calls", FULL, ("observed", None)),
    ("tool_calls", replace(FULL, unresolved_tool_count=1), ("partial", "result_missing")),
    ("tool_calls", EMPTY, ("unavailable", "not_in_record")),
    ("latency", FULL, ("observed", None)),
    ("latency", EMPTY, ("unavailable", "not_in_record")),
    ("cost_breakdown", FULL, ("observed", None)),
    ("cost_breakdown", replace(FULL, cost_line_count=0), ("partial", "total_only")),
    ("cost_breakdown", EMPTY, ("unavailable", "not_in_record")),
    ("recordings", FULL, ("observed", None)),
    ("recordings", EMPTY, ("unavailable", "not_in_record")),
    ("transfers", FULL, ("unavailable", "not_mapped")),
    ("agent_version", FULL, ("observed", None)),
    (
        "agent_version",
        replace(FULL, provider_ref=ProviderRef(provider="vapi", id="asst-1", version=None)),
        ("partial", "no_version_on_record"),
    ),
    ("agent_version", EMPTY, ("unavailable", "not_in_record")),
    ("end_reason", FULL, ("observed", None)),
    ("end_reason", EMPTY, ("unavailable", "not_in_record")),
    ("direction", FULL, ("observed", None)),
    ("direction", replace(FULL, direction="outbound"), ("observed", None)),
    ("direction", EMPTY, ("unavailable", "not_in_record")),
    ("provider_analysis", FULL, ("observed", None)),
    ("provider_analysis", EMPTY, ("unavailable", "not_in_record")),
    ("raw_log", FULL, ("observed", None)),
    ("raw_log", EMPTY, ("unavailable", "not_in_record")),
]


@pytest.mark.parametrize(
    "capability, facts, expected",
    RULE_TABLE,
    ids=[f"{row[0]}-{row[2][0]}-{i}" for i, row in enumerate(RULE_TABLE)],
)
def test_rule_table(capability, facts, expected):
    assert _entry(capability, facts) == expected


def test_in_progress_reason():
    entries = build_coverage(VOICE_CAPABILITIES, replace(EMPTY, status="in_progress"))
    assert all(e.status != "observed" and e.reason == "call_in_progress" for e in entries)
    # The status column still follows the table; only the reason changes.
    partial = build_coverage(["cost_breakdown"], replace(FULL, status="in_progress", cost_line_count=0))
    assert [(e.status, e.reason) for e in partial] == [("partial", "call_in_progress")]
    observed = build_coverage(VOICE_CAPABILITIES, replace(FULL, status="in_progress"))
    assert [(e.status, e.reason) for e in observed if e.capability != "transfers"] == [("observed", None)] * 11
    assert [(e.status, e.reason) for e in observed if e.capability == "transfers"] == [
        ("unavailable", "call_in_progress")
    ]
    # An errored call is over, so the table's reasons apply.
    assert {e.reason for e in build_coverage(VOICE_CAPABILITIES, replace(EMPTY, status="errored"))} == {
        "not_in_record",
        "not_mapped",
    }
