from __future__ import annotations

import decimal
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from fi.alk.episode import (
    EVENT_PAYLOAD_MODELS,
    CoverageEntry,
    EpisodeVersion,
    Latency,
    ProviderAnalysis,
    VoiceEpisode,
)
from fi.alk.episode.ids import derive_episode_event_id
from fi.alk.episode.models import money_to_micros, parse_record_time

STARTED = {"tool_call_id": "call-1", "name": "lookup", "arguments": {}}
COMPLETED = {"tool_call_id": "call-1", "result": {"ok": True}}
FAILED = {"tool_call_id": "call-1", "error": "timeout", "result": None}


def _renumbered(episode):
    for seq, event in enumerate(episode["events"]):
        event["seq"] = seq
        event["event_id"] = derive_episode_event_id(episode["id"], event["source"], seq)
    episode["version"]["event_count"] = len(episode["events"])
    return episode


def _seq_not_position(build):
    episode = build(body=[("utterance", {"role": "caller", "text": "hi", "duration_ms": None})])
    episode["events"][1]["seq"] = 7
    return episode


def _episode_id_differs(build):
    episode = build()
    episode["events"][0]["episode_id"] = "episode_00000000000000000000000000000000"
    return episode


def _event_id_not_derived(build):
    episode = build()
    episode["events"][0]["event_id"] = "event_00000000000000000000000000000000"
    return episode


def _event_count_wrong(build):
    episode = build()
    episode["version"]["event_count"] += 1
    return episode


def _ended_flag_wrong(build):
    episode = build()
    episode["version"]["ended"] = False
    return episode


def _second_started(build):
    return build(body=[("episode.started", {})])


def _first_not_started(build):
    episode = build(body=[("utterance", {"role": "caller", "text": "hi", "duration_ms": None})])
    episode["events"][0], episode["events"][1] = episode["events"][1], episode["events"][0]
    return _renumbered(episode)


def _no_ended_when_over(build):
    return build(status="ended", with_ended=False)


def _ended_while_in_progress(build):
    return build(status="in_progress", with_ended=True, ended_at=None)


def _second_ended(build):
    return build(body=[("episode.ended", {})])


def _ended_not_last(build):
    episode = build(body=[("utterance", {"role": "caller", "text": "hi", "duration_ms": None})])
    episode["events"][1], episode["events"][2] = episode["events"][2], episode["events"][1]
    return _renumbered(episode)


def _tool_started_without_close(build):
    return build(body=[("tool.started", STARTED)])


def _tool_closed_twice(build):
    return build(body=[("tool.started", STARTED), ("tool.completed", COMPLETED), ("tool.failed", FAILED)])


def _tool_close_without_start(build):
    return build(body=[("tool.completed", COMPLETED)])


def _duplicate_tool_call_id(build):
    return build(
        body=[
            ("tool.started", STARTED),
            ("tool.completed", COMPLETED),
            ("tool.started", STARTED),
            ("tool.completed", COMPLETED),
        ]
    )


def _bad_known_payload(build):
    return build(body=[("utterance", {"role": "caller", "text": "hi"})])


def _coverage_repeats_a_capability(build):
    entry = {"capability": "transcript", "status": "unavailable", "reason": "not_in_record"}
    return build(coverage=[entry, entry])


INVARIANT_CASES = {
    "seq_equals_position": _seq_not_position,
    "episode_id_on_every_event": _episode_id_differs,
    "event_id_derived": _event_id_not_derived,
    "event_count_equals_len": _event_count_wrong,
    "ended_iff_ended_at": _ended_flag_wrong,
    "only_one_started": _second_started,
    "started_is_first": _first_not_started,
    "ended_present_when_over": _no_ended_when_over,
    "no_ended_when_in_progress": _ended_while_in_progress,
    "ended_is_last": _ended_not_last,
    "only_one_ended": _second_ended,
    "tool_started_needs_a_close": _tool_started_without_close,
    "tool_closed_once": _tool_closed_twice,
    "tool_close_needs_a_start": _tool_close_without_start,
    "tool_call_id_unique": _duplicate_tool_call_id,
    "known_payload_valid": _bad_known_payload,
    "coverage_capability_unique": _coverage_repeats_a_capability,
}


def test_valid_builder_episode_is_accepted(build_episode, python_form):
    # Guards the parametrized cases below: each must fail for its own edit, not the base.
    for episode in (
        build_episode(),
        build_episode(status="in_progress"),
        build_episode(body=[("tool.started", STARTED), ("tool.completed", COMPLETED)]),
    ):
        VoiceEpisode(**python_form(episode))
        VoiceEpisode.model_validate_json(json.dumps(episode))


def test_errored_call_without_end_time_is_accepted(build_episode, python_form):
    # A call that never connected has no end time, yet still closes with episode.ended.
    errored = build_episode(status="errored", ended_at=None)
    for episode in (VoiceEpisode(**python_form(errored)), VoiceEpisode.model_validate_json(json.dumps(errored))):
        assert episode.version.ended is False
        assert episode.status == "errored"
        assert episode.ended_at is None
        assert episode.events[-1].type == "episode.ended" and episode.events[-1].wall_time is None


@pytest.mark.parametrize("make", INVARIANT_CASES.values(), ids=INVARIANT_CASES.keys())
def test_invariants_each_rejected(make, build_episode, python_form):
    episode = make(build_episode)
    with pytest.raises(ValidationError):
        VoiceEpisode(**python_form(episode))
    with pytest.raises(ValidationError):
        VoiceEpisode.model_validate_json(json.dumps(episode))


def test_version_orders_in_progress_before_ended():
    versions = [
        EpisodeVersion(ended=True, event_count=2),
        EpisodeVersion(ended=False, event_count=9),
        EpisodeVersion(ended=True, event_count=5),
        EpisodeVersion(ended=False, event_count=3),
    ]
    ordered = sorted(versions, key=EpisodeVersion.ordering_key)
    assert [(v.ended, v.event_count) for v in ordered] == [(False, 3), (False, 9), (True, 2), (True, 5)]
    assert max(versions, key=EpisodeVersion.ordering_key) == EpisodeVersion(ended=True, event_count=5)


def test_micros_and_rounding():
    assert money_to_micros(0.0165, in_cents=False) == 16500
    assert money_to_micros(12.5, in_cents=True) == 125000
    # Half up on the decimal the provider wrote, not on the nearest binary float.
    assert money_to_micros(0.0000005, in_cents=False) == 1
    assert money_to_micros(0.0000004, in_cents=False) == 0
    assert money_to_micros(0.00005, in_cents=True) == 1
    assert money_to_micros(3, in_cents=False) == 3_000_000
    for bad in (-0.01, float("nan"), float("inf"), "0.5", None, True):
        assert money_to_micros(bad, in_cents=False) is None
    # Past the decimal context's precision: null, not an exception.
    assert money_to_micros(1e22, in_cents=False) is None
    assert money_to_micros(10**22, in_cents=False) is None


def test_micros_ignore_the_ambient_decimal_context():
    # A host that lowers the thread's precision would round 1234567890.5 down first.
    with decimal.localcontext() as ambient:
        ambient.prec = 10
        assert money_to_micros(1234.5678905, in_cents=False) == 1234567891


def test_parse_record_time():
    expected = datetime(2026, 10, 12, 5, 0, tzinfo=timezone.utc)
    assert parse_record_time("2026-10-12T05:00:00.000Z") == expected
    assert parse_record_time("2026-10-12T05:00:00+00:00") == expected
    assert parse_record_time("2026-10-12T07:00:00+02:00") == expected
    assert parse_record_time("2026-10-12T05:00:00") == expected  # naive is UTC
    assert parse_record_time(1791781200000) == expected  # epoch milliseconds
    for parsed in (parse_record_time("2026-10-12T07:00:00+02:00"), parse_record_time("2026-10-12T05:00:00")):
        assert parsed.utcoffset().total_seconds() == 0
    for garbage in ("garbage", "", None, True, [], {}, float("nan"), float("inf")):
        assert parse_record_time(garbage) is None
    # Valid in their own zone, outside the calendar once shifted to UTC.
    assert parse_record_time("0001-01-01T00:00:00+01:00") is None
    assert parse_record_time("9999-12-31T23:59:59-01:00") is None


def test_coverage_entry_reason_null_exactly_when_observed():
    CoverageEntry(capability="transcript", status="observed", reason=None)
    CoverageEntry(capability="transcript", status="partial", reason="some_entries_untimed")
    CoverageEntry(capability="transcript", status="unavailable", reason="not_in_record")
    with pytest.raises(ValidationError):
        CoverageEntry(capability="transcript", status="observed", reason="not_in_record")
    with pytest.raises(ValidationError):
        CoverageEntry(capability="transcript", status="partial", reason=None)
    with pytest.raises(ValidationError):
        CoverageEntry(capability="transcript", status="unavailable", reason=None)


def test_latency_needs_one_number():
    Latency(metric="e2e", avg_ms=None, p50_ms=None, p90_ms=None, p99_ms=800)
    with pytest.raises(ValidationError):
        Latency(metric="e2e", avg_ms=None, p50_ms=None, p90_ms=None, p99_ms=None)


def test_header_models_do_not_coerce():
    # The exported schema refuses these, so the model must too.
    with pytest.raises(ValidationError):
        Latency(metric="e2e", avg_ms=True, p50_ms=None, p90_ms=None, p99_ms=None)
    with pytest.raises(ValidationError):
        ProviderAnalysis(summary=None, success="yes", sentiment=None, structured=None)


@pytest.mark.parametrize("duration_ms", ["5", True], ids=["string", "bool"])
def test_known_payloads_do_not_coerce(duration_ms, build_episode, python_form):
    payload = {"role": "caller", "text": "hi", "duration_ms": duration_ms}
    utterance = EVENT_PAYLOAD_MODELS["utterance"]
    with pytest.raises(ValidationError):
        utterance.model_validate(payload)
    with pytest.raises(ValidationError):
        utterance.model_validate_json(json.dumps(payload))
    episode = build_episode(body=[("utterance", payload)])
    with pytest.raises(ValidationError):
        VoiceEpisode(**python_form(episode))
    with pytest.raises(ValidationError):
        VoiceEpisode.model_validate_json(json.dumps(episode))


def test_known_payload_refuses_nan():
    with pytest.raises(ValidationError):
        EVENT_PAYLOAD_MODELS["tool.completed"].model_validate({"tool_call_id": "call-1", "result": float("nan")})
