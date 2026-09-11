"""Tests for the voice modality (deterministic voice-episode verifier)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fi.alk import bench
from fi.alk.bench._voice import score_voice_episode

ROOT = Path(__file__).parent.parent
VOICE_SUITE = ROOT / "examples" / "bench_suites" / "voice_starter.json"

_GOOD = [
    {"speaker": "caller", "start_ms": 0, "end_ms": 1500, "text": "I want a refund"},
    {"speaker": "agent", "start_ms": 1700, "end_ms": 3500, "text": "Our refund policy is 30 days."},
    {"speaker": "caller", "start_ms": 3100, "end_ms": 3300, "text": "wait", "interrupt": True},
    {"speaker": "agent", "start_ms": 3650, "end_ms": 4200, "text": "Yes?"},
]


def test_good_episode_passes_all_dimensions() -> None:
    r = score_voice_episode(_GOOD, budgets={"max_latency_ms": 1200}, required_content=["refund", "30"])
    assert r["result"]["pass_fail"]["voice"] is True
    assert r["result"]["scalar"] == 1.0
    assert set(r["result"]["components"]) == {"latency", "turn_taking", "barge_in", "content"}


def test_slow_response_fails_latency() -> None:
    slow = [
        {"speaker": "caller", "start_ms": 0, "end_ms": 1000, "text": "I want a refund"},
        {"speaker": "agent", "start_ms": 5000, "end_ms": 6000, "text": "30 day refund"},
    ]
    r = score_voice_episode(slow, budgets={"max_latency_ms": 1200}, required_content=["refund"])
    assert r["result"]["components"]["latency"] == 0.0
    assert r["result"]["pass_fail"]["voice"] is False


def test_talking_over_caller_fails_turn_taking() -> None:
    overlap = [
        {"speaker": "caller", "start_ms": 0, "end_ms": 3000, "text": "I want a refund now please"},
        {"speaker": "agent", "start_ms": 500, "end_ms": 2500, "text": "30 day refund"},  # overlaps, no barge-in
    ]
    r = score_voice_episode(overlap, budgets={"max_latency_ms": 1200}, required_content=["refund"])
    assert r["result"]["components"]["turn_taking"] < 1.0


def test_missing_content_fails() -> None:
    r = score_voice_episode(
        [{"speaker": "caller", "start_ms": 0, "end_ms": 1000, "text": "refund?"},
         {"speaker": "agent", "start_ms": 1200, "end_ms": 2000, "text": "hello there"}],
        budgets={"max_latency_ms": 1200}, required_content=["refund", "30"])
    assert r["result"]["components"]["content"] == 0.0


def test_voice_suite_through_facade() -> None:
    suite = json.loads(VOICE_SUITE.read_text())
    ref = {t["id"]: t["reference_dialogue"] for t in suite["tasks"]}
    res = bench.run_bench(VOICE_SUITE, control_mode="artifact_in", submission=ref,
                          evidence_class="local_gate", emit_telemetry=False)
    assert res["modalities"] == ["voice"]
    assert res["aggregate"]["pass_rate"] == 1.0
    assert all(r["world_kind"] == "voice_telephony" for r in res["per_task"])


def test_voice_missing_submission_is_void() -> None:
    res = bench.run_bench(VOICE_SUITE, control_mode="artifact_in", submission={}, emit_telemetry=False)
    assert res["per_task"][0]["verdict"] == "void"


def test_voice_requires_artifact_in() -> None:
    with pytest.raises(bench.BenchError):
        bench.run_bench(VOICE_SUITE, {"x": 1}, control_mode="pull", emit_telemetry=False)
