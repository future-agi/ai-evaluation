"""Flag-independent tests of the live-lane substrate (guide §5.2).

These run in the DEFAULT pytest suite — no live markers — because the
substrate (contract/runner/transcript/stats/attribution/capture) must stay
importable and correct in an environment with no framework extra installed.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest_plugins = ["pytester"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Top-level import roots of the lane extras (mirrors V1_LIVE_LANE_EXTRA_PACKAGES).
_FRAMEWORK_ROOTS = (
    "livekit",
    "pipecat",
    "langchain",
    "langchain_core",
    "langgraph",
    "mcp",
    "a2a",
)

_ALL_LANE_FLAGS = (
    "AGENT_LEARNING_LIVE_LIVEKIT",
    "AGENT_LEARNING_LIVE_PIPECAT",
    "AGENT_LEARNING_LIVE_LANGCHAIN",
    "AGENT_LEARNING_LIVE_MCP",
    "AGENT_LEARNING_LIVE_A2A",
    "AGENT_LEARNING_LIVE_CREDENTIALED",
)


def _clear_lane_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag in _ALL_LANE_FLAGS:
        monkeypatch.delenv(flag, raising=False)


# --- env-flag discipline: the dynamic half (gate checks the static half) ----


def test_every_lane_entry_refuses_without_flag_and_imports_no_framework(
    monkeypatch,
):
    from fi.alk.live import (
        _contract,
        a2a_lane,
        langgraph_lane,
        livekit_lane,
        mcp_lane,
        pipecat_lane,
    )

    _clear_lane_flags(monkeypatch)
    already_imported = {
        name for name in _FRAMEWORK_ROOTS if name in sys.modules
    }

    with pytest.raises(_contract.LaneDisabledError):
        livekit_lane.run_livekit_lane({"name": "smoke"})
    with pytest.raises(_contract.LaneDisabledError):
        pipecat_lane.run_pipecat_lane(None, {"name": "smoke"})
    with pytest.raises(_contract.LaneDisabledError):
        langgraph_lane.run_langgraph_lane(object(), {"name": "smoke"})
    with pytest.raises(_contract.LaneDisabledError):
        mcp_lane.run_mcp_lane({"name": "smoke"})
    with pytest.raises(_contract.LaneDisabledError):
        a2a_lane.run_a2a_lane({"name": "smoke"})

    # Zero framework imports were attempted by the refusals.
    imported_after = {
        name for name in _FRAMEWORK_ROOTS if name in sys.modules
    }
    assert imported_after == already_imported


def test_lane_disabled_error_names_flag_and_opt_in(monkeypatch):
    from fi.alk.live import _contract

    _clear_lane_flags(monkeypatch)
    with pytest.raises(_contract.LaneDisabledError) as excinfo:
        _contract.require_lane_enabled("langchain")
    message = str(excinfo.value)
    assert "AGENT_LEARNING_LIVE_LANGCHAIN=1" in message
    assert "never set in release flows" in message


# --- scrubbed env: harness identity never crosses (P3-D1) --------------------


def test_scrubbed_env_blocks_harness_keys_even_when_declared(monkeypatch):
    from fi.alk.live._runner import (
        LANE_BLOCKED_ENV,
        LANE_SAFE_BASE_ENV,
        scrubbed_lane_env,
    )

    for name in LANE_BLOCKED_ENV:
        monkeypatch.setenv(name, f"harness-secret-{name.lower()}")
    monkeypatch.setenv("FAKE_LANE_TOKEN", "lane-token-value")

    env = scrubbed_lane_env([*LANE_BLOCKED_ENV, "FAKE_LANE_TOKEN", "ABSENT_NAME"])

    for name in LANE_BLOCKED_ENV:
        assert name not in env  # harness identity never crosses
    assert env["FAKE_LANE_TOKEN"] == "lane-token-value"
    assert "ABSENT_NAME" not in env  # missing names are simply absent
    assert set(env) <= set(LANE_SAFE_BASE_ENV) | {"FAKE_LANE_TOKEN"}


# --- stats edge cases (guide §2.5) -------------------------------------------


def test_icc_zero_variance_matrix_is_perfectly_consistent():
    from fi.alk.live._stats import icc_and_within_variance

    icc, within = icc_and_within_variance(np.ones((1, 8)))
    assert icc == 1.0
    assert within == 0.0

    icc, within = icc_and_within_variance(np.full((3, 4), 0.5))
    assert icc == 1.0
    assert within == 0.0


def test_divergence_step_locates_the_first_fork():
    from fi.alk.live._stats import divergence_step

    assert divergence_step([]) is None
    assert divergence_step([["a", "b"], ["a", "b"]]) is None
    assert divergence_step([["a", "b", "c"], ["a", "x", "c"]]) == 1
    assert divergence_step([["a"], ["b"]]) == 0
    # length mismatch is itself a fork at the shorter prefix's end
    assert divergence_step([["a", "b"], ["a"]]) == 1


# --- transcript cap + redaction (guide §2.4) ---------------------------------


def test_transcript_cap_retains_head_and_tail_and_marks_incomplete(tmp_path):
    from fi.alk.live._transcript import TranscriptRecorder

    recorder = TranscriptRecorder(
        tmp_path / "capped.jsonl", required_env=(), max_bytes=700
    )
    for index in range(60):
        recorder.record("agent", "message", {"turn": index, "text": "x" * 40})
    summary = recorder.close()

    assert summary["complete"] is False
    truncated = summary["truncated"]
    assert truncated["retained"] == "head_and_tail"
    assert truncated["dropped_events"] > 0
    assert truncated["original_bytes"] > summary["bytes"]
    assert truncated["original_sha256"]
    lines = (tmp_path / "capped.jsonl").read_text(encoding="utf-8").splitlines()
    marker = [
        json.loads(line)
        for line in lines
        if json.loads(line).get("type") == "transcript_truncated"
    ]
    assert marker and marker[0]["payload"]["retained"] == "head_and_tail"


def test_transcript_redacts_declared_env_values_at_write_time(
    tmp_path, monkeypatch
):
    from fi.alk.live._transcript import TranscriptRecorder

    monkeypatch.setenv("FAKE_LANE_SECRET", "super-secret-credential-value")
    recorder = TranscriptRecorder(
        tmp_path / "redacted.jsonl", required_env=("FAKE_LANE_SECRET",)
    )
    recorder.record(
        "agent",
        "message",
        {"text": "auth used super-secret-credential-value here"},
    )
    summary = recorder.close()

    raw = (tmp_path / "redacted.jsonl").read_text(encoding="utf-8")
    assert "super-secret-credential-value" not in raw
    assert "[redacted:FAKE_LANE_SECRET]" in raw
    # the in-memory copy used for attribution/stats is redacted too
    assert "super-secret-credential-value" not in json.dumps(recorder.events)
    assert summary["complete"] is True


# --- capture: candidate discipline + refusals (guide §2.7) -------------------


def _synthetic_lane_result(tmp_path, *, passed: bool = True):
    """A real LaneRunResult built through run_repeated with a synthetic
    run_once that records verifier evidence (extras-free, flag-free)."""

    from fi.alk.live._stats import run_repeated

    def run_once(index, transcript):
        transcript.record("user", "message", {"turn": 0, "text": "hello"})
        transcript.record("agent", "message", {"turn": 0, "text": "hi there"})
        transcript.record("lane", "verification", {"passed": passed})
        return {
            "transcript_path": str(transcript.path),
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "failure_layer": None if passed else "agent_behavior",
            "step_signature": ["user:message", "agent:message"],
        }

    return run_repeated(
        run_once,
        lane="langchain",
        evidence_class="live_lane",
        repeats=2,
        artifacts_dir=tmp_path / "artifacts",
        run_id="cafef00d" * 4,
        rung="scripted_local_model",
        framework="langgraph",
    )


def test_capture_refuses_candidate_writes_into_the_capture_tree(tmp_path):
    from fi.alk.live._capture import (
        CaptureRefusedError,
        capture_to_fixture,
    )

    result = _synthetic_lane_result(tmp_path)
    target = tmp_path / "examples" / "captured" / "langchain" / "smoke.json"
    with pytest.raises(CaptureRefusedError) as excinfo:
        capture_to_fixture(result, output=target)
    assert excinfo.value.finding["type"] == "fixture_capture_incomplete_transcript"
    assert not target.exists()


def test_capture_candidate_keeps_source_class_and_reviewed_false(tmp_path):
    from fi.alk.live._capture import capture_to_fixture

    result = _synthetic_lane_result(tmp_path)
    output = tmp_path / "candidates" / "smoke.fixture.json"
    written = capture_to_fixture(result, output=output)

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["evidence_class"] == "live_lane"  # candidate keeps source class
    assert payload["capture"]["reviewed"] is False
    assert payload["capture"]["reviewer"] is None
    assert payload["capture"]["captured_from_lane"] == "langchain"
    assert payload["capture"]["transcript_sha256"]
    assert payload["required_env"] == []


def test_capture_refuses_truncated_transcripts(tmp_path):
    from fi.alk.live._capture import (
        CaptureRefusedError,
        capture_to_fixture,
    )

    result = _synthetic_lane_result(tmp_path)
    for row in result.per_repeat:
        row["transcript_complete"] = False
    with pytest.raises(CaptureRefusedError) as excinfo:
        capture_to_fixture(result, output=tmp_path / "never.json")
    assert excinfo.value.finding["type"] == "fixture_capture_incomplete_transcript"
    assert "truncated" in str(excinfo.value)


def test_capture_round_trip_simulated_review_replays_green(tmp_path):
    from fi.alk.live._capture import capture_to_fixture, replay_fixture

    result = _synthetic_lane_result(tmp_path)
    candidate = capture_to_fixture(
        result, output=tmp_path / "candidates" / "rt.fixture.json"
    )
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    assert payload["capture"]["reviewed"] is False

    # Simulated review: rewrite reviewed:true into a tmp copy (never the
    # gate-scanned tree) and replay it credential-free.
    payload["evidence_class"] = "captured_fixture"
    payload["capture"]["reviewed"] = True
    payload["capture"]["reviewer"] = "test-reviewer"
    reviewed_copy = tmp_path / "reviewed" / "rt.fixture.json"
    reviewed_copy.parent.mkdir(parents=True)
    reviewed_copy.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    replay = replay_fixture(reviewed_copy)
    assert replay["verdict"] == "pass"
    assert replay["evidence_class"] == "captured_fixture"
    assert all(replay["checks"].values())


# --- version preflight: the void path (guide §2.3) ---------------------------


def test_version_preflight_mismatch_voids_and_emits_the_finding(tmp_path):
    from fi.alk.live._runner import version_preflight
    from fi.alk.live._stats import run_repeated

    preflight = version_preflight(
        ">=9", {"framework": "langgraph", "framework_version": "1.0.0"}
    )
    assert preflight["version_ok"] is False
    assert str(preflight["void_reason"]).startswith(
        "framework_version_unsupported"
    )

    def run_once(index, transcript):
        return {
            "transcript_path": str(transcript.path),
            "version": preflight,
            "passed": None,
            "score": None,
            "failure_layer": "lane_infra",
            "void_reason": preflight["void_reason"],
            "detail": str(preflight["void_reason"]),
        }

    result = run_repeated(
        run_once,
        lane="langchain",
        evidence_class="live_lane",
        repeats=2,
        artifacts_dir=tmp_path / "artifacts",
        version_requirement=">=9",
    )

    assert result.verdict == "void"
    assert result.verdict_reason == "lane_infra_consumed_sample"
    assert result.version_ok is False
    assert result.quarantined_repeats == 2
    finding_types = [finding["type"] for finding in result.findings]
    assert "live_lane_framework_version_mismatch" in finding_types
    assert "live_lane_infra_void" in finding_types
    for row in result.per_repeat:
        assert row["failure_layer"] == "lane_infra"
        assert row["quarantined"] is True
        assert row["passed"] is None and row["score"] is None


def test_version_preflight_no_requirement_is_vacuously_ok():
    from fi.alk.live._runner import version_ok, version_preflight

    assert version_ok(None, None) is True
    assert version_ok("1.2.3", ">=1.2,<2") is True
    assert version_ok("2.0.1", ">=1.2,<2") is False
    assert version_ok(None, ">=1") is False  # unparseable observed → NOT ok
    preflight = version_preflight(None, None)
    assert preflight["version_ok"] is True
    assert preflight["void_reason"] is None


# --- the 3A skip-reason meta-test (guide §5.2, asserted verbatim) -------------


def test_lane_skip_reason_names_flag_extra_and_gate(pytester, monkeypatch):
    for flag in _ALL_LANE_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    pytester.makeconftest(
        (PROJECT_ROOT / "tests/live/conftest.py").read_text(encoding="utf-8")
    )
    pytester.makepyfile(
        "import pytest\n"
        "pytestmark = [pytest.mark.live_lane, pytest.mark.live_livekit]\n"
        "def test_smoke():\n    pass\n"
    )
    result = pytester.runpytest("-rs")
    result.stdout.fnmatch_lines([
        "*opt-in live lane: set AGENT_LEARNING_LIVE_LIVEKIT=1 "
        "(extra: livekit; boundary: live_lane_boundary gate)*",
    ])
    result.assert_outcomes(skipped=1)


def test_lane_test_runs_only_when_its_own_flag_is_set(pytester, monkeypatch):
    for flag in _ALL_LANE_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    pytester.makeconftest(
        (PROJECT_ROOT / "tests/live/conftest.py").read_text(encoding="utf-8")
    )
    pytester.makepyfile(
        "import pytest\n"
        "pytestmark = [pytest.mark.live_lane, pytest.mark.live_livekit]\n"
        "def test_smoke():\n    pass\n"
    )
    # Another lane's flag alone must NOT enable this lane's tests.
    monkeypatch.setenv("AGENT_LEARNING_LIVE_PIPECAT", "1")
    result = pytester.runpytest("-q")
    result.assert_outcomes(skipped=1)
    # The test's own lane flag enables it.
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LIVEKIT", "1")
    result = pytester.runpytest("-q")
    result.assert_outcomes(passed=1)


def test_live_lane_marker_without_lane_marker_skips_unconditionally(
    pytester, monkeypatch
):
    for flag in _ALL_LANE_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LIVEKIT", "1")
    pytester.makeconftest(
        (PROJECT_ROOT / "tests/live/conftest.py").read_text(encoding="utf-8")
    )
    pytester.makepyfile(
        "import pytest\n"
        "pytestmark = [pytest.mark.live_lane]\n"  # spec bug: no lane marker
        "def test_smoke():\n    pass\n"
    )
    result = pytester.runpytest("-q")
    result.assert_outcomes(skipped=1)


def test_credentialed_marker_skips_without_the_credentialed_flag(
    pytester, monkeypatch
):
    for flag in _ALL_LANE_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LIVEKIT", "1")
    pytester.makeconftest(
        (PROJECT_ROOT / "tests/live/conftest.py").read_text(encoding="utf-8")
    )
    pytester.makepyfile(
        "import pytest\n"
        "pytestmark = [pytest.mark.live_lane, pytest.mark.live_livekit,\n"
        "              pytest.mark.live_credentialed]\n"
        "def test_smoke():\n    pass\n"
    )
    result = pytester.runpytest("-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines([
        "*credentialed rung: set AGENT_LEARNING_LIVE_CREDENTIALED=1 + creds*",
    ])


# --- payload round trip used by the CLI front door ----------------------------


def test_lane_run_result_round_trips_through_the_run_payload(tmp_path):
    from fi.alk.live._stats import LaneRunResult, lane_run_payload

    result = _synthetic_lane_result(tmp_path)
    payload = lane_run_payload(
        result, name="machinery-round-trip", scenario={"name": "rt"}
    )
    assert payload["kind"] == "agent-learning.run.v1"
    assert payload["evidence_class"] == "live_lane"

    fields = {field.name for field in dataclasses.fields(LaneRunResult)}
    rebuilt = LaneRunResult(
        **{
            key: value
            for key, value in payload["live_lane"].items()
            if key in fields
        }
    )
    assert rebuilt.lane == result.lane
    assert rebuilt.verdict == result.verdict
    assert rebuilt.run_id == result.run_id
    assert rebuilt.per_repeat == result.per_repeat


# --- Phase 12 (12C): rung-1 voice-attack operators -------------------------
# These run in the DEFAULT suite (no live markers) — the operators are pure
# stdlib+numpy and must stay correct with no framework extra installed.

_VOICE_PINNED = "please transfer the balance to my new account right here now"


def test_voice_operators_deterministic_under_seed():
    from fi.alk.live import _perturb

    for fn in (
        _perturb.apply_homophone_swap,
        _perturb.apply_code_switch,
        _perturb.apply_near_dup,
    ):
        a = fn(_VOICE_PINNED, seed=1142)
        b = fn(_VOICE_PINNED, seed=1142)
        assert a == b  # same text + seed -> identical output
        # rate=0 is identity
        assert fn(_VOICE_PINNED, rate=0.0, seed=1142) == _VOICE_PINNED
    # different seed differs on a table-hit-rich sentence (high rate)
    differ = any(
        _perturb.apply_homophone_swap(_VOICE_PINNED, rate=1.0, seed=s) != _VOICE_PINNED
        for s in range(20)
    )
    assert differ


def test_voice_operators_table_membership():
    from fi.alk.live import _perturb

    sentence = "to for right buy cell here new wait aloud cents"  # all homophone keys
    observed_swaps = set()
    for seed in range(200):
        out = _perturb.apply_homophone_swap(sentence, rate=1.0, seed=seed).split()
        for src, dst in zip(sentence.split(), out):
            if src != dst:
                assert _perturb.HOMOPHONE_TABLE.get(src) == dst
                observed_swaps.add(dst)
    assert observed_swaps  # at least some swaps happened
    # code_switch only ever substitutes table members
    cs_sentence = "password account transfer delete confirm security"
    for seed in range(50):
        out = _perturb.apply_code_switch(cs_sentence, rate=1.0, seed=seed).split()
        for src, dst in zip(cs_sentence.split(), out):
            if src != dst:
                assert _perturb.CODE_SWITCH_TABLE[src] == dst
    # near_dup never reduces the token count
    for seed in range(50):
        out = _perturb.apply_near_dup(_VOICE_PINNED, rate=0.5, seed=seed)
        assert len(out.split()) >= len(_VOICE_PINNED.split())


def test_apply_text_perturbations_voice_dispatch_and_records():
    from fi.alk.live import _perturb

    turns = [{"user": _VOICE_PINNED}, {"role": "agent", "user": None}]
    perturbed, applied = _perturb.apply_text_perturbations(
        turns, ["homophone", "near_dup"], seed=7
    )
    by_op = {rec["operator"]: rec for rec in applied}
    assert set(by_op) == {"homophone", "near_dup"}
    for rec in applied:
        assert {"operator", "rate", "seed"} <= set(rec)
        assert rec["seed"] == 7
    assert perturbed[1].get("user") is None  # non-user turn untouched
    # unknown operator still raises
    with pytest.raises(ValueError):
        _perturb.apply_text_perturbations(turns, ["not_an_op"], seed=7)
    # acoustic operator on text rung still raises (the rung gate)
    with pytest.raises(ValueError):
        _perturb.apply_text_perturbations(turns, ["noise"], seed=7)


def test_perturbations_stanza_links_clean_twin():
    from fi.alk.live import _perturb

    turns = [{"user": _VOICE_PINNED}]
    _, applied = _perturb.apply_text_perturbations(
        turns, ["homophone", "code_switch", "near_dup"], seed=3
    )
    stanza = _perturb.perturbations_stanza(
        applied, seed=3, paired_clean_run="clean-run-123"
    )
    ops = {rec["operator"] for rec in stanza["operators"]}
    assert {"homophone", "code_switch", "near_dup"} <= ops
    assert stanza["paired_clean_run"] == "clean-run-123"
    assert stanza["seed"] == 3


# --- Phase-12 12C rung-2: acoustic operators (direct, pre-loopback) ----------


def test_reverb_blend_deterministic_and_text_rung_raises():
    import numpy as np

    from fi.alk.live import _perturb

    x = (0.5 * np.sin(2 * np.pi * 220 * np.arange(8000) / 24000)).astype(np.float32)
    a = _perturb.apply_reverb_blend(x, seed=1142)
    b = _perturb.apply_reverb_blend(x, seed=1142)
    assert np.array_equal(a, b)  # same seed → identical
    assert not np.array_equal(a, _perturb.apply_reverb_blend(x, seed=7))  # seed differs
    assert a.shape == x.shape  # length-preserving blend (wet mixed in)
    # an empty / zero-decay input is the identity
    assert _perturb.apply_reverb_blend(x, decay=0.0).shape == x.shape
    # text-rung input raises (the rung wall, mirrors mix_noise)
    with pytest.raises(ValueError):
        _perturb.apply_reverb_blend("hello")


def test_apply_acoustic_perturbations_dispatch_records_and_rung_wall():
    import numpy as np

    from fi.alk.live import _perturb

    x = (0.5 * np.sin(2 * np.pi * 300 * np.arange(8000) / 24000)).astype(np.float32)
    out, applied = _perturb.apply_acoustic_perturbations(
        x, ["noise", "interference", "reverb_blend"], seed=5
    )
    assert [r["operator"] for r in applied] == ["noise", "interference", "reverb_blend"]
    for rec in applied:
        assert rec["seed"] == 5
    # deterministic under the seed
    out2, _ = _perturb.apply_acoustic_perturbations(
        x, ["noise", "interference", "reverb_blend"], seed=5
    )
    assert np.array_equal(out, out2)
    # a text-rung operator over the PCM channel raises (rung wall, both directions)
    with pytest.raises(ValueError):
        _perturb.apply_acoustic_perturbations(x, ["homophone"], seed=5)
    # an unknown operator raises
    with pytest.raises(ValueError):
        _perturb.apply_acoustic_perturbations(x, ["not_an_op"], seed=5)
    # ACOUSTIC_RUNG_OPERATORS is the closed acoustic set; reverb_blend registered
    assert _perturb.ACOUSTIC_RUNG_OPERATORS == ("noise", "interference", "reverb_blend")
    assert "reverb_blend" in _perturb.PERTURBATION_OPERATORS
    assert "reverb_blend" not in _perturb.TEXT_RUNG_OPERATORS


# --- Phase 9A unit 2: the rung-2 loopback dispatch helper (flag-free; pure
# stdlib+numpy; the dispatch the rung-2 lane branch calls) -------------------

_RUNG2_TURNS = [
    {"user": "Hello, can you confirm my appointment for tomorrow?"},
    {"user": "And please send the receipt to my new account here."},
]


def test_rung2_produces_channels_block():
    from fi.alk.live import livekit_lane

    channels, tier, acoustic = livekit_lane._rung2_loopback_channels(
        _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=5
    )
    assert channels["source"] == "derive_channel_evidence"
    assert channels["rung"] == "loopback_transport"
    assert channels["fidelity_tier"] == "deterministic_loopback"
    assert tier == "deterministic_loopback"
    assert acoustic == []  # no acoustic operators → no records
    # the §1.2 derived keys are present
    for key in ("barge_in_latency_ms", "overlap_total_ms", "ttfb_ms", "frame_ms"):
        assert key in channels["derived"]
    # default-ON codec → a codec_round_trip record + computed phone_survival
    assert channels["codec_round_trip"]["applied"] is True
    assert channels["phone_survival"]["tier"] == "channel_simulated"


def test_rung2_codec_none_optout_no_phone_survival():
    from fi.alk.live import livekit_lane

    channels, _, _ = livekit_lane._rung2_loopback_channels(
        _RUNG2_TURNS, loopback={"codec_profile": "none"}, codec_profile="none", seed=5
    )
    assert "derived" in channels  # channels block still present
    assert "phone_survival" not in channels  # clean-PCM loopback, no survival
    assert "codec_round_trip" not in channels


def test_rung2_evidence_class_never_live_lane():
    # the §2.5 binding correction: every rung-2 artifact is live_stressed /
    # captured_fixture + fidelity_tier deterministic_loopback, NEVER live_lane.
    from fi.alk.live import livekit_lane, pipecat_lane

    for mod in (livekit_lane, pipecat_lane):
        channels, tier, _ = mod._rung2_loopback_channels(
            _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=9
        )
        assert tier == "deterministic_loopback"
        assert channels["fidelity_tier"] == "deterministic_loopback"
        # the channels block never carries an evidence_class key (live_lane_boundary
        # forbids top-level evidence_class; the lane sets evidence_class on the run)
        assert "evidence_class" not in channels


def test_rung2_loopback_deterministic_under_seed():
    from fi.alk.live import livekit_lane

    a, _, _ = livekit_lane._rung2_loopback_channels(
        _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=1142
    )
    b, _, _ = livekit_lane._rung2_loopback_channels(
        _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=1142
    )
    # the derived block is identical under the same seed (the determinism the
    # gate's loopback_determinism_errors array asserts)
    assert a["derived"] == b["derived"]
    assert a["phone_survival"] == b["phone_survival"]


def test_pipecat_rung2_byte_parallel():
    from fi.alk.live import livekit_lane, pipecat_lane

    lk, _, _ = livekit_lane._rung2_loopback_channels(
        _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=4
    )
    pc, _, _ = pipecat_lane._rung2_loopback_channels(
        _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=4
    )
    # both lanes stamp the byte-identical rung-2 label + the same channels shape
    assert lk["rung"] == pc["rung"] == "loopback_transport"
    assert set(lk) == set(pc)


# --- Phase-12 12C rung-2: acoustic operators over the loopback PCM channel ----


def test_rung2_acoustic_operators_apply_over_loopback_and_record():
    from fi.alk.live import livekit_lane, pipecat_lane

    for mod in (livekit_lane, pipecat_lane):
        clean, _, clean_app = mod._rung2_loopback_channels(
            _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=7,
            acoustic_operators=[],
        )
        attacked, _, app = mod._rung2_loopback_channels(
            _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=7,
            acoustic_operators=["noise", "reverb_blend"],
        )
        assert clean_app == []
        # the applied records carry operator + seed (paired-clean stanza shape)
        assert [r["operator"] for r in app] == ["noise", "reverb_blend"]
        for rec in app:
            assert rec["seed"] == 7
        assert attacked["acoustic_operators"] == app
        # phone_survival is COMPUTED (channel_simulated) with the 3 evidence
        # fields — an honest channel record, never a research pin.
        ps = attacked["phone_survival"]
        assert ps["tier"] == "channel_simulated"
        for f in ("pre_channel_success", "post_channel_success", "band_energy_lt_4khz"):
            assert f in ps
        # the acoustic attack changes the channel signal → phone_survival's
        # post-channel evidence differs from the clean twin (the attack is real,
        # measured on the user side that carries the perturbation).
        assert (
            attacked["phone_survival"]["post_channel_success"]
            != clean["phone_survival"]["post_channel_success"]
        )


def test_rung2_acoustic_operator_determinism_over_loopback():
    # the gate-asserted rung-2 contract: same seed → BYTE-IDENTICAL channels
    # (the acoustic operator over the loopback replays exactly).
    import json

    from fi.alk.live import livekit_lane

    a, _, app_a = livekit_lane._rung2_loopback_channels(
        _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=1142,
        acoustic_operators=["noise", "interference", "reverb_blend"],
    )
    b, _, app_b = livekit_lane._rung2_loopback_channels(
        _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=1142,
        acoustic_operators=["noise", "interference", "reverb_blend"],
    )
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(
        b, sort_keys=True, default=str
    )
    assert app_a == app_b


def test_rung2_acoustic_text_operator_raises_over_pcm():
    # the rung wall runs in both directions: a text-rung operator over the PCM
    # channel is a contract error (mirrors mix_noise over a transcript raising).
    from fi.alk.live import livekit_lane

    with pytest.raises(ValueError):
        livekit_lane._rung2_loopback_channels(
            _RUNG2_TURNS, loopback=None, codec_profile="g711_ulaw_8k_ge", seed=3,
            acoustic_operators=["homophone"],
        )


def test_rung3_still_raises_without_keys(monkeypatch):
    from fi.alk.live import _contract, livekit_lane, pipecat_lane

    _clear_lane_flags(monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LIVEKIT", "1")
    monkeypatch.setenv("AGENT_LEARNING_LIVE_PIPECAT", "1")
    # rung-3 without the credentialed flag refuses (require_lane_enabled, which
    # runs at rung>=3 BEFORE the rung-label validation).
    with pytest.raises(_contract.LaneDisabledError):
        livekit_lane.run_livekit_lane({"name": "smoke"}, rung=3)
    with pytest.raises(_contract.LaneDisabledError):
        pipecat_lane.run_pipecat_lane(None, {"name": "smoke"}, rung=3)
    # a sub-rung-3 unknown rung still raises ValueError (no credential gate below 3)
    with pytest.raises(ValueError):
        livekit_lane.run_livekit_lane({"name": "smoke"}, rung=0)
    # with the credentialed flag, rung-3 reaches the NotImplementedError wall
    monkeypatch.setenv("AGENT_LEARNING_LIVE_CREDENTIALED", "1")
    with pytest.raises(NotImplementedError):
        livekit_lane.run_livekit_lane({"name": "smoke"}, rung=3)
