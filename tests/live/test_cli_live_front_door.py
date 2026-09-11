"""CLI front door for live lanes (Phase 3 guide §6) — extras-free tests.

One test per finding type plus one per exit-policy branch (fail / >0.5 void /
<=0.5 void / unstable-only), all stub-worker, all in the DEFAULT suite: the
front door's refusals and exit policy must work in an env with no framework
extra installed and no lane flag set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fi.alk import cli

# Framework roots no sanctioned release-surface import may pull in. livekit /
# langchain / langchain_core are excluded: the four pre-existing vendored
# guarded-import sites (V1_LIVE_LANE_GUARDED_IMPORT_FILES) legally import
# them when the extra happens to be installed — via the vendored simulate
# engine the `run` command loads, NOT via the live-lane path.
_FRAMEWORK_ROOTS = (
    "pipecat",
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


def _write_manifest(tmp_path: Path, stanza=None, **extra) -> Path:
    manifest = {"name": "live-front-door", **extra}
    if stanza is not None:
        manifest["live_lane"] = stanza
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _run_cli(tmp_path: Path, manifest_path: Path, *args: str):
    out = tmp_path / "out.json"
    exit_code = cli.main(
        ["run", str(manifest_path), "-o", str(out), "--quiet", *args]
    )
    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return exit_code, payload


def _stub_lane_payload(verdict: str, *, findings=(), scenario=None):
    return {
        "kind": "agent-learning.run.v1",
        "name": f"stub-{verdict}",
        "evidence_class": "live_lane",
        "scenario": dict(scenario or {}),
        "live_lane": {
            "lane": "langchain",
            "evidence_class": "live_lane",
            "verdict": verdict,
            "verdict_reason": None,
            "repeats": 2,
            "repeats_completed": 2,
            "quarantined_repeats": 2 if verdict == "void" else 0,
            "icc": None if verdict == "void" else 1.0,
            "within_variance": 0.0,
            "divergence_step": None,
            "determinism": {
                "distinct_trajectory_count": 1,
                "trajectory_entropy": 0.0,
            },
            "per_repeat": [],
            "required_env": [],
            "end_state_diff": None,
            "run_id": "stub0000",
            "rung": "scripted_local_model",
            "framework": "langgraph",
            "framework_version": None,
            "version_requirement": None,
            "version_ok": None,
            "repeats_requested": 2,
            "budget_cap_s": 600.0,
            "budget_spent_s": 0.1,
            "findings": list(findings),
            "artifacts_dir": None,
        },
        "findings": list(findings),
        "summary": {"verdict": verdict},
    }


def _install_lane_stub(monkeypatch: pytest.MonkeyPatch, payload_by_scenario):
    import fi.alk.live as live

    calls = []

    def _stub_run_lane(lane, *args, **kwargs):
        scenario = args[1] if lane in {"pipecat", "langchain"} else args[0]
        calls.append({"lane": lane, "scenario": dict(scenario), **kwargs})
        return payload_by_scenario[str(scenario.get("name"))]

    monkeypatch.setattr(live, "run_lane", _stub_run_lane)
    monkeypatch.setattr(cli, "_live_lane_extra_available", lambda lane: True)
    return calls


_LANGCHAIN_STANZA = {
    "lane": "langchain",
    "factory": "stub_factory_mod:make_graph",
    "scenario": {"name": "s1"},
}


# --- finding: live_lane_flag_required (run + redteam; zero framework imports) -


def test_run_flag_required_finding_with_zero_framework_imports(
    tmp_path, monkeypatch
):
    _clear_lane_flags(monkeypatch)
    manifest_path = _write_manifest(tmp_path, dict(_LANGCHAIN_STANZA))
    already = {name for name in _FRAMEWORK_ROOTS if name in sys.modules}

    exit_code, payload = _run_cli(tmp_path, manifest_path)

    assert exit_code == 1
    assert payload["status"] == "failed"
    finding = payload["findings"][0]
    assert finding["type"] == "live_lane_flag_required"
    assert finding["flag"] == "AGENT_LEARNING_LIVE_LANGCHAIN"
    assert finding["lane"] == "langchain"
    assert payload["summary"]["lane_executed"] is False
    after = {name for name in _FRAMEWORK_ROOTS if name in sys.modules}
    assert after == already  # the refusal attempted zero framework imports


def test_redteam_flag_required_finding(tmp_path, monkeypatch):
    _clear_lane_flags(monkeypatch)
    manifest_path = _write_manifest(tmp_path, dict(_LANGCHAIN_STANZA))
    out = tmp_path / "redteam-out.json"

    exit_code = cli.main(
        ["redteam", str(manifest_path), "-o", str(out), "--quiet"]
    )

    assert exit_code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["findings"][0]["type"] == "live_lane_flag_required"


def test_credentialed_rung_without_credentialed_flag_is_flag_required(
    tmp_path, monkeypatch
):
    _clear_lane_flags(monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LANGCHAIN", "1")
    stanza = {**_LANGCHAIN_STANZA, "rung": 2}
    manifest_path = _write_manifest(tmp_path, stanza)

    exit_code, payload = _run_cli(tmp_path, manifest_path)

    assert exit_code == 1
    finding = payload["findings"][0]
    assert finding["type"] == "live_lane_flag_required"
    assert finding["flag"] == "AGENT_LEARNING_LIVE_CREDENTIALED"


# --- finding: live_credential_missing (names listed, values never) ----------


def test_credential_missing_finding_lists_names_only(tmp_path, monkeypatch):
    _clear_lane_flags(monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LANGCHAIN", "1")
    monkeypatch.setenv("AGENT_LEARNING_LIVE_CREDENTIALED", "1")
    monkeypatch.setenv("FAKE_LANE_TOKEN_A", "present-value")
    monkeypatch.delenv("FAKE_LANE_TOKEN_B", raising=False)
    stanza = {
        **_LANGCHAIN_STANZA,
        "rung": 2,
        "required_env": ["FAKE_LANE_TOKEN_A", "FAKE_LANE_TOKEN_B"],
    }
    manifest_path = _write_manifest(tmp_path, stanza)

    exit_code, payload = _run_cli(tmp_path, manifest_path)

    assert exit_code == 1
    finding = payload["findings"][0]
    assert finding["type"] == "live_credential_missing"
    assert finding["missing"] == ["FAKE_LANE_TOKEN_B"]
    assert "1 of 2" in finding["reason"]
    assert "present-value" not in json.dumps(payload)  # names only, never values


# --- exit policy branches (MF6: fail / >0.5 void / <=0.5 void / unstable) ----


def test_exit_policy_any_scenario_fail_exits_one(tmp_path, monkeypatch):
    _clear_lane_flags(monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LANGCHAIN", "1")
    _install_lane_stub(
        monkeypatch,
        {
            "s1": _stub_lane_payload("pass", scenario={"name": "s1"}),
            "s2": _stub_lane_payload("fail", scenario={"name": "s2"}),
        },
    )
    stanza = {
        "lane": "langchain",
        "factory": "stub_factory_mod:make_graph",
        "scenarios": [{"name": "s1"}, {"name": "s2"}],
    }
    manifest_path = _write_manifest(tmp_path, stanza)

    exit_code, payload = _run_cli(tmp_path, manifest_path)

    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["summary"]["verdicts"] == {
        "pass": 1,
        "fail": 1,
        "unstable": 0,
        "void": 0,
    }


def test_exit_policy_void_rate_above_half_exits_one_with_void_finding(
    tmp_path, monkeypatch
):
    _clear_lane_flags(monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LANGCHAIN", "1")
    void_finding = {
        "type": "live_lane_infra_void",
        "level": "error",
        "detail": "lane_infra consumed the sample (no scoreable repeats)",
    }
    _install_lane_stub(
        monkeypatch,
        {
            "s1": _stub_lane_payload(
                "void", findings=[void_finding], scenario={"name": "s1"}
            )
        },
    )
    manifest_path = _write_manifest(
        tmp_path,
        {
            "lane": "langchain",
            "factory": "stub_factory_mod:make_graph",
            "scenario": {"name": "s1"},
        },
    )

    exit_code, payload = _run_cli(tmp_path, manifest_path)

    assert exit_code == 1  # void rate 1.0 > 0.5: lane infrastructure unusable
    assert payload["summary"]["void_rate"] == 1.0
    types = [finding["type"] for finding in payload["findings"]]
    assert "live_lane_infra_void" in types
    assert payload["scenarios"][0]["failure_layer"] == "lane_infra"
    assert payload["scenarios"][0]["scored"] is False


def test_exit_policy_void_rate_at_or_below_half_exits_zero_but_keeps_finding(
    tmp_path, monkeypatch
):
    _clear_lane_flags(monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LANGCHAIN", "1")
    void_finding = {
        "type": "live_lane_infra_void",
        "level": "error",
        "detail": "lane_infra consumed the sample (no scoreable repeats)",
    }
    _install_lane_stub(
        monkeypatch,
        {
            "s1": _stub_lane_payload("pass", scenario={"name": "s1"}),
            "s2": _stub_lane_payload(
                "void", findings=[void_finding], scenario={"name": "s2"}
            ),
        },
    )
    manifest_path = _write_manifest(
        tmp_path,
        {
            "lane": "langchain",
            "factory": "stub_factory_mod:make_graph",
            "scenarios": [{"name": "s1"}, {"name": "s2"}],
        },
    )

    exit_code, payload = _run_cli(tmp_path, manifest_path)

    assert exit_code == 0  # voids at or below half exit 0 on voids alone
    assert payload["status"] == "passed"
    assert payload["summary"]["void_rate"] == 0.5
    # every void still emits its finding regardless of exit code
    types = [finding["type"] for finding in payload["findings"]]
    assert "live_lane_infra_void" in types


def test_exit_policy_unstable_only_exits_zero_with_quarantine_finding(
    tmp_path, monkeypatch
):
    _clear_lane_flags(monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LANGCHAIN", "1")
    unstable_finding = {
        "type": "live_lane_scenario_unstable",
        "level": "warning",
        "detail": {"reason": "mixed_outcomes", "icc": 0.31, "divergence_step": 2},
    }
    _install_lane_stub(
        monkeypatch,
        {
            "s1": _stub_lane_payload(
                "unstable", findings=[unstable_finding], scenario={"name": "s1"}
            )
        },
    )
    manifest_path = _write_manifest(
        tmp_path,
        {
            "lane": "langchain",
            "factory": "stub_factory_mod:make_graph",
            "scenario": {"name": "s1"},
        },
    )

    exit_code, payload = _run_cli(tmp_path, manifest_path)

    assert exit_code == 0  # unstable does NOT flip red — quarantined instead
    assert payload["status"] == "passed"
    types = [finding["type"] for finding in payload["findings"]]
    assert "live_lane_scenario_unstable" in types
    assert payload["scenarios"][0]["quarantined"] is True
    assert payload["scenarios"][0]["scored"] is False


# --- finding: live_lane_framework_version_mismatch surfaces ------------------


def test_version_mismatch_finding_surfaces_and_voids(tmp_path, monkeypatch):
    _clear_lane_flags(monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LANGCHAIN", "1")
    mismatch_finding = {
        "type": "live_lane_framework_version_mismatch",
        "level": "error",
        "repeat": 0,
        "detail": "framework_version_unsupported: observed '1.0.0', required '>=9'",
    }
    _install_lane_stub(
        monkeypatch,
        {
            "s1": _stub_lane_payload(
                "void", findings=[mismatch_finding], scenario={"name": "s1"}
            )
        },
    )
    manifest_path = _write_manifest(
        tmp_path,
        {
            "lane": "langchain",
            "factory": "stub_factory_mod:make_graph",
            "scenario": {"name": "s1"},
            "version_requirement": ">=9",
        },
    )

    exit_code, payload = _run_cli(tmp_path, manifest_path)

    assert exit_code == 1  # the whole sample voided -> void rate 1.0
    types = [finding["type"] for finding in payload["findings"]]
    assert "live_lane_framework_version_mismatch" in types


# --- --repeats plumbing -------------------------------------------------------


def test_repeats_flag_overrides_stanza_and_reaches_the_lane(
    tmp_path, monkeypatch
):
    _clear_lane_flags(monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LANGCHAIN", "1")
    calls = _install_lane_stub(
        monkeypatch, {"s1": _stub_lane_payload("pass", scenario={"name": "s1"})}
    )
    manifest_path = _write_manifest(
        tmp_path,
        {
            "lane": "langchain",
            "factory": "stub_factory_mod:make_graph",
            "scenario": {"name": "s1"},
            "repeats": 2,
        },
    )

    exit_code, payload = _run_cli(tmp_path, manifest_path, "--repeats", "5")

    assert exit_code == 0
    assert calls and calls[0]["repeats"] == 5  # CLI override beats the stanza
    assert payload["summary"]["repeats_per_scenario"] == 5


def test_repeats_without_live_lane_stanza_is_an_error_finding(
    tmp_path, monkeypatch
):
    _clear_lane_flags(monkeypatch)
    manifest_path = _write_manifest(tmp_path)  # no live_lane stanza

    exit_code, payload = _run_cli(tmp_path, manifest_path, "--repeats", "4")

    assert exit_code == 1
    assert payload["findings"][0]["type"] == "live_lane_repeats_requires_lane"
    assert payload["summary"]["lane_executed"] is False


def test_redteam_repeats_without_stanza_is_an_error_finding(
    tmp_path, monkeypatch
):
    _clear_lane_flags(monkeypatch)
    manifest_path = _write_manifest(tmp_path)
    out = tmp_path / "redteam-repeats.json"

    exit_code = cli.main(
        ["redteam", str(manifest_path), "-o", str(out), "--quiet",
         "--repeats", "4"]
    )

    assert exit_code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["findings"][0]["type"] == "live_lane_repeats_requires_lane"


# --- capture-fixture subcommand (finding: fixture_capture_incomplete_transcript)


def _live_run_artifact(tmp_path: Path, *, complete: bool = True) -> Path:
    """A REAL single-scenario lane artifact built through run_repeated with a
    synthetic run_once (extras-free, flag-free)."""

    from fi.alk.live._stats import lane_run_payload, run_repeated

    def run_once(index, transcript):
        transcript.record("user", "message", {"turn": 0, "text": "hello"})
        transcript.record("agent", "message", {"turn": 0, "text": "hi there"})
        transcript.record("lane", "verification", {"passed": True})
        return {
            "transcript_path": str(transcript.path),
            "passed": True,
            "score": 1.0,
            "failure_layer": None,
            "step_signature": ["user:message", "agent:message"],
        }

    result = run_repeated(
        run_once,
        lane="langchain",
        evidence_class="live_lane",
        repeats=2,
        artifacts_dir=tmp_path / "artifacts",
        run_id="feedc0de" * 4,
        rung="scripted_local_model",
        framework="langgraph",
    )
    if not complete:
        for row in result.per_repeat:
            row["transcript_complete"] = False
    payload = lane_run_payload(
        result, name="capture-source", scenario={"name": "s1"}
    )
    artifact = tmp_path / "live_run.json"
    artifact.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return artifact


# --- Phase 9A unit 5: the live_lane.loopback sub-stanza (rung == 2 only) ----

_LIVEKIT_RUNG2_STANZA = {
    "lane": "livekit",
    "rung": 2,
    "scenario": {"name": "s1"},
}


def test_dispatch_rung2_loopback_stanza_reaches_lane():
    # the dispatch reads the loopback sub-stanza ONLY at rung == 2 and passes
    # loopback= + codec_profile= into the lane runner (the unit-2 signature).
    import fi.alk.live as live

    captured = {}

    def _stub_run_lane(lane, *args, **kwargs):
        captured.update({"lane": lane, **kwargs})
        return {"live_lane": {}}

    import unittest.mock as mock
    with mock.patch.object(live, "run_lane", _stub_run_lane):
        cli._dispatch_live_lane_scenario(
            live,
            "livekit",
            {"name": "s1"},
            {"loopback": {"user_wav": "u.wav", "codec_profile": "g711_alaw_8k_ge"}},
            {"repeats": 4},
            2,
        )
    assert captured["loopback"]["user_wav"] == "u.wav"
    assert captured["codec_profile"] == "g711_alaw_8k_ge"


def test_dispatch_rung1_ignores_loopback_stanza():
    # rung-1 manifests are unaffected: the loopback stanza is NOT read.
    import fi.alk.live as live

    captured = {}

    def _stub_run_lane(lane, *args, **kwargs):
        captured.update(kwargs)
        return {"live_lane": {}}

    import unittest.mock as mock
    with mock.patch.object(live, "run_lane", _stub_run_lane):
        cli._dispatch_live_lane_scenario(
            live, "livekit", {"name": "s1"}, {"loopback": {"user_wav": "u.wav"}},
            {"repeats": 4}, 1,
        )
    assert "loopback" not in captured
    assert "codec_profile" not in captured


def test_dispatch_rung2_invalid_codec_profile_raises():
    import fi.alk.live as live

    with pytest.raises(ValueError):
        cli._dispatch_live_lane_scenario(
            live, "livekit", {"name": "s1"},
            {"loopback": {"codec_profile": "not_a_profile"}}, {"repeats": 4}, 2,
        )


def test_dispatch_rung2_invalid_tick_raises():
    import fi.alk.live as live

    with pytest.raises(ValueError):
        cli._dispatch_live_lane_scenario(
            live, "livekit", {"name": "s1"},
            {"loopback": {"tick_ms": -1}}, {"repeats": 4}, 2,
        )


def test_cli_loopback_missing_fixture_finding(tmp_path, monkeypatch):
    # a rung-2 run whose user_wav fixture is missing -> exit 1 +
    # loopback_user_fixture_missing naming the path; lane not "succeeded".
    _clear_lane_flags(monkeypatch)
    monkeypatch.setenv("AGENT_LEARNING_LIVE_LIVEKIT", "1")
    monkeypatch.setattr(cli, "_live_lane_extra_available", lambda lane: True)
    stanza = {
        "lane": "livekit",
        "rung": 2,
        "scenario": {"name": "s1", "turns": [{"user": "hi", "turn_id": "turn_1"}]},
        "loopback": {
            "user_wav": [{"turn_id": "turn_1", "wav": str(tmp_path / "absent.wav")}]
        },
    }
    manifest_path = _write_manifest(tmp_path, stanza)
    exit_code, payload = _run_cli(tmp_path, manifest_path)
    assert exit_code == 1
    finding = payload["findings"][0]
    assert finding["type"] == "loopback_user_fixture_missing"
    assert "absent.wav" in str(finding["missing"])


def test_capture_fixture_writes_a_candidate(tmp_path, capsys):
    artifact = _live_run_artifact(tmp_path)
    output = tmp_path / "candidates" / "s1.fixture.json"

    exit_code = cli.main(
        ["simulate", "capture-fixture", str(artifact), "-o", str(output)]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "passed"
    assert summary["fixture"]["reviewed"] is False
    fixture = json.loads(output.read_text(encoding="utf-8"))
    assert fixture["evidence_class"] == "live_lane"  # candidate keeps source class
    assert fixture["capture"]["reviewed"] is False


def test_capture_fixture_refuses_candidates_in_the_capture_tree(
    tmp_path, capsys
):
    artifact = _live_run_artifact(tmp_path)
    output = tmp_path / "examples" / "captured" / "langchain" / "s1.json"

    exit_code = cli.main(
        ["capture-fixture", str(artifact), "-o", str(output)]
    )

    assert exit_code == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "failed"
    assert (
        summary["findings"][0]["type"]
        == "fixture_capture_incomplete_transcript"
    )
    assert not output.exists()


def test_capture_fixture_refuses_truncated_transcripts(tmp_path, capsys):
    artifact = _live_run_artifact(tmp_path, complete=False)
    output = tmp_path / "candidates" / "s1.fixture.json"

    exit_code = cli.main(
        ["simulate", "capture-fixture", str(artifact), "-o", str(output)]
    )

    assert exit_code == 1
    summary = json.loads(capsys.readouterr().out)
    assert (
        summary["findings"][0]["type"]
        == "fixture_capture_incomplete_transcript"
    )
    assert not output.exists()


def test_capture_fixture_reviewed_by_stamps_and_replays_green(
    tmp_path, capsys
):
    artifact = _live_run_artifact(tmp_path)
    output = tmp_path / "reviewed" / "s1.fixture.json"

    exit_code = cli.main(
        [
            "simulate",
            "capture-fixture",
            str(artifact),
            "-o",
            str(output),
            "--reviewed-by",
            "test-reviewer",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["fixture"]["evidence_class"] == "captured_fixture"
    assert summary["fixture"]["reviewed"] is True
    assert summary["fixture"]["reviewer"] == "test-reviewer"
    assert summary["replay"]["verdict"] == "pass"
    fixture = json.loads(output.read_text(encoding="utf-8"))
    assert fixture["evidence_class"] == "captured_fixture"
    assert fixture["capture"]["reviewer"] == "test-reviewer"


def test_capture_fixture_selects_scenarios_from_a_multi_run_artifact(
    tmp_path, capsys
):
    single = json.loads(
        _live_run_artifact(tmp_path).read_text(encoding="utf-8")
    )
    run_one = dict(single)
    run_one["scenario_id"] = "s1"
    run_two = dict(single)
    run_two["scenario_id"] = "s2"
    multi = {
        "kind": "agent-learning.run.v1",
        "live_lane_runs": [run_one, run_two],
    }
    artifact = tmp_path / "multi_run.json"
    artifact.write_text(json.dumps(multi, default=str), encoding="utf-8")
    output = tmp_path / "candidates" / "s2.fixture.json"

    # without --scenario: refuse, naming the choices on stderr
    exit_code = cli.main(
        ["simulate", "capture-fixture", str(artifact), "-o", str(output)]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--scenario" in captured.err
    assert "s1" in captured.err and "s2" in captured.err

    exit_code = cli.main(
        [
            "simulate",
            "capture-fixture",
            str(artifact),
            "--scenario",
            "s2",
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    assert output.exists()
