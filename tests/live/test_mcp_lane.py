"""MCP live lane suite (3E) — opt-in, env-gated (guide §5.2).

Collected in every env, SKIPPED unless AGENT_LEARNING_LIVE_MCP=1 (the
conftest three-fact reason). The loopback server fixture is credential-free
but still needs the `mcp` extra (a real FastMCP process + ClientSession over
stdio — that IS the live graduation, P3-D6), so every running test here is
marked accordingly via the module-level lane markers.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

pytestmark = [pytest.mark.live_lane, pytest.mark.live_mcp]

_SCRIPTED_SCENARIO = {
    "name": "mcp-loopback-smoke",
    "calls": [
        {
            "tool": "echo",
            "arguments": {"text": "hello loopback"},
            "expect": {"contains": "hello loopback"},
        },
        {
            "tool": "add",
            "arguments": {"a": 2, "b": 3},
            "expect": {"contains": "5"},
        },
    ],
}


def test_lane_refuses_without_env_flag(monkeypatch):
    from fi.alk.live import _contract, mcp_lane

    monkeypatch.delenv("AGENT_LEARNING_LIVE_MCP", raising=False)
    with pytest.raises(_contract.LaneDisabledError):
        mcp_lane.run_mcp_lane({"name": "smoke"})


def test_rung1_loopback_server_repeats_and_snapshot(tmp_path):
    from fi.alk.live import _contract, mcp_lane

    result = mcp_lane.run_mcp_lane(
        _SCRIPTED_SCENARIO, repeats=2, artifacts_dir=tmp_path / "artifacts"
    )
    assert result["live_lane"]["evidence_class"] == "live_lane"
    assert result["live_lane"]["verdict"] in {"pass", "fail", "unstable", "void"}
    assert result["live_lane"]["repeats"] == 2
    assert result["live_lane"]["rung"] == "loopback_servers"
    for repeat in result["live_lane"]["per_repeat"]:
        assert repeat.get("failure_layer") in (None, *_contract.FAILURE_LAYERS)
    assert all(
        repeat["failure_layer"] != "lane_infra" or repeat.get("quarantined")
        for repeat in result["live_lane"]["per_repeat"]
    )
    if result["live_lane"]["verdict"] == "pass":
        # server-behavior snapshot stamp (R§1 #11)
        snapshot = result["live_lane"].get("server_snapshot")
        assert snapshot and snapshot.get("capability_hash")


def test_captured_fixture_round_trip_from_loopback_run(tmp_path):
    """live loopback run -> capture candidate -> simulated review ->
    replay_fixture green (guide §5.4 pattern, needs the mcp extra)."""

    from fi.alk.live import _capture, _stats, mcp_lane

    result = mcp_lane.run_mcp_lane(
        _SCRIPTED_SCENARIO, repeats=2, artifacts_dir=tmp_path / "artifacts"
    )
    assert result["live_lane"]["verdict"] == "pass"

    fields = {field.name for field in dataclasses.fields(_stats.LaneRunResult)}
    lane_result = _stats.LaneRunResult(
        **{
            key: value
            for key, value in result["live_lane"].items()
            if key in fields
        }
    )
    candidate = _capture.capture_to_fixture(
        lane_result,
        output=tmp_path / "candidates" / "mcp.fixture.json",
        scenario=result.get("scenario"),
    )
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    assert payload["evidence_class"] == "live_lane"  # source class kept
    assert payload["capture"]["reviewed"] is False

    payload["evidence_class"] = "captured_fixture"
    payload["capture"]["reviewed"] = True
    payload["capture"]["reviewer"] = "test-reviewer"
    reviewed_copy = tmp_path / "reviewed" / "mcp.fixture.json"
    reviewed_copy.parent.mkdir(parents=True)
    reviewed_copy.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    replay = _capture.replay_fixture(reviewed_copy)
    assert replay["verdict"] == "pass"
    assert replay["evidence_class"] == "captured_fixture"
