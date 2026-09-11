"""A2A live lane suite (3E) — opt-in, env-gated (guide §5.2).

Collected in every env, SKIPPED unless AGENT_LEARNING_LIVE_A2A=1 (the
conftest three-fact reason). The loopback peer pair needs the `a2a` extra.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.live_lane, pytest.mark.live_a2a]

_SCRIPTED_SCENARIO = {
    "name": "a2a-loopback-smoke",
    "stages": ["card_discovery", "task_lifecycle", "artifact_exchange"],
    "message": "ping from the lane suite",
}


def test_lane_refuses_without_env_flag(monkeypatch):
    from fi.alk.live import _contract, a2a_lane

    monkeypatch.delenv("AGENT_LEARNING_LIVE_A2A", raising=False)
    with pytest.raises(_contract.LaneDisabledError):
        a2a_lane.run_a2a_lane({"name": "smoke"})


def test_rung1_loopback_peer_protocol_stages(tmp_path):
    from fi.alk.live import _contract, a2a_lane

    result = a2a_lane.run_a2a_lane(
        _SCRIPTED_SCENARIO, repeats=2, artifacts_dir=tmp_path / "artifacts"
    )
    assert result["live_lane"]["evidence_class"] == "live_lane"
    assert result["live_lane"]["verdict"] in {"pass", "fail", "unstable", "void"}
    assert result["live_lane"]["repeats"] == 2
    assert result["live_lane"]["rung"] == "loopback_peers"
    for repeat in result["live_lane"]["per_repeat"]:
        assert repeat.get("failure_layer") in (None, *_contract.FAILURE_LAYERS)
    assert all(
        repeat["failure_layer"] != "lane_infra" or repeat.get("quarantined")
        for repeat in result["live_lane"]["per_repeat"]
    )
    assert result["protocol_trace"]["engine"] == "live_lane_a2a"
