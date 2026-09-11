"""Pipecat live lane suite (3C) — opt-in, env-gated (guide §5.2).

Collected in every env, SKIPPED unless AGENT_LEARNING_LIVE_PIPECAT=1 (the
conftest three-fact reason). Running for real needs the `pipecat` extra.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.live_lane, pytest.mark.live_pipecat]

_SCRIPTED_SCENARIO = {
    "name": "pipecat-rung1-smoke",
    "turns": [
        {"user": "Hello there."},
        {"user": "What can you help me with today?"},
    ],
    "responses": [
        "Hi! I am the scripted pipeline responder.",
        "I can run your frames through a real Pipecat pipeline.",
    ],
}


def test_lane_refuses_without_env_flag(monkeypatch):
    from fi.alk.live import _contract, pipecat_lane

    monkeypatch.delenv("AGENT_LEARNING_LIVE_PIPECAT", raising=False)
    with pytest.raises(_contract.LaneDisabledError):
        pipecat_lane.run_pipecat_lane(None, {"name": "smoke"})


def test_rung1_frame_injection_repeats_and_attributes():
    from fi.alk.live import _contract, pipecat_lane

    result = pipecat_lane.run_pipecat_lane(
        None, _SCRIPTED_SCENARIO, rung=1, repeats=2
    )
    assert result["live_lane"]["evidence_class"] == "live_lane"
    assert result["live_lane"]["verdict"] in {"pass", "fail", "unstable", "void"}
    assert result["live_lane"]["repeats"] == 2
    for repeat in result["live_lane"]["per_repeat"]:
        assert repeat.get("failure_layer") in (None, *_contract.FAILURE_LAYERS)
    assert all(
        repeat["failure_layer"] != "lane_infra" or repeat.get("quarantined")
        for repeat in result["live_lane"]["per_repeat"]
    )
    # rung-1 honesty rule: timing evidence only, NO channels block
    assert result["live_lane"]["rung"] == "frame_injection"
    assert "channels" not in result["live_lane"]
    assert "voice_timing" in result["metadata"]
