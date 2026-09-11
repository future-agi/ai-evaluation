"""LiveKit live lane suite (3B) — opt-in, env-gated (guide §5.2/§5.3).

Collected in every env, SKIPPED unless AGENT_LEARNING_LIVE_LIVEKIT=1 (the
conftest three-fact reason). Running for real needs the `livekit` extra.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.live_lane, pytest.mark.live_livekit]

_SCRIPTED_SCENARIO = {
    "name": "livekit-rung1-smoke",
    "turns": [
        {"user": "Hello, can you hear me?"},
        {"user": "Great - please confirm my appointment for tomorrow."},
    ],
    "responses": [
        "Yes, I can hear you loud and clear.",
        "Your appointment for tomorrow is confirmed.",
    ],
    "expect": {"contains": "confirmed"},
}


def test_lane_refuses_without_env_flag(monkeypatch):
    # The dynamic half of the env-flag discipline (gate checks the static half).
    # This test itself is skipped unless the flag is set, so flip it OFF inside:
    from fi.alk.live import _contract, livekit_lane

    monkeypatch.delenv("AGENT_LEARNING_LIVE_LIVEKIT", raising=False)
    with pytest.raises(_contract.LaneDisabledError):
        livekit_lane.run_livekit_lane({"name": "smoke"})


def test_rung1_virtual_clock_session_repeats_and_attributes():
    from fi.alk.live import _contract, livekit_lane

    result = livekit_lane.run_livekit_lane(_SCRIPTED_SCENARIO, rung=1, repeats=8)
    assert result["live_lane"]["evidence_class"] == "live_lane"
    assert result["live_lane"]["verdict"] in {"pass", "fail", "unstable", "void"}
    assert result["live_lane"]["repeats"] == 8
    if result["live_lane"]["verdict"] != "void":
        assert result["live_lane"]["icc"] is not None
    for repeat in result["live_lane"]["per_repeat"]:
        assert repeat.get("failure_layer") in (None, *_contract.FAILURE_LAYERS)
    # layer-attribution honesty: lane_infra rows are quarantined, never scored
    assert all(
        repeat["failure_layer"] != "lane_infra" or repeat.get("quarantined")
        for repeat in result["live_lane"]["per_repeat"]
    )
    # rung-1 honesty rule: timing-only voice metrics, NO channels block
    assert result["live_lane"]["rung"] == "virtual_clock"
    assert "channels" not in result["live_lane"]
    assert "voice_timing" in result["metadata"]


@pytest.mark.live_credentialed
def test_rung3_livekit_cloud_session():
    import os

    required = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")  # P3-D5 names
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip(f"credentialed rung needs: {', '.join(missing)}")
    from fi.alk.live import livekit_lane

    try:
        result = livekit_lane.run_livekit_lane(
            _SCRIPTED_SCENARIO, rung=3, repeats=2, required_env=required
        )
    except NotImplementedError as exc:
        pytest.skip(f"rung 3 (cloud_sip) not implemented in this build: {exc}")
    assert result["live_lane"]["verdict"] in {"pass", "unstable"}
    assert result["live_lane"]["required_env"] == list(required)  # names only
