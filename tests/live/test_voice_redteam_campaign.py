"""Voice escalation campaign runner tests (Phase 12, units 4/4b/4c).

Machinery tier (no extras, no flags) runs in the DEFAULT suite: arc-turn
compilation determinism + dial conditioning, timing-fidelity math, the rung
wall + flag refusal, authorization ordering, simulator hardening. Lane tier
(env-gated, auto-skip bare) runs one full clean+stressed campaign over the
livekit rung-1 lane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT_ROOT / "examples" / "voice_redteam"


def _scenario() -> dict:
    s = json.loads((FIXTURES / "scenarios/adversarial.json").read_text())
    return s


# --- machinery tier (no flags) ----------------------------------------------


def test_compile_arc_turns_determinism_and_dial_conditioning():
    from fi.alk.live import voice_redteam

    scenario = _scenario()
    turns_a = voice_redteam.compile_arc_turns(scenario)
    turns_b = voice_redteam.compile_arc_turns(scenario)
    assert turns_a == turns_b  # deterministic
    assert len(turns_a) == len(scenario["escalation"]["steps"])
    # rajas >= 0.7 (the attacker fixture) -> interruption-marked turns
    assert all(t["user"].startswith("--") for t in turns_a)
    # escalation_schedule overrides step pressure when present
    schedule = scenario["dataset"][0]["behavior_policy"]["escalation_schedule"]
    assert turns_a[0]["pressure"] == pytest.approx(schedule[0])


def test_timing_fidelity_is_a_labeled_rung1_proxy():
    from fi.alk.live import voice_redteam

    scenario = _scenario()
    turns = voice_redteam.compile_arc_turns(scenario)
    persona = scenario["dataset"][0]
    tf = voice_redteam.timing_fidelity(turns, persona, turns)
    assert tf["proxy"] == "timing_only"
    assert tf["rung"] == 1
    assert isinstance(tf["in_character_timing"], bool)


def test_simulator_hardening_voids_on_persona_jailbreak():
    from fi.alk.live import voice_redteam

    held = voice_redteam.simulator_hardening(
        [{"counter_pressure": True}, {"text": "ok"}]
    )
    assert held["simulator_held"] is True
    assert held["counter_pressure_probes"] == 1
    broken = voice_redteam.simulator_hardening(
        [{"counter_pressure": True, "persona_jailbroken": True}]
    )
    assert broken["simulator_held"] is False


def test_campaign_refuses_without_lane_flag(monkeypatch):
    from fi.alk.live import _contract, voice_redteam

    monkeypatch.delenv("AGENT_LEARNING_LIVE_LIVEKIT", raising=False)
    scenario = _scenario()
    with pytest.raises(_contract.LaneDisabledError):
        voice_redteam.run_voice_escalation_campaign(scenario, lane="livekit")


def test_campaign_rung_wall_and_authorization_ordering(monkeypatch):
    from fi.alk.live import voice_redteam

    scenario = _scenario()
    # acoustic operator raises at text rung (before any lane dispatch)
    with pytest.raises(ValueError):
        voice_redteam.run_voice_escalation_campaign(
            scenario, lane="livekit", operators=["noise"]
        )
    # a non-local target without the stanza refuses with the finding FIRST,
    # before LaneDisabledError could fire
    monkeypatch.delenv("AGENT_LEARNING_LIVE_LIVEKIT", raising=False)
    with pytest.raises(voice_redteam.VoiceAuthorizationError) as exc:
        voice_redteam.run_voice_escalation_campaign(
            scenario,
            lane="livekit",
            target={"kind": "live_lane", "lane": "livekit"},
            provider="custom",
        )
    assert exc.value.finding["type"] == "voice_target_authorization_missing"


# --- Phase 9A unit 3b: the honesty-pin UPGRADE (research-pin -> computed) ----
# These run flag-free by stubbing the lane runner with a deterministic payload
# (the lane dispatch itself is unit-2 tested; here we prove the rung-aware
# phone_survival / attack_rung flip in the campaign stanza).


def _stub_lane_runner(monkeypatch, *, channels=None):
    """Replace the campaign's lane runner with a deterministic stub so the
    rung-aware stanza logic can be tested without an env flag / framework."""
    from fi.alk.live import voice_redteam

    def runner(scenario, *, rung=1, repeats=4, stressed=False, perturbations=None,
               seed=0, required_env=None, artifacts_dir=None, **kw):
        payload = {
            "kind": "agent-learning.run.v1",
            "live_lane": {"run_id": f"stub{rung}{int(stressed)}"},
            "summary": {"verdict": "pass"},
            "realtime_trace": {"items": []},
            "evidence_class": "live_stressed" if (stressed or rung == 2) else "live_lane",
        }
        if channels is not None and rung == 2:
            payload["channels"] = channels
            payload["fidelity_tier"] = "deterministic_loopback"
        return payload

    monkeypatch.setattr(voice_redteam, "_resolve_lane_runner", lambda lane: runner)
    return voice_redteam


def test_rung1_campaign_keeps_research_pinned(monkeypatch):
    vr = _stub_lane_runner(monkeypatch)
    payload = vr.run_voice_escalation_campaign(
        _scenario(), lane="livekit", rung=1, seed=7, capture_candidates=False
    )
    assert payload["voice_redteam"]["phone_survival"] == {
        "status": "untested",
        "tier": "research_pinned",
    }
    assert payload["voice_redteam"]["attack_rung"] == "transcript_level"
    assert payload["attack_rung"] == "transcript_level"


def test_rung2_campaign_computes_phone_survival_and_flips_attack_rung(monkeypatch):
    computed = {
        "status": "partial",
        "tier": "channel_simulated",
        "reason": "codec=g711_ulaw ...",
        "pre_channel_success": 0.8,
        "post_channel_success": 0.5,
        "band_energy_lt_4khz": 0.9,
    }
    vr = _stub_lane_runner(
        monkeypatch,
        channels={
            "derived": {"ttfb_ms": 100.0},
            "source": "derive_channel_evidence",
            "rung": "loopback_transport",
            "fidelity_tier": "deterministic_loopback",
            "phone_survival": computed,
        },
    )
    payload = vr.run_voice_escalation_campaign(
        _scenario(), lane="livekit", rung=2, seed=7, capture_candidates=False
    )
    ps = payload["voice_redteam"]["phone_survival"]
    assert ps["tier"] == "channel_simulated"
    assert ps["status"] == "partial"
    assert "pre_channel_success" in ps  # the 3 computed-evidence fields ride
    # attack_rung flips to the canonical "acoustic" ONLY on the rung-2 record
    # (Phase-12 12C rung-2 reconciled 9A's interim "audio_level" → the
    # gate-pinned V1_VOICE_ATTACK_RUNGS token "acoustic").
    assert payload["voice_redteam"]["attack_rung"] == "acoustic"
    assert payload["attack_rung"] == "acoustic"
    # "acoustic" is in the canonical Phase-12 attack-rung vocabulary
    from fi.alk import trinity

    assert "acoustic" in trinity.V1_VOICE_ATTACK_RUNGS
    # the legacy 9A token is retained as a back-compat alias only
    assert vr.ATTACK_RUNG_AUDIO == "acoustic"
    # no rung-2 artifact carries evidence_class live_lane
    assert payload["evidence_class"] != "live_lane"


def test_rung1_acoustic_operator_raises_rung2_passes(monkeypatch):
    # Phase-12 12C rung-2: an acoustic operator raises at rung-1 (no audio
    # channel) but is accepted at rung-2 and forwarded to the lane runner.
    from fi.alk.live import voice_redteam

    scenario = _scenario()
    # rung-1: acoustic operator hits the campaign rung wall
    with pytest.raises(ValueError):
        voice_redteam.run_voice_escalation_campaign(
            scenario, lane="livekit", rung=1, operators=["noise"]
        )

    # rung-2: the acoustic operator flows through to the (stubbed) lane runner
    seen = {}

    def runner(scenario, *, rung=1, repeats=4, stressed=False, perturbations=None,
               seed=0, required_env=None, artifacts_dir=None, **kw):
        seen.setdefault(rung, []).append(list(perturbations or []))
        payload = {
            "kind": "agent-learning.run.v1",
            "live_lane": {"run_id": f"stub{rung}{int(stressed)}"},
            "summary": {"verdict": "pass"},
            "realtime_trace": {"items": []},
            "evidence_class": "live_stressed" if rung == 2 else "live_lane",
        }
        if rung == 2 and perturbations:
            payload["channels"] = {
                "derived": {"ttfb_ms": 100.0},
                "source": "derive_channel_evidence",
                "rung": "loopback_transport",
                "fidelity_tier": "deterministic_loopback",
                "acoustic_operators": [{"operator": "reverb_blend", "seed": seed}],
                "phone_survival": {
                    "status": "survives",
                    "tier": "channel_simulated",
                    "reason": "codec=g711_ulaw ...",
                    "pre_channel_success": 0.7,
                    "post_channel_success": 0.65,
                    "band_energy_lt_4khz": 0.95,
                },
            }
            payload["fidelity_tier"] = "deterministic_loopback"
        return payload

    monkeypatch.setattr(voice_redteam, "_resolve_lane_runner", lambda lane: runner)
    payload = voice_redteam.run_voice_escalation_campaign(
        scenario, lane="livekit", rung=2, operators=["reverb_blend"], seed=7,
        capture_candidates=False,
    )
    # the acoustic operator was forwarded to BOTH the clean and stressed lane runs
    assert any("reverb_blend" in ops for ops in seen.get(2, []))
    # the campaign earned the computed phone_survival + flipped attack_rung
    assert payload["voice_redteam"]["phone_survival"]["tier"] == "channel_simulated"
    assert payload["voice_redteam"]["attack_rung"] == "acoustic"
    assert payload["attack_rung"] == "acoustic"


# --- lane tier (env-gated, auto-skip bare) ----------------------------------

pytestmark = []


@pytest.mark.live_lane
@pytest.mark.live_livekit
def test_full_clean_stressed_campaign_over_livekit_rung1(tmp_path):
    from fi.alk.live import voice_redteam

    scenario = _scenario()
    scenario["responses"] = ["sure", "confirmed", "done", "ok"]
    payload = voice_redteam.run_voice_escalation_campaign(
        scenario,
        lane="livekit",
        operators=["homophone", "near_dup"],
        seed=7,
        repeats=4,
        artifacts_dir=tmp_path,
    )
    vr = payload["voice_redteam"]
    assert payload["attack_rung"] == "transcript_level"
    assert vr["attack_rung"] == "transcript_level"
    assert vr["phone_survival"] == {"status": "untested", "tier": "research_pinned"}
    assert payload["evidence_class"] == "live_stressed"
    # the stressed run's paired_clean_run is filled with the clean run id
    perturbations = payload["live_lane"]["perturbations"]
    assert perturbations["paired_clean_run"] == vr["paired"]["clean_run"]
    assert perturbations["paired_clean_run"] is not None
