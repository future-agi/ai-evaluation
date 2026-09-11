"""Phase 9A units 4 + 6 — the voice improvement loop + the voice_loopback gate.

No extras, no env flags, no network. Unit 4 (the multi-objective voice loss +
the Goodhart guard reuse + the V1_VOICE_FAILURE_SUBLAYERS attribution + the
whole-agent voice search space). Unit 6 (the ``voice_loopback_readiness`` status
function on tmp_path mini-repo trees, incl. the constructed-negative tripwires:
fidelity overclaim, phone-survival violation, single-timing loss, channels at
rung-1, non-determinism).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fi.alk import trinity, voice_loop

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- a valid declared voice objective (guards populated) ---------------------

def _voice_objective(*, terms=None) -> dict:
    terms = terms or [
        {"eval": "task_success", "weight": 1.0, "direction": "maximize"},
        {"eval": "barge_in_latency", "weight": 0.5, "direction": "minimize"},
        {"eval": "ttfb", "weight": 0.5, "direction": "minimize"},
        {"eval": "codec_survival", "weight": 0.8, "direction": "maximize"},
    ]
    return {
        "source": "declared",
        "evals": terms,
        "guards": {
            "sentinel_rows": [{"id": "no_pii_leak"}],
            "canary_evals": [{"eval": "repetition_canary"}],
            "min_guard_count": 1,
        },
    }


def _voice_search_space() -> dict:
    # the §4.5 whole-agent families — NOT prompt-only.
    return {
        "voice.id": ["alloy", "shimmer"],
        "voice.tts.rate": [0.9, 1.0, 1.1],
        "agent.first_message": ["Hi, how can I help?", "Thanks for calling."],
        "voice.endpointing.threshold": [200, 400],
        "voice.barge_in.policy": ["eager", "polite"],
        "agent.instructions": ["Be concise.", "Confirm every value."],
    }


# --- unit 4: the multi-objective voice loss ---------------------------------

def test_voice_loss_multi_objective_compiles():
    compiled = voice_loop.compile_voice_objective(_voice_objective())
    refs = [t["eval"] for t in compiled["evals"]]
    assert len(refs) >= 2
    assert any(r in voice_loop.V1_VOICE_LOSS_NON_TIMING_QUALITY_TERMS for r in refs)
    # guard block survived the compile
    assert compiled["guards"]["min_guard_count"] >= 1


def test_voice_loss_single_timing_term_rejected():
    # a single-timing-term objective is the 9A-A4 structural rejection
    single = _voice_objective(
        terms=[{"eval": "barge_in_latency", "weight": 1.0, "direction": "minimize"}]
    )
    with pytest.raises(voice_loop.VoiceLossCompositionError):
        voice_loop.compile_voice_objective(single)


def test_voice_loss_timing_only_multi_term_rejected():
    # >= 2 terms but ALL timing → still rejected (no non-timing quality anchor)
    timing_only = _voice_objective(
        terms=[
            {"eval": "barge_in_latency", "weight": 1.0, "direction": "minimize"},
            {"eval": "ttfb", "weight": 1.0, "direction": "minimize"},
        ]
    )
    with pytest.raises(voice_loop.VoiceLossCompositionError):
        voice_loop.compile_voice_objective(timing_only)


def test_voice_loss_guard_unconditional():
    # a valid multi-objective composition WITHOUT guards still raises (the
    # unedited loss.py:106-116 — "There is no override.")
    from fi.alk.loss import ObjectiveError

    no_guards = _voice_objective()
    no_guards.pop("guards")
    with pytest.raises(ObjectiveError):
        voice_loop.compile_voice_objective(no_guards)


def test_voice_sublayer_attribution_closed_set():
    # selectivity / endpointing weak → tts_endpointing (NOT llm)
    assert voice_loop.attribute_voice_sublayer(
        failure_layer="agent_behavior", signal="selectivity weak"
    ) == "tts_endpointing"
    # mis-heard the amount under clean audio → asr_mishear
    assert voice_loop.attribute_voice_sublayer(
        failure_layer="agent_behavior", signal="tool_argument mishear"
    ) == "asr_mishear"
    # claim died through the codec → acoustic_codec
    assert voice_loop.attribute_voice_sublayer(
        failure_layer="provider", signal="codec_survival died"
    ) == "acoustic_codec"
    # reasoning/policy default → llm
    assert voice_loop.attribute_voice_sublayer(
        failure_layer="agent_behavior", signal="wrong policy choice"
    ) == "llm"
    # every output is in the closed set
    for sig in ("selectivity", "codec", "asr", "policy", ""):
        out = voice_loop.attribute_voice_sublayer(failure_layer="agent_behavior", signal=sig)
        assert out in voice_loop.V1_VOICE_FAILURE_SUBLAYERS


def test_voice_search_space_whole_agent():
    manifest = voice_loop.build_voice_practice_loop_manifest(
        name="voice-loop-demo",
        base_agent={"model": "gpt-4o", "voice": {"id": "alloy"}},
        search_space=_voice_search_space(),
        objective=_voice_objective(),
        eval_budget=4,
        seed=7,
    )
    practice = manifest["practice"]
    # world.kind made executable for the loop substrate
    assert practice["simulation"]["inline"]["world"]["kind"] == "voice_telephony"
    paths = set(practice["search_space"])
    # NOT prompt-only: voice/TTS/endpointing families are present
    assert "voice.id" in paths
    assert "voice.tts.rate" in paths
    assert "voice.endpointing.threshold" in paths
    assert "agent.instructions" in paths
    # base_agent + search_space resolve against the emitted manifest
    assert practice["base_agent"]["voice"]["id"] == "alloy"


def test_voice_loop_ab_equal_budget():
    # the loop-vs-no-loop A/B compiles both arms at equal eval_budget (the
    # _experiment.py contract reused — no new harness).
    budget = 4
    on = voice_loop.build_voice_practice_loop_manifest(
        name="ab-loop-on",
        base_agent={"model": "gpt-4o"},
        search_space=_voice_search_space(),
        objective=_voice_objective(),
        eval_budget=budget,
        seed=7,
    )
    off = voice_loop.build_voice_practice_loop_manifest(
        name="ab-loop-off",
        base_agent={"model": "gpt-4o"},
        search_space=_voice_search_space(),
        objective=_voice_objective(),
        eval_budget=budget,
        seed=7,
    )
    assert on["practice"]["eval_budget"] == off["practice"]["eval_budget"] == budget


# --- unit 6: the voice_loopback_readiness status fn (tmp_path mini-repos) ----
# Build a minimal repo tree (the real fixtures + a doctored example) and run the
# status fn directly. See test_config_and_facades for the full release-check.

def _mini_repo(tmp_path: Path) -> Path:
    """Copy the real Phase-9A example files + fixture dir into a tmp repo so the
    status fn exec-loads them; the kit's installed packages resolve normally."""
    (tmp_path / "examples").mkdir(parents=True, exist_ok=True)
    for rel in trinity.V1_VOICE_LOOPBACK_FILES:
        shutil.copy(PROJECT_ROOT / rel, tmp_path / rel)
    shutil.copytree(
        PROJECT_ROOT / trinity.V1_VOICE_LOOPBACK_GATE_FIXTURE_DIR,
        tmp_path / trinity.V1_VOICE_LOOPBACK_GATE_FIXTURE_DIR,
    )
    return tmp_path


def test_release_voice_loopback_readiness_status_clean(tmp_path):
    root = _mini_repo(tmp_path)
    status = trinity._release_voice_loopback_readiness_status(root)
    assert status["kind"] == "agent-learning.voice-loopback-readiness.v1"
    for arr in (
        "missing_files",
        "loopback_determinism_errors",
        "codec_roundtrip_errors",
        "metrics_wiring_errors",
        "voice_loss_errors",
        "evidence_class_errors",
        "phone_survival_errors",
        "rung_honesty_errors",
    ):
        assert status[arr] == [], f"{arr}: {status[arr]}"


def _doctor_loopback_example(root: Path, *, block: str, replacement: str) -> None:
    """Patch the copied loopback example's evidence by string-replacing a return
    snippet — the gate exec-loads the doctored copy."""
    path = root / "examples/sdk_voice_loopback.py"
    text = path.read_text(encoding="utf-8")
    assert block in text, f"block not found: {block[:60]}"
    path.write_text(text.replace(block, replacement), encoding="utf-8")


def test_voice_loopback_flags_fidelity_overclaim(tmp_path):
    # a rung-2 artifact stamping evidence_class: live_lane -> the
    # loopback_fidelity_overclaim token in evidence_class_errors.
    root = _mini_repo(tmp_path)
    _doctor_loopback_example(
        root,
        block='"evidence_class": "live_stressed",  # NEVER live_lane (default codec ON)',
        replacement='"evidence_class": "live_lane",  # DOCTORED overclaim',
    )
    status = trinity._release_voice_loopback_readiness_status(root)
    reasons = " ".join(
        str(e.get("reason", "")) for e in status["evidence_class_errors"]
    )
    assert "loopback_fidelity_overclaim" in reasons
    assert status["evidence_class_errors"]


def test_voice_loopback_flags_phone_survival_violation(tmp_path):
    # a status: survives claim with tier: research_pinned (no channel record).
    root = _mini_repo(tmp_path)
    _doctor_loopback_example(
        root,
        block='"phone_survival": lk_channels["phone_survival"],',
        replacement='"phone_survival": {"status": "survives", "tier": "research_pinned"},',
    )
    status = trinity._release_voice_loopback_readiness_status(root)
    assert status["phone_survival_errors"]


def test_voice_loopback_flags_single_timing_loss(tmp_path):
    # break the improvement example's single-timing rejection signal.
    root = _mini_repo(tmp_path)
    path = root / "examples/sdk_voice_improvement.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '        "single_timing_rejected": single_timing_rejected,',
        '        "single_timing_rejected": False,  # DOCTORED',
    )
    path.write_text(text, encoding="utf-8")
    status = trinity._release_voice_loopback_readiness_status(root)
    assert status["voice_loss_errors"]


def test_voice_loopback_flags_channels_at_rung1(tmp_path):
    # a rung-1 artifact carrying a channels block -> metrics_wiring_errors.
    root = _mini_repo(tmp_path)
    _doctor_loopback_example(
        root,
        block='"has_channels_block": False,  # rung-1 NEVER emits channels',
        replacement='"has_channels_block": True,  # DOCTORED honesty violation',
    )
    status = trinity._release_voice_loopback_readiness_status(root)
    assert status["metrics_wiring_errors"]


def test_voice_loopback_flags_nondeterminism(tmp_path):
    # break the determinism signal -> loopback_determinism_errors.
    root = _mini_repo(tmp_path)
    _doctor_loopback_example(
        root,
        block='"user_pcm_byte_identical": np.array_equal(a["user_pcm"], b["user_pcm"]),',
        replacement='"user_pcm_byte_identical": False,  # DOCTORED nondeterminism',
    )
    status = trinity._release_voice_loopback_readiness_status(root)
    assert status["loopback_determinism_errors"]


def test_voice_loopback_flags_missing_fixture(tmp_path):
    root = _mini_repo(tmp_path)
    (root / "examples/voice_loopback_fixture/user_turns/turn_1.wav").unlink()
    status = trinity._release_voice_loopback_readiness_status(root)
    assert status["missing_files"]
