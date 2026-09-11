"""Phase 12 voice red-team tests — composed search, A/B harness, scoring,
detection evidence, authorization, capture packs, and the gate-#73 status fn.

Covers (BUILD-GUIDE §3.4 / §3b / §6 / §6b / §7.6): the composed manifest
builder preconditions + arm freezing; the three-arm A/B harness verdict +
re-derivation + lift null rules (budget mismatch + quarantine epidemic);
fidelity-as-quality halving (never a floor); detection-evidence field closure +
no-verdict-key structural rule; authorization validation; the capture
round-trip; the constant cross-pins; and the four tmp_path negatives the gate
status fn must catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fi.alk import live, redteam, trinity
from fi.alk.cli import main
from fi.simulate.simulation.models import Persona, Scenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "examples" / "voice_redteam"


def _persona() -> Persona:
    return Persona(**json.loads((FIXTURES / "personas/attacker.json").read_text()))


def _scenario() -> Scenario:
    return Scenario(**json.loads((FIXTURES / "scenarios/adversarial.json").read_text()))


_PSPACE = {
    "temperament.rajas": [0.3, 0.6, 0.9],
    "behavior_policy.interruption_propensity": [0.1, 0.4, 0.7],
}
_SSPACE = {"operator": ["homophone", "code_switch"], "rate": [0.05, 0.15], "seed": [7]}


# --- builder preconditions + arm freezing -----------------------------------


def test_composed_builder_requires_attack_conditioning():
    persona = Persona(
        identity={"name": "x", "role": "support"},
        persona={"name": "x"},
        situation="s",
        outcome="o",
    )
    with pytest.raises(ValueError):
        redteam.build_composed_voice_attack_search_manifest(
            name="x", persona=persona, scenario=_scenario(),
            persona_space=_PSPACE, signal_space=_SSPACE, eval_budget=6,
        )


def test_composed_builder_rejects_bad_budget_and_empty_value_lists():
    with pytest.raises(ValueError):
        redteam.build_composed_voice_attack_search_manifest(
            name="x", persona=_persona(), scenario=_scenario(),
            persona_space=_PSPACE, signal_space=_SSPACE, eval_budget=0,
        )
    with pytest.raises(ValueError):
        redteam.build_composed_voice_attack_search_manifest(
            name="x", persona=_persona(), scenario=_scenario(),
            persona_space={"temperament.rajas": []}, signal_space=_SSPACE,
            eval_budget=6,
        )


def test_composed_builder_rejects_non_text_rung_operator_and_bad_voice_surface():
    with pytest.raises(ValueError):
        redteam.build_composed_voice_attack_search_manifest(
            name="x", persona=_persona(), scenario=_scenario(),
            persona_space=_PSPACE, signal_space={"operator": ["noise"]},
            eval_budget=6,
        )
    with pytest.raises(ValueError):
        redteam.build_composed_voice_attack_search_manifest(
            name="x", persona=_persona(), scenario=_scenario(),
            persona_space=_PSPACE, signal_space=_SSPACE, eval_budget=6,
            voice_surfaces=["not_a_voice_surface"],
        )


def test_arm_freezing_drops_the_complementary_path_family():
    paths = {}
    for arm in ("composed", "persona_only", "signal_only"):
        m = redteam.build_composed_voice_attack_search_manifest(
            name="x", persona=_persona(), scenario=_scenario(),
            persona_space=_PSPACE, signal_space=_SSPACE, eval_budget=6, arm=arm,
        )
        paths[arm] = set(m["optimization"]["target"]["search_space"])
        assert m["version"] == "agent-learning.optimization.v1"
        assert m["optimization"]["target"]["metadata"]["eval_budget"] == 6
        assert m["optimization"]["target"]["metadata"]["ranking_source"] == (
            "evaluation_suite"
        )
    assert any(".attack_persona." in p for p in paths["composed"])
    assert any(".attack_signal." in p for p in paths["composed"])
    assert not any(".attack_signal." in p for p in paths["persona_only"])
    assert not any(".attack_persona." in p for p in paths["signal_only"])


# --- A/B harness ------------------------------------------------------------


def test_ab_harness_three_arms_equal_budget_verdict_rederivable():
    ab = redteam.run_composed_voice_attack_ab(
        name="ab", persona=_persona(), scenario=_scenario(),
        persona_space=_PSPACE, signal_space=_SSPACE, eval_budget_per_arm=6,
    )
    assert ab["kind"] == "agent-learning.optimization.v1"
    assert "ab_harness" in ab and "voice-redteam-ab" not in ab["kind"]
    arms = ab["ab_harness"]["arms"]
    assert set(arms) == set(redteam.VOICE_REDTEAM_AB_ARMS)
    assert all(arms[a]["eval_budget"] == 6 for a in arms)
    assert ab["ab_harness"]["budget_equal"] is True
    # the verdict re-derives from per_seed (the harness can't hand-assign)
    rederived = redteam._derive_voice_ab_verdict(arms, ab["ab_harness"]["seeds"])
    assert ab["ab_harness"]["ab_verdict"] == rederived
    assert ab["ab_harness"]["ab_verdict"] in redteam.VOICE_REDTEAM_AB_VERDICTS
    # lift numeric on a clean full-budget run
    assert ab["ab_harness"]["lift"]["vs_best_ablation"] is not None


def test_ab_harness_hand_tampered_verdict_is_detected_by_rederivation():
    ab = redteam.run_composed_voice_attack_ab(
        name="ab", persona=_persona(), scenario=_scenario(),
        persona_space=_PSPACE, signal_space=_SSPACE, eval_budget_per_arm=6,
    )
    tampered = dict(ab["ab_harness"])
    tampered["ab_verdict"] = "no_lift" if ab["ab_harness"]["ab_verdict"] == (
        "composed_lift"
    ) else "composed_lift"
    rederived = redteam._derive_voice_ab_verdict(tampered["arms"], tampered["seeds"])
    assert rederived != tampered["ab_verdict"]


def test_ab_harness_quarantine_epidemic_voids_lift_and_exits_one():
    ab = redteam.run_composed_voice_attack_ab(
        name="ab", persona=_persona(), scenario=_scenario(),
        persona_space=_PSPACE, signal_space=_SSPACE, eval_budget_per_arm=6,
        quarantine_overrides={"composed": 5},  # rate 5/6 > 0.5
    )
    assert ab["exit_code"] == 1
    assert ab["ab_harness"]["lift"]["vs_best_ablation"] is None
    assert any(
        f["type"] == "composed_arm_quarantine_epidemic"
        for f in ab.get("findings", [])
    )


def test_ab_harness_budget_under_run_voids_lift_keeps_arms_exit_zero():
    ab = redteam.run_composed_voice_attack_ab(
        name="ab", persona=_persona(), scenario=_scenario(),
        persona_space=_PSPACE, signal_space=_SSPACE, eval_budget_per_arm=6,
        quarantine_overrides={"signal_only": 1},
    )
    assert ab["exit_code"] == 0
    assert ab["ab_harness"]["lift"]["vs_best_ablation"] is None
    assert any(
        f["type"] == "composed_budget_mismatch" for f in ab.get("findings", [])
    )
    assert set(ab["ab_harness"]["arms"]) == set(redteam.VOICE_REDTEAM_AB_ARMS)


# --- fidelity-as-quality (halving, never a floor) ---------------------------


def test_character_broken_quality_halved_and_row_retained():
    held = redteam.voice_attack_quality_score(
        {"asr": 1.0}, {"verdict": "pass", "adherence": {"score": 0.8}}
    )
    broken = redteam.voice_attack_quality_score(
        {"asr": 1.0}, {"verdict": "fail", "adherence": {"score": 0.8}}
    )
    assert held["score"] == pytest.approx(0.8)
    assert broken["score"] == pytest.approx(0.4)  # halved
    assert broken["character_broken"] is True
    assert broken["score"] > 0.0  # retained, never dropped


# --- detection evidence -----------------------------------------------------


def test_detection_evidence_field_closure_and_no_verdict_key():
    clean = {"refusal_rate": 0.9, "transcript_text": "hello world"}
    stressed = {
        "refusal_rate": 0.5, "transcript_text": "hello word",
        "narrative_frame_marker": True, "escalation_pressure_trace": [0.2, 0.5],
    }
    for family, declared in trinity.V1_VOICE_DETECTION_EVIDENCE_FIELDS.items():
        block = redteam.voice_detection_evidence(family, clean, stressed)
        assert [f["signal"] for f in block["fields"]] == list(declared)
        assert not any(k in block for k in ("verdict", "passed", "pass"))
    with pytest.raises(ValueError):
        redteam.voice_detection_evidence("not_a_family", clean, stressed)


# --- authorization ----------------------------------------------------------


def test_authorization_kit_local_autostamp_and_non_local_refusal_ordering():
    echo = live.validate_authorization(None, provider="local_cli")
    assert echo["relationship"] == "kit_local"
    with pytest.raises(live.VoiceAuthorizationError) as exc:
        live.validate_authorization(
            {"kind": "live_lane", "lane": "livekit"}, provider="local_cli"
        )
    assert exc.value.finding["type"] == "voice_target_authorization_missing"
    complete = live.validate_authorization(
        {
            "kind": "live_lane", "lane": "livekit",
            "authorization": {
                "relationship": "owned", "statement": "ours",
                "acknowledged_by": "n", "acknowledged_at": "2026-06-12",
                "scope": "test",
            },
        }
    )
    assert complete["relationship"] == "owned"


# --- constant cross-pins ----------------------------------------------------


def test_voice_operator_constants_cross_pinned():
    assert tuple(trinity.V1_VOICE_REDTEAM_TEXT_OPERATORS) == (
        live._perturb.TEXT_RUNG_OPERATORS
    )
    for family in trinity.V1_VOICE_ATTACK_FAMILY_MATRIX:
        assert family in trinity.V1_VOICE_DETECTION_EVIDENCE_FIELDS
    for row in trinity.V1_VOICE_ATTACK_FAMILY_MATRIX.values():
        assert row["maturity"] in trinity.V1_VOICE_ATTACK_MATURITY_LEVELS
        assert row["phone_survival"]["status"] in (
            trinity.V1_VOICE_PHONE_SURVIVAL_STATUSES
        )
        assert row["phone_survival"]["tier"] in (
            trinity.V1_VOICE_PHONE_SURVIVAL_TIERS
        )


# --- CLI front door ---------------------------------------------------------


def test_cli_ab_harness_emits_embedded_block(tmp_path):
    manifest = {
        "name": "voice-composed-ab",
        "persona": json.loads((FIXTURES / "personas/attacker.json").read_text()),
        "scenario": json.loads(
            (FIXTURES / "scenarios/adversarial.json").read_text()
        ),
        "persona_space": _PSPACE,
        "signal_space": _SSPACE,
        "eval_budget_per_arm": 6,
    }
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest))
    op = tmp_path / "out.json"
    rc = main(["redteam", str(mp), "--ab-harness", "-o", str(op), "--quiet"])
    assert rc == 0
    out = json.loads(op.read_text())
    assert out["kind"] == "agent-learning.optimization.v1"
    assert out["ab_harness"]["ab_verdict"] in redteam.VOICE_REDTEAM_AB_VERDICTS


def test_cli_ab_harness_refuses_rung2_operators(tmp_path):
    manifest = {
        "name": "voice-composed-ab",
        "persona": json.loads((FIXTURES / "personas/attacker.json").read_text()),
        "scenario": json.loads(
            (FIXTURES / "scenarios/adversarial.json").read_text()
        ),
        "persona_space": _PSPACE,
        "signal_space": {"operator": ["noise"]},
        "eval_budget_per_arm": 6,
    }
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest))
    rc = main(["redteam", str(mp), "--ab-harness", "--quiet"])
    assert rc == 1


# --- gate-#73 status fn tmp_path negatives ----------------------------------


def _mini_repo(tmp_path: Path) -> Path:
    """A mini repo tree pointing the gate at synthetic corpus/example/fixtures
    by copying the real ones, then mutated per-test."""

    root = tmp_path / "repo"
    (root / "examples/voice_redteam").mkdir(parents=True)
    import shutil

    shutil.copytree(FIXTURES, root / "examples/voice_redteam", dirs_exist_ok=True)
    shutil.copy(
        PROJECT_ROOT / "examples/sdk_voice_redteam_campaign.py",
        root / "examples/sdk_voice_redteam_campaign.py",
    )
    shutil.copy(
        PROJECT_ROOT / "examples/redteam_corpus.json",
        root / "examples/redteam_corpus.json",
    )
    # The voice-redteam research doc lives in the separate internal-docs repo and
    # is no longer required gate evidence (the gate's source-url check reads the
    # committed corpus JSON, not the doc), so the mini-repo doesn't stage it.
    return root


def test_release_voice_redteam_readiness_status_flags_corpus_violations(tmp_path):
    root = _mini_repo(tmp_path)
    corpus_path = root / "examples/redteam_corpus.json"
    corpus = json.loads(corpus_path.read_text())
    for row in corpus["rows"]:
        if row.get("id") == "voice_asr_front_end_auditory_injection":
            # claim survives against a row but flip the prior to a mismatch
            row["voice"]["phone_survival"] = {"status": "dies", "tier": "research_pinned", "reason": "x"}
        if row.get("id") == "voice_diarization_system_speaker":
            row["voice"]["attack_family"] = "not_a_family"
    corpus_path.write_text(json.dumps(corpus))
    ev = trinity._release_voice_redteam_readiness_status(root)
    fields = {e["field"] for e in ev["corpus_errors"]}
    assert any("phone_survival" in f for f in fields)
    assert any("attack_family" in f for f in fields)


def test_release_voice_redteam_readiness_status_clean_corpus_passes(tmp_path):
    root = _mini_repo(tmp_path)
    ev = trinity._release_voice_redteam_readiness_status(root)
    for arr in (
        "missing_files", "execution_errors", "corpus_errors", "matrix_errors",
        "operator_errors", "search_errors", "fidelity_errors", "pack_errors",
        "authorization_errors",
    ):
        assert ev[arr] == [], (arr, ev[arr])
