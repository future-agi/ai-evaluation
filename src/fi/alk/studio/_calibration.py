"""Persona authoring + calibration lifecycle (Phase 7, unit 4).

Lifecycle = ``sampled -> validated -> interrogated -> admitted`` (ARCH §4
canon). Interrogation is the PICon battery — probe legs ``internal`` /
``external`` / ``retest`` — run against a SCRIPTED deterministic responder
(no LLM, no keys, no network). The retest leg is replay-based: the identical
battery re-runs ``repeats`` times under the same seed and the realization
vectors must agree fork-free (``_probe_divergence_step`` re-implements the
``live/_stats.divergence_step`` SEMANTIC — the live package is never imported
here, per the live_lane_boundary rule).

Emits ``agent-learning.persona-calibration.v1`` artifacts; bias-lint results
ride INSIDE them as a ``bias_lint`` block (never a standalone kind).
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from fi.simulate.simulation.behavior_policy import (
    BEHAVIOR_POLICY_AXIS_FIELDS,
    PERSONA_BEHAVIOR_AXES,
    PERSONA_BEHAVIOR_REALIZATION_METRICS,
    compile_behavior_policy,
)
from fi.simulate.simulation.models import (
    AttackConditioning,
    BehaviorPolicy,
    Persona,
    PersonaFact,
    PersonaIdentity,
    PersonaProvenance,
    PersonaTemperament,
    Scenario,
)

from ._library import (
    PERSONA_CALIBRATION_KIND,
    save_calibration,
)
from ._upgrade import upgrade_legacy_persona

PERSONA_CALIBRATION_STAGES = ("sampled", "validated", "interrogated", "admitted")
PERSONA_CALIBRATION_PROBES = ("internal", "external", "retest")

# Monotone upgrade lattice: calibration may only move a persona UP this rank.
# cloud_downloaded and trace_mined are provenance facts, never calibration
# outcomes — calibration stamps calibrated=True but leaves their class alone.
_CALIBRATABLE_RANK = {
    "legacy": 0,
    "hand_written": 1,
    "schema_sampled": 2,
    "policy_evolved": 3,
}
_CALIBRATION_TARGETS = ("hand_written", "schema_sampled", "policy_evolved")


def build_persona(
    *,
    name: str,
    situation: str,
    outcome: str,
    role: Optional[str] = None,
    summary: Optional[str] = None,
    language: Optional[str] = None,
    demographics: Optional[Mapping[str, Any]] = None,
    style_notes: Sequence[str] = (),
    temperament: Union[PersonaTemperament, Mapping[str, float], None] = None,
    behavior_policy: Union[BehaviorPolicy, Mapping[str, Any], None] = None,
    knowledge: Sequence[Union[PersonaFact, Mapping[str, Any]]] = (),
    attack: Union[AttackConditioning, Mapping[str, Any], None] = None,
    evidence_class: str = "hand_written",
) -> Persona:
    """Deterministic persona writer (hand_written / schema_sampled classes).

    Same inputs -> byte-identical persona (content hash stable). When a
    temperament is given without an explicit policy, the engine compiler
    derives layer 3 (explicit beats derived, ARCH Decision 4)."""
    if evidence_class not in ("hand_written", "schema_sampled"):
        raise ValueError(
            "build_persona writes hand_written or schema_sampled personas "
            f"only (got {evidence_class!r}); other classes have their own "
            "writers (calibration / pull / upgrade)"
        )
    temperament_model: Optional[PersonaTemperament] = None
    if temperament is not None:
        temperament_model = (
            temperament if isinstance(temperament, PersonaTemperament)
            else PersonaTemperament(**dict(temperament))
        )
    policy_model: Optional[BehaviorPolicy] = None
    if behavior_policy is not None:
        policy_model = (
            behavior_policy if isinstance(behavior_policy, BehaviorPolicy)
            else BehaviorPolicy(**dict(behavior_policy))
        )
    facts = [
        fact if isinstance(fact, PersonaFact) else PersonaFact(**dict(fact))
        for fact in knowledge
    ]
    attack_model: Optional[AttackConditioning] = None
    if attack is not None:
        attack_model = (
            attack if isinstance(attack, AttackConditioning)
            else AttackConditioning(**dict(attack))
        )
    embedded: Dict[str, Any] = {"name": name}
    if role:
        embedded["role"] = role
    persona = Persona(
        persona=embedded,
        situation=situation,
        outcome=outcome,
        identity=PersonaIdentity(
            name=name,
            role=role,
            summary=summary,
            language=language,
            demographics=dict(demographics or {}),
            style_notes=list(style_notes),
        ),
        temperament=temperament_model,
        behavior_policy=policy_model,
        knowledge=facts,
        attack=attack_model,
        provenance=PersonaProvenance(evidence_class=evidence_class),
    )
    if persona.behavior_policy is None and temperament_model is not None:
        compiled = compile_behavior_policy(persona)
        persona = _rebuild(persona, behavior_policy=compiled)
    return persona


def _rebuild(persona: Persona, **updates: Any) -> Persona:
    """Re-validate after an update so the content-address version re-stamps."""
    payload = persona.model_dump(exclude={"version"}, exclude_none=True)
    for key, value in updates.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return Persona(**payload)


def _taxonomy() -> Tuple[List[str], List[str]]:
    # Lazy read of the EXISTING gate-enforced 10x6 taxonomy constants —
    # facade-side membership check (the engine never imports trinity).
    from fi.alk import trinity

    return (
        list(trinity.V1_REDTEAM_RESEARCH_ATTACK_TYPES),
        list(trinity.V1_REDTEAM_RESEARCH_SURFACES),
    )


def validate_persona(persona_or_row: Union[Persona, Mapping[str, Any]]) -> Dict[str, Any]:
    """SPASM-style schema + realization-metric validation (UI §2.1 shape)."""
    findings: List[Dict[str, Any]] = []
    checks: Dict[str, str] = {}
    try:
        persona = upgrade_legacy_persona(persona_or_row)
        checks["schema"] = "pass"
    except Exception as exc:  # noqa: BLE001 — structured refusal, never a traceback
        return {
            "status": "invalid",
            "exit_code": 1,
            "checks": {"schema": "fail"},
            "findings": [{
                "type": "persona_schema_invalid",
                "level": "error",
                "reason": str(exc),
            }],
            "representativeness_claim": "none",
        }

    # every behavior axis carries its canon-paired metric (doctrine #7);
    # the pairing is structural: 1:1, ordered, and policy fields exist.
    pairing_ok = (
        len(PERSONA_BEHAVIOR_AXES) == len(PERSONA_BEHAVIOR_REALIZATION_METRICS) == 6
        and all(
            axis == pair[0] and pair[1] in BehaviorPolicy.model_fields
            for axis, pair in zip(PERSONA_BEHAVIOR_AXES, BEHAVIOR_POLICY_AXIS_FIELDS)
        )
    )
    checks["realization_metrics_per_axis"] = "pass" if pairing_ok else "fail"
    if not pairing_ok:
        findings.append({
            "type": "persona_axis_unobservable",
            "level": "error",
            "reason": "a behavior axis lost its canon realization-metric pairing",
        })

    try:
        first = compile_behavior_policy(persona)
        second = compile_behavior_policy(persona)
        checks["policy_compiles"] = (
            "pass" if first.model_dump() == second.model_dump() else "fail"
        )
    except Exception as exc:  # noqa: BLE001
        checks["policy_compiles"] = "fail"
        findings.append({
            "type": "persona_policy_compile_failed",
            "level": "error",
            "reason": str(exc),
        })

    # the persona never owns the task (2601.15290 separation)
    owns_task = any(key in persona.persona for key in ("goal", "goals", "task"))
    checks["goals_binding_is_scenario_scoped"] = "fail" if owns_task else "pass"
    if owns_task:
        findings.append({
            "type": "persona_owns_task",
            "level": "error",
            "reason": "goals belong to the Scenario, never the Persona",
        })

    demographics_present = bool(
        persona.identity is not None and persona.identity.demographics
    )
    checks["demographics"] = "flagged" if demographics_present else "absent"
    if demographics_present:
        findings.append({
            "type": "persona_demographics_flagged",
            "level": "info",
            "reason": (
                "demographic fields flag this persona for set-level bias "
                "lint; admit is blocked until the lint passes (P7-D4)"
            ),
        })

    if persona.attack is not None:
        attack_types, surfaces = _taxonomy()
        bad_strategies = sorted(set(persona.attack.strategies) - set(attack_types))
        bad_surfaces = sorted(set(persona.attack.surfaces) - set(surfaces))
        checks["attack_taxonomy"] = "fail" if (bad_strategies or bad_surfaces) else "pass"
        if bad_strategies or bad_surfaces:
            findings.append({
                "type": "persona_attack_taxonomy_violation",
                "level": "error",
                "reason": (
                    f"strategies {bad_strategies} / surfaces {bad_surfaces} "
                    "are outside the gate-enforced 10x6 taxonomy"
                ),
            })

    failed = any(value == "fail" for value in checks.values())
    return {
        "status": "invalid" if failed else "valid",
        "exit_code": 1 if failed else 0,
        "checks": checks,
        "findings": findings,
        "representativeness_claim": "none",
    }


def validate_scenario(scenario: Scenario) -> Dict[str, Any]:
    """Typed-scenario validation: kind contract + adversarial taxonomy."""
    findings: List[Dict[str, Any]] = []
    checks: Dict[str, str] = {"schema": "pass"}
    if scenario.kind == "adversarial":
        attack_types, surfaces = _taxonomy()
        ok = scenario.attack_type in attack_types and scenario.attack_surface in surfaces
        checks["attack_taxonomy"] = "pass" if ok else "fail"
        if not ok:
            findings.append({
                "type": "scenario_attack_taxonomy_violation",
                "level": "error",
                "reason": (
                    f"attack_type={scenario.attack_type!r} / "
                    f"attack_surface={scenario.attack_surface!r} outside the "
                    "gate-enforced 10x6 taxonomy"
                ),
            })
    failed = any(value == "fail" for value in checks.values())
    return {
        "status": "invalid" if failed else "valid",
        "exit_code": 1 if failed else 0,
        "checks": checks,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# PICon interrogation battery — deterministic scripted simulator
# ---------------------------------------------------------------------------

def _scripted_answer(
    persona: Persona,
    fact: PersonaFact,
    *,
    seed: int,
    repeat: int,
) -> str:
    """The scripted responder: answers fact probes from the knowledge store.

    ``retest_jitter`` in the embedded dict simulates a persona whose conduct
    is NOT seed-stable (per-repeat token) — the designed-to-fail retest
    fixture; within one battery run the answer is stable, across repeats it
    forks."""
    if fact.disclosure == "withhold":
        return "(withheld)"
    answer = fact.value
    if persona.persona.get("retest_jitter"):
        token = random.Random(f"{seed + repeat}:{fact.key}").random()
        answer = f"{answer} #{token:.6f}"
    return answer


def _battery_once(
    persona: Persona,
    scenario: Optional[Scenario],
    *,
    seed: int,
    repeat: int = 0,
) -> Dict[str, Any]:
    internal_probes: List[Dict[str, Any]] = []
    contradictions = 0
    seen_by_key: Dict[str, str] = {}
    for fact in persona.knowledge:
        first = _scripted_answer(persona, fact, seed=seed, repeat=repeat)
        paraphrase = _scripted_answer(persona, fact, seed=seed, repeat=repeat)
        consistent = first == paraphrase
        if fact.key in seen_by_key and seen_by_key[fact.key] != fact.value:
            consistent = False
            contradictions += 1
        seen_by_key.setdefault(fact.key, fact.value)
        internal_probes.append({
            "key": fact.key,
            "ask": first,
            "paraphrase": paraphrase,
            "consistent": consistent,
        })

    reality_breaks = 0
    external_probes: List[Dict[str, Any]] = []
    observable = (
        dict(scenario.constraints.observable_state)
        if scenario is not None and scenario.constraints is not None else {}
    )
    allowed_keys = (
        set(scenario.constraints.max_user_knowledge)
        if scenario is not None and scenario.constraints is not None
        and scenario.constraints.max_user_knowledge else None
    )
    for fact in persona.knowledge:
        breaks = False
        if fact.key in observable and str(observable[fact.key]) != fact.value:
            breaks = True  # contradicts declared world facts
        if allowed_keys is not None and fact.key not in allowed_keys:
            breaks = True  # knows state the scenario declares unobservable
        if breaks:
            reality_breaks += 1
        external_probes.append({"key": fact.key, "reality_break": breaks})

    probe_count = max(1, len(persona.knowledge))
    return {
        "internal": {
            "score": round(1.0 - contradictions / probe_count, 6),
            "probes": len(internal_probes),
            "contradictions": contradictions,
        },
        "external": {
            "score": round(1.0 - reality_breaks / probe_count, 6),
            "probes": len(external_probes),
            "reality_breaks": reality_breaks,
        },
        "trace": [probe["ask"] for probe in internal_probes]
        + [probe["paraphrase"] for probe in internal_probes],
    }


def _probe_divergence_step(
    first: Sequence[Any],
    second: Sequence[Any],
) -> Optional[int]:
    """First index where two probe trajectories fork (re-implementation of
    the live/_stats.divergence_step SEMANTIC — never imported)."""
    for index, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return index
    if len(first) != len(second):
        return min(len(first), len(second))
    return None


def calibrate_persona(
    persona: Union[Persona, Mapping[str, Any]],
    *,
    library: Optional[Any] = None,
    target_class: str = "schema_sampled",
    repeats: int = 2,
    seed: int = 7,
    scenario: Optional[Scenario] = None,
    bias_lint_result: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """sampled -> validated -> interrogated -> admitted (R§3.1 lifecycle).

    Deterministic, key-free, network-free. -> agent-learning.persona-calibration.v1.
    Monotone upgrades only; cloud_downloaded/trace_mined are provenance facts,
    never calibration outcomes. Any probe red -> the persona stays at its
    current class; the artifact records which probe failed. Uncalibrated
    personas RUN fine — they just carry the lowest class and cannot back
    release claims (PRD §4.2)."""
    if target_class not in _CALIBRATION_TARGETS:
        raise ValueError(
            f"target_class must be one of {list(_CALIBRATION_TARGETS)} "
            "(cloud_downloaded and trace_mined are provenance facts, never "
            "calibration outcomes)"
        )
    persona = upgrade_legacy_persona(persona)
    stages: List[str] = ["sampled"]

    validation = validate_persona(persona)
    artifact: Dict[str, Any] = {
        "kind": PERSONA_CALIBRATION_KIND,
        "persona": {
            "name": (persona.identity.name if persona.identity else None)
            or persona.persona.get("name"),
            "version_before": persona.version,
        },
        "validation": validation,
        "representativeness_claim": "none",
    }
    if validation["status"] != "valid":
        artifact.update({
            "stages": stages,
            "status": "failed",
            "verdict": "not_admit_eligible",
            "failed_probe": None,
            "bias_lint": dict(bias_lint_result) if bias_lint_result else None,
        })
        return artifact
    stages.append("validated")

    runs = [
        _battery_once(persona, scenario, seed=seed, repeat=index)
        for index in range(max(1, int(repeats)))
    ]
    base = runs[0]
    divergence: Optional[int] = None
    for repeat in runs[1:]:
        step = _probe_divergence_step(base["trace"], repeat["trace"])
        if step is not None:
            divergence = step if divergence is None else min(divergence, step)
    retest = {
        "score": 1.0 if divergence is None else 0.0,
        "replays": len(runs),
        "divergence_step": divergence,
        "method": "deterministic_replay",
    }
    probes = {
        "internal": base["internal"],
        "external": base["external"],
        "retest": retest,
    }
    stages.append("interrogated")

    failed_probe: Optional[str] = None
    for leg in PERSONA_CALIBRATION_PROBES:
        if probes[leg]["score"] < 1.0:
            failed_probe = leg
            break

    current_class = (
        persona.provenance.evidence_class if persona.provenance is not None else "legacy"
    )
    evidence = {"before": current_class, "after": current_class}
    if failed_probe is None:
        stages.append("admitted")
        if current_class in _CALIBRATABLE_RANK and (
            _CALIBRATABLE_RANK[current_class] < _CALIBRATABLE_RANK[target_class]
        ):
            evidence["after"] = target_class
        # else: monotone — keep the (equal-or-higher / provenance-fact) class

    artifact.update({
        "stages": stages,
        "status": "passed" if failed_probe is None else "failed",
        "verdict": "admit_eligible" if failed_probe is None else "not_admit_eligible",
        "failed_probe": failed_probe,
        "probes": probes,
        "seed": seed,
        "evidence_class": evidence,
        "bias_lint": dict(bias_lint_result) if bias_lint_result else None,
    })

    if failed_probe is None:
        ref_payload = json.dumps(
            {k: v for k, v in artifact.items() if k != "persona_payload"},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        calibration_ref = "sha256:" + hashlib.sha256(
            ref_payload.encode("utf-8")
        ).hexdigest()
        provenance = (persona.provenance or PersonaProvenance()).model_copy(update={
            "calibrated": True,
            "calibration_ref": calibration_ref,
            "evidence_class": evidence["after"],
        })
        persona = _rebuild(persona, provenance=provenance.model_dump(exclude_none=True))
        artifact["calibration_ref"] = calibration_ref
        artifact["persona"]["version_after"] = persona.version
    artifact["persona_payload"] = persona.model_dump(exclude_none=True)

    if library is not None:
        hex_digest = persona.content_hash().split(":", 1)[1]
        path = save_calibration(artifact, hex_digest, library=library)
        artifact["artifact_path"] = str(path)
    return artifact


__all__ = [
    "PERSONA_CALIBRATION_PROBES",
    "PERSONA_CALIBRATION_STAGES",
    "build_persona",
    "calibrate_persona",
    "validate_persona",
    "validate_scenario",
]
