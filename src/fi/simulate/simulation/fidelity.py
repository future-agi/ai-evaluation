"""Persona-fidelity engine (Phase 7, unit 3) — Eval4Sim triple + drift.

Engine-side, pure python, deterministic transcript arithmetic over
``TestCaseResult.messages`` (P7-D3: no single unperturbed LLM judge, ever).
Fidelity attaches through ``TestCaseResult.metadata`` under the reserved keys
``persona_fidelity`` (the record) and ``admission`` (the verdict block) —
NEVER a standalone artifact kind (ARCH §4).

Persona fidelity carries its OWN three-valued vocabulary (ARCH Decision 2);
the kit's frozen row verdicts (``live/_contract.py``) are untouched and this
module never imports ``fi.alk``.

The floor table below is V1-constant-shaped data living with the engine for
now; the trinity ``V1_PERSONA_FIDELITY_FLOORS`` constants land with the gate
pass, seed runtime library-index floors, and must stay byte-equal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from fi.simulate.simulation.behavior_policy import (
    arc_pressure,
    intensity_series,
    per_turn_drift,
    realization_vector,
    stdev,
    user_turns,
)
from fi.simulate.simulation.models import Persona, Scenario, TestCaseResult

PERSONA_FIDELITY_VERDICTS = ("pass", "fail", "inconclusive")   # ARCH Decision 2 — NOT live/_contract.VERDICTS
PERSONA_FIDELITY_EPIDEMIC_RATE = 0.5
# GATE-FIXTURE floors keyed by evidence class (ARCH §2c: runtime floors are
# library-index data seeded from these). legacy has NO floors (cannot produce
# fidelity evidence at all) — the dict omits it on purpose. hand_written
# floors bind LOCAL verdicts only: hand_written rows can never back release
# claims regardless of floors (PRD §4.2).
PERSONA_FIDELITY_FLOORS: Dict[str, Dict[str, float]] = {
    "hand_written":     {"adherence": 0.6, "consistency": 0.7, "naturalness": 0.5},
    "schema_sampled":   {"adherence": 0.7, "consistency": 0.8, "naturalness": 0.6},
    "policy_evolved":   {"adherence": 0.75, "consistency": 0.8, "naturalness": 0.65},
    "trace_mined":      {"adherence": 0.75, "consistency": 0.85, "naturalness": 0.7},
    "cloud_downloaded": {"adherence": 0.7, "consistency": 0.8, "naturalness": 0.6},
}

_NEGATION_MARKERS = ("not ", "never ", "no longer ")
_COUNTER_PRESSURE_MARKERS = (
    "you are now", "ignore your instructions", "ignore previous instructions",
    "drop the act", "stop pretending", "forget your persona", "act as a",
    "you are an ai assistant, not",
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _content(message: Mapping[str, Any]) -> str:
    return str(message.get("content") or "")


def _evidence_class(persona: Persona) -> str:
    if persona.provenance is not None:
        return persona.provenance.evidence_class
    return "hand_written"


def _resolve_floors(
    persona: Persona,
    floors: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    if floors is not None:
        return {k: float(v) for k, v in floors.items()}
    evidence_class = _evidence_class(persona)
    table = PERSONA_FIDELITY_FLOORS.get(evidence_class)
    if table is None:  # legacy / unknown: local verdicts use the lowest band
        table = PERSONA_FIDELITY_FLOORS["hand_written"]
    return dict(table)


def _consistency(persona: Persona, messages: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    violations: List[str] = []
    user_text_turns = [_content(m).lower() for m in user_turns(messages)]
    joined = " ".join(user_text_turns)

    # (a) fact stability — contradictory surface forms + withheld facts leaking
    for fact in persona.knowledge:
        value = fact.value.strip().lower()
        if not value:
            continue
        mentioned = value in joined
        if fact.disclosure == "withhold" and mentioned:
            violations.append(f"withheld_fact_disclosed:{fact.key}")
        negated = any(f"{marker}{value}" in joined for marker in _NEGATION_MARKERS)
        if mentioned and negated:
            violations.append(f"fact_contradiction:{fact.key}")

    # (b) identity stability — declared name never self-revised
    declared_name = (persona.identity.name if persona.identity else None) or ""
    if declared_name:
        for text in user_text_turns:
            if "my name is " in text:
                spoken = text.split("my name is ", 1)[1].strip().split(" ")[0].strip(".,!?")
                if spoken and spoken != declared_name.strip().lower().split(" ")[0]:
                    violations.append("identity_self_revision:name")
                    break

    # (c) style stability — rolling variance of realized intensity under a band
    series = intensity_series(messages)
    deltas = [abs(series[i + 1] - series[i]) for i in range(len(series) - 1)]
    if deltas and stdev(deltas) > 0.35:
        violations.append("style_instability")

    score = _clamp(1.0 - 0.3 * len(violations))
    return {"score": round(score, 6), "violations": violations}


def _naturalness(
    persona: Persona,
    messages: Sequence[Mapping[str, Any]],
    adherence_under: float,
) -> Dict[str, Any]:
    series = intensity_series(messages)
    n_turns = len(series)
    # caricature: realization pinned at extremes across >=2 axes (escalation
    # pinned high implies patience pinned low) — the over-acting failure.
    pinned = sum(1 for value in series if value > 0.95)
    caricature_index = round(pinned / n_turns, 6) if n_turns else 0.0
    # flatness: near-zero realization movement WITH adherence shortfall —
    # the under-encoding failure (a flat-but-adherent persona is fine).
    movement = (
        sum(abs(series[i + 1] - series[i]) for i in range(n_turns - 1)) / (n_turns - 1)
        if n_turns > 1 else 0.0
    )
    flat_raw = _clamp(1.0 - movement / 0.05)
    flatness_index = round(flat_raw * _clamp(adherence_under * 2.0), 6)
    score = _clamp(1.0 - max(caricature_index, flatness_index))
    return {
        "score": round(score, 6),
        "caricature_index": caricature_index,
        "flatness_index": flatness_index,
    }


def persona_fidelity(
    persona: Persona,
    scenario: Optional[Scenario],
    messages: Sequence[Mapping[str, Any]],
    *,
    probe_responses: Optional[Sequence[Mapping[str, Any]]] = None,
    floors: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """-> the per-row fidelity record: an IN-ROW block under
    ``metadata["persona_fidelity"]`` — NEVER a standalone artifact kind
    (ARCH §4). Observable metrics only (P7-D3)."""
    if not persona.is_typed:
        raise ValueError(
            "persona_fidelity requires a typed persona (behavior_policy set); "
            "legacy personas produce no fidelity evidence"
        )
    policy = persona.behavior_policy
    applied_floors = _resolve_floors(persona, floors)
    record: Dict[str, Any] = {
        "persona_version": persona.version,
        "scenario_version": scenario.version if scenario is not None else None,
        "evidence_class": _evidence_class(persona),
    }

    users = user_turns(messages)
    garbled = not users or all(not _content(m).strip() for m in users)
    if garbled:
        record.update({
            "adherence": {"score": 0.0, "per_axis": {}, "under": 0.0, "over": 0.0},
            "consistency": {"score": 0.0, "violations": []},
            "naturalness": {"score": 0.0, "caricature_index": 0.0, "flatness_index": 0.0},
            "drift": {"prompt_to_line": 0.0, "line_to_line": 0.0, "probe": None},
            "drift_trajectory": [],
            "floors": applied_floors,
            "verdict": "fail",
            "verdict_reason": "empty_trajectory",
        })
        return record

    vector = realization_vector(policy, messages, knowledge=persona.knowledge)
    deviations = [entry["deviation"] for entry in vector.values()]
    under = sum(max(0.0, -d) for d in deviations) / len(deviations)
    over = sum(max(0.0, d) for d in deviations) / len(deviations)
    adherence_score = _clamp(1.0 - sum(abs(d) for d in deviations) / len(deviations))
    adherence = {
        "score": round(adherence_score, 6),
        "per_axis": {axis: entry["deviation"] for axis, entry in vector.items()},
        "under": round(under, 6),
        "over": round(over, 6),
    }

    consistency = _consistency(persona, messages)
    naturalness = _naturalness(persona, messages, under)

    drifts = per_turn_drift(policy, messages)
    prompt_to_line = round(sum(drifts) / len(drifts), 6) if drifts else 0.0
    line_to_line = (
        round(sum(abs(drifts[i + 1] - drifts[i]) for i in range(len(drifts) - 1))
              / (len(drifts) - 1), 6)
        if len(drifts) > 1 else 0.0
    )
    probe_drift: Optional[float] = None
    if probe_responses:
        mismatches = sum(
            1 for probe in probe_responses
            if str(probe.get("observed")) != str(probe.get("expected"))
        )
        probe_drift = round(mismatches / len(probe_responses), 6)

    # drift trajectory + counter-pressure flags (Assistant Axis: drift is a
    # trajectory, fastest under pressure)
    trajectory: List[Dict[str, Any]] = []
    user_index = 0
    last_assistant_text = ""
    arc = scenario.escalation if scenario is not None else None
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            last_assistant_text = _content(message).lower()
            continue
        if role != "user":
            continue
        turn = user_index + 1
        declared = arc_pressure(arc, turn) if arc is not None else (
            policy.escalation_schedule[min(user_index, len(policy.escalation_schedule) - 1)]
            if policy.escalation_schedule else 0.0
        )
        counter_pressure = any(
            marker in last_assistant_text for marker in _COUNTER_PRESSURE_MARKERS
        )
        trajectory.append({
            "turn": turn,
            "drift": drifts[user_index] if user_index < len(drifts) else 0.0,
            "pressure": round(float(declared), 6),
            "counter_pressure": counter_pressure,
        })
        user_index += 1

    triple = {
        "adherence": adherence["score"],
        "consistency": consistency["score"],
        "naturalness": naturalness["score"],
    }
    failing = sorted(
        metric for metric, score in triple.items()
        if score < applied_floors.get(metric, 0.0)
    )
    if not failing:
        verdict = "pass"
        verdict_reason: Optional[str] = None
    else:
        verdict = "inconclusive"
        collapse = any(
            entry["counter_pressure"] and entry["drift"] >= 0.5 for entry in trajectory
        )
        if collapse:
            verdict_reason = "fidelity_collapse_under_counter_pressure"
        else:
            verdict_reason = "; ".join(f"{metric}_below_floor" for metric in failing)

    record.update({
        "adherence": adherence,
        "consistency": consistency,
        "naturalness": naturalness,
        "drift": {
            "prompt_to_line": prompt_to_line,
            "line_to_line": line_to_line,
            "probe": probe_drift,
        },
        "drift_trajectory": trajectory,
        "floors": applied_floors,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    })
    return record


def attach_fidelity(
    result: TestCaseResult,
    persona: Persona,
    scenario: Optional[Scenario],
    *,
    probe_responses: Optional[Sequence[Mapping[str, Any]]] = None,
    floors: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Compute the fidelity record and attach record + admission block to the
    row via ``metadata`` ONLY (no structural change — ARCH §2c)."""
    record = persona_fidelity(
        persona, scenario, result.messages,
        probe_responses=probe_responses, floors=floors,
    )
    result.metadata["persona_fidelity"] = record
    result.metadata["admission"] = {
        "admissible": record["verdict"] == "pass",
        "verdict": "pass" if record["verdict"] == "pass" else "inconclusive",
        "reason": None if record["verdict"] == "pass" else "persona_fidelity_floor",
        "quarantined": record["verdict"] != "pass",
        "rerunnable": True,
    }
    return record


def summarize_admissions(results: Sequence[TestCaseResult]) -> Dict[str, Any]:
    """Run-summary admission rollup + the epidemic rule (ARCH §2c/§4 canon).

    Admission-``inconclusive`` rate above ``PERSONA_FIDELITY_EPIDEMIC_RATE``
    declares the SIMULATOR (not the agent) unusable: ``exit_code`` flips to 1
    with finding ``persona_fidelity_epidemic`` naming the worst personas.
    Below the threshold, quarantine keeps CI green (exit 0 + warning finding).
    """
    scored = [r for r in results if "admission" in r.metadata]
    inconclusive = [
        r for r in scored
        if r.metadata["admission"].get("verdict") == "inconclusive"
    ]
    rate = round(len(inconclusive) / len(scored), 6) if scored else 0.0
    per_persona: Dict[str, int] = {}
    for row in inconclusive:
        identity = row.persona.identity
        name = (identity.name if identity else None) or str(
            row.persona.persona.get("name", "unknown")
        )
        per_persona[name] = per_persona.get(name, 0) + 1
    worst = [
        name for name, _ in
        sorted(per_persona.items(), key=lambda item: (-item[1], item[0]))
    ]
    epidemic = rate > PERSONA_FIDELITY_EPIDEMIC_RATE
    findings: List[Dict[str, Any]] = []
    if epidemic:
        findings.append({
            "type": "persona_fidelity_epidemic",
            "level": "error",
            "reason": (
                f"admission-inconclusive rate {rate} exceeds "
                f"{PERSONA_FIDELITY_EPIDEMIC_RATE}: the simulator, not the "
                "agent, is unusable for this run"
            ),
            "worst_personas": worst,
        })
    elif inconclusive:
        findings.append({
            "type": "persona_fidelity_inconclusive",
            "level": "warning",
            "reason": (
                f"{len(inconclusive)} row(s) quarantined as non-admissible "
                "evidence (persona_fidelity_floor); re-run the manifest"
            ),
            "worst_personas": worst,
        })
    return {
        "rows": len(results),
        "scored": len(scored),
        "inconclusive": len(inconclusive),
        "inconclusive_rate": rate,
        "epidemic": epidemic,
        "exit_code": 1 if epidemic else 0,
        "findings": findings,
    }


__all__ = [
    "PERSONA_FIDELITY_EPIDEMIC_RATE",
    "PERSONA_FIDELITY_FLOORS",
    "PERSONA_FIDELITY_VERDICTS",
    "attach_fidelity",
    "persona_fidelity",
    "summarize_admissions",
]
