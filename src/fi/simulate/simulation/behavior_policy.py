"""Behavior-policy compiler + per-axis realization metrics (Phase 7, unit 2).

Engine-side home (ARCH Decision 3): stdlib only — deterministic, no LLM, no
numpy. The six policy parameters map 1:1 onto the canon behavior axes, each
paired with its transcript-observable realization metric; a parameter without
one DOES NOT SHIP (RESEARCH §3.4 limit 4). The V1-constant-shaped data below
lives with the engine for now; the trinity ``V1_*`` constants land with the
gate pass and must stay byte-equal to these tuples.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from fi.simulate.simulation.models import (
    BehaviorPolicy,
    EscalationArc,
    Persona,
    PersonaFact,
    PersonaTemperament,
)

# Canon pairing (ARCH §4): axes <-> realization metrics, 1:1 and ordered.
PERSONA_BEHAVIOR_AXES = (
    "patience", "disclosure", "interruption", "escalation",
    "cooperation", "repair",
)
PERSONA_BEHAVIOR_REALIZATION_METRICS = (
    "turns_to_escalation", "info_withholding_rate", "interruption_count",
    "intensity_trajectory_match", "compliance_rate", "repair_turn_fraction",
)
AXIS_TO_METRIC = dict(zip(PERSONA_BEHAVIOR_AXES, PERSONA_BEHAVIOR_REALIZATION_METRICS))
# Axis -> BehaviorPolicy field, same order as the axes (pinned by tests).
BEHAVIOR_POLICY_AXIS_FIELDS = (
    ("patience", "patience_curve"),
    ("disclosure", "disclosure_policy"),
    ("interruption", "interruption_propensity"),
    ("escalation", "escalation_schedule"),
    ("cooperation", "cooperation_bounds"),
    ("repair", "repair_propensity"),
)

_DEFAULT_POLICY_TURNS = 6

# Deterministic lexicons for transcript-observable scoring. These are the
# measurement contract shared verbatim by fidelity, calibration retest, and
# bias-lint caricature checks — one implementation, three consumers.
_URGENCY_MARKERS = (
    "immediately", "urgent", "unacceptable", "supervisor", "manager",
    "escalate", "ridiculous", "fed up", "right now", "asap", "demand",
    "complaint", "lawyer", "cancel my", "last warning", "furious",
)
_INTERRUPT_MARKERS = (
    "(interrupting)", "let me stop you", "stop right there", "hold on, stop",
    "i'm cutting in",
)
_MISUNDERSTANDING_MARKERS = (
    "i don't understand", "could you clarify", "i'm not sure i follow",
    "can you rephrase", "i may have misunderstood", "that's not what i",
)
_REPAIR_MARKERS = (
    "i mean", "let me rephrase", "to clarify", "sorry, i meant",
    "what i meant", "let me explain again",
)
_AGENT_REQUEST_MARKERS = (
    "please provide", "can you share", "could you confirm", "what is your",
    "may i have", "please confirm", "i need your",
)
_REFUSAL_MARKERS = (
    "won't", "will not", "refuse", "not comfortable", "i cannot share",
    "not going to", "i'd rather not",
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _curve_at(curve: Sequence[float], turn: int, default: float) -> float:
    if not curve:
        return default
    index = min(max(turn, 0), len(curve) - 1)
    return float(curve[index])


def compile_behavior_policy(persona: Persona) -> BehaviorPolicy:
    """Temperament axes -> policy parameters. Pure, total, deterministic.

    If ``persona.behavior_policy`` is set it WINS (explicit beats derived);
    temperament only fills gaps. Mapping per R§3.4:
      rajas  -> interruption_propensity, escalation_schedule slope
      sattva -> disclosure_policy, cooperation_bounds, repair_propensity
      tamas  -> patience_curve decay, cooperation/disclosure damping
                (withdrawal realized through the patience+cooperation metrics;
                verbosity/tempo dials are post-v1.x — ARCH Decision 4)
    The exact arithmetic is fixture-pinned: same persona -> byte-identical
    policy, forever.
    """
    if persona.behavior_policy is not None:
        return persona.behavior_policy.model_copy(deep=True)
    temperament = persona.temperament or PersonaTemperament()
    rajas = float(temperament.rajas)
    sattva = float(temperament.sattva)
    tamas = float(temperament.tamas)
    patience_curve = [
        round(_clamp(1.0 - (0.04 + 0.16 * tamas) * index), 6)
        for index in range(_DEFAULT_POLICY_TURNS)
    ]
    escalation_schedule = [
        round(_clamp(rajas * index / (_DEFAULT_POLICY_TURNS - 1)), 6)
        for index in range(_DEFAULT_POLICY_TURNS)
    ]
    return BehaviorPolicy(
        patience_curve=patience_curve,
        disclosure_policy=round(_clamp((0.2 + 0.6 * sattva) * (1.0 - 0.3 * tamas)), 6),
        interruption_propensity=round(_clamp(0.05 + 0.6 * rajas), 6),
        escalation_schedule=escalation_schedule,
        cooperation_bounds=round(_clamp((0.4 + 0.5 * sattva) * (1.0 - 0.3 * tamas)), 6),
        repair_propensity=round(_clamp(0.2 + 0.7 * sattva), 6),
    )


def render_policy_directives(
    policy: BehaviorPolicy,
    turn: int,
    pressure: float = 0.0,
) -> Dict[str, float]:
    """Per-turn target dials — one dial per canon axis (ARCH §2b)."""
    return {
        "patience_level": round(_curve_at(policy.patience_curve, turn, 1.0), 6),
        "disclosure_rate": round(float(policy.disclosure_policy), 6),
        "interruption_propensity": round(float(policy.interruption_propensity), 6),
        "escalation_level": round(
            max(_curve_at(policy.escalation_schedule, turn, 0.0), _clamp(float(pressure))), 6
        ),
        "cooperation_level": round(float(policy.cooperation_bounds), 6),
        "repair_propensity": round(float(policy.repair_propensity), 6),
    }


def arc_pressure(arc: Optional[EscalationArc], turn: int) -> float:
    """Declared scenario pressure at a 1-based turn (last step at/before it)."""
    if arc is None or not arc.steps:
        return 0.0
    pressure = 0.0
    for step in arc.steps:
        if step.turn <= turn:
            pressure = float(step.pressure)
    return pressure


# ---------------------------------------------------------------------------
# Transcript primitives
# ---------------------------------------------------------------------------

def _content(message: Mapping[str, Any]) -> str:
    return str(message.get("content") or "")


def user_turns(messages: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [m for m in messages if m.get("role") == "user"]


def assistant_turns(messages: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [m for m in messages if m.get("role") == "assistant"]


def turn_intensity(message: Mapping[str, Any]) -> float:
    """Lexicon-scored urgency/pressure of one user turn, 0..1."""
    text = _content(message).lower()
    matches = sum(1 for marker in _URGENCY_MARKERS if marker in text)
    return _clamp(matches / 3.0)


def intensity_series(messages: Sequence[Mapping[str, Any]]) -> List[float]:
    return [round(turn_intensity(m), 6) for m in user_turns(messages)]


def _is_interrupt(message: Mapping[str, Any]) -> bool:
    if message.get("interrupt") is True:
        return True
    text = _content(message).lower()
    return any(marker in text for marker in _INTERRUPT_MARKERS)


# ---------------------------------------------------------------------------
# The six realization metrics (canon names; transcript-observable only)
# ---------------------------------------------------------------------------

def turns_to_escalation(messages: Sequence[Mapping[str, Any]]) -> int:
    """Turn index (0-based, user turns) where intensity first rises; the
    user-turn count when it never does."""
    series = intensity_series(messages)
    for index, value in enumerate(series):
        if value >= 0.34:
            return index
    return len(series)


def info_withholding_rate(
    facts: Sequence[PersonaFact],
    messages: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    """Facts withheld ÷ facts solicited (non-withhold facts). None when the
    persona declares no disclosable facts (unobservable — never fabricated)."""
    disclosable = [f for f in facts if f.disclosure != "withhold"]
    if not disclosable:
        return None
    text = " ".join(_content(m).lower() for m in user_turns(messages))
    revealed = sum(1 for fact in disclosable if fact.value.strip().lower() in text)
    return round(1.0 - revealed / len(disclosable), 6)


def interruption_count(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for m in user_turns(messages) if _is_interrupt(m))


def intensity_trajectory_match(
    policy: BehaviorPolicy,
    messages: Sequence[Mapping[str, Any]],
) -> float:
    """1 − mean L1 distance between realized per-turn pressure and the
    declared escalation schedule."""
    series = intensity_series(messages)
    if not series:
        return 0.0
    distance = sum(
        abs(value - _curve_at(policy.escalation_schedule, index, 0.0))
        for index, value in enumerate(series)
    ) / len(series)
    return round(_clamp(1.0 - distance), 6)


def compliance_rate(messages: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """Agent requests honored ÷ requests made by the agent of the simulated
    USER. None when the agent made no requests."""
    requests = 0
    honored = 0
    ordered = list(messages)
    for index, message in enumerate(ordered):
        if message.get("role") != "assistant":
            continue
        text = _content(message).lower()
        if not any(marker in text for marker in _AGENT_REQUEST_MARKERS):
            continue
        requests += 1
        for later in ordered[index + 1:]:
            if later.get("role") == "user":
                reply = _content(later).lower()
                if reply and not any(marker in reply for marker in _REFUSAL_MARKERS):
                    honored += 1
                break
    if requests == 0:
        return None
    return round(honored / requests, 6)


def repair_turn_fraction(messages: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """Good-faith repair turns after a flagged misunderstanding ÷
    misunderstanding turns. None when no misunderstanding was flagged."""
    misunderstandings = 0
    repairs = 0
    ordered = list(messages)
    for index, message in enumerate(ordered):
        if message.get("role") != "assistant":
            continue
        text = _content(message).lower()
        if not any(marker in text for marker in _MISUNDERSTANDING_MARKERS):
            continue
        misunderstandings += 1
        for later in ordered[index + 1:]:
            if later.get("role") == "user":
                reply = _content(later).lower()
                if any(marker in reply for marker in _REPAIR_MARKERS):
                    repairs += 1
                break
    if misunderstandings == 0:
        return None
    return round(repairs / misunderstandings, 6)


# ---------------------------------------------------------------------------
# Realization vector — shared by fidelity, calibration retest, and bias lint
# ---------------------------------------------------------------------------

def realization_vector(
    policy: BehaviorPolicy,
    messages: Sequence[Mapping[str, Any]],
    *,
    knowledge: Iterable[PersonaFact] = (),
) -> Dict[str, Dict[str, Any]]:
    """Observed values + signed deviations per canon axis.

    Each entry: ``{"metric", "value", "target", "observed", "deviation"}``
    where ``target``/``observed`` are normalized 0..1 in the same orientation
    and ``deviation = observed - target`` (signed; two-sided by construction).
    Unobservable axes (no facts / no requests / no misunderstandings) report
    ``value=None`` and zero deviation — never fabricated evidence.
    """
    facts = list(knowledge)
    series = intensity_series(messages)
    users = user_turns(messages)
    n_turns = len(users)

    # patience — observed per-turn patience proxy = 1 - intensity
    patience_target = (
        sum(_curve_at(policy.patience_curve, i, 1.0) for i in range(n_turns)) / n_turns
        if n_turns else _curve_at(policy.patience_curve, 0, 1.0)
    )
    patience_observed = (
        sum(1.0 - value for value in series) / n_turns if n_turns else patience_target
    )

    # disclosure — observed disclosure fraction vs the declared policy
    withholding = info_withholding_rate(facts, messages)
    disclosure_target = float(policy.disclosure_policy)
    disclosure_observed = (
        1.0 - withholding if withholding is not None else disclosure_target
    )

    # interruption — observed interruption rate vs propensity
    interruptions = interruption_count(messages)
    interruption_target = float(policy.interruption_propensity)
    interruption_observed = (
        interruptions / n_turns if n_turns else interruption_target
    )

    # escalation — realized mean pressure vs declared mean schedule
    escalation_target = (
        sum(_curve_at(policy.escalation_schedule, i, 0.0) for i in range(n_turns)) / n_turns
        if n_turns else _curve_at(policy.escalation_schedule, 0, 0.0)
    )
    escalation_observed = sum(series) / n_turns if n_turns else escalation_target
    match = intensity_trajectory_match(policy, messages)

    # cooperation — compliance rate vs cooperation bounds
    compliance = compliance_rate(messages)
    cooperation_target = float(policy.cooperation_bounds)
    cooperation_observed = compliance if compliance is not None else cooperation_target

    # repair — repair fraction vs repair propensity
    repair = repair_turn_fraction(messages)
    repair_target = float(policy.repair_propensity)
    repair_observed = repair if repair is not None else repair_target

    def _entry(metric: str, value: Any, target: float, observed: float) -> Dict[str, Any]:
        return {
            "metric": metric,
            "value": value,
            "target": round(target, 6),
            "observed": round(observed, 6),
            "deviation": round(observed - target, 6),
        }

    return {
        "patience": _entry(
            "turns_to_escalation", turns_to_escalation(messages),
            patience_target, patience_observed,
        ),
        "disclosure": _entry(
            "info_withholding_rate", withholding,
            disclosure_target, disclosure_observed,
        ),
        "interruption": _entry(
            "interruption_count", interruptions,
            interruption_target, interruption_observed,
        ),
        "escalation": _entry(
            "intensity_trajectory_match", match,
            escalation_target, escalation_observed,
        ),
        "cooperation": _entry(
            "compliance_rate", compliance,
            cooperation_target, cooperation_observed,
        ),
        "repair": _entry(
            "repair_turn_fraction", repair,
            repair_target, repair_observed,
        ),
    }


def per_turn_drift(
    policy: BehaviorPolicy,
    messages: Sequence[Mapping[str, Any]],
) -> List[float]:
    """Per-user-turn drift: mean |observed − declared| over the per-turn
    observable axes (patience, escalation)."""
    drifts: List[float] = []
    for index, value in enumerate(intensity_series(messages)):
        patience_gap = abs((1.0 - value) - _curve_at(policy.patience_curve, index, 1.0))
        escalation_gap = abs(value - _curve_at(policy.escalation_schedule, index, 0.0))
        drifts.append(round((patience_gap + escalation_gap) / 2.0, 6))
    return drifts


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


__all__ = [
    "AXIS_TO_METRIC",
    "BEHAVIOR_POLICY_AXIS_FIELDS",
    "PERSONA_BEHAVIOR_AXES",
    "PERSONA_BEHAVIOR_REALIZATION_METRICS",
    "arc_pressure",
    "assistant_turns",
    "compile_behavior_policy",
    "compliance_rate",
    "info_withholding_rate",
    "intensity_series",
    "intensity_trajectory_match",
    "interruption_count",
    "per_turn_drift",
    "realization_vector",
    "render_policy_directives",
    "repair_turn_fraction",
    "turn_intensity",
    "turns_to_escalation",
    "user_turns",
]
