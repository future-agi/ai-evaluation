"""Set-level bias lint — the library-admission gate (Phase 7, unit 5.4).

Operates on persona SETS, not individuals; four deterministic checks
(``demographic_clustering``, ``trait_demographic_cells``,
``subgroup_error_redistribution``, ``caricature_two_sided``), all stdlib
arithmetic, no network. Locale-sensitive: the lint re-runs per
``identity.language`` value present in the set (2604.23600's bilingual
finding) and the stamp records every locale linted. Results ride INSIDE
calibration artifacts as a ``bias_lint`` block and stamp the library index —
never a standalone artifact kind (ARCH §2f).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from fi.simulate.simulation.models import Persona

from ._upgrade import upgrade_legacy_persona

PERSONA_BIAS_LINT_CHECKS = (
    "demographic_clustering", "trait_demographic_cells",
    "subgroup_error_redistribution", "caricature_two_sided",
)

_DEMOGRAPHIC_VARIANCE_CEILING = 0.05   # SCOPE: demographics explain ~1.5%
_TRAIT_EXTREME = 0.8
_POLICY_EXTREME_HIGH = 0.95
_POLICY_EXTREME_LOW = 0.05
_SUBGROUP_REDISTRIBUTION_FACTOR = 2.0

_TEMPERAMENT_AXES = ("rajas", "sattva", "tamas")


def _coerce(personas: Sequence[Union[Persona, Mapping[str, Any]]]) -> List[Persona]:
    return [upgrade_legacy_persona(p) for p in personas]


def _policy_scalars(persona: Persona) -> List[float]:
    """The numeric behavior-policy parameter vector (curves as means)."""
    policy = persona.behavior_policy
    if policy is None:
        return []
    patience = (
        sum(policy.patience_curve) / len(policy.patience_curve)
        if policy.patience_curve else 1.0
    )
    escalation = (
        sum(policy.escalation_schedule) / len(policy.escalation_schedule)
        if policy.escalation_schedule else 0.0
    )
    return [
        patience,
        float(policy.disclosure_policy),
        float(policy.interruption_propensity),
        escalation,
        float(policy.cooperation_bounds),
        float(policy.repair_propensity),
    ]


def _demographic_fields(personas: Sequence[Persona]) -> Dict[str, Dict[str, List[int]]]:
    """field -> value -> member indexes (only fields actually present)."""
    fields: Dict[str, Dict[str, List[int]]] = {}
    for index, persona in enumerate(personas):
        demographics = (
            persona.identity.demographics if persona.identity is not None else {}
        )
        for field, value in (demographics or {}).items():
            fields.setdefault(str(field), {}).setdefault(str(value), []).append(index)
    return fields


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _demographic_clustering(personas: Sequence[Persona]) -> Dict[str, Any]:
    vectors = [_policy_scalars(p) for p in personas]
    usable = [v for v in vectors if v]
    fields = _demographic_fields(personas)
    if not fields or len(usable) < 2:
        return {
            "status": "pass",
            "variance_explained_by_demographics": 0.0,
            "ceiling": _DEMOGRAPHIC_VARIANCE_CEILING,
        }
    dimensions = len(usable[0])
    worst = 0.0
    for field, groups in sorted(fields.items()):
        explained_total = 0.0
        total_total = 0.0
        for dim in range(dimensions):
            values = [
                vectors[i][dim] for i in range(len(personas)) if vectors[i]
            ]
            total = _variance(values)
            grand_mean = sum(values) / len(values) if values else 0.0
            between = 0.0
            for _, members in sorted(groups.items()):
                member_values = [vectors[i][dim] for i in members if vectors[i]]
                if not member_values:
                    continue
                group_mean = sum(member_values) / len(member_values)
                between += len(member_values) * (group_mean - grand_mean) ** 2
            between /= max(1, len(values))
            explained_total += between
            total_total += total
        share = explained_total / total_total if total_total > 0 else 0.0
        worst = max(worst, share)
    status = "pass" if worst <= _DEMOGRAPHIC_VARIANCE_CEILING else "fail"
    result: Dict[str, Any] = {
        "status": status,
        "variance_explained_by_demographics": round(worst, 6),
        "ceiling": _DEMOGRAPHIC_VARIANCE_CEILING,
    }
    if status == "fail":
        result["reason"] = (
            "behavioral variance concentrates on identity.demographics fields "
            "— the set encodes stereotypes instead of behavior"
        )
    return result


def _trait_demographic_cells(personas: Sequence[Persona]) -> Dict[str, Any]:
    fields = _demographic_fields(personas)
    flagged: List[Dict[str, Any]] = []
    cells_tested = 0
    for axis in _TEMPERAMENT_AXES:
        extremes = [
            index for index, persona in enumerate(personas)
            if persona.temperament is not None
            and getattr(persona.temperament, axis) >= _TRAIT_EXTREME
        ]
        for field, groups in sorted(fields.items()):
            cells_tested += len(groups)
            if len(groups) < 2 or len(extremes) < 2:
                continue
            for value, members in sorted(groups.items()):
                if set(extremes) and set(extremes) <= set(members):
                    flagged.append({
                        "cell": f"{field}:{value} x high_{axis}",
                        "reason": (
                            f"{axis} weight >= {_TRAIT_EXTREME} applied ONLY "
                            f"to the {field}={value} personas"
                        ),
                    })
    return {
        "status": "fail" if flagged else "pass",
        "cells_tested": cells_tested,
        "flagged_cells": flagged,
    }


def _subgroup_error_redistribution(
    personas: Sequence[Persona],
    transcripts: Optional[Mapping[int, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Fidelity-floor failure rate per demographic subgroup vs global —
    runs only when fixture transcripts/fidelity outcomes are provided."""
    if not transcripts:
        return {"status": "pass", "probes": 0}
    failures = {
        index for index, outcome in transcripts.items()
        if outcome.get("verdict") not in (None, "pass")
    }
    total = len(transcripts)
    global_rate = len(failures) / total if total else 0.0
    fields = _demographic_fields(personas)
    flagged: List[Dict[str, Any]] = []
    for field, groups in sorted(fields.items()):
        for value, members in sorted(groups.items()):
            scored = [m for m in members if m in transcripts]
            if not scored:
                continue
            rate = sum(1 for m in scored if m in failures) / len(scored)
            if global_rate > 0 and rate > _SUBGROUP_REDISTRIBUTION_FACTOR * global_rate and rate > 0.5:
                flagged.append({
                    "subgroup": f"{field}:{value}",
                    "failure_rate": round(rate, 6),
                    "global_rate": round(global_rate, 6),
                })
    return {
        "status": "fail" if flagged else "pass",
        "probes": total,
        "flagged_subgroups": flagged,
    }


def _caricature_two_sided(
    personas: Sequence[Persona],
    transcripts: Optional[Mapping[int, Mapping[str, Any]]],
) -> Dict[str, Any]:
    over_acting: List[Dict[str, Any]] = []
    for index, persona in enumerate(personas):
        policy = persona.behavior_policy
        if policy is None:
            continue
        scalars = _policy_scalars(persona)
        pinned = sum(
            1 for value in scalars
            if value >= _POLICY_EXTREME_HIGH or value <= _POLICY_EXTREME_LOW
        )
        if pinned >= 3:
            name = (persona.identity.name if persona.identity else None) or str(
                persona.persona.get("name", f"persona[{index}]")
            )
            over_acting.append({
                "persona": name,
                "pinned_axes": pinned,
                "direction": "over_acting",
                "reason": "policy targets pinned at extremes across >=3 axes",
            })
    if transcripts:
        for index, outcome in sorted(transcripts.items()):
            record = outcome.get("naturalness") or {}
            if record.get("caricature_index", 0.0) >= 0.6:
                persona = personas[index] if index < len(personas) else None
                name = (
                    (persona.identity.name if persona and persona.identity else None)
                    or f"persona[{index}]"
                )
                over_acting.append({
                    "persona": name,
                    "direction": "over_acting",
                    "reason": "realized caricature_index >= 0.6",
                })
            if record.get("flatness_index", 0.0) >= 0.6:
                over_acting.append({
                    "persona": f"persona[{index}]",
                    "direction": "under_encoding",
                    "reason": "realized flatness_index >= 0.6",
                })
    return {
        "status": "fail" if over_acting else "pass",
        "over_acting_flags": len(over_acting),
        "flags": over_acting,
    }


def _lint_locale(
    personas: Sequence[Persona],
    transcripts: Optional[Mapping[int, Mapping[str, Any]]],
) -> Dict[str, Any]:
    return {
        "demographic_clustering": _demographic_clustering(personas),
        "trait_demographic_cells": _trait_demographic_cells(personas),
        "subgroup_error_redistribution": _subgroup_error_redistribution(personas, transcripts),
        "caricature_two_sided": _caricature_two_sided(personas, transcripts),
    }


def bias_lint(
    personas: Sequence[Union[Persona, Mapping[str, Any]]],
    *,
    transcripts: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Set-level lint over the four canon checks, re-run per locale.

    ``transcripts`` (optional): member-index -> per-row fidelity outcome
    (``verdict`` + ``naturalness`` block) for the run-history checks."""
    coerced = _coerce(personas)
    locales = sorted({
        (p.identity.language if p.identity is not None and p.identity.language else "und")
        for p in coerced
    }) or ["und"]
    per_locale: Dict[str, Dict[str, Any]] = {}
    for locale in locales:
        members = [
            (index, persona) for index, persona in enumerate(coerced)
            if (
                persona.identity.language
                if persona.identity is not None and persona.identity.language
                else "und"
            ) == locale
        ]
        member_personas = [persona for _, persona in members]
        member_transcripts = (
            {
                position: transcripts[original]
                for position, (original, _) in enumerate(members)
                if transcripts and original in transcripts
            }
            if transcripts else None
        )
        per_locale[locale] = _lint_locale(member_personas, member_transcripts)

    failed = any(
        check["status"] == "fail"
        for checks in per_locale.values()
        for check in checks.values()
    )
    with_demographics = sum(
        1 for persona in coerced
        if persona.identity is not None and persona.identity.demographics
    )
    # the headline checks block mirrors the first locale (single-locale sets
    # read flat, multi-locale sets read per_locale)
    headline = per_locale[locales[0]]
    return {
        "status": "failed" if failed else "passed",
        "exit_code": 1 if failed else 0,
        "checks": headline,
        "per_locale": per_locale,
        "locales_linted": locales,
        "set": {"personas": len(coerced), "with_demographics": with_demographics},
        "representativeness_claim": "none",
    }


__all__ = ["PERSONA_BIAS_LINT_CHECKS", "bias_lint"]
