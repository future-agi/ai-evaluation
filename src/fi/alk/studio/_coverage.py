"""Coverage machinery (Phase 7, unit 5): k-way expansion, obligation
coverage, budgeted residual estimator, coverage-guided generation.

Doctrine #3: the headline numbers are ``obligation_coverage`` and
``residual_uncovered`` — the payload MUST NOT carry ``library_size`` /
``scenario_count`` at top level (the ``no_global_aggregate`` move; library
size is never the reported number). Coverage output embeds as blocks of the
library index artifact (``agent-learning.persona-library.v1``); raw data
lives under ``coverage/`` — coverage is never a standalone kind.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from fi.simulate.simulation.models import (
    CoverageDeclaration,
    Scenario,
)

SCENARIO_COVERAGE_AXES = (
    "intents", "personas", "perturbations",
    "tool_obligations", "delegation_obligations",
)
COVERAGE_FORBIDDEN_HEADLINE_KEYS = ("library_size", "scenario_count")

_AXIS_TO_FIELD = {
    "intents": "intents",
    "personas": "personas",
    "perturbations": "perturbations",
    "tool_obligations": "tool_obligations",
    "delegation_obligations": "delegation_obligations",
}


def _declared_values(scenario: Scenario, axis: str) -> List[str]:
    declaration = scenario.coverage
    values: List[str] = []
    if declaration is not None:
        values.extend(getattr(declaration, _AXIS_TO_FIELD[axis], []) or [])
    if axis == "tool_obligations" and scenario.constraints is not None:
        # obligations are DERIVED from the declared surface (Decision 10):
        # each declared tool yields an exercise obligation.
        values.extend(f"allow:{tool}" for tool in scenario.constraints.declared_tools)
    return values


def expand_scenarios(
    base: Scenario,
    axes: Mapping[str, Sequence[str]],
    k: int = 2,
) -> List[Scenario]:
    """k-way covering-array expansion over typed axis values.

    Deterministic (sorted axes/values, greedy first-fit cover); each child is
    stamped ``parent_version=base.version``, inherits ``kind``, and gets a
    content-address ``version`` (ARCH §2d). Pure itertools."""
    axis_names = sorted(axes)
    if not axis_names:
        return []
    k = max(1, min(int(k), len(axis_names)))
    axis_values = {name: sorted(str(v) for v in axes[name]) for name in axis_names}

    required: List[Tuple[Tuple[str, str], ...]] = []
    for combo in itertools.combinations(axis_names, k):
        for values in itertools.product(*(axis_values[name] for name in combo)):
            required.append(tuple(zip(combo, values)))
    uncovered = set(required)

    children: List[Scenario] = []
    for values in itertools.product(*(axis_values[name] for name in axis_names)):
        if not uncovered:
            break
        assignment = tuple(zip(axis_names, values))
        cells = {
            tuple(sorted(pair_combo))
            for pair_combo in itertools.combinations(assignment, k)
        }
        newly = {cell for cell in uncovered if tuple(sorted(cell)) in cells}
        if not newly:
            continue
        uncovered -= newly
        suffix = "/".join(f"{name}={value}" for name, value in assignment)
        declaration = CoverageDeclaration(
            intents=[dict(assignment).get("intents")] if "intents" in axis_names else [],
            personas=[dict(assignment).get("personas")] if "personas" in axis_names else [],
            perturbations=[dict(assignment).get("perturbations")] if "perturbations" in axis_names else [],
            tool_obligations=[dict(assignment).get("tool_obligations")] if "tool_obligations" in axis_names else [],
            delegation_obligations=[dict(assignment).get("delegation_obligations")] if "delegation_obligations" in axis_names else [],
        )
        child = Scenario(
            **{
                **base.model_dump(exclude_none=True, exclude={"version", "parent_version", "coverage"}),
                "name": f"{base.name}::{suffix}",
                "coverage": declaration.model_dump(),
                "parent_version": base.version,
                "version": None,
            }
        )
        children.append(child)
    return children


def coverage_report(
    scenarios: Sequence[Scenario],
    *,
    axes: Sequence[str] = SCENARIO_COVERAGE_AXES,
) -> Dict[str, Any]:
    """Obligation coverage per axis: declared (anywhere in the set) vs
    covered (declared on a TYPED scenario — kind=None legacy rows declare but
    cannot exercise an obligation)."""
    per_axis: Dict[str, Any] = {}
    declared_total = 0
    covered_total = 0
    uncovered_cells: List[str] = []
    for axis in axes:
        declared = sorted({
            value for scenario in scenarios for value in _declared_values(scenario, axis)
        })
        covered = sorted({
            value for scenario in scenarios if scenario.kind is not None
            for value in _declared_values(scenario, axis)
        })
        uncovered = sorted(set(declared) - set(covered))
        per_axis[axis] = {
            "declared": len(declared),
            "covered": len(covered),
            "uncovered": uncovered,
        }
        declared_total += len(declared)
        covered_total += len(covered)
        uncovered_cells.extend(f"{axis}:{value}" for value in uncovered)
    report = {
        "obligation_coverage": {
            "declared": declared_total,
            "covered": covered_total,
            "rate": round(covered_total / declared_total, 6) if declared_total else 1.0,
            "per_axis": per_axis,
            "uncovered": uncovered_cells,
        },
        # residual_uncovered is filled by residual_uncovered_estimate when an
        # axis grid is available; reported here as the structural headline.
        "residual_uncovered": {
            "rate": round(1.0 - (covered_total / declared_total), 6) if declared_total else 0.0,
            "method": "declared_obligations",
        },
        "metadata": {
            # demoted by contract — never the headline (doctrine #3)
            "library_size": len(scenarios),
        },
    }
    for key in COVERAGE_FORBIDDEN_HEADLINE_KEYS:
        assert key not in report, "forbidden headline key leaked to top level"
    return report


def residual_uncovered_estimate(
    scenarios: Sequence[Scenario],
    axes: Mapping[str, Sequence[str]],
    *,
    budget: int = 64,
    steps: int = 4,
) -> Dict[str, Any]:
    """Budgeted SafeAudit-style enumerator over not-covered k=2-way cells.

    Deterministically enumerates uncovered pairwise cells, samples ``budget``
    candidates across ``steps`` increments, and reports the discovery curve
    (`uncovered_found / sampled` per step) + the plateau curve."""
    axis_names = sorted(axes)
    axis_values = {name: sorted(str(v) for v in axes[name]) for name in axis_names}
    all_cells: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    for first, second in itertools.combinations(axis_names, 2):
        for a, b in itertools.product(axis_values[first], axis_values[second]):
            all_cells.append(((first, a), (second, b)))

    covered_pairs = set()
    for scenario in scenarios:
        values_by_axis = {
            axis: set(_declared_values(scenario, axis)) for axis in axis_names
        }
        for first, second in itertools.combinations(axis_names, 2):
            for a in values_by_axis.get(first, ()):  # noqa: B007
                for b in values_by_axis.get(second, ()):
                    covered_pairs.add(((first, str(a)), (second, str(b))))

    budget = max(1, int(budget))
    steps = max(1, int(steps))
    per_step = max(1, budget // steps)
    sampled = 0
    found = 0
    step_rows: List[Dict[str, Any]] = []
    plateau_curve: List[float] = []
    candidates = list(all_cells)  # deterministic sorted-product order
    for step in range(steps):
        chunk = candidates[sampled:sampled + per_step]
        if not chunk:
            chunk = []
        step_found = sum(1 for cell in chunk if cell not in covered_pairs)
        sampled += len(chunk)
        found += step_found
        rate = round(found / sampled, 6) if sampled else 0.0
        step_rows.append({
            "step": step + 1,
            "sampled": len(chunk),
            "uncovered_found": step_found,
            "rate": rate,
        })
        plateau_curve.append(rate)
    plateau_reached = (
        len(plateau_curve) >= 2
        and abs(plateau_curve[-1] - plateau_curve[-2]) < 1e-9
    )
    return {
        "rate": plateau_curve[-1] if plateau_curve else 0.0,
        "method": "budgeted_enumerator",
        "budget_declared": budget,
        "budget_used": sampled,
        "steps": step_rows,
        "plateau_curve": plateau_curve,
        "plateau_reached": plateau_reached,
        "cells_enumerated": len(all_cells),
        "bound": "estimate" if plateau_reached else "lower",
    }


def synthesize_next_scenario(
    scenarios: Sequence[Scenario],
    axes: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    """Coverage-guided generation (the AgentAssay move): the spec for the
    weakest cell — lowest coverage, ties broken lexicographically."""
    axis_names = sorted(axes)
    axis_values = {name: sorted(str(v) for v in axes[name]) for name in axis_names}
    coverage_count: Dict[Tuple[str, str], int] = {
        (axis, value): 0 for axis in axis_names for value in axis_values[axis]
    }
    for scenario in scenarios:
        for axis in axis_names:
            for value in _declared_values(scenario, axis):
                if (axis, str(value)) in coverage_count:
                    coverage_count[(axis, str(value))] += 1
    weakest = min(coverage_count.items(), key=lambda item: (item[1], item[0]))
    (axis, value), rows = weakest
    return {
        "target_cell": {"axis": axis, "value": value, "rows": rows},
        "spec": {
            "kind": "task",
            "coverage": {_AXIS_TO_FIELD[axis]: [value]},
            "name": f"synth::{axis}={value}",
        },
    }


__all__ = [
    "COVERAGE_FORBIDDEN_HEADLINE_KEYS",
    "SCENARIO_COVERAGE_AXES",
    "coverage_report",
    "expand_scenarios",
    "residual_uncovered_estimate",
    "synthesize_next_scenario",
]
