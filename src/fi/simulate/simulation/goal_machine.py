"""Unit 2 (BBG U2 / ARCH §1.9 G3) — the executable goal/verification binding.

This module makes the EXISTING ``ScenarioGoal``/``VerificationSpec`` types
executable at three rungs (turn / settle / run). It imports ONLY from
``models.py`` and stdlib (the engine-side one-way rule). The closed vocabularies
declared here are the single canonical home; ``contract.py`` and ``trinity.py``
mirror them, never redeclare them (ARCH §3).

Everything is deterministic: no model calls, no wall-clock, sorted iteration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .models import ScenarioGoal, VerificationSpec

# --- canon (ARCH §3 / §2a; R5/A7 STAGED — the v1 5-kind set is frozen) -------
GOAL_CHECK_KINDS = (
    "state_predicate",
    "world_invariant",
    "world_success_condition",
    "eval_template",
    "keyword_fallback",
)
GOAL_PREDICATE_OPS = ("eq", "ne", "gte", "lte", "contains", "exists")
GOAL_CHECK_RUNGS = ("turn", "settle", "run")


def _resolve_path(state: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    """Dotted-path resolution over a nested mapping. Returns (found, value)."""
    cursor: Any = state
    for part in str(path).split("."):
        if isinstance(cursor, Mapping) and part in cursor:
            cursor = cursor[part]
        else:
            return False, None
    return True, cursor


def _eval_predicate(check: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Evaluate a ``state_predicate`` {path, op, value} over environment_state.
    Missing path ⇒ False except ``exists`` (which reports presence)."""
    path = check.get("path") or check.get("target")
    op = str(check.get("op") or "eq")
    expected = check.get("value")
    found, actual = _resolve_path(state, str(path)) if path is not None else (False, None)
    if op == "exists":
        return bool(found)
    if not found:
        return False
    try:
        if op == "eq":
            return actual == expected
        if op == "ne":
            return actual != expected
        if op == "gte":
            return float(actual) >= float(expected)
        if op == "lte":
            return float(actual) <= float(expected)
        if op == "contains":
            if isinstance(actual, (str, list, tuple, dict)):
                return expected in actual
            return False
    except (TypeError, ValueError):
        return False
    return False


def _world_status_pass(world_status: Mapping[str, Any], name: str, *, condition_type: str) -> Optional[bool]:
    """Bind a world_invariant/world_success_condition check BY NAME against the
    WorldContractEnvironment status the engine already holds in
    ``environment_state['world_contract']`` (invariant_results/success_results).
    Returns the pass bool, or None if no such named condition is present."""
    key = "invariant_results" if condition_type == "world_invariant" else "success_results"
    results = world_status.get(key) or []
    for entry in results:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("name") or entry.get("id") or "") == name:
            return bool(entry.get("pass"))
    return None


def _checks_at_rung(verification: Optional[VerificationSpec], rung: str) -> List[Mapping[str, Any]]:
    if verification is None:
        return []
    out: List[Mapping[str, Any]] = []
    for raw in verification.checks:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("rung") or "turn") == rung:
            out.append(raw)
    return out


def _evaluate_rung(
    goal: ScenarioGoal,
    verification: Optional[VerificationSpec],
    rung: str,
    *,
    environment_state: Mapping[str, Any],
    world_status: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate all checks at ``rung``; resolve state_predicate / world_* checks.
    eval_template / keyword_fallback are rung-run/legacy and skipped here."""
    states_reached: List[str] = []
    check_records: List[Dict[str, Any]] = []
    for raw in sorted(_checks_at_rung(verification, rung), key=lambda c: str(c.get("name") or "")):
        name = str(raw.get("name") or "")
        kind = str(raw.get("kind") or "state_predicate")
        if kind == "state_predicate":
            passed: Optional[bool] = _eval_predicate(raw, environment_state)
        elif kind in ("world_invariant", "world_success_condition"):
            passed = _world_status_pass(world_status, name, condition_type=kind)
        else:
            # eval_template / keyword_fallback are not evaluated at turn/settle.
            continue
        record = {"name": name, "kind": kind, "rung": rung, "passed": bool(passed)}
        check_records.append(record)
        if passed and name and name in (goal.states or []):
            states_reached.append(name)

    stop: Optional[str] = None
    success_state = goal.success_state
    failure_states = set(goal.failure_states or [])
    passed_names = {r["name"] for r in check_records if r["passed"]}
    if success_state and success_state in passed_names:
        stop = "goal_success"
    elif failure_states & passed_names:
        stop = "goal_failure"
    return {"states_reached": states_reached, "stop": stop, "checks": check_records}


def evaluate_turn(
    goal: ScenarioGoal,
    verification: Optional[VerificationSpec],
    *,
    environment_state: Mapping[str, Any],
    world_status: Optional[Mapping[str, Any]] = None,
    messages: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Evaluate all rung-``turn`` checks after each observe."""
    return _evaluate_rung(
        goal, verification, "turn",
        environment_state=environment_state,
        world_status=world_status or environment_state.get("world_contract") or {},
        messages=messages or [],
    )


def evaluate_settle(
    goal: ScenarioGoal,
    verification: Optional[VerificationSpec],
    *,
    environment_state: Mapping[str, Any],
    world_status: Optional[Mapping[str, Any]] = None,
    messages: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Same shape over rung-``settle`` checks at episode end."""
    return _evaluate_rung(
        goal, verification, "settle",
        environment_state=environment_state,
        world_status=world_status or environment_state.get("world_contract") or {},
        messages=messages or [],
    )


def evaluate_run(
    goal: ScenarioGoal,
    verification: Optional[VerificationSpec],
    *,
    environment_state: Optional[Mapping[str, Any]] = None,
    world_status: Optional[Mapping[str, Any]] = None,
    messages: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Rung-``run`` summary consumed later by loss.py — pure data, no eval
    execution here (eval templates run in the existing evaluation lineage; the
    goal machine only *names* them)."""
    names = sorted(
        str(c.get("name") or "")
        for c in _checks_at_rung(verification, "run")
        if isinstance(c, Mapping)
    )
    return {"rung": "run", "named_checks": names, "success_state": goal.success_state}
