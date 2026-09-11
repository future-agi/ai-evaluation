"""Unit 11 (BBG U11 / ARCH §2d phase 3) — deficit-targeted drills at measured ZPD.

Generator: 13a_t1_deficit when it lands; v1 fallback studio_perturbation. The
four scaffolds are MANIFEST TRANSFORMS (never engine features), each a pure
function with its own content-hash consequence. ZPD is MEASURED (k seeded
repeats + ICC via live/_stats), never asserted. Cell loss is ONLY ever computed
at intensity 0.0 (unscaffolded).
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .._schema import public_payload
from ..live._contract import DEFAULT_REPEATS, UNSTABLE_ICC_FLOOR
from ..live._stats import icc_and_within_variance
from ._contract import AGENT_LEARNING_PRACTICE_DRILL_KIND, SCAFFOLD_TYPES, ZPD_BAND


def child_seed(seed: int, phase: str, cell_key: str, index: int) -> int:
    """Determinism recipe (ARCH §2d): first 8 bytes of SHA-256 as int."""
    digest = hashlib.sha256(f"{seed}:{phase}:{cell_key}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _hash(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


# --- the four scaffolds as pure manifest transforms ------------------------
def scaffold_world_simplification(sim: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    """Drop a declared transition/precondition from the drill simulation world."""
    out = copy.deepcopy(dict(sim))
    world = out.setdefault("world", {})
    spec = world.setdefault("spec", {})
    drop = params.get("drop_transition")
    if drop is not None:
        transitions = [t for t in spec.get("transitions", []) if t.get("id") != drop]
        spec["transitions"] = transitions
    spec["_scaffold"] = {"world_simplification": dict(params)}
    return out


def scaffold_hint_tool(sim: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    """Add a status-volunteering tool variant to world.tools."""
    out = copy.deepcopy(dict(sim))
    world = out.setdefault("world", {})
    tools = list(world.get("tools") or [])
    tools.append({"name": params.get("tool_name", "hint_status"), "mock": {"level": "static_fixture"},
                  "_scaffold": "hint_tool"})
    world["tools"] = tools
    return out


def scaffold_worked_example(sim: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    """Inject a captured competent trajectory as context (rides the agent block
    context field / the binding oracle_solver slot, A10)."""
    out = copy.deepcopy(dict(sim))
    scenarios = out.setdefault("scenarios", [])
    if scenarios:
        scenarios[0].setdefault("oracle_solver", {
            "kind": "trajectory",
            "source": params.get("source", "captured://worked_example"),
            "content_hash": params.get("content_hash", "sha256:worked"),
        })
    out.setdefault("metadata", {})["_scaffold_worked_example"] = True
    return out


def scaffold_relaxed_success(sim: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    """Relax verification.threshold / success predicate."""
    out = copy.deepcopy(dict(sim))
    verification = dict(out.get("verification") or {})
    verification["threshold"] = float(params.get("threshold", 0.4))
    out["verification"] = verification
    return out


_SCAFFOLD_OPS: Dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], dict]] = {
    "world_simplification": scaffold_world_simplification,
    "hint_tool": scaffold_hint_tool,
    "worked_example": scaffold_worked_example,
    "relaxed_success": scaffold_relaxed_success,
}


def apply_scaffold(sim: Mapping[str, Any], scaffold_type: str, params: Mapping[str, Any]) -> dict:
    if scaffold_type not in SCAFFOLD_TYPES:
        raise ValueError(f"scaffold type {scaffold_type!r} not in {SCAFFOLD_TYPES}")
    return _SCAFFOLD_OPS[scaffold_type](sim, params)


def _zpd_verdict(unscaffolded_rate: float, scaffolded_rates: Mapping[str, float],
                 band: Sequence[float], icc: float, icc_floor: float) -> str:
    low, high = float(band[0]), float(band[1])
    if icc < icc_floor:
        return "unstable"
    if low <= unscaffolded_rate <= high:
        return "in_band"
    if unscaffolded_rate < low:
        # passes under >= 1 scaffold ⇒ vygotsky_form, else below_band.
        if any(rate > unscaffolded_rate for rate in scaffolded_rates.values()):
            return "vygotsky_form"
        return "below_band"
    return "above_band"


def drill(
    deficit: Mapping[str, Any],
    drill_simulation: Mapping[str, Any],
    *,
    seed: int,
    round_no: int,
    repeat_scorer: Callable[[Mapping[str, Any], int], float],
    scaffolds: Optional[Sequence[Mapping[str, Any]]] = None,
    fade_intensities: Sequence[float] = (1.0, 0.5, 0.0),
    k: int = DEFAULT_REPEATS,
    icc_floor: float = UNSTABLE_ICC_FLOOR,
    band: Sequence[float] = ZPD_BAND,
    admission: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Run a deficit-targeted drill. ``repeat_scorer(sim, seed) -> float`` scores
    one episode (1.0 pass / 0.0 fail) — injected for determinism. ZPD is measured
    over k seeded repeats; cell loss is ONLY computed at intensity 0.0."""
    if not fade_intensities or float(fade_intensities[-1]) != 0.0:
        raise ValueError("scaffold_fade.intensities MUST end at 0.0 (unscaffolded)")

    cell = deficit.get("cell") or {}
    cell_key = json.dumps(cell, sort_keys=True, default=str)

    # Admission before any run: a drill failing admission never runs (zero budget).
    admission = dict(admission or {})
    admitted = admission.get("admissible", True)
    if not admitted:
        return public_payload({
            "kind": AGENT_LEARNING_PRACTICE_DRILL_KIND,
            "target_cell": cell,
            "admission": admission,
            "unscaffolded_exit": False,
            "zpd_measurement": {"verdict": "below_band", "k": 0, "seeds": [], "icc": 0.0,
                                "unscaffolded_pass_rate": 0.0, "scaffolded_pass_rates": {}, "band": list(band)},
        }, kind=AGENT_LEARNING_PRACTICE_DRILL_KIND)

    # Unscaffolded ZPD measurement: k seeded repeats with derived seeds.
    seeds = [child_seed(seed, "drill", cell_key, i) for i in range(k)]
    scores = np.array([[repeat_scorer(drill_simulation, s) for s in seeds]], dtype=float)
    unscaffolded_rate = round(float(scores.mean()), 6)
    icc, _ = icc_and_within_variance(scores)
    icc = round(float(icc), 6)

    # Scaffolded pass rates (each scaffold a different simulation by construction).
    scaffolded_rates: Dict[str, float] = {}
    scaffold_records: List[dict] = []
    for spec in scaffolds or []:
        stype = spec.get("type")
        scaffolded = apply_scaffold(drill_simulation, stype, spec.get("params") or {})
        srate = round(
            float(np.mean([repeat_scorer(scaffolded, child_seed(seed, f"drill:{stype}", cell_key, i))
                           for i in range(k)])),
            6,
        )
        scaffolded_rates[stype] = srate
        scaffold_records.append({
            "type": stype, "params": spec.get("params") or {},
            "simulation_hash": _hash(scaffolded),
        })

    verdict = _zpd_verdict(unscaffolded_rate, scaffolded_rates, band, icc, icc_floor)

    record = {
        "kind": AGENT_LEARNING_PRACTICE_DRILL_KIND,
        "target_cell": cell,
        "generator": {"method": "studio_perturbation", "ref": deficit.get("harness_layer")},
        "drill_simulation": {"version": _hash(drill_simulation), "inline": dict(drill_simulation)},
        "admission": admission,
        "scaffolds": scaffold_records,
        "fade": {"intensities": list(fade_intensities), "step_outcomes": []},
        "zpd_measurement": {
            "k": k,
            "seeds": seeds,
            "icc": icc,
            "unscaffolded_pass_rate": unscaffolded_rate,
            "scaffolded_pass_rates": scaffolded_rates,
            "band": list(band),
            "verdict": verdict,
        },
        # unstable ⇒ drill quarantined (zero update budget); not an exit.
        "unscaffolded_exit": verdict == "in_band",
    }
    return public_payload(record, kind=AGENT_LEARNING_PRACTICE_DRILL_KIND)
