"""CUA improvement loop example (Phase 9C, the cua_loop_readiness gate).

Runs ENTIRELY offline -- zero network, zero API keys, zero lanes, no real browser,
no VM. ``run(output_path)`` returns the evidence payload the gate audits for the
CUA-loop improvement half:

  * a multi-objective CUA objective compiles (the unit-2 menu + the Goodhart guard
    + the fake/unsafe-completion guard rows);
  * a judge-only objective is rejected (the constructed negative);
  * a whole CUA-agent search space (the §3.3 families incl. agent.grounding.* +
    agent.observe.*/agent.escalation.* + agent.reflection.*/agent.memory.* --
    NOT prompt-only);
  * the loop-vs-no-loop A/B at equal budget (the capstone);
  * the cua_sublayer attribution on weak cells (V1_CUA_FAILURE_SUBLAYERS).

The 13D Practice Loop is reused on ``world.kind=browser``; NO new optimizer is
invented (9C-D4). The Goodhart guard is the unedited loss.py enforcement --
"There is no override."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import cua_loop

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURES = EXAMPLE_DIR / "cua_loop_fixture"
IMPROVEMENT_KIND = "agent-learning.cua-improvement.v1"

_SEED = 1142


def _objective(*, terms=None) -> dict[str, Any]:
    terms = terms or [
        {"eval": "task_success", "weight": 1.0, "direction": "maximize"},
        {"eval": "state_match", "weight": 0.9, "direction": "maximize"},
        {"eval": "grounding_mutation_resilience", "weight": 0.7, "direction": "maximize"},
        {"eval": "action_correctness", "weight": 0.7, "direction": "maximize"},
        {"eval": "step_efficiency", "weight": 0.5, "direction": "maximize"},
        {"eval": "safety_adherence", "weight": 0.8, "direction": "maximize"},
        {"eval": "tool_evidence", "weight": 0.5, "direction": "maximize"},
        {"eval": "trace_coverage", "weight": 0.5, "direction": "maximize"},
    ]
    return {
        "source": "declared",
        "evals": terms,
        "guards": {
            "sentinel_rows": [
                {"id": "fake_completion_sentinel", "kind": "fake_completion"},
                {"id": "no_silent_failure"},
            ],
            "canary_evals": [{"eval": "injected_dom_follow", "kind": "unsafe_completion"}],
            "min_guard_count": 2,
        },
    }


def _search_space() -> dict[str, Any]:
    return {
        "agent.model": ["gpt-4o", "claude"],
        "agent.grounding.mode": ["element-id", "coordinate", "selector"],
        "agent.grounding.selector_fallback": ["on", "off"],
        "agent.observe.channel": ["screenshot", "DOM", "AXTree"],
        "agent.observe.resolution": ["low", "high"],
        "agent.escalation.stuck_monitor": ["on", "off"],
        "agent.escalation.milestone_monitor": ["on", "off"],
        "agent.reflection.postmortems": ["on", "off"],
        "agent.memory.env_knowledge": ["retain", "drop"],
        "agent.tools.routing": ["strict", "flexible"],
        "agent.instructions": ["Verify the post-state.", "Use the fallback selector."],
        "agent.first_message": ["Refreshing the snapshot.", "Inspecting the mutation pack."],
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_path).expanduser() if output_path is not None else None

    compiled = cua_loop.compile_cua_objective(_objective())

    judge_only_rejected = False
    try:
        cua_loop.compile_cua_objective(
            _objective(terms=[
                {"eval": "completion_judge", "weight": 1.0, "direction": "maximize"},
                {"eval": "completion_judge", "weight": 0.5, "direction": "maximize"},
            ])
        )
    except cua_loop.CuaLossCompositionError:
        judge_only_rejected = True

    single_term_rejected = False
    try:
        cua_loop.compile_cua_objective(_objective(terms=[{"eval": "task_success", "weight": 1.0}]))
    except cua_loop.CuaLossCompositionError:
        single_term_rejected = True

    missing_anchor_rejected = False
    try:
        cua_loop.compile_cua_objective(
            _objective(terms=[
                {"eval": "action_correctness", "weight": 1.0},
                {"eval": "step_efficiency", "weight": 0.5},
            ])
        )
    except cua_loop.CuaLossCompositionError:
        missing_anchor_rejected = True

    ab_spec = json.loads((FIXTURES / "ab/toy_space.json").read_text(encoding="utf-8"))
    budget = int(ab_spec["eval_budget_per_arm"])
    arms: dict[str, Any] = {}
    for arm in ("loop_on", "loop_off"):
        manifest = cua_loop.build_cua_practice_loop_manifest(
            name=f"{ab_spec['name']}-{arm}",
            base_agent={"model": "gpt-4o"},
            search_space=_search_space(),
            objective=_objective(),
            eval_budget=budget,
            seed=_SEED,
        )
        arms[arm] = {
            "eval_budget": manifest["practice"]["eval_budget"],
            "world_kind": manifest["practice"]["simulation"]["inline"]["world"]["kind"],
            "anchored_loss": ab_spec["arms"][arm]["anchored_loss"],
            "fake_completion_canary_holds": ab_spec["arms"][arm]["fake_completion_canary_holds"],
            "unsafe_completion_canary_holds": ab_spec["arms"][arm]["unsafe_completion_canary_holds"],
        }

    manifest = cua_loop.build_cua_practice_loop_manifest(
        name="cua-improvement",
        base_agent={"model": "gpt-4o"},
        search_space=_search_space(),
        objective=_objective(),
        eval_budget=budget,
        seed=_SEED,
    )

    # the cua_sublayer attribution on weak cells (V1_CUA_FAILURE_SUBLAYERS).
    sublayers = {
        "stale screenshot didn't refresh": cua_loop.attribute_cua_sublayer(
            failure_layer="agent_behavior", signal="stale screenshot didn't refresh; missed an observed change"
        ),
        "selector drifted mis-clicked": cua_loop.attribute_cua_sublayer(
            failure_layer="agent_behavior", signal="selector drifted, mis-clicked; coordinate off"
        ),
        "looped on the same step": cua_loop.attribute_cua_sublayer(
            failure_layer="agent_behavior", signal="looped on the same step; touched injected banner"
        ),
        "wrong plan bad memory": cua_loop.attribute_cua_sublayer(
            failure_layer="agent_behavior", signal="right perception, wrong plan; bad memory of prior steps"
        ),
    }

    ss = manifest["practice"]["search_space"]
    ab_loop_improves = arms["loop_on"]["anchored_loss"] < arms["loop_off"]["anchored_loss"]
    ab_canaries_hold = (
        arms["loop_on"]["fake_completion_canary_holds"]
        and arms["loop_on"]["unsafe_completion_canary_holds"]
        and arms["loop_off"]["fake_completion_canary_holds"]
        and arms["loop_off"]["unsafe_completion_canary_holds"]
    )

    # a desktop objective with the narrower grounding_step_accuracy anchor compiles.
    desktop_compiles = False
    try:
        cua_loop.compile_cua_objective(
            {
                "source": "declared",
                "evals": [
                    {"eval": "grounding_step_accuracy", "weight": 1.0},
                    {"eval": "action_correctness", "weight": 0.6},
                ],
                "guards": {
                    "sentinel_rows": [{"id": "x", "kind": "fake_completion"}],
                    "min_guard_count": 1,
                },
            },
            cua_surface="desktop",
        )
        desktop_compiles = True
    except cua_loop.CuaLossCompositionError:
        desktop_compiles = False

    payload: dict[str, Any] = {
        "kind": IMPROVEMENT_KIND,
        "modality": "cua",
        "seed": _SEED,
        "world_kind": manifest["practice"]["simulation"]["inline"]["world"]["kind"],
        "cua_surface": manifest["practice"]["simulation"]["inline"]["world"]["spec"]["cua_surface"],
        "multi_objective_compiles": len(compiled["evals"]) >= 2
        and any(
            t["eval"] in cua_loop.V1_CUA_LOSS_DETERMINISTIC_ANCHOR_TERMS
            for t in compiled["evals"]
        ),
        "judge_only_rejected": judge_only_rejected,
        "single_term_rejected": single_term_rejected,
        "missing_anchor_rejected": missing_anchor_rejected,
        "desktop_objective_compiles": desktop_compiles,
        "guard_min_count": compiled["guards"]["min_guard_count"],
        "search_space_paths": sorted(ss),
        "search_space_is_whole_agent": all(
            p in ss
            for p in (
                "agent.grounding.mode", "agent.observe.channel",
                "agent.reflection.postmortems", "agent.memory.env_knowledge", "agent.model"
            )
        ),
        "ab_arms": arms,
        "ab_equal_budget": arms["loop_on"]["eval_budget"]
        == arms["loop_off"]["eval_budget"]
        == budget,
        "ab_loop_improves": ab_loop_improves,
        "ab_canaries_hold": ab_canaries_hold,
        "cua_sublayers": sublayers,
        "term_refs": list(cua_loop.V1_CUA_LOSS_TERM_REFS),
        "failure_sublayers": list(cua_loop.V1_CUA_FAILURE_SUBLAYERS),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return payload


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result = run(destination)
    if destination is None:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
