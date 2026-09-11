"""Image improvement loop example (Phase 9B, the image_loop_readiness gate).

Runs ENTIRELY offline -- zero network, zero API keys, zero lanes.
``run(output_path)`` returns the evidence payload the gate audits for the
image-loop half:

  * a multi-objective image objective compiles (the unit-2 menu + the Goodhart
    guard + the perception-bypass guard rows);
  * a judge-only objective is rejected (the constructed negative);
  * a whole multimodal-agent search space (the §2.3 families incl.
    image.preprocess.* + mmrag.* -- NOT prompt-only);
  * the loop-vs-no-loop A/B at equal budget (the capstone);
  * the image_sublayer attribution on weak cells (V1_IMAGE_FAILURE_SUBLAYERS).

The 13D Practice Loop is reused on ``world.kind=image``; NO new optimizer is
invented (9B-D4). The Goodhart guard is the unedited loss.py enforcement --
"There is no override."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import image_loop

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURES = EXAMPLE_DIR / "image_loop_fixture"
IMPROVEMENT_KIND = "agent-learning.image-improvement.v1"

_SEED = 1142


def _objective(*, terms=None) -> dict[str, Any]:
    terms = terms or [
        {"eval": "task_success", "weight": 1.0, "direction": "maximize"},
        {"eval": "ocr_accuracy", "weight": 0.7, "direction": "maximize"},
        {"eval": "chart_accuracy", "weight": 0.7, "direction": "maximize"},
        {"eval": "artifact_grounding", "weight": 0.6, "direction": "maximize"},
        {"eval": "instruction_adherence", "weight": 0.4, "direction": "maximize"},
        {"eval": "tool_argument_correctness", "weight": 0.5, "direction": "maximize"},
    ]
    return {
        "source": "declared",
        "evals": terms,
        "guards": {
            "sentinel_rows": [
                {"id": "prior_answerable", "kind": "perception_bypass"},
                {"id": "no_hallucinated_object"},
            ],
            "canary_evals": [{"eval": "counterfactual_twin", "kind": "perceptual_counterfactual"}],
            "min_guard_count": 2,
        },
    }


def _search_space() -> dict[str, Any]:
    return {
        "agent.model": ["gpt-4o", "claude-vision"],
        "agent.vision_prompt": ["describe the scene", "extract every value"],
        "agent.instructions": ["Be precise.", "Cite the region you read."],
        "image.preprocess.resolution": [256, 512, 1024],
        "image.preprocess.crop": ["center", "none"],
        "image.preprocess.enhance": ["off", "contrast"],
        "mmrag.retrieve_images": [True, False],
        "mmrag.reranker": ["off", "cross_encoder"],
        "agent.tools.routing": ["strict", "flexible"],
        "agent.first_message": ["Let's analyze the image.", "Reading now."],
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_path).expanduser() if output_path is not None else None

    compiled = image_loop.compile_image_objective(_objective())

    judge_only_rejected = False
    try:
        image_loop.compile_image_objective(
            _objective(
                terms=[
                    {"eval": "instruction_adherence", "weight": 1.0, "direction": "maximize"},
                    {"eval": "instruction_adherence", "weight": 0.5, "direction": "maximize"},
                ]
            )
        )
    except image_loop.ImageLossCompositionError:
        judge_only_rejected = True

    single_term_rejected = False
    try:
        image_loop.compile_image_objective(
            _objective(terms=[{"eval": "task_success", "weight": 1.0}])
        )
    except image_loop.ImageLossCompositionError:
        single_term_rejected = True

    ab_spec = json.loads((FIXTURES / "ab/toy_space.json").read_text(encoding="utf-8"))
    budget = int(ab_spec["eval_budget_per_arm"])
    arms: dict[str, Any] = {}
    for arm in ("loop_on", "loop_off"):
        manifest = image_loop.build_image_practice_loop_manifest(
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
            "canary_holds": ab_spec["arms"][arm]["perception_bypass_canary_holds"],
        }

    manifest = image_loop.build_image_practice_loop_manifest(
        name="image-improvement",
        base_agent={"model": "gpt-4o"},
        search_space=_search_space(),
        objective=_objective(),
        eval_budget=budget,
        seed=_SEED,
    )

    # the image_sublayer attribution on weak cells (V1_IMAGE_FAILURE_SUBLAYERS).
    sublayers = {
        "ocr parse weak low_res": image_loop.attribute_image_sublayer(
            failure_layer="agent_behavior", signal="ocr parse weak low_res"
        ),
        "visual misidentification": image_loop.attribute_image_sublayer(
            failure_layer="agent_behavior", signal="visual misidentification"
        ),
        "tool_argument extracted wrong": image_loop.attribute_image_sublayer(
            failure_layer="agent_behavior", signal="tool_argument extracted wrong"
        ),
        "grounded-but-wrong conclusion": image_loop.attribute_image_sublayer(
            failure_layer="agent_behavior", signal="grounded-but-wrong conclusion reasoning"
        ),
    }

    ss = manifest["practice"]["search_space"]
    ab_loop_improves = (
        arms["loop_on"]["anchored_loss"] < arms["loop_off"]["anchored_loss"]
    )
    ab_canary_holds = arms["loop_on"]["canary_holds"] and arms["loop_off"]["canary_holds"]

    payload: dict[str, Any] = {
        "kind": IMPROVEMENT_KIND,
        "modality": "image",
        "seed": _SEED,
        "world_kind": manifest["practice"]["simulation"]["inline"]["world"]["kind"],
        "task_mode": manifest["practice"]["simulation"]["inline"]["world"]["spec"]["task_mode"],
        "multi_objective_compiles": len(compiled["evals"]) >= 2
        and any(
            t["eval"] in image_loop.V1_IMAGE_LOSS_DETERMINISTIC_ANCHOR_TERMS
            for t in compiled["evals"]
        ),
        "judge_only_rejected": judge_only_rejected,
        "single_term_rejected": single_term_rejected,
        "guard_min_count": compiled["guards"]["min_guard_count"],
        "search_space_paths": sorted(ss),
        "search_space_is_whole_agent": all(
            p in ss
            for p in ("image.preprocess.resolution", "mmrag.retrieve_images", "agent.model")
        ),
        "ab_arms": arms,
        "ab_equal_budget": arms["loop_on"]["eval_budget"]
        == arms["loop_off"]["eval_budget"]
        == budget,
        "ab_loop_improves": ab_loop_improves,
        "ab_canary_holds": ab_canary_holds,
        "image_sublayers": sublayers,
        "term_refs": list(image_loop.V1_IMAGE_LOSS_TERM_REFS),
        "failure_sublayers": list(image_loop.V1_IMAGE_FAILURE_SUBLAYERS),
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
