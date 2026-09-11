"""Voice improvement loop example (Phase 9A, the voice_loopback_readiness gate).

Runs ENTIRELY offline — zero network, zero API keys, zero lanes. ``run(output_path)``
returns the evidence payload the gate audits for the voice-loop half:

  * a multi-objective voice objective compiles (the §4.2 menu + the Goodhart guard);
  * a single-timing objective is rejected (the constructed negative);
  * a whole voice-agent search space (the §4.5 families — NOT prompt-only);
  * the loop-vs-no-loop A/B at equal budget;
  * the voice_sublayer attribution on a weak cell (V1_VOICE_FAILURE_SUBLAYERS).

The 13D Practice Loop is reused on ``world.kind=voice_telephony``; NO new
optimizer is invented (9A-D5). The Goodhart guard is the unedited loss.py
enforcement — "There is no override."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import voice_loop

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURES = EXAMPLE_DIR / "voice_loopback_fixture"
IMPROVEMENT_KIND = "agent-learning.voice-improvement.v1"

_SEED = 1142


def _objective(*, terms=None) -> dict[str, Any]:
    terms = terms or [
        {"eval": "task_success", "weight": 1.0, "direction": "maximize"},
        {"eval": "tool_argument_correctness", "weight": 0.8, "direction": "maximize"},
        {"eval": "barge_in_latency", "weight": 0.4, "direction": "minimize"},
        {"eval": "ttfb", "weight": 0.4, "direction": "minimize"},
        {"eval": "wer_delta", "weight": 0.6, "direction": "minimize"},
        {"eval": "selectivity", "weight": 0.5, "direction": "maximize"},
        {"eval": "codec_survival", "weight": 0.7, "direction": "maximize"},
        {"eval": "perturbation_robustness", "weight": 0.5, "direction": "minimize"},
    ]
    return {
        "source": "declared",
        "evals": terms,
        "guards": {
            "sentinel_rows": [{"id": "no_pii_leak"}, {"id": "no_repetition"}],
            "canary_evals": [{"eval": "repetition_canary"}],
            "min_guard_count": 2,
        },
    }


def _search_space() -> dict[str, Any]:
    return {
        "voice.id": ["alloy", "shimmer"],
        "voice.tts.rate": [0.9, 1.0, 1.1],
        "agent.first_message": ["Hi, how can I help?", "Thanks for calling."],
        "voice.endpointing.threshold": [200, 400],
        "voice.barge_in.policy": ["eager", "polite"],
        "agent.instructions": ["Be concise.", "Confirm every value."],
        "agent.tools.routing": ["strict", "flexible"],
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_path).expanduser() if output_path is not None else None

    compiled = voice_loop.compile_voice_objective(_objective())

    single_timing_rejected = False
    try:
        voice_loop.compile_voice_objective(
            _objective(terms=[{"eval": "ttfb", "weight": 1.0, "direction": "minimize"}])
        )
    except voice_loop.VoiceLossCompositionError:
        single_timing_rejected = True

    ab_spec = json.loads((FIXTURES / "ab/toy_space.json").read_text(encoding="utf-8"))
    budget = int(ab_spec["eval_budget_per_arm"])
    arms = {}
    for arm in ("loop_on", "loop_off"):
        manifest = voice_loop.build_voice_practice_loop_manifest(
            name=f"{ab_spec['name']}-{arm}",
            base_agent={"model": "gpt-4o", "voice": {"id": "alloy"}},
            search_space=_search_space(),
            objective=_objective(),
            eval_budget=budget,
            seed=_SEED,
        )
        arms[arm] = {
            "eval_budget": manifest["practice"]["eval_budget"],
            "world_kind": manifest["practice"]["simulation"]["inline"]["world"]["kind"],
        }

    manifest = voice_loop.build_voice_practice_loop_manifest(
        name="voice-improvement",
        base_agent={"model": "gpt-4o", "voice": {"id": "alloy"}},
        search_space=_search_space(),
        objective=_objective(),
        eval_budget=budget,
        seed=_SEED,
    )

    # the voice_sublayer attribution on weak cells (V1_VOICE_FAILURE_SUBLAYERS)
    sublayers = {
        "selectivity weak": voice_loop.attribute_voice_sublayer(
            failure_layer="agent_behavior", signal="selectivity weak"
        ),
        "tool_argument mishear": voice_loop.attribute_voice_sublayer(
            failure_layer="agent_behavior", signal="tool_argument mishear"
        ),
        "codec_survival died": voice_loop.attribute_voice_sublayer(
            failure_layer="provider", signal="codec_survival died"
        ),
    }

    payload: dict[str, Any] = {
        "kind": IMPROVEMENT_KIND,
        "channel": "voice",
        "seed": _SEED,
        "world_kind": manifest["practice"]["simulation"]["inline"]["world"]["kind"],
        "multi_objective_compiles": len(compiled["evals"]) >= 2
        and any(
            t["eval"] in voice_loop.V1_VOICE_LOSS_NON_TIMING_QUALITY_TERMS
            for t in compiled["evals"]
        ),
        "single_timing_rejected": single_timing_rejected,
        "guard_min_count": compiled["guards"]["min_guard_count"],
        "search_space_paths": sorted(manifest["practice"]["search_space"]),
        "search_space_is_whole_agent": all(
            p in manifest["practice"]["search_space"]
            for p in ("voice.id", "voice.tts.rate", "voice.endpointing.threshold")
        ),
        "ab_arms": arms,
        "ab_equal_budget": arms["loop_on"]["eval_budget"]
        == arms["loop_off"]["eval_budget"]
        == budget,
        "voice_sublayers": sublayers,
        "term_refs": list(voice_loop.V1_VOICE_LOSS_TERM_REFS),
        "failure_sublayers": list(voice_loop.V1_VOICE_FAILURE_SUBLAYERS),
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
