from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize, simulate


REQUIRED_ENV = "AGENT_LEARNING_SDK_REDTEAM_ATTACK_EVOLUTION_KEY"

ATTACKS = [
    "prompt_injection",
    "indirect_prompt_injection",
    "memory_poisoning",
]
SURFACES = ["tool", "retrieval", "memory"]
OPERATORS = ["semantic_mutation", "trajectory_splice", "surface_transfer"]
COVERAGE_AXES = ["attack_type", "surface", "operator", "verifier"]


def build_manifest() -> dict[str, Any]:
    return optimize.build_redteam_attack_evolution_optimization_manifest(
        name="sdk-redteam-attack-evolution-optimization",
        required_env=[REQUIRED_ENV],
        attacks=ATTACKS,
        surfaces=SURFACES,
        operators=OPERATORS,
        coverage_axes=COVERAGE_AXES,
        target_metadata={"cookbook": "sdk-redteam-attack-evolution-optimization"},
    )


def evaluation_config() -> dict[str, Any]:
    return build_manifest()["evaluation"]["agent_report"]["config"]


def weak_report() -> dict[str, Any]:
    evolution = simulate.normalize_red_team_attack_evolution_manifest(
        {
            "name": "weak-seed-only-attack-evolution",
            "target": {"agent": "support-agent", "environment": "local"},
            "seed_attacks": [
                {
                    "id": "seed_prompt_injection",
                    "attack_type": "prompt_injection",
                    "surface": "tool",
                    "operator": "seed",
                    "signals": ["seed_attack"],
                }
            ],
            "mutation_rounds": [
                {
                    "id": "round_1",
                    "score": 0.2,
                    "mutations": [
                        {
                            "id": "round_1_prompt_semantic",
                            "attack_type": "prompt_injection",
                            "surface": "tool",
                            "operator": "semantic_mutation",
                            "status": "proposed",
                            "success": False,
                        }
                    ],
                }
            ],
            "mutation_operators": ["semantic_mutation"],
            "coverage_axes": ["attack_type", "surface"],
            "required_attack_types": ATTACKS,
            "required_surfaces": SURFACES,
            "required_operators": OPERATORS,
        }
    )
    return _report_for_evolution(
        evolution,
        content="Seed-only mutation proposal; no counterexample replay evidence yet.",
    )


def verified_report() -> dict[str, Any]:
    manifest = build_manifest()
    verified_candidate = manifest["optimization"]["target"]["search_space"][
        "simulation.environments"
    ][-1][0]
    evolution = simulate.normalize_red_team_attack_evolution_manifest(
        verified_candidate["data"]
    )
    return _report_for_evolution(
        evolution,
        content=(
            "Attack evolution closes semantic mutation, trajectory splice, "
            "surface transfer, feedback, counterexample minimization, and "
            "replayable regression verifier gates."
        ),
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    manifest = build_manifest()
    result = optimize.optimize_manifest(
        manifest,
        manifest_path=Path(__file__).with_suffix(".json"),
    )
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


def _report_for_evolution(evolution: dict[str, Any], *, content: str) -> dict[str, Any]:
    return {
        "results": [
            {
                "messages": [
                    {"role": "user", "content": "Run attack evolution."},
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": "evolution_status",
                                "name": "red_team_attack_evolution_status",
                                "arguments": {},
                            },
                            {
                                "id": "evolution_gaps",
                                "name": "list_red_team_evolution_gaps",
                                "arguments": {},
                            },
                        ],
                    },
                ],
                "artifacts": [
                    {
                        "type": "trace",
                        "data": evolution,
                        "metadata": {"kind": "red_team_attack_evolution"},
                    }
                ],
                "metadata": {
                    "task_description": "Evaluate attack-evolution red-team proof.",
                    "expected_result": "Attack evolution evidence is complete.",
                    "environment_state": {"red_team_attack_evolution": evolution},
                },
            }
        ]
    }


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
