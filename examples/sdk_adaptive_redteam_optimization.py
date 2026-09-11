from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_ADAPTIVE_REDTEAM_OPT_KEY"


def source_result() -> dict[str, Any]:
    return {
        "kind": "agent-learning.redteam.v1",
        "status": "passed",
        "redteam": {
            "attack_types": ["prompt_injection"],
            "surfaces": ["tool", "memory"],
            "channels": ["chat"],
            "providers": ["local_cli"],
            "frameworks": ["agent_learning_kit"],
        },
        "redteam_strategy": {
            "kind": "redteam_strategy_map",
            "status": "needs_attention",
            "attack_types": ["prompt_injection"],
            "surfaces": ["tool", "memory"],
            "channels": ["chat"],
            "providers": ["local_cli"],
            "frameworks": ["agent_learning_kit"],
            "missing_coverage_cells": ["prompt_injection|memory|chat|local_cli"],
            "missing_executed_cells": ["prompt_injection|memory|chat|local_cli"],
            "adaptive_surface_risk": {
                "status": "needs_attention",
                "blind_spot_surfaces": ["memory"],
                "worst_surface": "memory",
                "adaptive_gap_rate": 1.0,
            },
        },
        "findings": [
            {
                "type": "red_team_campaign_gap",
                "metric": "red_team_campaign_quality",
                "score": 0.0,
                "surface": "memory",
            }
        ],
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_adaptive_redteam_optimization_manifest(
        name="sdk-adaptive-redteam-optimization",
        required_env=[REQUIRED_ENV],
        source_result=source_result(),
        target_metadata={"cookbook": "sdk-adaptive-redteam-optimization"},
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


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
