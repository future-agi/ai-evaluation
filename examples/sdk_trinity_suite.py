from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, suite


REQUIRED_ENV = "AGENT_LEARNING_SDK_TRINITY_SUITE_KEY"

EXAMPLE_ENV_KEYS = [
    "AGENT_LEARNING_RUN_EXAMPLE_KEY",
    "AGENT_LEARNING_REDTEAM_EXAMPLE_KEY",
    "AGENT_LEARNING_WORLD_FRAMEWORK_OPT_EXAMPLE_KEY",
    "AGENT_LEARNING_SDK_WORLD_MODEL_KEY",
]


def build_suite() -> dict[str, Any]:
    return suite.build_trinity_suite_manifest(
        name="sdk-trinity-suite",
        required_env=[REQUIRED_ENV],
        run_path="run_manifest.json",
        eval_path="eval_suite.json",
        artifact_eval_path="artifact_task_eval_suite.json",
        artifact_report_path="fixtures/task_artifacts/refund_task_run.json",
        artifact_eval_config_path="artifact_task_eval_config.json",
        artifact_optimization_path="artifact_task_optimization_suite.json",
        redteam_path="redteam_manifest.json",
        eval_optimization_path="eval_suite_optimization.json",
        optimization_path="world_framework_memory_optimization.json",
        world_model_optimization_path="world_model_optimization.json",
        max_candidates=4,
        metadata={"cookbook": "sdk-trinity-suite"},
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)
    for key in EXAMPLE_ENV_KEYS:
        os.environ.setdefault(key, api_key)

    result = suite.run_suite(
        build_suite(),
        suite_path=Path(__file__).with_name("sdk_trinity_suite.json"),
    )
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
