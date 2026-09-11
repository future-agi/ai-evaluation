from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, suite


REQUIRED_ENV = "AGENT_LEARNING_SDK_SUITE_OPT_EXAMPLE_KEY"
CHILD_ENV = "AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY"


SEED_JOB = {
    "id": "framework-breadth",
    "command": "run",
    "path": "framework_langchain_manifest.json",
    "name": "sdk-suite-opt-single-framework-seed",
}

TEN_FRAMEWORK_JOB = {
    "id": "framework-breadth",
    "command": "suite",
    "path": "multi_framework_simulation_suite.json",
    "name": "sdk-suite-opt-ten-framework-candidate",
}


def build_suite() -> dict[str, Any]:
    manifest = suite.build_suite_manifest(
        name="sdk-suite-optimization",
        required_env=[REQUIRED_ENV, CHILD_ENV],
        jobs=[SEED_JOB],
        required_capabilities={
            "commands": ["suite"],
            "result_kinds": ["agent-learning.suite.v1"],
            "environment_state_keys": ["framework_runtime"],
            "frameworks": [
                "langchain",
                "langgraph",
                "llamaindex",
                "openai_agents",
                "autogen",
                "crewai",
                "pydantic_ai",
                "pipecat",
                "livekit",
                "custom_refund_orchestrator",
            ],
        },
        metadata={"cookbook": "sdk-suite-optimization"},
    )
    manifest["optimization"] = {
        "threshold": 1.0,
        "target": {
            "name": "sdk-suite-framework-breadth",
            "layers": ["harness", "framework", "world", "evaluator"],
            "base_config": {"jobs": [SEED_JOB]},
            "search_space": {"jobs.0": [SEED_JOB, TEN_FRAMEWORK_JOB]},
            "metadata": {
                "source": "examples/sdk_suite_optimization.py",
                "task_kind": "agent_learning_suite_optimization",
                "cookbook": "sdk-suite-optimization",
            },
        },
        "optimizer": {
            "algorithm": "agent",
            "max_candidates": 3,
            "include_seed": True,
            "auto_diagnose": False,
        },
    }
    return manifest


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)
    os.environ.setdefault(CHILD_ENV, api_key)

    result = suite.optimize_suite(
        build_suite(),
        suite_path=Path(__file__).with_name("sdk_suite_optimization.json"),
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
