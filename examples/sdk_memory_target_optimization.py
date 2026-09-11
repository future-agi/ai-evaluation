from __future__ import annotations

import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize, simulate


REQUIRED_ENV = "AGENT_LEARNING_SDK_MEMORY_TARGET_OPTIMIZATION_KEY"
TARGET_PATH = "simulation.environments.1.data.operations"


def _memory_example() -> Any:
    example_path = Path(__file__).with_name("sdk_memory_optimization.py")
    spec = importlib.util.spec_from_file_location(
        "sdk_memory_optimization",
        example_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operation_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    operations = deepcopy(
        _memory_example().strong_candidate()["agent_memory_lineage"]["operations"]
    )
    return [], operations


def _base_config() -> dict[str, Any]:
    memory_example = _memory_example()
    memory = deepcopy(memory_example.strong_candidate())
    weak_operations, _ = _operation_candidates()
    memory["agent_memory_lineage"]["operations"] = weak_operations
    base_manifest = simulate.build_memory_layer_run_manifest(
        name="sdk-memory-target-optimization-base",
        required_env=[REQUIRED_ENV],
        memory=memory,
        evaluation_config=memory_example.evaluation_config(),
        threshold=0.98,
        min_turns=1,
        metadata={"cookbook": "sdk-memory-target-optimization"},
    )
    return {
        "agent": base_manifest["agent"],
        "simulation": base_manifest["simulation"],
    }


def _evaluation_config() -> dict[str, Any]:
    return _memory_example().evaluation_config()


def _target_candidates() -> dict[str, list[list[dict[str, Any]]]]:
    weak_operations, operations = _operation_candidates()
    return {TARGET_PATH: [weak_operations, operations]}


def build_manifest() -> dict[str, Any]:
    return optimize.build_target_optimization_manifest(
        name="sdk-memory-target-optimization",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(),
        evaluation_config=_evaluation_config(),
        target_candidates=_target_candidates(),
        layers=["memory", "retrieval", "policy", "evaluator"],
        min_turns=1,
        max_turns=2,
        threshold=0.98,
        target_metadata={
            "cookbook": "sdk-memory-target-optimization",
            "optimized_surface": "agent_memory_lineage_operations",
        },
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    result = optimize.optimize_target(
        name="sdk-memory-target-optimization",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(),
        evaluation_config=_evaluation_config(),
        target_candidates=_target_candidates(),
        layers=["memory", "retrieval", "policy", "evaluator"],
        min_turns=1,
        max_turns=2,
        threshold=0.98,
        target_metadata={
            "cookbook": "sdk-memory-target-optimization",
            "optimized_surface": "agent_memory_lineage_operations",
        },
        manifest_path=Path(__file__).with_suffix(".json"),
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
