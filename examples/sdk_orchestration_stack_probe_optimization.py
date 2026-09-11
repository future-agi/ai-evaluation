from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


def _orchestration_example() -> Any:
    example_path = Path(__file__).with_name("sdk_orchestration_optimization.py")
    spec = importlib.util.spec_from_file_location(
        "sdk_orchestration_optimization",
        example_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_probe_optimization() -> dict[str, Any]:
    orchestration_example = _orchestration_example()
    return optimize.optimize_orchestration_stack_probe(
        name="sdk-orchestration-stack-probe-optimization",
        stack_candidates=[
            orchestration_example.weak_stack(),
            orchestration_example.strong_stack(),
        ],
        agent_candidates=[
            orchestration_example.weak_agent(),
            orchestration_example.strong_agent(),
        ],
        metadata={"cookbook": "sdk-orchestration-stack-probe-optimization"},
    )


def build_manifest() -> dict[str, Any]:
    orchestration_example = _orchestration_example()
    return optimize.build_orchestration_run_manifest_from_probe_optimization(
        build_probe_optimization(),
        name="sdk-orchestration-stack-probe-promotion-run",
        evaluation_config=orchestration_example.evaluation_config(),
        metadata={"cookbook": "sdk-orchestration-stack-probe-optimization"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    simulate.write_manifest_file(build_manifest(), manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts") / "sdk-orchestration-stack-probe-optimization.json"
    )
    run(destination)
