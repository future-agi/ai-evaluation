from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, simulate


REQUIRED_ENV = "AGENT_LEARNING_SDK_MULTI_AGENT_SIMULATION_KEY"


def _multi_agent_optimization_example() -> Any:
    example_path = Path(__file__).with_name("sdk_multi_agent_optimization.py")
    spec = importlib.util.spec_from_file_location(
        "sdk_multi_agent_optimization",
        example_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_manifest() -> dict[str, Any]:
    multi_agent_example = _multi_agent_optimization_example()
    return simulate.build_multi_agent_coordination_run_manifest(
        name="sdk-multi-agent-coordination-simulation",
        required_env=[REQUIRED_ENV],
        participants=multi_agent_example.participants(),
        agent=multi_agent_example.strong_agent(),
        room=multi_agent_example.strong_room(),
        evaluation_config=multi_agent_example.evaluation_config(),
        threshold=0.95,
        metadata={"cookbook": "sdk-multi-agent-coordination-simulation"},
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    manifest_path = (
        Path(output_path).expanduser().with_suffix(".manifest.json")
        if output_path is not None
        else Path(__file__).with_suffix(".json")
    )
    simulate.write_manifest_file(build_manifest(), manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
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
