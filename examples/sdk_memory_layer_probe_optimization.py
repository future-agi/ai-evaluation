from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


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


def build_probe_optimization() -> dict[str, Any]:
    memory_example = _memory_example()
    return optimize.optimize_memory_layer_probe(
        name="sdk-memory-layer-probe-optimization",
        memory_candidates=[
            memory_example.weak_candidate(),
            memory_example.strong_candidate(),
        ],
        cases=[
            {
                "id": "refund-memory",
                "input": "Recall the current refund policy memory.",
                "required_operations": ["read", "write", "recall"],
            }
        ],
        metadata={"cookbook": "sdk-memory-layer-probe-optimization"},
    )


def build_manifest() -> dict[str, Any]:
    memory_example = _memory_example()
    return optimize.build_memory_run_manifest_from_probe_optimization(
        build_probe_optimization(),
        name="sdk-memory-layer-probe-promotion-run",
        evaluation_config=memory_example.evaluation_config(),
        metadata={"cookbook": "sdk-memory-layer-probe-optimization"},
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
        else Path("artifacts") / "sdk-memory-layer-probe-optimization.json"
    )
    run(destination)
