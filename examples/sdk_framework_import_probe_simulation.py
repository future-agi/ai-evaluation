from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, simulate


REQUIRED_ENV = "AGENT_LEARNING_SDK_FRAMEWORK_IMPORT_PROBE_KEY"
EXAMPLE_DIR = Path(__file__).resolve().parent


def build_manifest() -> dict[str, Any]:
    if str(EXAMPLE_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_DIR))

    return simulate.build_framework_import_run_manifest(
        name="sdk-framework-import-probe-simulation",
        required_env=[REQUIRED_ENV],
        framework="langgraph",
        targets=[
            {
                "id": "langgraph_factory",
                "framework": "langgraph",
                "module": "framework_shims",
                "attribute": "build_langgraph_agent",
                "callable": True,
                "invoke": True,
                "signals": ["factory", "shim"],
            },
            {
                "id": "langchain_factory",
                "framework": "langchain",
                "module": "framework_shims",
                "attribute": "build_langchain_agent",
                "callable": True,
                "invoke": True,
                "signals": ["factory", "shim"],
            },
            {
                "id": "pipecat_factory",
                "framework": "pipecat",
                "module": "framework_shims",
                "attribute": "build_pipecat_pipeline",
                "callable": True,
                "invoke": True,
                "signals": ["factory", "voice", "shim"],
            },
        ],
        target={
            "name": "local-framework-shim-byo-agent",
            "provider": "futureagi",
            "repository": "examples/framework_shims.py",
            "modalities": ["chat", "voice"],
        },
        adapter={
            "name": "runtime-import-probe-adapter",
            "version": "2026-06",
            "runtime": "python",
        },
        observability={
            "logs": ["import_probe"],
            "events": ["module_import", "runtime_call"],
        },
        artifacts=[
            {
                "id": "framework-shims-source",
                "type": "probe_suite",
                "path": str(EXAMPLE_DIR / "framework_shims.py"),
                "signals": ["artifact", "probe_suite", "runtime_import"],
            }
        ],
        required_frameworks=["langgraph", "langchain", "pipecat"],
        required_export_types=["probe_suite"],
        required_signals=[
            "framework_import",
            "runtime_import",
            "python_import",
            "module_import",
            "callable",
            "runtime_call",
            "target",
            "adapter",
            "observability",
            "artifact",
        ],
        metadata={"cookbook": "sdk-framework-import-probe-simulation"},
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
