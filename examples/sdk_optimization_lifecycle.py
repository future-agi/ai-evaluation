from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fi.alk import configure, suite


REQUIRED_ENV = "AGENT_LEARNING_SDK_OPTIMIZATION_LIFECYCLE_KEY"


def build_manifest() -> dict[str, Any]:
    task_world = _task_world_module()
    manifest = task_world.build_manifest()
    manifest["name"] = "sdk-optimization-lifecycle"
    manifest["required_env"] = [REQUIRED_ENV]
    return manifest


def write_workspace(directory: str | Path) -> Path:
    root = Path(directory).expanduser().resolve()
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests / "optimize.json"
    manifest_path.write_text(
        json.dumps(build_manifest(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def build_plan(workspace: str | Path | None = None) -> dict[str, Any]:
    root = Path(workspace or ".").expanduser().resolve()
    return suite.build_optimization_lifecycle_plan(
        optimize_manifest_path=root / "manifests" / "optimize.json",
        workspace_dir=root,
        name="sdk-optimization-lifecycle",
        required_env=[REQUIRED_ENV],
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    output = Path(output_path).expanduser() if output_path is not None else None
    workspace = (
        output.parent / "sdk-optimization-lifecycle-workspace"
        if output is not None
        else Path(tempfile.gettempdir()) / "agent-learning-sdk-optimization-lifecycle"
    )
    manifest_path = write_workspace(workspace)
    result = suite.run_optimization_lifecycle_file(
        manifest_path,
        workspace_dir=workspace,
        name="sdk-optimization-lifecycle",
        required_env=[REQUIRED_ENV],
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return result


def _task_world_module() -> Any:
    path = Path(__file__).with_name("sdk_task_world_optimization.py")
    spec = importlib.util.spec_from_file_location("sdk_task_world_optimization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load sdk_task_world_optimization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
