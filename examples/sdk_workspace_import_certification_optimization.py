from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_WORKSPACE_IMPORT_CERTIFICATION_KEY"
EXAMPLE_DIR = Path(__file__).resolve().parent


def _targets() -> list[dict[str, Any]]:
    return [
        {
            "id": "langgraph_factory",
            "framework": "langgraph",
            "module": "framework_shims",
            "attribute": "build_langgraph_agent",
            "callable": True,
            "invoke": True,
            "signals": ["factory", "workspace", "shim"],
        },
        {
            "id": "langchain_factory",
            "framework": "langchain",
            "module": "framework_shims",
            "attribute": "build_langchain_agent",
            "callable": True,
            "invoke": True,
            "signals": ["factory", "workspace", "shim"],
        },
        {
            "id": "pipecat_factory",
            "framework": "pipecat",
            "module": "framework_shims",
            "attribute": "build_pipecat_pipeline",
            "callable": True,
            "invoke": True,
            "signals": ["factory", "voice", "workspace", "shim"],
        },
    ]


def build_manifest() -> dict[str, Any]:
    if str(EXAMPLE_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_DIR))

    return optimize.build_workspace_import_certification_optimization_manifest(
        name="sdk-workspace-import-certification-optimization",
        workspace_path=EXAMPLE_DIR,
        required_env=[REQUIRED_ENV],
        repository_url="https://github.com/future-agi/agent-learning-kit",
        commit_sha="local-example-worktree",
        framework="langgraph",
        targets=_targets(),
        target={
            "name": "local-framework-shim-workspace",
            "provider": "futureagi",
            "repository": "examples/framework_shims.py",
            "modalities": ["chat", "voice"],
        },
        adapter={
            "name": "workspace-import-certification-adapter",
            "version": "2026-06",
            "runtime": "python",
        },
        observability={
            "logs": ["workspace_probe", "framework_import_probe"],
            "metrics": [
                "workspace_run_quality",
                "framework_import_quality",
            ],
            "events": ["workspace_import_certified"],
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
        target_metadata={"cookbook": "sdk-workspace-import-certification"},
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    manifest = build_manifest()
    manifest_path = (
        Path(output_path).expanduser().with_suffix(".manifest.json")
        if output_path is not None
        else Path(__file__).with_suffix(".json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    result = optimize.optimize_manifest(
        manifest,
        manifest_path=manifest_path,
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
