from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize, simulate, suite


REQUIRED_ENV = "AGENT_LEARNING_SDK_ARTIFACT_ACTION_OPTIMIZATION_KEY"


def build_source_manifest() -> dict[str, Any]:
    return simulate.build_framework_certification_run_manifest(
        name="sdk-artifact-action-source",
        framework="langgraph",
        target_framework="openai_agents",
        required_env=[REQUIRED_ENV],
        metadata={"cookbook": "sdk-artifact-action-optimization"},
    )


def build_suite(
    *,
    artifact_path: str | Path,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_dir)
    return optimize.build_artifact_action_optimization_manifest(
        name="sdk-artifact-action-optimization",
        artifact_path=artifact_path,
        action_ids=[
            "report_framework_readiness",
            "rerun_framework_certification",
        ],
        required_env=[REQUIRED_ENV],
        cwd_root=workspace / "action-runs",
        outputs_root=workspace / "action-run-results",
        metadata={"cookbook": "sdk-artifact-action-optimization"},
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    if output_path is None:
        with tempfile.TemporaryDirectory(
            prefix="agent-learning-artifact-action-"
        ) as temp_dir:
            return _run_in_workspace(destination=None, workspace=Path(temp_dir))

    destination = (
        Path(output_path).expanduser()
    )
    workspace = destination.with_suffix("")
    return _run_in_workspace(destination=destination, workspace=workspace)


def _run_in_workspace(
    *,
    destination: Path | None,
    workspace: Path,
) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    artifact_path = workspace / "source-artifact.json"
    source_manifest_path = workspace / "source-manifest.json"
    suite_manifest_path = workspace / "artifact-action-optimization-suite.json"

    simulate.write_manifest_file(build_source_manifest(), source_manifest_path)
    source_artifact = asyncio.run(simulate.run_manifest_file(source_manifest_path))
    artifact_path.write_text(
        json.dumps(source_artifact, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    manifest = build_suite(
        artifact_path=artifact_path,
        workspace_dir=workspace,
    )
    suite.write_suite_file(manifest, suite_manifest_path)
    result = suite.optimize_suite(manifest, suite_path=suite_manifest_path)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(output)
    if output is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
