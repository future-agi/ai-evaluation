from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, simulate


REQUIRED_ENV = "AGENT_LEARNING_SDK_OPTIMIZER_GOVERNANCE_SIMULATION_KEY"


def build_manifest() -> dict[str, Any]:
    return simulate.build_optimizer_governance_run_manifest(
        name="sdk-optimizer-governance-simulation",
        required_env=[REQUIRED_ENV],
        threshold=0.95,
        metadata={"cookbook": "sdk-optimizer-governance-simulation"},
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
