from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_COMPONENT_OPTIMIZATION_KEY"


def build_manifest() -> dict[str, Any]:
    return optimize.build_component_optimization_manifest(
        name="sdk-component-optimization",
        required_env=[REQUIRED_ENV],
        observed_report=(
            "The agent has missing tool evidence, wrong tool routing, a "
            "framework trace gap, memory retrieval failure, missing memory "
            "lineage attribution, orchestration flow failure, and a world "
            "contract violation."
        ),
        target_metadata={"cookbook": "sdk-component-optimization"},
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    result = optimize.optimize_component(
        name="sdk-component-optimization",
        required_env=[REQUIRED_ENV],
        observed_report=(
            "The agent has missing tool evidence, wrong tool routing, a "
            "framework trace gap, memory retrieval failure, missing memory "
            "lineage attribution, orchestration flow failure, and a world "
            "contract violation."
        ),
        target_metadata={"cookbook": "sdk-component-optimization"},
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
