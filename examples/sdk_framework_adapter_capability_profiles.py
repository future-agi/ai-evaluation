from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import simulate


FRAMEWORKS = [
    "langchain",
    "langgraph",
    "openai_agents",
    "livekit",
    "pipecat",
]


def build_profiles() -> dict[str, Any]:
    """Build portable simulate/eval/opt profiles without importing frameworks."""

    matrix = simulate.framework_adapter_contract_matrix(FRAMEWORKS)
    return simulate.framework_adapter_capability_profiles(matrix=matrix)


def build_manifest() -> dict[str, Any]:
    """Build a run manifest that carries the profile bundle as evidence."""

    matrix = simulate.framework_adapter_contract_matrix(FRAMEWORKS)
    return simulate.build_framework_adapter_matrix_run_manifest(
        name="sdk-framework-adapter-capability-profiles",
        frameworks=FRAMEWORKS,
        matrix=matrix,
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    profiles = build_profiles()
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(profiles, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return profiles


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
