from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_OPENENV_OPTIMIZATION_KEY"


def _env_keys(required_env: Sequence[str] | None) -> list[str]:
    if required_env is None:
        return [REQUIRED_ENV]
    return [str(key) for key in required_env]


def build_manifest(
    *,
    required_env: Sequence[str] | None = None,
) -> dict[str, Any]:
    return optimize.build_openenv_optimization_manifest(
        name="sdk-openenv-environment-optimization",
        required_env=_env_keys(required_env),
        target_metadata={"cookbook": "sdk-openenv-environment-optimization"},
    )


def run(
    output_path: str | Path | None = None,
    *,
    required_env: Sequence[str] | None = None,
) -> dict[str, Any]:
    env_keys = _env_keys(required_env)
    if env_keys:
        api_key = os.environ.get(env_keys[0])
        if not api_key:
            raise RuntimeError(f"Set {env_keys[0]} before running this example.")
        configure(api_key=api_key)

    result = optimize.optimize_openenv(
        name="sdk-openenv-environment-optimization",
        required_env=env_keys,
        target_metadata={"cookbook": "sdk-openenv-environment-optimization"},
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
