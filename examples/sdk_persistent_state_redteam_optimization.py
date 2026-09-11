from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_PERSISTENT_REDTEAM_OPT_KEY"


def build_manifest() -> dict[str, Any]:
    return optimize.build_persistent_state_redteam_optimization_manifest(
        name="sdk-persistent-state-redteam-optimization",
        required_env=[REQUIRED_ENV],
        target_metadata={"cookbook": "sdk-persistent-state-redteam-optimization"},
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    manifest = build_manifest()
    manifest_path = Path(__file__).with_suffix(".json")
    result = optimize.optimize_manifest(manifest, manifest_path=manifest_path)
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
