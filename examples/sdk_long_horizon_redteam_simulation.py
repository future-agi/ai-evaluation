from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, redteam


REQUIRED_ENV = "AGENT_LEARNING_SDK_LONG_HORIZON_REDTEAM_KEY"


def build_manifest() -> dict[str, Any]:
    return redteam.build_long_horizon_redteam_manifest(
        name="sdk-long-horizon-redteam",
        required_env=[REQUIRED_ENV],
        target={
            "agent": "sdk-long-horizon-agent",
            "environment": "local-stateful-ci",
        },
        redteam={"campaign_name": "sdk-long-horizon-redteam-campaign"},
        threshold=0.9,
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    manifest = build_manifest()
    manifest_path = (
        Path(output_path).with_suffix(".manifest.json")
        if output_path
        else Path(__file__).with_suffix(".json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = asyncio.run(redteam.redteam_manifest_file(manifest_path))
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
