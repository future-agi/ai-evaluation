from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import simulate


def build_manifest() -> dict[str, Any]:
    return simulate.build_openenv_run_manifest(
        name="sdk-openenv-environment-simulation",
        metadata={"cookbook": "sdk-openenv-environment-simulation"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["openenv_environment_manifest"] = manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/sdk-openenv-environment-simulation.json")
    )
    payload = run(destination)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
