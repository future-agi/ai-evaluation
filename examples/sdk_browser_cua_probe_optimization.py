from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


def evaluation_config() -> dict[str, Any]:
    manifest = optimize.build_browser_cua_optimization_manifest(
        name="sdk-browser-cua-probe-evaluation-config",
    )
    return manifest["evaluation"]["agent_report"]["config"]


def build_probe_optimization() -> dict[str, Any]:
    return optimize.optimize_browser_cua_probe(
        name="sdk-browser-cua-probe-optimization",
        metadata={"cookbook": "sdk-browser-cua-probe-optimization"},
    )


def build_manifest() -> dict[str, Any]:
    return optimize.build_browser_cua_run_manifest_from_probe_optimization(
        build_probe_optimization(),
        name="sdk-browser-cua-probe-promotion-run",
        evaluation_config=evaluation_config(),
        metadata={"cookbook": "sdk-browser-cua-probe-optimization"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    simulate.write_manifest_file(build_manifest(), manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
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
        else Path("artifacts") / "sdk-browser-cua-probe-optimization.json"
    )
    run(destination)
