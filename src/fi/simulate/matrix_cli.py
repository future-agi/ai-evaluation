"""``python -m fi.simulate.matrix_cli`` — run a provider-matrix manifest.

The matrix manifest references a base SDK simulation manifest and a list
of provider × channel legs (see ``simulation.matrix.MatrixLeg``). Each
non-skipped leg is executed by delegating to the existing SDK ``run``
command surface via ``fi.simulate.manifest.run_manifest``. Skipped legs
emit ``TestCaseStatus.UNSUPPORTED`` rows explicitly.

The CLI is intentionally thin: it exists so pipelines and acceptance
runs can execute the matrix without also depending on the wider
``agent-learn simulate`` argparse tree. Programmatic callers should use
``fi.simulate.simulation.matrix.run_matrix`` directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fi.simulate.manifest import ManifestError, ManifestRunOptions, run_manifest
from fi.simulate.simulation.matrix import (
    MatrixLeg,
    MatrixLegResult,
    run_matrix,
)

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_base(manifest_dir: Path, matrix: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    base = matrix.get("base")
    if isinstance(base, str):
        base_path = (manifest_dir / base).resolve()
        return _load_json(base_path), base_path
    if isinstance(base, dict):
        return dict(base), manifest_dir / "matrix-inline.json"
    raise ManifestError("matrix manifest requires 'base' as path or object")


async def _default_run_leg(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    leg: MatrixLeg,
    max_concurrent_calls: int,
) -> Any:
    del max_concurrent_calls  # concurrency is engine-side today
    return await run_manifest(
        manifest=manifest,
        manifest_path=manifest_path,
        options=ManifestRunOptions(name=f"matrix:{leg.label}", no_eval=False),
    )


def _report_status(report: Any) -> str | None:
    if not isinstance(report, dict):
        return None
    status = report.get("status")
    return str(status) if status is not None else None


def _leg_succeeded(result: MatrixLegResult) -> bool:
    return result.skipped or (
        result.error is None and _report_status(result.report) == "passed"
    )


def _summary(results: list[MatrixLegResult]) -> dict[str, Any]:
    return {
        "schema_version": "agent-learning.matrix.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "legs": [
            {
                "leg": item.leg.label,
                "provider": item.leg.provider,
                "channel": item.leg.channel,
                "status": (
                    "unsupported"
                    if item.skipped
                    else "error"
                    if item.error is not None
                    else _report_status(item.report)
                    or "unknown"
                ),
                "started_at": item.started_at.isoformat(),
                "ended_at": item.ended_at.isoformat() if item.ended_at else None,
                "skipped": item.skipped,
                "skip_reason": item.leg.skip_reason,
                "error": item.error,
                "skipped_cases": item.skipped_cases,
                "manifest_name": item.manifest.get("name"),
                "summary": (
                    item.report.get("summary")
                    if isinstance(item.report, dict)
                    else None
                ),
            }
            for item in results
        ],
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_legs(raw: Any) -> list[MatrixLeg]:
    if not isinstance(raw, list):
        raise ManifestError("matrix manifest 'legs' must be a list")
    try:
        return [MatrixLeg.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ManifestError(f"invalid matrix leg: {exc}") from exc


async def _run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    matrix = _load_json(manifest_path)
    base, base_path = _resolve_base(manifest_path.parent, matrix)
    legs = _parse_legs(matrix.get("legs"))
    results = await run_matrix(
        base,
        legs,
        manifest_path=base_path,
        max_concurrent_calls=int(matrix.get("max_concurrent_calls", 1)),
        run_leg=_default_run_leg,
    )
    payload = _summary(results)
    if args.output:
        _write(Path(args.output).expanduser().resolve(), payload)
    if not args.quiet:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(_leg_succeeded(result) for result in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fi.simulate.matrix_cli",
        description="Run a provider × channel matrix against an SDK base manifest.",
    )
    parser.add_argument("manifest")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except ManifestError as exc:
        print(f"matrix: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
