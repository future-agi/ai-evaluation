"""Child-process entrypoint for a locally sandboxed harness job."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .executor import HarnessExecutor
from .job import HarnessJob


async def _run(
    job_path: Path,
    source: Path,
    output: Path,
    status_path: Path,
    adjustments_path: Path | None,
) -> int:
    job = HarnessJob.model_validate_json(job_path.read_text(encoding="utf-8"))
    status = await HarnessExecutor().run(
        job, source=source, output=output, adjustments_path=adjustments_path
    )
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(status.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(status_path)
    return 0 if status.stage.value == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alk-harness-sandbox-worker")
    parser.add_argument("job", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--adjustments", type=Path)
    args = parser.parse_args(argv)
    return asyncio.run(
        _run(args.job, args.source, args.output, args.status, args.adjustments)
    )


if __name__ == "__main__":
    raise SystemExit(main())
