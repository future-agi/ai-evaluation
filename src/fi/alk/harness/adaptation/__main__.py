"""python -m fi.alk.harness.adaptation run|hosted (`environment-releases.md` §7.5).

    run     --environment DIR [--extension DIR] --suite DIR (--profile FILE | --job FILE)
            --output DIR --postgres-dsn DSN [--answers FILE] [--drafts FILE] [--console]
    hosted  --job /work/job.json --releases /work/releases --output /work/authoring/adaptation

Exit codes: 0 completed, 2 failed with result.json written, anything else a crash.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..releases import BindingMode
from .askers import Asker, ConsoleAsker, ScriptedAsker
from .drafting import Drafter, FileDrafter, LlmDrafter
from .errors import AdaptationFailure
from .pipeline import Adaptation, AdaptationInputs


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{what} {path}: {exc}") from exc


def drafter_factory(drafts: Any) -> Any:
    def make(mode: BindingMode) -> Drafter:
        if drafts is not None:
            return FileDrafter(drafts)
        return LlmDrafter(mode=mode.value)

    return make


def failed_before_start(output: Path, code: str, message: str) -> int:
    """A failure found before the pipeline exists still leaves its result.json."""
    from .pipeline import RESULT, RESULT_SCHEMA, _write_json

    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / RESULT,
        {
            "schema_version": RESULT_SCHEMA,
            "status": "failed",
            "failure": {"code": code, "message": message},
            "environment": [],
            "base_suite": None,
            "suite": None,
            "binding": None,
            "alignment": None,
            "answers": {},
            "proposals": [],
        },
    )
    print(f"adaptation failed: {code}: {message}", file=sys.stderr)
    return 2


async def adapt(inputs: AdaptationInputs, asker: Asker) -> int:
    try:
        return await Adaptation(inputs, asker).run()
    finally:
        await asker.close()


def _run(args: argparse.Namespace) -> int:
    from ..job import HarnessJob
    from .profile import profile_from_file, profile_from_job

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty")
    if args.profile:
        body = _read_json(Path(args.profile), "profile")

        def load_profile() -> Any:
            return profile_from_file(body)

    else:
        job = HarnessJob.model_validate_json(Path(args.job).read_text(encoding="utf-8"))

        def load_profile() -> Any:
            return profile_from_job(job, credentials=os.environ)

    answers = _read_json(Path(args.answers), "answers") if args.answers else {}
    drafts = _read_json(Path(args.drafts), "drafts") if args.drafts else None
    asker: Asker = ConsoleAsker() if args.console else ScriptedAsker(answers)
    inputs = AdaptationInputs(
        environments=[
            Path(args.environment),
            *([Path(args.extension)] if args.extension else []),
        ],
        suite_root=Path(args.suite),
        output=output,
        postgres_dsn=args.postgres_dsn,
        load_profile=load_profile,
        drafter_for=drafter_factory(drafts),
    )
    code = asyncio.run(adapt(inputs, asker))
    print((output / "result.json").read_text(encoding="utf-8"))
    return code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fi.alk.harness.adaptation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="adapt a suite to an agent locally")
    run.add_argument("--environment", required=True)
    run.add_argument("--extension")
    run.add_argument("--suite", required=True)
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--profile", help="a saved inspect_provider_target result")
    target.add_argument(
        "--job", help="a hosted job; the provider key comes from the environment"
    )
    run.add_argument("--output", required=True)
    run.add_argument(
        "--postgres-dsn",
        default=os.environ.get("ALK_RELEASE_POSTGRES_DSN"),
        required=not os.environ.get("ALK_RELEASE_POSTGRES_DSN"),
    )
    run.add_argument("--answers", help="question id -> value, proposal id -> bool")
    run.add_argument(
        "--drafts", help="operator-supplied binding, trajectories and proposals"
    )
    run.add_argument("--console", action="store_true", help="answer interactively")
    hosted = commands.add_parser("hosted", help="adapt inside the hosted sandbox")
    hosted.add_argument("--job", required=True)
    hosted.add_argument("--releases", required=True)
    hosted.add_argument("--output", required=True)
    hosted.add_argument("--capabilities", help="conversation capability file")
    hosted.add_argument(
        "--postgres-dsn", help="use this server instead of starting a scratch cluster"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    from .hosted import run_hosted

    try:
        return run_hosted(args)
    except AdaptationFailure as exc:
        return failed_before_start(Path(args.output), exc.code, exc.message)


if __name__ == "__main__":
    raise SystemExit(main())
