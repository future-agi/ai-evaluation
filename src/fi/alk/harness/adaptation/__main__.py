"""python -m fi.alk.harness.adaptation run|hosted (`environment-releases.md` §7.5).

    run     --environment DIR [--extension DIR] --suite DIR (--profile FILE | --job FILE)
            --output DIR --postgres-dsn DSN [--drafts FILE] [--approve | --console]
    hosted  --job /work/job.json --releases /work/releases --output /work/authoring/adaptation

`run` binds locally: with `--drafts` (binding, trajectories, extension_candidates, conflicts) no
model is involved; without it the builder's binding session drafts. `hosted` is the sandbox's
control process; the conversation's builder does the binding (see `hosted.py`).

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
from .errors import AdaptationFailure
from .hosted import failed_before_start
from .workbench import RESULT, Approve, Workbench

_NUDGES = 3


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{what} {path}: {exc}") from exc


def _approver(args: argparse.Namespace) -> Approve:
    async def approve(question: str, options: list[str]) -> str:
        if args.console:
            print(f"\n{question}", file=sys.stderr)
            return await asyncio.to_thread(input, f"[{'/'.join(options)}] > ")
        print(f"\n{question}\n> {'yes' if args.approve else 'no'}", file=sys.stderr)
        return "yes" if args.approve else "no"

    return approve


async def _from_drafts(workbench: Workbench, drafts: Any, approve: Approve) -> None:
    if workbench.mode is BindingMode.TOOL_ROUTED:
        accepted, said = workbench.submit_binding(drafts)
        if not accepted:
            return workbench.fail("binding_invalid", said)
        passed, said = await asyncio.to_thread(workbench.run_proof)
        if not passed:
            return workbench.fail("binding_unproved", said)
    accepted, said = workbench.report_conflicts(drafts.get("conflicts") or [])
    if not accepted:
        return workbench.fail("conflicts_invalid", said)
    finished, said = await workbench.finish(approve)
    if not finished:
        workbench.fail(workbench.stop_code(), said)


async def _with_session(workbench: Workbench, approve: Approve) -> None:
    from ..builder import open_for_adapting

    stage = open_for_adapting(workbench, approve=approve)
    async with stage:
        await stage.say(
            "Bind this agent to the environment: write the binding, prove it, report where "
            "the agent's instructions conflict with the policy, then finish."
        )
        for _ in range(_NUDGES):
            if workbench.done:
                return
            await stage.say(
                "[The adaptation is not finished:\n- "
                + "\n- ".join(
                    workbench.unfinished() or ["finish_adaptation was not called"]
                )
                + "\nCarry on, and call finish_adaptation, or stop_adaptation with the reason.]"
            )
    if not workbench.done:
        workbench.fail("adaptation_unfinished", "the session ended without finishing")


def _run(args: argparse.Namespace) -> int:
    from ..job import HarnessJob
    from .profile import profile_from_file, profile_from_job

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty")
    try:
        if args.profile:
            profile = profile_from_file(_read_json(Path(args.profile), "profile"))
        else:
            job = HarnessJob.model_validate_json(
                Path(args.job).read_text(encoding="utf-8")
            )
            profile = profile_from_job(job, credentials=os.environ)
        workbench = Workbench(
            environments=[
                Path(args.environment),
                *([Path(args.extension)] if args.extension else []),
            ],
            suite_root=Path(args.suite),
            output=output,
            postgres_dsn=args.postgres_dsn,
            profile=profile,
        )
    except AdaptationFailure as exc:
        return failed_before_start(output, exc.code, exc.message)
    approve = _approver(args)
    try:
        if args.drafts:
            asyncio.run(
                _from_drafts(
                    workbench, _read_json(Path(args.drafts), "drafts"), approve
                )
            )
        else:
            asyncio.run(_with_session(workbench, approve))
    except AdaptationFailure as exc:
        workbench.fail(exc.code, exc.message)
    body = (output / RESULT).read_text(encoding="utf-8")
    print(body)
    return 0 if json.loads(body)["status"] == "completed" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fi.alk.harness.adaptation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="bind an agent to a release locally")
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
    run.add_argument(
        "--drafts",
        help="operator-supplied binding, trajectories, extension_candidates and conflicts",
    )
    answer = run.add_mutually_exclusive_group()
    answer.add_argument("--approve", action="store_true", help="approve the binding")
    answer.add_argument("--console", action="store_true", help="answer interactively")
    hosted = commands.add_parser("hosted", help="the sandbox's control process")
    hosted.add_argument("--job", required=True)
    hosted.add_argument("--releases", required=True)
    hosted.add_argument("--output", required=True)
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
        return failed_before_start(
            Path(args.output), exc.code, exc.message, hosted=True
        )


if __name__ == "__main__":
    raise SystemExit(main())
