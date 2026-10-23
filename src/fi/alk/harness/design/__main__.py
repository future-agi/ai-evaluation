"""python -m fi.alk.harness.design run|hosted

    run  --workspace DIR --name NAME (--requirements TEXT | --requirements-file FILE)
         [--template-environment DIR --template-suite DIR] [--modality chat|voice]
         --output DIR --postgres-dsn DSN
         [--answers auto|console] [--question-window SECONDS] [--park-at N]

Designs an environment locally with the configured harness model. The person is played by
`--answers`: `auto` approves the proposal and leaves every other choice to the builder, `console`
asks on the terminal. A question unanswered within `--question-window`, or the `--park-at`th one
asked, parks the design; running again on the same workspace re-asks it and resumes.

`hosted --job /work/job.json --releases /work/releases --workspace /work/authoring` is the
sandbox's control process; the conversation's builder does the design (see `hosted.py`).

Exit codes: 0 sealed, 2 failed or stopped (result.json says why), 3 parked (`run` only; a hosted
design parks as a failure with the code `design_waiting`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..backends import resolve
from .tools import (
    APPROVE,
    PARKED,
    PENDING_QUESTION,
    RESULT,
    settle_proposal,
)
from .workspace import load_design, open_design

_NUDGES = 3
_ROUNDS = 6
_PARKED_EXIT = 3


def _labels(options: list[Any]) -> list[str]:
    return [
        str(option.get("label") or option.get("id") or "")
        if isinstance(option, dict)
        else str(option)
        for option in options
    ]


class Person:
    """Whoever answers the design's questions in a local run."""

    def __init__(
        self, workspace: Path, *, mode: str, window: float, park_at: int
    ) -> None:
        self.workspace = workspace
        self.mode = mode
        self.window = window
        self.park_at = park_at
        self.asked = 0
        self.parked = False

    async def answer(self, question: str, options: list[Any]) -> str:
        self.asked += 1
        pending = self.workspace / PENDING_QUESTION
        pending.write_text(
            json.dumps({"question": question, "options": options}, indent=2),
            encoding="utf-8",
        )
        print(f"\n[question {self.asked}] {question}", file=sys.stderr)
        if options:
            print(f"[options] {' | '.join(_labels(options))}", file=sys.stderr)
        if self.asked == self.park_at:
            self.parked = True
            print("[parked]", file=sys.stderr)
            return PARKED
        if self.mode == "console":
            try:
                answer = await asyncio.wait_for(
                    asyncio.to_thread(input, "> "), timeout=self.window
                )
            except TimeoutError:
                self.parked = True
                print("[parked: no answer in time]", file=sys.stderr)
                return PARKED
        else:
            ids = [option.get("id") for option in options if isinstance(option, dict)]
            answer = (
                APPROVE
                if APPROVE in ids
                else _labels(options)[0]
                if options
                else "Use your judgement from the requirements; keep it simple, and say what "
                "you chose."
            )
        pending.unlink(missing_ok=True)
        print(f"[answer] {answer}", file=sys.stderr)
        return answer

    async def approve(self, question: str, options: list[dict[str, str]]) -> str:
        return await self.answer(question, options)

    async def ask(self, _tool_name: str, payload: dict[str, Any], _context: Any) -> Any:
        asked = [one for one in payload.get("questions") or [] if isinstance(one, dict)]
        question = str(
            payload.get("question")
            or "\n".join(str(one.get("question") or "") for one in asked)
            or "What should I use?"
        )
        options = (
            payload.get("options") or (asked[0].get("options") if asked else None) or []
        )
        answer = await self.answer(question, list(options))
        if asked and resolve().name == "claude":
            from claude_agent_sdk.types import PermissionResultAllow

            return PermissionResultAllow(
                updated_input={
                    "questions": asked,
                    "answers": {
                        str(one.get("question") or ""): answer for one in asked
                    },
                }
            )
        return {"content": [{"type": "text", "text": answer}]}


def _shown(event: Any) -> None:
    """The session's own words and tool calls, as the run goes."""
    if event.kind in ("text", "tool", "result"):
        print(event.line()[:600], file=sys.stderr, flush=True)


async def _resumed(person: Person) -> str:
    """The question the design parked on, asked again; what to tell the session about it."""
    path = person.workspace / PENDING_QUESTION
    if not path.is_file():
        return ""
    pending = json.loads(path.read_text(encoding="utf-8"))
    answer = await person.answer(
        str(pending["question"]), list(pending.get("options") or [])
    )
    if person.parked:
        return ""
    approved = settle_proposal(person.workspace, answer)
    return (
        "[Resumed from the checkpoint. You had asked:\n"
        f"{pending['question']}\nThe person answered: {answer}"
        + ("\nThe design is approved: build it." if approved else "")
        + "]"
    )


async def _design(args: argparse.Namespace) -> int:
    from .. import spend
    from .proof import ViewProof
    from .session import open_for_designing, unfinished

    workspace, output = Path(args.workspace), Path(args.output)
    requirements = (
        Path(args.requirements_file).read_text(encoding="utf-8")
        if args.requirements_file
        else args.requirements or ""
    )
    template = (
        (Path(args.template_environment), Path(args.template_suite))
        if args.template_environment
        else None
    )
    from .documents import parse_documents, parsed_root

    open_design(
        workspace,
        name=args.name,
        requirements=requirements,
        modality=args.modality,
        template=template,
        documents=parse_documents(Path(args.documents), parsed_root(workspace))
        if args.documents
        else None,
    )
    spend.journal_to(workspace / "cost.json")
    person = Person(
        workspace, mode=args.answers, window=args.question_window, park_at=args.park_at
    )
    message = await _resumed(person) or (
        "Customize the template in the workspace to the person's requirements in your brief."
        if template
        else "Design this environment from the person's requirements in your brief."
    )
    if person.parked:
        return _PARKED_EXIT
    view_proof = ViewProof(workspace, lambda: args.postgres_dsn)
    try:
        for _ in range(_ROUNDS):
            stage, tools = open_for_designing(
                workspace,
                approve=person.approve,
                ask=person.ask,
                postgres_dsn=lambda: args.postgres_dsn,
                output=output,
                view_proof=view_proof,
            )
            async with stage:
                await stage.say(message, on_event=_shown)
                for _ in range(_NUDGES):
                    if tools.done or tools.reopen or person.parked:
                        break
                    await stage.say(
                        unfinished(load_design(workspace).stage), on_event=_shown
                    )
            if tools.done:
                break
            if person.parked:
                return _PARKED_EXIT
            if not tools.reopen:
                break
            message = (
                "[You were opened again with the tools for what you saved: carry on where you "
                "were.]"
            )
    finally:
        view_proof.close()
    result_path = output / RESULT
    if not result_path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failure": {
                        "code": "design_unfinished",
                        "message": "the session ended without finishing",
                    },
                }
            ),
            encoding="utf-8",
        )
    body = result_path.read_text(encoding="utf-8")
    print(body)
    return 0 if json.loads(body).get("status") == "completed" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fi.alk.harness.design")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="design an environment locally")
    run.add_argument("--workspace", required=True)
    run.add_argument("--name", required=True)
    requirements = run.add_mutually_exclusive_group()
    requirements.add_argument("--requirements")
    requirements.add_argument("--requirements-file")
    run.add_argument("--template-environment")
    run.add_argument("--template-suite")
    run.add_argument("--modality", choices=("chat", "voice"), default="chat")
    run.add_argument("--documents", help="a folder of the person's documents")
    run.add_argument("--output", required=True)
    run.add_argument(
        "--postgres-dsn",
        default=os.environ.get("ALK_RELEASE_POSTGRES_DSN"),
        required=not os.environ.get("ALK_RELEASE_POSTGRES_DSN"),
    )
    run.add_argument("--answers", choices=("auto", "console"), default="auto")
    run.add_argument("--question-window", type=float, default=900.0)
    run.add_argument("--park-at", type=int, default=0)
    hosted = commands.add_parser("hosted", help="the sandbox's control process")
    hosted.add_argument("--job", required=True)
    hosted.add_argument("--releases", required=True)
    hosted.add_argument("--workspace", required=True)
    hosted.add_argument("--documents", default="/work/documents")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "hosted":
        from .hosted import run_hosted

        return run_hosted(args)
    if bool(args.template_environment) != bool(args.template_suite):
        raise SystemExit("--template-environment and --template-suite go together")
    return asyncio.run(_design(args))


if __name__ == "__main__":
    raise SystemExit(main())
