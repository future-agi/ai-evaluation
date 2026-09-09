"""Deciding a judged sub-goal against the world the session left behind.

A judged sub-goal is one nothing observable settles, so a model reaches the verdict. Until now
nothing did: judged sub-goals carried a placeholder check returning None, which the scheduler read
as "held", so every one passed before anything looked. This runs in the sandbox at the end of the
session, while the world is still alive, so the judge reads real state rather than a snapshot.

Nothing here is modality-specific. It reasons over the world's tables and the actions the agent
took, which a voice call, a typed conversation and a browser the agent drives all leave behind in
the same shape, so the wording stays neutral rather than naming a call.

A judge that cannot decide returns held None: the unjudged path the platform already skips, because
a judge that failed to run is not evidence against the agent.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from .backends import SessionSpec, tool, tool_server
from .config import chosen_model
from .session import Stage
from .tools import schema

JUDGE_MODEL_ALIAS = "ALK_JUDGE_MODEL"
_LIMIT = 600

_INSTRUCTIONS = """
You decide one claim about a session that already happened, against the world it left behind.

Look before you answer: read the actions you were given, inspect the tables the claim touches, query
for a specific row when the claim is about one. The world is the run's final state. Its SQL dialect
is PostgreSQL, not SQLite. Use inspect_world to discover tables; do not query sqlite_master or make
up a schema.

Then call decide, once. `passed` true when the claim holds, false when it does not, and an
explanation citing what you saw: a value, a row, an action and its arguments. An explanation that
only restates the claim is not a verdict. If you genuinely cannot tell, pass undecided true rather
than guess. Judge only the claim you were given.
""".strip()


def judge_model() -> str:
    """The judge's model. Set ALK_JUDGE_MODEL to run it on something other than the harness model."""
    return os.environ.get(JUDGE_MODEL_ALIAS, "").strip() or chosen_model()


def _short(value: object) -> object:
    if isinstance(value, str) and len(value) > _LIMIT:
        return value[:_LIMIT] + f"... [{len(value)} chars]"
    if isinstance(value, dict):
        return {key: _short(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_short(item) for item in value[:20]]
    return value


def _dump(value: object) -> str:
    return json.dumps(_short(value), default=str, indent=1)


async def judge(goal: Any, world: Any, calls: Sequence[Any]) -> tuple[bool | None, str]:
    """Decide one judged sub-goal: (passed, explanation). Never raises."""
    verdict: dict[str, tuple[bool | None, str]] = {}

    @tool(
        "inspect_world",
        "The world's tables: without a name, every table and its row count; with one, its rows.",
        schema({"table": str}, []),
    )
    async def inspect_world(args: dict[str, Any]) -> dict[str, Any]:
        table = str(args.get("table") or "").strip()
        state = world.state(table or None)
        if table:
            return _say(_dump(state))
        return _say(_dump({name: len(rows or []) for name, rows in state.items()}))

    @tool(
        "query_world",
        "Run one read-only PostgreSQL query for a specific row. Use inspect_world for schema discovery.",
        schema({"sql": str}, ["sql"]),
    )
    async def query_world(args: dict[str, Any]) -> dict[str, Any]:
        sql = str(args.get("sql") or "")
        try:
            return _say(_dump(world.query(sql)))
        except Exception as exc:  # noqa: BLE001 - bad model SQL is feedback, not a stage crash
            return _say(
                _dump(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "dialect": "postgresql",
                        "recovery": "Use inspect_world to discover real tables, then retry once.",
                    }
                ),
                error=True,
            )

    @tool(
        "decide",
        "Commit to the verdict, once, after looking.",
        schema({"passed": bool, "explanation": str, "undecided": bool}, ["explanation"]),
    )
    async def decide(args: dict[str, Any]) -> dict[str, Any]:
        explanation = str(args.get("explanation") or "").strip()
        if not explanation:
            return _say("a verdict needs an explanation naming what you saw", error=True)
        held = None if bool(args.get("undecided")) else bool(args.get("passed"))
        verdict["it"] = (held, explanation)
        return _say("recorded")

    spec = SessionSpec(
        system_prompt=_INSTRUCTIONS,
        servers={"world": tool_server(name="world", tools=[inspect_world, query_world, decide])},
        max_turns=12,
        model=judge_model(),
    )
    prompt = (
        f"Claim {getattr(goal, 'name', '')!r}.\n"
        f"What it means: {getattr(goal, 'what', '') or '(none written)'}\n"
        f"Why a model must decide it: {getattr(goal, 'judged', '')}\n\n"
        f"Actions the agent took:\n{_dump([_call(c) for c in calls])}\n\n"
        "Inspect the world as needed, then call decide."
    )
    try:
        async with Stage(spec, name="judge-sub-goals") as stage:
            await stage.say(prompt)
    except Exception as exc:  # noqa: BLE001 - a judge that could not run is not a failed agent
        return None, f"the judge could not run: {type(exc).__name__}: {exc}"
    return verdict.get("it", (None, "the judge finished without a verdict"))


def _call(call: Any) -> dict[str, Any]:
    return {
        "name": getattr(call, "name", ""),
        "arguments": getattr(call, "arguments", None) or {},
        "result": getattr(call, "result", None),
        "ok": bool(getattr(call, "ok", False)),
        "refused": bool(getattr(call, "refused", False)),
        "error": str(getattr(call, "error", "") or ""),
    }


def _say(text: str, *, error: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if error:
        body["is_error"] = True
    return body
