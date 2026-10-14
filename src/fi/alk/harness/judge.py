"""Deciding a judged sub-goal against the world the session left behind.

A judged sub-goal is one nothing observable settles, so a model reaches the verdict. Until now
nothing did: judged sub-goals carried a placeholder check returning None, which the scheduler read
as "held", so every one passed before anything looked. This runs in the sandbox at the end of the
session, while the world is still alive, so the judge reads real state rather than a snapshot.

Nothing here is modality-specific. It reasons over the world's tables, the actions the agent
took and what was said, which a voice call, a typed conversation and a browser the agent drives all
leave behind in the same shape, so the wording stays neutral rather than naming a call.

The explanation speaks only about the agent; judging details go to the log. A judge that cannot
decide after one retry returns held None.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Sequence

from .backends import SessionSpec, tool, tool_server
from .catalogue import criteria_text
from .config import chosen_model
from .session import Stage
from .tools import schema
from .world.observation import ObservationWorld

logger = logging.getLogger(__name__)

JUDGE_MODEL_ALIAS = "ALK_JUDGE_MODEL"
_LIMIT = 600
_TRANSCRIPT_LIMIT = 24000

_INSTRUCTIONS = """
You are a judge. You evaluate one sub-goal of a conversation an agent has already had, and you
always deliver a verdict.

What you are given:
- The sub-goal's criteria. They are the only standard you judge against: when the sub-goal applies,
  what passes, what fails, and what the verdict is when its situation does not arise.
- The agent's own instructions: what the agent is meant to do. Never hold the agent to more than
  they require.
- The customer's side: what the customer set out to do and how they were meant to behave. It is
  context for reading the conversation, nothing more. The customer is not being evaluated: never
  pass or fail the agent for something the customer did or did not do, and never comment on the
  customer.
- The evidence: what was said, the actions the agent took, and the records of its system
  afterwards.

How to decide:
1. Read the criteria.
2. Read the evidence. Read the conversation for what was said, the actions for what the agent did,
   and inspect the records when the criteria are about what changed. Use inspect_world to discover
   what the environment supports. Only query records when a query tool is available; never invent
   records or infer unobserved state from capability names.
3. Establish whether the sub-goal's situation arose. If it did not, work out why from the evidence:
   the criteria say what that means, and whether the agent's own behaviour kept it from arising
   matters.
4. Apply the criteria exactly as written. Do not add requirements they do not state, and do not
   excuse what they state.
5. Where nothing you were given records the right answer, never call what the agent said accurate
   or correct, and never assume it is.
6. What was said reached you through speech recognition and can contain its errors. A garbled
   phrase, or a word that sounds like one the context calls for, is a mishearing, not something the
   agent said: judge what the agent evidently said.
7. The channel can add content of its own that the agent did not produce, such as a recorded
   notice a phone line plays or a system message around a chat. When the agent's instructions do
   not ask for it, it is not the agent's: judge the agent only on what it said and did itself.

Always decide. The evidence you were given is what there is; never answer that you cannot tell.
Call decide once, with `passed` true or false.

`explanation` is shown to the agent's owner. Write one or two plain sentences about what the agent
said or did and what followed from it, in the terms of their business, citing what actually
happened. Never mention tables, rows, queries, tool names, records you could or could not read,
criteria, this review, a test, a scenario, a simulation, the customer's instructions or any error.
Put anything about how you checked into `evidence`, which only the reviewers of this process see.
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


def _dump_calls(calls: Sequence[Any]) -> str:
    """Every action, with each field trimmed."""
    return json.dumps([_short(_call(c)) for c in calls], default=str, indent=1)


async def judge(
    goal: Any,
    world: Any,
    calls: Sequence[Any],
    *,
    messages: Sequence[Any] = (),
    scenario: Any = None,
    agent_instructions: str = "",
) -> tuple[bool | None, str]:
    """Decide one judged sub-goal: (passed, explanation). Never raises."""
    verdict: dict[str, tuple[bool | None, str]] = {}
    native = isinstance(world, ObservationWorld)

    @tool(
        "inspect_world",
        (
            "The native environment's declared capabilities and evidence limits; no database state."
            if native
            else "The world's tables: without a name, every table and its row count; with one, its rows."
        ),
        schema({} if native else {"table": str}, []),
    )
    async def inspect_world(args: dict[str, Any]) -> dict[str, Any]:
        if native:
            return _say(
                _dump(
                    {
                        "kind": "native_observation",
                        "capabilities": list(world.capabilities),
                        "state_access": "No local database or SQL query is available to the judge. Capability names are not state evidence.",
                        "evidence": "Use recorded actions and their results, and read_transcript for the captured conversation.",
                    }
                )
            )
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
        "read_transcript",
        "What was said, in order: the Customer and the Agent being judged.",
        schema({}, []),
    )
    async def read_transcript(args: dict[str, Any]) -> dict[str, Any]:
        if not messages:
            return _say(
                "Nothing was captured of what was said. Decide from the actions and the records."
            )
        return _say(_transcript(messages))

    @tool(
        "decide",
        "Commit to the verdict, once, after looking.",
        schema(
            {"passed": bool, "explanation": str, "evidence": str},
            ["passed", "explanation"],
        ),
    )
    async def decide(args: dict[str, Any]) -> dict[str, Any]:
        passed = args.get("passed")
        explanation = str(args.get("explanation") or "").strip()
        if not isinstance(passed, bool):
            return _say("passed must be true or false", error=True)
        if not explanation:
            return _say("a verdict needs an explanation of what the agent did", error=True)
        evidence = str(args.get("evidence") or "").strip()
        if evidence:
            logger.info("judge evidence for %s: %s", getattr(goal, "name", ""), evidence)
        verdict["it"] = (passed, explanation)
        return _say("recorded")

    spec = SessionSpec(
        system_prompt=_INSTRUCTIONS,
        servers={
            "world": tool_server(
                name="world",
                tools=[inspect_world, read_transcript, decide]
                if native
                else [inspect_world, query_world, read_transcript, decide],
            )
        },
        max_turns=12,
        model=judge_model(),
    )
    prompt = (
        f"Sub-goal {getattr(goal, 'name', '')!r}: {getattr(goal, 'what', '') or '(no summary)'}\n\n"
        f"Criteria:\n{criteria_text(getattr(goal, 'judged', ''))}\n\n"
        f"The agent's own instructions:\n{agent_instructions.strip() or '(not given)'}\n\n"
        f"{_customer_side(scenario)}"
        f"Actions the agent took:\n{_dump_calls(calls)}\n\n"
        + (
            ""
            if calls
            else "No actions were recorded, so only the conversation is observable: judge what the "
            "agent did from what it said, confirmed and did not contradict.\n\n"
        )
        + "Look as needed, then call decide."
    )
    retry = (
        f"{prompt}\n\nWhat was said:\n{_transcript(messages) if messages else '(nothing captured)'}\n\n"
        "Everything needed is above. Call decide now with passed true or false."
    )
    for attempt, text in enumerate((prompt, retry), start=1):
        try:
            async with Stage(spec, name="judge-sub-goals", overheard=False) as stage:
                await stage.say(text)
        except Exception as exc:  # noqa: BLE001 - a judge that could not run is not a failed agent
            logger.warning(
                "judge attempt %d for %s could not run: %s: %s",
                attempt, getattr(goal, "name", ""), type(exc).__name__, exc,
            )
        if "it" in verdict:
            return verdict["it"]
    logger.warning("judge gave no verdict for %s", getattr(goal, "name", ""))
    return None, ""


def _customer_side(scenario: Any) -> str:
    """What the customer set out to do, as context only; nothing here is judged."""
    if scenario is None:
        return ""
    presented = getattr(scenario, "presented", None) or {}
    situation = getattr(scenario, "instruction", "") or presented.get("situation", "")
    purpose = getattr(scenario, "tests", "") or presented.get("outcome", "")
    if not situation and not purpose:
        return ""
    return (
        "The customer's side, for context only (not evaluated):\n"
        f"What they set out to do: {situation or '(not given)'}\n"
        f"What the conversation was meant to show about the agent: {purpose or '(not given)'}\n\n"
    )


_SPEAKERS = {"user": "Customer", "assistant": "Agent"}


def _transcript(messages: Sequence[Any]) -> str:
    """The turns as spoken, oldest first. A long call keeps its tail, where a readback would be."""
    body = "\n".join(
        f"{_SPEAKERS.get(str(m.get('role')), m.get('role') or 'unknown')}: "
        f"{str(m.get('content') or '').strip()}"
        for m in messages
        if isinstance(m, dict)
    )
    return body if len(body) <= _TRANSCRIPT_LIMIT else "...\n" + body[-_TRANSCRIPT_LIMIT:]


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
