"""Deciding all of a call's judged sub-goals in one conversation, the default (ALK_JUDGE_MODE=separate turns it off).

The evidence, its tools and the order of looking are `judge.py`'s; only the deciding differs. The
recording, transcript, actions and records are read once, and each sub-goal is decided against its
own criteria with its own `decide` call. The scheduler judges every sub-goal of the call alone,
with `judge.judge_sub_goal`, whenever this leaves any of them undecided.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .backends import tool
from .catalogue import criteria_text
from .judge import (
    JUDGE_EXPLANATION,
    JUDGE_GIVEN,
    JUDGE_METHOD,
    VERDICT_FIELD,
    VERDICT_RULES,
    _STEPS,
    compared,
    CallEvidence,
    _say,
    _transcript,
    describe_call,
    request_final_verdicts,
    run_judge_conversation,
    verdict_of,
)
from .tools import schema

logger = logging.getLogger(__name__)

_INSTRUCTIONS = f"""
You are a judge. You evaluate the sub-goals of a conversation an agent has already had, and you
always deliver a verdict for every one of them.

What you are given:
- Each sub-goal's criteria. They are the only standard you judge that sub-goal against: when it
  applies, what passes, what fails, and what the verdict is when its situation does not arise.
{JUDGE_GIVEN}

Each sub-goal stands alone. Judge it only against its own criteria. Another sub-goal's criteria,
situation or verdict never decides it: a sub-goal can pass while another fails, and the reverse.
Its explanation and evidence speak only to its own criteria.

{JUDGE_METHOD}

{VERDICT_RULES}
Decide a sub-goal only once you have looked at what each part of its criteria turns on. Call decide
once per sub-goal, with its name in `sub_goal` (never its criteria), its `verdict`, and `evidence`
saying what supports that verdict and where it came from: the recording, the transcript, the
actions or the records.

The customer is simulated and was given a side to play. Separately from the sub-goals, if the
customer contradicted a fact their side fixed for them (not a detail they were free to choose), or
never gave the agent the chance the conversation was meant to show although they could have, call
customer_off_script once with what they did. The call is then not counted for or against the agent.
Never call it because the agent kept something from happening, or for a detail the customer was free
to choose.

{JUDGE_EXPLANATION}
""".strip()

OFF_SCRIPT = "__customer_off_script__"


async def judge_sub_goals_jointly(
    goals: Sequence[Any],
    world: Any,
    calls: Sequence[Any],
    *,
    messages: Sequence[Any] = (),
    scenario: Any = None,
    agent_instructions: str = "",
    recording: str | None = None,
) -> dict[str, tuple[bool | None, str]]:
    """{name: (held, explanation)} for a call's judged sub-goals, plus `OFF_SCRIPT`; never raises."""
    names = [str(getattr(goal, "name", "")) for goal in goals]
    evidence = CallEvidence(world, messages, recording, ", ".join(names))
    verdict: dict[str, tuple[bool | None, str]] = {}

    @tool(
        "customer_off_script",
        "Report, once, that the simulated customer contradicted a fact fixed for them or withheld "
        "the chance the conversation was meant to show.",
        schema({"what": str, "evidence": str}, ["what", "evidence"]),
    )
    async def customer_off_script(args: dict[str, Any]) -> dict[str, Any]:
        what = str(args.get("what") or "").strip()
        if not what:
            return _say("say what the customer did", error=True)
        logger.info("customer off script: %s (%s)", what, args.get("evidence"))
        verdict[OFF_SCRIPT] = (False, what)
        return _say("recorded")

    @tool(
        "decide",
        "Commit to one sub-goal's verdict, once, after looking. Name the sub-goal; never repeat its criteria.",
        schema(
            {"sub_goal": str, "verdict": VERDICT_FIELD, "explanation": str, "evidence": str},
            ["sub_goal", "verdict", "explanation", "evidence"],
        ),
    )
    async def decide(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("sub_goal") or "").strip()
        if name not in names:
            return _say(f"sub_goal must be one of: {', '.join(names)}", error=True)
        if name in verdict:
            return _say(f"{name} is already decided", error=True)
        refusal = evidence.verdict_problem(args)
        if refusal:
            return _say(refusal, error=True)
        logger.info("judge evidence for %s: %s", name, args["evidence"])
        verdict[name] = verdict_of(args)
        return _say(f"recorded {name}")

    def missing() -> list[str]:
        return [name for name in names if name not in verdict]

    def ask() -> str:
        return (
            f"No verdict yet for: {', '.join(missing())}. Judge each on its own criteria and call "
            "decide once for each, with its verdict."
        )

    tools = [*evidence.tools(), decide, customer_off_script]
    prompt = (
        f"Sub-goals to judge ({len(goals)}). Judge each one on its own criteria only.\n\n"
        + "".join(
            f"=== Sub-goal {index} of {len(goals)}: {name} ===\n"
            f"What it checks: {getattr(goal, 'what', '') or '(no summary)'}\n"
            f"{compared(goal)}"
            f"Criteria:\n{criteria_text(getattr(goal, 'judged', ''))}\n"
            f"=== End of {name} ===\n\n"
            for index, (name, goal) in enumerate(zip(names, goals), start=1)
        )
        + describe_call(agent_instructions, scenario, calls, recording, messages)
        + "Look as needed, then call decide once for each sub-goal."
    )
    conversation = await run_judge_conversation(
        [{"role": "system", "content": _INSTRUCTIONS}, {"role": "user", "content": prompt}],
        tools,
        evidence,
        done=lambda: not missing(),
        nudge=lambda: (
            f"What was said:\n{_transcript(messages) if messages else '(nothing captured)'}\n\n" + ask()
        ),
        steps=_STEPS + 2 * (len(goals) - 1),
    )
    if missing():
        await request_final_verdicts(_INSTRUCTIONS, prompt, conversation, evidence, decide, "Everything needed is above. " + ask())
    if missing():
        logger.warning("judge gave no verdict for %s", ", ".join(missing()))
    return verdict
