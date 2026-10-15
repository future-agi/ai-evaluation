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

_INSTRUCTIONS = """
You are a judge. You evaluate the sub-goals of a conversation an agent has already had, and you
always deliver a verdict for every one of them.

What you are given:
- Each sub-goal's criteria. They are the only standard you judge that sub-goal against: when it
  applies, what passes, what fails, and what the verdict is when its situation does not arise.
- The agent's own instructions: what the agent is meant to do. Never hold the agent to more than
  they require.
- The customer's side: what the customer set out to do and how they were meant to behave. It is
  context for reading the conversation, nothing more. The customer is not being evaluated: never
  pass or fail the agent for something the customer did or did not do, and never comment on the
  customer.
- The evidence: what was said, the actions the agent took, and the records of its system
  afterwards.

Each sub-goal stands alone. Judge it only against its own criteria. Another sub-goal's criteria,
situation or verdict never decides it: a sub-goal can pass while another fails, and the reverse.
Its explanation and evidence speak only to its own criteria.

How to decide:
1. Read the criteria.
2. Read the evidence. Read the conversation for what was said, the actions for what the agent did,
   and inspect the records when the criteria are about what changed. The records are PostgreSQL;
   use inspect_world to discover them rather than guessing names. When listen is offered, listen
   to the recording: it is the fullest record of what was said. Use the transcript where it is
   unclear, and judge from the transcript and the actions when it cannot be heard.
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

{VERDICT_RULES}
Decide a sub-goal only once you have looked at what each part of its criteria turns on. Call decide
once per sub-goal, with its name in `sub_goal` (never its criteria), its `verdict`, and `evidence`
saying what supports that verdict and where it came from: the recording, the transcript, the
actions or the records.

The customer is simulated and was given a side to play. Separately from the sub-goals, if the
customer contradicted a fact their side fixed for them (not a detail they were free to choose), or
never gave the agent the chance the conversation was meant to show although they could have, call
customer_off_script once with what they did. The call is then not counted for or against the agent. Never call it
because the agent kept something from happening, or for a detail the customer was free to choose.

`explanation` is shown to the agent's owner. Write one or two plain sentences about what the agent
said or did and what followed from it, in the terms of their business, citing what actually
happened. Write in the third person ("the agent", "the caller"); never address anyone as "you". Never mention tables, rows, queries, tool names, records you could or could not read,
criteria, this review, a test, a scenario, a simulation, the customer's instructions or any error.
Put anything about how you checked into `evidence`, which only the reviewers of this process see.
""".strip().replace("{VERDICT_RULES}", VERDICT_RULES)

# The key under which a report that the simulated customer left their side comes back.
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
    """Decide a call's judged sub-goals in one conversation: {name: (held, explanation)}, plus
    `OFF_SCRIPT` when the simulated customer left their side.

    A sub-goal missing from the result was not decided. Never raises.
    """
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
        + describe_call(agent_instructions, scenario, calls, recording)
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
