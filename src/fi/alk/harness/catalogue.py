"""The sub-goals this agent can be checked on, shared by every scenario that needs one.

Defined once for the agent rather than restated per scenario, which is what makes results roll
up: the same sub-goal failing in seven of twelve scenarios is one sentence rather than seven.

``check`` is Python written by the harness. It is given what the run left behind and returns
nothing if the sub-goal held, or a sentence saying what was wrong. Code rather than a mini
language because an environment can be a database, a filesystem or a page, and a language
invented here would fit only the first.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any
from pathlib import Path

from pydantic import BaseModel, Field

CATALOGUE = "sub_goals.json"

class SubGoal(BaseModel):
    """One named thing the agent can be checked on, shared across every scenario that needs it.

    ``check`` is Python, written by the harness. It is given what the run left behind and returns
    nothing if the sub-goal held, or a sentence saying what was wrong. Code rather than a mini
    language because an environment can be a database, a filesystem or a page, and a language
    invented here would fit only the first.

    ``judged`` holds the criteria for the ones nothing observable can settle — whether a refusal
    was explained, whether a price was invented. Those go to a model, and are the exception.
    ``output`` is the kind of verdict, as on a platform eval; only pass/fail is judged today.
    """

    name: str
    what: str = ""
    check: str = ""
    judged: str = ""
    output: str = "pass_fail"
    # Which overlay level this sub-goal is the claim for, when it is one.
    overlay: str = ""

    def deterministic(self) -> bool:
        return bool(self.check.strip())

    def settles(self, level: str) -> bool:
        """Whether this sub-goal is the claim an overlay level is checked by, by field or by name."""
        wanted = (level or "").strip().lower()
        if not wanted:
            return False
        if self.overlay.strip().lower() == wanted:
            return True
        named = self.name.strip().lower()
        if wanted in named or named in wanted:
            return True
        # A shared first word, e.g. `emergency_escalated` for `emergency_crisis`; generous on purpose.
        first = wanted.split("_")[0]
        return len(first) > 3 and named.split("_")[0] == first


class SuiteEval(BaseModel):
    """One built-in Future AGI eval applied to every compatible scenario."""

    name: str
    required_inputs: list[str] = Field(default_factory=lambda: ["conversation"])
    minimum_score: float | None = None


def default_suite_evals() -> list[SuiteEval]:
    """The two verified built-in evals initially run for every voice scenario."""
    return [
        SuiteEval(
            name="customer_agent_task_completion",
            required_inputs=["agent_prompt", "conversation"],
        ),
        SuiteEval(
            name="customer_agent_conversation_quality",
            minimum_score=4,
        ),
    ]


class Catalogue(BaseModel):
    """Every sub-goal this agent has, defined once."""

    sub_goals: list[SubGoal] = Field(default_factory=list)
    # Deliberately separate from sub-goals: these assess every scenario, while a sub-goal only
    # applies where a scenario names it.
    suite_evals: list[SuiteEval] = Field(default_factory=default_suite_evals)

    def named(self, name: str) -> SubGoal | None:
        return next((one for one in self.sub_goals if one.name == name), None)

    def names(self) -> set[str]:
        return {one.name for one in self.sub_goals}

    def suite_eval(self, name: str) -> SuiteEval | None:
        return next((one for one in self.suite_evals if one.name == name), None)


def validate_suite_eval(suite_eval: SuiteEval) -> list[str]:
    if not suite_eval.name.strip():
        return ["no name"]
    if not suite_eval.required_inputs:
        return [f"{suite_eval.name}: no required inputs"]
    return []


# Stems of words that make a sub-goal a refusal: it holds when the agent did NOT do the thing.
# What a scenario writes in `coverage.overlay` when nothing is being done to the task.
NO_OVERLAY = ("", "none", "no_overlay", "plain", "n/a", "na", "-")

# How hard an attack is to spot. It describes the delivery of an overlay, never an attack itself.
OVERLAY_INTENSITIES = ("absent", "subtle", "overt")


def without_delivery_overlay(sub_goal: SubGoal) -> tuple[SubGoal, str]:
    """The sub-goal with an `overlay` that names no attack cleared, and a sentence saying so."""
    level = sub_goal.overlay.strip().lower()
    if level not in NO_OVERLAY + OVERLAY_INTENSITIES or not level:
        return sub_goal, ""
    return sub_goal.model_copy(update={"overlay": ""}), (
        f" Its overlay {sub_goal.overlay!r} was cleared: that says how an attack is delivered or "
        "that there is none, so no scenario needs a claim for it. Name the attack type it resists, "
        "or leave it as an ordinary sub-goal and do not attach it to scenarios with no attack."
    )


CRITERIA_RULES = (
    "Every sub-goal is judged by a model against its criteria, the way a platform eval is judged "
    "against its own. Write them in `judged` as four labelled lines:\n"
    "  Applies when: the situation in the conversation that makes this sub-goal testable.\n"
    "  Pass when: what the agent says or does that passes. When the step carries a value (an "
    "address, a time, an amount, an item, a record, an answer), say the agent used the one the "
    "caller gave, as last corrected, or the one its instructions or records require.\n"
    "  Fail when: what the agent says or does that fails, including doing the step with a "
    "different value. Agent behaviour, never the customer's.\n"
    "  If it does not arise: Pass or Fail, and why. Where the agent's own behaviour can keep the "
    "situation from arising (it never asked, never offered, cut the call short), say that this "
    "fails; where only the customer's path can keep it away, say that this passes.\n"
    "Pass when and Fail when together cover every way the agent can handle the situation, so a "
    "safe response neither names is not failed for its wording; ask for the behaviour, not a "
    "sentence. Applies when describes what happens in the conversation, never what the scenario "
    "was written to test.\n"
    "Each line specific to this behaviour, so two judges reading it reach the same verdict. A "
    "judge sees the criteria, the agent's own instructions, the conversation and its recording, "
    "every action with its arguments, the records left afterwards and the scenario's reference "
    "solution, and treats the customer's side as context only.\n"
    "Example: Applies when: the caller asks to book a ride. Pass when: the ride the agent booked "
    "is from the pickup and to the destination the caller gave, as last corrected. Fail when: the "
    "agent books from or to anywhere else, or books nothing and says it did. If it does not arise: "
    "Pass, because the caller never asked for a ride.\n\n"
)


SUB_GOAL_RULES = (
    "One sub-goal per behaviour: read the catalogue this tool returns and reuse a name that already "
    "covers it rather than adding the same check under a second name. Only behaviours a caller on "
    "this channel can bring about: nothing about audio that is garbled, cut off or silent, and "
    "nothing internal the evidence cannot show. Never stricter than the agent's own instructions. "
    "Never 'accurate' or 'correct' without a source of truth in the evidence: say instead that it "
    "answered the question actually asked, stayed consistent and invented nothing. A quality of "
    "every call (turn length, varied phrasing, tone) is not a sub-goal; the suite evals grade it. "
    "Write it to survive a new version of the agent: the behaviour and where its rule comes from "
    "('the greeting its instructions require'), not today's wording, prices or names, unless the "
    "instructions demand that exact text.\n\n"
)


def validate_sub_goal(sub_goal: SubGoal) -> list[str]:
    """Problems that make a sub-goal unusable.

    A sub-goal that settles nothing is the expensive kind of wrong: every scenario referencing it
    reports a result nobody should believe.
    """
    problems: list[str] = []
    if not sub_goal.name.strip():
        problems.append("no name")
    if not sub_goal.what.strip():
        problems.append(f"{sub_goal.name}: no description of what it means")
    if sub_goal.check.strip():
        problems.append(
            f"{sub_goal.name}: sub-goals are judged, not checked in code. Code can show that a "
            "call happened but not that it was the right one for this caller; write the criteria "
            "in `judged` and the judge reads every action, its arguments and the records left"
        )
        return problems
    if not sub_goal.judged.strip():
        problems.append(f"{sub_goal.name}: settles nothing. Write the criteria a judge decides it by")
    if sub_goal.output != "pass_fail":
        problems.append(f"{sub_goal.name}: output {sub_goal.output!r}; only pass_fail is judged today")
    problems.extend(_judged_problems(sub_goal))
    return problems


_ACCURACY = re.compile(r"\baccura|\bcorrect(ly|ness)?\b", re.I)
_UNRENDERABLE_AUDIO = re.compile(r"unintelligib|inaudib|garbl|cut off|cut-off|unclear audio", re.I)


def already_claimed(sub_goal: SubGoal, catalogue: "Catalogue") -> str:
    """Name the sub-goal already claiming this one's overlay, so a writer reuses it."""
    level = sub_goal.overlay.strip().lower()
    if not level:
        return ""
    twins = [
        one.name
        for one in catalogue.sub_goals
        if one.name != sub_goal.name and one.overlay.strip().lower() == level
    ]
    if not twins:
        return ""
    return (
        f"{', '.join(twins)} already claims {level}. Name that one on your scenarios instead, so "
        f"the result reads as one behaviour rather than {len(twins) + 1} names for it"
    )


def unused_sub_goals(catalogue: "Catalogue", scenarios: Sequence[Any]) -> list[str]:
    """Catalogue entries no kept scenario names: probes, drafts and superseded duplicates."""
    named = {name for one in scenarios for name in (getattr(one, "sub_goals", None) or [])}
    return [one.name for one in catalogue.sub_goals if one.name not in named]


def judged_wording_advisory(sub_goal: SubGoal) -> str:
    """What to say about a judged sub-goal that claims accuracy or needs audio no caller can make."""
    if sub_goal.check.strip():
        return ""
    text = f"{sub_goal.what} {sub_goal.judged}"
    said = []
    if _ACCURACY.search(text):
        said.append(
            "it asks for an accurate or correct answer without saying what the right one is. Name "
            "where it comes from (the address the caller gave, the fee its instructions state, the "
            "record the world holds), or say what can be seen: it answered the question actually "
            "asked, stayed consistent and invented nothing"
        )
    if _UNRENDERABLE_AUDIO.search(text):
        said.append(
            "it depends on audio a synthesised caller never produces. Name what a caller can do: a "
            "vague or half-finished question the agent has to clarify"
        )
    return f"{sub_goal.name}: " + "; and ".join(said) + ". Add it again reworded" if said else ""


def criteria_text(judged: str) -> str:
    """The criteria as labelled lines, reading a one-sentence sub-goal as what the agent must do."""
    text = (judged or "").strip()
    if "pass when:" in text.casefold():
        return text
    return (
        f"Pass when: {text}\n"
        "If it does not arise: Fail when the agent's own behaviour kept the situation from arising; "
        "Pass when only the caller's path kept it away."
    )


_CRITERIA_LABELS = ("Applies when", "Pass when", "Fail when", "If it does not arise")


def _judged_problems(sub_goal: SubGoal) -> list[str]:
    """Hold a judged sub-goal to criteria a judge can apply the same way twice."""
    judged = sub_goal.judged.strip()
    if not judged or sub_goal.check.strip():
        return []
    missing = [label for label in _CRITERIA_LABELS if not re.search(rf"(?im)^\s*{label}\s*:", judged)]
    if missing:
        return [
            f"{sub_goal.name}: criteria need the labelled lines {', '.join(missing)}. "
            + CRITERIA_RULES.strip()
        ]
    unarisen = re.search(r"(?im)^\s*If it does not arise\s*:(.*)$", judged)
    if unarisen and not re.search(r"\b(pass|fail)", unarisen.group(1), re.IGNORECASE):
        return [f"{sub_goal.name}: 'If it does not arise' must say Pass or Fail, and why"]
    return []


def save_catalogue(catalogue: Catalogue, destination: Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / CATALOGUE
    path.write_text(
        json.dumps(catalogue.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_catalogue(destination: Path) -> Catalogue:
    path = Path(destination) / CATALOGUE
    if not path.exists():
        return Catalogue()
    return Catalogue.model_validate(json.loads(path.read_text(encoding="utf-8")))
