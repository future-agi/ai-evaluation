"""The sub-goals this agent can be checked on, shared by every scenario that needs one.

Defined once for the agent rather than restated per scenario, which is what makes results roll
up: the same sub-goal failing in seven of twelve scenarios is one sentence rather than seven.

``check`` is Python written by the harness. It is given what the run left behind and returns
nothing if the sub-goal held, or a sentence saying what was wrong. Code rather than a mini
language because an environment can be a database, a filesystem or a page, and a language
invented here would fit only the first.
"""

from __future__ import annotations

import ast
import json
import textwrap
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

CATALOGUE = "sub_goals.json"


class SubGoal(BaseModel):
    """One named thing the agent can be checked on, shared across every scenario that needs it.

    ``check`` is Python, written by the harness. It is given what the run left behind and returns
    nothing if the sub-goal held, or a sentence saying what was wrong. Code rather than a mini
    language because an environment can be a database, a filesystem or a page, and a language
    invented here would fit only the first.

    ``judged`` marks the ones nothing observable can settle — whether a refusal was explained,
    whether a price was invented. Those go to a model, and are the exception.
    """

    name: str
    what: str = ""
    check: str = ""
    judged: str = ""

    def deterministic(self) -> bool:
        return bool(self.check.strip())


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
    if not sub_goal.check.strip() and not sub_goal.judged.strip():
        problems.append(
            f"{sub_goal.name}: settles nothing. Give a check in code, or say what a judge has "
            "to decide and why nothing observable can settle it"
        )
    if sub_goal.check.strip() and "def check(" not in sub_goal.check:
        problems.append(
            f"{sub_goal.name}: a check must define check(world, calls) and return a problem as "
            "a string, or None when the sub-goal held"
        )
    problems.extend(_presence_only_problems(sub_goal))
    problems.extend(_judged_problems(sub_goal))
    return problems


# What a check has to touch to be about the outcome rather than about reaching a tool. `.arguments`
# is what the agent passed, `.result`/`.error` is what came back, and `world` is the state left
# behind. A check touching none of these can only be matching call names.
_OUTCOME_ATTRIBUTES = frozenset({"arguments", "args", "result", "error", "refused"})


def _reads_outcome(source: str) -> bool:
    """Whether a check reads what happened, rather than only that a call happened.

    Parsed rather than string-matched. A substring test passes a check that merely mentions
    ``world`` in a comment or names an unused variable, and the whole point of this gate is that a
    check which looks right and settles nothing is the expensive kind of wrong.
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        # An uncompilable check fails run_check with `broken` anyway, and reporting it as
        # outcome-blind here would hide the real reason.
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _OUTCOME_ATTRIBUTES:
            return True
        if isinstance(node, ast.Subscript):
            # world.state()["orders"] parses as a Subscript over a Call; the attribute walk above
            # catches `.state`, but a check handed a plain mapping is still reading state.
            return True
        if isinstance(node, ast.Name) and node.id == "world":
            # Reading the world at all is reading state. The parameter itself is a Name in the
            # signature, so only uses inside the body reach here.
            for function in ast.walk(tree):
                if isinstance(function, ast.FunctionDef) and function.name == "check":
                    names = {
                        inner.id
                        for statement in function.body
                        for inner in ast.walk(statement)
                        if isinstance(inner, ast.Name)
                    }
                    if "world" in names:
                        return True
    return False


def _presence_only_problems(sub_goal: SubGoal) -> list[str]:
    """Refuse a check that only asks whether a tool was reached, rather than what it did."""
    body = sub_goal.check
    if not body.strip():
        return []
    if _reads_outcome(body):
        return []
    return [
        f"{sub_goal.name}: the check only asks whether a tool was called, which any agent reaching "
        "it passes and any agent doing the right thing another way fails. Assert the arguments it "
        "was given, or the state the world was left in. Whether a call was ended is never a "
        "sub-goal; what the agent did before stopping is"
    ]


def _judged_problems(sub_goal: SubGoal) -> list[str]:
    """Hold a judged sub-goal to the reason it is judged.

    The catalogue guidance already says a judge is the fallback, and nothing enforced it, so the
    fallback became the default: one run reported six sub-goals judged rather than settled by code.
    A judged sub-goal has to say what a model must decide and why nothing observable settles it,
    because that sentence is the thing a reviewer can disagree with.
    """
    judged = sub_goal.judged.strip()
    if not judged or sub_goal.check.strip():
        return []
    if len(judged.split()) < 8:
        return [
            f"{sub_goal.name}: judged, but does not say what a model has to decide and why nothing "
            "observable settles it. Name the judgement and the reason code cannot make it, or "
            "write a check"
        ]
    return []


def catalogue_problems(
    sub_goals: Sequence[SubGoal], *, world_is_observable: bool = True
) -> list[str]:
    """Problems with the catalogue taken as a whole, rather than with one sub-goal.

    Per-sub-goal validation cannot see the shape of the set, and the shape is what decides whether
    a suite grades anything: a catalogue that is mostly judged reports opinions.

    ``world_is_observable`` is False for a target we cannot see into -- a conversational agent with
    no executable tools and no state leaves nothing behind for a check to read, so judging is the
    only thing available and is correct rather than lazy. It is never a way around writing a check
    for a world that does have state.
    """
    usable = [one for one in sub_goals if one.name.strip()]
    if not usable or not world_is_observable:
        return []
    judged = [one for one in usable if not one.deterministic()]
    if len(judged) * 2 > len(usable):
        return [
            f"{len(judged)} of {len(usable)} sub-goals are judged rather than settled by code. A "
            "judge is the fallback, not the method: most of these are answerable from the "
            "arguments the agent passed or the state it left. Rewrite the ones that are, and keep "
            "judging only what nothing observable can settle"
        ]
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
