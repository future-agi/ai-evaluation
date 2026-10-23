"""Proving a scenario is worth keeping, before anything is ever run against the agent.

Three gates, all pure code. No model is asked whether a scenario is good; the environment
decides. Terminal-bench keeps its tasks honest this way, and it is the cheapest useful thing in
the whole harness: no tokens, no network, a few milliseconds.

**Ready.** Reset the world, run the scenario's own ``setup.py``, then its ``ready.py``. The world
has to hold what the scenario presumes. A scenario about the last five chocolates is only a test
of the agent if there really are five; otherwise the agent fails for something we got wrong and
it reads as the agent's fault. This gate is why a missing precondition can never be mistaken for
a finding.

**Solvable.** Then run the reference solution and the checks. They must pass. If they do not,
either the scenario cannot be passed at all or its checks are wrong, and both have happened
here: one scenario asserted a value the agent was never permitted to send; another demanded
confirmation of an item that could not be ordered. Neither was noticed until a live run failed
and read as a finding about the agent.

**Not vacuous.** Then reset, set up again, run *nothing*, and run the checks. They must fail. A
check that passes with no actions taken grades nothing while reporting a result, which is how a
suite goes quietly green. This one earns its keep: on a third-party benchmark it caught three
sub-goals that passed trivially because the seeded world already contained a cancelled order.

Only a scenario that clears all three is kept. That is the green light.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from .catalogue import Catalogue
from .checks import Outcome, reference_of, run_check
from .folder import apply_setup, check_ready
from .scenario import Scenario
from .world.runtime import Call, GeneratedWorld
from .world.snapshot import restore


@dataclass
class Proof:
    """Whether a scenario holds up, and what happened when it was tried."""

    ready: bool = False
    solvable: bool = False
    vacuous: bool = True
    why_not_ready: str = ""
    # Checks that held with nothing done. The scenario is only vacuous when *every* check does
    # that, but a single one still grades nothing, and since sub-goals are shared it will report
    # itself as held for an agent that did nothing at all. Named rather than refused: on a
    # scenario about a refusal, "no order was placed" holding on an untouched world is correct.
    weak: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    with_solution: list[Outcome] = field(default_factory=list)
    with_nothing: list[Outcome] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)
    # Solution steps that were recorded without being executed, because the tool they name has no
    # endpoint bound in this lane. The condition was already detected and logged and then went
    # nowhere, so a scenario whose entire solution was assumed saved as "proved". Carried here so
    # whoever keeps the scenario is told what the proof did not cover.
    assumed: list[str] = field(default_factory=list)
    # Graded by judgement alone, because the world holds nothing a check could read. True only for
    # a target we cannot see into, never a way around writing a check for a world that has one.
    judged_only: bool = False
    # Checks that fail when the agent makes its own choice for an argument the scenario does not
    # mark fixed: they hold the agent to a value nothing fixed. Each names those arguments.
    too_strict: list[str] = field(default_factory=list)
    # Values the scenario marks fixed that no code check notices when they are wrong.
    unchecked: list[str] = field(default_factory=list)
    # The reference solution calls an attached MCP server, which only the PostgreSQL proof runs,
    # so this pass checked setup and readiness and left the other gates to it.
    postgres_only: bool = False

    @property
    def holds(self) -> bool:
        if self.postgres_only:
            return self.ready and not self.broken
        return (
            self.ready
            and self.solvable
            and not self.vacuous
            and not self.weak
            and not self.failed_attempts
            and not self.broken
            and not self.too_strict
            and not self.unchecked
        )

    def gates(self) -> dict[str, bool]:
        """The three answers, for anything that wants to show them."""
        return {
            "ready": self.ready,
            "solvable": self.solvable,
            "not_vacuous": not self.vacuous and not self.failed_attempts,
        }

    def why(self) -> str:
        """What to fix, in the order worth fixing it."""
        if self.broken and not self.why_not_ready:
            return "these checks are broken, not failing:\n  - " + "\n  - ".join(self.broken)
        if not self.ready:
            return (
                "the world is not ready for this scenario, so running it would test us rather "
                f"than the agent:\n  - {self.why_not_ready}\n\n"
                "Either setup.py does not make the change this scenario needs, or ready.py is "
                "checking for something the setup never creates."
            )
        if self.broken:
            return "these checks are broken, not failing:\n  - " + "\n  - ".join(
                self.broken
            )
        if not self.solvable:
            failed = [one for one in self.with_solution if not one.held]
            said = "\n  - ".join(f"{one.name}: {one.said}" for one in failed)
            refusals = (
                "\n\nThe solution's own calls were refused by the world:\n  - "
                + "\n  - ".join(self.refused)
                if self.refused
                else ""
            )
            return (
                "the reference solution does not pass this scenario's own checks, so either the "
                "scenario cannot be passed or the checks are wrong:\n  - "
                + said
                + refusals
            )
        if self.vacuous:
            passed = [one.name for one in self.with_nothing if one.held]
            return (
                "these checks pass without the agent doing anything, so they grade nothing:\n  - "
                + "\n  - ".join(passed)
                + "\n\nIf the point of this scenario is that nothing should happen, checking "
                "the world alone cannot show it — an untouched world looks identical to one "
                "where the agent did nothing at all. Check the calls instead: that the agent "
                "tried, and that the attempt was refused rather than succeeding.\n"
                "    def check(world, calls):\n"
                "        tried = [c for c in calls if c.name == 'add']\n"
                "        if not tried: return 'never attempted it'\n"
                "        if any(c.ok for c in tried): return 'it succeeded'\n"
                "        return None"
            )
        if self.unchecked:
            return (
                "these values are marked fixed, yet every check still passes when the agent gets "
                "them wrong:\n  - "
                + "\n  - ".join(self.unchecked)
                + "\n\nA value is fixed so that something checks it. Fix the check that should "
                "compare it, or take it out of `fixed` if only what is said settles it or the agent words it "
                "itself (a reason, a note, a message), which no check compares as text."
            )
        if self.too_strict:
            return (
                "these checks fail an agent that does every fixed thing right but makes its own "
                "choice, for an argument this scenario does not mark fixed or a way its "
                "instructions allow:\n  - "
                + "\n  - ".join(self.too_strict)
                + "\n\nA check may hold the agent only to what the caller, their record or a "
                "rule of the agent fixes. Check the outcome instead, or mark the argument fixed "
                "if one of those really fixes it."
            )
        if self.failed_attempts:
            return (
                "these checks award credit when every tool attempt crashes without changing "
                "the world or returning a result:\n  - "
                + "\n  - ".join(self.failed_attempts)
                + "\n\nArguments and a tool name are not evidence of success. Require a "
                "successful outcome or the corresponding world effect. For a refusal, require "
                "evidence of the intended refusal, not an arbitrary execution error."
            )
        if self.weak:
            return (
                "these individual checks pass without the agent doing anything, so keeping them "
                "would display credit unsupported by evidence:\n  - "
                + "\n  - ".join(self.weak)
                + "\n\nMake each check assert the relevant attempt or observation as well as the "
                "final state. For a refusal, require that the call was attempted and refused; "
                "an untouched world is not evidence that the agent refused correctly."
            )
        return "holds"


def _checks_for(scenario: Scenario, catalogue: Catalogue) -> list[tuple[str, str]]:
    """The deterministic checks this scenario is graded by, in catalogue order."""
    chosen: list[tuple[str, str]] = []
    for name in scenario.sub_goals:
        sub_goal = catalogue.named(name)
        if sub_goal is not None and sub_goal.deterministic():
            chosen.append((name, sub_goal.check))
    return chosen


def prepared(
    scenario: Scenario, world_root: Path
) -> tuple[GeneratedWorld, Outcome, Outcome]:
    """A fresh world with this scenario's setup applied, and how that went."""
    world = restore(world_root)
    world.reset()
    applied = apply_setup(scenario, world)
    ready = check_ready(scenario, world) if applied.ok else Outcome(False, applied.said)
    # The setup's own calls are not the agent's. Clearing them keeps a check that counts calls
    # from crediting the agent with work the scenario did on its behalf.
    world.calls = []
    return world, applied, ready


logger = logging.getLogger(__name__)


def executes(world: GeneratedWorld, tool: str) -> bool:
    """Whether a reference step for `tool` really runs here, rather than being recorded unrun."""
    if tool not in set(getattr(world, "runtime_tools", set())):
        return True
    return bool((getattr(world, "endpoint_for", {}) or {}).get(tool))


def play_reference_step(world: GeneratedWorld, step: object) -> Call:
    """Play one correct-agent step without confusing its API with its dependency's API.

    A generated world can execute the agent-facing call directly.  A source-provisioned world
    cannot: its model-facing tool may enforce local ordering and inject session state before it
    reaches a raw HTTP service.  For those worlds, drive the raw endpoint with the explicitly
    declared environment payload, but record only the semantic call the agent was expected to
    make.  Purely local state-machine tools have no dependency effect and are recorded as such.
    """
    name = str(getattr(step, "tool", "") or "")
    arguments = dict(getattr(step, "arguments", {}) or {})
    environment_arguments = _resolve_reference_values(
        dict(getattr(step, "environment_arguments", {}) or {}), world.calls
    )
    runtime_tools = set(getattr(world, "runtime_tools", set()))
    if name not in runtime_tools:
        return world.call(name, arguments)

    endpoint = getattr(world, "endpoint_for", {}).get(name)
    forward = getattr(world, "forward", None)
    if endpoint and callable(forward):
        dependency_arguments = environment_arguments or arguments
        effect = forward(endpoint, dependency_arguments, record=False)
        semantic = Call(
            name=name,
            arguments=arguments,
            result=effect.result,
            ok=effect.ok,
            error=effect.error,
            refused=effect.refused,
            at=effect.at,
        )
    else:
        # Reaching here means the tool is a declared runtime tool with nothing bound to call.
        # Two very different situations arrive at the same place: a local orchestration action
        # that genuinely has no environment effect, and a lane where the runtime is built after
        # authoring so no endpoint exists yet. The second cannot be proved, and recording it as
        # a pass is what lets a scenario be kept on a solution nothing ever executed. Say so.
        logger.warning(
            "reference step assumed rather than executed: tool=%s reason=%s. Its result is "
            "recorded ok with no call made, so this step proves nothing about the world.",
            name,
            "no endpoint bound" if not endpoint else "no forwarder on the world",
        )
        semantic = Call(name=name, arguments=arguments)
    world.calls.append(semantic)
    return semantic


def _resolve_reference_values(value: object, calls: list[Call]) -> object:
    """Resolve a dependency value produced by an earlier reference call.

    ``$call.book_ride.booking_ref`` refers to the named field on the most recent successful
    ``book_ride`` result. This drives real chained effects without pretending the model knew a
    backend-generated id.
    """
    if isinstance(value, dict):
        return {
            key: _resolve_reference_values(item, calls) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_reference_values(item, calls) for item in value]
    if not isinstance(value, str) or not value.startswith("$call."):
        return value
    parts = value.split(".")
    if len(parts) < 3:
        return value
    call_name = parts[1]
    found = next(
        (call for call in reversed(calls) if call.name == call_name and call.ok), None
    )
    current = found.result if found is not None else None
    for key in parts[2:]:
        if not isinstance(current, dict) or key not in current:
            return value
        current = current[key]
    return current


def _run(
    scenario: Scenario, world_root: Path, *, with_solution: bool, steps: Sequence[Any] | None = None
) -> tuple[GeneratedWorld, list[Call], list[str], list[str]]:
    """A world set up for this scenario, optionally with the solution (or `steps`) played through it."""
    world, _applied, _ready = prepared(scenario, world_root)
    refused: list[str] = []
    assumed: list[str] = []
    if with_solution:
        for step in scenario.solution if steps is None else steps:
            call = play_reference_step(world, step)
            if not call.ok:
                refused.append(f"{call.name}({step.arguments}): {call.error}")
        assumed = [
            str(getattr(step, "tool", ""))
            for step in (scenario.solution if steps is None else steps)
            if not executes(world, str(getattr(step, "tool", "")))
        ]
        if assumed:
            logger.warning(
                "scenario %s: %d of %d solution steps were assumed, not executed (%s). The "
                "proof below covers only the remaining steps.",
                scenario.name,
                len(assumed),
                len(scenario.solution),
                ", ".join(sorted(set(assumed))),
            )
        else:
            logger.info(
                "scenario %s: all %d solution steps executed against the world",
                scenario.name,
                len(scenario.solution),
            )
    return world, list(world.calls), refused, sorted(set(assumed))


def _declared_options(world_root: Path) -> set[str]:
    """`tool.argument` names whose values the agent's own tool definitions limit to a set."""
    try:
        tools = json.loads((Path(world_root) / "contract.json").read_text(encoding="utf-8")).get("tools") or []
    except (OSError, ValueError):
        return set()
    return {
        f"{tool.get('name')}.{name}"
        for tool in tools
        for name, values in (tool.get("arg_values") or {}).items()
        if values
    }


def _with_value(step: Any, name: str, value: Any) -> Any:
    """The step with one argument changed where it is executed too, or None when what is executed
    does not carry that argument, so changing it could not change what the step does."""
    update: dict[str, Any] = {"arguments": {**step.arguments, name: value}}
    if step.environment_arguments:
        if step.environment_arguments.get(name) != step.arguments.get(name):
            return None
        update["environment_arguments"] = {**step.environment_arguments, name: value}
    return step.model_copy(update=update)


def _end_state(world: GeneratedWorld) -> dict[str, Any] | None:
    try:
        return world.state()
    except Exception:  # noqa: BLE001 - an unreadable state compares as different, never as equal
        return None


def _other_choices(scenario: Scenario, declared: set[str] = frozenset()) -> tuple[list[Any], set[str]]:
    """The solution with every free text argument the scenario does not mark fixed set otherwise."""
    steps = list(getattr(scenario, "solution", None) or ())
    if any(step.fixed is None for step in steps):
        return steps, set()
    varied, changed = [], set()
    for step in steps:
        for name, value in step.arguments.items():
            free = name not in step.fixed and f"{step.tool}.{name}" not in declared
            if free and isinstance(value, str) and value and not value.startswith("$"):
                other = _with_value(step, name, f"another {name}")
                if other is not None:
                    step = other
                    changed.add(f"{step.tool}.{name}")
        varied.append(step)
    return varied, changed


def _with_one_wrong(scenario: Scenario) -> list[tuple[str, list[Any], bool]]:
    """For each fixed value, the solution with only that value changed, and whether a later step
    sets that argument again (a draft the solution itself corrects)."""
    steps = list(getattr(scenario, "solution", None) or ())
    wrong = []
    for at, step in enumerate(steps):
        for name in step.fixed or ():
            value = step.arguments.get(name)
            if isinstance(value, bool):
                other: Any = not value
            elif isinstance(value, (int, float)):
                other = value + 1
            elif isinstance(value, str) and value and not value.startswith("$"):
                other = f"{value}-other"
            else:
                continue
            changed = _with_value(step, name, other)
            if changed is not None:
                corrected = any(name in (later.arguments or {}) for later in steps[at + 1 :])
                wrong.append((f"{step.tool}.{name}", [*steps[:at], changed, *steps[at + 1 :]], corrected))
    return wrong


def prove(
    scenario: Scenario,
    catalogue: Catalogue,
    world_root: Path,
    *,
    postgres_only: frozenset[str] = frozenset(),
) -> Proof:
    """Run all three gates and say whether this scenario is worth keeping.

    A scenario whose solution calls a tool in `postgres_only` is checked for readiness here and
    proved in full by the PostgreSQL proof, which runs those tools."""
    proof = Proof()
    checks = _checks_for(scenario, catalogue)
    reference = reference_of(scenario.solution)
    if not checks:
        # Every sub-goal is judged on the call itself, so the gates that still mean something are
        # that setup and ready ran; the writer proves the reference solution with try_calls.
        world, applied, ready = prepared(scenario, world_root)
        world.close()
        proof.why_not_ready = (
            "" if applied.ok and ready.ok else (applied.said or ready.said)
        )
        proof.ready = applied.ok and ready.ok
        proof.solvable = proof.ready
        proof.vacuous = False
        proof.judged_only = True
        return proof

    # Gate 1: is the world ready for this scenario at all?
    world, applied, ready = prepared(scenario, world_root)
    world.close()
    if not applied.ok:
        proof.why_not_ready = applied.said
        if applied.broken:
            proof.broken = [applied.said]
        return proof
    if not ready.ok:
        proof.why_not_ready = ready.said
        if ready.broken:
            proof.broken = [ready.said]
        return proof
    proof.ready = True
    if postgres_only & {str(step.tool) for step in scenario.solution}:
        proof.postgres_only = True
        return proof

    # Gate 2: does the reference solution pass this scenario's own checks?
    world, calls, refused, assumed = _run(scenario, world_root, with_solution=True)
    # What the proof below could not cover. Detected here since forever and logged into a void; a
    # scenario whose whole solution was assumed used to save as proved.
    proof.assumed = assumed
    try:
        proof.with_solution = [
            run_check(source, world, calls, name=name, reference=reference)
            for name, source in checks
        ]
        passing = {one.name for one in proof.with_solution if one.held}
        held = [(name, source) for name, source in checks if name in passing]
        right_state = _end_state(world)
    finally:
        world.close()
    varied, changed = _other_choices(scenario, _declared_options(world_root))
    if changed:
        other_world, other_calls, other_refused, _ = _run(
            scenario, world_root, with_solution=True, steps=varied
        )
        try:
            proof.too_strict = [
                f"{name} (the agent chose: {', '.join(sorted(changed))})"
                for name, source in held
                if not other_refused
                and not run_check(source, other_world, other_calls, name=name, reference=reference).held
            ]
        finally:
            other_world.close()
    for number, other in enumerate(getattr(scenario, "alternatives", None) or (), start=1):
        alt_world, alt_calls, _, _ = _run(scenario, world_root, with_solution=True, steps=other)
        try:
            proof.too_strict += [
                f"{name} (another way the instructions allow, alternative {number}: "
                f"{', '.join(step.tool for step in other)})"
                for name, source in held
                if not run_check(source, alt_world, alt_calls, name=name, reference=reference).held
            ]
        finally:
            alt_world.close()
    for label, steps, corrected in _with_one_wrong(scenario) if held else ():
        wrong_world, wrong_calls, _, _ = _run(scenario, world_root, with_solution=True, steps=steps)
        try:
            # Only a value a later step sets again, leaving the right end state, is a corrected draft
            # no check must notice; a read-only lookup leaves the same state and is still checked.
            ended = _end_state(wrong_world)
            if not (corrected and ended is not None and ended == right_state) and all(
                run_check(source, wrong_world, wrong_calls, name=name, reference=reference).held
                for name, source in held
            ):
                proof.unchecked.append(label)
        finally:
            wrong_world.close()
    proof.refused = refused
    proof.broken = [one.name for one in proof.with_solution if one.broken]
    proof.solvable = all(one.held for one in proof.with_solution) and not proof.broken

    # Gate 3: do those same checks fail when nothing is done?
    untouched, nothing, _, _ = _run(scenario, world_root, with_solution=False)
    try:
        proof.with_nothing = [
            run_check(source, untouched, nothing, name=name, reference=reference)
            for name, source in checks
        ]
        # A call-presence check can pass the no-op gate yet credit failed operations.
        # Keep the arguments, remove all results/effects, and inject an execution failure.
        # This is not a business refusal: a timeout does not prove correct refusal either.
        crashed = [
            replace(
                call, ok=False, result=None, error="execution failed", refused=False
            )
            for call in calls
        ]
        if crashed:
            proof.failed_attempts = [
                name
                for name, source in checks
                if not any(one.name == name and one.held for one in proof.with_nothing)
                and run_check(source, untouched, crashed, name=name, reference=reference).held
            ]
    finally:
        untouched.close()
    # Record every check that passes without an action. Even when another checkpoint makes the
    # overall scenario non-vacuous, showing this one as green would award unsupported credit —
    # exactly the misleading partial-pass display the proof gate exists to prevent.
    proof.weak = [one.name for one in proof.with_nothing if one.held]
    # A judged sub-goal reads what the agent said, and an agent that did nothing said nothing, so
    # it cannot be passed by an empty run the way a state check can. It still does not excuse a
    # deterministic checkpoint that independently awards credit with no evidence.
    judged = [
        name
        for name in scenario.sub_goals
        if (found := catalogue.named(name)) is not None and not found.deterministic()
    ]
    proof.vacuous = (
        bool(proof.with_nothing)
        and len(proof.weak) == len(proof.with_nothing)
        and not judged
    )
    return proof
