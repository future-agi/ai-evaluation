"""Stage three: write the scenarios the agent will be tested with.

Reads the contract and the world that was built from it, and produces scenarios grounded in both.
The stage can look at the world and run calls against throwaway copies of it, which is what keeps
a scenario about a real record rather than a plausible-sounding one.

Like the other stages it stays open. A suite is usually right on the second look, and "make three
of these harder" is the next thing said rather than a regeneration from nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .backends import SessionSpec, ToolServer, tool, tool_server

from .config import artifact_dir, chosen_model, discovered_skills, load_skill
from .catalogue import load_catalogue
from .contract import AgentContract
from .scenario import Scenario, suite_diversity_problems, voicemail_enabled
from .scenario_tools import (
    journalled,
    worth_delegating,
    SCENARIO_SERVER,
    load_scenarios,
    scenario_tools,
    world_summary,
    write_scenarios,
)
from .session import Stage
from .tools import schema

logger = logging.getLogger(__name__)

SKILL = "write-scenarios"
PLAN_SKILL = "plan-suite"

# The review pass runs its own tool server, kept apart from the writers' one so a reviewer can
# only report gaps and never submit or save a scenario itself.
REVIEW_SERVER = "suite-review"


# Turns a scenario costs in practice: look at the world, rehearse the calls, submit, and often
# one more to correct what a gate refused.
TURNS_EACH = 3
# Enough to write a handful without the budget being the thing that stops it.
TURNS_FLOOR = 120


def turns_for(wanted: int) -> int:
    """A turn budget that grows with the suite being asked for.

    A fixed ceiling is what made asking for a large suite pointless: generation stopped partway
    through, and `save_scenarios` refuses a count that does not match what was asked for, so a run
    that asked for fifty and reached twenty-eight saved nothing at all. The budget has to follow
    the request, or the request cannot be honoured.
    """
    return max(TURNS_FLOOR, wanted * TURNS_EACH + 40)


def open_stage(
    contract: AgentContract,
    *,
    out: Path | None = None,
    wanted: int = 10,
    ask: Callable[..., Any] | None = None,
    max_turns: int = 0,
) -> tuple[Stage, Path]:
    """A live write-the-scenarios stage, and where it will write."""
    destination = out or artifact_dir(contract.agent)
    server, kept = scenario_tools(contract, destination, destination, wanted=wanted)
    spec = SessionSpec(
        # Same ordering as the slice writer: the agent and its world before the method.
        system_prompt=(
            f"## This agent\n\n{contract.brief(with_data=True)}"
            f"\n\n## Its world\n\n{world_summary(destination)}"
            f"\n\n{load_skill(SKILL)}"
            # Planning and writing are two stages. This session does the first, so it gets both;
            # a slice writer gets the writing skill alone, because the plan is already made and
            # widening a slice is the one thing it must not do.
            + (f"\n\n{load_skill(PLAN_SKILL)}" if worth_delegating(wanted) else "")
            # Whatever this kind of agent adds on top. A file under skills/kinds/ that
            # declares `applies_to: modality=<kind>` is appended here, so supporting a
            # new kind of agent is adding that file and nothing else.
            # `voicemail` gates the mailbox skill the way `modality` gates this one.
            + discovered_skills(
                modality=contract.modality,
                voicemail="on" if voicemail_enabled() else "off",
            )
            + (
                f"\n\nWrite {wanted} scenarios."
                if not kept
                else f"\n\n{len(kept)} scenarios already exist and are loaded: "
                + ", ".join(scenario.name for scenario in kept)
                + ". Submitting one under an existing name replaces it."
            )
        ),
        servers={SCENARIO_SERVER: server},
        builtins=("AskUserQuestion",),
        cwd=str(destination.parent if destination.parent.exists() else Path.cwd()),
        max_turns=max_turns or turns_for(wanted),
        model=chosen_model(),
        ask=ask,
        thinking=True,
        # Silence means something different once the writing is delegated: see the constant.
        idle_timeout_seconds=(
            QUIET_WHILE_DELEGATING_SECONDS if worth_delegating(wanted) else 0.0
        ),
    )
    return Stage(spec, name=SKILL), destination


def opening(contract: AgentContract, wanted: int = 10, existing: int = 0) -> str:
    if existing:
        return (
            f"There are already {existing} scenarios for {contract.agent!r}, and they are "
            "loaded. Use inspect_scenario before changing each one so every unchanged field is "
            "preserved exactly. Say what you want changed, or add to them. Anything you submit "
            "under an existing name replaces it."
        )
    return (
        f"Write {wanted} scenarios for {contract.agent!r}.\n\n"
        "Look at the world first with inspect_world so every scenario names real records, and "
        "read the sub-goals already defined. After that inspection, immediately work out and "
        "submit one scenario at a time; never hold the whole suite in one long response. Emit a "
        "tool call after each scenario so progress is visible and proved work survives a stop. "
        "Work out each scenario's solution with try_calls before you submit it, because a "
        "scenario is only kept if its solution passes its own "
        "checks and those checks fail without it. In a source-provisioned world, keep each "
        "solution step's arguments exactly model-facing. If the raw dependency needs trusted "
        "fields injected by the worker, put its complete payload in environment_arguments; "
        "never pretend the model supplied an internal identifier, a resolved lookup, a priced "
        "result, or any other value it could not have known. Treat every contract phrase that ties "
        "a value to this conversation literally: the reference "
        "solution must create that state earlier in the same conversation. Never pre-seed "
        "opaque state that the agent has no public tool or session state to retrieve. Cover the "
        "ordinary case, the request that has "
        "to be refused, the rule under pressure, and at least one where state has to carry "
        "across several turns. If a proof says an intended check is vacuous or broken, repair "
        "that named sub-goal with add_sub_goal and resubmit. Never evade a gate by deleting a "
        "check for behavior the scenario still claims to test. Then save_scenarios."
        + (
            "\n\nFor a suite rather than one scenario, say briefly how you are splitting it "
            "across the agent's use cases and then write it with generate_suite in the same "
            "turn: it runs a writer per use case at the same time and saves what they prove."
            if worth_delegating(wanted)
            else ""
        )
    )


def load(destination: Path) -> list[Scenario]:
    """The scenarios written for this agent, if any have been."""
    return load_scenarios(Path(destination))


# What a suite costs, and what it is allowed to cost.
#
# Writers run as separate model sessions, so wall clock is roughly the number of scenarios
# divided by how many run at once. The two ceilings below exist for different reasons: one
# protects the machine, the other protects the person waiting. Asking for a thousand scenarios
# is a reasonable thing to want and an unreasonable thing to do in one go, so a large ask is
# served a batch at a time with the rest offered back.
AT_ONCE = 4
# Writers each drive their own model session, so this is a request rate as much as a concurrency.
# Eight of them exhausts the provider quota and the writers that get the 429 lose their whole
# slice, which costs more scenarios than the extra concurrency buys.
MOST_AT_ONCE = int(os.environ.get("HARNESS_WRITERS_AT_ONCE") or 4)
# How many a single generate_suite pass will write. A hosted run is unattended, so a cap here
# does not pause for a person, it just returns fewer than were asked for and stops. Kept as a
# backstop against a runaway ask rather than as a batch size.
MOST_IN_ONE_GO = 1000

# How many times the suite is reviewed and topped up after the first pass. One is enough to
# catch a slice that came back short or a use case nobody covered; more turns it into a loop
# that keeps finding smaller things to say.
TOP_UP_ROUNDS = 1

# How long a session may say nothing before the harness treats it as hung, where the default of
# ten minutes is wrong for this stage.
#
# The planning session's whole turn is one `generate_suite` call, and that call does not return
# until the writers it started have finished. It is working the entire time and has nothing to
# emit while it works, so the default bound kills a fan-out mid-flight and throws away everything
# the writers proved. A hundred scenarios across four writers is comfortably an hour, and any
# writer may also be waiting out a quota refusal inside that.
#
# Still bounded, because the reason the bound exists is real: a dropped provider stream leaves a
# session alive forever. The outer bound is the run's own authoring deadline.
QUIET_WHILE_DELEGATING_SECONDS = 5400.0
# A writer is quiet while one of its own tool calls runs, and its longest is a proof: restore the
# world, apply setup, play the solution, then play it again against an untouched world. Minutes,
# not an hour, and keeping this shorter than the planner's bound is what frees a stuck writer's
# slot for the next slice instead of holding it until the whole stage times out.
QUIET_WHILE_WRITING_SECONDS = 1800.0


# A session refused by the provider is retried rather than abandoned: its work is still worth doing and
# a slice keeps what it already proved. The quota is measured over a minute, so each wait clears a
# minute; a shorter one asks inside the same window and is refused again for the same reason. What is
# bounded is the total, not the number of tries: five minutes of waiting is worth a slice, and a run
# that waits longer than that is not going to be rescued by waiting more.
RATE_LIMIT_BACKOFF_SECONDS = 60
RATE_LIMIT_JITTER_SECONDS = 30
RATE_LIMIT_TOTAL_WAIT_SECONDS = 300


def _rate_limited(said: str) -> bool:
    """Whether this is the provider refusing for rate or quota rather than for what was asked."""
    lowered = said.lower()
    return any(
        mark in lowered
        for mark in ("429", "resource_exhausted", "resourceexhausted", "rate limit", "quota")
    )


def _refusal_in(broke: BaseException | None, ended: Any) -> str:
    """The refusal, when a turn or an exception is one for rate or quota, and empty otherwise.

    Two shapes because a backend has two ways of reporting a dead model call, and only one of them
    is an exception. The Vertex backend never raises: it catches everything and finishes the turn
    with `outcome` "failed" and the provider's own words in `error`. A retry that watched only for
    exceptions therefore never fired on the backend the hosted run actually uses, which is how a
    quota refusal went on costing a whole slice while the waiting code looked correct.
    """
    if broke is not None:
        said = f"{type(broke).__name__} {broke}"
        return said if _rate_limited(said) else ""
    if str(getattr(ended, "outcome", "") or "") != "failed":
        return ""
    said = str(getattr(ended, "error", "") or "")
    return said if _rate_limited(said) else ""


def _refusal_pause() -> float:
    """How long to wait before asking again, spread out so sessions do not return together.

    Sessions are refused at the same moment because they ask at the same moment, so a fixed wait has
    them all wake together and refuse together. Each wait clears the quota's minute and carries
    jitter on top, which is what breaks the lockstep.
    """
    return round(
        RATE_LIMIT_BACKOFF_SECONDS + random.uniform(0, RATE_LIMIT_JITTER_SECONDS), 1
    )


async def survive_refusal(
    run: Callable[[], Awaitable[Any]],
    *,
    what: str,
    on_event: Callable[..., Any] | None = None,
    enough: Callable[[], bool] | None = None,
) -> Any:
    """Run something that talks to the model, waiting out a refusal for rate or quota.

    Used by every session that drives its own model turn, because any of them can be the one the
    provider refuses, and losing the planning turn costs the whole suite rather than one slice.
    Anything that is not a rate or quota refusal is raised, so a real fault still fails fast.
    """
    waited = 0.0
    while True:
        broke: BaseException | None = None
        ended: Any = None
        try:
            ended = await run()
        except Exception as raised:  # noqa: BLE001 - classified below, re-raised when not a refusal
            broke = raised
        refusal = _refusal_in(broke, ended)
        pause = _refusal_pause()
        spent_out = waited + pause > RATE_LIMIT_TOTAL_WAIT_SECONDS
        if not refusal or spent_out or (enough is not None and enough()):
            if broke is not None:
                raise broke
            return ended
        waited += pause
        logger.warning(
            "%s was refused for rate or quota; waiting %ss (%ss of %ss spent): %s",
            what, pause, round(waited), RATE_LIMIT_TOTAL_WAIT_SECONDS, refusal[:200],
        )
        if on_event:
            on_event({"type": "waiting_on_provider", "what": what, "seconds": pause})
        await asyncio.sleep(pause)


@dataclass(frozen=True)
class Slice:
    """One writer's share of a suite: what to write, how much, and why it is worth writing."""

    use_case: str
    angle: str = ""
    count: int = 1
    why: str = ""

    def named(self) -> str:
        return f"{self.use_case}: {self.angle}" if self.angle else self.use_case


def even_slices(wanted: int, use_cases: list[str]) -> list[Slice]:
    """The fallback split, when nobody said how the work should be divided.

    Evenly, with the remainder going to the ones named first, because a contract lists its
    primary use cases before its marginal ones. It is a poor plan and it is meant to be: a use
    case with one real branch gets the same share as one with six, so the first pads and the
    second under-covers. It exists so a caller that supplies no plan still gets a suite.
    """
    if not use_cases:
        return []
    if wanted <= len(use_cases):
        return [Slice(use_case=case, count=1) for case in use_cases[:wanted]]
    each, extra = divmod(wanted, len(use_cases))
    return [
        Slice(use_case=case, count=each + (1 if i < extra else 0))
        for i, case in enumerate(use_cases)
    ]


def planned(wanted: int, use_cases: list[str], given: list[dict] | None) -> list[Slice]:
    """The split this suite will actually be written to.

    A plan supplied by the caller wins, because whoever is talking to the person has just read
    the contract and the world and knows which use cases have something in them. Sizing every
    use case identically is the thing that made suites pad in one place and under-cover in
    another, and the plan is the only part of the process that knows the difference.

    Anything the plan leaves out is filled in evenly, and anything it over-asks for is trimmed,
    so a plan can be rough without producing a suite nobody asked for.
    """
    if not given:
        return even_slices(wanted, use_cases)

    known = {case.strip().lower(): case for case in use_cases}
    slices: list[Slice] = []
    for one in given:
        if not isinstance(one, dict):
            continue
        case = str(one.get("use_case") or "").strip()
        if not case:
            continue
        # Match the contract's own wording where the plan paraphrased it, so a slice is filed
        # under a use case the coverage count recognises rather than a near-miss of one.
        case = known.get(case.lower(), case)
        try:
            count = max(1, int(one.get("count") or 1))
        except (TypeError, ValueError):
            count = 1
        slices.append(
            Slice(
                use_case=case,
                angle=str(one.get("angle") or "").strip(),
                count=count,
                why=str(one.get("why") or "").strip(),
            )
        )
    if not slices:
        return even_slices(wanted, use_cases)

    # Trim from the end rather than scaling everything down: the plan put its most valuable
    # slices first, and shaving one scenario off each is how a deliberate plan becomes an even
    # one again.
    total = sum(one.count for one in slices)
    while total > wanted and slices:
        last = slices[-1]
        if last.count > 1:
            slices[-1] = Slice(last.use_case, last.angle, last.count - 1, last.why)
        else:
            slices.pop()
        total = sum(one.count for one in slices)
    return slices


# Initial letters dealt out so parallel writers cannot invent the same people. Three per writer, which
# is enough choice to suit a scenario and keeps seven writers disjoint before the letters wrap; beyond
# that two writers share initials but still choose different names. Numbers are partitioned by a
# three-digit prefix instead: a hundred slots collided twice on a run with twenty slices, which is
# what the birthday arithmetic predicts, and a thousand makes it rare. A repeated verification code is
# a real collision; a repeated initial is not.
_NAME_LETTERS = "ABCDEFGHIJKLMNOPRSTVWY"
_LETTER_BLOCK = "abc"


def _slot(of: str, index: int) -> int:
    """A stable number for one slice, so its share of the value space does not move between passes.

    Using the position in the current batch looked right and was not: a second `generate_suite` pass
    numbers its slices from zero again, so its first writer is handed the same letters and the same
    leading digits as the first writer of the pass before it, and their codes collide. Measured on a
    377-scenario run: two verification codes shared, both between passes. Derived from the slice's own
    name instead, which does not change when the batch does, and deterministically so two runs of the
    same plan partition the same way.
    """
    if not of:
        return index
    return int(hashlib.sha256(of.encode("utf-8")).hexdigest()[:8], 16)


def callers_for(index: int, wanted: int, slice_name: str = "") -> str:
    """Which callers this slice should write, so the suite varies across slices as well as within.

    Instruction alone cannot do this. Each writer is blind to the others, so each independently
    picks the safest value and the suite converges on it: measured across three suites, more
    than half the callers came out "Professional and formal" and over three quarters American,
    with nobody doing anything wrong. Worse, a slice writing a single scenario has nothing to
    vary at all.

    So the spread is dealt out here, the same way the work is. Each slice is handed a different
    starting point in the platform's own vocabularies and told to begin there. It is a
    suggestion rather than a rule, because the caller still has to suit the scenario: a stolen
    phone is not a cheerful call whatever this hands out.
    """
    from .persona_guides import offered

    people = offered("personality")
    accents = offered("accent")
    if not people:
        return ""
    picks = [people[(index + step) % len(people)] for step in range(max(1, wanted))]
    said = (
        "\n\nStart from these callers, and move off them only where the scenario calls for "
        f"somebody else: {', '.join(picks)}."
    )
    # Writers cannot see each other, so left to themselves they invent the same handful of people and
    # the same round numbers, and the suite comes back with one name on a dozen scenarios and one
    # verification code shared between them. Partitioning the space of values costs nothing and makes
    # a collision impossible: each writer owns some initial letters and one leading digit, so no
    # shared list of names or codes has to exist for the values to stay distinct.
    slot = _slot(slice_name, index)
    # More letters where the slice is larger: a writer inventing twelve people from three initials
    # reuses a name, which is most of why distinctness measured 73 percent rather than the 90 the
    # suite rule wants.
    block = max(len(_LETTER_BLOCK), min(8, (max(1, wanted) + 2) // 3))
    letters = "".join(
        _NAME_LETTERS[(slot * block + step) % len(_NAME_LETTERS)] for step in range(block)
    )
    said += (
        f"\n\nEvery person you invent must have a given name beginning with one of {letters}, and "
        "every number you invent that the agent will look up, a code or a reference or an account "
        f"number, must begin with {slot % 1000:03d}. Other writers own the other letters and "
        "prefixes, so this is what keeps two scenarios from sharing a name or a code. Within your own "
        "slice, no two people may share a given name and no two scenarios may share a code or a "
        "reference: the prefix keeps you clear of other writers, it does not keep you clear of "
        "yourself."
    )
    if accents:
        # Spread several offered accents across this writer's callers rather than naming just one,
        # so the suite does not collapse to a single default accent and the agent's speech handling
        # is genuinely varied.
        spread = [
            accents[(index + step) % len(accents)]
            for step in range(min(len(accents), max(2, wanted)))
        ]
        said += (
            " Give your callers varied accents from the offered set, a different one per caller "
            f"where it fits rather than defaulting everyone to the same accent: {', '.join(spread)}. "
            "A suite where every caller sounds the same is a missed test of the agent's speech "
            "handling, so do not make them all American unless a scenario truly requires it."
        )
    return said


def brief_for(
    contract: AgentContract, mine: Slice, siblings: list[Slice], callers: str
) -> str:
    """What one writer is told: its share, what everyone else holds, and the bar.

    Written as a brief rather than a template because a writer that cannot see its siblings
    will otherwise write what they are writing. Naming their angles is cheaper than discovering
    the overlap at the merge and throwing the loser away.
    """
    others = "\n".join(f"  - {one.named()}" for one in siblings if one is not mine)
    aim = f"    {mine.use_case}"
    if mine.angle:
        aim += f"\n    Angle: {mine.angle}"
    if mine.why:
        aim += f"\n    Worth testing because: {mine.why}"

    return (
        f"Write {mine.count} scenario{'s' if mine.count != 1 else ''} for {contract.agent!r}, "
        "all of them within this one slice:\n\n"
        f"{aim}\n\n"
        + (
            "The rest of the suite is being written at the same time by others, covering:\n"
            f"{others}\n\nStay out of theirs. A scenario that strays is either a duplicate of "
            "somebody else's or a gap in yours.\n\n"
            if others
            else ""
        )
        + "Every scenario carries this use case verbatim in `use_case`, and its own one-line "
        "`branch` saying what makes it different from the others you write here. Branches are "
        "where the variety lives: the ordinary path, the branch that cannot be completed, the "
        "rule under pressure, state that has to carry across turns, the same request against a "
        "differently seeded world.\n\n"
        "What each one has to be, before you submit it:\n"
        "  - every value real, read out of the world with inspect_world, never invented\n"
        "  - an instruction that is a circumstance the person is living through, not a script "
        "of lines to say\n"
        "  - a setup that makes true whatever the instruction presumes, and a ready check that "
        "proves it\n"
        "  - a solution worked out with try_calls first, so the gates are not where you find "
        "out it cannot be passed\n"
        "  - sub-goals named from the shared catalogue, and checks that assert the right call "
        "with the right arguments or the right end state, never that something merely happened\n"
        "  - a scenario a competent agent could plausibly fail. If any correct implementation "
        "passes it for free, it teaches nothing and is not worth the run\n\n"
        "Look at the world first, and read the sub-goals already defined. Submit each scenario "
        "with submit_scenario and then stop: do not save, and do not ask what to do next. "
        "Whoever asked for this collects the suite and writes it." + callers
    )


async def _write_slice(
    contract: AgentContract,
    mine: Slice,
    siblings: list[Slice],
    *,
    index: int,
    destination: Path,
    on_event: Callable[..., Any] | None,
    ask: Callable[..., Any] | None,
) -> list[Scenario]:
    """One slice, written by its own session. Returns what it proved, unsaved."""
    server, kept = scenario_tools(
        contract,
        destination,
        destination,
        wanted=mine.count,
        can_save=False,
        start_from=[],
    )
    logger.info("slice starting: %s (wants %s)", mine.named(), mine.count)
    seen = 0

    def watch(event: Any) -> None:
        # Report as they land rather than at the end. A slice that proves its first scenario
        # four minutes in is the difference between a run that looks alive and one that does not.
        nonlocal seen
        if len(kept) != seen:
            seen = len(kept)
            logger.info("slice %s proved %s of %s", mine.named(), seen, mine.count)
        if on_event:
            on_event(event)

    # A slice writer never saves the suite; withholding the tool structurally means no backend
    # has to be told to deny it.
    sliced = SessionSpec(
        # The agent and its world come first, the method second. Grounding evidence read
        # before the instructions that operate on it is followed more closely than the
        # same evidence buried between the instructions and the task.
        system_prompt=(
            f"## This agent\n\n{contract.brief(with_data=True)}"
            f"\n\n## Its world\n\n{world_summary(destination)}"
            f"\n\n{load_skill(SKILL)}"
            # Whatever this kind of agent adds on top. A file under skills/kinds/ that
            # declares `applies_to: modality=<kind>` is appended here, so supporting a
            # new kind of agent is adding that file and nothing else.
            # `voicemail` gates the mailbox skill the way `modality` gates this one.
            + discovered_skills(
                modality=contract.modality,
                voicemail="on" if voicemail_enabled() else "off",
            )
            + f"\n\n## Your slice\n\nYou are writing only: {mine.named()}"
        ),
        servers={
            SCENARIO_SERVER: ToolServer(
                name=server.name,
                version=server.version,
                tools=[spec for spec in server.tools if spec.name != "save_scenarios"],
            )
        },
        cwd=str(destination.parent if destination.parent.exists() else Path.cwd()),
        max_turns=turns_for(mine.count),
        model=chosen_model(),
        ask=ask,
        thinking=True,
        idle_timeout_seconds=QUIET_WHILE_WRITING_SECONDS,
    )
    # Each writer drives its own model session, so several of them together are a request rate. When
    # the provider refuses one, the slice is not wrong and its brief is still worth writing, so wait
    # for the quota to recover and run it again. `kept` belongs to this slice's own tool server, so a
    # second attempt adds to what the first proved rather than starting over.
    async def attempt_slice() -> Any:
        stage = Stage(sliced, name=f"{SKILL}:{mine.named()[:40]}")
        async with stage:
            return await stage.say(
                brief_for(contract, mine, siblings, callers_for(index, mine.count)),
                on_event=watch,
            )

    try:
        # A refusal for rate or quota is waited out; the slice keeps what it already proved, so a
        # second attempt adds to it. Stop early if the count is already met.
        await survive_refusal(
            attempt_slice,
            what=f"slice {mine.named()}",
            on_event=on_event,
            enough=lambda: len(kept) >= mine.count,
        )
    except Exception as broke:  # noqa: BLE001 - one slice failing must not lose the others
        logger.warning("slice %s failed after %s: %s", mine.named(), len(kept), broke)
        if on_event:
            on_event({"type": "slice_failed", "slice": mine.named(), "why": str(broke)[:300]})
        return list(kept)
    logger.info("slice %s finished with %s of %s", mine.named(), len(kept), mine.count)
    return list(kept)


def merged(written: list[list[Scenario]]) -> list[Scenario]:
    """One suite out of several writers, with folder-name collisions renamed rather than dropped.

    Asking for twenty scenarios has to return twenty. Two scenarios may legitimately share a use
    case and a branch and still test different things, so sharing them is not a reason to discard
    one; an earlier version dropped those and quietly returned eighteen.

    The one collision that cannot be tolerated is the folder name, because the folder is where a
    scenario lives on disk and the loser would overwrite the winner. Those are given a numbered
    suffix instead of being thrown away, so nothing generated is ever lost.
    """
    suite: list[Scenario] = []
    taken: set[str] = set()
    for batch in written:
        for one in batch:
            if one.name in taken:
                stem, suffix = one.name, 2
                while f"{stem}-{suffix}" in taken:
                    suffix += 1
                one = one.model_copy(update={"name": f"{stem}-{suffix}", "scenario_key": ""})
                logger.info("renamed a duplicate folder name to %s", one.name)
            taken.add(one.name)
            suite.append(one)
    return suite


def _suite_summary(suite: list[Scenario]) -> str:
    """The whole suite as a reviewer needs to see it: what each row claims to test."""
    return "\n".join(
        f"  {one.name} | use case: {one.use_case} | branch: {one.branch} | passes when: {one.tests}"
        for one in suite
    )


async def gaps_in(
    contract: AgentContract,
    suite: list[Scenario],
    *,
    destination: Path,
    wanted: int,
    ask: Callable[..., Any] | None = None,
) -> list[Slice]:
    """What the finished suite is missing, as slices that would fill it.

    Nobody looks at a suite written in parallel. Each writer sees its own slice and the merge
    only removes collisions, so a use case that came back one short, or an obvious branch that
    every writer assumed somebody else had, survives to the end and nobody notices. This is the
    one pass that reads the suite as a whole.
    """
    if not suite:
        return []
    found: list[Slice] = []

    @tool(
        "submit_gaps",
        "The gaps worth filling in this suite, as the slices that would fill them. Return "
        "nothing when the suite covers what it should: a suite that is finished is a real "
        "answer, and inventing work to report is worse than saying so.",
        schema(
            {
                "gaps": {
                    "type": "array",
                    "description": "One entry per gap. Empty when the suite is covering what "
                    "it should.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "use_case": {"type": "string"},
                            "angle": {
                                "type": "string",
                                "description": "The scenario that is missing, in one line.",
                            },
                            "why": {"type": "string"},
                        },
                        "required": ["use_case", "angle"],
                    },
                }
            },
            ["gaps"],
        ),
    )
    async def submit_gaps(args: dict[str, Any]) -> dict[str, Any]:
        for one in args.get("gaps") or []:
            if not isinstance(one, dict):
                continue
            case = str(one.get("use_case") or "").strip()
            if case:
                found.append(
                    Slice(
                        use_case=case,
                        angle=str(one.get("angle") or "").strip(),
                        count=1,
                        why=str(one.get("why") or "").strip(),
                    )
                )
        return {
            "content": [
                {"type": "text", "text": f"{len(found)} gap(s) recorded. Nothing else to do."}
            ]
        }

    server = tool_server(name=REVIEW_SERVER, version="0.1.0", tools=[submit_gaps])
    review = SessionSpec(
        system_prompt=(
            "You are reviewing a suite of tests somebody else wrote for an AI agent, in "
            "parallel, each writer blind to the others. Your only job is to say what is "
            "missing.\n\n"
            "Look for: a use case of this agent that nothing covers; a use case covered only "
            "on its ordinary path, where the branch that cannot be completed or the rule under "
            "pressure is the interesting one; two rows that are the same test under different "
            "names, leaving the branch one of them claimed uncovered.\n\n"
            "Judge coverage of the agent, not of the plan. Do not ask for more of what is "
            "already well covered, and do not report a gap you cannot name a scenario for. "
            "A suite of the right size that covers what matters is finished, and saying so is "
            f"the useful answer.\n\n## This agent\n\n{contract.brief()}"
        ),
        servers={REVIEW_SERVER: server},
        cwd=str(destination.parent if destination.parent.exists() else Path.cwd()),
        max_turns=8,
        model=chosen_model(),
        ask=ask,
    )
    stage = Stage(review, name=f"{SKILL}:review")
    try:
        async with stage:
            # The suite-level checks are reported at save and enforced nowhere, deliberately:
            # refusing a save leaves proved work in memory only. This is the one place that can act
            # on them, because it is the only pass that reads the whole suite and can commission
            # replacements. Without this they are a message nobody reads.
            skew = suite_diversity_problems(suite)
            already_wrong = (
                "Reading the suite as a whole, these are already wrong with it, and a gap that "
                "fixes one of them is worth more than a new use case:\n"
                + "\n".join(f"- {problem}" for problem in skew)
                + "\n\n"
                if skew
                else ""
            )
            asked = (
                f"This suite has {len(suite)} scenarios against a target of {wanted}:\n\n"
                f"{_suite_summary(suite)}\n\n"
                + already_wrong
                + "Say what it is missing, then submit_gaps. Submit an empty list if it is "
                "covering what it should."
            )
            await survive_refusal(
                lambda: stage.say(asked), what="the suite review", on_event=on_event
            )
    except Exception:  # noqa: BLE001 - a review that fails leaves the suite as written
        return []
    return found


async def write_in_parallel(
    contract: AgentContract,
    *,
    out: Path | None = None,
    wanted: int = 10,
    use_cases: list[str] | None = None,
    slices: list[dict] | None = None,
    at_once: int = AT_ONCE,
    rounds: int = TOP_UP_ROUNDS,
    on_event: Callable[..., Any] | None = None,
    ask: Callable[..., Any] | None = None,
) -> list[Scenario]:
    """Write a suite with one session per slice, review it, fill what it missed, and save once.

    Sequentially, a suite costs roughly three turns a scenario against one budget, which is why
    asking for forty stopped around twenty-five. Here the work is split into slices that run at
    the same time, so the wall clock is the slowest slice rather than the sum of all of them.

    Saving stays here, once, for a reason: ``save_scenarios`` regenerates the index and deletes
    any folder it does not know about, so letting the writers save would have each of them
    remove the others' work.
    """
    destination = out or artifact_dir(contract.agent)
    cases = [case for case in (use_cases or contract.real_use_cases) if case.strip()]
    if not cases and not slices:
        # Nothing to partition on. One writer, the ordinary path, rather than no scenarios.
        return await write(contract, out=destination, wanted=wanted, on_event=on_event, ask=ask)

    at_once = max(1, min(at_once or AT_ONCE, MOST_AT_ONCE))
    allocation = planned(wanted, cases, slices)
    logger.info(
        "writing %s scenarios across %s slices, %s at a time: %s",
        wanted,
        len(allocation),
        at_once,
        ", ".join(f"{one.named()} x{one.count}" for one in allocation),
    )
    if on_event:
        on_event(
            {
                "type": "planned",
                "slices": [(one.named(), one.count) for one in allocation],
                "at_once": at_once,
            }
        )

    limit = asyncio.Semaphore(at_once)

    async def guarded(mine: Slice, siblings: list[Slice], index: int) -> list[Scenario]:
        async with limit:
            return await _write_slice(
                contract,
                mine,
                siblings,
                index=index,
                destination=destination,
                on_event=on_event,
                ask=ask,
            )

    written = await asyncio.gather(
        *(guarded(one, allocation, index) for index, one in enumerate(allocation)),
        return_exceptions=False,
    )
    proved = merged([load_scenarios(destination), *written])
    # What a writer proved but never handed back, because its session died after proving it. Matched
    # by name so a scenario already in hand is not added twice under a numbered name.
    recovered = [
        one for one in journalled(destination) if one.name not in {x.name for x in proved}
    ]
    if recovered:
        logger.warning(
            "recovered %s scenarios from the journal that no writer returned: %s",
            len(recovered),
            ", ".join(one.name for one in recovered),
        )
        if on_event:
            on_event({"type": "recovered", "kept": len(recovered)})
    suite = merged([proved, recovered])

    # Read the whole thing and fill what nobody covered. Bounded, because a reviewer asked
    # twice will always find something smaller to say.
    for _ in range(max(0, rounds)):
        if len(suite) >= wanted:
            break
        missing = await gaps_in(
            contract, suite, destination=destination, wanted=wanted, ask=ask
        )
        missing = missing[: max(0, wanted - len(suite))]
        if not missing:
            break
        if on_event:
            on_event({"type": "topping_up", "slices": [one.named() for one in missing]})
        logger.info(
            "topping up %s of %s with %s more slices: %s",
            len(suite),
            wanted,
            len(missing),
            ", ".join(f"{one.named()} x{one.count}" for one in missing),
        )
        more = await asyncio.gather(
            *(
                guarded(one, missing, len(allocation) + index)
                for index, one in enumerate(missing)
            ),
            return_exceptions=False,
        )
        before = len(suite)
        suite = merged([suite, *more])
        allocation = [*allocation, *missing]
        if len(suite) == before:
            break

    write_scenarios(suite, destination, load_catalogue(destination))
    logger.info("suite saved: %s of %s asked for", len(suite), wanted)
    if on_event:
        on_event({"type": "saved", "kept": len(suite), "asked": wanted})
    return load(destination)


async def write(
    contract: AgentContract,
    *,
    out: Path | None = None,
    wanted: int = 10,
    follow_ups: list[str] | None = None,
    on_event: Callable[..., Any] | None = None,
    ask: Callable[..., Any] | None = None,
    max_turns: int = 0,
) -> list[Scenario]:
    """Run the stage start to finish. Returns whatever scenarios were saved."""
    stage, destination = open_stage(
        contract, out=out, wanted=wanted, ask=ask, max_turns=max_turns
    )
    async with stage:
        # The planning turn is the expensive one to lose: a refusal here costs the whole suite, not
        # one slice, so it waits the same way a writer does.
        await survive_refusal(
            lambda: stage.say(opening(contract, wanted), on_event=on_event),
            what="the opening turn",
            on_event=on_event,
        )
        for follow_up in follow_ups or []:
            await survive_refusal(
                lambda message=follow_up: stage.say(message, on_event=on_event),
                what="a follow-up turn",
                on_event=on_event,
            )
    return load(destination)
