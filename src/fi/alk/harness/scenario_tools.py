"""The tools that write scenarios, and the gates that decide one may be kept.

A scenario is accepted by being *proved*, not by looking right. ``submit_scenario`` puts it
through three gates, in order: the world must end up holding what the scenario presumes, the
reference solution must pass the scenario's own checks, and those same checks must fail when
nothing is done at all.

Every gate is code. No model is asked whether a scenario is good; the environment decides. A
scenario that clears all three is written out as its own folder of runnable files.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .backends import tool, tool_server

from .amend import add_rule, drop_rule, fix_tool, widen
from .catalogue import (
    Catalogue,
    SubGoal,
    load_catalogue,
    save_catalogue,
    validate_sub_goal,
)
from .contract import CALL_DIRECTIONS, AgentContract
from .folder import INDEX, SCENARIOS, apply_setup, read_all, write_folder, write_index
from .prove import play_reference_step, prepared, prove
from .scenario import (
    ANSWERED_BY,
    CALLER_AWARENESS,
    VOICEMAIL_STYLES,
    Scenario,
    Step,
    contract_sequence_problems,
    suite_diversity_problems,
    validate_scenario,
    voicemail_enabled,
)
from .simulator import load_simulator_prompt
from .tools import brief, schema
from .world.snapshot import restore

logger = logging.getLogger(__name__)

SCENARIO_SERVER = "scenarios"


def _mailbox_fields() -> dict[str, Any]:
    """The scenario fields that only mean anything when a mailbox may answer.

    Empty when ``ALK_VOICEMAIL_SCENARIOS`` is off: withheld rather than offered and then refused,
    because a field in the schema is an invitation.
    """
    if not voicemail_enabled():
        return {}
    return {
        "answered_by": {
            "type": "string",
            "enum": list(ANSWERED_BY),
            "description": "Who picked up, and only for a scenario that states call_direction "
            "outbound. Leave it out for the ordinary case where a person answers. 'voicemail' "
            "replaces the person with a mailbox that plays its greeting once and then says nothing "
            "whatever the agent asks, which tests whether the agent notices it is talking to a "
            "machine, leaves a message that stands on its own, and stops. A mailbox can supply "
            "nothing, so such a scenario never asks the agent to collect a value or reach agreement.",
        },
        "voicemail_style": {
            "type": "string",
            "enum": list(VOICEMAIL_STYLES),
            "description": "Which kind of mailbox answered. 'personal' carries the person's name, "
            "'carrier' names nobody, 'operator' is a long announcement a careless agent talks over, "
            "'full' cannot record at all and is the only style with no tone. State one rather than "
            "leaving it out: left out it is personal, the easiest of the four, and most suites have "
            "room for only one mailbox.",
        },
    }


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


# Below this, fanning out costs more than it saves: measured at ten scenarios, fifty four turns
# without writers against a hundred and nineteen with, for output that was identical scenario by
# scenario. Above it, one at a time runs out of turns long before the number is reached.
FEWEST_WORTH_DELEGATING = 20


def worth_delegating(wanted: int) -> bool:
    """Whether a request of this size should be written by several writers at once.

    Decided from the number asked for, which is the one fact that settles it, rather than from a
    setting. An environment variable had to survive four separate allowlists between the platform
    and the process that reads it, three of which silently dropped it, and it exposed as an
    operator choice something no operator should have to make.
    """
    return int(wanted or 0) >= FEWEST_WORTH_DELEGATING


def persona_field(name: str) -> dict[str, Any]:
    """The schema for one persona field, carrying the platform's own values where it has them.

    Offered as an enum so the values arrive right the first time. Without the platform's model
    to read, it stays a plain string rather than an enum of nothing.
    """
    from .persona_guides import offered

    allowed = offered(name)
    return {"type": "string", "enum": allowed} if allowed else {"type": "string"}


def persona_vocabulary_note() -> str:
    """A sentence about why the persona fields are constrained, when they are."""
    from .persona_guides import vocabulary

    if not vocabulary():
        return ""
    return (
        " The listed values are the ones the platform understands: they carry behaviour "
        "guidance into the call and select the caller's voice. Anything else about this person "
        "goes in metadata, where it is free text."
    )


def write_scenarios(
    scenarios: list[Scenario], destination: Path, catalogue: Catalogue | None = None
) -> Path:
    """Write every scenario out as its own folder, and regenerate the index over them."""
    catalogue = catalogue if catalogue is not None else load_catalogue(destination)
    if not scenarios and (Path(destination) / SCENARIOS).is_dir():
        # An empty save would take every folder with it, because dropping a scenario is expressed by
        # saving the suite without it. Nothing legitimately saves an empty suite over a full one: a
        # session whose own list is empty is a session that has not written anything yet, and on a run
        # this emptied 30 folders and then let a second fan-out pass write the suite again from zero.
        logger.warning(
            "refusing to save an empty suite over %s existing scenarios",
            len(load_scenarios(destination)),
        )
        return Path(destination) / INDEX
    for one in scenarios:
        write_folder(one, catalogue, destination)
    _forget_dropped(scenarios, destination)
    return write_index(scenarios, destination)


def _forget_dropped(scenarios: list[Scenario], destination: Path) -> None:
    """Remove the folders of scenarios that are no longer in the suite.

    The folders are the truth, and they are what gets read back. Writing the survivors without
    taking the others away means a dropped scenario returns on the next load, still failing, and
    dropping it appears to do nothing at all.
    """
    import shutil

    root = Path(destination) / SCENARIOS
    if not root.exists():
        return
    keeping = {one.name for one in scenarios}
    for folder in root.iterdir():
        if folder.is_dir() and folder.name not in keeping:
            shutil.rmtree(folder)


def load_scenarios(destination: Path) -> list[Scenario]:
    """Every scenario on disk, read from its folder.

    The folders are the truth. The index beside them is regenerated from these, so it can
    describe them but never contradict them.
    """
    return read_all(destination)


JOURNAL = "written.jsonl"


def journal_scenario(scenario: Scenario, destination: Path) -> None:
    """Append one proved scenario to the journal, which is the only record a dead writer leaves.

    A delegated writer cannot write folders: saving the suite deletes every folder it does not know
    about, so a writer persisting its own would delete its siblings' work. It therefore keeps what it
    proved in memory, and until now a writer whose session died took its scenarios with it. A whole
    hundred-scenario run was lost that way. This file is append-only and nothing prunes it, so what
    was proved survives the session that proved it.
    """
    try:
        path = Path(destination) / JOURNAL
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(scenario.model_dump(), ensure_ascii=False) + "\n")
    except Exception as broke:  # noqa: BLE001 - a scenario is never lost over bookkeeping
        logger.warning("could not journal %s: %s", scenario.name, broke)


def journalled(destination: Path) -> list[Scenario]:
    """Every scenario the journal holds, newest wins, skipping anything unreadable.

    Appended by writers as they prove, so a retried slice re-journals and the same name appears more
    than once. Read by the caller that saves, to recover what a writer proved and never returned.
    """
    path = Path(destination) / JOURNAL
    if not path.is_file():
        return []
    found: dict[str, Scenario] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            one = Scenario.model_validate(json.loads(line))
        except Exception:  # noqa: BLE001 - a half-written line is expected while writers run
            continue
        found[one.name] = one
    return list(found.values())


def accept_scenario(
    payload: dict[str, Any],
    *,
    world_root: Path,
    catalogue: Catalogue,
    kept: list[Scenario],
    simulator_prompt: str = "",
    hard_constraints: list[str] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Validate one scenario, then prove it. A plain function so both halves are testable.

    ``persist`` is off for a writer that shares the destination with siblings: writing the suite
    removes every folder not in the writer's own list, so persisting here would delete whatever
    the others have proved. Those writers keep their work in ``kept`` and the caller saves once.
    """
    try:
        scenario = Scenario.model_validate(payload)
    except Exception as invalid:
        return _err(f"Not kept. {invalid}"[:600])

    # Read against the world this scenario actually runs in, so a setup that creates the table
    # a check reads is not reported as referring to something that does not exist.
    trial, _applied, _ready = prepared(scenario, world_root)
    try:
        problems = validate_scenario(
            scenario, catalogue, trial.state(), simulator_prompt
        )
        problems.extend(contract_sequence_problems(scenario, hard_constraints or []))
    finally:
        trial.close()

    if problems:
        return _err(
            "Not kept. Fix these and submit again:\n  - " + "\n  - ".join(problems)
        )

    proof = prove(scenario, catalogue, world_root)
    if not proof.holds:
        said = f"Not kept. {proof.why()}"
        # Code written against the wrong collection shape is the commonest way setup, ready and a
        # check fail here, and the exception alone does not say which collections are mappings and
        # which are lists. The world is asked, so the answer names them.
        if "attribute" in said.lower() or "not subscriptable" in said.lower():
            world = restore(world_root)
            try:
                said += f"\n\n{world.shapes()}"
            finally:
                world.close()
        return _err(said)

    replaced = any(one.name == scenario.name for one in kept)
    kept[:] = [one for one in kept if one.name != scenario.name]
    kept.append(scenario)
    # A proved scenario is already valuable work. Persist it immediately so a stopped model,
    # browser refresh, process restart, or later scenario failure cannot make the UI say none
    # were written. ``save_scenarios`` remains the suite-level diversity/finality gate.
    if persist:
        write_scenarios(kept, world_root, catalogue)
    else:
        # A writer sharing the destination cannot write folders, so the journal is where its proved
        # work survives the session that proved it.
        journal_scenario(scenario, world_root)
    # Say what the proof did not cover. On a lane where the target's tools have no endpoints, every
    # solution step is recorded without running, so "all three gates pass" is true and misleading:
    # the checks were exercised, the solution was not.
    unproved = (
        "\nNOT PROVED: "
        + f"{len(proof.assumed)} of {len(scenario.solution)} solution steps were recorded without "
        "running, because these tools have no endpoint in this environment: "
        + ", ".join(proof.assumed)
        + ". The checks ran, the solution did not, so this scenario is kept as written rather than "
        "as demonstrated."
        if proof.assumed
        else ""
    )
    return _ok(
        f"{scenario.name} {'replaced' if replaced else 'kept'}. All three gates pass: the world "
        "is ready for it, the reference solution passes its checks, and those checks fail when "
        f"nothing is done.{unproved}\n{len(kept)} so far: " + ", ".join(one.name for one in kept)
    )


def not_ready(kept: list[Scenario], wanted: int, catalogue: Catalogue) -> list[str]:
    """Why this suite is not worth saving yet."""
    problems: list[str] = []
    if len(kept) < wanted:
        problems.append(
            f"{len(kept)} of the {wanted} asked for. The ones that find something are usually "
            "the awkward ones, so this is worth finishing rather than stopping here. If nobody "
            f"asked for {wanted}, record what they did ask for with aim_for."
        )
    elif len(kept) > wanted:
        problems.append(
            f"{len(kept)} scenarios against a target of {wanted}. If they asked for more, "
            "aim_for records the new size; reopening a suite starts with the target set to what "
            "is already there, so adding to one always reads like this. If you wrote extra "
            "nobody asked for, drop_scenario takes them off."
        )
    # Two scenarios claiming the same use case are either the same test twice, or one of them is
    # mislabelled. Both happened in the same suite: a delivered-order refusal was filed under
    # "cancel a pending order", which is neither what it tests nor distinguishable afterwards
    # from the scenario that really does test that. A use case is how coverage is counted, so a
    # duplicate quietly overstates it.
    # Keyed on the pair, not the use case alone. A use case fans out into several branches and
    # each is a separate test, so keying on the use case alone caps a suite at one scenario per
    # use case — which is how a request for forty against fourteen use cases became unsaveable.
    claimed: dict[tuple[str, str], list[str]] = {}
    for one in kept:
        case = (one.use_case or "").strip().lower()
        branch = (one.branch or "").strip().lower()
        if case:
            claimed.setdefault((case, branch), []).append(one.name)
    for (case, branch), names in claimed.items():
        if len(names) > 1:
            where = f"{case!r}" if not branch else f"{case!r} / {branch!r}"
            problems.append(
                f"{' and '.join(names)} both claim {where}. Give each the branch it actually "
                "exercises, or drop the one that duplicates the other. Coverage is counted by "
                "use case and branch, so two scenarios sharing both hides a gap."
            )

    # Sub-goals are shared so results roll up. A suite where every scenario invents its own is a
    # suite whose results cannot be added together.
    used = [name for one in kept for name in one.sub_goals]
    if kept and len(used) > 2 and len(set(used)) == len(used):
        problems.append(
            "no sub-goal is used by more than one scenario, so nothing rolls up across the "
            "suite. Reuse the catalogue where the same thing is being checked."
        )
    return problems


def scenario_tools(
    contract: AgentContract,
    world_root: Path,
    destination: Path,
    *,
    wanted: int,
    can_save: bool = True,
    start_from: list[Scenario] | None = None,
) -> tuple[Any, list[Scenario]]:
    """A server for writing scenarios against one built environment.

    ``can_save`` is what makes several writers safe at once. Saving rewrites the index and
    removes any folder not in the saver's own list, so two writers saving concurrently delete
    each other's work. A writer that only submits keeps its scenarios in ``kept``, and whoever
    spawned it merges the lists and writes once.

    ``start_from`` seeds that list. A parallel writer starts empty rather than from disk, so it
    is never counted as already having what a sibling wrote.
    """
    kept: list[Scenario] = (
        list(start_from) if start_from is not None else load_scenarios(destination)
    )
    catalogue = load_catalogue(destination)
    simulator_prompt = load_simulator_prompt(destination)
    target = {"count": wanted}
    exploration = {"since_submit": 0}

    # ``branch`` is required because coverage is counted on the use case and branch pair, and the
    # merge drops a repeat of that pair. A writer that leaves it out gives every scenario in its
    # slice the same pair, and all but the first are silently thrown away.
    scenario_required = ["name", "branch", "instruction", "solution", "sub_goals"]
    if contract.conversational:
        scenario_required.append("persona")

    @tool(
        "inspect_world",
        "Look at what is in the world. Without a table, lists the tables and how many rows each "
        "holds; with one, returns rows from it. `matching` is plain text, not SQL.",
        schema({"table": str, "limit": int, "matching": str}, []),
    )
    async def inspect_world(args: dict[str, Any]) -> dict[str, Any]:
        world = restore(world_root)
        try:
            state = world.state()
            table = str(args.get("table") or "")
            if not table:
                lines = [f"{n}: {len(r)} rows" for n, r in sorted(state.items())]
                if catalogue.sub_goals:
                    lines.append(
                        "\nsub-goals available: " + ", ".join(sorted(catalogue.names()))
                    )
                return _ok("\n".join(lines) or "this world has no tables")
            if table not in state:
                return _err(
                    f"no table {table!r}; this world has {', '.join(sorted(state))}"
                )
            rows = state[table]
            # A provisioned store keys rows by id; the generated one keeps a list. Either
            # way what gets shown is rows, not the index over them.
            if isinstance(rows, dict):
                rows = list(rows.values())
            matching = str(args.get("matching") or "").strip()
            if matching:
                needle = matching.lower()
                found = [
                    r for r in rows if needle in json.dumps(r, default=str).lower()
                ]
                if not found:
                    return _ok(
                        f"nothing in {table} contains {matching!r}, but it holds {len(rows)} rows."
                    )
                rows = found
            shown = rows[: int(args.get("limit") or 20)]
            return _ok(
                f"{len(rows)} rows, showing {len(shown)}:\n"
                + "\n".join(json.dumps(r, default=str) for r in shown)
            )
        finally:
            world.close()

    @tool(
        "inspect_scenario",
        "Read one already-kept scenario in full before replacing it. This is the source of "
        "truth for incremental edits after a restart; do not reconstruct a saved scenario from "
        "memory or from its one-line suite summary.",
        schema({"name": str}, ["name"]),
    )
    async def inspect_scenario(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or "")
        found = next((one for one in kept if one.name == name), None)
        if found is None:
            return _err(
                f"no scenario called {name!r}; available: "
                + (", ".join(one.name for one in kept) or "none")
            )
        return _ok(found.model_dump_json(indent=2))

    @tool(
        "try_calls",
        "Run calls against a throwaway copy of the world and see the state they leave. Use it to "
        "work out a scenario's solution and what its checks should assert.\n\n"
        "`setup_code` is optional: pass the same code you intend to give the scenario and the "
        "calls run against a world it has already changed, so you can see what the agent would "
        "actually face. Nothing is saved.",
        schema({"calls": list, "setup_code": str}, ["calls"]),
    )
    async def try_calls(args: dict[str, Any]) -> dict[str, Any]:
        if exploration["since_submit"] >= 4:
            return _err(
                "Four throwaway probes have run since the last saved scenario. Submit and prove "
                "one scenario now; if its gate identifies a concrete problem, use the next "
                "probe to correct that problem. Do not map the whole suite before saving work."
            )
        exploration["since_submit"] += 1
        world = restore(world_root)
        try:
            world.reset()
            trial = Scenario(name="trial", setup_code=str(args.get("setup_code") or ""))
            applied = apply_setup(trial, world)
            if not applied.ok:
                return _err(f"the setup did not run: {applied.said}")
            world.calls = []
            lines: list[str] = []
            for step in args.get("calls") or []:
                if not isinstance(step, dict):
                    return _err("each call must be an object with a tool and arguments")
                try:
                    reference_step = Step.model_validate(step)
                except Exception as invalid:
                    return _err(f"invalid reference call: {invalid}"[:600])
                call = play_reference_step(world, reference_step)
                if call.refused:
                    lines.append(f"{call.name}: refused — {call.error}")
                elif not call.ok:
                    lines.append(f"{call.name}: CRASHED — {call.error}")
                else:
                    lines.append(f"{call.name}: ok — {brief(call.result)}")
            state = world.state()
            lines.append(
                "state afterwards: "
                + ", ".join(f"{n}.count={len(r)}" for n, r in sorted(state.items()))
            )
            for name, rows in sorted(state.items()):
                if rows and len(rows) <= 6:
                    lines.append(f"{name}: " + brief(rows, limit=1200))
            return _ok("\n".join(lines) or "no calls were made")
        finally:
            world.close()

    @tool(
        "add_sub_goal",
        "Add a named thing this agent can be checked on, shared by every scenario that needs it. "
        "`check` is Python: define check(world, calls) returning a sentence when something is "
        "wrong, or None when it held. `world` is the environment afterwards; `calls` is every "
        "tool call made, each with .name, .arguments, .ok and .refused — so a check can insist a "
        "call happened with the right arguments, not merely that it happened. Check the named "
        "outcome using the smallest sufficient evidence. Do not require preparatory or discovery "
        "calls when a later successful state-changing call already proves the outcome; valid "
        "agents may reach the same result through different safe trajectories.\n\n"
        "Use `judged` only where nothing observable settles it, saying what a model must decide "
        "and why code cannot.",
        schema(
            {"name": str, "what": str, "check": str, "judged": str}, ["name", "what"]
        ),
    )
    async def add_sub_goal(args: dict[str, Any]) -> dict[str, Any]:
        sub_goal = SubGoal(
            name=str(args.get("name") or ""),
            what=str(args.get("what") or ""),
            check=str(args.get("check") or ""),
            judged=str(args.get("judged") or ""),
        )
        problems = validate_sub_goal(sub_goal)
        if problems:
            return _err("Not added:\n  - " + "\n  - ".join(problems))
        catalogue.sub_goals = [
            one for one in catalogue.sub_goals if one.name != sub_goal.name
        ]
        catalogue.sub_goals.append(sub_goal)
        save_catalogue(catalogue, destination)
        return _ok(
            f"{sub_goal.name} added"
            + ("" if sub_goal.deterministic() else " (judged, not deterministic)")
            + f". The catalogue has {len(catalogue.sub_goals)}: "
            + ", ".join(sorted(catalogue.names()))
        )

    @tool(
        "submit_scenario",
        "Keep one scenario. It is put through three gates before it is kept, and told which one "
        "failed if any does:\n"
        "  1. ready     — the world is restored, setup_code runs, then ready_code. The world "
        "must end up holding what this scenario presumes.\n"
        "  2. solvable  — the reference solution is played through that world and the checks of "
        "every sub-goal named must pass.\n"
        "  3. not vacuous — the same checks run again with nothing done at all, and must fail.\n\n"
        "A scenario that clears all three is written out as its own folder of runnable files.",
        schema(
            {
                "name": {
                    "type": "string",
                    "description": "Short identifier, lower case with hyphens or underscores. "
                    "It becomes this scenario's folder name.",
                },
                "use_case": {
                    "type": "string",
                    "description": "Which of the agent's use cases this belongs to.",
                },
                "branch": {
                    "type": "string",
                    "description": "The condition that makes this scenario different from the "
                    "others in the same use case, in one line: what is true here that is not "
                    "true of its siblings.",
                },
                "tests": {
                    "type": "string",
                    "description": "One line: what this scenario is trying to find out.",
                },
                "background_noise": {
                    "type": "string",
                    "description": "Where the caller is phoning from: street, transit, vehicle, "
                    "outdoors, retail, office or home. Name it whenever the instruction implies "
                    "somewhere, a caller leaving a hotel or standing on a street is not in a "
                    "quiet room. Left out, it is decided from the scenario name.",
                },
                "call_direction": {
                    "type": "string",
                    "enum": list(CALL_DIRECTIONS),
                    "description": "Who placed the call. Match the contract unless this scenario "
                    "deliberately tests the other one. Outbound changes what the instruction has "
                    "to be: a person who did not dial has no objective to pursue.",
                },
                "caller_awareness": {
                    "type": "string",
                    "enum": list(CALLER_AWARENESS),
                    "description": "Outbound only, and the thing the scenario is really varying: "
                    "whether this person was told to expect the call, half remembers arranging "
                    "something, or has no idea why anyone is ringing. Left out it is unaware, "
                    "which the agent has to work hardest for.",
                },
                **_mailbox_fields(),
                "instruction": {
                    "type": "string",
                    "description": "What this person is trying to achieve, written to them. "
                    "State the objective first, in their own terms, so they pursue it rather "
                    "than narrate a situation: 'Get the cancellation fee refunded', not 'You "
                    "were charged a fee'. On an OUTBOUND scenario invert that: they did not "
                    "call anyone and have no objective, so give them their situation and what "
                    "they would agree to if asked, never an opening request. Then give them "
                    "everything they need to hold the "
                    "conversation without inventing anything: the facts they know, the values "
                    "they can be asked for, and what they will only say once asked. Every value "
                    "real and read out of the world.\n"
                    "Write only what this person knows before the call starts. Never write what "
                    "the agent will do, in any phrasing: not what it will send, offer, ask for, "
                    "disclose or decide, and no closing line about what counts as done. Those "
                    "are the behaviours under test, and a person primed to expect them plays "
                    "along whether or not they happen, so the check passes on a conversation "
                    "that never earned it. Give them the value, the preference or the problem "
                    "they arrived with, and let the agent's handling of it be what is measured.\n"
                    "Test every sentence by asking whether this person could say it out loud. "
                    "They have never seen the agent's design, so a parenthetical explaining "
                    "where the agent should find a value fails that test just as much as a "
                    "sentence predicting what it will say. Worst of all is agreeing in advance "
                    "to something the agent has not done yet: that hands over a pass the "
                    "conversation never earned.",
                },
                "persona": {
                    "type": "object",
                    "description": "Who the simulated person is, separate from the task. Use "
                    "the established voice-scenario shape and only grounded, test-relevant "
                    "details. This fills the simulator prompt's persona slot."
                    + persona_vocabulary_note(),
                    "properties": {
                        "name": {"type": "string"},
                        "gender": persona_field("gender"),
                        "age_group": persona_field("age_group"),
                        "occupation": persona_field("occupation"),
                        "location": persona_field("location"),
                        "personality": persona_field("personality"),
                        "communication_style": persona_field("communication_style"),
                        "initial_message": {
                            "type": "string",
                            "description": "The caller's natural opening request, specific to "
                            "this scenario. Do not use a generic greeting.",
                        },
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "languages": {
                            "type": "array",
                            "items": persona_field("languages"),
                        },
                        "accent": persona_field("accent"),
                        "multilingual": {"type": "boolean"},
                        "metadata": {"type": "object"},
                    },
                    "required": [
                        "name",
                        "personality",
                        "communication_style",
                        "initial_message",
                        "languages",
                        "accent",
                        "keywords",
                    ],
                },
                "variables": {
                    "type": "object",
                    "description": "Any other slot the simulator prompt asks for, by name. Do "
                    "not put persona here; use the structured persona field.",
                },
                "fixture": {
                    "type": "object",
                    "description": "Readable manifest of the concrete data behind this test. "
                    "Include origin (seed/generated/mixed) and the identity, credentials, "
                    "location, account state or other facts the instruction/setup depends on. "
                    "Never put hidden pass/fail checks here.",
                },
                "setup_code": {
                    "type": "string",
                    "description": "Python defining setup(world): the changes this scenario "
                    "makes to the environment before the run. Leave empty to run on the base "
                    "world unchanged. Use world.call(tool, args) to act through the agent's own "
                    "tools, or world.put, world.change and world.drop for what no tool can produce. This is code and not a list of "
                    "rows because a scenario may need more than a table changed.",
                },
                "ready_code": {
                    "type": "string",
                    "description": "Python defining ready(world): return None when the world "
                    "holds what this scenario presumes, or a sentence naming what is missing. "
                    "This is the precondition. If the scenario is about the last five items, "
                    "check there are five. A scenario whose world was never right tests us, not "
                    "the agent.",
                },
                "solution": {
                    "type": "array",
                    "description": "What a correct agent would do: the reference trajectory. "
                    "Never run against the agent under test; it exists to prove the scenario "
                    "can be passed at all.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "description": "Exactly the model-facing arguments defined by "
                                "the agent's tool schema. Never include hidden session state.",
                            },
                            "environment_arguments": {
                                "type": "object",
                                "description": "Only for a source-provisioned tool whose raw "
                                "dependency needs fields the worker injects: the complete raw "
                                "dependency payload used to prove the real state effect. This "
                                "is never shown to or credited to the agent. Omit for local "
                                "tools and when the two payloads are identical. A value like "
                                "`$call.book_ride.booking_ref` resolves that field from the "
                                "most recent successful earlier reference call.",
                            },
                        },
                        "required": ["tool", "arguments"],
                    },
                },
                "sub_goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names from the shared catalogue that must hold. Use the "
                    "existing names wherever one fits, so results add up across the suite.",
                },
                "max_turns": {"type": "integer"},
            },
            scenario_required,
        ),
    )
    async def submit_scenario(args: dict[str, Any]) -> dict[str, Any]:
        # A writer working one slice of a suite stops at the size it was given. Its turn budget is
        # far larger than its slice, and left to itself it keeps writing: one run proved 559
        # scenarios against a target of 200, spending three times the quota and three times the wall
        # clock, and the surplus is trimmed at the end anyway. Replacing a scenario it already has
        # stays allowed, because fixing a refused one is how a writer finishes its slice.
        if not can_save and wanted:
            named = str(args.get("name") or "").strip()
            already = any(one.name == named for one in kept)
            if not already and len(kept) >= wanted:
                return _err(
                    f"This slice is complete: {len(kept)} of {wanted} written. Do not write another. "
                    "Say what you covered and what you could not, and stop. Submitting again under "
                    "an existing name is the only submission left to you, for fixing one of yours."
                )
        result = accept_scenario(
            args,
            world_root=world_root,
            catalogue=catalogue,
            kept=kept,
            simulator_prompt=simulator_prompt,
            hard_constraints=contract.hard_constraints,
            persist=can_save,
        )
        if not result.get("is_error"):
            exploration["since_submit"] = 0
        return result

    @tool(
        "amend_contract",
        "Let one of the agent's tools accept values it did not before, when the world holds "
        "something the agent has no way to name. Say why; it is recorded on the contract.",
        schema(
            {"tool_name": str, "argument": str, "values": list, "why": str},
            ["tool_name", "argument", "values", "why"],
        ),
    )
    async def amend_contract(args: dict[str, Any]) -> dict[str, Any]:
        done, said = widen(
            contract,
            world_root,
            tool_name=str(args.get("tool_name") or ""),
            argument=str(args.get("argument") or ""),
            values=[str(v) for v in (args.get("values") or [])],
            why=str(args.get("why") or ""),
        )
        return _ok(said) if done else _err(said)

    @tool(
        "add_rule",
        "Give the agent a hard rule its source did not state, when asked for one. It is told to "
        "the agent under test and graded, so this changes what is being tested. Say why.",
        schema({"rule": str, "why": str}, ["rule", "why"]),
    )
    async def add_rule_tool(args: dict[str, Any]) -> dict[str, Any]:
        done, said = add_rule(
            contract,
            world_root,
            rule=str(args.get("rule") or ""),
            why=str(args.get("why") or ""),
        )
        return _ok(said) if done else _err(said)

    @tool(
        "drop_rule",
        "Take away a hard rule the agent does not really have. Say why.",
        schema({"rule": str, "why": str}, ["rule", "why"]),
    )
    async def drop_rule_tool(args: dict[str, Any]) -> dict[str, Any]:
        done, said = drop_rule(
            contract,
            world_root,
            rule=str(args.get("rule") or ""),
            why=str(args.get("why") or ""),
        )
        return _ok(said) if done else _err(said)

    @tool(
        "fix_tool",
        "Correct a tool that was read wrong, or remove one the agent does not have. Everything "
        "is built from these, so a wrong argument name produces a world that refuses everything.",
        schema(
            {
                "tool_name": str,
                "args": list,
                "arg_types": dict,
                "description": str,
                "remove": bool,
                "why": str,
            },
            ["tool_name", "why"],
        ),
    )
    async def fix_tool_tool(args: dict[str, Any]) -> dict[str, Any]:
        done, said = fix_tool(
            contract,
            world_root,
            tool_name=str(args.get("tool_name") or ""),
            why=str(args.get("why") or ""),
            args=[str(a) for a in args["args"]] if args.get("args") else None,
            arg_types={
                str(k): str(v) for k, v in (args.get("arg_types") or {}).items()
            },
            description=str(args.get("description") or ""),
            remove=bool(args.get("remove")),
        )
        return _ok(said) if done else _err(said)

    @tool(
        "aim_for",
        "Set how many scenarios are wanted. Call it whenever the person changes what they are "
        "asking for: a number outright, or asking for more without naming one, in which case the "
        "count is the size of the suite once you have written them. Adding to an existing suite "
        "always needs this, because reopening one starts with the target set to what is already "
        "there.\n\n"
        "What it is not for is saving a suite nobody asked for. Writing extra and then raising "
        "the target to match is how a request for four becomes thirteen that nobody reviews.",
        schema({"count": int}, ["count"]),
    )
    async def aim_for(args: dict[str, Any]) -> dict[str, Any]:
        count = int(args.get("count") or 0)
        if count < 1:
            return _err("that is not a number of scenarios worth writing")
        target["count"] = count
        return _ok(f"aiming for {count}. {len(kept)} written so far")

    @tool(
        "drop_scenario",
        "Remove a scenario by name, or all of them with name '*'.",
        schema({"name": str}, ["name"]),
    )
    async def drop_scenario(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or "")
        if name == "*":
            kept.clear()
            write_scenarios(kept, destination, catalogue)
            return _ok("all scenarios dropped")
        before = len(kept)
        kept[:] = [one for one in kept if one.name != name]
        if len(kept) == before:
            return _err(f"no scenario called {name!r}")
        write_scenarios(kept, destination, catalogue)
        return _ok(f"{name} dropped. {len(kept)} left")

    @tool(
        "generate_suite",
        "Write a whole suite at once by splitting it across the agent's use cases, one writer "
        "per slice, several running at the same time, then reviewing what came back and "
        "filling what it missed. Use this whenever somebody asks for a number of scenarios "
        "rather than one in particular: writing twenty or fifty one at a time runs out of "
        "turns long before it finishes.\n\n"
        "Pass `slices` when you know how the suite should be divided, which you do once you "
        "have looked at the world: give each use case a share in proportion to how much can "
        "genuinely go wrong in it, and name the angle each slice should take. Without it the "
        "work is divided evenly, which pads the thin use cases and under-covers the rich ones. "
        "Everything produced clears the same three gates, and the suite is saved.",
        schema(
            {
                "count": int,
                "at_once": int,
                "slices": {
                    "type": ["array", "null"],
                    "description": "How to divide the suite. One entry per writer.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "use_case": {
                                "type": "string",
                                "description": "One of the agent's use cases, worded as the "
                                "contract words it.",
                            },
                            "angle": {
                                "type": "string",
                                "description": "What this slice should look for: the ordinary "
                                "path, the branch that cannot be completed, the rule under "
                                "pressure, state that has to carry.",
                            },
                            "count": {
                                "type": "integer",
                                "description": "How many scenarios this slice is worth, in "
                                "proportion to how much can genuinely go wrong in it.",
                            },
                            "why": {
                                "type": "string",
                                "description": "Why it earns that share.",
                            },
                        },
                        "required": ["use_case", "count"],
                    },
                },
            },
            ["count"],
        ),
    )
    async def generate_suite(args: dict[str, Any]) -> dict[str, Any]:
        from .scenarios import MOST_AT_ONCE, MOST_IN_ONE_GO, write_in_parallel

        asked = int(args.get("count") or 0)
        if asked < 1:
            return _err("say how many scenarios the suite should have")
        cases = [one for one in contract.real_use_cases if one.strip()]
        given = args.get("slices") or None
        if not cases and not given:
            return _err(
                "this contract names no use cases, so there is nothing to split the work "
                "across. Write them one at a time with submit_scenario, or fix the contract."
            )

        # Never write more than the suite still needs. A pass that comes back short is told how many
        # are outstanding and to call again, and a model that calls again with the original number
        # instead of the remainder gets a second full suite: measured at 377 kept against a target of
        # 200, three times the quota for scenarios that are trimmed away again. The count on disk is
        # the only honest measure of what is left, so it is read here rather than trusted from the
        # argument.
        if wanted:
            # Counted from the folders, which are what a suite actually is.
            #
            # Counting the journal as well looked better, because a save prunes the directory before
            # rewriting it and a pass asking during that window reads zero and writes a second suite.
            # It was worse: a retried attempt starts with the folders gone and the journal intact, so
            # the count said the suite was complete, this refused to write anything, and the attempt
            # saved 14 of 200 and failed the platform's cardinality check. Overproduction wastes
            # quota; refusing to produce loses the run, so this counts the conservative thing and the
            # prune window stays a known cost.
            asked = min(asked, max(0, wanted - len(load_scenarios(destination))))
        if not asked:
            return _ok(
                f"The suite already holds the {wanted} it was asked for. Nothing more to write: "
                "review what is there, replace any scenario you are unhappy with by name, and "
                "call save_scenarios."
            )
        # A large ask is served a batch at a time. Spinning up a writer per scenario would put
        # hundreds of model sessions on one machine, and the person waiting would see nothing
        # for an hour. A batch they can read, and an offer of the rest, is the better trade.
        count = min(asked, MOST_IN_ONE_GO)
        at_once = max(1, min(int(args.get("at_once") or 0) or 4, MOST_AT_ONCE))

        produced = await write_in_parallel(
            contract,
            out=destination,
            wanted=count,
            use_cases=cases,
            slices=given,
            at_once=at_once,
        )
        # The suite is already on disk. The open session's own list has to be brought level with
        # it, or a later save_scenarios here would write out the stale list and delete every
        # folder the fan-out just produced.
        kept[:] = produced
        target["count"] = len(produced)

        by_case: dict[str, int] = {}
        for one in produced:
            name = one.use_case or "unassigned"
            by_case[name] = by_case.get(name, 0) + 1
        lines = "\n".join(f"  {n} x {case[:70]}" for case, n in sorted(by_case.items()))
        said = (
            f"{len(produced)} scenarios across {len(by_case)} use cases, {at_once} writers at a "
            f"time. Each cleared all three gates and the suite is saved.\n{lines}"
        )
        if asked > count:
            # Never ask here. A hosted run has nobody to answer and no ask tool, so asking ends the
            # stage with fewer scenarios than were requested and no explanation of why.
            said += (
                f"\n\n{asked - count} of the {asked} asked for are still to write. Call "
                f"generate_suite again now for the remaining {asked - count}, with slices for the "
                "use cases still short. Do not stop at this batch and do not ask first."
            )
        return _ok(said)

    @tool(
        "save_scenarios",
        "Write the kept scenarios out. Every one has already been proved by submit_scenario, so "
        "this always saves; anything else worth knowing comes back alongside.",
        schema({}, []),
    )
    async def save_scenarios(_args: dict[str, Any]) -> dict[str, Any]:
        # Always written. Each of these already cleared all three gates on its way in, so this is
        # persistence and not a second opinion: refusing here left proved work in memory only,
        # which is how a suite that asked for fifty and reached twenty-eight saved nothing at all.
        # What is off about the suite is said, not enforced.
        noted = not_ready(kept, target["count"], catalogue)
        path = write_scenarios(kept, destination, catalogue)
        diversity = suite_diversity_problems(kept)
        judged = sum(
            1
            for one in kept
            for name in one.sub_goals
            if (found := catalogue.named(name)) and not found.deterministic()
        )
        said = (
            f"Saved {len(kept)} scenarios. Each has its own folder under "
            f"{destination / 'scenarios'} holding scenario.json, setup.py, ready.py and one "
            f"runnable file per check; {path.name} indexes them.\n"
            "Every one cleared all three gates: the world is ready for it, the reference "
            "solution passes its checks, and those checks fail when nothing is done.\n"
            f"{judged} sub-goal references are judged rather than settled by code."
        )
        if noted:
            said += (
                "\n\nWorth looking at, none of it stopping the save:\n  - "
                + "\n  - ".join(noted)
            )
        if diversity:
            return _err(
                said
                + "\n\nSaved as a checkpoint, but the suite is not ready to run because its "
                "fixtures/personas are repetitive:\n  - "
                + "\n  - ".join(diversity)
                + "\nReplace the repeated scenarios, then save again."
            )
        return _ok(said)

    server = tool_server(
        name=SCENARIO_SERVER,
        version="0.1.0",
        tools=[
            inspect_world,
            inspect_scenario,
            try_calls,
            add_sub_goal,
            submit_scenario,
            amend_contract,
            add_rule_tool,
            drop_rule_tool,
            fix_tool_tool,
            aim_for,
            drop_scenario,
        ]
        # Only the session a person is talking to may fan out. A writer that is itself one slice
        # of a fan-out calling this would split its own slice again, and so on.
        + (
            [generate_suite, save_scenarios]
            if can_save and worth_delegating(wanted)
            else [save_scenarios]
            if can_save
            else []
        ),
    )
    return server, kept


_ALWAYS = (
    "inspect_world",
    "inspect_scenario",
    "try_calls",
    "add_sub_goal",
    "submit_scenario",
    "amend_contract",
    "add_rule",
    "drop_rule",
    "fix_tool",
    "aim_for",
    "drop_scenario",
    "save_scenarios",
)


def tool_names(wanted: int = 0) -> tuple[str, ...]:
    """The tools a saving session publishes, which depends on how large a suite it was asked for."""
    if worth_delegating(wanted):
        return (*_ALWAYS[:-1], "generate_suite", "save_scenarios")
    return _ALWAYS


# The whole surface, for anything that needs to name every tool this module can publish rather than
# the subset one request gets. Which of them a session actually receives is decided per request.
TOOL_NAMES = tool_names(FEWEST_WORTH_DELEGATING)


def world_summary(world_root: Path) -> str:
    """What is in the built environment, for grounding the writer before it asks."""
    world = restore(world_root)
    try:
        state = world.state()
        lines = [f"  {name}: {len(rows)} rows" for name, rows in sorted(state.items())]
        catalogue = load_catalogue(world_root)
        if catalogue.sub_goals:
            lines.append(
                "\nSUB-GOALS already defined (reuse these, do not restate them):"
            )
            lines += [f"  {one.name}: {one.what}" for one in catalogue.sub_goals]
        return "THE BUILT WORLD (restored fresh for every scenario):\n" + "\n".join(
            lines
        )
    finally:
        world.close()
