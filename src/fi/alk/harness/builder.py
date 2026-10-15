"""One agent for a built environment, with every tool and every skill, and no stages.

The staged conversation opens one stage at a time and hands a request between them as a string, so
a question about runs met a session that could not see scenarios and an edit met one that never
saw the proposal. Here one session holds the whole environment: it reads the contract, the world,
the agent's own source, every scenario and every run, picks the method it needs as a skill, and
changes the environment only through the tools that prove each change.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .backends import (
    ASK_TOOL,
    DELEGATE_TOOL,
    FILE_TOOLS,
    SessionSpec,
    ToolServer,
    tool,
    tool_server,
)
from .config import SKILLS_ROOT, chosen_model, discovered_skills, load_skill
from .contract import AgentContract
from .scenario import voicemail_enabled
from .scenario_tools import SCENARIO_SERVER, load_scenarios, scenario_tools, world_summary
from .scenarios import writer_worker
from .session import Stage
from .tools import schema

BUILDER = "builder"
BIND_AGENT = "bind-agent"
KNOWLEDGE_SERVER = "knowledge"
BUILDER_TURNS = 200
# The builder's own skill is its system prompt; every other skill is loaded when needed.
_NOT_LOADABLE = {BUILDER}


def _description(path: Path) -> str:
    head = path.read_text(encoding="utf-8").split("---")
    if len(head) > 2:
        for line in head[1].splitlines():
            if line.strip().lower().startswith("description:"):
                return line.split(":", 1)[1].strip()
    return ""


def skill_index() -> dict[str, str]:
    """Every loadable skill and knowledge topic by name, with its own description."""
    found: dict[str, str] = {}
    for path in sorted(SKILLS_ROOT.glob("*/SKILL.md")):
        if path.parent.name not in _NOT_LOADABLE:
            found[path.parent.name] = _description(path)
        for topic in sorted((path.parent / "topics").glob("*.md")):
            found[f"{path.parent.name}/{topic.stem}"] = _description(topic)
    return found


def _skill_text(name: str) -> str:
    if "/" in name:
        folder, topic = name.split("/", 1)
        return (SKILLS_ROOT / folder / "topics" / f"{topic}.md").read_text(encoding="utf-8")
    return load_skill(name, preamble=False)


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def knowledge_server(contract: AgentContract | None, destination: Path) -> ToolServer:
    """Skills, the platform's offered values, and how this environment was built."""
    index = skill_index()

    @tool(
        "read_skill",
        "Load one skill or knowledge topic by name. Load the method a task needs before doing it, "
        "and the topic a question is about before answering it. Names are listed in your "
        "instructions under 'What you know'.",
        schema({"name": str}, ["name"]),
    )
    async def read_skill(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or "").strip()
        if name not in index:
            return _err(f"no skill called {name!r}. Available: {', '.join(index)}")
        text = _skill_text(name)
        if name in {"write-scenarios", "plan-suite"} and contract is not None:
            text += discovered_skills(
                modality=contract.modality,
                voicemail="on" if voicemail_enabled() else "off",
                conversational="yes" if contract.conversational else "no",
            )
        return _ok(text)

    @tool(
        "offered_values",
        "The values the platform actually offers for a caller: accents, languages, genders, age "
        "groups, personalities, communication styles, locations, occupations, and the background "
        "noise places. A value not listed here does not exist on the platform; say so rather than "
        "promising it.",
        schema({}, []),
    )
    async def offered_values(_args: dict[str, Any]) -> dict[str, Any]:
        from .background_noise import places
        from .persona_guides import vocabulary

        return _ok(
            json.dumps(
                {"persona": vocabulary(), "background_noise_places": sorted(places())},
                indent=1,
                sort_keys=True,
            )
        )

    @tool(
        "read_build_log",
        "How this environment was built: the events its build recorded, oldest first, one page at "
        "a time. Use it to answer how something was decided or why a step failed or was repaired.",
        schema({"page": int, "contains": str}, []),
    )
    async def read_build_log(args: dict[str, Any]) -> dict[str, Any]:
        path = destination / "harness-events.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return _ok("This environment kept no build log.")
        wanted = str(args.get("contains") or "").strip().lower()
        events = []
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            said = json.dumps(event.get("payload") or {}, sort_keys=True)[:400]
            kind = event.get("type") or event.get("event_type")
            entry = f"{kind}: {said}"
            if not wanted or wanted in entry.lower():
                events.append(entry)
        size = 60
        page = max(1, int(args.get("page") or 1))
        shown = events[(page - 1) * size : page * size]
        pages = -(-len(events) // size)
        return _ok(f"{len(events)} events, page {page} of {max(pages, 1)}\n" + "\n".join(shown))

    return tool_server(
        name=KNOWLEDGE_SERVER, tools=[read_skill, offered_values, read_build_log]
    )


def suite_server(
    out: Path,
    kept: list,
    contract: AgentContract | None = None,
    source_root: Path | None = None,
    sealed: bool = True,
) -> ToolServer:
    """Whole-suite operations the single-scenario tools would take one call per scenario to do."""
    from .catalogue import load_catalogue
    from .prove import prove
    from .scenario_tools import write_scenarios

    @tool(
        "drop_scenarios",
        "Remove several scenarios at once, by exact name. Resolve numbers or a description to names "
        "with find_scenarios first. Says which names were removed and which were not found.",
        schema({"names": {"type": "array", "items": {"type": "string"}}}, ["names"]),
    )
    async def drop_scenarios(args: dict[str, Any]) -> dict[str, Any]:
        wanted = {str(one).strip() for one in args.get("names") or [] if str(one).strip()}
        present = {one.name for one in kept}
        gone = sorted(wanted & present)
        missing = sorted(wanted - present)
        if not gone:
            return _err(f"none of these is in the suite: {', '.join(missing)}")
        if len(gone) == len(kept):
            return _err("that would remove every scenario; an environment keeps at least one")
        kept[:] = [one for one in kept if one.name not in wanted]
        write_scenarios(kept, out, load_catalogue(out))
        return _ok(
            f"dropped {len(gone)}: {', '.join(gone)}. {len(kept)} left"
            + (f". Not found: {', '.join(missing)}" if missing else "")
        )

    @tool(
        "prove_scenarios",
        "Run the three gates (ready, solvable, not vacuous) again on existing scenarios by name, "
        "without changing them, and report each one's result.",
        schema({"names": {"type": "array", "items": {"type": "string"}}}, ["names"]),
    )
    async def prove_scenarios(args: dict[str, Any]) -> dict[str, Any]:
        wanted = [str(one).strip() for one in args.get("names") or [] if str(one).strip()]
        by_name = {one.name: one for one in kept}
        catalogue = load_catalogue(out)
        lines = []
        for name in wanted:
            scenario = by_name.get(name)
            if scenario is None:
                lines.append(f"{name}: not in the suite")
                continue
            proof = prove(scenario, catalogue, out)
            gates = ", ".join(f"{gate} {'yes' if held else 'no'}" for gate, held in proof.gates().items())
            lines.append(
                f"{name}: {'holds' if proof.holds else 'does not hold'} ({gates})"
                + ("" if proof.holds else f". {proof.why()}")
                + (" Judged-only: nothing observable to check." if proof.judged_only else "")
            )
        return _ok("\n".join(lines) or "name at least one scenario")

    @tool(
        "check_health",
        "Check the whole environment for anything broken: scenarios naming sub-goals the "
        "catalogue lacks, solutions calling tools the contract lacks, the index or the sealed "
        "bundle out of step with the scenario folders, and (with `prove`) any scenario whose "
        "proof no longer holds. Run it when the conversation opens and after every change, and "
        "repair what it finds.",
        schema({"prove": bool}, []),
    )
    async def check_health(args: dict[str, Any]) -> dict[str, Any]:
        problems = suite_health(contract_of(out), out, kept, prove_all=bool(args.get("prove")), sealed=sealed)
        return _ok("Healthy: nothing broken found." if not problems else "\n".join(problems))

    @tool(
        "impact_of",
        "Before changing the world, a tool, the contract or a sub-goal, find everything that "
        "depends on it: the scenarios whose setup, ready check, solution, checks or sub-goals "
        "mention it, and the sub-goals whose checks do. `name` is a table, column, tool, "
        "argument, value or sub-goal name.",
        schema({"name": str}, ["name"]),
    )
    async def impact_of(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or "").strip()
        if not name:
            return _err("name what is about to change")
        return _ok(dependents(out, kept, name))

    @tool(
        "change_world",
        "Change the world's data with one INSERT, UPDATE or DELETE statement, the way the world "
        "is seeded for every scenario and every run. Run impact_of on the tables and values first "
        "and confirm the blast radius with the person; afterwards re-prove the affected scenarios "
        "and save_changes, which carries the new data into the sealed seed. Returns the rows "
        "changed and each table's count.",
        schema({"sql": str}, ["sql"]),
    )
    async def change_world(args: dict[str, Any]) -> dict[str, Any]:
        return _ok(world_change(out, str(args.get("sql") or ""))) if (out / "world.sqlite").is_file() else _err(
            "This environment keeps no world data the chat can change: its agent's data lives "
            "where the agent runs, or its world is described in a form only a new build changes."
        )

    @tool(
        "declare_tool",
        "Add a tool the agent really has to the contract, when the contract lacks it (the agent "
        "gained a tool, or the contract lost one). `evidence` names a file in the agent's source "
        "and quotes the line that defines the tool; it is checked before anything changes. The "
        "addition is recorded on the contract as an amendment with `why`.",
        schema(
            {
                "name": str,
                "description": str,
                "args": {"type": "array", "items": {"type": "string"}},
                "arg_types": {"type": "object"},
                "arg_values": {"type": "object"},
                "evidence_file": str,
                "evidence_quote": str,
                "why": str,
            },
            ["name", "description", "evidence_file", "evidence_quote", "why"],
        ),
    )
    async def declare_tool(args: dict[str, Any]) -> dict[str, Any]:
        from .amend import _persist
        from .contract import ToolSpec, validate_contract

        if contract is None:
            return _err("there is no contract loaded to add a tool to")
        name = str(args.get("name") or "").strip()
        if name in contract.tool_names():
            return _err(f"{name} is already in the contract; use fix_tool to correct it")
        quote = str(args.get("evidence_quote") or "").strip()
        root = source_root or Path("/nonexistent")
        target = (root / str(args.get("evidence_file") or "")).resolve()
        if source_root is None or root.resolve() not in target.parents or not target.is_file():
            return _err("evidence_file must be a file inside the agent's source")
        if not quote or quote not in target.read_text(encoding="utf-8", errors="replace"):
            return _err(f"the quoted line is not in {args.get('evidence_file')}; quote it exactly")
        if not str(args.get("why") or "").strip():
            return _err("say why: an unexplained change to the contract is indistinguishable from a guess")
        spec = ToolSpec.model_validate(
            {
                "name": name,
                "description": str(args.get("description") or ""),
                "args": list(args.get("args") or []),
                "arg_types": dict(args.get("arg_types") or {}),
                "arg_values": dict(args.get("arg_values") or {}),
            }
        )
        contract.tools.append(spec)
        contract.amendments.append(f"{name} added from {args['evidence_file']}: {args['why']}")
        problems = validate_contract(contract)
        if problems:
            contract.tools.pop()
            contract.amendments.pop()
            return _err("the contract would not be valid: " + "; ".join(problems))
        _persist(contract, out)
        return _ok(f"{name} added to the contract. Re-prove the scenarios that use it, then save_changes.")

    return tool_server(
        name="suite",
        tools=[drop_scenarios, prove_scenarios, check_health, impact_of, change_world, declare_tool],
    )


def world_change(out: Path, statement: str) -> str:
    """Apply one data statement to the authoring world, all or nothing, foreign keys enforced."""
    import sqlite3

    verb = statement.strip().split(None, 1)[0].upper() if statement.strip() else ""
    if verb not in ("INSERT", "UPDATE", "DELETE") or ";" in statement.strip().rstrip(";"):
        return "Not changed: give exactly one INSERT, UPDATE or DELETE statement."
    connection = sqlite3.connect(out / "world.sqlite")
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            changed = connection.execute(statement).rowcount
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        counts = ", ".join(
            "{}: {}".format(name, connection.execute('SELECT COUNT(*) FROM "%s"' % name).fetchone()[0])
            for name in tables
        )
    except sqlite3.Error as failed:
        return f"Not changed, the world refused it: {failed}"
    finally:
        connection.close()
    return f"{changed} rows changed. The world now holds {counts}. Re-prove what depends on it, then save_changes."


def contract_of(out: Path) -> AgentContract | None:
    from .understand import load

    return load(out)


def dependents(out: Path, kept: list, name: str) -> str:
    """Which scenarios and sub-goals mention one name, and where."""
    from .catalogue import load_catalogue

    needle = name.lower()
    hits: list[str] = []
    for scenario in kept:
        where = []
        if needle in scenario.setup_code.lower():
            where.append("setup")
        if needle in scenario.ready_code.lower():
            where.append("ready check")
        if any(needle in (step.tool + json.dumps(step.arguments, default=str)).lower() for step in scenario.solution):
            where.append("solution")
        if any(needle == goal.lower() for goal in scenario.sub_goals):
            where.append("names the sub-goal")
        if needle in (scenario.instruction or "").lower():
            where.append("situation")
        if where:
            hits.append(f"{scenario.name}: {', '.join(where)}")
    goals = [
        one.name
        for one in load_catalogue(out).sub_goals
        if needle in (one.check or "").lower() or needle in (one.judged or "").lower()
    ]
    using = {
        scenario.name for scenario in kept if set(scenario.sub_goals) & set(goals)
    }
    return (
        f"{len(hits)} scenarios mention {name!r} directly"
        + (":\n  " + "\n  ".join(hits) if hits else "")
        + f"\n{len(goals)} sub-goals mention it: {', '.join(goals) or 'none'}"
        + (
            f"\n{len(using)} scenarios name those sub-goals, so their grading depends on it too: "
            + ", ".join(sorted(using))
            if using
            else ""
        )
    )


def suite_health(
    contract: AgentContract | None,
    out: Path,
    kept: list,
    *,
    prove_all: bool = False,
    sealed: bool = True,
) -> list[str]:
    """Everything broken in the suite, one line each, or nothing."""
    from .catalogue import load_catalogue

    problems: list[str] = []
    catalogue = load_catalogue(out)
    defined = {one.name for one in catalogue.sub_goals}
    tools = contract.tool_names() if contract is not None else set()
    for scenario in kept:
        missing = [goal for goal in scenario.sub_goals if goal not in defined]
        if missing:
            problems.append(f"{scenario.name}: names sub-goals the catalogue lacks: {', '.join(missing)}")
        unknown = sorted({step.tool for step in scenario.solution if tools and step.tool not in tools})
        if unknown:
            problems.append(f"{scenario.name}: its solution calls tools the contract lacks: {', '.join(unknown)}")
    folders = {path.name for path in (out / "scenarios").iterdir() if path.is_dir()} if (out / "scenarios").is_dir() else set()
    names = {scenario.name for scenario in kept}
    if folders != names:
        problems.append(
            "scenario folders and the loaded suite differ: "
            + ", ".join(sorted(folders ^ names)[:20])
        )
    try:
        index = json.loads((out / "scenarios.json").read_text(encoding="utf-8"))
        indexed = {str(one.get("name")) for one in index if isinstance(one, dict)}
        if indexed != names:
            problems.append(
                "the index (scenarios.json) does not list the same scenarios: "
                + ", ".join(sorted(indexed ^ names)[:20])
            )
    except (OSError, ValueError):
        problems.append("the index (scenarios.json) is missing or unreadable")
    bundle = out / "environment-bundle"
    if sealed and (bundle / "manifest.json").is_file():
        sealed = {path.name for path in (bundle / "scenarios").iterdir() if path.is_dir()} if (bundle / "scenarios").is_dir() else set()
        if sealed != names:
            problems.append(
                f"the sealed bundle runs play is out of step with the suite ({len(sealed - names)} "
                f"sealed scenarios no longer in it, {len(names - sealed)} in it but not sealed): "
                "runs would not play what the suite says; save_changes with force reseals it"
            )
        else:
            from .bundle_v2 import load_bundle_v2
            from .process_preflight import PreflightError, _verify_digest

            try:
                _verify_digest(bundle, load_bundle_v2(bundle))
            except (PreflightError, Exception) as failed:  # noqa: BLE001 - reported, not raised
                problems.append(f"the sealed bundle does not verify: {failed}; save_changes with force reseals it")
    if prove_all:
        from .prove import prove

        for scenario in kept:
            proof = prove(scenario, catalogue, out)
            if not proof.holds:
                problems.append(f"{scenario.name}: proof no longer holds. {proof.why()}"[:600])
    return problems



def environment_brief(contract: AgentContract, out: Path, overview: dict[str, Any] | None) -> str:
    """What the builder knows about this environment before the first question, all derived."""
    from .catalogue import load_catalogue

    lines = ["## This environment, in brief", ""]
    if overview:
        env = overview.get("environment") or {}
        lines.append(
            f"- Environment `{env.get('id')}` ({env.get('name') or 'unnamed'}), state "
            f"{env.get('state')}; agent connected by {env.get('connector') or 'unknown'}"
            + (f", mode {env.get('mode')}" if env.get("mode") else "")
            + (f", source {env.get('source_kind')}" if env.get("source_kind") else "")
        )
    lines.append(
        f"- Modality {contract.modality}; {len(contract.tools)} tools: "
        + (", ".join(tool.name for tool in contract.tools) or "none (judged from the conversation)")
    )
    if contract.amendments:
        lines.append("- Contract amendments made after the agent was read:")
        lines.extend(f"  - {one}" for one in contract.amendments[-10:])
    catalogue = load_catalogue(out)
    checked = [one.name for one in catalogue.sub_goals if one.check]
    judged = [one.name for one in catalogue.sub_goals if not one.check]
    lines.append(
        f"- Sub-goal catalogue: {len(catalogue.sub_goals)} ({len(checked)} checked in code, "
        f"{len(judged)} judged). Checked: {', '.join(checked[:30]) or 'none'}"
        + (f"; judged: {', '.join(judged[:30])}" if judged else "")
    )
    events = out / "harness-events.jsonl"
    if events.is_file():
        lines.append(
            f"- The build recorded {sum(1 for _ in events.open(encoding='utf-8'))} events; "
            "`read_build_log` has them"
        )
    if overview:
        lines.append(f"- {overview.get('scenario_count')} scenarios. By use case (tab numbers):")
        for one in overview.get("use_cases") or []:
            shown = one.get("names") or one.get("examples") or []
            lines.append(
                f"  - {one['use_case']} ({one['count']}): {one['numbers']}"
                + (f". {'Names' if one.get('names') else 'For example'}: {'; '.join(shown)}" if shown else "")
            )
        coverage = overview.get("coverage") or {}
        lines.append(
            f"- Coverage: {coverage.get('axes_varied')} of 8 axes varied; forced situations present: "
            f"{', '.join(coverage.get('forced_present') or []) or 'none'}; missing: "
            f"{', '.join(coverage.get('forced_missing') or []) or 'none'}"
        )
        for axis in coverage.get("axes") or []:
            counts = ", ".join(f"{level} {n}" for level, n in list(axis["counts"].items())[:8])
            lines.append(f"  - {axis['axis']} ({axis['levels']} levels): {counts}")
        runs = overview.get("recent_runs") or []
        lines.append(
            "- Recent runs: "
            + (
                "; ".join(
                    f"{run['run_id'][:8]} {run['state']} ({run['completed']} done, "
                    f"{run['failed']} failed of {run['calls']} calls)"
                    + (f", {run['failure'].get('code')}" if run.get("failure") else "")
                    for run in runs
                )
                or "none yet"
            )
        )
    else:
        lines.append(f"- {len(load_scenarios(out))} scenarios on disk (tab numbers unavailable)")
    changes = (overview or {}).get("build_changes") or []
    if changes:
        lines.append("- Changes the person asked for while the environment was being built:")
        lines.extend(
            f"  - [{one.get('target_stage')}, {one.get('status')}] {one.get('instruction')}" for one in changes
        )
        lines.append(
            "  Check that each one is reflected in what exists now; finish any that is not, on your own."
        )
    earlier = (overview or {}).get("earlier_messages") or []
    if earlier:
        lines.append("- The conversation so far (most recent last):")
        lines.extend(f"  - {one.get('role')}: {one.get('text')}" for one in earlier)
    lines.append(
        "\nThis is a summary taken when the conversation opened. For any detail, read the real "
        "value with a tool and quote it."
    )
    return "\n".join(lines)


def open_for_building(
    contract: AgentContract | None,
    *,
    out: Path,
    source: Any,
    job: Any = None,
    platform: ToolServer | None = None,
    ask: Callable[..., Any] | None = None,
    hosted_prompt: str = "",
) -> Stage:
    """The same harness before its environment has a contract or a world: it has the tools for
    what can be done next, and gains the rest when it is opened again after saving."""
    if contract is None:
        from .understand import open_stage as understand_stage

        offered = ((getattr(job, "metadata", None) or {}) if job is not None else {}).get("available_evals")
        work, _ = understand_stage(source, out=out, available_evals=offered if isinstance(offered, list) else None)
    else:
        from .build import open_stage as build_stage

        work, _ = build_stage(
            contract,
            out=out,
            source_root=str(source.workdir() if source is not None else ""),
            deferred_runtime=True,
            external_runtime=getattr(source, "kind", "") == "provider",
        )
    servers: dict[str, ToolServer] = {
        **work.spec.servers,
        KNOWLEDGE_SERVER: knowledge_server(contract, out),
    }
    if platform is not None:
        servers[platform.name] = platform
    spec = SessionSpec(
        system_prompt=(
            load_skill(BUILDER)
            + "\n\n## What you know\n\nLoad any of these with `read_skill` before you answer a "
            "question it covers or do work it governs.\n\n"
            + "\n".join(f"- `{name}`: {about}" for name, about in skill_index().items())
            + "\n\n## The work in front of you now\n\n"
            + work.spec.system_prompt
            + f"\n\nThe environment is being built in {out}."
            + hosted_prompt
        ),
        servers=servers,
        builtins=tuple(dict.fromkeys((*work.spec.builtins, *FILE_TOOLS, ASK_TOOL))),
        workers=dict(work.spec.workers),
        cwd=str(out),
        max_turns=BUILDER_TURNS,
        model=chosen_model(),
        ask=ask,
    )
    return Stage(spec, name=BUILDER)


def open_for_adapting(
    workbench: Any,
    *,
    approve: Callable[[str, list[str]], Awaitable[str]],
    ask: Callable[..., Any] | None = None,
    hosted_prompt: str = "",
) -> Stage:
    """The same harness on a prebuilt environment it does not own, binding the person's agent to
    it. The environment and its suite are sealed: the binding tools are the only way to change
    anything, and they, not the session, decide whether the binding is proved and sealed."""
    from .adaptation.workbench import binding_server

    adaptation = binding_server(workbench, approve)
    spec = SessionSpec(
        system_prompt=(
            load_skill(BUILDER)
            + "\n\n## What you know\n\nLoad any of these with `read_skill` before you answer a "
            "question it covers or do work it governs.\n\n"
            + "\n".join(f"- `{name}`: {about}" for name, about in skill_index().items())
            + "\n\n## The work in front of you now\n\n"
            + load_skill(BIND_AGENT, preamble=False)
            + "\n\n## This adaptation\n\n"
            + workbench.brief()
            + hosted_prompt
        ),
        servers={
            adaptation.name: adaptation,
            KNOWLEDGE_SERVER: knowledge_server(None, workbench.output),
        },
        builtins=(ASK_TOOL,) if ask is not None else (),
        cwd=str(workbench.output),
        max_turns=BUILDER_TURNS,
        model=chosen_model(),
        ask=ask,
    )
    return Stage(spec, name=BUILDER)


def open_for_composition(
    composition: Any,
    *,
    platform: ToolServer | None = None,
    ask: Callable[..., Any] | None = None,
    hosted_prompt: str = "",
) -> Stage:
    """The same harness on a composition: a prebuilt environment it does not own, bound to the
    person's agent. Only the suite changes, so the tools that change the world or the contract
    are left out; saving seals a child suite and re-proves the binding (`composition_edit`)."""
    from .composition_edit import FIXED_ENVIRONMENT_TOOLS, trajectory_server

    return open_builder(
        composition.contract,
        out=composition.workspace,
        source_root=None,
        platform=platform,
        ask=ask,
        hosted_prompt=hosted_prompt,
        brief=composition.brief(),
        sealed=False,
        more_servers=(
            {"composition": trajectory_server(composition)} if composition.routed else None
        ),
        files_note=composition.files_note(),
        without=FIXED_ENVIRONMENT_TOOLS,
    )


def open_builder(
    contract: AgentContract,
    *,
    out: Path,
    source_root: Path | None,
    platform: ToolServer | None = None,
    ask: Callable[..., Any] | None = None,
    scenario_problem: Callable[[dict[str, Any]], str] | None = None,
    hosted_prompt: str = "",
    brief: str = "",
    sealed: bool = True,
    more_servers: dict[str, ToolServer] | None = None,
    files_note: str = "",
    without: frozenset[str] = frozenset(),
) -> Stage:
    """The one session that owns an environment once it has a world: every tool, every skill.
    ``sealed`` is False while the environment is still being built and has no sealed bundle yet.
    ``without`` names tools this session must not have; ``files_note`` replaces the file listing."""
    existing = load_scenarios(out)
    server, kept = scenario_tools(
        contract,
        out,
        out,
        wanted=max(len(existing), 1),
        can_grow=True,
        scenario_problem=scenario_problem,
    )
    servers: dict[str, ToolServer] = {
        SCENARIO_SERVER: server,
        KNOWLEDGE_SERVER: knowledge_server(contract, out),
        "suite": suite_server(out, kept, contract, source_root, sealed=sealed),
    }
    if platform is not None:
        servers[platform.name] = platform
    servers.update(more_servers or {})
    if without:
        servers = {
            name: replace(one, tools=[spec for spec in one.tools if spec.name not in without])
            for name, one in servers.items()
        }
    health = suite_health(contract, out, kept, sealed=sealed) if kept else []
    health_note = (
        "\n\n## Health when this conversation opened\n\n"
        + (
            "Nothing broken was found."
            if not health
            else "These are broken. Repair them first, on your own, before anything else, then "
            "tell the person what you found and fixed:\n- " + "\n- ".join(health[:40])
        )
    )
    where = files_note or (
        f"\n\nThe environment's files are in {out}: contract.json, sub_goals.json, coverage.json, "
        "simulator_prompt.md, scenarios/ and the sealed environment-bundle/."
        + (f" The agent's own source is in {source_root}." if source_root else "")
    )
    spec = SessionSpec(
        system_prompt=(
            load_skill(BUILDER)
            + "\n\n## What you know\n\nLoad any of these with `read_skill` before you answer a "
            "question it covers or do work it governs. Answer internals from them, never from "
            "general knowledge.\n\n"
            + "\n".join(f"- `{name}`: {about}" for name, about in skill_index().items())
            + f"\n\n## This agent\n\n{contract.brief(with_data=True, sample_rows=3)}"
            + f"\n\n## Its world\n\n{world_summary(out)}"
            + where
            + (f"\n\n{brief}" if brief else "")
            + health_note
            + hosted_prompt
        ),
        servers=servers,
        builtins=(*FILE_TOOLS, ASK_TOOL, DELEGATE_TOOL),
        workers=writer_worker(contract, out, server, BUILDER_TURNS),
        cwd=str(out),
        max_turns=BUILDER_TURNS,
        model=chosen_model(),
        ask=ask,
    )
    stage = Stage(spec, name=BUILDER)
    stage.health_at_open = health
    return stage
