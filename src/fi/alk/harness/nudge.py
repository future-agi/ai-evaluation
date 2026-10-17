"""A nudge turn: one builder turn that brings an environment or its suite up to a new agent.

The platform briefs the turn; the turn changes the piece incrementally with a reduced tool set,
then ``report_update`` runs a static gate for its stage and writes the result the platform lands.
The chat sandbox runs no world and holds no agent credentials, so every check here is static
or runs against the frozen world on disk.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .backends import FILE_TOOLS, SessionSpec, ToolServer, tool, tool_server
from .builder import _err, _ok, scenario_problems
from .catalogue import (
    Catalogue,
    catalogue_problems,
    load_catalogue,
    save_catalogue,
    validate_sub_goal,
)
from .contract import validate_contract
from .outbound import _redact_validation_error
from .proofs import load_proofs, record_proof, record_unprovable
from .prove import prove
from .scenario import Scenario, _slug
from .scenario_tools import load_scenarios, write_scenarios
from .source_data_invariants import ARTIFACT, source_files, validate_evidence
from .tools import schema
from .understand import load as load_contract
from .world.snapshot import (
    DATABASE,
    HANDLERS,
    MANIFEST,
    STATE,
    WORLD_MODULE,
    read_manifest,
)
from .world.stores import SAVED
from .world.stores.inprocess import InProcessStore

NUDGE_RESULT = ".futureagi-nudge-result.json"
NUDGE_SERVER = "nudge"
ENVIRONMENT = "environment"
SUITE = "suite"
ENVIRONMENT_TOOLS = frozenset(
    {
        "amend_contract",
        "add_rule",
        "drop_rule",
        "fix_tool",
        "declare_tool",
        "add_sub_goal",
        "retire_sub_goal",
        "impact_of",
        "check_health",
        "create_schema",
        "seed",
        "change_data",
        "adopt_tool",
        "adopt_state",
        "write_simulator_prompt",
        "set_modality",
        "write_store_ops",
        "save_world",
        "read_skill",
        "offered_values",
        "read_build_log",
        "report_update",
        *FILE_TOOLS,
    }
)
SUITE_TOOLS = frozenset(
    {
        "aim_for",
        "submit_scenario",
        "drop_scenario",
        "drop_scenarios",
        "retire_sub_goal",
        "prove_scenarios",
        "find_scenarios",
        "inspect_scenario",
        "check_health",
        "impact_of",
        "read_skill",
        "offered_values",
        "report_update",
        *FILE_TOOLS,
    }
)
# How far past the suite a rebuild's aim_for may set the target at one time.
REBUILD_ADDS_AT_MOST = 5
# What a nudge turn can call without changing the workspace; anything else may change it.
READ_ONLY_TOOLS = frozenset(
    {
        "impact_of",
        "check_health",
        "prove_scenarios",
        "find_scenarios",
        "inspect_scenario",
        "read_skill",
        "offered_values",
        "read_build_log",
        "report_update",
        *FILE_TOOLS,
    }
)
_WORLD_JSON = (MANIFEST, STATE, SAVED, InProcessStore.FILE, "environment.json")
_TEXT_MAX = 500


def _problem(item: str, kind: str, message: str) -> dict[str, str]:
    return {"item": item, "kind": kind, "message": message[:_TEXT_MAX]}


def _failed(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        return f"invalid: {_redact_validation_error(exc)}"[:_TEXT_MAX]
    return f"{type(exc).__name__}: {exc}"[:_TEXT_MAX]


def filtered_spec(spec: SessionSpec, allowed: frozenset[str]) -> SessionSpec:
    """The same session with only the tools a nudge stage may use; no workers, no Delegate."""
    servers = {}
    for name, server in spec.servers.items():
        kept = [one for one in server.tools if one.name in allowed]
        if kept:
            servers[name] = tool_server(
                name=server.name, version=server.version, tools=kept
            )
    builtins = tuple(one for one in spec.builtins if one in allowed)
    return replace(spec, servers=servers, builtins=builtins, workers={})


def retire_sub_goal(
    workspace: Path,
    name: str,
    why: str,
    *,
    stage: str,
    catalogue: Catalogue | None = None,
    kept: list[Scenario] | None = None,
) -> str:
    """Retire a sub-goal: the catalogue entry (environment) or the scenarios' use of it (suite).

    ``catalogue`` and ``kept`` are the session's live objects, changed in place so the session's
    later writes keep the retirement; without them both are read from disk.
    """
    name = (name or "").strip()
    if not name or not (why or "").strip():
        return "Not retired: name the sub-goal and say why."
    catalogue = catalogue if catalogue is not None else load_catalogue(workspace)
    if stage == ENVIRONMENT:
        if catalogue.named(name) is None:
            return f"Not retired: {name} is not in the catalogue."
        catalogue.sub_goals[:] = [
            one for one in catalogue.sub_goals if one.name != name
        ]
        save_catalogue(catalogue, workspace)
        return f"{name} retired from the catalogue: {why}"
    if catalogue.named(name) is not None:
        return (
            f"Not retired: the environment still defines {name}. A suite rebuild only drops "
            "sub-goals the environment has already retired."
        )
    kept = kept if kept is not None else load_scenarios(workspace)
    touched = [one for one in kept if name in one.sub_goals]
    if not touched:
        return f"No scenario names {name}; nothing to retire."
    for one in touched:
        one.sub_goals = [goal for goal in one.sub_goals if goal != name]
    write_scenarios(kept, workspace, catalogue)
    empty = [one.name for one in touched if not one.sub_goals]
    return f"{name} dropped from {len(touched)} scenario(s): {why}" + (
        f". These now grade nothing; repair or drop them: {', '.join(empty)}"
        if empty
        else ""
    )


def _world_problems(workspace: Path) -> tuple[list[dict], list[str]]:
    problems: list[dict] = []
    for name in _WORLD_JSON:
        path = workspace / name
        if path.is_file():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(_problem(name, "world", _failed(exc)))
    for path in sorted(workspace.glob("*.sql")):
        try:
            path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            problems.append(_problem(path.name, "world", _failed(exc)))
    if not (workspace / MANIFEST).is_file() or problems:
        return problems, []
    if str(read_manifest(workspace).get("store") or "sqlite") != "sqlite":
        return problems, ["world"]
    # A scratch copy in memory; the saved file is never written.
    try:
        scratch = sqlite3.connect(":memory:")
        try:
            origin = sqlite3.connect(f"file:{workspace / DATABASE}?mode=ro", uri=True)
            try:
                origin.backup(scratch)
            finally:
                origin.close()
            broken = scratch.execute("PRAGMA integrity_check").fetchone()[0]
            dangling = scratch.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            scratch.close()
    except sqlite3.Error as exc:
        return [*problems, _problem(DATABASE, "world", _failed(exc))], []
    if broken != "ok":
        problems.append(_problem(DATABASE, "world", str(broken)))
    if dangling:
        problems.append(
            _problem(DATABASE, "world", f"{len(dangling)} rows break a foreign key")
        )
    return problems, []


def environment_gate(workspace: Path) -> tuple[list[dict], list[str]]:
    """Static checks of the environment piece: (problems, not_checked)."""
    workspace = Path(workspace)
    problems: list[dict] = []
    try:
        contract = load_contract(workspace)
    except (OSError, ValueError) as exc:
        contract = None
        problems.append(_problem("contract.json", "contract", _failed(exc)))
    else:
        if contract is None:
            problems.append(_problem("contract.json", "contract", "no contract"))
        else:
            problems.extend(
                _problem("contract.json", "contract", one)
                for one in validate_contract(contract)
            )

    world, not_checked = _world_problems(workspace)
    problems.extend(world)

    # Compiled, never imported: the agent's dependencies and credentials are not here.
    handlers = sorted((workspace / HANDLERS).glob("*.py"))
    for path in [workspace / WORLD_MODULE, *handlers]:
        if path.is_file():
            item = path.relative_to(workspace).as_posix()
            try:
                compile(path.read_text(encoding="utf-8"), item, "exec")
            except (SyntaxError, ValueError, OSError) as exc:
                problems.append(_problem(item, "handler", _failed(exc)))

    try:
        catalogue = load_catalogue(workspace)
    except (OSError, ValueError) as exc:
        return [
            *problems,
            _problem("sub_goals.json", "sub_goal", _failed(exc)),
        ], not_checked
    for one in catalogue.sub_goals:
        problems.extend(
            _problem(one.name, "sub_goal", said) for said in validate_sub_goal(one)
        )
        if one.deterministic():
            try:
                compile(one.check, f"sub_goals/{one.name}", "exec")
            except (SyntaxError, ValueError) as exc:
                problems.append(_problem(one.name, "sub_goal", _failed(exc)))
    manifest = (
        read_manifest(workspace)
        if (workspace / MANIFEST).is_file() and not world
        else {}
    )
    problems.extend(
        _problem("sub_goals.json", "sub_goal", said)
        for said in catalogue_problems(
            catalogue.sub_goals,
            world_is_observable=bool(manifest.get("tables"))
            or bool(manifest.get("tools")),
        )
    )
    return problems, not_checked


def suite_gate(
    workspace: Path,
    user_draft_keys: set[str],
    *,
    started_with: set[str] | None = None,
) -> tuple[list[dict], list[str], list[dict]]:
    """Prove every scenario into proofs.json: (problems, not_checked, left_out).

    ``started_with`` is the suite's keys when the turn began; without it the keys proofs.json
    held before this proving stand in for them.
    """
    workspace = Path(workspace)
    earlier = load_proofs(workspace)
    catalogue = load_catalogue(workspace)
    try:
        contract = load_contract(workspace)
    except (OSError, ValueError):
        contract = None
    tools = contract.tool_names() if contract is not None else set()
    scenarios = load_scenarios(workspace)
    present = {one.scenario_key for one in scenarios}
    before = set(earlier) if started_with is None else set(started_with)
    left_out = [
        {
            "key": key,
            "why": str((earlier.get(key) or {}).get("why") or "dropped")[:_TEXT_MAX],
        }
        for key in sorted(before - present)
    ]
    problems: list[dict] = []
    not_checked: list[str] = []
    for scenario in scenarios:
        key = scenario.scenario_key
        # prove() skips names the catalogue lacks, so it would hold on what is left.
        said = scenario_problems(scenario, catalogue.names(), tools)
        if not scenario.sub_goals:
            said.append("no sub_goals: nothing would be graded")
        if said:
            why = "; ".join(said)
            record_unprovable(workspace, key, why)
            if key in user_draft_keys:
                not_checked.append(key)
            else:
                problems.append(_problem(key, "scenario", why))
            continue
        try:
            proof = prove(
                scenario,
                catalogue,
                workspace,
                allow_judged_only_with_state=not scenario.solution,
            )
        except Exception as exc:  # noqa: BLE001 - a scenario that cannot be proved is the finding
            record_unprovable(workspace, key, type(exc).__name__)
            holds, why = False, type(exc).__name__
        else:
            record_proof(workspace, key, proof)
            holds, why = proof.holds, ("" if proof.holds else proof.why())
        if holds:
            continue
        if key in user_draft_keys:
            not_checked.append(key)
        else:
            problems.append(_problem(key, "scenario", why))
    return problems, not_checked, left_out


def source_review_stale(source: Path, workspace: Path) -> list[str]:
    """Warnings for source-review evidence the agent's source no longer contains."""
    try:
        review = json.loads((Path(workspace) / ARTIFACT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    checks = review.get("checks") if isinstance(review, dict) else None
    if not isinstance(checks, list) or not checks:
        return []
    if not Path(source).is_dir():
        return ["source not available; source review not re-checked"]
    files = source_files(Path(source))
    warnings = []
    for check in checks:
        try:
            validate_evidence(check, files)
        except (ValueError, AttributeError, TypeError):
            # Names only: a quote is source text and stays out of the result.
            paths = sorted(
                {str(one.get("path", "")) for one in check.get("evidence") or []}
            )
            warnings.append(
                f"source review is out of date: {check.get('name') or 'a check'} quotes "
                f"{', '.join(paths) or 'a file'}, which no longer holds the quoted line"
            )
    return warnings


def write_result(
    workspace: Path,
    *,
    nudge_id: str,
    stage: str,
    changed: bool,
    notes: str,
    problems: list[dict],
    warnings: list[str],
    not_checked: list[str],
    left_out: list[dict],
) -> None:
    """Write the result the platform lands; the last write wins."""
    body = {
        "nudge_id": nudge_id,
        "stage": stage,
        "changed": bool(changed),
        "notes": notes,
        "checks": {
            "passed": not problems,
            "problems": problems,
            "warnings": warnings,
            "not_checked": not_checked,
        },
        "left_out": left_out,
    }
    target = Path(workspace) / NUDGE_RESULT
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(target)


def report_update(
    workspace: Path,
    *,
    nudge: dict[str, Any],
    changed: bool,
    notes: str,
    source: Path | None = None,
    started_with: set[str] | None = None,
) -> dict[str, Any]:
    """Run the stage gate and write the result file. ``nudge`` is the command payload."""
    # A gate that raises leaves no result rather than an earlier one.
    (Path(workspace) / NUDGE_RESULT).unlink(missing_ok=True)
    stage = str(nudge.get("stage") or "")
    warnings: list[str] = []
    left_out: list[dict] = []
    if stage == ENVIRONMENT:
        problems, not_checked = environment_gate(workspace)
        if source is not None:
            warnings = source_review_stale(source, workspace)
    else:
        problems, not_checked, left_out = suite_gate(
            workspace,
            set(nudge.get("user_draft_keys") or []),
            started_with=started_with,
        )
    write_result(
        workspace,
        nudge_id=str(nudge.get("nudge_id") or ""),
        stage=stage,
        changed=changed,
        notes=notes,
        problems=problems,
        warnings=warnings,
        not_checked=not_checked,
        left_out=left_out,
    )
    return {
        "passed": not problems,
        "problems": problems,
        "warnings": warnings,
        "not_checked": not_checked,
        "left_out": left_out,
    }


def rename_refusal(workspace: Path, payload: dict[str, Any]) -> str:
    """Why a submitted scenario renames one already in the suite, or "" when it does not."""
    name = str(payload.get("name") or "").strip()
    key = str(payload.get("scenario_key") or "").strip() or _slug(name)
    for one in load_scenarios(workspace):
        if one.scenario_key == key and one.name != name:
            return f"Not kept: {name} would rename {one.name}. A rebuild never renames a scenario."
        if one.name == name and one.scenario_key != key:
            return f"Not kept: {name} would change its key. A rebuild never renames a scenario."
    return ""


def aim_refusal(args: dict[str, Any], floor: int, suite: int) -> str:
    """Why this aim_for would loosen the suite's limits in a rebuild, or ""."""
    if any(value for key, value in args.items() if key != "count"):
        return "Not set: a rebuild only raises the count with aim_for."
    count = int(args.get("count") or 0)
    if count < floor or count > suite + REBUILD_ADDS_AT_MOST:
        return (
            f"Not set: aim between {floor} and {suite + REBUILD_ADDS_AT_MOST}, "
            "the suite plus what this turn adds."
        )
    return ""


def guard_renames(server: ToolServer, workspace: Path) -> ToolServer:
    """The same server, with ``submit_scenario`` refusing renames and ``aim_for`` only
    raising the count (suite nudge turns)."""
    floor = max(len(load_scenarios(workspace)), 1)
    tools = []
    for one in server.tools:
        if one.name == "aim_for":
            aim = one.handler

            async def aimed(args: dict[str, Any], _aim=aim) -> dict[str, Any]:
                said = aim_refusal(args, floor, len(load_scenarios(workspace)))
                if said:
                    return _err(said)
                return await _aim(args)

            one = replace(one, handler=aimed)
        if one.name == "submit_scenario":
            submit = one.handler

            async def guarded(args: dict[str, Any], _submit=submit) -> dict[str, Any]:
                said = rename_refusal(workspace, args)
                if said:
                    return _err(said)
                return await _submit(args)

            one = replace(one, handler=guarded)
        tools.append(one)
    return tool_server(name=server.name, version=server.version, tools=tools)


def nudge_server(
    workspace: Path,
    nudge: dict[str, Any],
    *,
    source: Path | None = None,
    catalogue: Catalogue | None = None,
    kept: list[Scenario] | None = None,
    reported: dict[str, Any] | None = None,
) -> ToolServer:
    """``retire_sub_goal`` and ``report_update`` for one nudge turn.

    ``catalogue`` and ``kept`` are the ones the session's scenario server holds. ``kept`` must be
    its whole list: saving it removes every scenario folder it does not name. ``reported``
    receives each report_update's arguments, so the turn can run the gate again at its end.
    """
    workspace = Path(workspace)
    stage = str(nudge.get("stage") or "")
    started_with = (
        {
            one.scenario_key
            for one in (kept if kept is not None else load_scenarios(workspace))
        }
        if stage == SUITE
        else None
    )

    @tool(
        "retire_sub_goal",
        "Retire a sub-goal instead of renaming it. In the environment stage it leaves the "
        "catalogue; in the suite stage it is dropped from every scenario naming it, once the "
        "environment no longer defines it. Say why.",
        schema({"name": str, "why": str}, ["name", "why"]),
    )
    async def retire(args: dict[str, Any]) -> dict[str, Any]:
        said = retire_sub_goal(
            workspace,
            str(args.get("name") or ""),
            str(args.get("why") or ""),
            stage=stage,
            catalogue=catalogue,
            kept=kept,
        )
        return _err(said) if said.startswith("Not retired") else _ok(said)

    @tool(
        "report_update",
        "End the update with this. Runs this stage's checks and records what changed; fix what "
        "it reports and call it again. `changed` is false when nothing needed to change.",
        schema({"changed": bool, "notes": str}, ["changed", "notes"]),
    )
    async def report(args: dict[str, Any]) -> dict[str, Any]:
        done = {
            "changed": bool(args.get("changed")),
            "notes": str(args.get("notes") or ""),
            "source": source,
            "started_with": started_with,
        }
        result = report_update(workspace, nudge=nudge, **done)
        if reported is not None:
            reported.update(done)
        said = json.dumps(result, indent=1)
        return _ok(said) if result["passed"] else _err(said)

    return tool_server(name=NUDGE_SERVER, tools=[retire, report])


_USAGE_TOTALS = ("tokens_in", "tokens_out", "tokens_cached")


def usage_delta(before: dict, after: dict) -> dict:
    """What one turn spent, from two ``spend.snapshot()`` readings."""

    def by_stage(reading: dict) -> dict[str, dict]:
        return {str(one.get("stage")): one for one in reading.get("stages") or []}

    earlier, later = by_stage(before), by_stage(after)
    delta: dict[str, Any] = {
        field: sum(
            max(
                0, int(one.get(field) or 0) - int(earlier.get(name, {}).get(field) or 0)
            )
            for name, one in later.items()
        )
        for field in _USAGE_TOTALS
    }
    delta["usd"] = round(
        max(
            0.0,
            float(after.get("total_usd") or 0) - float(before.get("total_usd") or 0),
        ),
        6,
    )
    delta["unpriced_turns"] = max(
        0,
        int(after.get("unpriced_turns") or 0) - int(before.get("unpriced_turns") or 0),
    )
    # Models of the stages that took turns since ``before``.
    delta["models"] = sorted(
        {
            model
            for name, one in later.items()
            if int(one.get("turns") or 0) > int(earlier.get(name, {}).get("turns") or 0)
            for model in one.get("models") or []
        }
    )
    return delta
