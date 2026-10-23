"""The design's own tools: the proposal the person approves, the environment's tools and rules,
and the gates that seal it.

Each tool changes the design view and checks the change before it answers. Nothing is built
before the person approves the proposal: until then the build tools refuse. The world they
change is the session's own in-memory world (the world server's while the world is being built,
a restored copy once the scenarios are), and every change is written straight back to the
workspace, so a checkpoint holds the schema, the rows and the handlers.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import traceback
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..backends import ToolServer, tool, tool_server
from ..catalogue import load_catalogue
from ..contract import AgentContract
from ..folder import read_all
from ..releases import (
    ATTACHED,
    POLICY,
    POLICY_TABLE,
    SERVERS,
    TOOLS,
    WORLD_IMPLEMENTATION,
    Policy,
    PolicyRule,
    ReleaseError,
    RuleClass,
    ToolDefinition,
    attached_tool_names,
    load_policy,
    load_tools,
    world_tool_specs,
)
from ..tools import schema
from ..world.runtime import GeneratedWorld
from ..world.snapshot import HANDLERS, read_manifest, save, saved
from ..world.tools import DRAFT
from .contract import write_contract
from .handlers import ALLOWED_IMPORTS, behaviour_problems, source_problems
from .workspace import Design, Proposal, load_design, save_design, seal_design

DESIGN_SERVER = "design"
FINISHED_MARKER = ".futureagi-build-finished.json"
# The question the person has not answered yet, written by whoever asked it; while it is there the
# design is waiting (or parked), and nothing changes.
PENDING_QUESTION = ".futureagi-pending-question.json"
PARKED = (
    "The person has not answered yet, so the design is parked until they do. End your turn now: "
    "say nothing more and call nothing else."
)
RESULT = "result.json"
# Writers and the judge read only the first 14 hard constraints.
MOST_AGENT_RULES = 14
APPROVE, CHANGE = "approve", "change"
APPROVAL_OPTIONS = [
    {
        "id": APPROVE,
        "label": "Build this",
        "description": "Approve the design and start building",
    },
    {
        "id": CHANGE,
        "label": "Change something",
        "description": "Say what to change first",
    },
]

Approve = Callable[[str, list[dict[str, str]]], Awaitable[str]]


def chosen_option(answer: str, options: list[dict[str, str]]) -> str | None:
    """The id of the option the person picked: its id, its label, or the card's flattened
    `label: description`."""
    said = answer.strip()
    for option in options:
        flattened = (
            f"{option['label']}: {option['description']}"
            if option.get("description")
            else option["label"]
        )
        if said in {option["id"], option["label"], flattened}:
            return option["id"]
    return None


def approval_question(name: str, proposal: Proposal) -> str:
    return (
        f"Approve this design for {name}?\n\n{proposal.one_liner}\n\n{proposal.summary}"
        "\n\nUse cases:\n- " + "\n- ".join(proposal.use_cases)
    )


def settle_proposal(workspace: Path, answer: str) -> bool:
    """Settle the pending proposal with the person's answer; whether they approved it."""
    design = load_design(workspace)
    proposal = design.pending_proposal
    approved = (
        proposal is not None and chosen_option(answer, APPROVAL_OPTIONS) == APPROVE
    )
    update: dict[str, Any] = {"pending_proposal": None}
    if approved:
        update.update(
            proposal=proposal,
            stage="world" if design.stage == "interview" else design.stage,
        )
    design = design.model_copy(update=update)
    save_design(workspace, design)
    if approved:
        write_contract(workspace, design)
    return approved


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text[:4000]}], "is_error": True}


def _safe(handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]) -> Any:
    """A tool's failure reported to the model as an error; raised, it would end the turn."""

    async def run(args: dict[str, Any]) -> dict[str, Any]:
        try:
            return await handler(args)
        except Exception as exc:  # noqa: BLE001 - reported to the model, never swallowed
            traceback.print_exc()
            return _err(f"The tool failed: {type(exc).__name__}: {exc}")

    return run


_POLICY_TABLE_NAMED = re.compile(rf"\b{POLICY_TABLE}\b", re.IGNORECASE)
_OWN_POLICY_TABLE = (
    f"{POLICY_TABLE} belongs to the policy: add_policy_rule makes it from each configurable "
    "rule's parameters, one row per parameter (name TEXT PRIMARY KEY, value TEXT NOT NULL, the "
    "value JSON). Read it in handlers; never create or fill it yourself."
)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class DesignTools:
    """The design tools over one workspace, one world and the contract the session reads."""

    def __init__(
        self,
        workspace: Path,
        world: GeneratedWorld,
        contract: AgentContract,
        *,
        approve: Approve,
        postgres_dsn: Callable[[], str],
        output: Path,
    ) -> None:
        self.workspace = workspace
        self.world = world
        self.contract = contract
        self.approve = approve
        self.postgres_dsn = postgres_dsn
        self.output = output
        self.done = False
        # Set when the session has to be opened again: the world was saved (the scenario tools
        # open), or the contract the writers were given changed.
        self.reopen = False

    # --- state --------------------------------------------------------------------------------

    def design(self) -> Design:
        return load_design(self.workspace)

    def _unapproved(self) -> str:
        if (self.workspace / PENDING_QUESTION).is_file():
            return PARKED
        if self.design().stage == "interview":
            return (
                "Nothing is built before the person approves the design. Ask what you need, then "
                "call propose_design."
            )
        return ""

    def _tools(self) -> list[ToolDefinition]:
        return load_tools(self.workspace) if (self.workspace / TOOLS).is_file() else []

    def _policy(self) -> Policy:
        return (
            load_policy(self.workspace)
            if (self.workspace / POLICY).is_file()
            else Policy(rules=[])
        )

    def refresh_contract(self) -> None:
        """The contract the session's tools hold, rebuilt from the design view in place, and
        written to `contract.json`."""
        fresh = write_contract(self.workspace, self.design())
        for field in ("one_liner", "tools", "hard_constraints", "real_use_cases"):
            setattr(self.contract, field, getattr(fresh, field))

    def snapshot(self) -> None:
        """The world written back to the workspace, ungated, with the sequences and world checks
        the world server holds (its draft while one is open)."""
        draft_path = self.workspace / DRAFT
        recorded = read_manifest(self.workspace) if saved(self.workspace) else {}
        draft = (
            json.loads(draft_path.read_text(encoding="utf-8"))
            if draft_path.is_file()
            else {}
        )
        self.world.tools = world_tool_specs(self.workspace) if self._tools() else []
        save(
            self.world,
            self.workspace,
            notes=str(recorded.get("notes") or ""),
            sequences=list(draft.get("sequences", recorded.get("sequences")) or []),
            world_checks=dict(
                draft.get("world_checks", recorded.get("world_checks")) or {}
            ),
            implementation=WORLD_IMPLEMENTATION,
        )

    def _changed(self) -> None:
        self.refresh_contract()
        self.snapshot()
        if self.design().stage == "suite":
            self.reopen = True

    def _using(self, name: str) -> list[str]:
        return sorted(
            scenario.name
            for scenario in read_all(self.workspace)
            if any(step.tool == name for step in scenario.solution)
        )

    def brief(self) -> str:
        design = self.design()
        lines = [
            f"Design: {design.name} ({design.modality}), stage {design.stage}.",
        ]
        if design.template is not None:
            lines.append(
                f"Customizing the template {design.template.environment.name} "
                f"({design.template.environment.digest[:19]}): its environment and suite are "
                "unpacked here. The result is a new environment, not a change to the template."
            )
        else:
            lines.append("Built from scratch: there is no template.")
        if design.inherited_schema:
            lines.append(
                "Inherited schema that PostgreSQL runs do not enforce (the handlers do): "
                + "; ".join(design.inherited_schema)
            )
        lines.append(
            f"\nThe person's requirements:\n{design.requirements or '(none given)'}"
        )
        if design.proposal is not None:
            lines.append(
                f"\nThe approved design:\n{design.proposal.one_liner}\n\n{design.proposal.summary}"
                "\n\nUse cases:\n- " + "\n- ".join(design.proposal.use_cases)
            )
        tools = self._tools()
        lines.append(
            f"\nTools: {', '.join(tool.name for tool in tools) or 'none yet'}. "
            f"Policy rules: {', '.join(rule.id for rule in self._policy().rules) or 'none yet'}. "
            f"Scenarios: {len(read_all(self.workspace))}."
        )
        return "\n".join(lines)

    # --- tools --------------------------------------------------------------------------------

    async def propose_design(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            proposal = Proposal.model_validate(
                {
                    "one_liner": args.get("one_liner"),
                    "summary": args.get("summary"),
                    "use_cases": args.get("use_cases"),
                }
            )
        except ValidationError as exc:
            return _err(f"Not proposed: {exc}")
        design = self.design()
        save_design(
            self.workspace, design.model_copy(update={"pending_proposal": proposal})
        )
        answer = await self.approve(
            approval_question(design.name, proposal), APPROVAL_OPTIONS
        )
        if (self.workspace / PENDING_QUESTION).is_file():
            return _err(PARKED)
        if not self.answer_proposal(answer):
            return _ok(
                f"Not approved. The person said: {answer or '(nothing)'}\n"
                "Change the design accordingly and propose it again; ask if it is unclear."
            )
        return _ok(
            "Approved. Build it now: the schema and seed rows, each tool with define_tool, the "
            "rules with add_policy_rule, sequences, world checks, sub-goals and the simulator "
            "prompt, then save_world."
        )

    def answer_proposal(self, answer: str) -> bool:
        approved = settle_proposal(self.workspace, answer)
        if approved:
            self.refresh_contract()
        return approved

    async def _put_tool(self, args: dict[str, Any], *, new: bool) -> dict[str, Any]:
        if refused := self._unapproved():
            return _err(refused)
        name = str(args.get("name") or "")
        tools = self._tools()
        exists = any(tool.name == name for tool in tools)
        if new and exists:
            return _err(f"{name} already exists; use change_tool")
        if not new and not exists:
            return _err(
                f"there is no tool {name}; tools: "
                f"{', '.join(tool.name for tool in tools) or 'none'}"
            )
        if "__" in name or name in attached_tool_names(self.workspace):
            return _err(f"{name} is an attached server's tool name; choose another")
        try:
            definition = ToolDefinition.model_validate(
                {
                    "name": name,
                    "description": args.get("description"),
                    "parameters": args.get("parameters"),
                    "mutates_state": args.get("mutates_state"),
                    "returns": args.get("returns"),
                }
            )
        except ValidationError as exc:
            return _err(f"Not saved: {exc}")
        source = str(args.get("handler") or "")
        if problems := source_problems(source):
            return _err(
                "Not saved. The handler "
                + "; ".join(problems)
                + ". It runs against the world alone: define handle(args, db), raise "
                "ToolError(message) to refuse, and import only "
                + ", ".join(sorted(ALLOWED_IMPORTS))
                + "."
            )
        sample = args.get("sample_arguments")
        if not isinstance(sample, dict):
            return _err(
                "sample_arguments must be an object of arguments the tool accepts"
            )
        if problems := behaviour_problems(self.world, definition, source, sample):
            return _err("Not saved:\n- " + "\n- ".join(problems))
        (self.workspace / HANDLERS).mkdir(exist_ok=True)
        (self.workspace / HANDLERS / f"{name}.py").write_text(source, encoding="utf-8")
        self.world.handlers[name] = source
        kept = [tool for tool in tools if tool.name != name]
        _json(
            self.workspace / TOOLS,
            [
                tool.model_dump(mode="json", exclude_none=True)
                for tool in (
                    [*kept, definition]
                    if new
                    else [definition if tool.name == name else tool for tool in tools]
                )
            ],
        )
        self._changed()
        using = self._using(name)
        return _ok(
            f"{name} {'defined' if new else 'changed'}; the sample call holds."
            + (
                f" Re-prove the scenarios that call it: {', '.join(using)}."
                if using and not new
                else ""
            )
        )

    async def define_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self._put_tool(args, new=True)

    async def change_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self._put_tool(args, new=False)

    async def drop_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        if refused := self._unapproved():
            return _err(refused)
        name = str(args.get("name") or "")
        tools = self._tools()
        if not any(tool.name == name for tool in tools):
            return _err(f"there is no tool {name}")
        # The design decides and the suite follows: scenarios that call a dropped tool no longer
        # prove, so nothing seals until they are rewritten or dropped.
        using = self._using(name)
        _json(
            self.workspace / TOOLS,
            [
                tool.model_dump(mode="json", exclude_none=True)
                for tool in tools
                if tool.name != name
            ],
        )
        (self.workspace / HANDLERS / f"{name}.py").unlink(missing_ok=True)
        self.world.handlers.pop(name, None)
        self._changed()
        return _ok(
            f"{name} dropped: its definition, handler and contract entry are gone."
            + (
                f" These scenarios call it and must be rewritten or dropped before the design "
                f"seals: {', '.join(using)}."
                if using
                else ""
            )
        )

    def _rows_from_policy(self, policy: Policy) -> None:
        """The world's `environment_policy` holds the parameters' defaults, which the handlers
        read. The design tools own the table: it is made again in its one shape every time."""
        defaults = {
            name: parameter.default for name, parameter in policy.parameters().items()
        }
        if not defaults and POLICY_TABLE not in self.world.state():
            return
        store = self.world.store
        store.execute(f"DROP TABLE IF EXISTS {POLICY_TABLE}")
        store.execute(
            f"CREATE TABLE {POLICY_TABLE} (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        for name, value in defaults.items():
            store.execute(
                f"INSERT INTO {POLICY_TABLE} (name, value) VALUES (?, ?)",
                (name, json.dumps(value)),
            )

    def _put_policy(
        self, rules: list[PolicyRule], graded: dict[str, list[str] | None]
    ) -> str:
        try:
            policy = Policy(rules=rules)
        except ValidationError as exc:
            return f"Not saved: {exc}"
        agent_rules = policy.rule_ids(RuleClass.AGENT_POLICY)
        if len(agent_rules) > MOST_AGENT_RULES:
            return (
                f"Not saved: {len(agent_rules)} agent rules; writers and the judge read at most "
                f"{MOST_AGENT_RULES}. Merge some."
            )
        catalogue = {one.name for one in load_catalogue(self.workspace).sub_goals}
        design = self.design()
        scored = {
            rule: names
            for rule, names in design.policy_profile.scored_agent_rules.items()
            if rule in agent_rules
        }
        for rule, names in graded.items():
            if names is None:
                continue
            if rule not in agent_rules:
                return f"Not saved: only agent_policy rules are graded, and {rule} is not one"
            if unknown := sorted(set(names) - catalogue):
                return (
                    f"Not saved: {', '.join(unknown)} not in the sub-goal catalogue; add them "
                    "with add_sub_goal first"
                )
            if names:
                scored[rule] = list(names)
            else:
                scored.pop(rule, None)
        # The world first: if its rows cannot be written, nothing else changes.
        self._rows_from_policy(policy)
        _json(
            self.workspace / POLICY,
            {
                "rules": [
                    rule.model_dump(mode="json", by_alias=True, exclude_defaults=True)
                    for rule in policy.rules
                ]
            },
        )
        save_design(
            self.workspace,
            design.model_copy(
                update={
                    "policy_profile": design.policy_profile.model_copy(
                        update={"scored_agent_rules": scored}
                    )
                }
            ),
        )
        self._changed()
        return ""

    @staticmethod
    def _rule(args: dict[str, Any], before: PolicyRule | None = None) -> PolicyRule:
        current = before.model_dump(by_alias=True) if before else {}
        return PolicyRule.model_validate(
            {
                "id": args.get("id"),
                "text": args.get("text") or current.get("text"),
                "class": args.get("rule_class") or current.get("class"),
                "parameters": args.get("parameters")
                if args.get("parameters") is not None
                else current.get("parameters") or {},
            }
        )

    async def add_policy_rule(self, args: dict[str, Any]) -> dict[str, Any]:
        if refused := self._unapproved():
            return _err(refused)
        rules = self._policy().rules
        if any(rule.id == args.get("id") for rule in rules):
            return _err(f"{args.get('id')} already exists; use change_policy_rule")
        try:
            rule = self._rule(args)
        except ValidationError as exc:
            return _err(f"Not saved: {exc}")
        if said := self._put_policy([*rules, rule], {rule.id: args.get("graded_by")}):
            return _err(said)
        return _ok(f"{rule.id} added ({rule.rule_class.value}).")

    async def change_policy_rule(self, args: dict[str, Any]) -> dict[str, Any]:
        if refused := self._unapproved():
            return _err(refused)
        rules = self._policy().rules
        before = next((rule for rule in rules if rule.id == args.get("id")), None)
        if before is None:
            return _err(f"there is no rule {args.get('id')}")
        try:
            rule = self._rule(args, before)
        except ValidationError as exc:
            return _err(f"Not saved: {exc}")
        replaced = [rule if one.id == rule.id else one for one in rules]
        if said := self._put_policy(replaced, {rule.id: args.get("graded_by")}):
            return _err(said)
        return _ok(f"{rule.id} changed.")

    async def drop_policy_rule(self, args: dict[str, Any]) -> dict[str, Any]:
        if refused := self._unapproved():
            return _err(refused)
        rules = self._policy().rules
        if not any(rule.id == args.get("id") for rule in rules):
            return _err(f"there is no rule {args.get('id')}")
        kept = [rule for rule in rules if rule.id != args.get("id")]
        if said := self._put_policy(kept, {}):
            return _err(said)
        return _ok(f"{args.get('id')} dropped.")

    def _seal(self, out: Path) -> tuple[Any, Any]:
        return seal_design(self.workspace, out, postgres_dsn=self.postgres_dsn())

    def _not_ready(self) -> str:
        if (self.workspace / PENDING_QUESTION).is_file():
            return PARKED
        stage = self.design().stage
        if stage != "suite":
            return (
                "Not yet: the world is not saved. Build it and call save_world, then write the "
                "scenarios."
                if stage == "world"
                else self._unapproved()
            )
        return ""

    async def check_design(self, _args: dict[str, Any]) -> dict[str, Any]:
        if refused := self._not_ready():
            return _err(refused)
        with tempfile.TemporaryDirectory(prefix="design-check-") as scratch:
            try:
                _environment, suite = await asyncio.to_thread(
                    self._seal, Path(scratch) / "out"
                )
            except ReleaseError as exc:
                return _err(f"Would not seal. {exc.code}: {exc.detail}")
        unused = sorted(
            {tool.name for tool in self._tools()}
            - {
                step.tool
                for scenario in read_all(self.workspace)
                for step in scenario.solution
            }
        )
        return _ok(
            f"It seals: {len(suite.scenarios)} scenarios proved on SQLite and PostgreSQL, every "
            "agent rule graded and every use case covered."
            + (
                f" No scenario's solution calls {', '.join(unused)}: if the approved design "
                "removed them, drop_tool now; otherwise give them a scenario."
                if unused
                else ""
            )
            + " Call finish_design when the person's requirements are met."
        )

    async def detach_server(self, args: dict[str, Any]) -> dict[str, Any]:
        if refused := self._unapproved():
            return _err(refused)
        path = self.workspace / SERVERS
        servers = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        found = next((one for one in servers if one.get("id") == args.get("id")), None)
        if found is None:
            return _err(
                f"no attached server {args.get('id')!r}; attached: "
                f"{', '.join(str(one.get('id')) for one in servers) or 'none'}"
            )
        kept = [one for one in servers if one is not found]
        if kept:
            _json(path, kept)
        else:
            path.unlink()
        folder = str((found.get("attach") or {}).get("directory") or "")
        # A folder under attached/ that no remaining server reads goes with it.
        if (
            Path(folder).parts[:1] == (ATTACHED,)
            and ".." not in Path(folder).parts
            and not any(
                (one.get("attach") or {}).get("directory") == folder for one in kept
            )
        ):
            shutil.rmtree(self.workspace / folder, ignore_errors=True)
        self._changed()
        prefix = f"{found['id']}__"
        using = sorted(
            scenario.name
            for scenario in read_all(self.workspace)
            if any(step.tool.startswith(prefix) for step in scenario.solution)
        )
        return _ok(
            f"{found['id']} detached"
            + (f", and its folder {folder} removed" if folder else "")
            + "."
            + (
                f" These scenarios call its tools and must be rewritten or dropped: "
                f"{', '.join(using)}."
                if using
                else ""
            )
        )

    async def finish_design(self, _args: dict[str, Any]) -> dict[str, Any]:
        if refused := self._not_ready():
            return _err(refused)
        try:
            environment, suite = await asyncio.to_thread(self._seal, self.output)
        except ReleaseError as exc:
            return _err(f"Not finished; nothing was sealed. {exc.code}: {exc.detail}")
        self._finish(
            {
                "status": "completed",
                "environment": {"digest": environment.digest, "name": environment.name},
                "suite": {
                    "digest": suite.digest,
                    "name": suite.name,
                    "scenarios": len(suite.scenarios),
                },
            }
        )
        return _ok(
            f"Sealed: environment {environment.name} {environment.digest[:19]} and suite "
            f"{suite.name} {suite.digest[:19]} with {len(suite.scenarios)} scenarios. Tell the "
            "person it is ready, in a few lines."
        )

    async def stop_design(self, args: dict[str, Any]) -> dict[str, Any]:
        reason = str(args.get("reason") or "").strip()
        if not reason:
            return _err("say why the design cannot be finished")
        self._finish(
            {
                "status": "failed",
                "failure": {"code": "design_stopped", "message": reason},
            }
        )
        return _ok("Stopped. Tell the person why, in a line or two.")

    def _finish(self, result: dict[str, Any]) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        _json(self.output / RESULT, result)
        _json(self.workspace / FINISHED_MARKER, {"status": result["status"]})
        self.done = True

    def guard_world(self, server: ToolServer, left_out: frozenset[str]) -> ToolServer:
        """The world server for a design: without `left_out`, refusing to build before approval,
        writing the world back after every change, and opening the scenario tools on save."""
        building = {"create_schema", "seed", "change_data", "run_tool"}
        drafted = {"declare_sequence", "drop_sequence", "add_world_check"}

        def guarded(spec: Any) -> Any:
            handler = spec.handler

            async def run(args: dict[str, Any]) -> dict[str, Any]:
                if spec.name in building | drafted | {"save_world"} and (
                    refused := self._unapproved()
                ):
                    return _err(refused)
                if (
                    spec.name in {"create_schema", "change_data"}
                    and _POLICY_TABLE_NAMED.search(str(args.get("sql") or ""))
                ) or (
                    spec.name == "seed"
                    and str(args.get("table") or "").lower() == POLICY_TABLE
                ):
                    return _err(_OWN_POLICY_TABLE)
                result = await handler(args)
                if result.get("is_error"):
                    return result
                if spec.name in building | drafted:
                    self.snapshot()
                if spec.name == "save_world":
                    design = self.design()
                    save_design(
                        self.workspace, design.model_copy(update={"stage": "suite"})
                    )
                    self.refresh_contract()
                    self.reopen = True
                return result

            return type(spec)(
                name=spec.name,
                description=spec.description,
                input_schema=spec.input_schema,
                handler=_safe(run),
            )

        return tool_server(
            name=server.name,
            version=server.version,
            tools=[guarded(spec) for spec in server.tools if spec.name not in left_out],
        )

    def server(self) -> ToolServer:
        parameters = {
            "type": "object",
            "description": "JSON Schema of the arguments: type object, properties, required",
        }
        definition = {
            "name": str,
            "description": str,
            "parameters": parameters,
            "mutates_state": bool,
            "returns": {
                "type": "object",
                "description": "JSON Schema of what it returns",
            },
            "handler": str,
            "sample_arguments": {
                "type": "object",
                "description": "arguments the tool accepts and acts on in the world as it stands",
            },
        }
        rule = {
            "id": str,
            "text": str,
            "rule_class": {
                "type": "string",
                "enum": [one.value for one in RuleClass],
            },
            "parameters": {
                "type": "object",
                "description": "configurable_environment_policy only: {name: {type, default, allowed}}",
            },
            "graded_by": {
                "type": "array",
                "items": {"type": "string"},
                "description": "agent_policy only: the sub-goals that grade it",
            },
        }
        return tool_server(
            name=DESIGN_SERVER,
            tools=[
                tool(
                    "propose_design",
                    "Ask the person to approve the design before anything is built: a one-line "
                    "description, a summary (who the agent serves and over which channel, each "
                    "tool and what it changes, the rules, the data, what failure looks like) and "
                    "the use cases every scenario belongs to. Waits for their answer.",
                    schema(
                        {
                            "one_liner": str,
                            "summary": str,
                            "use_cases": {"type": "array", "items": {"type": "string"}},
                        },
                        ["one_liner", "summary", "use_cases"],
                    ),
                )(_safe(self.propose_design)),
                tool(
                    "define_tool",
                    "Add one of the environment's tools: its definition and its handler, "
                    "`handle(args, db)` against the world's database (`db.query`, `db.one`, "
                    "`db.execute` with `?` placeholders), raising ToolError(message) to refuse. "
                    "It is called with sample_arguments on a copy of the world before it is "
                    "kept: the call must succeed, change the world exactly when mutates_state "
                    "says so, refuse a missing required argument, and respond to every argument.",
                    schema(definition, list(definition)),
                )(_safe(self.define_tool)),
                tool(
                    "change_tool",
                    "Replace an existing tool's definition and handler, with the same checks as "
                    "define_tool.",
                    schema(definition, list(definition)),
                )(_safe(self.change_tool)),
                tool(
                    "drop_tool",
                    "Remove a tool: its definition, handler and contract entry.",
                    schema({"name": str}, ["name"]),
                )(_safe(self.drop_tool)),
                tool(
                    "add_policy_rule",
                    "Add one of the environment's rules to policy.json. system_invariant: the "
                    "world enforces it. configurable_environment_policy: the world enforces it "
                    "with parameters the handlers read from environment_policy. agent_policy: the "
                    "agent must follow it, and a sub-goal grades it (graded_by).",
                    schema(rule, ["id", "text", "rule_class"]),
                )(_safe(self.add_policy_rule)),
                tool(
                    "change_policy_rule",
                    "Change a rule's text, parameters or the sub-goals that grade it.",
                    schema(rule, ["id"]),
                )(_safe(self.change_policy_rule)),
                tool(
                    "drop_policy_rule",
                    "Remove a rule.",
                    schema({"id": str}, ["id"]),
                )(_safe(self.drop_policy_rule)),
                tool(
                    "detach_server",
                    "Remove an attached MCP server from the environment, with the folder under "
                    "attached/ it reads. A customization drops what no longer fits this way.",
                    schema({"id": str}, ["id"]),
                )(_safe(self.detach_server)),
                tool(
                    "check_design",
                    "Run every sealing gate on a copy, PostgreSQL proofs included, and say what "
                    "would refuse it. Changes nothing.",
                    schema({}, []),
                )(_safe(self.check_design)),
                tool(
                    "finish_design",
                    "Seal the environment and its suite. Refused until every gate passes.",
                    schema({}, []),
                )(_safe(self.finish_design)),
                tool(
                    "stop_design",
                    "Stop: the design cannot be finished. Say why.",
                    schema({"reason": str}, ["reason"]),
                )(_safe(self.stop_design)),
            ],
        )
