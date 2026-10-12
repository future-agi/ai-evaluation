"""What adaptation asks the harness model to write, and the operator-supplied alternative.

The model writes three things: the `tool_routed` mapping with agent-vocabulary trajectories,
persona/instruction rewrites of base scenarios, and complete custom scenarios. It never decides
whether its work is good: every submission is checked by the caller's validator (sealing,
proving) inside the tool call, and the reasons go straight back to the model. Structured output
is a submit tool, the pattern every authoring stage uses; nested documents travel as JSON text
because Vertex function declarations cannot describe the mapping language's recursive values.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import AdaptationFailure

DEFAULT_MODEL = "gemini-3.7-flash"
_ROW_LIMIT = 200


@dataclass
class BindingDraft:
    mapping: dict[str, Any]
    trajectories: dict[str, list[dict[str, Any]]]
    extension_candidates: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class DraftingContext:
    environment_tools: list[dict[str, Any]]
    policy: dict[str, Any]
    agent_tools: list[dict[str, Any]]
    agent: dict[str, Any]
    scenarios: list[dict[str, Any]]
    world_database: Path
    simulator_prompt: str
    # The sealed tool_routed mapping, once there is one: custom scenarios carry trajectories
    # through it.
    binding_mapping: dict[str, Any] | None = None


BindingValidator = Callable[[BindingDraft], list[str]]
ItemValidator = Callable[[dict[str, Any]], Awaitable[list[str]]]


class Drafter:
    async def binding(
        self, context: DraftingContext, validate: BindingValidator, feedback: str | None
    ) -> BindingDraft | None:
        raise NotImplementedError

    async def specializations(
        self, context: DraftingContext, validate: ItemValidator, limit: int
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def custom_scenarios(
        self, context: DraftingContext, validate: ItemValidator, limit: int
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def parse_binding_draft(body: Mapping[str, Any]) -> BindingDraft:
    mapping = body.get("binding")
    trajectories = body.get("trajectories")
    candidates = body.get("extension_candidates") or []
    if not isinstance(mapping, Mapping) or not isinstance(trajectories, Mapping):
        raise ValueError("a draft needs a `binding` object and a `trajectories` object")
    if not isinstance(candidates, list) or not all(
        isinstance(one, Mapping) and {"agent_tool", "reason"} <= set(one)
        for one in candidates
    ):
        raise ValueError("extension_candidates must be a list of {agent_tool, reason}")
    return BindingDraft(
        mapping={
            key: mapping[key]
            for key in ("tools", "fixtures", "aliases", "rendering")
            if key in mapping
        },
        trajectories={str(key): list(calls) for key, calls in trajectories.items()},
        extension_candidates=[
            {"agent_tool": str(one["agent_tool"]), "reason": str(one["reason"])}
            for one in candidates
        ],
    )


class FileDrafter(Drafter):
    """Operator-supplied drafts: `{binding, trajectories, extension_candidates?, specialize?,
    add?}`. A file cannot answer feedback, so a draft that fails its proof fails adaptation."""

    def __init__(self, body: Mapping[str, Any]) -> None:
        self.body = dict(body)

    async def binding(
        self, context: DraftingContext, validate: BindingValidator, feedback: str | None
    ) -> BindingDraft | None:
        if feedback is not None:
            return None
        try:
            draft = parse_binding_draft(self.body)
        except ValueError as exc:
            raise AdaptationFailure("drafts_invalid", str(exc)) from exc
        if problems := validate(draft):
            raise AdaptationFailure("binding_invalid", "\n".join(problems))
        return draft

    async def _items(
        self, key: str, validate: ItemValidator, limit: int
    ) -> list[dict[str, Any]]:
        items = self.body.get(key) or []
        if not isinstance(items, list) or not all(
            isinstance(one, Mapping) for one in items
        ):
            raise AdaptationFailure(
                "drafts_invalid", f"`{key}` must be a list of objects"
            )
        accepted = []
        for item in items[:limit]:
            if problems := await validate(dict(item)):
                raise AdaptationFailure(
                    "drafts_invalid",
                    f"{key} {item.get('scenario_key')}: " + "; ".join(problems),
                )
            accepted.append(dict(item))
        return accepted

    async def specializations(
        self, context: DraftingContext, validate: ItemValidator, limit: int
    ) -> list[dict[str, Any]]:
        return await self._items("specialize", validate, limit)

    async def custom_scenarios(
        self, context: DraftingContext, validate: ItemValidator, limit: int
    ) -> list[dict[str, Any]]:
        return await self._items("add", validate, limit)


# --- the harness model ----------------------------------------------------------------------------

MAPPING_LANGUAGE = """\
A tool_routed binding maps each agent tool onto environment operations, declaratively.
binding = {"tools": [ToolBinding, one per agent tool, each agent tool exactly once],
           "fixtures": {name: JSON}, "aliases": {table: {agent value: environment value}},
           "rendering": {slot: string}}
ToolBinding is either mapped:
  {"agent_tool": "<agent tool name>",
   "steps": [{"id": "<slug unique in the entry>", "tool": "<environment tool>",
              "arguments": {"<environment argument>": Value}}],
   "response": {"template": "<Jinja2>"} or {"value": Value}   (optional)}
or {"agent_tool": "<name>", "unsupported": "<reason>"} when no environment tool can serve it.
A Value is exactly one of:
  {"arg": "<JMESPath over the agent's arguments>"}   a plain name is a rename; the first field
                                                    must be a parameter of the agent tool
  {"step": "<earlier step id>", "path": "<JMESPath>"} a value from an earlier step's result
  {"fixture": "<name>"}
  {"const": JSON}       only for an argument the environment tool does NOT require
  {"concat": [Value, ...], "separator": "<string>"}
  {"object": {key: Value}} or {"list": [Value, ...]}
Any Value may add modifiers, applied in order: "convert": one of string, integer, number,
boolean, lower, upper, trim, digits, date (to YYYY-MM-DD from ISO, MM/DD/YYYY, "Month D, YYYY",
"D Month YYYY"), time (to HH:MM), datetime (to YYYY-MM-DDTHH:MM); then "map": {"<from>": JSON}
(unmapped value is invalid); then "alias": "<table in aliases>" (unknown value is invalid).
Every required environment argument must be present and must come from the agent's arguments,
an earlier step or a fixture: a binding must never supply the answer itself.
Templates see `args`, `steps` (id -> result) and `fixtures`, with filters `json` and
`alias_back(table)`. Without a response the agent receives the last step's result.
One agent call runs its steps in order in one transaction; the first refusal rolls it back.
"""

PROOF_RULES = """\
How the binding is proved: for every scenario, on a fresh world with its setup applied, your
trajectory (the calls the agent would make, in the agent's own tool vocabulary and argument
formats, {"tool": agent tool, "arguments": {...}}) runs through the binding and must leave the
world so that the scenario's checks hold, exactly as its reference solution (environment
vocabulary) does. Then adversarial variants must FAIL the checks: no calls at all, each
state-changing call removed, and each state-changing call with every string argument replaced by
garbage. A scenario whose reference solution uses an environment tool no mapped step reaches is
unscorable and needs no trajectory. Write trajectories the way the agent really would call its
tools (its names, formats and casing, as its tool schemas and prompt describe them).
"""


def harness_model() -> str:
    from ..config import refuse_a_model_we_cannot_afford

    model = os.environ.get("ALK_HARNESS_MODEL", "").strip() or DEFAULT_MODEL
    refuse_a_model_we_cannot_afford(model)
    # Deployments route the authoring harness through a gateway ("vertex_ai/<model>"); the
    # drafter calls Vertex directly, which knows only the bare model id.
    return model.rsplit("/", 1)[-1]


def _say(text: str, *, error: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if error:
        body["is_error"] = True
    return body


def _json_argument(args: Mapping[str, Any], name: str, default: Any = None) -> Any:
    raw = args.get(name)
    if raw in (None, ""):
        return default
    if not isinstance(raw, str):
        return raw
    return json.loads(raw)


def _world_rows(database: Path, table: str) -> str:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        tables = {
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if table not in tables:
            return f"no table {table!r}; tables: {', '.join(sorted(tables))}"
        cursor = connection.execute(f'SELECT * FROM "{table}" LIMIT {_ROW_LIMIT}')
        columns = [column[0] for column in cursor.description]
        return json.dumps(
            [dict(zip(columns, row)) for row in cursor.fetchall()], default=str
        )
    finally:
        connection.close()


def _tables(database: Path) -> str:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        names = [
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        counts = {
            name: connection.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
            for name in names
        }
        return ", ".join(f"{name} ({count} rows)" for name, count in counts.items())
    finally:
        connection.close()


def _agent_summary(agent: Mapping[str, Any], limit: int = 8000) -> str:
    keep = {
        key: agent.get(key)
        for key in (
            "name",
            "begin_message",
            "general_prompt",
            "first_message",
            "model",
            "states",
            "conversation_flow",
        )
        if agent.get(key) not in (None, "", [], {})
    }
    text = json.dumps(keep, indent=1, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "\n...(truncated)"


def _common_context(context: DraftingContext) -> str:
    return (
        "ENVIRONMENT TOOLS (tools.json):\n"
        + json.dumps(context.environment_tools, indent=1)
        + "\n\nENVIRONMENT POLICY:\n"
        + json.dumps(context.policy, indent=1)
        + "\n\nWORLD TABLES: "
        + _tables(context.world_database)
        + " (read rows with world_rows)\n\nTHE AGENT (provider profile, redacted):\n"
        + _agent_summary(context.agent)
        + "\n\nTHE AGENT'S ADMITTED TOOLS:\n"
        + json.dumps(context.agent_tools, indent=1)
    )


def _scenarios_text(scenarios: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "scenario_key": one["key"],
                "instruction": one["instruction"],
                "persona": one["persona"],
                "setup_code": one["setup_code"],
                "reference_solution": one["solution"],
                "checks": one["checks"],
            }
            for one in scenarios
        ],
        indent=1,
        ensure_ascii=False,
    )


CUSTOM_SCENARIO_FORMAT = """\
A custom scenario is one JSON object:
{"name": "<folder slug, [a-z][a-z0-9_]*>", "scenario_key": "<kebab-case key>",
 "use_case": "<short use case>", "branch": "<what makes this case distinct>",
 "tests": "<what it shows about the agent>",
 "instruction": "<second-person task for the simulated customer: who they are, every fact they
                 may need (names, dates of birth, IDs they know), what they want>",
 "persona": {"name", "gender", "age_group", "personality", "communication_style",
             "initial_message", "languages": ["English"]},
 "setup_code": "def setup(world):\\n    world.put('<table>', {...})\\n",
 "ready_code": "def ready(world):\\n    ...return None when the setup holds, else a reason\\n",
 "solution": [{"tool": "<environment tool>", "arguments": {...}}],
 "sub_goals": [{"name": "<snake_case, new>", "what": "<one sentence>",
                "check": "def check(world, calls):\\n    rows = world.state()['<table>']\\n    ...return None when the outcome holds, else a reason string\\n"}],
 "trajectory": [{"tool": "<agent tool>", "arguments": {...}}]   (tool_routed agents only),
 "reason": "<why this scenario matters for this agent>"}
`world.state()` maps each table to a list of row dicts; `world.put(table, row)` inserts a row.
A check reads only the final world state, must hold after the reference solution and must fail
when nothing is done. Use only facts present in the world or created by setup_code.
"""


class LlmDrafter(Drafter):
    """The harness model on Vertex Gemini (`ALK_HARNESS_MODEL`, default gemini-3.7-flash)."""

    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.model = harness_model()
        self._binding_stage: Any = None
        self._accepted: dict[str, BindingDraft] = {}

    def _stage(self, name: str, system_prompt: str, tools: list[Any]) -> Any:
        from ..backends import SessionSpec, resolve, tool_server
        from ..session import Stage

        spec = SessionSpec(
            system_prompt=system_prompt,
            servers={"adaptation": tool_server(name="adaptation", tools=tools)},
            max_turns=40,
            model=self.model,
        )
        return Stage(spec, name=name, backend=resolve("vertex-gemini"), overheard=False)

    def _world_tool(self, context: DraftingContext) -> Any:
        from ..backends import tool
        from ..tools import schema

        @tool(
            "world_rows",
            "Rows of one world table (at most 200).",
            schema({"table": str}, ["table"]),
        )
        async def world_rows(args: dict[str, Any]) -> dict[str, Any]:
            return _say(
                _world_rows(context.world_database, str(args.get("table") or ""))
            )

        return world_rows

    async def binding(
        self, context: DraftingContext, validate: BindingValidator, feedback: str | None
    ) -> BindingDraft | None:
        from ..backends import tool
        from ..tools import schema

        self._accepted.clear()
        if self._binding_stage is None:

            @tool(
                "submit_binding",
                "Submit the complete binding and trajectories. Each argument is JSON text.",
                schema(
                    {
                        "binding": {
                            "type": "string",
                            "description": "JSON object {tools, fixtures, aliases, rendering}",
                        },
                        "trajectories": {
                            "type": "string",
                            "description": "JSON object {scenario_key: [{tool, arguments}]}",
                        },
                        "extension_candidates": {
                            "type": "string",
                            "description": "JSON list [{agent_tool, reason}]: unsupported agent "
                            "tools that act on business state this environment lacks",
                        },
                    },
                    ["binding", "trajectories"],
                ),
            )
            async def submit_binding(args: dict[str, Any]) -> dict[str, Any]:
                try:
                    draft = parse_binding_draft(
                        {
                            "binding": _json_argument(args, "binding"),
                            "trajectories": _json_argument(args, "trajectories"),
                            "extension_candidates": _json_argument(
                                args, "extension_candidates", []
                            ),
                        }
                    )
                except ValueError as exc:
                    return _say(f"not accepted: {exc}", error=True)
                if problems := validate(draft):
                    return _say("not accepted:\n" + "\n".join(problems), error=True)
                self._accepted["binding"] = draft
                return _say("accepted; it will now be proved. Stop here.")

            self._binding_stage = self._stage(
                "adaptation-mapping",
                "You bind a customer's AI agent to a prebuilt test environment. "
                "You write a declarative mapping from the agent's own tools onto the "
                "environment's tools, plus the calls the agent would make in each scenario.\n\n"
                + MAPPING_LANGUAGE
                + "\n"
                + PROOF_RULES,
                [self._world_tool(context), submit_binding],
            )
            await self._binding_stage.__aenter__()
            message = (
                _common_context(context)
                + "\n\nSCENARIOS (reference solutions in environment vocabulary):\n"
                + _scenarios_text(context.scenarios)
                + "\n\nRead the world rows you need (for aliases and argument formats), then "
                "call submit_binding once with a binding entry for every admitted agent tool "
                "and a trajectory for every scenario the binding can reach."
            )
        else:
            message = (
                "The binding was proved and failed:\n"
                + (feedback or "")
                + "\n\nFix the binding and/or the trajectories and call submit_binding again."
            )
        for text in (message, "Call submit_binding now with the complete binding."):
            turn = await self._binding_stage.say(text)
            if "binding" in self._accepted:
                return self._accepted["binding"]
            if turn.error:
                raise RuntimeError(f"the harness model failed: {turn.error}")
        return None

    async def _collect(
        self,
        name: str,
        system_prompt: str,
        message: str,
        tool_name: str,
        description: str,
        validate: ItemValidator,
        limit: int,
        context: DraftingContext,
    ) -> list[dict[str, Any]]:
        from ..backends import tool
        from ..tools import schema

        accepted: list[dict[str, Any]] = []

        @tool(
            tool_name,
            description,
            schema(
                {"item": {"type": "string", "description": "one JSON object"}}, ["item"]
            ),
        )
        async def submit(args: dict[str, Any]) -> dict[str, Any]:
            if len(accepted) >= limit:
                return _say(f"the limit of {limit} is reached; stop.")
            try:
                item = _json_argument(args, "item")
            except ValueError as exc:
                return _say(f"not accepted: invalid JSON: {exc}", error=True)
            if not isinstance(item, dict):
                return _say("not accepted: the item must be a JSON object", error=True)
            if problems := await validate(item):
                return _say("not accepted:\n" + "\n".join(problems), error=True)
            accepted.append(item)
            left = limit - len(accepted)
            return _say(
                f"accepted and proved. {left} more may be submitted."
                if left
                else "accepted and proved; stop."
            )

        stage = self._stage(name, system_prompt, [self._world_tool(context), submit])
        async with stage:
            turn = await stage.say(message)
        if turn.error and not accepted:
            raise RuntimeError(f"the harness model failed: {turn.error}")
        return accepted

    async def specializations(
        self, context: DraftingContext, validate: ItemValidator, limit: int
    ) -> list[dict[str, Any]]:
        return await self._collect(
            "adaptation-specialize",
            "You adapt generic test scenarios to one customer's business. You rewrite only the "
            "simulated customer's persona and instruction so the conversation sounds like this "
            "agent's domain (its brand, services and wording), keeping every fact the reference "
            "solution depends on (names, dates of birth, dates, IDs, amounts) exactly as given, "
            "and the same goal. The persona's name must not change.",
            _common_context(context)
            + "\n\nSCENARIOS:\n"
            + _scenarios_text(context.scenarios)
            + f"\n\nPick at most {limit} scenarios where a domain-specific rewrite makes the test "
            "more realistic for this agent and submit each with submit_specialization as "
            '{"scenario_key", "instruction", "persona": {...full persona...}, "reason"}. '
            "Submit nothing if the generic wording already fits.",
            "submit_specialization",
            "Submit one rewritten persona and instruction for an existing scenario.",
            validate,
            limit,
            context,
        )

    async def custom_scenarios(
        self, context: DraftingContext, validate: ItemValidator, limit: int
    ) -> list[dict[str, Any]]:
        routed = self.mode == "tool_routed"
        return await self._collect(
            "adaptation-add",
            "You write new test scenarios for a customer's AI agent against a prebuilt "
            "environment: setup, reference solution and executable checks, each proved before "
            "it is accepted.\n\n"
            + CUSTOM_SCENARIO_FORMAT
            + ("\n" + MAPPING_LANGUAGE + "\n" + PROOF_RULES if routed else ""),
            _common_context(context)
            + "\n\nEXISTING SCENARIOS (do not duplicate them):\n"
            + _scenarios_text(context.scenarios)
            + f"\n\nWrite at most {limit} new scenarios covering what this agent's prompt and "
            "tools emphasise that the existing scenarios miss. Submit each with submit_scenario; "
            "fix and resubmit when one is not accepted."
            + (
                " Include `trajectory`: the agent-vocabulary calls that reach the reference "
                "solution through this binding:\n"
                + json.dumps(context.binding_mapping, indent=1)
                if routed and context.binding_mapping
                else ""
            ),
            "submit_scenario",
            "Submit one complete custom scenario (JSON text).",
            validate,
            limit,
            context,
        )

    async def close(self) -> None:
        if self._binding_stage is not None:
            await self._binding_stage.__aexit__(None, None, None)
            self._binding_stage = None
