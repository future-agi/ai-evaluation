"""Scenario editing on a composition (`environment-releases.md` §7.8).

A composition is a prebuilt environment and a suite bound to a person's agent. Its chat opens the
environment's builder on a workspace the platform assembles: the delivered releases under
`releases/`, the sealed `binding.json` and its `binding-proof.json`, `composition.json`, and
`trajectories.json` (the agent's own proved calls, for a `tool_routed` binding). The world, its
tools and its policy are shared and fixed; only the suite changes.

`open_composition` verifies those inputs and, the first time, copies the world and the suite into
the workspace root, where the builder's scenario tools edit them as they would an authored
environment's. `seal_change` turns the edited copy into a child `SuiteRelease` of the composition's
suite, proved on PostgreSQL, proves the binding against it, and writes both to
`composition-change/` for the platform to publish. Code decides whether a change is sealed; the
session only drafts scenarios and trajectories.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .binding_proof import BindingProofV1, prove_binding
from .bindings import MappedToolBinding
from .contract import AgentContract
from .folder import document_for, read_folder, write_index
from .releases import (
    RELEASES_SUITE,
    SCENARIOS,
    SUB_GOALS,
    WORLD,
    AgentBindingV1,
    BindingMode,
    EnvironmentReleaseV1,
    Policy,
    ReleaseError,
    ScenarioOrigin,
    SuiteReleaseV1,
    SuiteScenario,
    _contract,
    apply_policy_profile,
    environment_directory,
    load_agent_binding,
    load_environment_release,
    load_policy,
    load_suite_release,
    load_tools,
    seal_suite_release,
)
from .scenario_tools import load_scenarios

COMPOSITION = "composition.json"
COMPOSITION_SCHEMA = "futureagi.composition-workspace.v1"
RELEASES = "releases"
BINDING = "binding.json"
BINDING_PROOF = "binding-proof.json"
TRAJECTORIES = "trajectories.json"
CHANGE = "composition-change"
CHANGE_MANIFEST = "change.json"
CHANGE_SCHEMA = "futureagi.composition-change.v1"
SIMULATOR_PROMPT = "simulator_prompt.md"

# The builder's tools that change the world or the contract. A composition's environment is a
# shared release, so its session never gets them.
FIXED_ENVIRONMENT_TOOLS = frozenset(
    {
        "change_world",
        "declare_tool",
        "amend_contract",
        "add_rule",
        "drop_rule",
        "fix_tool",
    }
)


class CompositionEditError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def is_composition(workspace: Path) -> bool:
    return (Path(workspace) / COMPOSITION).is_file()


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        raise CompositionEditError(
            "composition_workspace_invalid", f"{path.name}: {exc}"
        ) from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@dataclass
class Composition:
    """One composition's workspace, verified."""

    workspace: Path
    document: dict[str, Any]
    environments: list[EnvironmentReleaseV1]
    environment_root: Path
    suite_root: Path
    suite: SuiteReleaseV1
    binding: AgentBindingV1
    policy: Policy

    @property
    def routed(self) -> bool:
        return self.binding.mode is BindingMode.TOOL_ROUTED

    @property
    def modality(self) -> str:
        return "chat" if self.binding.agent.connector == "retell_chat" else "voice"

    @property
    def contract(self) -> AgentContract:
        return AgentContract.model_validate(
            _contract(self.suite.name, self.modality, load_tools(self.environment_root))
        )

    def trajectories(self) -> dict[str, list[dict[str, Any]]]:
        stored = _read_json(self.workspace / TRAJECTORIES, {})
        return stored if isinstance(stored, dict) else {}

    def proof(self) -> dict[str, Any]:
        stored = _read_json(self.workspace / BINDING_PROOF, {})
        return stored if isinstance(stored, dict) else {}

    def agent_tool_names(self) -> list[str]:
        return sorted(tool.name for tool in self.binding.agent_tools)

    def brief(self) -> str:
        """What the builder must know about this composition before the first question."""
        environment = self.environments[-1]
        lines = [
            "## This composition",
            "",
            (
                f"This environment is the prebuilt environment `{environment.name}` "
                f"({environment.digest[:19]}) with the suite `{self.suite.name}` "
                f"({self.suite.digest[:19]}, {len(self.suite.scenarios)} scenarios), bound to the "
                f"person's {self.binding.agent.connector} agent in `{self.binding.mode.value}` mode."
            ),
            "",
            (
                "The world, its tools and its policy are shared releases that other agents run too. "
                "Nothing in this conversation can change them: there is no tool to change the world's "
                "data, the tools or the contract, and none must be promised. What changes is the "
                "suite: add, rewrite and drop scenarios and add sub-goals, written in the "
                "environment's own vocabulary (the tools in the contract above), then call "
                "`save_changes`. Saving seals a new version of the suite whose parent is the current "
                "one; runs already made keep the version they ran."
            ),
            "",
            (
                f"The suite is proved under this policy profile: "
                f"{json.dumps(self.suite.policy_profile.model_dump(mode='json'), sort_keys=True)}. "
                "The policy rules every scenario's checks must respect:"
            ),
        ]
        lines.extend(
            f"- `{rule.id}` ({rule.rule_class.value}): {rule.text}"
            for rule in self.policy.rules
        )
        if self.routed:
            lines.extend(
                [
                    "",
                    (
                        "The person's agent calls its own tools; the binding maps them onto the "
                        "environment's:"
                    ),
                ]
            )
            for entry in self.binding.tools:
                if isinstance(entry, MappedToolBinding):
                    lines.append(
                        f"- `{entry.agent_tool}` -> "
                        + ", ".join(f"`{step.tool}`" for step in entry.steps)
                    )
                else:
                    lines.append(
                        f"- `{entry.agent_tool}`: unsupported ({entry.unsupported})"
                    )
            proof = self.proof().get("scenarios") or {}
            unscorable = sorted(
                key for key, one in proof.items() if one.get("status") == "unscorable"
            )
            missing = self.missing_trajectories()
            lines.extend(
                [
                    "",
                    (
                        f"The full binding is in {BINDING}; the agent's proved trajectories are in "
                        f"{TRAJECTORIES}. Every scenario the binding can reach needs the agent's own "
                        "trajectory: the calls the agent would make, in its own tool names and "
                        "argument formats, that solve the scenario. Write one with `write_trajectory` "
                        "for every scenario you add or rewrite. `save_changes` proves each one through "
                        "the binding on PostgreSQL and saves nothing until every reachable scenario "
                        "holds and every adversarial variant fails. A scenario whose solution needs a "
                        "tool no mapped step reaches is unscorable for this agent: it stays in the "
                        "suite, is not scored, and needs no trajectory."
                    ),
                ]
            )
            if unscorable:
                lines.append("Unscorable now: " + ", ".join(unscorable) + ".")
            if missing:
                lines.append(
                    "Reachable scenarios without a trajectory now: "
                    + ", ".join(missing)
                    + "."
                )
        return "\n".join(lines)

    def files_note(self) -> str:
        return (
            f"\n\nThe environment's working copy is in {self.workspace}: contract.json (the "
            "environment's tools, fixed), the world files, sub_goals.json, simulator_prompt.md "
            f"and scenarios/. The sealed releases it came from are under {RELEASES}/ and are "
            "read-only."
        )

    def missing_trajectories(self) -> list[str]:
        """Reachable scenarios in the working copy that have no trajectory."""
        if not self.routed:
            return []
        reached = {
            step.tool
            for entry in self.binding.tools
            if isinstance(entry, MappedToolBinding)
            for step in entry.steps
        }
        stored = self.trajectories()
        return sorted(
            scenario.scenario_key
            for scenario in load_scenarios(self.workspace)
            if {step.tool for step in scenario.solution} <= reached
            and scenario.scenario_key not in stored
        )

    def write_trajectory(self, scenario: str, calls: Sequence[Any]) -> tuple[bool, str]:
        if not self.routed:
            return False, f"a {self.binding.mode.value} binding needs no trajectories"
        wanted = str(scenario or "").strip()
        found = next(
            (
                one
                for one in load_scenarios(self.workspace)
                if wanted in {one.scenario_key, one.name}
            ),
            None,
        )
        if found is None:
            return False, f"no scenario {wanted!r} in the working copy"
        tools = self.agent_tool_names()
        problems = [
            f"call {index} must be {{tool: one of {tools}, arguments: object}}"
            for index, call in enumerate(calls, start=1)
            if not (
                isinstance(call, Mapping)
                and call.get("tool") in tools
                and isinstance(call.get("arguments", {}), Mapping)
            )
        ]
        if not calls:
            problems.append("a trajectory has at least one call")
        if problems:
            return False, "\n".join(problems)
        stored = self.trajectories()
        stored[found.scenario_key] = [
            {"tool": call["tool"], "arguments": dict(call.get("arguments") or {})}
            for call in calls
        ]
        _write_json(self.workspace / TRAJECTORIES, stored)
        return True, f"trajectory for {found.scenario_key} kept; save_changes proves it"


def open_composition(workspace: Path) -> tuple[Composition, bool]:
    """Verify a composition workspace; lay out its working copy if it has none.

    Returns the composition and whether the working copy was laid out now.
    """
    workspace = Path(workspace)
    document = _read_json(workspace / COMPOSITION, None)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != COMPOSITION_SCHEMA
    ):
        raise CompositionEditError(
            "composition_workspace_invalid", f"{COMPOSITION} is not a composition"
        )
    releases = workspace / RELEASES
    directories = [
        environment_directory(releases, str(entry.get("digest") or ""))
        for entry in document.get("environment") or []
    ]
    if not 1 <= len(directories) <= 2:
        raise CompositionEditError(
            "composition_workspace_invalid",
            "a composition names one environment or an extension",
        )
    try:
        environments = [load_environment_release(directories[0])]
        if len(directories) == 2:
            environments.append(
                load_environment_release(directories[1], parent_root=directories[0])
            )
        suite_root = releases / RELEASES_SUITE
        suite = load_suite_release(suite_root)
        binding = load_agent_binding(workspace / BINDING)
    except ReleaseError as exc:
        raise CompositionEditError(exc.code, exc.detail) from exc
    if suite.digest != (document.get("suite") or {}).get("digest"):
        raise CompositionEditError(
            "composition_workspace_invalid",
            "the delivered suite is not the composition's",
        )
    if binding.environment.digest != environments[-1].digest:
        raise CompositionEditError(
            "composition_workspace_invalid", "the binding targets another environment"
        )
    composition = Composition(
        workspace=workspace,
        document=document,
        environments=environments,
        environment_root=directories[-1],
        suite_root=suite_root,
        suite=suite,
        binding=binding,
        policy=load_policy(directories[-1]),
    )
    if (workspace / "contract.json").is_file():
        return composition, False
    _lay_out(composition)
    return composition, True


def _lay_out(composition: Composition) -> None:
    """The world with the suite's profile applied, the contract, and the suite to edit."""
    workspace = composition.workspace
    for path in (composition.environment_root / WORLD).iterdir():
        target = workspace / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(path, target)
    apply_policy_profile(
        workspace, composition.policy, composition.suite.policy_profile
    )
    _write_json(
        workspace / "contract.json", composition.contract.model_dump(mode="json")
    )
    shutil.rmtree(workspace / SCENARIOS, ignore_errors=True)
    for entry in composition.suite.scenarios:
        shutil.copytree(composition.suite_root / entry.path, workspace / entry.path)
    for name in (SUB_GOALS, SIMULATOR_PROMPT):
        if (composition.suite_root / name).is_file():
            shutil.copyfile(composition.suite_root / name, workspace / name)
    write_index(load_scenarios(workspace), workspace)


@dataclass
class ChangePlan:
    composition: Composition
    entries: list[SuiteScenario]
    sources: dict[str, Path]
    added: list[str]
    changed: list[str]
    removed: list[str]


def plan_change(composition: Composition) -> ChangePlan | None:
    """How the working copy differs from the composition's suite; None when it does not."""
    workspace, parent = composition.workspace, composition.suite
    kept = load_scenarios(workspace)
    if not kept:
        raise CompositionEditError("suite_empty", "a suite keeps at least one scenario")
    held = {
        entry.key: (entry, read_folder(composition.suite_root, Path(entry.path).name))
        for entry in parent.scenarios
    }
    entries: list[SuiteScenario] = []
    sources: dict[str, Path] = {}
    added, changed = [], []
    for scenario in sorted(kept, key=lambda one: one.scenario_key):
        key = scenario.scenario_key
        before = held.get(key)
        if (
            before is not None
            and before[1] is not None
            and document_for(before[1]) == document_for(scenario)
        ):
            # Unchanged: carried over byte for byte, so the sealer can check it is the parent's.
            entries.append(
                SuiteScenario(
                    key=key,
                    path=before[0].path,
                    origin=ScenarioOrigin.BASE,
                    derived_from=key,
                )
            )
            sources[before[0].path] = composition.suite_root / before[0].path
            continue
        path = f"{SCENARIOS}/{scenario.name}"
        if before is not None:
            entries.append(
                SuiteScenario(
                    key=key,
                    path=path,
                    origin=ScenarioOrigin.SPECIALIZED,
                    derived_from=key,
                )
            )
            changed.append(key)
        else:
            entries.append(
                SuiteScenario(key=key, path=path, origin=ScenarioOrigin.CUSTOM)
            )
            added.append(key)
        sources[path] = workspace / path
    removed = sorted(set(held) - {entry.key for entry in entries})
    same_files = all(
        _same(workspace / name, composition.suite_root / name)
        for name in (SUB_GOALS, SIMULATOR_PROMPT)
    )
    if not (added or changed or removed) and same_files:
        return None
    return ChangePlan(composition, entries, sources, added, changed, removed)


def _same(first: Path, second: Path) -> bool:
    if not (first.is_file() or second.is_file()):
        return True
    return (
        first.is_file()
        and second.is_file()
        and first.read_bytes() == second.read_bytes()
    )


def seal_change(plan: ChangePlan, *, postgres_dsn: str) -> dict[str, Any]:
    """Seal the child suite and, for a routed binding, prove the binding against it.

    Writes `composition-change/` only when both hold; raises `CompositionEditError` otherwise.
    """
    composition = plan.composition
    change = composition.workspace / CHANGE
    shutil.rmtree(change, ignore_errors=True)
    staging = change / RELEASES_SUITE
    (staging / SCENARIOS).mkdir(parents=True)
    for path, source in plan.sources.items():
        shutil.copytree(source, staging / path)
    for name in (SUB_GOALS, SIMULATOR_PROMPT):
        source = composition.workspace / name
        if source.is_file():
            shutil.copyfile(source, staging / name)
    try:
        suite = seal_suite_release(
            staging,
            name=composition.suite.name,
            environment_root=composition.environment_root,
            scenarios=plan.entries,
            parent_root=composition.suite_root,
            policy_profile=composition.suite.policy_profile,
            postgres_dsn=postgres_dsn,
        )
    except ReleaseError as exc:
        shutil.rmtree(change, ignore_errors=True)
        raise CompositionEditError(exc.code, exc.detail) from exc
    keys = {entry.key for entry in suite.scenarios}
    trajectories = {
        key: calls for key, calls in composition.trajectories().items() if key in keys
    }
    proof: BindingProofV1 | None = None
    if composition.routed:
        try:
            proof = prove_binding(
                composition.environment_root,
                staging,
                composition.binding,
                trajectories,
                postgres_dsn=postgres_dsn,
            )
        except ReleaseError as exc:
            shutil.rmtree(change, ignore_errors=True)
            raise CompositionEditError(exc.code, exc.detail) from exc
        if not proof.passed or not any(
            one.status == "proved" for one in proof.scenarios.values()
        ):
            shutil.rmtree(change, ignore_errors=True)
            failed = [
                f"{key}: {one.reason or one.status}"
                for key, one in sorted(proof.scenarios.items())
                if one.status == "failed"
            ]
            raise CompositionEditError(
                "binding_unproved",
                "the binding does not prove the edited suite; nothing was saved:\n- "
                + "\n- ".join(failed or ["no scenario is proved"]),
            )
        _write_json(change / BINDING_PROOF, proof.model_dump(mode="json"))
    _write_json(change / TRAJECTORIES, trajectories)
    summary = {
        "schema_version": CHANGE_SCHEMA,
        "parent": composition.suite.digest,
        "suite": suite.digest,
        "added": plan.added,
        "changed": plan.changed,
        "removed": plan.removed,
        "unscorable": {
            key: one.reason
            for key, one in sorted((proof.scenarios if proof else {}).items())
            if one.status == "unscorable"
        },
    }
    _write_json(change / CHANGE_MANIFEST, summary)
    return summary


def accept_change(composition: Composition) -> None:
    """After the platform published the change, make it this workspace's suite."""
    workspace, change = composition.workspace, composition.workspace / CHANGE
    summary = _read_json(change / CHANGE_MANIFEST, None)
    if not isinstance(summary, dict):
        return
    shutil.rmtree(composition.suite_root)
    shutil.move(str(change / RELEASES_SUITE), str(composition.suite_root))
    for name in (BINDING_PROOF, TRAJECTORIES):
        if (change / name).is_file():
            shutil.move(str(change / name), str(workspace / name))
    suite = load_suite_release(composition.suite_root)
    composition.document["suite"] = {"digest": suite.digest, "name": suite.name}
    _write_json(workspace / COMPOSITION, composition.document)
    composition.suite = suite
    shutil.rmtree(change, ignore_errors=True)


def trajectory_server(composition: Composition) -> Any:
    """The one tool a routed composition adds: the agent's own calls for a scenario."""
    from .backends import tool, tool_server
    from .tools import schema

    @tool(
        "write_trajectory",
        "Keep the person's agent's own trajectory for one scenario: the calls it would make, in "
        "its own tool names and argument formats, that solve the scenario through the binding. "
        "Replaces any trajectory the scenario had. `scenario` is its name or scenario_key. "
        "save_changes proves it.",
        schema(
            {
                "scenario": str,
                "calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["tool", "arguments"],
                    },
                },
            },
            ["scenario", "calls"],
        ),
    )
    async def write_trajectory(args: dict[str, Any]) -> dict[str, Any]:
        calls = args.get("calls")
        if isinstance(calls, str):
            try:
                calls = json.loads(calls)
            except ValueError:
                calls = None
        if not isinstance(calls, list):
            ok, said = False, "`calls` must be a JSON list of {tool, arguments}"
        else:
            ok, said = composition.write_trajectory(
                str(args.get("scenario") or ""), calls
            )
        body: dict[str, Any] = {"content": [{"type": "text", "text": said}]}
        if not ok:
            body["is_error"] = True
        return body

    return tool_server(name="composition", version="0.1.0", tools=[write_trajectory])
