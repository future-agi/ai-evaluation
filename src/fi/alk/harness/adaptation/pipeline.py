"""Adaptation end to end (`environment-releases.md` §7.5).

inspect -> map -> prove -> ask -> propose -> seal. Every decision that matters is made by the
release sealer or the binding proof; the harness model only drafts, and the asker only chooses
among proved options. The outputs are written to one directory: `result.json` always (failure
included), `profile.json` once the agent is inspected, and `binding.json`, `binding-proof.json`
and `suite/` on completion.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..catalogue import Catalogue, SubGoal, load_catalogue, validate_sub_goal
from ..folder import read_folder, write_folder
from ..releases import (
    SCENARIOS,
    BindingMode,
    PolicyProfile,
    ReleaseError,
    RuleClass,
    ScenarioOrigin,
    SuiteScenario,
    load_environment_release,
    load_policy,
    load_suite_release,
    load_tools,
    seal_suite_release,
)
from ..scenario import Persona, Scenario, Step
from .askers import Asker, Option, Proposal, Question
from .drafting import BindingDraft, Drafter, DraftingContext
from .errors import AdaptationFailure
from .profile import AgentProfile, seal_binding, select_mode
from .proofs import (
    SIMULATOR_PROMPT,
    BindingVerdict,
    Candidate,
    binding_verdicts,
    prove_binding_on,
    prove_candidates,
    write_suite_tree,
)

RESULT_SCHEMA = "futureagi.adaptation-result.v1"
RESULT = "result.json"
PROFILE = "profile.json"
BINDING = "binding.json"
BINDING_PROOF = "binding-proof.json"
SUITE = "suite"
STAGES = ("inspecting", "mapping", "proving", "asking", "proposing", "sealing")
MAX_ROUNDS = 3
SPECIALIZE_LIMIT = 3
ADD_LIMIT = 2


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug if slug and slug[0].isalpha() else fallback


def _label(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


@dataclass
class AdaptationInputs:
    environments: list[Path]  # base first, then the extension
    suite_root: Path
    output: Path
    postgres_dsn: str
    load_profile: Callable[[], AgentProfile]
    drafter_for: Callable[[BindingMode], Drafter]


@dataclass
class _Built:
    """A proposed scenario folder that proved, with what it needs in the final suite."""

    candidate: Candidate
    trajectory: list[dict[str, Any]] | None
    sub_goals: list[SubGoal] = field(default_factory=list)


class Adaptation:
    def __init__(self, inputs: AdaptationInputs, asker: Asker) -> None:
        self.inputs = inputs
        self.asker = asker
        self.stage: str | None = None
        self.result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA,
            "status": "failed",
            "failure": None,
            "environment": [],
            "base_suite": None,
            "suite": None,
            "binding": None,
            "alignment": None,
            "answers": {},
            "proposals": [],
        }
        self.drafter: Drafter | None = None

    # --- driving -----------------------------------------------------------------------------

    async def run(self) -> int:
        output = self.inputs.output
        output.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix="alk-adaptation-"))
        try:
            await self._adapt()
            self.result["status"] = "completed"
            code = 0
        except AdaptationFailure as exc:
            self.result["status"] = "failed"
            self.result["failure"] = {"code": exc.code, "message": exc.message}
            for name in (BINDING, BINDING_PROOF):
                (output / name).unlink(missing_ok=True)
            shutil.rmtree(output / SUITE, ignore_errors=True)
            await self.asker.say(f"Adaptation failed ({exc.code}): {exc.message}")
            code = 2
        finally:
            if self.drafter is not None:
                await self.drafter.close()
            shutil.rmtree(self.work, ignore_errors=True)
        _write_json(output / RESULT, self.result)
        return code

    async def _enter(self, stage: str) -> None:
        await self.asker.stage(self.stage, stage)
        self.stage = stage

    async def _adapt(self) -> None:
        self._load_releases()
        await self._enter("inspecting")
        self.profile = await asyncio.to_thread(self.inputs.load_profile)
        _write_json(self.inputs.output / PROFILE, self.profile.document())
        await self._say_inspected()

        await self._enter("mapping")
        self.mode = select_mode(self.profile, self.environment_tools)
        self.result["alignment"] = self.alignment = {
            "mode": self.mode.value,
            "admission": [
                one.model_dump(mode="json") for one in self.profile.admission
            ],
            "mapped": [],
            "unsupported": [],
            "extension_candidates": [],
            "unscorable": [],
        }
        self.trajectories: dict[str, list[dict[str, Any]]] = {}
        self.map_verdicts: dict[str, BindingVerdict] = {}
        if self.mode is BindingMode.TOOL_ROUTED:
            await self._map_tool_routed()
        else:
            self.binding = await asyncio.to_thread(
                seal_binding,
                self.profile,
                self.mode,
                None,
                environment_root=self.target_root,
            )
            if self.mode is BindingMode.CANONICAL:
                self.alignment["mapped"] = [
                    {"agent_tool": one.name, "steps": [one.name]}
                    for one in self.profile.admitted
                ]
            else:
                self.alignment["unscorable"] = [
                    {
                        "scenario_key": entry.key,
                        "reason": "the agent reaches no environment tool",
                    }
                    for entry in self.base_suite.scenarios
                ]
            await self.asker.say(f"Binding mode: {self.mode.value}.")

        await self._enter("asking")
        self.profile_answer = await self._ask_policy()

        await self._enter("proposing")
        decisions = await self._propose()

        await self._enter("sealing")
        await self._seal(decisions)

    # --- releases ----------------------------------------------------------------------------

    def _load_releases(self) -> None:
        environments = self.inputs.environments
        try:
            releases = [load_environment_release(environments[0])]
            if len(environments) == 2:
                releases.append(
                    load_environment_release(
                        environments[1], parent_root=environments[0]
                    )
                )
            self.base_suite = load_suite_release(self.inputs.suite_root)
        except ReleaseError as exc:
            raise AdaptationFailure(
                "release_invalid", f"{exc.code}: {exc.detail}"
            ) from exc
        self.result["environment"] = [
            {"digest": one.digest, "name": one.name} for one in releases
        ]
        self.result["base_suite"] = {"digest": self.base_suite.digest}
        self.target_root = environments[-1]
        self.target_digest = releases[-1].digest
        self.environment_tools = load_tools(self.target_root)
        self.policy = load_policy(self.target_root)
        self.catalogue = load_catalogue(self.inputs.suite_root)
        self.simulator_prompt = (self.inputs.suite_root / SIMULATOR_PROMPT).read_text(
            encoding="utf-8"
        )
        self.base_candidates = [
            Candidate(entry=entry, source=self.inputs.suite_root / entry.path)
            for entry in self.base_suite.scenarios
        ]
        self.base_profile = self.base_suite.policy_profile

    def _effective(
        self, profile: PolicyProfile
    ) -> tuple[dict[str, Any], frozenset[str]]:
        parameters = {
            name: profile.parameters.get(name, parameter.default)
            for name, parameter in self.policy.parameters().items()
        }
        return parameters, frozenset(profile.scored_agent_rules)

    def _base_proof_holds_for(self, profile: PolicyProfile) -> bool:
        """The base suite's own proof stands for `profile` on the target environment."""
        return (
            self.base_suite.environment.digest == self.target_digest
            and self._effective(profile) == self._effective(self.base_profile)
        )

    async def _prove(
        self,
        candidates: Sequence[Candidate],
        profile: PolicyProfile,
        catalogue: Catalogue,
    ) -> tuple[dict[str, str], Path | None]:
        try:
            return await asyncio.to_thread(
                prove_candidates,
                self.work,
                candidates,
                catalogue,
                self.simulator_prompt,
                environment_root=self.target_root,
                profile=profile,
                postgres_dsn=self.inputs.postgres_dsn,
            )
        except ReleaseError as exc:
            raise AdaptationFailure(
                "suite_unproved", f"{exc.code}: {exc.detail}"
            ) from exc

    async def _binding_proof(
        self, suite_root: Path, trajectories: Mapping[str, list[dict[str, Any]]]
    ) -> Any:
        try:
            return await asyncio.to_thread(
                prove_binding_on,
                suite_root,
                self.binding,
                trajectories,
                environment_root=self.target_root,
                postgres_dsn=self.inputs.postgres_dsn,
            )
        except ReleaseError as exc:
            raise AdaptationFailure(
                "binding_unproved", f"{exc.code}: {exc.detail}"
            ) from exc

    # --- inspect and map ----------------------------------------------------------------------

    async def _say_inspected(self) -> None:
        by_action: dict[str, list[str]] = {}
        for one in self.profile.admission:
            by_action.setdefault(one.action, []).append(one.name)
        summary = "; ".join(
            f"{action}: {', '.join(names)}" for action, names in by_action.items()
        )
        await self.asker.say(
            f"Inspected {self.profile.target.get('name') or 'the agent'} "
            f"({self.profile.connector}). Tools {summary or 'none'}."
        )

    def _drafting_context(self) -> DraftingContext:
        scenarios = []
        for candidate in self.base_candidates:
            scenario = read_folder(self.inputs.suite_root, candidate.folder)
            assert scenario is not None
            scenarios.append(self._describe(candidate.key, scenario, self.catalogue))
        return DraftingContext(
            environment_tools=[
                tool.model_dump(mode="json") for tool in self.environment_tools
            ],
            policy=json.loads(
                (self.target_root / "policy.json").read_text(encoding="utf-8")
            ),
            agent_tools=self.profile.agent_tools,
            agent=self.profile.target,
            scenarios=scenarios,
            world_database=self.target_root / "world" / "world.sqlite",
            simulator_prompt=self.simulator_prompt,
        )

    @staticmethod
    def _describe(key: str, scenario: Scenario, catalogue: Catalogue) -> dict[str, Any]:
        return {
            "key": key,
            "folder": scenario.name,
            "instruction": scenario.instruction,
            "persona": scenario.persona.model_dump(mode="json")
            if scenario.persona
            else {},
            "setup_code": scenario.setup_code,
            "solution": [
                {"tool": step.tool, "arguments": step.arguments}
                for step in scenario.solution
            ],
            "checks": {
                name: goal.check
                for name in scenario.sub_goals
                if (goal := catalogue.named(name)) is not None
            },
        }

    def _validate_binding(self, draft: BindingDraft) -> list[str]:
        try:
            seal_binding(
                self.profile,
                BindingMode.TOOL_ROUTED,
                draft.mapping,
                environment_root=self.target_root,
            )
        except ReleaseError as exc:
            return exc.detail.splitlines() or [exc.code]
        problems = []
        agent_tools = {one["name"] for one in self.profile.agent_tools}
        keys = {entry.key for entry in self.base_suite.scenarios}
        for key, calls in draft.trajectories.items():
            if key not in keys:
                problems.append(f"trajectory {key}: the suite has no such scenario")
                continue
            for index, call in enumerate(calls, start=1):
                if (
                    not isinstance(call, Mapping)
                    or call.get("tool") not in agent_tools
                    or not isinstance(call.get("arguments"), Mapping)
                ):
                    problems.append(
                        f"trajectory {key} call {index}: must be {{tool: one of "
                        f"{sorted(agent_tools)}, arguments: object}}"
                    )
        problems.extend(
            f"extension candidate {one['agent_tool']}: not an admitted agent tool"
            for one in draft.extension_candidates
            if one["agent_tool"] not in agent_tools
        )
        return problems

    async def _context_suite(
        self, profile: PolicyProfile
    ) -> tuple[dict[str, str], Path | None]:
        """The base scenarios proved under `profile` on the target, as a sealed suite."""
        if self._base_proof_holds_for(profile):
            return {}, self.inputs.suite_root
        return await self._prove(self.base_candidates, profile, self.catalogue)

    async def _map_tool_routed(self) -> None:
        self.context = self._drafting_context()
        self.drafter = self.inputs.drafter_for(self.mode)
        _, suite_root = await self._context_suite(self.base_profile)
        if suite_root is None:
            raise AdaptationFailure(
                "binding_unproved", "no base scenario proves on this environment"
            )
        feedback: str | None = None
        failures: list[str] = ["the harness produced no binding that seals"]
        for round_number in range(1, MAX_ROUNDS + 1):
            if round_number > 1:
                await self._enter("mapping")
            try:
                draft = await self.drafter.binding(
                    self.context, self._validate_binding, feedback
                )
            except AdaptationFailure:
                raise
            except Exception as exc:  # noqa: BLE001 - a drafting outage fails the job, not crashes it
                raise AdaptationFailure(
                    "drafting_failed", f"{type(exc).__name__}: {exc}"
                ) from exc
            if draft is None:
                break
            self.binding = await asyncio.to_thread(
                seal_binding,
                self.profile,
                BindingMode.TOOL_ROUTED,
                draft.mapping,
                environment_root=self.target_root,
            )
            await self._enter("proving")
            proof = await self._binding_proof(suite_root, draft.trajectories)
            verdicts = binding_verdicts(proof)
            failures = [
                f"{key}: {verdict.reason}"
                for key, verdict in verdicts.items()
                if verdict.status == "failed"
            ]
            if not failures and not any(
                v.status == "proved" for v in verdicts.values()
            ):
                failures = [
                    "no scenario is proved: every one is unscorable, so the binding reaches "
                    "none of the environment operations the scenarios need"
                ]
            if not failures:
                self.draft = draft
                self.trajectories = draft.trajectories
                self.map_verdicts = verdicts
                break
            feedback = "\n".join(failures)
            await self.asker.say(
                f"Binding proof round {round_number} failed for {len(failures)} scenario(s)."
            )
        if not self.map_verdicts:
            raise AdaptationFailure(
                "binding_unproved",
                f"no binding proved (at most {MAX_ROUNDS} drafting rounds):\n"
                + "\n".join(failures),
            )
        mapped, unsupported = [], []
        for entry in self.binding.tools:
            body = entry.model_dump(mode="json")
            if "unsupported" in body:
                unsupported.append(
                    {"agent_tool": body["agent_tool"], "reason": body["unsupported"]}
                )
            else:
                mapped.append(
                    {
                        "agent_tool": body["agent_tool"],
                        "steps": [step["tool"] for step in body["steps"]],
                    }
                )
        unsupported_names = {one["agent_tool"] for one in unsupported}
        self.alignment.update(
            mapped=mapped,
            unsupported=unsupported,
            extension_candidates=[
                one
                for one in self.draft.extension_candidates
                if one["agent_tool"] in unsupported_names
            ],
        )
        await self.asker.say(
            f"Binding proved: {len(mapped)} agent tool(s) mapped, {len(unsupported)} unsupported, "
            f"{sum(v.status == 'unscorable' for v in self.map_verdicts.values())} scenario(s) "
            "unscorable."
        )

    # --- ask ------------------------------------------------------------------------------------

    def _questions(self) -> list[Question]:
        questions = []
        for rule in self.policy.rules:
            if rule.rule_class is RuleClass.CONFIGURABLE_ENVIRONMENT_POLICY:
                for name, parameter in rule.parameters.items():
                    default = self.base_profile.parameters.get(name, parameter.default)
                    if parameter.allowed is not None:
                        values = parameter.allowed
                    elif parameter.type == "boolean":
                        values = [True, False]
                    else:
                        values = [default]
                    questions.append(
                        Question(
                            id=f"parameter:{name}",
                            kind="parameter",
                            name=name,
                            default=default,
                            prompt=f"{rule.text} Which value of {name} does your business use?",
                            options=tuple(
                                Option(value, _label(value)) for value in values
                            ),
                        )
                    )
            elif rule.rule_class is RuleClass.AGENT_POLICY:
                scored = rule.id in self.base_profile.scored_agent_rules
                questions.append(
                    Question(
                        id=f"agent_rule:{rule.id}",
                        kind="agent_rule",
                        name=rule.id,
                        default="yes" if scored else "no",
                        prompt=f"Should your agent be held to this rule? {rule.text}",
                        options=(Option("yes", "yes"), Option("no", "no")),
                    )
                )
        return questions

    async def _ask_policy(self) -> PolicyProfile:
        parameters: dict[str, Any] = {}
        scored: list[str] = []
        for question in self._questions():
            value = await self.asker.ask(question)
            self.result["answers"][question.id] = value
            if question.kind == "parameter":
                parameters[question.name] = value
            elif value == "yes":
                scored.append(question.name)
        return PolicyProfile(parameters=parameters, scored_agent_rules=scored)

    # --- propose ----------------------------------------------------------------------------------

    async def _propose(self) -> dict[str, Any]:
        profile = self.profile_answer
        routed = self.mode is BindingMode.TOOL_ROUTED
        unproved, profile_suite = await self._context_suite(profile)
        verdicts: dict[str, BindingVerdict] = {}
        if routed and profile_suite is not None:
            if profile_suite == self.inputs.suite_root and self.map_verdicts:
                verdicts = self.map_verdicts
            else:
                verdicts = binding_verdicts(
                    await self._binding_proof(profile_suite, self.trajectories)
                )
        proposals: list[tuple[Proposal, bool]] = []  # (proposal, may be rejected)
        self.alignment["unscorable"] = self.alignment["unscorable"] or [
            {"scenario_key": key, "reason": verdict.reason}
            for key, verdict in verdicts.items()
            if verdict.status == "unscorable"
        ]
        dropped_keys: set[str] = set()
        for candidate in self.base_candidates:
            key = candidate.key
            if key in unproved:
                reason, optional = (
                    f"It does not prove under the answered policy: {unproved[key]}",
                    False,
                )
            elif routed and key in verdicts and verdicts[key].status == "unscorable":
                reason, optional = (
                    f"Your agent's tools cannot reach it: {verdicts[key].reason}",
                    True,
                )
            elif routed and key in verdicts and verdicts[key].status == "failed":
                reason, optional = (
                    f"The binding does not prove it under the answered policy: {verdicts[key].reason}",
                    False,
                )
            else:
                continue
            dropped_keys.add(key)
            proposals.append(
                (
                    Proposal(
                        id=f"drop:{key}",
                        action="drop",
                        scenario_key=key,
                        reason=reason,
                        preview=self._preview(candidate),
                    ),
                    optional,
                )
            )
        keep = [one for one in self.base_candidates if one.key not in dropped_keys]
        if self.drafter is None:
            self.drafter = self.inputs.drafter_for(self.mode)
            self.context = self._drafting_context()
        self.context = replace(
            self.context,
            scenarios=[
                one
                for one in self.context.scenarios
                if one["key"] in {c.key for c in keep}
            ],
            binding_mapping=(
                self.binding.model_dump(
                    mode="json", include={"tools", "fixtures", "aliases"}
                )
                if routed
                else None
            ),
        )
        self.specialized: dict[str, _Built] = {}
        self.custom: dict[str, _Built] = {}
        try:
            for item in await self.drafter.specializations(
                self.context,
                self._specialization_validator(keep, profile),
                SPECIALIZE_LIMIT,
            ):
                built = self.specialized[str(item["scenario_key"])]
                proposals.append(
                    (
                        Proposal(
                            id=f"specialize:{item['scenario_key']}",
                            action="specialize",
                            scenario_key=str(item["scenario_key"]),
                            reason=str(
                                item.get("reason")
                                or "Rewritten in your agent's domain."
                            ),
                            preview=self._preview(built.candidate),
                        ),
                        True,
                    )
                )
            for item in await self.drafter.custom_scenarios(
                self.context, self._custom_validator(profile), ADD_LIMIT
            ):
                built = self.custom[str(item["scenario_key"])]
                proposals.append(
                    (
                        Proposal(
                            id=f"add:{item['scenario_key']}",
                            action="add",
                            scenario_key=str(item["scenario_key"]),
                            reason=str(
                                item.get("reason") or "A new scenario for your agent."
                            ),
                            preview=self._preview(built.candidate),
                        ),
                        True,
                    )
                )
        except AdaptationFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - a drafting outage fails the job, not crashes it
            raise AdaptationFailure(
                "drafting_failed", f"{type(exc).__name__}: {exc}"
            ) from exc

        decisions: dict[str, bool] = {}
        for proposal, optional in proposals:
            approved = await self.asker.confirm(proposal)
            self.result["proposals"].append(
                {
                    "id": proposal.id,
                    "action": proposal.action,
                    "scenario_key": proposal.scenario_key,
                    "reason": proposal.reason,
                    "approved": approved,
                }
            )
            if not approved and not optional:
                raise AdaptationFailure(
                    "suite_unproved",
                    f"{proposal.scenario_key} cannot be kept under the answered policy: "
                    f"{proposal.reason}",
                )
            decisions[proposal.id] = approved
        return decisions

    @staticmethod
    def _preview(candidate: Candidate) -> dict[str, Any]:
        body = json.loads(
            (candidate.source / "scenario.json").read_text(encoding="utf-8")
        )
        return {
            "name": body.get("name") or candidate.folder,
            "instruction": body.get("instruction") or "",
            "persona": body.get("persona") or {},
        }

    async def _proves_with_binding(
        self, built: _Built, profile: PolicyProfile, catalogue: Catalogue
    ) -> list[str]:
        failures, suite_root = await self._prove([built.candidate], profile, catalogue)
        if failures:
            return [f"does not prove: {reason}" for reason in failures.values()]
        if self.mode is not BindingMode.TOOL_ROUTED:
            return []
        assert suite_root is not None
        verdict = binding_verdicts(
            await self._binding_proof(
                suite_root, {built.candidate.key: built.trajectory or []}
            )
        ).get(built.candidate.key)
        if verdict is None or verdict.status != "proved":
            status = verdict.status if verdict else "no verdict"
            reason = verdict.reason if verdict else ""
            return [f"the binding does not prove it ({status}): {reason}"]
        return []

    def _specialization_validator(
        self, keep: Sequence[Candidate], profile: PolicyProfile
    ) -> Any:
        parents = {one.key: one for one in keep}

        async def validate(item: dict[str, Any]) -> list[str]:
            key = str(item.get("scenario_key") or "")
            parent = parents.get(key)
            if parent is None:
                return [f"scenario_key must be one of {sorted(parents)}"]
            if key in self.specialized:
                return [f"{key} is already specialized"]
            instruction = item.get("instruction")
            persona = item.get("persona")
            if not isinstance(instruction, str) or not instruction.strip():
                return ["instruction must be a non-empty string"]
            if not isinstance(persona, Mapping):
                return ["persona must be an object"]
            body = json.loads(
                (parent.source / "scenario.json").read_text(encoding="utf-8")
            )
            original = body.get("persona") or {}
            if persona.get("name") != original.get("name"):
                return [f"the persona's name must stay {original.get('name')!r}"]
            known = set(Persona.model_fields)
            if unknown := sorted(set(persona) - known):
                return [f"unknown persona fields {unknown}"]
            if instruction == body.get("instruction") and dict(persona) == original:
                return ["nothing was rewritten"]
            new_key = f"{key}-specialized"
            folder = f"{parent.folder}_specialized"
            root = self.work / "specialized" / folder
            shutil.rmtree(root, ignore_errors=True)
            shutil.copytree(parent.source, root)
            body.update(
                name=folder,
                scenario_key=new_key,
                instruction=instruction,
                persona={**original, **persona},
            )
            (root / "scenario.json").write_text(
                json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            built = _Built(
                candidate=Candidate(
                    entry=SuiteScenario(
                        key=new_key,
                        path=f"{SCENARIOS}/{folder}",
                        origin=ScenarioOrigin.SPECIALIZED,
                        derived_from=key,
                    ),
                    source=root,
                ),
                trajectory=self.trajectories.get(key),
            )
            if problems := await self._proves_with_binding(
                built, profile, self.catalogue
            ):
                return problems
            self.specialized[key] = built
            return []

        return validate

    def _custom_validator(self, profile: PolicyProfile) -> Any:
        environment_tools = {tool.name for tool in self.environment_tools}
        agent_tools = {one["name"] for one in self.profile.agent_tools}

        async def validate(item: dict[str, Any]) -> list[str]:
            taken_keys = {one.key for one in self.base_candidates} | {
                built.candidate.key
                for built in (*self.specialized.values(), *self.custom.values())
            }
            taken_folders = {one.folder for one in self.base_candidates} | {
                built.candidate.folder
                for built in (*self.specialized.values(), *self.custom.values())
            }
            taken_goals = set(self.catalogue.names()) | {
                goal.name for built in self.custom.values() for goal in built.sub_goals
            }
            problems: list[str] = []
            name = str(item.get("name") or "")
            key = str(item.get("scenario_key") or "")
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in taken_folders:
                problems.append(
                    f"name must be a new slug, not one of {sorted(taken_folders)}"
                )
            if not key or key in taken_keys:
                problems.append(
                    f"scenario_key must be new, not one of {sorted(taken_keys)}"
                )
            goals_raw = item.get("sub_goals")
            goals: list[SubGoal] = []
            if not isinstance(goals_raw, list) or not goals_raw:
                problems.append("sub_goals must be a non-empty list")
            else:
                for raw in goals_raw:
                    try:
                        goal = SubGoal.model_validate(raw)
                    except ValueError as exc:
                        problems.append(f"sub_goal: {exc}")
                        continue
                    if goal.name in taken_goals:
                        problems.append(f"sub_goal {goal.name}: the name is taken")
                    if goal.judged or not goal.check.strip():
                        problems.append(
                            f"sub_goal {goal.name}: needs a code check, no judged criteria"
                        )
                    problems.extend(validate_sub_goal(goal))
                    goals.append(goal)
            try:
                solution = [
                    Step.model_validate(step) for step in item.get("solution") or []
                ]
                persona = Persona.model_validate(item.get("persona") or {})
            except ValueError as exc:
                problems.append(str(exc))
                solution, persona = [], Persona()
            if not solution:
                problems.append("solution must list the reference calls")
            problems.extend(
                f"solution tool {step.tool} is not an environment tool"
                for step in solution
                if step.tool not in environment_tools
            )
            trajectory = item.get("trajectory")
            if self.mode is BindingMode.TOOL_ROUTED:
                if not isinstance(trajectory, list) or not all(
                    isinstance(call, Mapping)
                    and call.get("tool") in agent_tools
                    and isinstance(call.get("arguments"), Mapping)
                    for call in trajectory
                ):
                    problems.append(
                        f"trajectory must be [{{tool: one of {sorted(agent_tools)}, arguments}}]"
                    )
            if problems:
                return problems
            scenario = Scenario(
                name=name,
                scenario_key=key,
                use_case=str(item.get("use_case") or ""),
                branch=str(item.get("branch") or ""),
                tests=str(item.get("tests") or ""),
                instruction=str(item.get("instruction") or ""),
                persona=persona,
                setup_code=str(item.get("setup_code") or ""),
                ready_code=str(item.get("ready_code") or ""),
                solution=solution,
                sub_goals=[goal.name for goal in goals],
            )
            catalogue = self.catalogue.model_copy(
                update={"sub_goals": [*self.catalogue.sub_goals, *goals]}
            )
            root = self.work / "custom" / name
            shutil.rmtree(root, ignore_errors=True)
            write_folder(scenario, catalogue, root)
            built = _Built(
                candidate=Candidate(
                    entry=SuiteScenario(
                        key=key,
                        path=f"{SCENARIOS}/{name}",
                        origin=ScenarioOrigin.CUSTOM,
                    ),
                    source=root / SCENARIOS / name,
                ),
                trajectory=list(trajectory) if isinstance(trajectory, list) else None,
                sub_goals=goals,
            )
            if problems := await self._proves_with_binding(built, profile, catalogue):
                return problems
            self.custom[key] = built
            return []

        return validate

    # --- seal --------------------------------------------------------------------------------------

    async def _seal(self, decisions: Mapping[str, bool]) -> None:
        profile = self.profile_answer
        final: list[Candidate] = []
        trajectories: dict[str, list[dict[str, Any]]] = {}
        changed = False
        for candidate in self.base_candidates:
            key = candidate.key
            if decisions.get(f"drop:{key}"):
                changed = True
                continue
            built = self.specialized.get(key)
            if built is not None and decisions.get(f"specialize:{key}"):
                changed = True
                final.append(built.candidate)
                trajectories[built.candidate.key] = built.trajectory or []
                continue
            final.append(
                Candidate(
                    entry=candidate.entry.model_copy(
                        update={"origin": ScenarioOrigin.BASE, "derived_from": key}
                    ),
                    source=candidate.source,
                )
            )
            if key in self.trajectories:
                trajectories[key] = self.trajectories[key]
        custom_goals: list[SubGoal] = []
        for key, built in self.custom.items():
            if decisions.get(f"add:{key}"):
                changed = True
                final.append(built.candidate)
                custom_goals.extend(built.sub_goals)
                trajectories[key] = built.trajectory or []
        if not final:
            raise AdaptationFailure("suite_empty", "every scenario was dropped")

        if not changed and self._base_proof_holds_for(profile):
            suite_root, suite = self.inputs.suite_root, self.base_suite
            self.result["suite"] = {
                "digest": suite.digest,
                "name": suite.name,
                "derived": False,
            }
            await self.asker.say("Nothing changed: the base suite is used as it is.")
        else:
            suite_root = self.inputs.output / SUITE
            shutil.rmtree(suite_root, ignore_errors=True)
            catalogue = self.catalogue.model_copy(
                update={"sub_goals": [*self.catalogue.sub_goals, *custom_goals]}
            )
            write_suite_tree(suite_root, final, catalogue, self.simulator_prompt)
            name = _slug(
                f"{self.base_suite.name}_{self.profile.target.get('name') or 'adapted'}",
                "adapted_suite",
            )[:96]
            try:
                suite = await asyncio.to_thread(
                    seal_suite_release,
                    suite_root,
                    name=name,
                    environment_root=self.target_root,
                    scenarios=[one.entry for one in final],
                    parent_root=self.inputs.suite_root,
                    policy_profile=profile,
                    postgres_dsn=self.inputs.postgres_dsn,
                )
            except ReleaseError as exc:
                raise AdaptationFailure(
                    "suite_unproved", f"{exc.code}: {exc.detail}"
                ) from exc
            self.result["suite"] = {
                "digest": suite.digest,
                "name": suite.name,
                "derived": True,
            }
            await self.asker.say(
                f"Sealed suite {suite.name} with {len(final)} scenario(s), proved on PostgreSQL."
            )

        if self.mode is BindingMode.TOOL_ROUTED:
            proof = await self._binding_proof(suite_root, trajectories)
            document = proof.model_dump(mode="json")
            verdicts = binding_verdicts(proof)
            if not (
                document.get("passed")
                and document["binding"]["digest"] == self.binding.digest
                and document["suite"]["digest"] == suite.digest
                and any(verdict.status == "proved" for verdict in verdicts.values())
            ):
                failed = [
                    f"{key}: {verdict.status}: {verdict.reason}"
                    for key, verdict in verdicts.items()
                    if verdict.status != "proved"
                ]
                raise AdaptationFailure(
                    "binding_unproved",
                    "the binding proves no runnable scenario of the final suite:\n"
                    + "\n".join(failed),
                )
            _write_json(self.inputs.output / BINDING_PROOF, document)
            known = {one["scenario_key"] for one in self.alignment["unscorable"]}
            self.alignment["unscorable"].extend(
                {"scenario_key": key, "reason": verdict.reason}
                for key, verdict in verdicts.items()
                if verdict.status == "unscorable" and key not in known
            )
        _write_json(
            self.inputs.output / BINDING,
            self.binding.model_dump(mode="json", by_alias=True),
        )
        self.result["binding"] = {
            "digest": self.binding.digest,
            "mode": self.mode.value,
        }
