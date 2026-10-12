"""Binding proofs (`environment-releases.md` §2.4.3).

A `tool_routed` binding is used for runs only once it is shown, on PostgreSQL and through the tool
server's own execution path, that the agent's own reference trajectory reaches every scenario's
expected world, and that the binding does not reach it for the agent: doing nothing, skipping a
state-changing call, or sending nonsense in one must each fail the scenario's checks, unless the
same change made to the environment's own calls holds too (the profile does not need that call:
the variant is `exempt`).

    python -m fi.alk.harness.binding_proof --environment DIR --suite DIR --binding FILE \
        --trajectories FILE --postgres-dsn DSN --out FILE

`--binding` is a sealed binding document (`seal_agent_binding`): the proof names its digest, and
the environment it validates against is the one given. The command writes the proof and exits 0
when it passed, 3 when it did not, and 1 on invalid input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from .bindings import AgentTool, MappedToolBinding
from .releases import (
    AgentBindingV1,
    BindingMode,
    DigestRef,
    PostgresProof,
    ReleaseError,
    _check_digest,
    check_tool_routed,
    load_agent_binding,
    load_environment_release,
    load_suite_release,
)

BINDING_PROOF_SCHEMA = "futureagi.binding-proof.v1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrajectoryCall(_Strict):
    tool: str
    arguments: JsonValue = Field(default_factory=dict)


_TRAJECTORIES = TypeAdapter(dict[str, list[TrajectoryCall]])


class ScenarioBindingProof(_Strict):
    status: Literal["proved", "failed", "unscorable"]
    reason: str = ""
    adversarial: dict[str, Literal["failed", "held", "exempt"]] = Field(
        default_factory=dict
    )


class BindingProofV1(_Strict):
    schema_version: Literal["futureagi.binding-proof.v1"] = BINDING_PROOF_SCHEMA
    binding: DigestRef
    suite: DigestRef
    environment: DigestRef
    scenarios: dict[str, ScenarioBindingProof]
    passed: bool


def perturb(value: Any, index: int) -> Any:
    """Every string leaf replaced, numbers increased by 7, booleans negated."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return f"zz-perturbed-{index}"
    if isinstance(value, int | float):
        return value + 7
    if isinstance(value, Mapping):
        return {key: perturb(item, index) for key, item in value.items()}
    if isinstance(value, list):
        return [perturb(item, index) for item in value]
    return value


Counterpart = tuple[AgentBindingV1, list[tuple[str, Any]]]
Variant = tuple[str, list[tuple[str, Any]], Counterpart | None]


def _replay(
    binding: AgentBindingV1,
    outcomes: Sequence[Any],
    position: int,
    perturbed: int | None,
) -> Counterpart:
    """The proved trajectory's committed agent calls, replayed in the environment's own
    vocabulary, with call `position` left out (`perturbed` None) or its operations' arguments
    perturbed.

    Each committed agent call becomes one replay call whose steps are its recorded environment
    operations with constant arguments, so it runs through `execute_agent_call` in one
    transaction: a refused later step rolls the earlier ones back exactly as it did for the agent.
    """
    agent_tools: list[AgentTool] = []
    entries: list[MappedToolBinding] = []
    calls: list[tuple[str, Any]] = []
    for at, outcome in enumerate(outcomes):
        if outcome.status != "ok" or (at == position and perturbed is None):
            continue
        name = f"call_{at + 1}"
        steps = [
            {
                "id": f"op_{number}",
                "tool": operation.tool,
                "arguments": {
                    key: {"const": value}
                    for key, value in (
                        perturb(operation.arguments, perturbed)
                        if at == position
                        else operation.arguments
                    ).items()
                },
            }
            for number, operation in enumerate(outcome.operations, start=1)
        ]
        agent_tools.append(
            AgentTool(
                name=name,
                description="",
                parameters={"type": "object"},
                admission="routed",
            )
        )
        entries.append(
            MappedToolBinding.model_validate({"agent_tool": name, "steps": steps})
        )
        calls.append((name, {}))
    replay = binding.model_copy(
        update={
            "agent_tools": agent_tools,
            "tools": entries,
            "fixtures": {},
            "aliases": {},
        }
    )
    return replay, calls


def _variants(
    binding: AgentBindingV1,
    calls: Sequence[tuple[str, Any]],
    outcomes: Sequence[Any],
    mutates: set[str],
) -> list[Variant]:
    """`nothing`, then `skip:i` and `perturb:i` (1-based) for every call that can change state,
    each with its canonical counterpart: the same change made to the environment operations the
    proved trajectory committed (none for `nothing`; the suite proof already shows an untouched
    world fails).

    A call the reference trajectory itself saw refused changed nothing, so removing or garbling
    it cannot fail the scenario whatever the binding does; it gets no variants.
    """
    variants: list[Variant] = [("nothing", [], None)]
    for position, (tool, arguments) in enumerate(calls):
        if tool not in mutates or outcomes[position].status != "ok":
            continue
        index = position + 1
        variants.append(
            (
                f"skip:{index}",
                [*calls[:position], *calls[index:]],
                _replay(binding, outcomes, position, None),
            )
        )
        variants.append(
            (
                f"perturb:{index}",
                [*calls[:position], (tool, perturb(arguments, index)), *calls[index:]],
                _replay(binding, outcomes, position, index),
            )
        )
    return variants


def _why(attempt: Any) -> str:
    if not attempt.ready.ok:
        return attempt.ready.said
    return "; ".join([*attempt.faults, *attempt.failed]) or "no deterministic check"


def prove_binding(
    environment_root: str | Path,
    suite_root: str | Path,
    binding: AgentBindingV1,
    trajectories: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    postgres_dsn: str,
) -> BindingProofV1:
    """Prove a sealed `tool_routed` binding against a sealed suite of the environment it targets.

    Raises `ReleaseError` only for inputs that cannot be proved at all; a scenario that fails, or
    an adversarial variant that holds, is reported and makes `passed` false.
    """
    from .folder import read_folder

    environment_root, suite_root = Path(environment_root), Path(suite_root)
    environment = load_environment_release(environment_root)
    suite = load_suite_release(suite_root)
    problems = []
    if binding.mode is not BindingMode.TOOL_ROUTED:
        problems.append(f"a {binding.mode.value} binding has nothing to prove")
    if binding.environment.digest != environment.digest:
        problems.append("the binding targets another environment")
    if suite.environment.digest != environment.digest:
        problems.append("the suite was proved against another environment")
    try:
        parsed = _TRAJECTORIES.validate_python(dict(trajectories))
    except ValidationError as exc:
        problems.append(f"trajectories: {exc}")
        parsed = {}
    keys = {entry.key for entry in suite.scenarios}
    if unknown := sorted(set(parsed) - keys):
        problems.append(
            f"trajectories for scenarios not in the suite: {', '.join(unknown)}"
        )
    if problems:
        raise ReleaseError("binding_proof_invalid", "; ".join(problems))
    _check_digest(binding, "binding_digest_mismatch")

    scenarios: dict[str, ScenarioBindingProof] = {}
    with PostgresProof(
        suite_root, environment_root, suite.policy_profile, postgres_dsn
    ) as proof:
        check_tool_routed(binding, proof.environment_tools)
        changing = {tool.name for tool in proof.environment_tools if tool.mutates_state}
        mapped = [
            entry for entry in binding.tools if isinstance(entry, MappedToolBinding)
        ]
        reached = {step.tool for entry in mapped for step in entry.steps}
        mutates = {
            entry.agent_tool
            for entry in mapped
            if any(step.tool in changing for step in entry.steps)
        }
        for entry in suite.scenarios:
            scenario = read_folder(suite_root, Path(entry.path).name)
            unreached = sorted({step.tool for step in scenario.solution} - reached)
            if unreached:
                scenarios[entry.key] = ScenarioBindingProof(
                    status="unscorable",
                    reason=f"the reference solution uses {', '.join(unreached)}, "
                    "which no mapped step reaches",
                )
                continue
            if entry.key not in parsed:
                scenarios[entry.key] = ScenarioBindingProof(
                    status="failed", reason="no trajectory"
                )
                continue
            calls = [(call.tool, call.arguments) for call in parsed[entry.key]]
            solved = proof.attempt(scenario, calls, binding)
            if not solved.holds:
                scenarios[entry.key] = ScenarioBindingProof(
                    status="failed", reason=_why(solved)
                )
                continue
            adversarial: dict[str, Any] = {}
            for name, variant, counterpart in _variants(
                binding, calls, solved.outcomes, mutates
            ):
                if not proof.attempt(scenario, variant, binding).checks_hold:
                    adversarial[name] = "failed"
                # A variant that holds shows the binding helps only if the same change, made in
                # the environment's own vocabulary, would fail; otherwise the policy profile
                # simply does not need the call.
                elif (
                    counterpart is not None
                    and proof.attempt(
                        scenario, counterpart[1], counterpart[0]
                    ).checks_hold
                ):
                    adversarial[name] = "exempt"
                else:
                    adversarial[name] = "held"
            held = [name for name, outcome in adversarial.items() if outcome == "held"]
            exempt = [
                name for name, outcome in adversarial.items() if outcome == "exempt"
            ]
            # A held variant means the binding, not the agent, reached the expected world; the
            # scenario then measures nothing about the agent and is not used for runs.
            scenarios[entry.key] = ScenarioBindingProof(
                status="failed" if held else "proved",
                reason=f"adversarial variants held: {', '.join(held)}"
                if held
                else f"exempt: {', '.join(exempt)} (the call is not needed under this "
                "profile: the canonical calls hold without it)"
                if exempt
                else "",
                adversarial=adversarial,
            )
    return BindingProofV1(
        binding=DigestRef(digest=binding.digest),
        suite=DigestRef(digest=suite.digest),
        environment=DigestRef(digest=environment.digest),
        scenarios=scenarios,
        passed=all(item.status != "failed" for item in scenarios.values())
        and not any(
            outcome == "held"
            for item in scenarios.values()
            for outcome in item.adversarial.values()
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fi.alk.harness.binding_proof")
    parser.add_argument("--environment", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--binding", required=True, help="a sealed binding.json")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--postgres-dsn", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        try:
            trajectories = json.loads(
                Path(args.trajectories).read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ReleaseError("binding_proof_invalid", f"trajectories: {exc}") from exc
        if not isinstance(trajectories, dict):
            raise ReleaseError(
                "binding_proof_invalid", "trajectories must be a JSON object"
            )
        result = prove_binding(
            args.environment,
            args.suite,
            load_agent_binding(args.binding),
            trajectories,
            postgres_dsn=args.postgres_dsn,
        )
    except ReleaseError as exc:
        print(json.dumps({"error": exc.code, "detail": exc.detail}), file=sys.stderr)
        return 1
    Path(args.out).write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": result.passed,
                "scenarios": {
                    key: item.status for key, item in result.scenarios.items()
                },
            }
        )
    )
    return 0 if result.passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
