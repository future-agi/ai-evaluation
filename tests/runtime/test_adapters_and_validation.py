"""Gap-close tests: adapter enums, simulator/world_kind validation, and
config-declared tool mocking (tool mocking as a world capability, any loop)."""

from __future__ import annotations

import asyncio

import pytest

from fi.simulate.adapters import (
    EnvironmentAdapters,
    SimulatorAdapters,
    TargetAdapters,
    WorldKinds,
)
from fi.simulate.agent.wrapper import AgentInput, AgentResponse
from fi.simulate.registry import AdapterNotFound
from fi.simulate.runtime.planner import UnsupportedWorldKind, build_plan
from fi.simulate.runtime.runner import SimulationRunner
from fi.simulate.runtime.spec import (
    AgentEndpointSpec,
    EnvironmentSpec,
    SimulationSpec,
    SimulatorPolicySpec,
)
from fi.simulate.simulation.models import Persona, Scenario


def _spec(*, sim=SimulatorAdapters.SYNTHETIC_USER, world_kind=WorldKinds.CONVERSATION,
          env=EnvironmentAdapters.CHAT, config=None):
    # Enums at the consumer surface; callers pass raw strings only to exercise
    # the invalid-name paths (typos aren't enum members by definition).
    return SimulationSpec(
        run_id="r",
        environment=EnvironmentSpec(adapter=env, world_kind=world_kind, config=config or {}),
        target=AgentEndpointSpec(adapter=TargetAdapters.FACTORY),
        simulator=SimulatorPolicySpec(adapter=sim),
        scenario=Scenario(name="s", dataset=[
            Persona(persona={"name": "x"}, situation="y", outcome="z")]),
    )


def test_enum_members_are_the_strings():
    assert EnvironmentAdapters.CHAT == "chat"
    assert TargetAdapters.VAPI_WEBSOCKET == "vapi_websocket"
    assert SimulatorAdapters.SYNTHETIC_USER == "synthetic_user"
    assert WorldKinds.CONVERSATION == "conversation"
    assert WorldKinds.TOOL_API == "tool_api"
    assert isinstance(TargetAdapters.FACTORY, str)


def test_worldkinds_mirror_contract():
    # GAP A: the runtime WorldKinds enum is a faithful mirror of the frozen
    # canon (contract.SIMULATION_WORLD_KINDS). Byte-compare so any drift on
    # either side fails here instead of silently forking the vocabulary.
    from fi.simulate.simulation.contract import SIMULATION_WORLD_KINDS

    assert tuple(wk.value for wk in WorldKinds) == SIMULATION_WORLD_KINDS


def test_enum_spec_hash_equals_bare_string_spec_hash():
    with_enum = SimulationSpec(
        run_id="r",
        environment=EnvironmentSpec(adapter=EnvironmentAdapters.CHAT,
                                    world_kind=WorldKinds.CONVERSATION, config={"max_turns": 2}),
        target=AgentEndpointSpec(adapter=TargetAdapters.FACTORY),
        simulator=SimulatorPolicySpec(adapter=SimulatorAdapters.SYNTHETIC_USER),
        scenario=Scenario(name="s", dataset=[Persona(persona={"name": "x"}, situation="y", outcome="z")]),
    )
    with_str = _spec(config={"max_turns": 2})
    assert with_enum.spec_hash == with_str.spec_hash
    assert with_enum.model_dump()["target"]["adapter"] == "factory"


def test_unknown_simulator_adapter_rejected_at_plan():
    with pytest.raises(AdapterNotFound):
        build_plan(_spec(sim="syntetic_user"))


@pytest.mark.parametrize("wk", [WorldKinds.CONVERSATION, WorldKinds.TOOL_API])
def test_known_world_kinds_pass(wk):
    # Both EXECUTABLE_WORLD_KINDS_V1 kinds are admitted by the chat plugin: the
    # tool surface runs on the same text loop (tools = capability, not engine).
    build_plan(_spec(world_kind=wk))  # no raise


def test_unknown_world_kind_rejected():
    with pytest.raises(UnsupportedWorldKind):
        build_plan(_spec(world_kind="sql"))


def test_config_declared_mock_tools_execute_in_any_loop():
    class ToolAgent:
        async def call(self, ai: AgentInput) -> AgentResponse:
            if ai.turn_index == 0:
                return AgentResponse(content="Approving.",
                                     tool_calls=[{"id": "c0", "name": "approve_refund", "arguments": {}}])
            return AgentResponse(content="Done.")

    spec = SimulationSpec(
        run_id="mockcfg",
        environment=EnvironmentSpec(adapter=EnvironmentAdapters.CHAT,
                                    world_kind=WorldKinds.CONVERSATION, config={
            "max_turns": 2, "min_turns": 1,
            "mock_tools": {"approve_refund": {"content": "refund approved",
                                              "state_updates": {"refund": {"status": "approved"}}}}}),
        target=AgentEndpointSpec(adapter=TargetAdapters.CALLABLE),
        simulator=SimulatorPolicySpec(adapter=SimulatorAdapters.SYNTHETIC_USER),
        scenario=Scenario(name="s", dataset=[
            Persona(persona={"name": "Sam"}, situation="refund pls", outcome="refund approved")]),
    )
    report = asyncio.run(SimulationRunner().run(spec, target=ToolAgent()))
    assert report.status.value == "completed"
    assert "refund approved" in report.test_cases[0].result.transcript
