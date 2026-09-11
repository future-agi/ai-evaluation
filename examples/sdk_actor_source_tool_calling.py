"""Drop in a tool-calling agent via the ActorSource + Environment abstractions.

Offline, deterministic, no credentials. Showcases the refactored gym model end
to end:

* an ``EnvironmentAdapter`` world (``RefundWorld``) that declares an action space
  (``lookup_order`` / ``approve_refund``) and owns state,
* a plain tool-calling agent class dropped in through the ``factory``
  **ActorSource** (``target``/``factory`` — the same vocabulary a manifest
  ``agent:`` block uses), resolved via the one endpoint registry,
* driven through ``SimulationRunner`` — the same spine chat and voice use.

The agent calls ``approve_refund``; the world executes it and moves to
``status: approved``; we assert the tool actually drove the world.

Run:  python examples/sdk_actor_source_tool_calling.py artifacts/actor-tool.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from fi.simulate.agent.wrapper import AgentInput, AgentResponse
from fi.simulate.endpoints.profiles import get_profile
from fi.simulate.environment import EnvironmentAdapter, EnvironmentSnapshot, ToolExecutionResult
from fi.simulate.runtime import (
    AgentEndpointSpec,
    EnvironmentSpec,
    RunStatus,
    SimulationSpec,
    SimulatorPolicySpec,
)
from fi.simulate.runtime.runner import SimulationRunner
from fi.simulate.simulation.models import Persona, Scenario

_TOOL_SCHEMAS = [
    {"name": "lookup_order", "description": "Look up an order by id."},
    {"name": "approve_refund", "description": "Approve a refund for an order."},
]


class RefundWorld(EnvironmentAdapter):
    """A tiny executable world: two tools + refund state."""

    name = "refund_world"

    def __init__(self) -> None:
        self.state: dict[str, Any] = {"refund": {"status": "pending"}}

    def reset(self, **_context: Any) -> EnvironmentSnapshot:
        self.state = {"refund": {"status": "pending"}}
        return EnvironmentSnapshot(tools=list(_TOOL_SCHEMAS), state=dict(self.state))

    def handle_tool_call(
        self, tool_call: Mapping[str, Any], **_context: Any
    ) -> Optional[ToolExecutionResult]:
        name = tool_call.get("name") or (tool_call.get("function") or {}).get("name")
        call_id = tool_call.get("id") or tool_call.get("tool_call_id")
        if name == "lookup_order":
            return ToolExecutionResult(
                tool_call_id=call_id, tool_name=name,
                content="order A1: eligible for refund", result={"eligible": True},
            )
        if name == "approve_refund":
            self.state["refund"]["status"] = "approved"
            return ToolExecutionResult(
                tool_call_id=call_id, tool_name=name,
                content="refund approved", result={"status": "approved"},
                state_updates={"refund": {"status": "approved"}},
            )
        return None


class ToolCallingRefundAgent:
    """Scripted target agent: looks up the order, then approves the refund.

    Dropped in via the ``factory`` ActorSource — the harness sets up the
    environment around it; the agent just acts in the action space.
    """

    async def call(self, agent_input: AgentInput) -> AgentResponse:
        turn = agent_input.turn_index
        if turn == 0:
            return AgentResponse(
                content="Let me look up your order.",
                tool_calls=[{"id": "c0", "name": "lookup_order",
                             "arguments": {"order_id": "A1"}}],
            )
        if turn == 1:
            return AgentResponse(
                content="It's eligible — approving the refund now.",
                tool_calls=[{"id": "c1", "name": "approve_refund",
                             "arguments": {"order_id": "A1"}}],
            )
        return AgentResponse(content="Your refund is approved. Anything else?")


def run(output_path: str | os.PathLike[str]) -> dict[str, Any]:
    # Make this module importable by "module:attr" so the ActorSource factory can
    # resolve the agent the same way a real job would.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    module_stem = Path(__file__).stem

    # Drop the agent in through the factory ActorSource (local resolution).
    target = get_profile("factory").resolve_target(
        {"target": f"{module_stem}:ToolCallingRefundAgent", "factory": True},
        hosted=False,
    )

    spec = SimulationSpec(
        run_id="run_actor_tool_calling",
        environment=EnvironmentSpec(
            adapter="chat", world_kind="conversation",
            config={"max_turns": 3, "min_turns": 1},
        ),
        target=AgentEndpointSpec(adapter="factory"),
        simulator=SimulatorPolicySpec(adapter="synthetic_user"),
        scenario=Scenario(
            name="refund",
            dataset=[Persona(
                persona={"name": "Sam"},
                situation="My order A1 arrived damaged.",
                outcome="the refund is approved",
            )],
        ),
    )

    world = RefundWorld()
    report = asyncio.run(SimulationRunner().run(spec, target=target, environment=world))

    tool_ran = world.state["refund"]["status"] == "approved"
    data = {
        "kind": "agent-learning.actor-source-example.v1",
        "status": "passed" if (report.status == RunStatus.COMPLETED and tool_ran) else "failed",
        "run_status": report.status.value,
        "world_final_state": world.state,
        "tool_drove_world": tool_ran,
        "transcript": report.test_cases[0].result.transcript if report.test_cases else "",
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/actor-tool.json"
    result = run(target_path)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "passed" else 1)
