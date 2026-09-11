"""SimulatorPolicy contract (plan §4.2).

The concrete simulator lives in ``fi.simulate.simulation.voice_prompt``
and the LiveKit-hosted worker (``livekit-infra/.../simulator_agent.py``).
This module publishes the Protocol so alternate policies (script-only,
adversarial, learned) can plug in without reaching into the LiveKit
engine.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field, JsonValue

from fi.simulate.simulation.models import Persona


class PolicyContext(BaseModel):
    run_id: str
    test_case_id: str
    persona: Persona
    call_type: str = "inbound"
    agent_name: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PolicyState(BaseModel):
    session_id: str | None = None
    turn_index: int = 0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PolicySummary(BaseModel):
    turns: int = 0
    ended_naturally: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SimulatorPolicy(Protocol):
    async def initialize(self, context: PolicyContext) -> PolicyState: ...

    async def next_action(self, state: PolicyState, observation: Any) -> Any: ...

    async def on_event(self, state: PolicyState, event: Any) -> None: ...

    async def finalize(self, state: PolicyState) -> PolicySummary: ...


__all__ = [
    "PolicyContext",
    "PolicyState",
    "PolicySummary",
    "SimulatorPolicy",
]
