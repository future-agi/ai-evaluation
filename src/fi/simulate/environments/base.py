"""Environment plugin contract (canonical plan §3).

An environment owns one *world*: it drives the simulator against the target,
emits a legacy ``TestReport`` (which ``SimulationRunner`` converts to a
``SimulationReport``), and declares its action/observation surface through a
manifest. ``SimulationRunner`` stays world-agnostic — it looks the plugin up in
the ``environment_registry`` and never references chat- or voice-specific fields.

The manifest mirrors ``endpoints.base.AgentEndpointManifest`` so the planner can
negotiate capabilities the same way for environments and target endpoints.
"""

from __future__ import annotations

from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, Field, JsonValue

from fi.simulate.agent.wrapper import AgentWrapper, SimulationArtifact, SimulationEvent
from fi.simulate.environment import EnvironmentAdapter
from fi.simulate.runtime.capabilities import EndpointCapabilities
from fi.simulate.runtime.spec import SimulationSpec
from fi.simulate.simulation.models import Persona, TestReport


class EnvironmentManifest(BaseModel):
    """Static declaration of an environment plugin's identity + shape."""

    name: str
    version: str = "1"
    world_kinds: list[str] = Field(default_factory=list)
    capabilities: EndpointCapabilities = Field(default_factory=EndpointCapabilities)
    isolation: str = "shared_runner_process"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


@runtime_checkable
class EnvironmentPlugin(Protocol):
    """Session/episode owner for one world kind.

    ``run`` returns a legacy ``TestReport``; ``SimulationRunner`` owns planning,
    canonical events, timeout, failure classification, and the conversion to
    ``SimulationReport`` — identically for every environment.
    """

    manifest: EnvironmentManifest

    async def run(
        self,
        spec: SimulationSpec,
        *,
        target: Callable[..., Any] | AgentWrapper | Any,
        artifacts: Optional[List[SimulationArtifact | dict[str, Any]]] = None,
        events: Optional[List[SimulationEvent | dict[str, Any]]] = None,
        environment: Optional[EnvironmentAdapter | Iterable[EnvironmentAdapter]] = None,
        auto_execute_tools: bool = True,
        stop_when: Optional[Callable[[list[dict[str, Any]], Persona], bool]] = None,
        agent_wrapper_kwargs: Optional[dict[str, Any]] = None,
        on_case_complete: Optional[Callable[[int, Any], Awaitable[None]]] = None,
        on_case_start: Optional[Callable[[int], Awaitable[None]]] = None,
    ) -> TestReport: ...


__all__ = ["EnvironmentManifest", "EnvironmentPlugin"]
