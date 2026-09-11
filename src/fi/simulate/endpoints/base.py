"""AgentEndpoint Protocol + shared data types (plan §4.1)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, Field, JsonValue

from fi.simulate.realtime.events import RealtimeEvent
from fi.simulate.realtime.media import AudioFrame
from fi.simulate.runtime.capabilities import EndpointCapabilities


class AgentEndpointManifest(BaseModel):
    """Static declaration of an endpoint adapter's identity + shape.

    The planner records the manifest on the plan so hosted runs can
    reconstruct which adapter (name + version) executed a case.
    """

    name: str
    version: str = "1"
    provider: str
    world_kinds: list[str] = Field(default_factory=lambda: ["voice"])
    capabilities: EndpointCapabilities = Field(default_factory=EndpointCapabilities)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class DiscoveryRequest(BaseModel):
    run_id: str
    test_case_id: str
    required_capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class DiscoverySnapshot(BaseModel):
    capabilities: EndpointCapabilities
    supported: bool = True
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EndpointHandle(BaseModel):
    """Opaque adapter handle returned by ``prepare`` and reused for the case."""

    handle_id: str
    endpoint_name: str
    created_at: datetime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ReadinessResult(BaseModel):
    ready: bool
    latency_ms: float | None = None
    reason: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ReconciliationResult(BaseModel):
    reconciled: bool
    orphan_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AgentEndpoint(Protocol):
    """Session-oriented target-agent adapter contract (plan §4.1).

    Turn-based agents can still use the legacy ``AgentWrapper.call``
    surface; this Protocol is the target for realtime/voice/session
    agents where the engine needs prepare/wait_ready/stop lifecycle.
    """

    manifest: AgentEndpointManifest
    capabilities: EndpointCapabilities

    async def discover(self, request: DiscoveryRequest) -> DiscoverySnapshot: ...

    async def prepare(self, plan) -> EndpointHandle: ...  # SimulationPlan

    async def wait_ready(self, handle: EndpointHandle) -> ReadinessResult: ...

    async def send(
        self, handle: EndpointHandle, event: RealtimeEvent | AudioFrame
    ) -> None: ...

    async def receive(
        self, handle: EndpointHandle
    ) -> AsyncIterator[RealtimeEvent | AudioFrame]: ...

    async def stop(self, handle: EndpointHandle) -> None: ...

    async def cleanup(self, handle: EndpointHandle) -> None: ...

    async def reconcile(self, handle: EndpointHandle) -> ReconciliationResult: ...


__all__ = [
    "AgentEndpoint",
    "AgentEndpointManifest",
    "DiscoveryRequest",
    "DiscoverySnapshot",
    "EndpointHandle",
    "ReadinessResult",
    "ReconciliationResult",
]
