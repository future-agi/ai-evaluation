"""HTTP target-agent adapter — capability declaration + Stage-6 seam."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from pydantic import AnyHttpUrl

from fi.simulate.realtime.events import RealtimeEvent
from fi.simulate.realtime.media import AudioFrame
from fi.simulate.runtime.capabilities import EndpointCapabilities

from .base import (
    AgentEndpointManifest,
    DiscoveryRequest,
    DiscoverySnapshot,
    EndpointHandle,
    ReadinessResult,
    ReconciliationResult,
)


class HttpAgentEndpoint:
    """Points at an HTTP agent surface. Wire-up lands in Stage 6."""

    def __init__(self, *, name: str, url: AnyHttpUrl | str) -> None:
        self.manifest = AgentEndpointManifest(
            name=name,
            provider="http",
            world_kinds=["chat"],
            capabilities=EndpointCapabilities(text=True, tool_events=True),
            metadata={"url": str(url)},
        )
        self.capabilities = self.manifest.capabilities

    async def discover(self, request: DiscoveryRequest) -> DiscoverySnapshot:
        del request
        return DiscoverySnapshot(capabilities=self.capabilities)

    async def prepare(self, plan) -> EndpointHandle:  # noqa: ANN001
        return EndpointHandle(
            handle_id=f"http-{uuid.uuid4().hex[:12]}",
            endpoint_name=self.manifest.name,
            created_at=datetime.now(timezone.utc),
            metadata={"plan_id": getattr(plan, "plan_id", None)},
        )

    async def wait_ready(self, handle: EndpointHandle) -> ReadinessResult:
        del handle
        raise NotImplementedError("HttpAgentEndpoint readiness lands in Stage 6")

    async def send(
        self, handle: EndpointHandle, event: RealtimeEvent | AudioFrame
    ) -> None:
        raise NotImplementedError("HttpAgentEndpoint send lands in Stage 6")

    async def receive(
        self, handle: EndpointHandle
    ) -> AsyncIterator[RealtimeEvent | AudioFrame]:
        raise NotImplementedError("HttpAgentEndpoint receive lands in Stage 6")
        yield  # type: ignore[unreachable]

    async def stop(self, handle: EndpointHandle) -> None:
        del handle

    async def cleanup(self, handle: EndpointHandle) -> None:
        del handle

    async def reconcile(self, handle: EndpointHandle) -> ReconciliationResult:
        del handle
        return ReconciliationResult(reconciled=True)


__all__ = ["HttpAgentEndpoint"]
