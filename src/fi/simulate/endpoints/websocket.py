"""WebSocket target-agent adapter — capability declaration + Stage-6 seam."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

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


class WebSocketAgentEndpoint:
    """Points at a WebSocket agent surface. Wire-up lands in Stage 6."""

    def __init__(self, *, name: str, url: str) -> None:
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("websocket_url_invalid: must use ws:// or wss://")
        self.manifest = AgentEndpointManifest(
            name=name,
            provider="websocket",
            world_kinds=["chat", "voice"],
            capabilities=EndpointCapabilities(text=True, streaming=True, tool_events=True),
            metadata={"url": url},
        )
        self.capabilities = self.manifest.capabilities

    async def discover(self, request: DiscoveryRequest) -> DiscoverySnapshot:
        del request
        return DiscoverySnapshot(capabilities=self.capabilities)

    async def prepare(self, plan) -> EndpointHandle:  # noqa: ANN001
        return EndpointHandle(
            handle_id=f"ws-{uuid.uuid4().hex[:12]}",
            endpoint_name=self.manifest.name,
            created_at=datetime.now(timezone.utc),
            metadata={"plan_id": getattr(plan, "plan_id", None)},
        )

    async def wait_ready(self, handle: EndpointHandle) -> ReadinessResult:
        del handle
        raise NotImplementedError("WebSocketAgentEndpoint readiness lands in Stage 6")

    async def send(
        self, handle: EndpointHandle, event: RealtimeEvent | AudioFrame
    ) -> None:
        raise NotImplementedError("WebSocketAgentEndpoint send lands in Stage 6")

    async def receive(
        self, handle: EndpointHandle
    ) -> AsyncIterator[RealtimeEvent | AudioFrame]:
        raise NotImplementedError("WebSocketAgentEndpoint receive lands in Stage 6")
        yield  # type: ignore[unreachable]

    async def stop(self, handle: EndpointHandle) -> None:
        del handle

    async def cleanup(self, handle: EndpointHandle) -> None:
        del handle

    async def reconcile(self, handle: EndpointHandle) -> ReconciliationResult:
        del handle
        return ReconciliationResult(reconciled=True)


__all__ = ["WebSocketAgentEndpoint"]
