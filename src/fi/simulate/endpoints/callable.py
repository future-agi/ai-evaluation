"""Callable target-agent adapter for turn-based chat runs."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any

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


class CallableAgentEndpoint:
    """Wraps a plain callable/coroutine target agent."""

    def __init__(
        self,
        agent_callable: Callable[..., Any],
        *,
        name: str = "callable-agent",
    ) -> None:
        self._callable = agent_callable
        self.manifest = AgentEndpointManifest(
            name=name,
            provider="callable",
            world_kinds=["chat"],
            capabilities=EndpointCapabilities(text=True, streaming=False),
        )
        self.capabilities = self.manifest.capabilities

    async def discover(self, request: DiscoveryRequest) -> DiscoverySnapshot:
        missing = [
            cap
            for cap in request.required_capabilities
            if cap not in self.capabilities.supported()
        ]
        return DiscoverySnapshot(
            capabilities=self.capabilities,
            supported=not missing,
            reasons=[f"unsupported:{cap}" for cap in missing],
        )

    async def prepare(self, plan) -> EndpointHandle:  # noqa: ANN001
        return EndpointHandle(
            handle_id=f"cb-{uuid.uuid4().hex[:12]}",
            endpoint_name=self.manifest.name,
            created_at=datetime.now(timezone.utc),
            metadata={"plan_id": getattr(plan, "plan_id", None)},
        )

    async def wait_ready(self, handle: EndpointHandle) -> ReadinessResult:
        del handle
        return ReadinessResult(ready=True)

    async def send(
        self, handle: EndpointHandle, event: RealtimeEvent | AudioFrame
    ) -> None:
        raise NotImplementedError("CallableAgentEndpoint is turn-based; use invoke()")

    async def receive(
        self, handle: EndpointHandle
    ) -> AsyncIterator[RealtimeEvent | AudioFrame]:
        raise NotImplementedError("CallableAgentEndpoint is turn-based; use invoke()")
        # unreachable, kept for Protocol conformance
        yield  # type: ignore[unreachable]

    async def stop(self, handle: EndpointHandle) -> None:
        del handle

    async def cleanup(self, handle: EndpointHandle) -> None:
        del handle

    async def reconcile(self, handle: EndpointHandle) -> ReconciliationResult:
        del handle
        return ReconciliationResult(reconciled=True)

    async def invoke(self, *args: Any, **kwargs: Any) -> Any:
        result = self._callable(*args, **kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result


__all__ = ["CallableAgentEndpoint"]
