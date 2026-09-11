"""LiveKit target-agent adapter.

Thin adapter around the existing ``LiveKitEngine`` that lets hosted +
matrix runners see a LiveKit target through the ``AgentEndpoint``
contract. The actual media/lifecycle work still lives inside the
engine's per-case runner; this class exposes a stable seam so a full
``endpoints/livekit.py`` rewrite can land later without breaking
callers.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from fi.simulate.agent.definition import AgentDefinition, TelephonyTransport
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


def _capabilities_for(agent_definition: AgentDefinition) -> EndpointCapabilities:
    transport = agent_definition.transport or TelephonyTransport()
    kind = transport.kind
    return EndpointCapabilities(
        audio=True,
        text=True,
        streaming=True,
        interruption=True,
        dtmf=kind in {"sip_inbound", "sip_outbound"},
        transfer=kind in {"sip_inbound", "sip_outbound"},
        transcript_events=True,
        tool_events=False,
        usage_events=True,
        internal_metrics=True,
        recording=True,
        web_rtc=kind == "webrtc",
        sip=kind in {"sip_inbound", "sip_outbound"},
    )


class LiveKitAgentEndpoint:
    """Spec-level LiveKit endpoint. Engine still owns the media path."""

    def __init__(self, agent_definition: AgentDefinition, *, name: str | None = None) -> None:
        self._agent_definition = agent_definition
        self.manifest = AgentEndpointManifest(
            name=name or agent_definition.name,
            provider="livekit",
            world_kinds=["voice"],
            capabilities=_capabilities_for(agent_definition),
            metadata={
                "url": str(agent_definition.url),
                "room_name_template": agent_definition.room_name,
                "room_mode": agent_definition.room_mode,
                "transport": (
                    agent_definition.transport.kind
                    if agent_definition.transport
                    else "webrtc"
                ),
            },
        )
        self.capabilities = self.manifest.capabilities

    @property
    def agent_definition(self) -> AgentDefinition:
        return self._agent_definition

    async def discover(self, request: DiscoveryRequest) -> DiscoverySnapshot:
        supported = self.capabilities.supported()
        missing = [
            cap for cap in request.required_capabilities if cap not in supported
        ]
        return DiscoverySnapshot(
            capabilities=self.capabilities,
            supported=not missing,
            reasons=[f"unsupported:{cap}" for cap in missing],
        )

    async def prepare(self, plan) -> EndpointHandle:  # noqa: ANN001
        return EndpointHandle(
            handle_id=f"lk-{uuid.uuid4().hex[:12]}",
            endpoint_name=self.manifest.name,
            created_at=datetime.now(timezone.utc),
            metadata={
                "plan_id": getattr(plan, "plan_id", None),
                "transport": (
                    self._agent_definition.transport.kind
                    if self._agent_definition.transport
                    else "webrtc"
                ),
            },
        )

    async def wait_ready(self, handle: EndpointHandle) -> ReadinessResult:
        del handle
        # Real readiness is driven inside LiveKitEngine per case today.
        # This adapter reports ready=True optimistically; hosted callers
        # observe actual readiness through emitted CanonicalEvents.
        return ReadinessResult(ready=True)

    async def send(
        self, handle: EndpointHandle, event: RealtimeEvent | AudioFrame
    ) -> None:
        raise NotImplementedError(
            "LiveKitAgentEndpoint.send belongs to the future realtime router"
        )

    async def receive(
        self, handle: EndpointHandle
    ) -> AsyncIterator[RealtimeEvent | AudioFrame]:
        raise NotImplementedError(
            "LiveKitAgentEndpoint.receive belongs to the future realtime router"
        )
        yield  # type: ignore[unreachable]

    async def stop(self, handle: EndpointHandle) -> None:
        del handle

    async def cleanup(self, handle: EndpointHandle) -> None:
        del handle

    async def reconcile(self, handle: EndpointHandle) -> ReconciliationResult:
        del handle
        return ReconciliationResult(reconciled=True)


__all__ = ["LiveKitAgentEndpoint"]
