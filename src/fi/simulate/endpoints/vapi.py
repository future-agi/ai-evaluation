"""Vapi target-agent adapter — capability declaration + Stage 6/8 seam.

The phone leg to a Vapi assistant runs through ``LiveKitAgentEndpoint``
with ``TelephonyTransport(kind="sip_outbound")`` today; this class
exists so a future direct-Vapi execution path (or hosted-runner
selection matrix) can register an adapter with the same shape as the
LiveKit and Retell ones. Post-call evidence still flows through
``fi.simulate.evidence.providers.vapi.VapiEvidenceSource``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

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


class VapiOriginatorConfigError(ValueError):
    pass


@dataclass(frozen=True)
class VapiCall:
    call_id: str
    status: str | None


class VapiCallOriginator:
    """Create an opt-in Vapi call to the LiveKit inbound DID."""

    _base_url = "https://api.vapi.ai"

    def __init__(
        self,
        *,
        api_key: str,
        assistant_id: str,
        phone_number_id: str,
        destination: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._assistant_id = assistant_id
        self._phone_number_id = phone_number_id
        self._destination = destination
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.AsyncClient(
            base_url=os.environ.get("VAPI_API_BASE_URL", self._base_url),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._owns_client = client is None

    @classmethod
    def from_env(cls) -> "VapiCallOriginator":
        names = (
            "VAPI_API_KEY",
            "VAPI_ASSISTANT_ID",
            "VAPI_PHONE_NUMBER_ID",
            "LIVEKIT_INBOUND_DID",
        )
        values = {name: os.environ.get(name, "").strip() for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(
                "vapi_originator_config_missing: " + ", ".join(sorted(missing))
            )
        return cls(
            api_key=values["VAPI_API_KEY"],
            assistant_id=values["VAPI_ASSISTANT_ID"],
            phone_number_id=values["VAPI_PHONE_NUMBER_ID"],
            destination=values["LIVEKIT_INBOUND_DID"],
        )

    async def start(self) -> VapiCall:
        phone_response = await self._client.get(
            f"/phone-number/{self._phone_number_id}",
            headers=self._headers,
        )
        phone_response.raise_for_status()
        response = await self._client.post(
            "/call",
            headers=self._headers,
            json={
                "assistantId": self._assistant_id,
                "phoneNumberId": self._phone_number_id,
                "customer": {"number": self._destination},
            },
        )
        response.raise_for_status()
        payload = response.json()
        call_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("vapi_call_response_missing_id")
        status = payload.get("status") if isinstance(payload, dict) else None
        return VapiCall(
            call_id=call_id,
            status=str(status) if status is not None else None,
        )

    async def stop(self, call_id: str) -> None:
        response = await self._client.delete(
            f"/call/{call_id}", headers=self._headers
        )
        if response.status_code not in {200, 202, 204, 404}:
            response.raise_for_status()

    async def reconcile_and_stop(
        self, *, started_after_ms: int, ended_before_ms: int
    ) -> list[str]:
        # Vapi's stop() always has a call id in hand; no orphan guard needed.
        return []

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class VapiAgentEndpoint:
    def __init__(self, *, name: str, assistant_id: str | None = None) -> None:
        self.manifest = AgentEndpointManifest(
            name=name,
            provider="vapi",
            world_kinds=["voice"],
            capabilities=EndpointCapabilities(
                audio=True,
                text=True,
                streaming=True,
                interruption=True,
                dtmf=True,
                transfer=True,
                transcript_events=True,
                tool_events=True,
                usage_events=True,
                internal_metrics=True,
                recording=True,
                web_rtc=False,
                sip=True,
            ),
            metadata={"assistant_id": assistant_id} if assistant_id else {},
        )
        self.capabilities = self.manifest.capabilities

    async def discover(self, request: DiscoveryRequest) -> DiscoverySnapshot:
        del request
        return DiscoverySnapshot(capabilities=self.capabilities)

    async def prepare(self, plan) -> EndpointHandle:  # noqa: ANN001
        return EndpointHandle(
            handle_id=f"vapi-{uuid.uuid4().hex[:12]}",
            endpoint_name=self.manifest.name,
            created_at=datetime.now(timezone.utc),
            metadata={"plan_id": getattr(plan, "plan_id", None)},
        )

    async def wait_ready(self, handle: EndpointHandle) -> ReadinessResult:
        del handle
        raise NotImplementedError(
            "Vapi direct execution seam; live path uses LiveKit SIP outbound today"
        )

    async def send(
        self, handle: EndpointHandle, event: RealtimeEvent | AudioFrame
    ) -> None:
        raise NotImplementedError("VapiAgentEndpoint.send is a Stage-8 seam")

    async def receive(
        self, handle: EndpointHandle
    ) -> AsyncIterator[RealtimeEvent | AudioFrame]:
        raise NotImplementedError("VapiAgentEndpoint.receive is a Stage-8 seam")
        yield  # type: ignore[unreachable]

    async def stop(self, handle: EndpointHandle) -> None:
        del handle

    async def cleanup(self, handle: EndpointHandle) -> None:
        del handle

    async def reconcile(self, handle: EndpointHandle) -> ReconciliationResult:
        del handle
        return ReconciliationResult(reconciled=True)


__all__ = [
    "VapiAgentEndpoint",
    "VapiCall",
    "VapiCallOriginator",
    "VapiOriginatorConfigError",
]
