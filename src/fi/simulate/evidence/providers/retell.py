"""Retell post-call evidence adapter.

Retell has no PSTN-outbound API. This adapter only supports inbound legs
(Retell agent dialed our number) and web-call bridge legs (already
established elsewhere in the run). It matches the Retell call by
``from_number`` + start-time window through ``POST /v3/list-calls``, then
fetches the full call with ``GET /get-call/{call_id}``. Credentials are
read from env.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import timedelta
from typing import Any

import httpx

from fi.simulate.agent.definition import ProviderEvidenceConfig
from fi.simulate.artifacts.manifest import ArtifactManifestEntry
from fi.simulate.evidence.base import (
    EvidenceCapabilities,
    EvidenceClass,
    EvidenceSourceSummary,
)

from .base import (
    EvidenceContext,
    ProviderConfigError,
    ProviderFetchResult,
    checksum_bytes,
    coerce_json,
    redact_phone,
)

_RETELL_API_BASE = "https://api.retellai.com"
# "completed" is not a real Retell status; a never-connected dial is terminal.
_TERMINAL_STATUSES = {"ended", "not_connected", "error"}
# the two hint sources route to _poll_call; polling_window goes to list-calls
_HINT_CALL_ID_SOURCES = frozenset({"participant_attribute", "originator_response"})
_ADAPTER = "retell"
logger = logging.getLogger(__name__)


class RetellEvidenceSource:
    capabilities = EvidenceCapabilities(
        transcript=True,
        audio=True,
        tool_calls=True,
        tool_results=False,
        usage=True,
        internal_latency=False,
        configuration_snapshot=False,
    )

    def __init__(
        self,
        config: ProviderEvidenceConfig,
        *,
        api_key: str | None = None,
        api_base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if config.provider != "retell":
            raise ProviderConfigError(
                f"retell adapter requires provider='retell', got {config.provider!r}"
            )
        self._config = config
        self._api_key = api_key or os.environ.get("RETELL_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "RETELL_API_KEY is required for the Retell adapter"
            )
        self._client = client or httpx.AsyncClient(
            base_url=api_base_url or _RETELL_API_BASE,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._owns_client = client is None
        self._context: EvidenceContext | None = None
        self._source_id = f"retell:{uuid.uuid4().hex[:12]}"

    async def connect(self, context: EvidenceContext) -> None:
        self._context = context

    async def fetch_final(self) -> ProviderFetchResult:
        if self._context is None:
            raise RuntimeError("retell_adapter_not_connected")
        try:
            call_payload = await self._locate_and_fetch_call()
        except httpx.HTTPError as exc:
            return self._unavailable("retell_fetch_failed", error=type(exc).__name__)
        if call_payload is None:
            if (
                self._config.call_id_source in _HINT_CALL_ID_SOURCES
                and self._context.call_id_hint
            ):
                # hint path has no matching step; None here means the GET failed
                return self._unavailable("retell_get_call_failed")
            return self._unavailable("retell_call_not_matched")
        artifacts = await self._download_recording(call_payload)
        summary = self._summarize(call_payload, artifacts)
        return ProviderFetchResult(summary=summary, artifacts=artifacts)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _locate_and_fetch_call(self) -> dict[str, Any] | None:
        assert self._context is not None
        context = self._context
        if (
            self._config.call_id_source in _HINT_CALL_ID_SOURCES
            and context.call_id_hint
        ):
            return await self._poll_call(context.call_id_hint)
        window = self._config.polling_window_seconds
        if not window:
            return None
        started = context.started_at
        upper = int((started + timedelta(seconds=window)).timestamp() * 1000)
        lower = int((started - timedelta(seconds=window)).timestamp() * 1000)
        filters: dict[str, Any] = {
            "start_timestamp": {
                "type": "range",
                "op": "bt",
                "value": [lower, upper],
            },
        }
        if context.caller_phone:
            filters["from_number"] = {
                "type": "string",
                "op": "eq",
                "value": context.caller_phone,
            }
        deadline = (
            asyncio.get_running_loop().time() + self._config.poll_deadline_seconds
        )
        while True:
            response = await self._client.post(
                "/v3/list-calls",
                json={"limit": 5, "filter_criteria": filters},
            )
            response.raise_for_status()
            payload = response.json()
            candidates = _select_calls(payload)
            terminal = [
                item
                for item in candidates
                if str(item.get("call_status") or "").lower() in _TERMINAL_STATUSES
            ]
            if terminal:
                # Fetch the full call payload for the newest terminal match.
                terminal.sort(
                    key=lambda item: (
                        item.get("end_timestamp") or item.get("start_timestamp") or 0
                    ),
                    reverse=True,
                )
                call_id = terminal[0].get("call_id")
                if call_id:
                    return await self._get_call(str(call_id))
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(self._config.poll_interval_seconds)

    async def _poll_call(self, call_id: str) -> dict[str, Any] | None:
        # Mirrors vapi._poll_call: Retell fills transcript/recording asynchronously.
        deadline = (
            asyncio.get_running_loop().time() + self._config.poll_deadline_seconds
        )
        while True:
            payload = await self._get_call(call_id)
            if payload is None:
                return (
                    None  # 4xx/5xx on get-call: preserve existing return-None behaviour
                )
            status = str(payload.get("call_status") or "").lower()
            if status in _TERMINAL_STATUSES:
                return payload
            if asyncio.get_running_loop().time() >= deadline:
                return payload
            await asyncio.sleep(self._config.poll_interval_seconds)

    async def _get_call(self, call_id: str) -> dict[str, Any] | None:
        try:
            response = await self._client.get(f"/v2/get-call/{call_id}")
            response.raise_for_status()
        except httpx.HTTPStatusError:
            return None
        return response.json()

    async def _download_recording(
        self, payload: dict[str, Any]
    ) -> list[ArtifactManifestEntry]:
        assert self._context is not None
        artifact_dir = self._context.case_directory / "provider"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        entries: list[ArtifactManifestEntry] = []
        call_id = str(payload.get("call_id") or "call")
        for label, url in _retell_recording_urls(payload).items():
            if not url:
                continue
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(60.0, connect=10.0)
                ) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.content
            except httpx.HTTPError as exc:
                logger.warning(
                    "retell recording download failed",
                    extra={"label": label, "error": type(exc).__name__},
                )
                continue
            path = artifact_dir / f"retell_{call_id}_{label}.wav"
            path.write_bytes(data)
            entries.append(
                ArtifactManifestEntry(
                    artifact_id=f"{self._source_id}:{label}",
                    test_case_id=self._context.test_case_id,
                    type="audio",
                    path=str(path),
                    checksum=checksum_bytes(data),
                    size_bytes=len(data),
                    mime_type="audio/wav",
                    codec="pcm",
                    evidence_class=EvidenceClass.PROVIDER_REPORTED,
                    evidence_source_id=self._source_id,
                    leg_id=label,
                    metadata={"provider": "retell", "recording_label": label},
                )
            )
        return entries

    def _summarize(
        self,
        payload: dict[str, Any],
        artifacts: list[ArtifactManifestEntry],
    ) -> EvidenceSourceSummary:
        assert self._context is not None
        transcript_events = payload.get("transcript_with_tool_calls") or []
        tool_calls = _extract_retell_tool_calls(transcript_events)
        cost = payload.get("call_cost") or {}
        metadata: dict[str, Any] = {
            "provider": "retell",
            "call_id": payload.get("call_id"),
            "status": payload.get("call_status"),
            "end_reason": payload.get("disconnection_reason"),
            "start_timestamp": payload.get("start_timestamp"),
            "end_timestamp": payload.get("end_timestamp"),
            "tool_call_count": len(tool_calls),
            "message_count": len(transcript_events),
            "cost": coerce_json(cost) if cost else None,
            "usage": coerce_json(payload.get("llm_token_usage")),
            "recording_labels": [entry.leg_id for entry in artifacts if entry.leg_id],
        }
        if self._context.caller_phone:
            metadata["caller_phone"] = redact_phone(self._context.caller_phone)
        return EvidenceSourceSummary(
            source_id=self._source_id,
            adapter=_ADAPTER,
            evidence_class=EvidenceClass.PROVIDER_REPORTED,
            capabilities=self.capabilities,
            available=str(payload.get("call_status") or "").lower()
            in _TERMINAL_STATUSES,
            redactions=["auth", "phone_e164"],
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    def _unavailable(self, code: str, **details: Any) -> ProviderFetchResult:
        summary = EvidenceSourceSummary(
            source_id=self._source_id,
            adapter=_ADAPTER,
            evidence_class=EvidenceClass.PROVIDER_REPORTED,
            capabilities=self.capabilities,
            available=False,
            redactions=["auth", "phone_e164"],
            metadata={"provider": "retell", "reason": code, **coerce_json(details)},
        )
        return ProviderFetchResult(summary=summary, artifacts=[])


def _select_calls(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("calls", "results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _retell_recording_urls(payload: dict[str, Any]) -> dict[str, str | None]:
    return {
        "mono": payload.get("recording_url"),
        "stereo": payload.get("recording_multi_channel_url"),
    }


def _extract_retell_tool_calls(events: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for entry in events:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("role") or "").lower() != "tool_call_invocation":
            continue
        calls.append(
            {
                "id": entry.get("tool_call_id"),
                "name": entry.get("name"),
            }
        )
    return calls
