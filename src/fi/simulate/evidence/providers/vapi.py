"""Vapi post-call evidence adapter.

Polls ``GET https://api.vapi.ai/call/{call_id}`` after the SDK's phone
leg finishes. Downloads the provider recording, extracts transcript,
tool calls, analysis, cost, and latency, and returns those as an
``EvidenceSourceSummary`` plus ``ArtifactManifestEntry`` rows. The
adapter never imports from ``futureagi/`` and reads credentials from
env only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import timedelta, timezone
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

_VAPI_API_BASE = "https://api.vapi.ai"
_TERMINAL_STATUSES = {"ended", "failed", "cancelled"}
_ADAPTER = "vapi"
_RECORDING_ENDPOINTS = {
    "combined": "mono-recording",
    "assistant": "assistant-recording",
    "customer": "customer-recording",
    "stereo": "stereo-recording",
}
logger = logging.getLogger(__name__)


class VapiEvidenceSource:
    capabilities = EvidenceCapabilities(
        transcript=True,
        audio=True,
        tool_calls=True,
        tool_results=True,
        usage=True,
        internal_latency=True,
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
        if config.provider != "vapi":
            raise ProviderConfigError(
                f"vapi adapter requires provider='vapi', got {config.provider!r}"
            )
        self._config = config
        self._api_key = api_key or os.environ.get("VAPI_API_KEY")
        if not self._api_key:
            raise ProviderConfigError("VAPI_API_KEY is required for the Vapi adapter")
        self._client = client or httpx.AsyncClient(
            base_url=api_base_url or _VAPI_API_BASE,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._owns_client = client is None
        self._context: EvidenceContext | None = None
        self._source_id = f"vapi:{uuid.uuid4().hex[:12]}"

    async def connect(self, context: EvidenceContext) -> None:
        self._context = context

    async def fetch_final(self) -> ProviderFetchResult:
        if self._context is None:
            raise RuntimeError("vapi_adapter_not_connected")
        context = self._context
        call_id = context.call_id_hint
        try:
            if not call_id and self._config.call_id_source == "polling_window":
                call_id = await self._locate_call_id()
            if not call_id:
                return self._unavailable("vapi_call_id_missing")
            call_payload = await self._poll_call(call_id)
        except httpx.HTTPError as exc:
            return self._unavailable(
                "vapi_fetch_failed",
                error=type(exc).__name__,
            )
        artifacts = await self._download_recordings(call_payload, call_id)
        summary = self._summarize(call_payload, call_id, artifacts)
        return ProviderFetchResult(summary=summary, artifacts=artifacts)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _locate_call_id(self) -> str | None:
        assert self._context is not None
        window = self._config.polling_window_seconds
        if not window:
            return None
        started = self._context.started_at.astimezone(timezone.utc)
        params = {
            "limit": 100,
            "createdAtGt": (started - timedelta(seconds=window)).isoformat(),
            "createdAtLt": (started + timedelta(seconds=window)).isoformat(),
        }
        deadline = (
            asyncio.get_running_loop().time() + self._config.poll_deadline_seconds
        )
        while True:
            response = await self._client.get("/call", params=params)
            response.raise_for_status()
            candidates = _select_vapi_calls(response.json())
            matches = [
                call
                for call in candidates
                if _matches_call_numbers(
                    call,
                    caller=self._context.caller_phone,
                    callee=self._context.callee_phone,
                )
            ]
            if matches:
                matches.sort(
                    key=lambda call: str(
                        call.get("startedAt") or call.get("createdAt") or ""
                    ),
                    reverse=True,
                )
                call_id = matches[0].get("id")
                if call_id:
                    return str(call_id)
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(self._config.poll_interval_seconds)

    async def _poll_call(self, call_id: str) -> dict[str, Any]:
        deadline = (
            asyncio.get_running_loop().time() + self._config.poll_deadline_seconds
        )
        while True:
            response = await self._client.get(f"/call/{call_id}")
            response.raise_for_status()
            payload = response.json()
            status = str(payload.get("status") or "").lower()
            if status in _TERMINAL_STATUSES:
                return payload
            if asyncio.get_running_loop().time() >= deadline:
                return payload
            await asyncio.sleep(self._config.poll_interval_seconds)

    async def _download_recordings(
        self,
        payload: dict[str, Any],
        call_id: str,
    ) -> list[ArtifactManifestEntry]:
        assert self._context is not None
        artifact_dir = self._context.case_directory / "provider"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        urls = _extract_vapi_recording_urls(payload)
        entries: list[ArtifactManifestEntry] = []
        for label, url in urls.items():
            if not url:
                continue
            try:
                data = await self._get_recording(call_id, label)
            except httpx.HTTPError as exc:
                logger.warning(
                    "vapi recording download failed",
                    extra={"label": label, "error": type(exc).__name__},
                )
                continue
            filename = f"vapi_{call_id}_{label}.wav"
            path = artifact_dir / filename
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
                    metadata={"provider": "vapi", "recording_label": label},
                )
            )
        return entries

    async def _get_recording(self, call_id: str, label: str) -> bytes:
        endpoint = _RECORDING_ENDPOINTS[label]
        response = await self._client.get(
            f"/call/{call_id}/{endpoint}",
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.content

    def _summarize(
        self,
        payload: dict[str, Any],
        call_id: str,
        artifacts: list[ArtifactManifestEntry],
    ) -> EvidenceSourceSummary:
        assert self._context is not None
        artifact = payload.get("artifact") or {}
        performance = (
            artifact.get("performanceMetrics")
            or payload.get("performanceMetrics")
            or {}
        )
        transcript_messages = artifact.get("messages") or payload.get("messages") or []
        tool_calls = _extract_tool_calls(transcript_messages)
        tool_results = _extract_tool_results(transcript_messages)
        cost_summary = _cost_summary(payload)
        metadata: dict[str, Any] = {
            "provider": "vapi",
            "call_id": call_id,
            "status": payload.get("status"),
            "ended_reason": payload.get("endedReason"),
            "started_at": payload.get("startedAt"),
            "ended_at": payload.get("endedAt"),
            "tool_call_count": len(tool_calls),
            "tool_calls": tool_calls or None,
            "tool_result_count": len(tool_results),
            "tool_results": tool_results or None,
            "message_count": len(transcript_messages),
            "latency": coerce_json(performance) or None,
            "cost": cost_summary,
            "analysis_summary": coerce_json(
                (payload.get("analysis") or {}).get("summary")
            ),
            "recording_labels": [entry.leg_id for entry in artifacts if entry.leg_id],
        }
        if self._context.caller_phone:
            metadata["caller_phone"] = redact_phone(self._context.caller_phone)
        if self._context.termination_source:
            metadata["termination_source"] = self._context.termination_source
            if payload.get("endedReason") == "call-deleted":
                metadata["ended_reason_interpretation"] = "sdk_originator_teardown"
        return EvidenceSourceSummary(
            source_id=self._source_id,
            adapter=_ADAPTER,
            evidence_class=EvidenceClass.PROVIDER_REPORTED,
            capabilities=self.capabilities,
            available=str(payload.get("status") or "").lower() in _TERMINAL_STATUSES,
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
            metadata={"provider": "vapi", "reason": code, **coerce_json(details)},
        )
        return ProviderFetchResult(summary=summary, artifacts=[])


def _select_vapi_calls(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("calls", "results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _matches_call_numbers(
    payload: dict[str, Any],
    *,
    caller: str | None,
    callee: str | None,
) -> bool:
    if caller:
        customer = payload.get("customer") or {}
        if _normalized_phone(customer.get("number")) != _normalized_phone(caller):
            return False
    if callee:
        phone_number = payload.get("phoneNumber") or {}
        target_number = (
            phone_number.get("number")
            or payload.get("phoneNumberNumber")
            or payload.get("toNumber")
        )
        if target_number and _normalized_phone(target_number) != _normalized_phone(
            callee
        ):
            return False
    return True


def _normalized_phone(value: Any) -> str:
    return "".join(character for character in str(value or "") if character.isdigit())


def _extract_vapi_recording_urls(payload: dict[str, Any]) -> dict[str, str | None]:
    artifact = payload.get("artifact") or {}
    recording = artifact.get("recording") or payload.get("recording") or {}
    mono = recording.get("mono") if isinstance(recording, dict) else {}
    urls: dict[str, str | None] = {
        "combined": (
            (mono or {}).get("combinedUrl") if isinstance(mono, dict) else None
        )
        or (recording if isinstance(recording, str) else None)
        or artifact.get("recordingUrl")
        or payload.get("recordingUrl"),
        "assistant": (mono or {}).get("assistantUrl")
        if isinstance(mono, dict)
        else None,
        "customer": (mono or {}).get("customerUrl") if isinstance(mono, dict) else None,
        "stereo": (recording.get("stereoUrl") if isinstance(recording, dict) else None)
        or artifact.get("stereoRecordingUrl")
        or payload.get("stereoRecordingUrl"),
    }
    return urls


def _extract_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for entry in messages:
        if not isinstance(entry, dict):
            continue
        tool_calls = (
            entry.get("toolCalls")
            or entry.get("tool_calls")
            or entry.get("toolCallList")
            or []
        )
        if not tool_calls and isinstance(entry.get("toolWithToolCallList"), list):
            tool_calls = [
                wrapped.get("toolCall")
                for wrapped in entry["toolWithToolCallList"]
                if isinstance(wrapped, dict)
                and isinstance(wrapped.get("toolCall"), dict)
            ]
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if isinstance(call, dict):
                function = call.get("function") or {}
                calls.append(
                    {
                        key: value
                        for key, value in {
                            "id": call.get("id") or call.get("toolCallId"),
                            "name": function.get("name") or call.get("name"),
                            "arguments": function.get("arguments")
                            or call.get("parameters")
                            or call.get("arguments"),
                        }.items()
                        if value is not None
                    }
                )
    return calls


def _extract_tool_results(messages: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for entry in messages:
        if not isinstance(entry, dict):
            continue
        candidates = (
            entry.get("toolCallResultList")
            or entry.get("tool_call_results")
            or entry.get("results")
            or []
        )
        if not candidates and entry.get("role") == "tool":
            candidates = [entry]
        if not isinstance(candidates, list):
            continue
        for result in candidates:
            if not isinstance(result, dict):
                continue
            results.append(
                {
                    key: value
                    for key, value in {
                        "tool_call_id": result.get("toolCallId")
                        or result.get("tool_call_id"),
                        "name": result.get("name"),
                        "result": result.get("result"),
                        "content": result.get("content") or result.get("message"),
                        "error": result.get("error"),
                    }.items()
                    if value is not None
                }
            )
    return results


def _cost_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    total = payload.get("cost")
    breakdown = payload.get("costBreakdown")
    if total is None and not breakdown:
        return None
    return {"total": total, "breakdown": coerce_json(breakdown)}
