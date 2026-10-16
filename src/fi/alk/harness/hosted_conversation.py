from __future__ import annotations

import asyncio
import hashlib
import json
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .outbound import RequestsTransport, Transport, TransportError

CONVERSATION_SCHEMA_VERSION = "futureagi.harness-conversation.v1"
EVENT_SCHEMA_VERSION = "futureagi.harness-conversation-event.v1"
DEFAULT_CAPABILITIES_PATH = Path("/run/futureagi/conversation.json")
# Finish retries before the platform's 90-second heartbeat expiry can replace this process.
_UNAVAILABLE_BUDGET_SECONDS = 60.0


class ConversationTransportError(RuntimeError):
    pass


class ConversationEndpoints(BaseModel):
    # A newer platform may offer endpoints this harness does not use yet.
    model_config = ConfigDict(extra="ignore")

    commands: str
    events: str
    rerun: str
    adjust: str
    run_status: str
    workspace: str

    session_store: str
    session_store_append: str
    # Absent on platforms that predate the builder; its tools say so instead of failing.
    scenarios: str = ""
    publish: str = ""
    runs: str = ""
    run_start: str = ""
    overview: str = ""

    @field_validator(
        "commands",
        "events",
        "rerun",
        "adjust",
        "run_status",
        "workspace",
        "session_store",
        "session_store_append",
        "scenarios",
        "publish",
        "runs",
        "run_start",
        "overview",
    )
    @classmethod
    def _https_endpoint(cls, value: str) -> str:
        if not value:
            return value
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc or not value.endswith("/"):
            raise ValueError(
                "conversation endpoints must be absolute HTTPS URLs ending in '/'"
            )
        return value


class ConversationIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_name: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=255)


class ConversationCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    conversation_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    token: str = Field(min_length=1)
    fence: str = Field(min_length=1)
    expires_at: datetime
    identity: ConversationIdentity
    turn_context: dict[str, Any]
    endpoints: ConversationEndpoints

    @model_validator(mode="after")
    def _valid(self) -> ConversationCapabilities:
        if self.schema_version != CONVERSATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.expires_at <= datetime.now(timezone.utc):
            raise ValueError("conversation capability is expired")
        for endpoint in self.endpoints.model_dump().values():
            if endpoint and self.conversation_id not in urlparse(endpoint).path.split("/"):
                raise ValueError(
                    "conversation endpoint is bound to another conversation"
                )
        return self

    def auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "X-Harness-Conversation-Fence": self.fence,
        }


def load_conversation_capabilities(
    path: str | Path = DEFAULT_CAPABILITIES_PATH,
    *,
    unlink: bool = True,
) -> ConversationCapabilities:
    target = Path(path)
    try:
        body = json.loads(target.read_text(encoding="utf-8"))
        capabilities = ConversationCapabilities.model_validate(body)
    except (OSError, ValueError) as exc:
        raise ConversationTransportError(
            f"conversation capability could not be loaded: {type(exc).__name__}"
        ) from exc
    if unlink:
        try:
            target.unlink()
        except OSError:
            pass
    return capabilities


class ConversationClient:
    def __init__(
        self,
        capabilities: ConversationCapabilities,
        *,
        transport: Transport | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.transport = transport or RequestsTransport()
        self.command_watermark = int(
            capabilities.turn_context.get("command_watermark") or 0
        )
        self.event_watermark = int(
            capabilities.turn_context.get("event_watermark") or 0
        )
        self.turn_context = dict(capabilities.turn_context)
        self._emit_lock = asyncio.Lock()

    async def commands(self) -> list[dict[str, Any]]:
        separator = "&" if "?" in self.capabilities.endpoints.commands else "?"
        url = (
            f"{self.capabilities.endpoints.commands}{separator}"
            f"after={self.command_watermark}"
        )
        body = await self._request("GET", url)
        if isinstance(body.get("turn_context"), dict):
            self.turn_context = body["turn_context"]
        return list(body.get("commands") or [])

    async def emit(
        self,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
        message_id: str | None = None,
        stage: str = "",
        invocation_id: str | None = None,
        function_call_id: str | None = None,
        acknowledge_through: int | None = None,
    ) -> dict[str, Any]:
        # Provider IDs may include long signed metadata. Only the correlation key
        # crosses this channel; the SDK retains the original ID for tool replies.
        if function_call_id and len(function_call_id) > 255:
            function_call_id = "call-" + hashlib.sha256(
                function_call_id.encode("utf-8")
            ).hexdigest()
        async with self._emit_lock:
            sequence = self.event_watermark + 1
            event: dict[str, Any] = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event_id": f"ce-{uuid.uuid4().hex}",
                "conversation_id": self.capabilities.conversation_id,
                "sequence": sequence,
                "kind": kind,
                "message_id": message_id,
                "stage": stage,
                "invocation_id": invocation_id,
                "function_call_id": function_call_id,
                "emitted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "payload": payload or {},
            }
            event["digest"] = _canonical_digest(event)
            acknowledged = (
                self.command_watermark
                if acknowledge_through is None
                else acknowledge_through
            )
            body = await self._request(
                "POST",
                self.capabilities.endpoints.events,
                json_body={
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "acknowledged_through": acknowledged,
                    "events": [event],
                },
            )
            self.event_watermark = int(body["acked_through_sequence"])
            if acknowledge_through is not None:
                self.command_watermark = max(
                    self.command_watermark, acknowledge_through
                )
            return event

    async def checkpoint(self, workspace: Path) -> dict[str, Any]:
        body = await asyncio.to_thread(_workspace_archive, workspace)
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        headers = {
            **self.capabilities.auth_headers(),
            "Content-Type": "application/gzip",
            "X-Workspace-Digest": digest,
            "X-Workspace-Size": str(len(body)),
        }
        response = await asyncio.to_thread(
            self.transport.request,
            "PUT",
            self.capabilities.endpoints.workspace,
            headers=headers,
            data=body,
            timeout=120.0,
        )
        if response.status_code != 200 or response.body is None:
            raise ConversationTransportError(
                f"workspace checkpoint returned HTTP {response.status_code}"
            )
        return response.body

    async def rerun(self, workspace: Path) -> dict[str, Any]:
        await self.checkpoint(workspace)
        return await self._request(
            "POST",
            self.capabilities.endpoints.rerun,
            json_body={},
        )

    async def run_status(self) -> dict[str, Any]:
        return await self._request(
            "GET",
            self.capabilities.endpoints.run_status,
        )

    async def scenarios(self, query: dict[str, Any]) -> dict[str, Any]:
        return await self._platform("scenarios", query)

    async def publish(self, digest: str) -> dict[str, Any]:
        return await self._platform("publish", {"digest": digest})

    async def runs(self, run_id: str = "", limit: int = 10) -> dict[str, Any]:
        return await self._platform(
            "runs", {"run_id": run_id, "limit": limit} if run_id else {"limit": limit}
        )

    async def start_run(
        self, numbers: list[int], ranges: list[dict[str, int]], trials: int
    ) -> dict[str, Any]:
        # One id for every retry of this call, so a lost response never starts a second run.
        return await self._platform(
            "run_start",
            {"numbers": numbers, "ranges": ranges, "trials": trials, "operation_id": uuid.uuid4().hex},
        )

    async def overview(self) -> dict[str, Any]:
        return await self._platform("overview", {})

    async def _platform(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        url = getattr(self.capabilities.endpoints, endpoint)
        if not url:
            raise ConversationTransportError(
                f"this platform does not offer the {endpoint} endpoint"
            )
        return await self._request("POST", url, json_body=body)

    async def adjust(
        self, instruction: str, target_stage: str = "", scenario_delta: int = 0
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            self.capabilities.endpoints.adjust,
            json_body={
                "instruction": instruction,
                "client_request_id": f"conversation-{uuid.uuid4().hex}",
                **({"target_stage": target_stage} if target_stage else {}),
                **({"scenario_delta": scenario_delta} if scenario_delta > 0 else {}),
            },
        )

    async def load_session(self, key: dict[str, Any]) -> dict[str, Any]:
        query = urlencode(
            {
                "project_key": str(key["project_key"]),
                "session_id": str(key["session_id"]),
                "subpath": str(key.get("subpath") or ""),
            }
        )
        return await self._request(
            "GET",
            f"{self.capabilities.endpoints.session_store}?{query}",
        )

    async def append_session(
        self,
        key: dict[str, Any],
        entries: list[dict[str, Any]],
    ) -> None:
        await self._request(
            "POST",
            self.capabilities.endpoints.session_store_append,
            json_body={
                "project_key": str(key["project_key"]),
                "session_id": str(key["session_id"]),
                "subpath": str(key.get("subpath") or ""),
                "entries": entries,
            },
        )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _UNAVAILABLE_BUDGET_SECONDS
        delay = 0.25
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ConversationTransportError("conversation endpoint remained unavailable")
            try:
                response = await asyncio.to_thread(
                    self.transport.request,
                    method,
                    url,
                    headers=self.capabilities.auth_headers(),
                    json_body=json_body,
                    timeout=min(35.0, remaining),
                )
            except TransportError:
                response = None
            if (
                response is not None
                and response.status_code == 200
                and response.body is not None
            ):
                return response.body
            if response is not None and response.status_code in {
                400,
                401,
                403,
                404,
                409,
                422,
            }:
                said = (
                    response.body.get("message") or response.body.get("detail") or ""
                    if isinstance(response.body, dict)
                    else ""
                )
                raise ConversationTransportError(
                    f"conversation endpoint returned HTTP {response.status_code}"
                    + (f": {said}" if said else "")
                )
            if loop.time() + delay > deadline:
                status = response.status_code if response is not None else "unreachable"
                raise ConversationTransportError(
                    f"conversation endpoint remained unavailable ({status})"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)


class PlatformSessionStore:
    """Claude Agent SDK transcript storage over the scoped outbound channel."""

    def __init__(self, client: ConversationClient) -> None:
        self._client = client

    async def append(
        self,
        key: dict[str, Any],
        entries: list[dict[str, Any]],
    ) -> None:
        await self._client.append_session(dict(key), list(entries))

    async def load(
        self,
        key: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        body = await self._client.load_session(dict(key))
        entries = body.get("entries")
        return list(entries) if isinstance(entries, list) else None

    async def list_subkeys(self, key: dict[str, Any]) -> list[str]:
        body = await self._client.load_session(dict(key))
        return [str(value) for value in body.get("subkeys") or []]


def _workspace_archive(workspace: Path) -> bytes:
    root = workspace.resolve()
    with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as file:
        with tarfile.open(fileobj=file, mode="w:gz") as archive:
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if not relative.parts or any(
                    part in {"__pycache__", "outbound-spool", "telemetry"}
                    for part in relative.parts
                ):
                    continue
                if path.is_symlink():
                    continue
                # The model's CLI writes and removes its own files under the workspace while we pack.
                try:
                    archive.add(path, arcname=relative.as_posix(), recursive=False)
                except FileNotFoundError:
                    continue
        file.seek(0)
        return file.read()


def _canonical_digest(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()
