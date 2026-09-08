"""Built-in, ownership-safe import adapters for provider-hosted voice targets.

The adapter copies configuration; it never fabricates tool implementations. Custom HTTP tools are
rewired to an already-running world endpoint. Provider-native tools are preserved. Every created
resource is returned in a validated lifecycle receipt so cleanup is exact and retryable.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field
from pydantic import model_validator

from .provider_lifecycle import (
    ProviderCleanupReceipt,
    ProviderContext,
    ProviderProvisionReceipt,
    ProviderResource,
    ProviderTarget,
    ProviderType,
)


class ProviderImportError(RuntimeError):
    """The provider definition could not be copied without guessing."""


class ProviderImportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ProviderType
    source_target_id: str = Field(min_length=1)
    public_capability: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    environment_tools: list[str] = Field(default_factory=list)
    event_path: str = "/provider/events"
    tool_path: str = "/provider/tools"
    api_base_url: str | None = None
    target_modality: Literal["voice", "chat"] = "voice"

    @model_validator(mode="after")
    def _provider_api_origin_is_fixed(self) -> "ProviderImportSpec":
        if self.api_base_url:
            expected = {
                ProviderType.VAPI: "https://api.vapi.ai",
                ProviderType.RETELL: "https://api.retellai.com",
            }[self.type]
            if self.api_base_url.rstrip("/") != expected:
                raise ValueError("provider_import_api_base_url_not_allowed")
        for name, value in (
            ("event_path", self.event_path),
            ("tool_path", self.tool_path),
        ):
            if (
                not value.startswith("/")
                or value.startswith("//")
                or ".." in value.split("/")
            ):
                raise ValueError(f"provider_import_{name}_invalid")
        return self

    @model_validator(mode="after")
    def _environment_tool_names_are_unambiguous(self) -> "ProviderImportSpec":
        normalized = [name.strip() for name in self.environment_tools]
        if any(not name for name in normalized) or len(set(normalized)) != len(
            normalized
        ):
            raise ValueError("provider_import_environment_tools_invalid")
        self.environment_tools = normalized
        return self


JsonRequest = Callable[[str, str, str, Mapping[str, Any] | None], dict[str, Any]]


def _safe_profile_value(value: Any, key: str = "") -> Any:
    """Return provider configuration that is safe to give the authoring model.

    Imported assistant definitions are part of the source of truth for contract and scenario
    authoring, but provider responses may also contain credentials or signed callback URLs. Keep
    behavioral configuration while removing secrets and URL userinfo/query fragments.
    """
    lowered = key.lower()
    if any(
        marker in lowered
        for marker in (
            "secret",
            "token",
            "api_key",
            "apikey",
            "credential",
            "authorization",
        )
    ):
        return "[redacted]"
    if isinstance(value, Mapping):
        return {
            str(item_key): _safe_profile_value(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_profile_value(item, key) for item in value]
    if isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            hostname = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            return urlunsplit((parsed.scheme, hostname + port, parsed.path, "", ""))
    return value


def inspect_provider_target(
    provider: ProviderType | str,
    *,
    source_target_id: str,
    api_key: str,
    api_base_url: str | None = None,
    target_modality: Literal["voice", "chat"] = "voice",
    request: JsonRequest | None = None,
) -> dict[str, Any]:
    """Fetch a sanitized, read-only behavioral profile for hosted authoring.

    This deliberately performs no create/update/delete operation. The profile lets contract and
    scenario generation see the externally hosted prompt, model/voice configuration and exact
    tool schemas instead of guessing solely from a submitted webhook implementation.
    """
    provider = ProviderType(provider)
    if not api_key:
        raise ProviderImportError(f"{provider.value}_api_key_missing")
    if not source_target_id:
        raise ProviderImportError("provider_import_source_target_id_missing")
    request = request or _request_json
    if provider is ProviderType.VAPI:
        base = (api_base_url or "https://api.vapi.ai").rstrip("/")
        assistant = request(
            "GET", f"{base}/assistant/{source_target_id}", api_key, None
        )
        model = assistant.get("model")
        reusable_tools: list[dict[str, Any]] = []
        if isinstance(model, Mapping):
            for tool_id in model.get("toolIds") or []:
                reusable_tools.append(
                    request("GET", f"{base}/tool/{tool_id}", api_key, None)
                )
        profile = {
            "provider": "vapi",
            "source_target_id": source_target_id,
            "name": assistant.get("name"),
            "first_message": assistant.get("firstMessage"),
            "first_message_mode": assistant.get("firstMessageMode"),
            "model": model,
            "voice": assistant.get("voice"),
            "transcriber": assistant.get("transcriber"),
            "end_call_message": assistant.get("endCallMessage"),
            "end_call_phrases": assistant.get("endCallPhrases"),
            "reusable_tools": reusable_tools,
        }
    else:
        base = (api_base_url or "https://api.retellai.com").rstrip("/")
        agent_path = "get-chat-agent" if target_modality == "chat" else "get-agent"
        agent = request("GET", f"{base}/{agent_path}/{source_target_id}", api_key, None)
        engine = agent.get("response_engine")
        if not isinstance(engine, Mapping):
            raise ProviderImportError(
                "retell_response_engine_unsupported: response_engine must be an object"
            )
        engine_type = str(engine.get("type") or "").strip()
        profile: dict[str, Any] = {
            "provider": "retell",
            "modality": target_modality,
            "source_target_id": source_target_id,
            "name": agent.get("agent_name"),
            "voice_id": agent.get("voice_id"),
            "language": agent.get("language"),
            "responsiveness": agent.get("responsiveness"),
            "interruption_sensitivity": agent.get("interruption_sensitivity"),
            "enable_backchannel": agent.get("enable_backchannel"),
            "response_engine_type": engine_type,
        }
        if engine_type == "retell-llm":
            llm_id = str(engine.get("llm_id") or "").strip()
            if not llm_id:
                raise ProviderImportError("retell_response_engine_missing_llm_id")
            llm = request("GET", f"{base}/get-retell-llm/{llm_id}", api_key, None)
            profile.update(
                {
                    "begin_message": llm.get("begin_message"),
                    "general_prompt": llm.get("general_prompt"),
                    "general_tools": llm.get("general_tools"),
                    "states": llm.get("states"),
                    "model": llm.get("model"),
                }
            )
        elif engine_type == "conversation-flow":
            flow_id = str(engine.get("conversation_flow_id") or "").strip()
            if not flow_id:
                raise ProviderImportError(
                    "retell_response_engine_missing_conversation_flow_id"
                )
            flow = request(
                "GET", f"{base}/get-conversation-flow/{flow_id}", api_key, None
            )
            profile["conversation_flow"] = flow
        else:
            raise ProviderImportError(
                "retell_response_engine_unsupported: provider_import supports "
                "retell-llm and conversation-flow"
            )
    return _safe_profile_value(profile)


def _request_json(
    method: str, url: str, api_key: str, body: Mapping[str, Any] | None
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Vapi's Cloudflare policy rejects urllib's default Python user
            # agent as Error 1010 from hosted sandboxes. Use a stable product
            # identity for provider API traffic instead.
            "User-Agent": "FutureAGI-ALK/1.0",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed provider hosts
            raw = response.read()
    except HTTPError as exc:
        # Provider APIs commonly return a safe machine-readable explanation (for
        # example an account/IP policy or an invalid target id). Preserve a
        # bounded, single-line excerpt so hosted failures are actionable. Never
        # include request headers or the caller's credential.
        try:
            detail = exc.read(512).decode("utf-8", errors="replace")
        except (AttributeError, OSError):
            detail = ""
        detail = " ".join(detail.split())
        suffix = f": {detail}" if detail else ""
        raise ProviderImportError(
            f"provider_api_error: {method} returned HTTP {exc.code}{suffix}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise ProviderImportError(
            f"provider_api_unavailable: {method} {type(exc).__name__}"
        ) from exc
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ProviderImportError("provider_api_response_invalid_json") from exc
    if not isinstance(value, dict):
        raise ProviderImportError("provider_api_response_must_be_object")
    return value


def _without(body: Mapping[str, Any], names: set[str]) -> dict[str, Any]:
    return {key: value for key, value in body.items() if key not in names}


def _join(base: str, original: Any) -> str:
    """Preserve a declared route while moving only its origin into the world."""
    from urllib.parse import urlsplit

    path = urlsplit(str(original or "")).path
    if not path or path == "/":
        return base
    base_path = urlsplit(base).path.rstrip("/")
    # A common source definition already uses the same logical mount
    # (`/provider/tools/...`) as ALK's signed world endpoint. Do not append the
    # mount twice when only the origin is changing.
    if base_path and (path == base_path or path.startswith(base_path + "/")):
        path = path[len(base_path) :]
    return base.rstrip("/") + "/" + path.lstrip("/")


def _rewire_vapi_tool(tool: Mapping[str, Any], tool_base_url: str) -> dict[str, Any]:
    copied = dict(tool)
    kind = str(copied.get("type") or "")
    if kind == "apiRequest" and copied.get("url"):
        copied["url"] = _join(tool_base_url, copied["url"])
    elif kind == "function":
        server = copied.get("server")
        if isinstance(server, Mapping) and server.get("url"):
            copied["server"] = {**server, "url": _join(tool_base_url, server["url"])}
    return copied


def _vapi_custom_tool_name(tool: Mapping[str, Any]) -> str | None:
    kind = str(tool.get("type") or "")
    if kind == "function":
        function = tool.get("function")
        if isinstance(function, Mapping):
            return str(function.get("name") or "").strip() or None
    if kind == "apiRequest":
        return str(tool.get("name") or "").strip() or None
    return None


def _require_environment_tool(
    *, provider: str, name: str | None, available: set[str]
) -> None:
    if not name:
        raise ProviderImportError(f"{provider}_custom_tool_missing_name")
    if name not in available:
        raise ProviderImportError(
            f"provider_tool_implementation_missing: {provider} tool {name!r} "
            "is not declared by the submitted environment"
        )


def _rewire_retell_tools(value: Any, tool_base_url: str) -> Any:
    if isinstance(value, list):
        return [_rewire_retell_tools(item, tool_base_url) for item in value]
    if not isinstance(value, Mapping):
        return value
    copied = {
        key: _rewire_retell_tools(item, tool_base_url) for key, item in value.items()
    }
    if str(copied.get("type") or "") == "custom" and copied.get("url"):
        copied["url"] = _join(tool_base_url, copied["url"])
    return copied


def _validate_retell_custom_tools(value: Any, environment_tools: set[str]) -> None:
    """Validate every custom tool wherever Retell nests it in an engine graph."""
    if isinstance(value, list):
        for item in value:
            _validate_retell_custom_tools(item, environment_tools)
        return
    if not isinstance(value, Mapping):
        return
    if str(value.get("type") or "") == "custom":
        _require_environment_tool(
            provider="retell",
            name=str(value.get("name") or "").strip() or None,
            available=environment_tools,
        )
    for item in value.values():
        _validate_retell_custom_tools(item, environment_tools)


_RETELL_AGENT_READ_ONLY_FIELDS = {
    "agent_id",
    "last_modification_timestamp",
    "version",
    "base_version",
    "is_published",
}

_RETELL_ENGINE_READ_ONLY_FIELDS = {
    "llm_id",
    "conversation_flow_id",
    "last_modification_timestamp",
    "version",
    "is_published",
}


@dataclass(frozen=True)
class _CloneResult:
    target_id: str
    target_kind: str
    resources: tuple[ProviderResource, ...]
    metadata: dict[str, Any]


def _clone_vapi(
    spec: ProviderImportSpec,
    context: ProviderContext,
    api_key: str,
    request: JsonRequest,
) -> _CloneResult:
    base = (spec.api_base_url or "https://api.vapi.ai").rstrip("/")
    original = request(
        "GET", f"{base}/assistant/{spec.source_target_id}", api_key, None
    )
    assistant = _without(
        original,
        {
            "id",
            "orgId",
            "createdAt",
            "updatedAt",
            "latestVersion",
            "isServerUrlSecretSet",
        },
    )
    assistant["name"] = context.provider_resource_prefix[:40]
    assistant["metadata"] = {
        **(
            assistant.get("metadata")
            if isinstance(assistant.get("metadata"), dict)
            else {}
        ),
        "alk_attempt_id": context.attempt_id,
        "alk_world_id": context.world_id,
    }
    assistant["server"] = {"url": context.event_url}
    resources: list[ProviderResource] = []
    environment_tools = set(spec.environment_tools)
    try:
        model = assistant.get("model")
        if isinstance(model, dict):
            model = dict(model)
            inline = model.get("tools")
            if isinstance(inline, list):
                rewritten_inline: list[Any] = []
                for item in inline:
                    if isinstance(item, Mapping) and str(item.get("type") or "") in {
                        "function",
                        "apiRequest",
                    }:
                        _require_environment_tool(
                            provider="vapi",
                            name=_vapi_custom_tool_name(item),
                            available=environment_tools,
                        )
                        item = _rewire_vapi_tool(item, context.tool_base_url)
                    rewritten_inline.append(item)
                model["tools"] = rewritten_inline
            tool_ids = model.get("toolIds")
            if isinstance(tool_ids, list):
                cloned_ids: list[str] = []
                for source_id in tool_ids:
                    source = request("GET", f"{base}/tool/{source_id}", api_key, None)
                    if str(source.get("type") or "") in {"function", "apiRequest"}:
                        _require_environment_tool(
                            provider="vapi",
                            name=_vapi_custom_tool_name(source),
                            available=environment_tools,
                        )
                    create = _rewire_vapi_tool(
                        _without(source, {"id", "orgId", "createdAt", "updatedAt"}),
                        context.tool_base_url,
                    )
                    created = request("POST", f"{base}/tool", api_key, create)
                    target_id = str(created.get("id") or "").strip()
                    if not target_id:
                        raise ProviderImportError("vapi_tool_create_missing_id")
                    cloned_ids.append(target_id)
                    resources.append(
                        ProviderResource(kind="tool", id=target_id, owned=True)
                    )
                model["toolIds"] = cloned_ids
            assistant["model"] = model
        created = request("POST", f"{base}/assistant", api_key, assistant)
    except ProviderImportError:
        for resource in reversed(resources):
            try:
                request("DELETE", f"{base}/tool/{resource.id}", api_key, None)
            except ProviderImportError:
                pass
        raise
    target_id = str(created.get("id") or "").strip()
    if not target_id:
        for resource in reversed(resources):
            try:
                request("DELETE", f"{base}/tool/{resource.id}", api_key, None)
            except ProviderImportError:
                pass
        raise ProviderImportError("vapi_assistant_create_missing_id")
    resources.append(ProviderResource(kind="assistant", id=target_id, owned=True))
    return _CloneResult(
        target_id=target_id,
        target_kind="assistant",
        resources=tuple(resources),
        metadata={
            "source_target_id": spec.source_target_id,
            "clone_kind": "provider_import",
        },
    )


def _clone_retell(
    spec: ProviderImportSpec,
    context: ProviderContext,
    api_key: str,
    request: JsonRequest,
) -> _CloneResult:
    base = (spec.api_base_url or "https://api.retellai.com").rstrip("/")
    agent_path = "get-chat-agent" if spec.target_modality == "chat" else "get-agent"
    original = request(
        "GET", f"{base}/{agent_path}/{spec.source_target_id}", api_key, None
    )
    engine = original.get("response_engine")
    if not isinstance(engine, Mapping):
        raise ProviderImportError(
            "retell_response_engine_unsupported: response_engine must be an object"
        )
    environment_tools = set(spec.environment_tools)
    engine_type = str(engine.get("type") or "").strip()
    engine_resource: ProviderResource
    cloned_engine: dict[str, Any]
    source_engine_id: str
    delete_engine_url: str

    if engine_type == "retell-llm":
        source_engine_id = str(engine.get("llm_id") or "").strip()
        if not source_engine_id:
            raise ProviderImportError("retell_response_engine_missing_llm_id")
        source_engine = request(
            "GET", f"{base}/get-retell-llm/{source_engine_id}", api_key, None
        )
        _validate_retell_custom_tools(source_engine, environment_tools)
        engine_create = _rewire_retell_tools(
            _without(source_engine, _RETELL_ENGINE_READ_ONLY_FIELDS),
            context.tool_base_url,
        )
        created_engine = request(
            "POST", f"{base}/create-retell-llm", api_key, engine_create
        )
        cloned_engine_id = str(created_engine.get("llm_id") or "").strip()
        if not cloned_engine_id:
            raise ProviderImportError("retell_llm_create_missing_id")
        engine_resource = ProviderResource(
            kind="retell_llm", id=cloned_engine_id, owned=True
        )
        cloned_engine = {"type": "retell-llm", "llm_id": cloned_engine_id}
        delete_engine_url = f"{base}/delete-retell-llm/{cloned_engine_id}"
    elif engine_type == "conversation-flow":
        source_engine_id = str(engine.get("conversation_flow_id") or "").strip()
        if not source_engine_id:
            raise ProviderImportError(
                "retell_response_engine_missing_conversation_flow_id"
            )
        source_engine = request(
            "GET",
            f"{base}/get-conversation-flow/{source_engine_id}",
            api_key,
            None,
        )
        _validate_retell_custom_tools(source_engine, environment_tools)
        engine_create = _rewire_retell_tools(
            _without(source_engine, _RETELL_ENGINE_READ_ONLY_FIELDS),
            context.tool_base_url,
        )
        created_engine = request(
            "POST", f"{base}/create-conversation-flow", api_key, engine_create
        )
        cloned_engine_id = str(created_engine.get("conversation_flow_id") or "").strip()
        if not cloned_engine_id:
            raise ProviderImportError("retell_conversation_flow_create_missing_id")
        engine_resource = ProviderResource(
            kind="conversation_flow", id=cloned_engine_id, owned=True
        )
        cloned_engine = {
            "type": "conversation-flow",
            "conversation_flow_id": cloned_engine_id,
        }
        delete_engine_url = f"{base}/delete-conversation-flow/{cloned_engine_id}"
    else:
        raise ProviderImportError(
            "retell_response_engine_unsupported: provider_import supports "
            "retell-llm and conversation-flow"
        )

    agent_create = _without(
        original,
        _RETELL_AGENT_READ_ONLY_FIELDS,
    )
    agent_create["agent_name"] = context.provider_resource_prefix
    agent_create["response_engine"] = cloned_engine
    agent_create["webhook_url"] = context.event_url
    create_path = (
        "create-chat-agent" if spec.target_modality == "chat" else "create-agent"
    )
    try:
        created_agent = request("POST", f"{base}/{create_path}", api_key, agent_create)
    except ProviderImportError:
        try:
            request("DELETE", delete_engine_url, api_key, None)
        except ProviderImportError:
            pass
        raise
    target_id = str(created_agent.get("agent_id") or "").strip()
    if not target_id:
        try:
            request("DELETE", delete_engine_url, api_key, None)
        except ProviderImportError:
            pass
        raise ProviderImportError("retell_agent_create_missing_id")
    return _CloneResult(
        target_id=target_id,
        target_kind="chat_agent" if spec.target_modality == "chat" else "voice_agent",
        resources=(
            engine_resource,
            ProviderResource(
                kind="chat_agent" if spec.target_modality == "chat" else "voice_agent",
                id=target_id,
                owned=True,
            ),
        ),
        metadata={
            "source_target_id": spec.source_target_id,
            "source_response_engine_id": source_engine_id,
            "source_response_engine_type": engine_type,
            "target_modality": spec.target_modality,
            "clone_kind": "provider_import",
        },
    )


def clone_provider_target(
    spec: ProviderImportSpec,
    *,
    context: ProviderContext,
    api_key: str,
    request: JsonRequest = _request_json,
) -> ProviderProvisionReceipt:
    if not api_key:
        raise ProviderImportError(f"{spec.type.value}_api_key_missing")
    result = (
        _clone_vapi(spec, context, api_key, request)
        if spec.type is ProviderType.VAPI
        else _clone_retell(spec, context, api_key, request)
    )
    return ProviderProvisionReceipt(
        schema_version="1",
        provider=spec.type,
        attempt_id=context.attempt_id,
        world_id=context.world_id,
        target=ProviderTarget(kind=result.target_kind, id=result.target_id),
        resources=list(result.resources),
        cleanup=ProviderCleanupReceipt(idempotency_key=context.idempotency_key),
        metadata=result.metadata,
    )


def destroy_imported_target(
    spec: ProviderImportSpec,
    *,
    receipt: ProviderProvisionReceipt,
    api_key: str,
    request: JsonRequest = _request_json,
) -> None:
    base = (
        spec.api_base_url
        or (
            "https://api.vapi.ai"
            if spec.type is ProviderType.VAPI
            else "https://api.retellai.com"
        )
    ).rstrip("/")
    # Reverse dependency order: target first, then tools/response engine.
    for resource in reversed(receipt.resources):
        if not resource.owned:
            continue
        if spec.type is ProviderType.VAPI:
            path = "assistant" if resource.kind == "assistant" else "tool"
            try:
                request("DELETE", f"{base}/{path}/{resource.id}", api_key, None)
            except ProviderImportError as exc:
                if "HTTP 404" not in str(exc):
                    raise
        elif resource.kind in {"voice_agent", "chat_agent"}:
            delete_path = (
                "delete-chat-agent" if resource.kind == "chat_agent" else "delete-agent"
            )
            try:
                request("DELETE", f"{base}/{delete_path}/{resource.id}", api_key, None)
            except ProviderImportError as exc:
                if "HTTP 404" not in str(exc):
                    raise
        elif resource.kind == "retell_llm":
            try:
                request(
                    "DELETE", f"{base}/delete-retell-llm/{resource.id}", api_key, None
                )
            except ProviderImportError as exc:
                if "HTTP 404" not in str(exc):
                    raise
        elif resource.kind == "conversation_flow":
            try:
                request(
                    "DELETE",
                    f"{base}/delete-conversation-flow/{resource.id}",
                    api_key,
                    None,
                )
            except ProviderImportError as exc:
                if "HTTP 404" not in str(exc):
                    raise


__all__ = [
    "ProviderImportError",
    "ProviderImportSpec",
    "clone_provider_target",
    "destroy_imported_target",
    "inspect_provider_target",
]
