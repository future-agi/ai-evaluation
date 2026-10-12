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

from .request_binding import SECRET_HEADER
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
    admission: Literal["legacy", "per_tool"] = "legacy"
    request_binding: bool = False

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


class ToolAdmission(BaseModel):
    """How a clone treats one provider tool (`environment-releases.md` §6.3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    provider_type: str
    action: Literal["routed", "replaced", "kept", "disabled"]
    reason: str


_VAPI_KEPT = frozenset({"endCall", "dtmf", "voicemail", "query"})
_VAPI_BUILTIN_NAMES = {"transferCall": "transfer_call", "sms": "send_sms"}
# Typed tools that may carry a `function` block but still act outside the call.
_VAPI_NEVER_REPLACED = frozenset(
    {"handoff", "mcp", "code", "bash", "computer", "textEditor"}
)
_RETELL_REPLACED = frozenset(
    {"transfer_call", "send_sms", "check_availability_cal", "book_appointment_cal"}
)
_RETELL_KEPT = frozenset({"end_call", "press_digit", "extract_dynamic_variable"})
_RETELL_UNCONTROLLED_NODES = frozenset(
    {"transfer_call", "sms", "code", "agent_swap", "bridge_transfer", "mcp"}
)
_RETELL_TOOL_LISTS = ("general_tools", "tools")
# Assistant fields that can still act outside the call once every tool is admitted: hooks run
# transfers/tools on events, the legacy forwarding number transfers, and the legacy server URL
# would receive tool calls with the customer's secret.
_VAPI_PER_TOOL_DROPPED = frozenset(
    {"hooks", "forwardingPhoneNumber", "serverUrl", "serverUrlSecret"}
)
_EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


def _vapi_tool_admission(tool: Mapping[str, Any]) -> ToolAdmission:
    kind = str(tool.get("type") or "")
    function = tool.get("function")
    function_name = (
        str(function.get("name") or "").strip() if isinstance(function, Mapping) else ""
    )
    if kind == "function":
        return ToolAdmission(
            name=function_name,
            provider_type=kind,
            action="routed",
            reason="custom function served by the world",
        )
    if kind == "apiRequest":
        return ToolAdmission(
            name=str(tool.get("name") or "").strip(),
            provider_type=kind,
            action="replaced",
            reason="HTTP request replaced by a world-served function",
        )
    if kind in _VAPI_BUILTIN_NAMES:
        return ToolAdmission(
            name=_VAPI_BUILTIN_NAMES[kind],
            provider_type=kind,
            action="replaced",
            reason="provider action replaced by a world-served function",
        )
    if kind in _VAPI_KEPT:
        return ToolAdmission(
            name=function_name or kind,
            provider_type=kind,
            action="kept",
            reason="no effect outside the call",
        )
    if function_name and kind not in _VAPI_NEVER_REPLACED:
        return ToolAdmission(
            name=function_name,
            provider_type=kind,
            action="replaced",
            reason="integration replaced by its function definition",
        )
    return ToolAdmission(
        name=function_name or str(tool.get("name") or "").strip() or kind,
        provider_type=kind,
        action="disabled",
        reason=f"{kind or 'untyped'} tool acts outside the call",
    )


def _retell_tool_admission(tool: Mapping[str, Any]) -> ToolAdmission:
    kind = str(tool.get("type") or "")
    name = str(tool.get("name") or "").strip()
    if kind == "custom":
        action, reason = "routed", "custom function served by the world"
    elif kind in _RETELL_REPLACED:
        action, reason = (
            "replaced",
            "provider action replaced by a world-served function",
        )
    elif kind in _RETELL_KEPT:
        action, reason = "kept", "no effect outside the call"
    else:
        action, reason = "disabled", f"{kind or 'untyped'} tool acts outside the call"
    return ToolAdmission(
        name=name or kind, provider_type=kind, action=action, reason=reason
    )


def _retell_tool_lists(value: Any, visit: Callable[[list[Any]], list[Any]]) -> Any:
    """Apply `visit` to every tool list wherever Retell nests one (states, flow, components)."""
    if isinstance(value, list):
        return [_retell_tool_lists(item, visit) for item in value]
    if not isinstance(value, Mapping):
        return value
    return {
        key: (
            visit(item)
            if key in _RETELL_TOOL_LISTS and isinstance(item, list)
            else _retell_tool_lists(item, visit)
        )
        for key, item in value.items()
    }


def _retell_uncontrolled_nodes(value: Any, disabled_tool_ids: set[str]) -> list[str]:
    """Flow nodes that act outside the call themselves, or call a tool the clone removes."""
    found: list[str] = []

    def walk(item: Any, in_nodes: bool) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child, in_nodes)
            return
        if not isinstance(item, Mapping):
            return
        if in_nodes:
            kind = str(item.get("type") or "")
            if kind in _RETELL_UNCONTROLLED_NODES or (
                kind == "function"
                and str(item.get("tool_id") or "") in disabled_tool_ids
            ):
                found.append(
                    f"{kind} node {str(item.get('name') or item.get('id') or '?')!r}"
                )
        for key, child in item.items():
            walk(child, key == "nodes")

    walk(value, False)
    return found


def _require_controlled_flow(flow: Any) -> None:
    disabled_ids: set[str] = set()

    def collect(tools: list[Any]) -> list[Any]:
        for tool in tools:
            if (
                isinstance(tool, Mapping)
                and _retell_tool_admission(tool).action == "disabled"
            ):
                disabled_ids.add(str(tool.get("tool_id") or ""))
        return tools

    _retell_tool_lists(flow, collect)
    disabled_ids.discard("")
    nodes = _retell_uncontrolled_nodes(flow, disabled_ids)
    if nodes:
        raise ProviderImportError("provider_tool_uncontrolled: " + ", ".join(nodes))


def _profile_tools(
    provider: ProviderType, target: Mapping[str, Any]
) -> list[tuple[Mapping[str, Any], ToolAdmission]]:
    if provider is ProviderType.VAPI:
        model = target.get("model")
        inline = model.get("tools") if isinstance(model, Mapping) else None
        return [
            (tool, _vapi_tool_admission(tool))
            for tool in [*(inline or []), *(target.get("reusable_tools") or [])]
            if isinstance(tool, Mapping)
        ]
    found: list[tuple[Mapping[str, Any], ToolAdmission]] = []

    def visit(tools: list[Any]) -> list[Any]:
        found.extend(
            (tool, _retell_tool_admission(tool))
            for tool in tools
            if isinstance(tool, Mapping)
        )
        return tools

    _retell_tool_lists(
        {
            key: target.get(key)
            for key in ("general_tools", "states", "conversation_flow")
        },
        visit,
    )
    if target.get("conversation_flow") is not None:
        _require_controlled_flow(target["conversation_flow"])
    return found


def admit_provider_tools(
    provider: ProviderType | str, target: Mapping[str, Any]
) -> list[ToolAdmission]:
    """Per-tool admission over an `inspect_provider_target` profile (§6.3).

    Pure: the clone applies exactly this decision, so Adaptation can show it beforehand. A
    conversation flow with a node that acts outside the call raises `provider_tool_uncontrolled`
    here too, because no clone of it can be made safe.
    """
    return list(
        dict.fromkeys(
            admission for _, admission in _profile_tools(ProviderType(provider), target)
        )
    )


def admitted_tool_definitions(
    provider: ProviderType | str, target: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """`{name, description, parameters, admission}` of every routed or replaced tool, exactly as
    the per-tool clone exposes it to the agent, in `admit_provider_tools` order."""
    provider = ProviderType(provider)
    definitions: dict[str, dict[str, Any]] = {}
    for tool, admission in _profile_tools(provider, target):
        if (
            admission.action not in {"routed", "replaced"}
            or admission.name in definitions
        ):
            continue
        if provider is ProviderType.VAPI:
            admitted = _vapi_admitted_tool(
                tool, admission, tool_base_url="", headers={}
            )
            function = (admitted or {}).get("function") or {}
            description, parameters = (
                function.get("description"),
                function.get("parameters"),
            )
        else:
            admitted = (
                _retell_admitted_tool(tool, admission, tool_base_url="", headers={})
                or {}
            )
            description, parameters = (
                admitted.get("description"),
                admitted.get("parameters"),
            )
        definitions[admission.name] = {
            "name": admission.name,
            "description": str(description or ""),
            "parameters": dict(parameters)
            if isinstance(parameters, Mapping)
            else dict(_EMPTY_PARAMETERS),
            "admission": admission.action,
        }
    return list(definitions.values())


def _admission_metadata(
    spec: ProviderImportSpec, admissions: list[ToolAdmission]
) -> dict[str, Any]:
    if spec.admission != "per_tool":
        return {}
    return {
        "admission": "per_tool",
        "tool_admissions": [
            admission.model_dump() for admission in dict.fromkeys(admissions)
        ],
    }


def _admitted_tool_headers(
    spec: ProviderImportSpec, context: ProviderContext
) -> dict[str, str]:
    """Headers on every admitted tool; the customer's own headers never reach the world."""
    if not spec.request_binding:
        return {}
    if not context.tool_secret:
        raise ProviderImportError("provider_tool_secret_missing")
    return {SECRET_HEADER: context.tool_secret}


def _routed_url(tool_base_url: str, name: str) -> str:
    return tool_base_url.rstrip("/") + "/" + name


def _require_admitted_name(
    admission: ToolAdmission, provider: str, environment_tools: set[str]
) -> None:
    if admission.action in {"routed", "replaced"}:
        _require_environment_tool(
            provider=provider,
            name=admission.name or None,
            available=environment_tools,
        )


def _transfer_parameters(destinations: list[str], description: str) -> dict[str, Any]:
    # Neither provider documents the arguments its model passes to a built-in transfer; the
    # tool's own configured destinations are the only schema that is not a guess.
    destination: dict[str, Any] = {"type": "string", "description": description}
    if destinations:
        destination["enum"] = destinations
    return {
        "type": "object",
        "properties": {"destination": destination},
        "required": ["destination"],
    }


def _vapi_replacement_parameters(tool: Mapping[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if isinstance(function, Mapping) and isinstance(
        function.get("parameters"), Mapping
    ):
        return dict(function["parameters"])
    kind = str(tool.get("type") or "")
    if kind == "apiRequest" and isinstance(tool.get("body"), Mapping):
        return dict(tool["body"])
    if kind == "transferCall":
        destinations = [
            str(item.get("number") or item.get("sipUri") or item.get("assistantName"))
            for item in tool.get("destinations") or []
            if isinstance(item, Mapping)
            and (item.get("number") or item.get("sipUri") or item.get("assistantName"))
        ]
        return _transfer_parameters(destinations, "Where to transfer the call.")
    return dict(_EMPTY_PARAMETERS)


def _vapi_admitted_tool(
    tool: Mapping[str, Any],
    admission: ToolAdmission,
    *,
    tool_base_url: str,
    headers: Mapping[str, str],
) -> dict[str, Any] | None:
    if admission.action == "disabled":
        return None
    if admission.action == "kept":
        return dict(tool)
    source_server = tool.get("server")
    server: dict[str, Any] = {
        "url": _routed_url(tool_base_url, admission.name),
        "headers": dict(headers),
    }
    if isinstance(source_server, Mapping) and source_server.get("timeoutSeconds"):
        server["timeoutSeconds"] = source_server["timeoutSeconds"]
    if admission.action == "routed":
        return {**dict(tool), "server": server}
    function = tool.get("function")
    description = str(
        (function.get("description") if isinstance(function, Mapping) else None)
        or tool.get("description")
        or ""
    )
    replaced: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": admission.name,
            "description": description,
            "parameters": _vapi_replacement_parameters(tool),
        },
        "server": server,
    }
    if isinstance(tool.get("messages"), list):
        replaced["messages"] = tool["messages"]
    return replaced


def _retell_replacement_parameters(tool: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(tool.get("parameters"), Mapping):
        return dict(tool["parameters"])
    kind = str(tool.get("type") or "")
    if kind == "transfer_call":
        target = tool.get("transfer_destination")
        target = target if isinstance(target, Mapping) else {}
        number = str(target.get("number") or "").strip()
        return _transfer_parameters(
            [number] if target.get("type") == "predefined" and number else [],
            str(target.get("prompt") or "Where to transfer the call."),
        )
    if kind == "send_sms":
        # Retell's SendSMSTool exposes no model arguments; an inferred message is the one value
        # the model chooses (create-retell-llm, SmsContent).
        content = tool.get("sms_content")
        if isinstance(content, Mapping) and content.get("type") == "inferred":
            return {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": str(content.get("prompt") or "The message."),
                    }
                },
                "required": ["content"],
            }
    return dict(_EMPTY_PARAMETERS)


def _retell_admitted_tool(
    tool: Mapping[str, Any],
    admission: ToolAdmission,
    *,
    tool_base_url: str,
    headers: Mapping[str, str],
) -> dict[str, Any] | None:
    if admission.action == "disabled":
        return None
    if admission.action == "kept":
        return dict(tool)
    routed = {
        "url": _routed_url(tool_base_url, admission.name),
        "method": "POST",
        "headers": dict(headers),
        "args_at_root": False,
    }
    if admission.action == "routed":
        return {**dict(tool), **routed}
    replaced: dict[str, Any] = {
        "type": "custom",
        "name": admission.name,
        "description": str(tool.get("description") or ""),
        "parameters": _retell_replacement_parameters(tool),
        "speak_during_execution": bool(tool.get("speak_during_execution", False)),
        "speak_after_execution": True,
        **routed,
    }
    for key in ("tool_id", "execution_message_description", "execution_message_type"):
        if tool.get(key) is not None:
            replaced[key] = tool[key]
    return replaced


def _admit_retell_engine(
    engine: Any,
    *,
    environment_tools: set[str],
    tool_base_url: str,
    headers: Mapping[str, str],
    admissions: list[ToolAdmission],
) -> Any:
    def visit(tools: list[Any]) -> list[Any]:
        kept: list[Any] = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                kept.append(tool)
                continue
            admission = _retell_tool_admission(tool)
            _require_admitted_name(admission, "retell", environment_tools)
            admissions.append(admission)
            admitted = _retell_admitted_tool(
                tool, admission, tool_base_url=tool_base_url, headers=headers
            )
            if admitted is not None:
                kept.append(admitted)
        return kept

    return _retell_tool_lists(engine, visit)


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
    per_tool = spec.admission == "per_tool"
    headers = _admitted_tool_headers(spec, context) if per_tool else {}
    admissions: list[ToolAdmission] = []
    if per_tool:
        assistant = _without(assistant, set(_VAPI_PER_TOOL_DROPPED))

    def admit(item: Any) -> Any:
        if not isinstance(item, Mapping):
            return item
        if per_tool:
            admission = _vapi_tool_admission(item)
            _require_admitted_name(admission, "vapi", environment_tools)
            admissions.append(admission)
            return _vapi_admitted_tool(
                item, admission, tool_base_url=context.tool_base_url, headers=headers
            )
        if str(item.get("type") or "") in {"function", "apiRequest"}:
            _require_environment_tool(
                provider="vapi",
                name=_vapi_custom_tool_name(item),
                available=environment_tools,
            )
        return _rewire_vapi_tool(item, context.tool_base_url)

    try:
        model = assistant.get("model")
        if isinstance(model, dict):
            model = dict(model)
            inline = model.get("tools")
            if isinstance(inline, list):
                model["tools"] = [
                    tool for tool in map(admit, inline) if tool is not None
                ]
            tool_ids = model.get("toolIds")
            if isinstance(tool_ids, list):
                cloned_ids: list[str] = []
                for source_id in tool_ids:
                    source = request("GET", f"{base}/tool/{source_id}", api_key, None)
                    create = admit(
                        _without(source, {"id", "orgId", "createdAt", "updatedAt"})
                    )
                    if create is None:
                        continue
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
            **_admission_metadata(spec, admissions),
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
    per_tool = spec.admission == "per_tool"
    headers = _admitted_tool_headers(spec, context) if per_tool else {}
    admissions: list[ToolAdmission] = []

    def engine_body(source_engine: dict[str, Any], *, flow: bool) -> Any:
        if not per_tool:
            _validate_retell_custom_tools(source_engine, environment_tools)
            return _rewire_retell_tools(
                _without(source_engine, _RETELL_ENGINE_READ_ONLY_FIELDS),
                context.tool_base_url,
            )
        if flow:
            _require_controlled_flow(source_engine)
        return _admit_retell_engine(
            _without(source_engine, _RETELL_ENGINE_READ_ONLY_FIELDS),
            environment_tools=environment_tools,
            tool_base_url=context.tool_base_url,
            headers=headers,
            admissions=admissions,
        )

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
        engine_create = engine_body(source_engine, flow=False)
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
        engine_create = engine_body(source_engine, flow=True)
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
            **_admission_metadata(spec, admissions),
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
    "ToolAdmission",
    "admit_provider_tools",
    "admitted_tool_definitions",
    "clone_provider_target",
    "destroy_imported_target",
    "inspect_provider_target",
]
