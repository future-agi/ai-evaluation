from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse

from fi.simulate.agent.wrapper import (
    AgentInput,
    AgentResponse,
    SimulationArtifact,
    SimulationEvent,
)
from fi.simulate.agent.wrapper import AgentWrapper


class HTTPAgentWrapper(AgentWrapper):
    """HTTP/OpenAI-compatible target adapter for external agent simulation."""

    def __init__(
        self,
        *,
        endpoint: str,
        protocol: str = "fi.alk",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: float = 30.0,
        include_tools: bool = True,
        system_prompt: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        self.endpoint = endpoint
        self.protocol = _normalize_protocol(protocol)
        self.model = model
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.headers = {str(k): str(v) for k, v in dict(headers or {}).items()}
        self.timeout = float(timeout)
        self.include_tools = bool(include_tools)
        self.system_prompt = system_prompt
        self.metadata = dict(metadata or {})

    async def call(self, input: AgentInput) -> AgentResponse:
        started = time.time()
        request_payload = self._request_payload(input)
        headers = self._request_headers()
        status_code = 0
        response_payload: dict[str, Any] = {}
        error: Optional[str] = None
        try:
            status_code, response_payload = await asyncio.to_thread(
                self._post_json,
                request_payload,
                headers,
            )
            if status_code >= 400:
                error = _response_error_text(response_payload) or (
                    f"HTTP target returned status {status_code}"
                )
            response = self._agent_response_from_payload(response_payload)
        except Exception as exc:
            error = str(exc)
            response = AgentResponse(content=f"HTTP target failed: {exc}")

        latency_ms = round((time.time() - started) * 1000, 4)
        trace = {
            "kind": "external_agent_http_trace",
            "protocol": self.protocol,
            "endpoint": _redacted_endpoint(self.endpoint),
            "endpoint_host": urlparse(self.endpoint).netloc,
            "model": self.model,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "request_message_count": len(input.messages),
            "request_tool_count": len(input.tools) if self.include_tools else 0,
            "response_tool_call_count": len(response.tool_calls or []),
            "success": error is None and 200 <= status_code < 300,
            "request_header_names": sorted(headers),
            "auth": {
                "mode": "bearer" if self._resolved_api_key() else "none",
                "api_key_env": self.api_key_env,
                "redacted": bool(self._resolved_api_key()),
            },
            "error": error,
            **self.metadata,
        }
        response.events.append(
            SimulationEvent(
                type="external_agent",
                name="external_agent_http_call",
                payload=trace,
            )
        )
        response.artifacts.append(
            SimulationArtifact(
                type="trace",
                role="agent",
                data=trace,
                metadata={"kind": "external_agent_http_trace"},
            )
        )
        state = dict(response.state or {})
        state["external_agent"] = trace
        state["external_agent_trace"] = trace
        response.state = state
        metadata = dict(response.metadata or {})
        metadata["external_agent"] = trace
        metadata["external_agent_trace"] = trace
        response.metadata = metadata
        return response

    def _request_payload(self, input: AgentInput) -> dict[str, Any]:
        messages = _messages_for_protocol(input.messages, self.protocol)
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}, *messages]
        if self.protocol == "openai_chat":
            payload: dict[str, Any] = {
                "model": self.model or "agent-learning-target",
                "messages": messages,
            }
            if self.include_tools and input.tools:
                payload["tools"] = [_openai_tool_spec(tool) for tool in input.tools]
                payload["tool_choice"] = "auto"
            return payload
        return {
            "thread_id": input.thread_id,
            "execution_id": input.execution_id,
            "turn_index": input.turn_index,
            "scenario_name": input.scenario_name,
            "persona": input.persona,
            "situation": input.situation,
            "expected_outcome": input.expected_outcome,
            "messages": messages,
            "new_message": input.new_message,
            "tools": list(input.tools) if self.include_tools else [],
            "metadata": input.metadata,
        }

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.headers}
        api_key = self._resolved_api_key()
        if api_key and not any(key.lower() == "authorization" for key in headers):
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _resolved_api_key(self) -> str:
        if self.api_key not in (None, ""):
            return str(self.api_key)
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    def _post_json(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> tuple[int, dict[str, Any]]:
        body = json.dumps(payload, default=str).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", 200))
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            text = exc.read().decode("utf-8")
        if not text:
            return status, {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"HTTP target returned non-JSON response: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("HTTP target response must be a JSON object")
        return status, parsed

    def _agent_response_from_payload(self, payload: Mapping[str, Any]) -> AgentResponse:
        if self.protocol == "openai_chat":
            message = _openai_message(payload)
            return AgentResponse(
                content=_content_text(message.get("content")),
                tool_calls=_openai_tool_calls(message.get("tool_calls")),
                metadata={
                    "finish_reason": _openai_finish_reason(payload),
                    "usage": dict(payload.get("usage") or {}),
                },
            )
        return AgentResponse(
            content=_content_text(payload.get("content") or payload.get("message")),
            tool_calls=_tool_call_list(payload.get("tool_calls")),
            tool_responses=_tool_response_list(payload.get("tool_responses")),
            artifacts=_artifact_list(payload.get("artifacts")),
            events=_event_list(payload.get("events")),
            memory_updates=_optional_mapping(payload.get("memory_updates")),
            state=_optional_mapping(payload.get("state")),
            metadata=_optional_mapping(payload.get("metadata")),
        )


def _normalize_protocol(value: str) -> str:
    protocol = str(value or "fi.alk").lower().replace("-", "_")
    aliases = {
        "openai": "openai_chat",
        "openai_compatible": "openai_chat",
        "chat_completions": "openai_chat",
        "agent_learning_http": "fi.alk",
        "http": "fi.alk",
    }
    protocol = aliases.get(protocol, protocol)
    if protocol not in {"fi.alk", "openai_chat"}:
        raise ValueError("protocol must be one of: fi.alk, openai_chat")
    return protocol


def _messages_for_protocol(
    messages: Sequence[Mapping[str, Any]], protocol: str
) -> list[dict[str, Any]]:
    """Encode canonical ALK history for the selected external-agent protocol.

    ALK keeps tool calls structured internally. OpenAI-compatible endpoints require that array
    unchanged, while the FutureAGI callback ``AgentInput`` schema represents historical
    ``tool_calls`` as a JSON string. Normalizing here keeps the conversation runner generic and
    prevents a retry from accidentally executing a side-effecting tool twice.
    """
    normalized = [dict(message) for message in messages]
    if protocol != "fi.alk":
        return normalized
    for message in normalized:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, Sequence) and not isinstance(
            tool_calls, (str, bytes)
        ):
            message["tool_calls"] = json.dumps(
                list(tool_calls), separators=(",", ":"), default=str
            )
    return normalized


def _openai_tool_spec(tool: Mapping[str, Any]) -> dict[str, Any]:
    # Accept both the flat SDK tool shape ({name, description, parameters}) and
    # the OpenAI-nested shape ({"type": "function", "function": {...}}). Without
    # reading the nested ``function`` block, a nested spec loses its name and the
    # model is handed a tool literally called "tool" — so it can never call the
    # real tool and the environment's mock never matches.
    fn = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
    name = str(
        tool.get("name")
        or fn.get("name")
        or tool.get("tool")
        or tool.get("id")
        or "tool"
    )
    parameters = tool.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = fn.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}}
    description = tool.get("description") or fn.get("description") or f"Tool {name}"
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(description),
            "parameters": dict(parameters),
        },
    }


def _openai_message(payload: Mapping[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        if choices:
            choice = choices[0]
            if isinstance(choice, Mapping):
                message = choice.get("message")
                if isinstance(message, Mapping):
                    return dict(message)
    message = payload.get("message")
    return dict(message) if isinstance(message, Mapping) else dict(payload)


def _openai_finish_reason(payload: Mapping[str, Any]) -> Optional[str]:
    choices = payload.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        if choices and isinstance(choices[0], Mapping):
            value = choices[0].get("finish_reason")
            return str(value) if value is not None else None
    return None


def _openai_tool_calls(value: Any) -> list[dict[str, Any]]:
    calls = _tool_call_list(value)
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(calls, start=1):
        function = call.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            name = call.get("name") or call.get("tool")
            arguments = call.get("arguments", call.get("args", {}))
        normalized.append(
            {
                "id": str(call.get("id") or f"call_{index}"),
                "type": str(call.get("type") or "function"),
                "function": {
                    "name": str(name or ""),
                    "arguments": (
                        arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments or {}, default=str)
                    ),
                },
            }
        )
    return normalized


def _tool_call_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _tool_response_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _artifact_list(value: Any) -> list[SimulationArtifact]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    artifacts: list[SimulationArtifact] = []
    for item in value:
        if isinstance(item, SimulationArtifact):
            artifacts.append(item)
        elif isinstance(item, Mapping):
            artifacts.append(SimulationArtifact(**dict(item)))
    return artifacts


def _event_list(value: Any) -> list[SimulationEvent]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    events: list[SimulationEvent] = []
    for item in value:
        if isinstance(item, SimulationEvent):
            events.append(item)
        elif isinstance(item, Mapping):
            events.append(SimulationEvent(**dict(item)))
    return events


def _optional_mapping(value: Any) -> Optional[dict[str, Any]]:
    return dict(value) if isinstance(value, Mapping) else None


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content") or item.get("refusal")
                if text not in (None, ""):
                    parts.append(str(text))
            elif item not in (None, ""):
                parts.append(str(item))
        return "\n".join(parts)
    return "" if value is None else str(value)


def _response_error_text(payload: Mapping[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, Mapping):
        return _content_text(error.get("message") or error.get("detail") or error)
    if error not in (None, ""):
        return _content_text(error)
    for key in ("detail", "message", "status"):
        if payload.get(key) not in (None, ""):
            return _content_text(payload.get(key))
    return ""


def _redacted_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if not parsed.query:
        return endpoint
    return parsed._replace(query="<redacted>").geturl()
