from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote, urlparse

from fi._diagnostics import safe_endpoint, sanitize_diagnostic

from fi.simulate.agent.wrapper import (
    AgentInput,
    AgentResponse,
    SimulationArtifact,
    SimulationEvent,
)
from fi.simulate.agent.wrapper import AgentWrapper


class HTTPCallFailure(RuntimeError):
    def __init__(self, trace: Mapping[str, Any]) -> None:
        super().__init__(str(trace.get("error") or "HTTP target failed"))
        self.trace = dict(trace)


class _ResponseFailure(RuntimeError):
    def __init__(self, stage: str, status: int, message: str) -> None:
        super().__init__(message)
        self.stage, self.status = stage, status


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
        total_timeout: Optional[float] = None,
        include_tools: bool = True,
        system_prompt: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        request_template: Any = None,
        response_path: str = "",
        setup_requests: Optional[Sequence[Mapping[str, Any]]] = None,
        failure_rule: Optional[Mapping[str, Any]] = None,
        secret_values: Optional[Mapping[str, str]] = None,
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
        self.total_timeout = total_timeout
        self._deadline: float | None = None
        self.include_tools = bool(include_tools)
        self.system_prompt = system_prompt
        self.metadata = dict(metadata or {})
        self.request_template = request_template
        self.response_path = str(response_path or "").strip()
        self.setup_requests = [dict(item) for item in (setup_requests or [])]
        self.failure_rule = dict(failure_rule) if failure_rule else None
        self.secret_values = dict(secret_values) if secret_values is not None else None
        self._response_text = ""
        self._response_status = 0
        self._setup_complete = False
        self._setup_values: dict[str, Any] = {}

    async def call(self, input: AgentInput) -> AgentResponse:
        started = time.time()
        self._deadline = (
            time.monotonic() + self.total_timeout
            if self.total_timeout is not None
            else None
        )
        headers: dict[str, str] = {}
        status_code = 0
        error: Optional[str] = None
        stage = "setup"
        failure_stage = None
        network_error = False
        response = AgentResponse(content="")
        self._response_text = ""
        self._response_status = 0
        secrets = None
        try:
            headers = self._request_headers()
            await self._run_request(self._ensure_setup, input, headers)
            self._response_text = ""
            self._response_status = 0
            stage = "transport"
            request_payload = self._request_payload(input)
            endpoint = _render_endpoint_path(
                self.endpoint, _template_context(input, extra=self._setup_values)
            )
            status_code, response_payload = await self._run_request(
                self._post_json, request_payload, headers, endpoint=endpoint
            )
            if not 200 <= status_code < 300:
                raise _ResponseFailure(
                    "http",
                    status_code,
                    _response_error_text(response_payload)
                    or f"HTTP target returned status {status_code}",
                )
            if self.failure_rule:
                stage = "application"
                status = _select_response_value(
                    response_payload, self.failure_rule["status_path"]
                )
                if any(
                    type(status) is type(value) and status == value
                    for value in self.failure_rule["failure_values"]
                ):
                    message_path = self.failure_rule.get("message_path", "")
                    message = (
                        _content_text(
                            _select_response_value(response_payload, message_path)
                        )
                        if message_path
                        else "Declared application failure"
                    )
                    raise _ResponseFailure("application", status_code, message)
            stage = "extract"
            response = self._agent_response_from_payload(response_payload)
        except Exception as exc:
            network_error = isinstance(
                exc, (urllib.error.URLError, TimeoutError, ConnectionError)
            )
            if self.secret_values is not None:
                secrets = {
                    str(index): value
                    for index, value in enumerate(
                        (
                            *self.secret_values.values(),
                            self._resolved_api_key(),
                            *self.headers.values(),
                            json.dumps(self._setup_values, default=str),
                            *(
                                _encoded_path_segment(str(value))
                                for value in self._setup_values.values()
                            ),
                        )
                    )
                }
            status_code = getattr(exc, "status", self._response_status or status_code)
            failure_stage = (
                "setup" if stage == "setup" else getattr(exc, "stage", stage)
            )
            error = sanitize_diagnostic(str(exc), secrets)

        latency_ms = round((time.time() - started) * 1000, 4)
        safe_metadata = self.metadata
        if error:
            try:
                safe_metadata = json.loads(
                    sanitize_diagnostic(json.dumps(self.metadata, default=str), secrets)
                )
            except ValueError:
                safe_metadata = {}
        trace = {
            **safe_metadata,
            "kind": "external_agent_http_trace",
            "protocol": self.protocol,
            "endpoint": sanitize_diagnostic(_redacted_endpoint(self.endpoint), secrets)
            if error
            else _redacted_endpoint(self.endpoint),
            "endpoint_host": urlparse(self.endpoint).hostname,
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
            "failure_stage": failure_stage,
            "network_error": network_error,
            "response_excerpt": sanitize_diagnostic(self._response_text, secrets)
            if error
            else "",
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

    async def _run_request(self, function, *args, **kwargs):
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # A cancelled thread can still mutate the target. Drain before world reset.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    def _remaining_timeout(self):
        if self._deadline is None:
            return self.timeout
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("HTTP call deadline exceeded")
        return min(self.timeout, remaining)

    def _request_payload(self, input: AgentInput) -> dict[str, Any]:
        if self.protocol == "json_template":
            rendered = _render_template(
                self.request_template,
                _template_context(input, extra=self._setup_values),
            )
            if not isinstance(rendered, Mapping):
                raise ValueError("json_template request must render to a JSON object")
            return dict(rendered)
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

    def _ensure_setup(
        self,
        input: AgentInput,
        headers: Mapping[str, str],
    ) -> None:
        if self._setup_complete or not self.setup_requests:
            return
        context = _template_context(input, extra=self._setup_values)
        parsed = urlparse(self.endpoint)
        origin = f"{parsed.scheme}://{parsed.netloc}/"
        for item in self.setup_requests:
            endpoint = _render_endpoint_path(
                origin + str(item.get("path") or "").lstrip("/"), context
            )
            method = str(item.get("method") or "POST").upper()
            body = _render_template(item.get("body_template", {}), context)
            accepted = {
                int(value)
                for value in item.get("accepted_statuses", [200, 201, 204, 409])
            }
            status, payload = self._request_json(
                endpoint,
                method=method,
                payload=body,
                headers=headers,
            )
            if status not in accepted:
                detail = _response_error_text(payload)
                raise RuntimeError(
                    f"HTTP setup request returned status {status}"
                    + (f": {detail}" if detail else "")
                )
            captures = item.get("capture") or {}
            if not isinstance(captures, Mapping):
                raise ValueError("HTTP setup capture must be an object")
            for raw_name, raw_path in captures.items():
                name = str(raw_name)
                self._setup_values[name] = _select_response_value(
                    payload, str(raw_path)
                )
                context[name] = self._setup_values[name]
        self._setup_complete = True

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
        *,
        endpoint: str | None = None,
    ) -> tuple[int, Any]:
        return self._request_json(
            endpoint if endpoint is not None else self.endpoint,
            method="POST",
            payload=payload,
            headers=headers,
        )

    def _request_json(
        self,
        endpoint: str,
        *,
        method: str,
        payload: Any,
        headers: Mapping[str, str],
    ) -> tuple[int, Any]:
        self._response_text = ""
        self._response_status = 0
        body = (
            None
            if method == "GET"
            else json.dumps(payload, default=str).encode("utf-8")
        )
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._remaining_timeout()
            ) as response:
                status = int(getattr(response, "status", 200))
                self._response_status = status
                raw = self._read_response(response)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            self._response_status = status
            with exc:
                raw = self._read_response(exc)
        self._response_status = status
        # An incomplete response can split a credential, so retain no excerpt on overflow.
        if len(raw) > 1024 * 1024:
            self._response_text = ""
            raise _ResponseFailure(
                "size", status, "HTTP response exceeds 1048576 bytes"
            )
        text = raw.decode("utf-8", errors="replace")
        self._response_text = text
        if not text:
            return status, {}
        try:
            return status, _decode_json_or_sse(text)
        except ValueError as exc:
            if status >= 400:
                return status, text
            raise _ResponseFailure("decode", status, str(exc)) from exc

    def _read_response(self, response):
        if self._deadline is None:
            return response.read(1024 * 1024 + 1)
        if isinstance(response, urllib.error.HTTPError):
            response = response.fp
        chunks = []
        size = 0
        while size <= 1024 * 1024:
            if response.isclosed():
                break
            remaining = self._remaining_timeout()
            # urllib exposes the socket through its buffered HTTP response reader.
            response.fp.raw._sock.settimeout(remaining)
            chunk = response.read1(min(65536, 1024 * 1024 + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        self._remaining_timeout()
        return b"".join(chunks)

    def _agent_response_from_payload(self, payload: Any) -> AgentResponse:
        if self.protocol == "json_template":
            selected = _select_response_value(payload, self.response_path)
            tool_calls, tool_responses = _nested_tool_evidence(payload)
            return AgentResponse(
                content=_content_text(selected),
                tool_calls=tool_calls,
                tool_responses=tool_responses,
            )
        if not isinstance(payload, Mapping):
            raise ValueError("HTTP target response must be a JSON object")
        if self.protocol == "openai_chat":
            message = _openai_message(payload)
            return AgentResponse(
                content=_content_text(message.get("content") or message.get("refusal")),
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
    if protocol not in {"fi.alk", "openai_chat", "json_template"}:
        raise ValueError("protocol must be one of: fi.alk, openai_chat, json_template")
    return protocol


_WHOLE_PLACEHOLDER = re.compile(r"^\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}$")
_PLACEHOLDER = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


def _template_context(
    input: AgentInput, *, extra: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    context = {
        "thread_id": input.thread_id,
        "execution_id": input.execution_id,
        "turn_index": input.turn_index,
        "scenario_name": input.scenario_name,
        "persona": input.persona,
        "situation": input.situation,
        "expected_outcome": input.expected_outcome,
        "messages": [dict(message) for message in input.messages],
        "new_message": dict(input.new_message or {}),
        "new_message_content": str((input.new_message or {}).get("content") or ""),
        "tools": list(input.tools),
        "metadata": dict(input.metadata),
    }
    context.update(dict(extra or {}))
    return context


def _encoded_path_segment(value: str) -> str:
    return quote(value, safe="").replace(".", "%2E")


def _render_endpoint_path(endpoint: str, context: Mapping[str, Any]) -> str:
    if not _PLACEHOLDER.search(endpoint):
        return endpoint
    parsed = urlparse(endpoint)
    if any(
        _PLACEHOLDER.search(part)
        for part in (
            parsed.scheme,
            parsed.netloc,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    ):
        raise ValueError("HTTP endpoint placeholders are supported only in the path")
    encoded = {}
    for name in _PLACEHOLDER.findall(parsed.path):
        value = context.get(name)
        if value is None or value == "":
            raise ValueError(f"HTTP path placeholder {name!r} has no value")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"HTTP path placeholder {name!r} requires a scalar")
        encoded[name] = _encoded_path_segment(str(value))
    return parsed._replace(path=_render_template(parsed.path, encoded)).geturl()


def _render_template(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _render_template(item, context) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_render_template(item, context) for item in value]
    if not isinstance(value, str):
        return value
    whole = _WHOLE_PLACEHOLDER.fullmatch(value)
    if whole:
        return context.get(whole.group(1), "")

    def replacement(match: re.Match[str]) -> str:
        item = context.get(match.group(1), "")
        if isinstance(item, (Mapping, list)):
            return json.dumps(item, separators=(",", ":"), default=str)
        return str(item)

    return _PLACEHOLDER.sub(replacement, value)


def _decode_json_or_sse(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        events: list[Any] = []
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                events.append(data)
        if events:
            return events
        raise ValueError(
            "HTTP target returned neither JSON nor JSON server-sent events"
        )


def _select_response_value(payload: Any, path: str) -> Any:
    if not path:
        return payload
    segments = [raw for raw in path.removeprefix("$").strip(".").split(".") if raw]

    def select(current: Any, offset: int) -> Any:
        if offset >= len(segments):
            return current
        raw = segments[offset]
        if isinstance(current, Mapping):
            if raw not in current:
                raise ValueError(f"response_path segment not found: {raw}")
            return select(current[raw], offset + 1)
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                index = int(raw)
            except ValueError:
                # JSON/SSE APIs often return a top-level event list while their documented
                # response schema describes one event.  A path such as ``content.parts.0.text``
                # is therefore still meaningful: select the latest event matching that shape.
                # This is envelope-driven rather than framework-specific and leaves explicit
                # numeric paths (including negative indexes) unchanged.
                last_error: ValueError | None = None
                for item in reversed(current):
                    try:
                        return select(item, offset)
                    except ValueError as exc:
                        last_error = exc
                raise ValueError(
                    f"response_path segment not found in list elements: {raw}"
                ) from last_error
            try:
                return select(current[index], offset + 1)
            except IndexError as exc:
                raise ValueError(f"response_path list index invalid: {raw}") from exc
        raise ValueError(f"response_path cannot traverse segment: {raw}")

    return select(payload, 0)


def _nested_named_values(value: Any, names: set[str]) -> list[Any]:
    """Find semantic values in an arbitrary JSON/SSE envelope.

    Source-owned agent servers commonly expose model events rather than an OpenAI response.  The
    surrounding envelope varies by framework, but function-call and function-response objects use
    stable semantic field names.  Walking the returned data keeps ``json_template`` framework
    neutral while preserving tool evidence that would otherwise be thrown away.
    """

    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in names:
                found.append(item)
            found.extend(_nested_named_values(item, names))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found.extend(_nested_named_values(item, names))
    return found


def _mapping_items(values: Sequence[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, Mapping):
            items.append(dict(value))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items.extend(dict(item) for item in value if isinstance(item, Mapping))
    return items


def _nested_tool_evidence(
    payload: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_calls = _mapping_items(
        _nested_named_values(payload, {"function_call", "functionCall", "tool_calls"})
    )
    calls = _openai_tool_calls(raw_calls)

    raw_responses = _mapping_items(
        _nested_named_values(
            payload, {"function_response", "functionResponse", "tool_responses"}
        )
    )
    responses: list[dict[str, Any]] = []
    for index, response in enumerate(raw_responses, start=1):
        responses.append(
            {
                "tool_call_id": str(
                    response.get("tool_call_id")
                    or response.get("id")
                    or f"call_{index}"
                ),
                "name": str(response.get("name") or response.get("tool") or ""),
                "result": response.get(
                    "result", response.get("response", response.get("content"))
                ),
                "error": response.get("error"),
            }
        )
    return _deduplicate_tool_records(calls), _deduplicate_tool_records(responses)


def _deduplicate_tool_records(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        marker = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(value)
    return unique


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


def _response_error_text(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return _content_text(payload)
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
    return safe_endpoint(endpoint)
