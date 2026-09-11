import inspect
import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Union

from fi.simulate.environment import (
    normalize_agent_control_plane,
    normalize_agent_trust_boundary_model,
    normalize_framework_adapter_conformance,
    normalize_framework_lifecycle_trace,
    normalize_framework_trace_events,
    normalize_framework_trace_export,
    normalize_mcp_tool_session_export,
    normalize_orchestration_trace_events,
    normalize_orchestration_trace_export,
)
from fi.simulate.agent.wrapper import (
    AgentInput,
    AgentResponse,
    AgentWrapper,
    SimulationArtifact,
    SimulationEvent,
)

InputMode = Literal["auto", "agent_input", "dict", "messages", "text"]

_KEYWORD_INPUT_NAMES = (
    "inputs",
    "input",
    "payload",
    "frame",
    "request",
    "contents",
    "arguments",
    "task",
    "user_prompt",
    "prompt",
    "message",
    "messages",
    "query",
    "data",
)

_METHOD_INPUT_KEY_PREFERENCES = {
    "execute_task": ("task", "input", "payload"),
    "kickoff": ("inputs", "input", "payload"),
    "run": ("task", "user_prompt", "prompt", "input"),
    "arun": ("task", "user_prompt", "prompt", "input"),
    "run_stream": ("task", "user_prompt", "prompt", "input"),
    "send_message": ("message", "payload", "input"),
    "message_send": ("message", "payload", "input"),
    "send": ("message", "messages", "input"),
    "achat": ("message", "messages", "input"),
    "chat": ("message", "messages", "input"),
    "query": ("query", "input", "message"),
    "respond": ("message", "input", "payload"),
    "process": ("frame", "payload", "input", "data"),
    "process_frame": ("frame", "payload", "input", "data"),
    "responses.create": ("input", "messages", "payload"),
    "chat.completions.create": ("messages", "input", "payload"),
    "messages.create": ("messages", "input", "payload"),
    "completion": ("request", "payload", "input"),
    "call_tool": ("payload", "input", "arguments"),
    "invoke_model": ("payload", "input", "request"),
    "generate_content": ("contents", "input", "payload"),
    "generate": ("prompt", "input", "payload"),
}

_AUTO_METHOD_ORDER = (
    "call",
    "ainvoke",
    "invoke",
    "astream",
    "stream",
    "stream_events",
    "execute_task",
    "kickoff",
    "process_frame",
    "process",
    "responses.create",
    "chat.completions.create",
    "messages.create",
    "run_stream",
    "arun",
    "run",
    "send_message",
    "message_send",
    "send",
    "respond",
    "achat",
    "chat",
    "query",
    "completion",
    "call_tool",
    "invoke_model",
    "generate_content",
    "generate",
)

_METHOD_INPUT_MODES: dict[str, InputMode] = {
    "ainvoke": "dict",
    "invoke": "dict",
    "astream": "dict",
    "stream": "dict",
    "stream_events": "dict",
    "execute_task": "dict",
    "kickoff": "dict",
    "process": "dict",
    "process_frame": "dict",
    "responses.create": "text",
    "chat.completions.create": "messages",
    "messages.create": "messages",
    "completion": "dict",
    "call_tool": "dict",
    "invoke_model": "dict",
    "generate_content": "dict",
    "generate": "dict",
    "call": "agent_input",
    "achat": "text",
    "chat": "text",
    "query": "text",
    "respond": "text",
    "run": "text",
    "run_stream": "text",
    "arun": "text",
    "send_message": "dict",
    "message_send": "dict",
    "send": "text",
}


class GenericAgentWrapper(AgentWrapper):
    """
    Framework-neutral adapter for agent objects, callables, and orchestration SDKs.

    The wrapper intentionally depends on conventions instead of optional imports:
    LangChain/LangGraph expose invoke/ainvoke, AutoGen and OpenAI-style runners often
    expose run/arun/run_stream, voice stacks usually expose send/respond/chat, and
    plain Python agents are just callables. Users can override method/input_mode when
    a framework has a custom shape.
    """

    def __init__(
        self,
        agent: Any,
        *,
        method: str | Callable[..., Any] | None = None,
        input_mode: InputMode = "auto",
        input_key: str | None = None,
        input_kwargs: Optional[Mapping[str, Any]] = None,
        output_key: str | None = None,
        system_prompt: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_runtime: bool = False,
        runtime_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent = agent
        self.method = method
        self.input_mode = input_mode
        self.input_key = input_key
        self.input_kwargs = dict(input_kwargs or {})
        self.output_key = output_key
        self.system_prompt = system_prompt
        self.metadata = metadata or {}
        self.trace_runtime = trace_runtime
        self.runtime_metadata = runtime_metadata or {}

    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
        method = self._resolve_method()
        method_name = (
            self.method
            if isinstance(self.method, str)
            else getattr(method, "__name__", None)
        )
        runtime_input_mode = (
            self._infer_input_mode(method_name)
            if self.input_mode == "auto"
            else self.input_mode
        )
        payload = self._build_payload(input, method_name=method_name)
        started_at = time.time()
        streamed = False

        raw, call_style, selected_input_key = _invoke_method_with_payload(
            method,
            payload,
            method_name=method_name,
            input_key=self.input_key,
            input_kwargs=self.input_kwargs,
        )

        if inspect.isawaitable(raw):
            raw = await raw

        if _is_async_stream(raw):
            streamed = True
            raw = await self._coerce_async_stream(raw)
        elif _is_sync_stream(raw):
            streamed = True
            raw = self._coerce_sync_stream(raw)

        response = self._coerce_response(raw)
        if not self.trace_runtime:
            return response
        trace = _framework_runtime_trace(
            framework=str(self.metadata.get("framework") or "generic"),
            method=method,
            method_name=method_name,
            input_mode=runtime_input_mode,
            payload=payload,
            response=response,
            duration_ms=int((time.time() - started_at) * 1000),
            streamed=streamed,
            call_style=call_style,
            input_key=selected_input_key,
            input_kwargs_keys=sorted(str(key) for key in self.input_kwargs),
            wrapper_metadata=self.metadata,
            runtime_metadata=self.runtime_metadata,
        )
        return _attach_framework_runtime_trace(response, trace)

    def _resolve_method(self) -> Callable[..., Any]:
        if isinstance(self.agent, AgentWrapper):
            return self.agent.call

        if callable(self.method):
            return self.method

        if isinstance(self.method, str):
            candidate = _resolve_callable_attr_path(self.agent, self.method)
            if callable(candidate):
                return candidate
            raise AttributeError(f"Agent does not expose method '{self.method}'.")

        for name in _AUTO_METHOD_ORDER:
            candidate = _resolve_callable_attr_path(self.agent, name)
            if callable(candidate):
                return candidate

        if callable(self.agent):
            return self.agent

        raise TypeError(
            "GenericAgentWrapper needs a callable agent or an object exposing one "
            "of the supported framework adapter method names."
        )

    def _build_payload(self, input: AgentInput, *, method_name: str | None) -> Any:
        mode = self.input_mode
        if mode == "auto":
            mode = self._infer_input_mode(method_name)

        if mode == "agent_input":
            return input

        messages = self._messages_with_system(input.messages)
        latest_text = _message_content(input.new_message) if input.new_message else ""

        if mode == "messages":
            return messages
        if mode == "text":
            return latest_text
        if mode == "dict":
            return {
                "messages": messages,
                "input": latest_text,
                "thread_id": input.thread_id,
                "execution_id": input.execution_id,
                "turn_index": input.turn_index,
                "scenario_name": input.scenario_name,
                "persona": input.persona,
                "situation": input.situation,
                "expected_outcome": input.expected_outcome,
                "modality": input.modality,
                "artifacts": [_model_to_dict(artifact) for artifact in input.artifacts],
                "events": [_model_to_dict(event) for event in input.events],
                "memory": input.memory,
                "tools": input.tools,
                "metadata": {**input.metadata, **self.metadata},
            }

        return input

    def _infer_input_mode(self, method_name: str | None) -> InputMode:
        return (
            _METHOD_INPUT_MODES.get(str(method_name or ""))
            or _METHOD_INPUT_MODES.get(_method_leaf(method_name))
            or "agent_input"
        )

    def _messages_with_system(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = [dict(message) for message in messages]
        if not self.system_prompt:
            return normalized
        if normalized and normalized[0].get("role") == "system":
            return normalized
        return [{"role": "system", "content": self.system_prompt}, *normalized]

    def _coerce_response(self, raw: Any) -> str | AgentResponse:
        if isinstance(raw, AgentResponse):
            return raw
        if isinstance(raw, str):
            return raw
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")

        content = self._extract_content(raw)
        tool_calls = self._extract_tool_calls(raw)
        tool_responses = self._extract_tool_responses(raw)
        artifacts = self._extract_artifacts(raw)
        events = self._extract_events(raw)
        memory_updates = self._extract_memory_updates(raw)
        state = self._extract_state(raw)
        metadata = self._extract_metadata(raw)
        if self.metadata:
            metadata = {**metadata, **self.metadata}

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            tool_responses=tool_responses,
            artifacts=artifacts,
            events=events,
            memory_updates=memory_updates,
            state=state or None,
            metadata=metadata or None,
        )

    async def _coerce_async_stream(self, raw: Any) -> AgentResponse:
        chunks: List[Any] = []
        async for chunk in raw:
            chunks.append(chunk)
        return self._coerce_stream_chunks(chunks)

    def _coerce_sync_stream(self, raw: Any) -> AgentResponse:
        return self._coerce_stream_chunks(list(raw))

    def _coerce_stream_chunks(self, chunks: List[Any]) -> AgentResponse:
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_responses: List[Dict[str, Any]] = []
        artifacts: List[SimulationArtifact] = []
        events: List[SimulationEvent] = []

        for index, chunk in enumerate(chunks, start=1):
            text = _stream_chunk_text(chunk)
            if text:
                content_parts.append(text)
            tool_calls.extend(self._extract_tool_calls(chunk) or [])
            tool_responses.extend(self._extract_tool_responses(chunk) or [])
            artifacts.extend(self._extract_artifacts(chunk))
            events.extend(self._extract_events(chunk))
            events.append(_stream_chunk_event(chunk, index=index, text=text))

        trace = _streaming_trace_from_chunks(chunks, self.metadata)
        state: Dict[str, Any] = {}
        if trace.get("events"):
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=trace,
                    metadata={
                        "kind": "streaming_trace",
                        "framework": trace.get("framework", "generic"),
                        "source": "generic_agent_wrapper",
                    },
                )
            )
            state["streaming_trace"] = trace

        metadata = {
            "streaming": {
                "chunk_count": len(chunks),
                "content_part_count": len(content_parts),
                "signals": list(trace.get("signals", [])),
                "summary": dict(trace.get("summary", {})),
            },
            **self.metadata,
        }
        return AgentResponse(
            content="".join(content_parts),
            tool_calls=tool_calls or None,
            tool_responses=tool_responses or None,
            artifacts=artifacts,
            events=events,
            state=state or None,
            metadata=metadata,
        )

    def _extract_content(self, raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")

        raw_mapping = _object_mapping(raw)
        if raw_mapping is not None:
            if self.output_key and self.output_key in raw_mapping:
                return _stringify(raw_mapping[self.output_key])
            for key in (
                "content",
                "output",
                "response",
                "text",
                "final_output",
                "answer",
                "result",
                "data",
            ):
                if key in raw_mapping and raw_mapping[key] is not None:
                    if key == "content":
                        block_text = _content_blocks_text(raw_mapping[key])
                        if block_text:
                            return block_text
                    return _stringify(raw_mapping[key])
            if "message" in raw_mapping:
                return _message_content(raw_mapping["message"])
            if "messages" in raw_mapping:
                return _last_message_content(raw_mapping["messages"])
            if "choices" in raw_mapping:
                return _choices_content(raw_mapping["choices"])
            realtime_text = _realtime_last_text(raw_mapping)
            if realtime_text:
                return realtime_text

        for attr in ("content", "output", "response", "text", "final_output", "answer"):
            if hasattr(raw, attr):
                value = getattr(raw, attr)
                if value is not None:
                    return _stringify(value)

        if hasattr(raw, "message"):
            return _message_content(getattr(raw, "message"))
        if hasattr(raw, "messages"):
            return _last_message_content(getattr(raw, "messages"))
        if isinstance(raw, (list, tuple)):
            return _last_message_content(raw)
        realtime_text = _realtime_last_text(raw)
        if realtime_text:
            return realtime_text

        return str(raw)

    def _extract_tool_calls(self, raw: Any) -> Optional[List[Dict[str, Any]]]:
        tool_calls = _extract_list_field(
            raw,
            ("tool_calls", "toolCalls", "tool_call_chunks", "toolCallChunks"),
        )
        provider_tool_calls = _provider_tool_calls(raw)
        history_tool_calls = _message_history_tool_calls(raw)
        realtime_tool_calls = _realtime_tool_calls(raw)
        framework_trace_tool_calls = _framework_trace_tool_calls(raw)
        orchestration_tool_calls = _orchestration_trace_tool_calls(raw)
        mcp_tool_calls = _mcp_tool_session_tool_calls(raw)
        workflow_tool_calls = _workflow_trace_tool_calls(raw)
        browser_tool_calls = _browser_cua_tool_calls(raw)
        return [
            *(tool_calls or []),
            *provider_tool_calls,
            *history_tool_calls,
            *realtime_tool_calls,
            *framework_trace_tool_calls,
            *orchestration_tool_calls,
            *mcp_tool_calls,
            *workflow_tool_calls,
            *browser_tool_calls,
        ] or None

    def _extract_tool_responses(self, raw: Any) -> Optional[List[Dict[str, Any]]]:
        tool_responses = _extract_list_field(raw, ("tool_responses", "toolResponses", "tool_outputs", "toolOutputs"))
        history_tool_responses = _message_history_tool_responses(raw)
        realtime_tool_responses = _realtime_tool_responses(raw)
        framework_trace_tool_responses = _framework_trace_tool_responses(raw)
        orchestration_tool_responses = _orchestration_trace_tool_responses(raw)
        mcp_tool_responses = _mcp_tool_session_tool_responses(raw)
        return [
            *(tool_responses or []),
            *history_tool_responses,
            *realtime_tool_responses,
            *framework_trace_tool_responses,
            *orchestration_tool_responses,
            *mcp_tool_responses,
        ] or None

    def _extract_metadata(self, raw: Any) -> Dict[str, Any]:
        raw_mapping = _object_mapping(raw)
        metadata: Dict[str, Any] = {}
        if raw_mapping is not None:
            value = raw_mapping.get("metadata")
            if isinstance(value, dict):
                metadata.update(dict(value))
            metadata.update(_provider_metadata(raw_mapping))
            return metadata
        value = getattr(raw, "metadata", None)
        if isinstance(value, dict):
            metadata.update(dict(value))
        metadata.update(_provider_metadata(raw))
        return metadata

    def _extract_memory_updates(self, raw: Any) -> Optional[Dict[str, Any]]:
        raw_mapping = _object_mapping(raw)
        for name in ("memory_updates", "memoryUpdates"):
            value = raw_mapping.get(name) if raw_mapping is not None else getattr(raw, name, None)
            plain = _plain_value(value)
            if isinstance(plain, Mapping):
                return dict(plain)
        memory_updates = _framework_memory_updates(raw)
        if memory_updates:
            return memory_updates
        return None

    def _extract_state(self, raw: Any) -> Dict[str, Any]:
        raw_mapping = _object_mapping(raw)
        state: Dict[str, Any] = {}
        for name in ("state", "output_state", "outputState"):
            value = raw_mapping.get(name) if raw_mapping is not None else getattr(raw, name, None)
            plain = _plain_value(value)
            if isinstance(plain, Mapping):
                state.update(dict(plain))

        if raw_mapping is not None:
            for name in ("typed_output", "structured_output", "validated_output"):
                value = raw_mapping.get(name)
                if value not in (None, "", [], {}):
                    state[name] = _plain_value(value)
            output_value = raw_mapping.get("output")
            output_payload = _plain_value(output_value)
            if (
                isinstance(output_payload, Mapping)
                and output_payload
                and "typed_output" not in state
            ):
                state["typed_output"] = dict(output_payload)
            provider_state = _provider_response_state(raw_mapping)
            if provider_state:
                state.setdefault("provider_response", provider_state)
        openenv_state = _openenv_trace_state(state) or _openenv_trace_state(raw)
        if openenv_state:
            state["openenv"] = openenv_state
            state.pop("open_env", None)
            state.pop("gymnasium_env", None)
        history_state = _message_history_state(raw)
        if history_state:
            state.setdefault("message_history", history_state)
        handoff_state = _message_history_handoff_state(raw)
        if handoff_state:
            state.setdefault("framework_handoffs", handoff_state)
        realtime_state = _realtime_trace_state(raw)
        if realtime_state:
            state.setdefault("realtime_trace", realtime_state)
        lifecycle_state = _framework_lifecycle_state(raw)
        if lifecycle_state:
            state.setdefault("framework_lifecycle_trace", lifecycle_state)
        framework_trace_state = _framework_trace_state(raw)
        if framework_trace_state:
            state.setdefault("framework_trace", framework_trace_state)
        orchestration_state = _orchestration_trace_state(raw)
        if orchestration_state:
            state.setdefault("orchestration_trace", orchestration_state)
        mcp_state = _mcp_tool_session_state(raw)
        if mcp_state:
            state.setdefault("mcp_tool_session", mcp_state)
        a2a_state = _a2a_protocol_state(raw)
        if a2a_state:
            state.setdefault("a2a_protocol_trace", a2a_state)
        workflow_state = _workflow_trace_state(raw)
        if workflow_state:
            state.setdefault("workflow_trace", workflow_state)
        memory_state = _framework_memory_state(raw)
        if memory_state:
            state.setdefault("framework_memory", memory_state)
            retrieval_memory = _framework_memory_retrieval_memory(raw)
            if retrieval_memory:
                state.setdefault("retrieval_memory", retrieval_memory)
            agent_memory_lineage = _framework_memory_agent_lineage(raw)
            if agent_memory_lineage:
                state.setdefault("agent_memory_lineage", agent_memory_lineage)
        browser_state = _browser_cua_state(raw)
        if browser_state:
            state.setdefault("browser_cua", browser_state)
        trust_boundary_state = _agent_trust_boundary_state(raw)
        if trust_boundary_state:
            state.setdefault("agent_trust_boundary_model", trust_boundary_state)
        control_plane_state = _agent_control_plane_state(raw)
        if control_plane_state:
            state.setdefault("agent_control_plane", control_plane_state)
        return state

    def _extract_artifacts(self, raw: Any) -> List[SimulationArtifact]:
        values = _extract_list_field(raw, ("artifacts", "media", "attachments"))
        artifacts: List[SimulationArtifact] = []
        for value in values or []:
            try:
                artifacts.append(SimulationArtifact(**value))
            except Exception:
                continue
        realtime_state = _realtime_trace_state(raw)
        if realtime_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=realtime_state,
                    metadata={
                        "kind": "realtime_trace",
                        "source": "generic_agent_wrapper",
                    },
                )
            )
        lifecycle_state = _framework_lifecycle_state(raw)
        if lifecycle_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=lifecycle_state,
                    metadata={
                        "kind": "framework_lifecycle_trace",
                        "source": "generic_agent_wrapper",
                    },
                )
            )
        framework_trace_state = _framework_trace_state(raw)
        if framework_trace_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=framework_trace_state,
                    metadata={
                        "kind": "framework_trace",
                        "framework": framework_trace_state.get("framework", "generic"),
                        "source": "generic_agent_wrapper",
                    },
                )
            )
        orchestration_state = _orchestration_trace_state(raw)
        if orchestration_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=orchestration_state,
                    metadata={
                        "kind": "orchestration_trace",
                        "framework": orchestration_state.get("framework", "generic"),
                        "source": "generic_agent_wrapper",
                    },
                )
            )
        mcp_state = _mcp_tool_session_state(raw)
        if mcp_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=mcp_state,
                    metadata={
                        "kind": "mcp_tool_session",
                        "source": "generic_agent_wrapper",
                    },
                )
            )
        a2a_state = _a2a_protocol_state(raw)
        if a2a_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=a2a_state,
                    metadata={
                        "kind": "a2a_protocol_trace",
                        "source": "generic_agent_wrapper",
                    },
                )
            )
            for artifact in _plain_list(a2a_state.get("artifacts")):
                artifact_dict = _plain_mapping(artifact)
                artifacts.append(
                    SimulationArtifact(
                        type=_a2a_simulation_artifact_type(artifact_dict),
                        role="assistant",
                        data=artifact_dict,
                        metadata={
                            "kind": "a2a_artifact",
                            "source": "generic_agent_wrapper",
                            "id": str(artifact_dict.get("id") or ""),
                        },
                    )
                )
        memory_state = _framework_memory_state(raw)
        if memory_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=memory_state,
                    metadata={
                        "kind": "framework_memory",
                        "source": "generic_agent_wrapper",
                    },
                )
            )
        workflow_state = _workflow_trace_state(raw)
        if workflow_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=_workflow_trace_payload(raw),
                    metadata={
                        "kind": "workflow_trace",
                        "source": "generic_agent_wrapper",
                    },
                )
            )
        browser_state = _browser_cua_state(raw)
        if browser_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=_browser_cua_trace_payload(raw),
                    metadata={
                        "kind": "browser_trace",
                        "source": "generic_agent_wrapper",
                    },
                )
            )
            for screenshot in browser_state.get("screenshots", []):
                uri = screenshot.get("uri") or screenshot.get("screenshot_uri")
                if not uri:
                    continue
                artifacts.append(
                    SimulationArtifact(
                        type="screenshot",
                        uri=str(uri),
                        role="assistant",
                        metadata={
                            "kind": "browser_screenshot",
                            "source": "generic_agent_wrapper",
                            "id": str(screenshot.get("id") or ""),
                        },
                    )
                )
        openenv_state = _openenv_trace_state(raw)
        if openenv_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=openenv_state,
                    metadata={
                        "kind": "openenv_trace",
                        "source": "generic_agent_wrapper",
                    },
                )
            )
        trust_boundary_state = _agent_trust_boundary_state(raw)
        if trust_boundary_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=trust_boundary_state,
                    metadata={
                        "kind": "agent_trust_boundary_model",
                        "source": "generic_agent_wrapper",
                    },
                )
            )
        control_plane_state = _agent_control_plane_state(raw)
        if control_plane_state:
            artifacts.append(
                SimulationArtifact(
                    type="trace",
                    role="assistant",
                    data=control_plane_state,
                    metadata={
                        "kind": "agent_control_plane",
                        "source": "generic_agent_wrapper",
                    },
                )
            )
        return artifacts

    def _extract_events(self, raw: Any) -> List[SimulationEvent]:
        values = _extract_list_field(raw, ("events", "trajectory", "spans"))
        events: List[SimulationEvent] = []
        for value in values or []:
            try:
                events.append(SimulationEvent(**value))
            except Exception:
                continue
        events.extend(_provider_events(raw))
        events.extend(_message_history_events(raw))
        events.extend(_message_history_coordination_events(raw))
        events.extend(_realtime_trace_events(raw))
        events.extend(_framework_lifecycle_events(raw))
        events.extend(_framework_trace_events(raw))
        events.extend(_orchestration_trace_events(raw))
        events.extend(_mcp_tool_session_events(raw))
        events.extend(_a2a_protocol_events(raw))
        events.extend(_workflow_trace_events(raw))
        events.extend(_framework_memory_events(raw))
        events.extend(_browser_cua_events(raw))
        events.extend(_openenv_trace_events(raw))
        events.extend(_agent_trust_boundary_events(raw))
        events.extend(_agent_control_plane_events(raw))
        return events


class _NoPayload:
    pass


_NO_PAYLOAD = _NoPayload()


def wrap_agent(
    agent: Any,
    *,
    method: str | Callable[..., Any] | None = None,
    input_mode: InputMode = "auto",
    input_key: str | None = None,
    input_kwargs: Optional[Mapping[str, Any]] = None,
    output_key: str | None = None,
    system_prompt: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_runtime: bool = False,
    runtime_metadata: Optional[Dict[str, Any]] = None,
) -> AgentWrapper:
    """Return an AgentWrapper for an existing AgentWrapper, object, or callable."""

    if isinstance(agent, AgentWrapper) and method is None and input_mode == "auto":
        return agent
    return GenericAgentWrapper(
        agent,
        method=method,
        input_mode=input_mode,
        input_key=input_key,
        input_kwargs=input_kwargs,
        output_key=output_key,
        system_prompt=system_prompt,
        metadata=metadata,
        trace_runtime=trace_runtime,
        runtime_metadata=runtime_metadata,
    )


def _extract_list_field(raw: Any, names: Iterable[str]) -> Optional[List[Dict[str, Any]]]:
    value = None
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        for name in names:
            value = raw_mapping.get(name)
            if value is not None:
                break
    else:
        for name in names:
            if hasattr(raw, name):
                value = getattr(raw, name)
                break
    if not isinstance(value, (list, tuple)):
        return None
    items: List[Dict[str, Any]] = []
    for item in value:
        item_mapping = _object_mapping(item)
        if item_mapping is not None:
            items.append(dict(item_mapping))
    return items or None


def _provider_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        calls.extend(_tool_calls_from_message(raw_mapping, include_direct_keys=False))
        for choice in _provider_choices(raw_mapping):
            message = _provider_choice_message(choice)
            calls.extend(_tool_calls_from_message(message))
    return calls


def _tool_calls_from_message(
    message: Mapping[str, Any],
    *,
    include_direct_keys: bool = True,
) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    if include_direct_keys:
        for key in ("tool_calls", "toolCalls", "tool_call_chunks", "toolCallChunks"):
            calls.extend(_list_of_mappings(message.get(key)))
    function_call = _object_mapping(message.get("function_call"))
    if function_call:
        calls.append(
            {
                "id": str(function_call.get("id") or "function_call"),
                "type": "function",
                "function": {
                    "name": function_call.get("name"),
                    "arguments": function_call.get("arguments"),
                },
            }
        )
    for block in _content_blocks(message.get("content")):
        block_type = str(block.get("type") or block.get("kind") or "")
        block_type_key = block_type.lower().replace("_", "").replace("-", "")
        has_tool_shape = bool(block.get("name")) and (
            "arguments" in block or "input" in block or "args" in block
        )
        if block_type != "tool_use" and "functioncall" not in block_type_key and not has_tool_shape:
            continue
        name = str(block.get("name") or block.get("tool") or "")
        calls.append(
            {
                "id": str(block.get("id") or name or "tool_use"),
                "type": "tool_use",
                "name": name,
                "arguments": block.get("input") or block.get("arguments") or {},
                "function": {
                    "name": name,
                    "arguments": block.get("input") or block.get("arguments") or {},
                },
            }
        )
    return calls


def _provider_events(raw: Any) -> List[SimulationEvent]:
    events: List[SimulationEvent] = []
    raw_mapping = _object_mapping(raw)
    if raw_mapping is None:
        return events
    for index, choice in enumerate(_provider_choices(raw_mapping), start=1):
        finish_reason = str(choice.get("finish_reason") or choice.get("stop_reason") or "")
        if finish_reason:
            events.append(
                SimulationEvent(
                    type="provider_choice",
                    name=finish_reason,
                    payload={
                        "index": index,
                        "finish_reason": finish_reason,
                    },
                )
            )
    for index, call in enumerate(_provider_tool_calls(raw_mapping), start=1):
        name = str(
            call.get("name")
            or call.get("tool")
            or dict(call.get("function") or {}).get("name")
            or f"provider_tool_call_{index}"
        )
        events.append(
            SimulationEvent(
                type="provider_tool_call",
                name=name,
                payload=call,
            )
        )
    return events


def _provider_metadata(raw: Any) -> Dict[str, Any]:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is None:
        return {}
    metadata: Dict[str, Any] = {}
    for key in ("id", "model", "object", "type", "role", "stop_reason", "stop_sequence"):
        value = raw_mapping.get(key)
        if value not in (None, "", [], {}):
            metadata[f"provider_{key}"] = value
    usage = _object_mapping(raw_mapping.get("usage"))
    if usage:
        metadata["provider_usage"] = usage
    return metadata


def _provider_response_state(raw: Any) -> Dict[str, Any]:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is None:
        return {}
    choices = _provider_choices(raw_mapping)
    tool_calls = _provider_tool_calls(raw_mapping)
    usage = _object_mapping(raw_mapping.get("usage"))
    has_provider_envelope = bool(
        choices
        or tool_calls
        or usage
        or raw_mapping.get("model")
        or raw_mapping.get("object")
        or raw_mapping.get("id")
    )
    if not has_provider_envelope:
        return {}
    finish_reasons = sorted(
        {
            str(choice.get("finish_reason") or choice.get("stop_reason") or "")
            for choice in choices
            if choice.get("finish_reason") or choice.get("stop_reason")
        }
    )
    state: Dict[str, Any] = {}
    if choices:
        state["choice_count"] = len(choices)
    if finish_reasons:
        state["finish_reasons"] = finish_reasons
    if tool_calls:
        state["tool_call_count"] = len(tool_calls)
        state["tool_names"] = sorted(
            {
                str(
                    call.get("name")
                    or call.get("tool")
                    or dict(call.get("function") or {}).get("name")
                    or ""
                )
                for call in tool_calls
                if isinstance(call, Mapping)
            }
        )
    if usage:
        state["usage"] = usage
    for key in ("id", "model", "object", "type", "role", "stop_reason"):
        value = raw_mapping.get(key)
        if value not in (None, "", [], {}):
            state[key] = value
    return state


def _agent_trust_boundary_state(raw: Any) -> Dict[str, Any]:
    payload = _agent_trust_boundary_payload(raw)
    if not payload:
        return {}
    framework = str(
        payload.get("framework")
        or _agent_control_plane_framework(raw)
        or "custom"
    )
    return normalize_agent_trust_boundary_model(payload, framework=framework)


def _agent_control_plane_state(raw: Any) -> Dict[str, Any]:
    payload = _agent_control_plane_payload(raw)
    if not payload:
        return {}
    framework = str(
        payload.get("framework")
        or _agent_control_plane_framework(raw)
        or "custom"
    )
    return normalize_agent_control_plane(payload, framework=framework)


def _agent_trust_boundary_payload(raw: Any) -> Dict[str, Any]:
    for key in (
        "agent_trust_boundary_model",
        "agent_trust_boundary",
        "trust_boundary",
        "trust_boundary_model",
        "threat_model",
    ):
        payload = _agent_control_plane_payload_field(raw, key)
        if payload:
            return payload
    return {}


def _agent_control_plane_payload(raw: Any) -> Dict[str, Any]:
    for key in (
        "agent_control_plane",
        "control_plane",
        "runtime_governance",
        "agency_control",
    ):
        payload = _agent_control_plane_payload_field(raw, key)
        if payload:
            return payload
    return {}


def _agent_control_plane_payload_field(raw: Any, key: str) -> Dict[str, Any]:
    raw_mapping = _object_mapping(raw)
    candidates: List[Any] = []
    if raw_mapping is not None:
        candidates.append(raw_mapping.get(key))
        state = _plain_mapping(raw_mapping.get("state"))
        candidates.append(state.get(key))
        output = _plain_mapping(raw_mapping.get("output"))
        candidates.append(output.get(key))
    else:
        candidates.append(getattr(raw, key, None))
        state = _plain_mapping(getattr(raw, "state", None))
        candidates.append(state.get(key))
    for candidate in candidates:
        payload = _plain_mapping(candidate)
        if payload:
            return payload
    return {}


def _agent_control_plane_framework(raw: Any) -> str:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        metadata = _plain_mapping(raw_mapping.get("metadata"))
        for source in (raw_mapping, metadata):
            value = source.get("framework") or source.get("runtime")
            if value not in (None, "", [], {}):
                return str(value)
    metadata = _plain_mapping(getattr(raw, "metadata", None))
    value = metadata.get("framework") or metadata.get("runtime")
    return str(value or "")


def _agent_trust_boundary_events(raw: Any) -> List[SimulationEvent]:
    state = _agent_trust_boundary_state(raw)
    if not state:
        return []
    summary = _plain_mapping(state.get("summary"))
    events = [
        SimulationEvent(
            type="agent_trust_boundary_ready",
            name=str(state.get("name") or "agent_trust_boundary_ready"),
            payload={"framework": state.get("framework"), "summary": summary},
        ),
        SimulationEvent(
            type="agent_trust_boundary_status",
            name=str(state.get("name") or "agent_trust_boundary_status"),
            payload=summary,
        ),
    ]
    if _plain_list(summary.get("gaps")):
        events.append(
            SimulationEvent(
                type="agent_trust_gaps_listed",
                name="agent_trust_gaps_listed",
                payload={"gaps": _plain_list(summary.get("gaps"))},
            )
        )
    else:
        events.append(
            SimulationEvent(
                type="agent_trust_gaps_listed",
                name="agent_trust_gaps_listed",
                payload={"gaps": []},
            )
        )
    for event_type, key in (
        ("agent_trust_assets_listed", "assets"),
        ("agent_trust_tools_listed", "tools"),
        ("agent_trust_surfaces_listed", "surfaces"),
    ):
        events.append(
            SimulationEvent(
                type=event_type,
                name=event_type,
                payload={"items": _plain_list(state.get(key))},
            )
        )
    controls = _plain_list(state.get("controls"))
    if controls:
        events.append(
            SimulationEvent(
                type="agent_trust_control_inspected",
                name="agent_trust_control_inspected",
                payload=_plain_mapping(controls[0]),
            )
        )
    return events


def _agent_control_plane_events(raw: Any) -> List[SimulationEvent]:
    state = _agent_control_plane_state(raw)
    if not state:
        return []
    summary = _plain_mapping(state.get("summary"))
    events = [
        SimulationEvent(
            type="agent_control_plane_ready",
            name=str(state.get("name") or "agent_control_plane_ready"),
            payload={"framework": state.get("framework"), "summary": summary},
        ),
        SimulationEvent(
            type="agent_control_plane_status",
            name=str(state.get("name") or "agent_control_plane_status"),
            payload=summary,
        ),
        SimulationEvent(
            type="agent_control_gaps_listed",
            name="agent_control_gaps_listed",
            payload={"gaps": _plain_list(summary.get("gaps"))},
        ),
        SimulationEvent(
            type="agent_control_actions_listed",
            name="agent_control_actions_listed",
            payload={"items": _plain_list(state.get("actions"))},
        ),
        SimulationEvent(
            type="agent_control_budgets_listed",
            name="agent_control_budgets_listed",
            payload={"items": _plain_list(state.get("budgets"))},
        ),
        SimulationEvent(
            type="agent_control_incidents_listed",
            name="agent_control_incidents_listed",
            payload={"items": _plain_list(state.get("incidents"))},
        ),
    ]
    actions = _plain_list(state.get("actions"))
    if actions:
        events.append(
            SimulationEvent(
                type="agent_control_action_inspected",
                name="agent_control_action_inspected",
                payload=_plain_mapping(actions[0]),
            )
        )
    return events


def _provider_choices(raw: Any) -> List[Dict[str, Any]]:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is None:
        return []
    return _list_of_mappings(raw_mapping.get("choices"))


def _provider_choice_message(choice: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("message", "delta"):
        mapping = _object_mapping(choice.get(key))
        if mapping:
            return mapping
    return dict(choice)


def _content_blocks(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return _list_of_mappings(value)


def _content_blocks_text(value: Any) -> str:
    parts: List[str] = []
    if isinstance(value, str):
        return value
    if not isinstance(value, (list, tuple)):
        return ""
    for item in value:
        if isinstance(item, str):
            parts.append(item)
            continue
        block = _object_mapping(item)
        if not block:
            continue
        for key in ("text", "content"):
            text = block.get(key)
            if text not in (None, "", [], {}):
                parts.append(_stringify(text))
                break
    return " ".join(part for part in parts if part)


def _list_of_mappings(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    items: List[Dict[str, Any]] = []
    for item in value:
        mapping = _object_mapping(item)
        if mapping:
            items.append(mapping)
    return items


def _message_history_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for message in _message_history(raw):
        calls.extend(_tool_calls_from_message(message))
    return calls


def _message_history_tool_responses(raw: Any) -> List[Dict[str, Any]]:
    responses: List[Dict[str, Any]] = []
    for message in _message_history(raw):
        message_type = str(message.get("type") or message.get("kind") or "")
        content_blocks = _content_blocks(message.get("content"))
        if not content_blocks and (
            "ToolCallExecution" in message_type
            or str(message.get("role") or "") == "tool"
        ):
            content = message.get("content") or message.get("result") or message.get("output")
            if content not in (None, "", [], {}):
                responses.append(
                    {
                        "id": str(
                            message.get("id")
                            or message.get("call_id")
                            or message.get("tool_call_id")
                            or "tool_response"
                        ),
                        "name": str(message.get("name") or message.get("tool") or ""),
                        "content": _plain_value(content),
                        "is_error": bool(message.get("is_error") or message.get("error")),
                    }
                )
            continue
        for block in content_blocks:
            block_type = str(block.get("type") or block.get("kind") or "")
            block_type_key = block_type.lower().replace("_", "").replace("-", "")
            is_response = (
                "toolcallresult" in block_type_key
                or "toolresult" in block_type_key
                or bool(block.get("call_id") or block.get("tool_call_id"))
                and ("content" in block or "result" in block or "output" in block)
            )
            if not is_response:
                continue
            responses.append(
                {
                    "id": str(
                        block.get("id")
                        or block.get("call_id")
                        or block.get("tool_call_id")
                        or "tool_response"
                    ),
                    "name": str(block.get("name") or block.get("tool") or ""),
                    "content": _plain_value(
                        block.get("content")
                        if "content" in block
                        else block.get("result", block.get("output"))
                    ),
                    "is_error": bool(block.get("is_error") or block.get("error")),
                }
            )
    return responses


def _message_history_events(raw: Any) -> List[SimulationEvent]:
    events: List[SimulationEvent] = []
    for index, message in enumerate(_message_history(raw), start=1):
        message_type = str(
            message.get("type")
            or message.get("kind")
            or message.get("role")
            or "message_history"
        )
        source = str(
            message.get("source")
            or message.get("name")
            or message.get("speaker")
            or message.get("role")
            or ""
        )
        payload = {
            "index": index,
            "type": message_type,
            "role": message.get("role"),
            "source": source,
            "content_length": len(_message_content(message)),
            "tool_call_count": len(_tool_calls_from_message(message)),
            "tool_response_count": len(
                _message_history_tool_responses({"messages": [message]})
            ),
        }
        for key in ("handoff_from", "handoff_to", "recipient", "task", "stop_reason"):
            value = message.get(key)
            if value not in (None, "", [], {}):
                payload[key] = value
        events.append(
            SimulationEvent(
                type=message_type,
                name=source or message_type,
                payload=payload,
                metadata={"kind": "message_history", "message_index": index},
            )
        )
    return events


def _message_history_coordination_events(raw: Any) -> List[SimulationEvent]:
    state = _message_history_handoff_state(raw)
    if not state:
        return []
    events: List[SimulationEvent] = []
    for index, handoff in enumerate(state.get("handoffs", []), start=1):
        handoff_dict = dict(handoff)
        events.append(
            SimulationEvent(
                type="framework_handoff",
                name=str(
                    handoff_dict.get("name")
                    or f"{handoff_dict.get('from', '')}->{handoff_dict.get('to', '')}"
                ),
                payload={**handoff_dict, "sequence": index},
                metadata={"kind": "framework_coordination", "coordination": "handoff"},
            )
        )
    for index, review in enumerate(state.get("reviews", []), start=1):
        review_dict = dict(review)
        events.append(
            SimulationEvent(
                type="framework_review",
                name=str(review_dict.get("name") or review_dict.get("reviewer") or "review"),
                payload={**review_dict, "sequence": index},
                metadata={"kind": "framework_coordination", "coordination": "review"},
            )
        )
    for index, reconciliation in enumerate(state.get("reconciliations", []), start=1):
        reconciliation_dict = dict(reconciliation)
        events.append(
            SimulationEvent(
                type="framework_reconciliation",
                name=str(
                    reconciliation_dict.get("name")
                    or reconciliation_dict.get("accepted_source")
                    or "reconciliation"
                ),
                payload={**reconciliation_dict, "sequence": index},
                metadata={
                    "kind": "framework_coordination",
                    "coordination": "reconciliation",
                },
            )
        )
    return events


def _message_history_handoff_state(raw: Any) -> Dict[str, Any]:
    messages = _message_history(raw)
    if not messages:
        return {}
    handoffs: List[Dict[str, Any]] = []
    reviews: List[Dict[str, Any]] = []
    reconciliations: List[Dict[str, Any]] = []
    participants: set[str] = set()

    for index, message in enumerate(messages, start=1):
        source = str(
            message.get("source")
            or message.get("speaker")
            or message.get("name")
            or message.get("role")
            or ""
        )
        if source:
            participants.add(source)
        target = str(message.get("handoff_to") or message.get("recipient") or "")
        if target:
            participants.add(target)
        if _is_handoff_message(message):
            handoffs.append(
                {
                    "index": index,
                    "name": str(message.get("name") or "framework_handoff"),
                    "from": str(message.get("handoff_from") or source),
                    "to": target,
                    "task": str(message.get("task") or _message_content(message)),
                    "reason": str(message.get("reason") or message.get("rationale") or ""),
                    "message_type": str(
                        message.get("type") or message.get("kind") or message.get("role") or ""
                    ),
                }
            )
        review = _review_payload_from_message(message, index=index, source=source)
        if review:
            reviewer = str(review.get("reviewer") or "")
            if reviewer:
                participants.add(reviewer)
            review_target = str(review.get("target") or "")
            if review_target:
                participants.add(review_target)
            reviews.append(review)
        reconciliation = _reconciliation_payload_from_message(
            message,
            index=index,
            source=source,
        )
        if reconciliation:
            accepted_source = str(reconciliation.get("accepted_source") or "")
            if accepted_source:
                participants.add(accepted_source)
            reconciliations.append(reconciliation)

    if not handoffs and not reviews and not reconciliations:
        return {}
    return {
        "handoff_count": len(handoffs),
        "review_count": len(reviews),
        "reconciliation_count": len(reconciliations),
        "participants": sorted(participants),
        "handoffs": handoffs,
        "reviews": reviews,
        "reconciliations": reconciliations,
    }


def _is_handoff_message(message: Mapping[str, Any]) -> bool:
    if message.get("handoff_to") or message.get("recipient"):
        return True
    message_type = str(message.get("type") or message.get("kind") or "").lower()
    name = str(message.get("name") or "").lower()
    return "handoff" in message_type or "handoff" in name


def _review_payload_from_message(
    message: Mapping[str, Any],
    *,
    index: int,
    source: str,
) -> Dict[str, Any]:
    review = _object_mapping(message.get("review"))
    if review:
        payload = {
            "index": index,
            "name": str(message.get("name") or review.get("name") or "framework_review"),
            "reviewer": str(review.get("reviewer") or review.get("by") or source),
            "target": str(review.get("target") or review.get("target_agent") or ""),
            "status": str(review.get("status") or review.get("verdict") or ""),
            "message_type": str(message.get("type") or message.get("kind") or ""),
        }
        if review.get("notes") not in (None, "", [], {}):
            payload["notes"] = _plain_value(review.get("notes"))
        return payload
    if not (
        message.get("review_target")
        or message.get("reviewer")
        or "review" in str(message.get("type") or message.get("kind") or "").lower()
        or "review" in str(message.get("name") or "").lower()
    ):
        return {}
    return {
        "index": index,
        "name": str(message.get("name") or "framework_review"),
        "reviewer": str(message.get("reviewer") or source),
        "target": str(message.get("review_target") or message.get("target") or ""),
        "status": str(message.get("review_status") or message.get("status") or ""),
        "message_type": str(message.get("type") or message.get("kind") or ""),
        "content": _message_content(message),
    }


def _reconciliation_payload_from_message(
    message: Mapping[str, Any],
    *,
    index: int,
    source: str,
) -> Dict[str, Any]:
    reconciliation = _object_mapping(message.get("reconciliation"))
    if reconciliation:
        payload = {
            "index": index,
            "name": str(
                message.get("name")
                or reconciliation.get("name")
                or "framework_reconciliation"
            ),
            "source": source,
            "accepted_source": str(reconciliation.get("accepted_source") or ""),
            "status": str(reconciliation.get("status") or reconciliation.get("verdict") or ""),
            "message_type": str(message.get("type") or message.get("kind") or ""),
        }
        if reconciliation.get("notes") not in (None, "", [], {}):
            payload["notes"] = _plain_value(reconciliation.get("notes"))
        return payload
    if not (
        message.get("accepted_source")
        or message.get("reconciliation_status")
        or "reconciliation" in str(message.get("type") or message.get("kind") or "").lower()
        or "reconciliation" in str(message.get("name") or "").lower()
    ):
        return {}
    return {
        "index": index,
        "name": str(message.get("name") or "framework_reconciliation"),
        "source": source,
        "accepted_source": str(message.get("accepted_source") or ""),
        "status": str(message.get("reconciliation_status") or message.get("status") or ""),
        "message_type": str(message.get("type") or message.get("kind") or ""),
        "content": _message_content(message),
    }


def _framework_lifecycle_state(raw: Any) -> Dict[str, Any]:
    if not _has_framework_lifecycle_shape(raw):
        return {}
    explicit = _framework_lifecycle_explicit_trace(raw)
    phases = _framework_lifecycle_phases(raw, explicit_trace=explicit)
    state = (
        _plain_mapping(_framework_lifecycle_field(raw, "lifecycle_state"))
        or _plain_mapping(_framework_lifecycle_field(raw, "framework_state"))
        or _plain_mapping(explicit.get("state"))
    )
    metadata = {
        **_plain_mapping(explicit.get("metadata")),
        **_plain_mapping(_framework_lifecycle_field(raw, "lifecycle_metadata")),
    }
    framework = str(
        _framework_lifecycle_field(raw, "framework")
        or explicit.get("framework")
        or ""
    )
    session_id = str(
        _framework_lifecycle_field(raw, "session_id")
        or _framework_lifecycle_field(raw, "thread_id")
        or explicit.get("session_id")
        or explicit.get("thread_id")
        or ""
    )
    source = {**explicit}
    if phases:
        source["phases"] = phases
    if state:
        source["state"] = state
    if metadata:
        source["metadata"] = metadata
    if framework:
        source["framework"] = framework
    if session_id:
        source["session_id"] = session_id
    return normalize_framework_lifecycle_trace(
        source,
        name=str(source.get("name") or "framework-adapter-lifecycle-trace"),
        framework=framework or "custom",
        session_id=session_id or None,
        phases=phases or None,
        state=state,
        metadata=metadata,
    )


def _framework_lifecycle_events(raw: Any) -> List[SimulationEvent]:
    if not _has_framework_lifecycle_shape(raw):
        return []
    trace = _framework_lifecycle_state(raw)
    events: List[SimulationEvent] = []
    for index, phase in enumerate(_plain_list(trace.get("phases")), start=1):
        phase_dict = _plain_mapping(phase)
        events.append(
            SimulationEvent(
                type="framework_lifecycle_phase",
                name=str(
                    phase_dict.get("name")
                    or phase_dict.get("stage")
                    or f"phase_{index}"
                ),
                payload={**phase_dict, "sequence": index},
                metadata={
                    "kind": "framework_lifecycle_trace",
                    "source": "framework_adapter_output",
                },
            )
        )
    events.append(
        SimulationEvent(
            type="framework_lifecycle_trace",
            name=str(trace.get("name") or "framework_lifecycle_trace"),
            payload=trace,
            metadata={
                "kind": "framework_lifecycle_trace",
                "source": "framework_adapter_output",
            },
        )
    )
    return events


def _has_framework_lifecycle_shape(raw: Any) -> bool:
    raw_mapping = _object_mapping(raw)
    names = (
        "framework_lifecycle_trace",
        "lifecycle_trace",
        "framework_lifecycle",
        "lifecycle_phases",
        "framework_lifecycle_phases",
        "framework_phases",
        "lifecycle_events",
        "framework_lifecycle_events",
        "lifecycle_sessions",
        "lifecycle_state",
        "lifecycle_metadata",
        "setup_events",
        "teardown_events",
        "retry_events",
        "recovery_events",
        "cancellation_events",
        "resume_events",
    )
    if raw_mapping is not None:
        return any(raw_mapping.get(name) not in (None, "", [], {}) for name in names)
    return any(
        hasattr(raw, name) and getattr(raw, name) not in (None, "", [], {})
        for name in names
    )


def _framework_lifecycle_explicit_trace(raw: Any) -> Dict[str, Any]:
    for name in ("framework_lifecycle_trace", "lifecycle_trace", "framework_lifecycle"):
        trace = _plain_mapping(_framework_lifecycle_field(raw, name))
        if trace:
            return trace
    return {}


def _framework_lifecycle_phases(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Any]:
    trace = _plain_mapping(explicit_trace) or _framework_lifecycle_explicit_trace(raw)
    values: List[Any] = []
    for name in (
        "lifecycle_phases",
        "framework_lifecycle_phases",
        "framework_phases",
        "lifecycle_events",
        "framework_lifecycle_events",
    ):
        values.extend(_plain_list(_framework_lifecycle_field(raw, name)))
    for name in ("phases", "events", "lifecycle"):
        values.extend(_plain_list(trace.get(name)))
    for stage, field_name in (
        ("initialize", "setup_events"),
        ("teardown", "teardown_events"),
        ("retry", "retry_events"),
        ("resume", "resume_events"),
        ("cancel", "cancellation_events"),
    ):
        for item in _plain_list(_framework_lifecycle_field(raw, field_name)):
            item_dict = _plain_mapping(item)
            values.append({**item_dict, "stage": item_dict.get("stage") or stage})
    for item in _plain_list(_framework_lifecycle_field(raw, "recovery_events")):
        item_dict = _plain_mapping(item)
        values.append(
            {
                **item_dict,
                "stage": item_dict.get("stage") or "retry",
                "status": item_dict.get("status") or "recovered",
                "recovered": True,
            }
        )
    return [_plain_value(item) for item in values if _plain_value(item) not in ({}, [])]


def _framework_lifecycle_field(raw: Any, name: str) -> Any:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        return raw_mapping.get(name)
    return getattr(raw, name, None)


def _framework_trace_state(raw: Any) -> Dict[str, Any]:
    spans = _framework_trace_spans(raw)
    events = _framework_trace_event_records(raw)
    if not spans and not events:
        return {}

    framework = _framework_trace_framework(raw)
    records = [*spans, *events]
    signals = sorted(
        {
            "framework_trace",
            *[
                str(signal)
                for record in records
                for signal in _plain_list(record.get("signals"))
                if str(signal)
            ],
        }
    )
    tool_names = sorted(
        {
            _framework_trace_record_tool_name(record)
            for record in records
            if _framework_trace_record_has_tool_signal(record)
            and _framework_trace_record_tool_name(record)
        }
    )
    checkpoints = _dedupe_framework_trace_mappings(
        _plain_mapping(record.get("checkpoint"))
        for record in records
        if _plain_mapping(record.get("checkpoint"))
    )
    sessions = _dedupe_framework_trace_mappings(
        _plain_mapping(record.get("session"))
        for record in records
        if _plain_mapping(record.get("session"))
    )
    metadata = _framework_trace_metadata(raw)
    summary = {
        "span_count": len(spans),
        "event_count": len(events),
        "signal_count": len(signals),
        "signals": signals,
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "model_span_count": _framework_trace_signal_count(records, "model"),
        "tool_span_count": _framework_trace_signal_count(records, "tool"),
        "retrieval_span_count": _framework_trace_signal_count(records, "retrieval"),
        "memory_span_count": _framework_trace_signal_count(records, "memory"),
        "state_span_count": _framework_trace_signal_count(records, "state"),
        "latency_span_count": _framework_trace_signal_count(records, "latency"),
        "cost_span_count": _framework_trace_signal_count(records, "cost"),
        "error_count": sum(
            1
            for record in records
            if record.get("error") or _framework_trace_record_has_signal(record, "error")
        ),
        "checkpoint_count": len(checkpoints),
        "session_count": len(sessions),
    }
    state = {
        "kind": "framework_trace",
        "framework": framework,
        "spans": spans,
        "events": events,
        "checkpoints": checkpoints,
        "sessions": sessions,
        "signals": signals,
        "state": _framework_trace_runtime_state(raw),
        "summary": summary,
        "metadata": metadata,
    }
    adapter_spec = _framework_trace_adapter_spec(raw)
    if adapter_spec:
        state["adapter_conformance"] = normalize_framework_adapter_conformance(
            framework,
            records,
            required_signals=adapter_spec.get("required_signals")
            or adapter_spec.get("signals"),
            required_mappings=(
                adapter_spec.get("required_mappings")
                or adapter_spec.get("mappings")
                or adapter_spec.get("field_mappings")
            ),
            metadata=adapter_spec,
        )
    return state


def _framework_trace_events(raw: Any) -> List[SimulationEvent]:
    state = _framework_trace_state(raw)
    if not state:
        return []
    events: List[SimulationEvent] = []
    for index, span in enumerate(_plain_list(state.get("spans")), start=1):
        span_dict = _plain_mapping(span)
        events.append(
            SimulationEvent(
                type="framework_trace_span",
                name=str(span_dict.get("name") or span_dict.get("id") or f"span_{index}"),
                payload={**span_dict, "sequence": index},
                metadata={
                    "kind": "framework_trace",
                    "source": "framework_adapter_output",
                    "signals": _plain_list(span_dict.get("signals")),
                },
            )
        )
    for index, event in enumerate(_plain_list(state.get("events")), start=1):
        event_dict = _plain_mapping(event)
        events.append(
            SimulationEvent(
                type="framework_trace_event",
                name=str(event_dict.get("name") or event_dict.get("id") or f"event_{index}"),
                payload={**event_dict, "sequence": index},
                metadata={
                    "kind": "framework_trace",
                    "source": "framework_adapter_output",
                    "signals": _plain_list(event_dict.get("signals")),
                },
            )
        )
    events.append(
        SimulationEvent(
            type="framework_trace",
            name="framework_trace",
            payload=state,
            metadata={
                "kind": "framework_trace",
                "source": "framework_adapter_output",
            },
        )
    )
    return events


def _framework_trace_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(_framework_trace_spans(raw), start=1):
        if not _framework_trace_record_has_tool_call_shape(record):
            continue
        name = _framework_trace_record_tool_name(record)
        if not name:
            continue
        call_id = _framework_trace_record_call_id(record, index=index)
        arguments = _framework_trace_record_arguments(record)
        signature = f"{call_id}:{name}"
        if signature in seen:
            continue
        seen.add(signature)
        calls.append(
            {
                "id": call_id,
                "type": "framework_trace_tool_call",
                "name": name,
                "arguments": arguments,
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return calls


def _framework_trace_tool_responses(raw: Any) -> List[Dict[str, Any]]:
    responses: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(_framework_trace_spans(raw), start=1):
        if not _framework_trace_record_has_tool_signal(record):
            continue
        output = _framework_trace_record_output(record)
        error = _framework_trace_record_error(record)
        if output in (None, "", [], {}) and not error and not (
            _framework_trace_record_has_signal(record, "tool_result")
            or _framework_trace_record_has_signal(record, "mcp_tool_result")
            or _framework_trace_record_has_signal(record, "tool_error")
            or _framework_trace_record_has_signal(record, "mcp_tool_error")
        ):
            continue
        name = _framework_trace_record_tool_name(record)
        if not name:
            continue
        call_id = _framework_trace_record_call_id(record, index=index)
        signature = f"{call_id}:{name}:{bool(error)}"
        if signature in seen:
            continue
        seen.add(signature)
        responses.append(
            {
                "id": f"{call_id}_response",
                "tool_call_id": call_id,
                "name": name,
                "content": _plain_value(error if error else output),
                "success": not bool(error),
                "result": _plain_value(output),
                "error": _plain_value(error),
            }
        )
    return responses


def _framework_trace_spans(raw: Any) -> List[Dict[str, Any]]:
    if not _has_framework_trace_shape(raw):
        return []
    framework = _framework_trace_framework(raw)
    spans: List[Dict[str, Any]] = []
    for trace_export in _framework_trace_exports(raw):
        spans.extend(
            _plain_mapping(span)
            for span in normalize_framework_trace_export(
                trace_export,
                framework=framework,
            )
            if _plain_mapping(span)
        )
    span_records = _framework_trace_span_records(raw)
    if span_records:
        spans.extend(
            _plain_mapping(span)
            for span in normalize_framework_trace_events(
                framework,
                span_records,
                category="span",
            )
            if _plain_mapping(span)
        )
    return _dedupe_framework_trace_records(spans)


def _framework_trace_event_records(raw: Any) -> List[Dict[str, Any]]:
    if not _has_framework_trace_shape(raw):
        return []
    framework = _framework_trace_framework(raw)
    records = _framework_trace_raw_event_records(raw)
    if not records:
        return []
    return _dedupe_framework_trace_records(
        _plain_mapping(event)
        for event in normalize_framework_trace_events(
            framework,
            records,
            category="event",
        )
        if _plain_mapping(event)
    )


def _has_framework_trace_shape(raw: Any) -> bool:
    if raw in (None, "", [], {}):
        return False
    if isinstance(raw, (list, tuple)):
        return any(_looks_like_framework_trace_record(item) for item in raw)
    raw_mapping = _object_mapping(raw)
    explicit_names = (
        "framework_trace",
        "framework_trace_export",
        "trace_export",
        "traceai_export",
        "otel_trace_export",
        "otlp_export",
        "opentelemetry_export",
        "open_telemetry_export",
        "framework_spans",
        "trace_spans",
        "span_records",
        "framework_events",
        "trace_events",
        "framework_trace_events",
    )
    if raw_mapping is None:
        return any(
            hasattr(raw, name) and getattr(raw, name) not in (None, "", [], {})
            for name in explicit_names
        )
    if any(raw_mapping.get(name) not in (None, "", [], {}) for name in explicit_names):
        return True
    if any(
        raw_mapping.get(name) not in (None, "", [], {})
        for name in ("resourceSpans", "resource_spans", "scopeSpans", "scope_spans")
    ):
        return True
    if _looks_like_framework_trace_record(raw_mapping):
        return True
    if not _framework_trace_has_marker(raw_mapping):
        return False
    return any(
        raw_mapping.get(name) not in (None, "", [], {})
        for name in ("spans", "events", "records", "items", "results")
    )


def _framework_trace_exports(raw: Any) -> List[Any]:
    exports: List[Any] = []
    for name in (
        "framework_trace_export",
        "trace_export",
        "traceai_export",
        "otel_trace_export",
        "otlp_export",
        "opentelemetry_export",
        "open_telemetry_export",
    ):
        value = _plain_value(_framework_trace_field(raw, name))
        if value not in (None, "", [], {}):
            exports.append(value)
    explicit = _framework_trace_explicit_payload(raw)
    if explicit and _framework_trace_payload_is_export(explicit):
        exports.append(explicit)
    raw_mapping = _object_mapping(raw)
    if raw_mapping and _framework_trace_payload_is_export(raw_mapping):
        exports.append(raw_mapping)
    return _dedupe_framework_trace_values(exports)


def _framework_trace_span_records(raw: Any) -> List[Any]:
    if isinstance(raw, (list, tuple)):
        return [_plain_value(item) for item in raw]
    records: List[Any] = []
    explicit = _framework_trace_explicit_payload(raw)
    for key in ("spans", "records", "items", "results"):
        records.extend(_plain_list(explicit.get(key)))
    for name in ("framework_spans", "trace_spans", "span_records"):
        records.extend(_plain_list(_framework_trace_field(raw, name)))
    raw_mapping = _object_mapping(raw)
    if raw_mapping and _framework_trace_has_marker(raw_mapping):
        for key in ("spans", "records", "items", "results"):
            records.extend(_plain_list(raw_mapping.get(key)))
    if raw_mapping and _looks_like_framework_trace_record(raw_mapping):
        records.append(raw_mapping)
    return [
        _plain_value(record)
        for record in records
        if _plain_value(record) not in (None, "", [], {})
    ]


def _framework_trace_raw_event_records(raw: Any) -> List[Any]:
    records: List[Any] = []
    explicit = _framework_trace_explicit_payload(raw)
    for key in ("events", "framework_events", "trace_events"):
        records.extend(_plain_list(explicit.get(key)))
    for name in ("framework_events", "trace_events", "framework_trace_events"):
        records.extend(_plain_list(_framework_trace_field(raw, name)))
    raw_mapping = _object_mapping(raw)
    if raw_mapping and _framework_trace_has_marker(raw_mapping):
        records.extend(_plain_list(raw_mapping.get("events")))
    return [
        _plain_value(record)
        for record in records
        if _plain_value(record) not in (None, "", [], {})
    ]


def _framework_trace_explicit_payload(raw: Any) -> Dict[str, Any]:
    for name in ("framework_trace", "trace"):
        value = _plain_mapping(_framework_trace_field(raw, name))
        if value:
            return value
    return {}


def _framework_trace_payload_is_export(value: Mapping[str, Any]) -> bool:
    return any(
        value.get(name) not in (None, "", [], {})
        for name in (
            "resourceSpans",
            "resource_spans",
            "scopeSpans",
            "scope_spans",
            "traces",
        )
    )


def _framework_trace_framework(raw: Any) -> str:
    explicit = _framework_trace_explicit_payload(raw)
    metadata = _plain_mapping(_framework_trace_field(raw, "metadata"))
    value = (
        _framework_trace_field(raw, "framework")
        or _framework_trace_field(raw, "trace_framework")
        or _framework_trace_field(raw, "trace_provider")
        or explicit.get("framework")
        or explicit.get("trace_provider")
        or metadata.get("framework")
        or metadata.get("trace_provider")
        or "generic"
    )
    key = _framework_trace_key(value)
    if key in {"otel", "opentelemetry", "open_telemetry", "otlp"}:
        return "opentelemetry"
    if key in {"traceai", "futureagi", "future_agi"}:
        return "traceai"
    return str(value or "generic")


def _framework_trace_metadata(raw: Any) -> Dict[str, Any]:
    explicit = _framework_trace_explicit_payload(raw)
    metadata = {
        **_plain_mapping(explicit.get("metadata")),
        **_plain_mapping(_framework_trace_field(raw, "trace_metadata")),
        **_plain_mapping(_framework_trace_field(raw, "metadata")),
    }
    if _framework_trace_exports(raw):
        metadata.setdefault("trace_export", {})["source"] = "framework_adapter_output"
    return metadata


def _framework_trace_runtime_state(raw: Any) -> Dict[str, Any]:
    explicit = _framework_trace_explicit_payload(raw)
    return (
        _plain_mapping(_framework_trace_field(raw, "framework_state"))
        or _plain_mapping(_framework_trace_field(raw, "trace_state"))
        or _plain_mapping(explicit.get("state"))
    )


def _framework_trace_adapter_spec(raw: Any) -> Dict[str, Any]:
    explicit = _framework_trace_explicit_payload(raw)
    metadata = _plain_mapping(_framework_trace_field(raw, "metadata"))
    explicit_metadata = _plain_mapping(explicit.get("metadata"))
    spec: Dict[str, Any] = {}
    for source in (
        _plain_mapping(metadata.get("adapter_conformance")),
        _plain_mapping(metadata.get("adapter_spec")),
        _plain_mapping(metadata.get("framework_adapter")),
        _plain_mapping(explicit_metadata.get("adapter_conformance")),
        _plain_mapping(explicit_metadata.get("adapter_spec")),
        _plain_mapping(explicit.get("adapter_conformance")),
        _plain_mapping(explicit.get("adapter_spec")),
        _plain_mapping(_framework_trace_field(raw, "adapter_conformance")),
        _plain_mapping(_framework_trace_field(raw, "adapter_spec")),
        _plain_mapping(_framework_trace_field(raw, "framework_adapter")),
    ):
        spec.update(source)
    required_signals = (
        _framework_trace_field(raw, "adapter_required_signals")
        or explicit.get("adapter_required_signals")
        or metadata.get("adapter_required_signals")
    )
    if required_signals not in (None, "", [], {}):
        spec["required_signals"] = _plain_list(required_signals)
    required_mappings = (
        _framework_trace_field(raw, "adapter_required_mappings")
        or explicit.get("adapter_required_mappings")
        or metadata.get("adapter_required_mappings")
    )
    if required_mappings not in (None, "", [], {}):
        spec["required_mappings"] = _plain_mapping(required_mappings)
    return {key: _plain_value(value) for key, value in spec.items() if value not in (None, "", [], {})}


def _framework_trace_field(raw: Any, name: str) -> Any:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        return raw_mapping.get(name)
    return getattr(raw, name, None)


def _framework_trace_has_marker(value: Mapping[str, Any]) -> bool:
    markers = {
        "framework_trace",
        "traceai",
        "futureagi",
        "future_agi",
        "otel",
        "otlp",
        "opentelemetry",
        "open_telemetry",
        "langchain",
        "langgraph",
        "openai_agents",
        "crewai",
        "autogen",
        "llamaindex",
        "dspy",
        "livekit",
        "pipecat",
    }
    for key in ("kind", "type", "framework", "protocol", "telemetry", "trace_provider", "provider"):
        marker = _framework_trace_key(value.get(key))
        if marker in markers:
            return True
    metadata = _plain_mapping(value.get("metadata"))
    return bool(metadata) and _framework_trace_has_marker(metadata)


def _looks_like_framework_trace_record(value: Any) -> bool:
    record = _plain_mapping(value)
    if not record:
        return False
    if any(
        record.get(key) not in (None, "", [], {})
        for key in ("spanId", "span_id", "traceId", "trace_id", "parentSpanId", "parent_span_id")
    ):
        return True
    if record.get("run_id") not in (None, "", [], {}) and any(
        key in record for key in ("name", "type", "kind", "event", "attributes", "span_data", "status")
    ):
        return True
    if any(key in record for key in ("attributes", "attrs", "span_data", "resource", "scope")) and any(
        key in record for key in ("name", "type", "kind", "event", "status")
    ):
        return True
    if _framework_trace_has_marker(record) and any(
        record.get(key) not in (None, "", [], {})
        for key in ("name", "attributes", "spans", "events", "records")
    ):
        return True
    return False


def _framework_trace_record_has_signal(record: Mapping[str, Any], signal: str) -> bool:
    normalized = _framework_trace_key(signal)
    return normalized in {
        _framework_trace_key(item)
        for item in _plain_list(record.get("signals"))
        if _framework_trace_key(item)
    }


def _framework_trace_record_has_tool_signal(record: Mapping[str, Any]) -> bool:
    attributes = _plain_mapping(record.get("attributes"))
    if attributes.get("mcp.tool.name") or attributes.get("gen_ai.tool.name") or attributes.get("tool.name"):
        return True
    return any(
        _framework_trace_record_has_signal(record, signal)
        for signal in (
            "tool",
            "tool_call",
            "tool_result",
            "tool_error",
            "mcp_tool_call",
            "mcp_tool_result",
            "mcp_tool_error",
        )
    )


def _framework_trace_record_has_tool_call_shape(record: Mapping[str, Any]) -> bool:
    if not _framework_trace_record_has_tool_signal(record):
        return False
    text = " ".join(
        [
            str(record.get("type") or ""),
            str(record.get("name") or ""),
            " ".join(str(signal) for signal in _plain_list(record.get("signals"))),
        ]
    ).lower()
    return not (
        ("schema" in text or "tools/list" in text)
        and not any(token in text for token in ("call", "result", "error"))
    )


def _framework_trace_record_tool_name(record: Mapping[str, Any]) -> str:
    attributes = _plain_mapping(record.get("attributes"))
    event = _plain_mapping(record.get("framework_event"))
    for source in (record, event, attributes):
        for key in ("tool_name", "tool", "name", "mcp.tool.name", "gen_ai.tool.name", "tool.name"):
            value = source.get(key)
            if value in (None, "", [], {}):
                continue
            parsed = _framework_trace_tool_name_from_span_name(str(value))
            if parsed:
                return parsed
            if key == "name" and source is record:
                continue
            return str(value)
    return _framework_trace_tool_name_from_span_name(str(record.get("name") or ""))


def _framework_trace_tool_name_from_span_name(name: str) -> str:
    lowered = name.lower()
    prefixes = (
        "mcp tool result ",
        "mcp tool error ",
        "mcp tool call ",
        "tool result ",
        "tool error ",
        "tool call ",
        "function call ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return name[len(prefix):].strip(" :")
    return ""


def _framework_trace_record_call_id(record: Mapping[str, Any], *, index: int) -> str:
    attributes = _plain_mapping(record.get("attributes"))
    return str(
        record.get("tool_call_id")
        or record.get("call_id")
        or attributes.get("tool_call_id")
        or attributes.get("mcp.tool.call_id")
        or record.get("span_id")
        or record.get("id")
        or f"framework_trace_tool_call_{index}"
    )


def _framework_trace_record_arguments(record: Mapping[str, Any]) -> Any:
    attributes = _plain_mapping(record.get("attributes"))
    return _plain_value(
        record.get("arguments")
        if "arguments" in record
        else record.get(
            "input",
            attributes.get(
                "arguments",
                attributes.get("mcp.tool.arguments", attributes.get("gen_ai.tool.arguments", {})),
            ),
        )
    )


def _framework_trace_record_output(record: Mapping[str, Any]) -> Any:
    attributes = _plain_mapping(record.get("attributes"))
    return _plain_value(
        record.get("result")
        if "result" in record
        else record.get(
            "output",
            attributes.get(
                "result",
                attributes.get("mcp.tool.result", attributes.get("gen_ai.tool.result")),
            ),
        )
    )


def _framework_trace_record_error(record: Mapping[str, Any]) -> Any:
    attributes = _plain_mapping(record.get("attributes"))
    error = record.get("error") or attributes.get("error") or attributes.get("exception")
    if error:
        return _plain_value(error)
    if _framework_trace_record_has_signal(record, "tool_error") or _framework_trace_record_has_signal(record, "mcp_tool_error"):
        return "tool_error"
    return None


def _framework_trace_signal_count(records: Sequence[Mapping[str, Any]], signal: str) -> int:
    return sum(1 for record in records if _framework_trace_record_has_signal(record, signal))


def _dedupe_framework_trace_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        record_dict = _plain_mapping(record)
        if not record_dict:
            continue
        signature = json.dumps(
            {
                "id": record_dict.get("id"),
                "span_id": record_dict.get("span_id"),
                "name": record_dict.get("name"),
                "type": record_dict.get("type"),
            },
            sort_keys=True,
            default=str,
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(record_dict)
    return deduped


def _dedupe_framework_trace_values(values: Iterable[Any]) -> List[Any]:
    deduped: List[Any] = []
    seen: set[str] = set()
    for value in values:
        signature = json.dumps(_plain_value(value), sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(value)
    return deduped


def _dedupe_framework_trace_mappings(values: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        mapping = _plain_mapping(value)
        signature = json.dumps(mapping, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(mapping)
    return deduped


def _framework_trace_key(value: Any) -> str:
    aliases = {
        "llm": "model",
        "generation": "model",
        "chat_model": "model",
        "function": "tool",
        "function_call": "tool",
        "tool_call": "tool",
        "tool_output": "tool_result",
        "exception": "error",
        "failure": "error",
        "duration": "latency",
        "duration_ms": "latency",
        "tokens": "cost",
        "usage": "cost",
    }
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")
    return aliases.get(normalized, normalized)


def _mcp_tool_session_state(raw: Any) -> Dict[str, Any]:
    spans = _mcp_tool_session_spans(raw)
    if not spans:
        return {}
    tool_calls = _mcp_tool_session_tool_calls(raw)
    tool_responses = _mcp_tool_session_tool_responses(raw)
    tool_names = sorted(
        {
            _mcp_span_tool_name(span)
            for span in spans
            if _mcp_span_has_tool_signal(span) and _mcp_span_tool_name(span)
        }
    )
    server_names = sorted(
        {
            str(_plain_mapping(span.get("attributes")).get("mcp.server.name") or "")
            for span in spans
            if _plain_mapping(span.get("attributes")).get("mcp.server.name")
        }
    )
    session_ids = sorted(
        {
            str(_plain_mapping(span.get("attributes")).get("mcp.session.id") or "")
            for span in spans
            if _plain_mapping(span.get("attributes")).get("mcp.session.id")
        }
    )
    record_types = sorted(
        {
            _mcp_span_event_type(span)
            for span in spans
            if _mcp_span_event_type(span)
        }
    )
    signals = sorted(
        {
            str(signal)
            for span in spans
            for signal in _plain_list(span.get("signals"))
            if str(signal)
        }
    )
    result_count = sum(1 for span in spans if _mcp_span_has_signal(span, "mcp_tool_result"))
    error_count = sum(1 for span in spans if _mcp_span_has_signal(span, "mcp_tool_error"))
    schema_count = sum(1 for span in spans if _mcp_span_has_signal(span, "mcp_tool_schema"))
    resource_count = sum(1 for span in spans if _mcp_span_event_type(span) == "mcp_resource")
    server_count = sum(1 for span in spans if _mcp_span_event_type(span) == "mcp_server")
    summary = {
        "span_count": len(spans),
        "server_count": server_count,
        "schema_count": schema_count,
        "resource_count": resource_count,
        "call_count": len(tool_calls),
        "result_count": result_count,
        "error_count": error_count,
        "tool_response_count": len(tool_responses),
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "server_names": server_names,
        "session_ids": session_ids,
        "record_types": record_types,
        "signals": signals,
    }
    return {
        "kind": "mcp_tool_session",
        "framework": _mcp_tool_session_framework(raw),
        "server_name": next(iter(server_names), ""),
        "session_id": next(iter(session_ids), ""),
        **summary,
        "spans": spans,
        "tool_calls": tool_calls,
        "tool_responses": tool_responses,
        "summary": summary,
    }


def _mcp_tool_session_events(raw: Any) -> List[SimulationEvent]:
    spans = _mcp_tool_session_spans(raw)
    if not spans:
        return []
    events: List[SimulationEvent] = []
    for index, span in enumerate(spans, start=1):
        span_dict = _plain_mapping(span)
        event_type = _mcp_span_event_type(span_dict)
        events.append(
            SimulationEvent(
                type=event_type,
                name=str(span_dict.get("name") or event_type),
                payload={**span_dict, "sequence": index},
                metadata={
                    "kind": "mcp_tool_session",
                    "source": "framework_adapter_output",
                },
            )
        )
    state = _mcp_tool_session_state(raw)
    events.append(
        SimulationEvent(
            type="mcp_tool_session",
            name=str(state.get("server_name") or "mcp_tool_session"),
            payload=state,
            metadata={
                "kind": "mcp_tool_session",
                "source": "framework_adapter_output",
            },
        )
    )
    return events


def _mcp_tool_session_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, span in enumerate(_mcp_tool_session_spans(raw), start=1):
        if not _mcp_span_has_signal(span, "mcp_tool_call"):
            continue
        name = _mcp_span_tool_name(span)
        if not name:
            continue
        call_id = _mcp_span_call_id(span, index=index)
        signature = f"{call_id}:{name}"
        if signature in seen:
            continue
        seen.add(signature)
        arguments = _mcp_span_arguments(span)
        calls.append(
            {
                "id": call_id,
                "type": "mcp_tool_call",
                "name": name,
                "arguments": arguments,
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return calls


def _mcp_tool_session_tool_responses(raw: Any) -> List[Dict[str, Any]]:
    responses: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, span in enumerate(_mcp_tool_session_spans(raw), start=1):
        is_result = _mcp_span_has_signal(span, "mcp_tool_result")
        is_error = _mcp_span_has_signal(span, "mcp_tool_error")
        output = _mcp_span_output(span)
        error = _mcp_span_error(span)
        if not is_result and not is_error and output in (None, "", [], {}) and not error:
            continue
        name = _mcp_span_tool_name(span)
        if not name:
            continue
        call_id = _mcp_span_call_id(span, index=index)
        signature = f"{call_id}:{name}:{bool(error)}"
        if signature in seen:
            continue
        seen.add(signature)
        content = error if error else output
        responses.append(
            {
                "id": f"{call_id}_response",
                "tool_call_id": call_id,
                "name": name,
                "content": _plain_value(content),
                "success": not bool(error),
                "result": _plain_value(output),
                "error": _plain_value(error),
            }
        )
    return responses


def _mcp_tool_session_spans(raw: Any) -> List[Dict[str, Any]]:
    if not _has_mcp_tool_session_shape(raw):
        return []
    session_export = _mcp_tool_session_export(raw)
    if session_export in (None, "", [], {}):
        return []
    return [
        _plain_mapping(span)
        for span in normalize_mcp_tool_session_export(
            session_export,
            framework=_mcp_tool_session_framework(raw),
            server_name=_mcp_tool_session_server_name(raw) or None,
        )
        if _plain_mapping(span)
    ]


def _has_mcp_tool_session_shape(raw: Any) -> bool:
    if raw in (None, "", [], {}):
        return False
    if isinstance(raw, (list, tuple)):
        return any(_looks_like_mcp_session_record(item) for item in raw)
    raw_mapping = _object_mapping(raw)
    explicit_names = (
        "mcp_tool_session",
        "mcp_session",
        "mcp_sessions",
        "mcp_records",
        "mcp_events",
        "mcp_messages",
        "mcp_requests",
        "mcp_responses",
        "mcp_tools",
        "mcp_tool_specs",
        "mcp_tool_schemas",
        "mcp_resources",
        "mcp_resource_templates",
        "mcp_calls",
        "mcp_tool_calls",
        "mcp_tool_results",
        "tool_session_export",
        "tool_protocol_trace",
    )
    if raw_mapping is None:
        return any(
            hasattr(raw, name) and getattr(raw, name) not in (None, "", [], {})
            for name in explicit_names
        )
    if any(raw_mapping.get(name) not in (None, "", [], {}) for name in explicit_names):
        return True
    if _looks_like_mcp_session_record(raw_mapping):
        return True
    if not _mcp_has_protocol_marker(raw_mapping):
        return False
    protocol_fields = (
        "sessions",
        "runs",
        "tools",
        "tool_specs",
        "toolSchemas",
        "tool_schemas",
        "schemas",
        "available_tools",
        "calls",
        "tool_calls",
        "invocations",
        "executions",
        "tool_invocations",
        "events",
        "records",
        "requests",
        "responses",
        "items",
        "resources",
        "resource_templates",
    )
    if any(raw_mapping.get(name) not in (None, "", [], {}) for name in protocol_fields):
        return True
    return _mcp_jsonrpc_sequence(raw_mapping.get("messages"))


def _mcp_tool_session_export(raw: Any) -> Any:
    if isinstance(raw, (list, tuple)):
        return [_plain_value(item) for item in raw]
    for name in (
        "mcp_tool_session",
        "mcp_session",
        "tool_session_export",
        "tool_protocol_trace",
    ):
        value = _mcp_tool_session_field(raw, name)
        plain = _plain_value(value)
        if plain in (None, "", [], {}):
            continue
        if isinstance(plain, Mapping):
            return _mcp_tool_session_payload_with_defaults(dict(plain), raw)
        return plain

    raw_mapping = _object_mapping(raw) or {}
    payload = _mcp_tool_session_payload_aliases(raw_mapping)
    return _mcp_tool_session_payload_with_defaults(payload, raw)


def _mcp_tool_session_payload_aliases(source: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    alias_targets = {
        "mcp_sessions": "sessions",
        "mcp_records": "records",
        "mcp_events": "events",
        "mcp_messages": "messages",
        "mcp_requests": "requests",
        "mcp_responses": "responses",
        "mcp_tools": "tools",
        "mcp_tool_specs": "tools",
        "mcp_tool_schemas": "tools",
        "mcp_resources": "resources",
        "mcp_resource_templates": "resource_templates",
        "mcp_calls": "calls",
        "mcp_tool_calls": "calls",
        "mcp_tool_results": "calls",
    }
    for source_key, target_key in alias_targets.items():
        value = source.get(source_key)
        if value not in (None, "", [], {}):
            payload[target_key] = _plain_value(value)
    for key in (
        "sessions",
        "runs",
        "tools",
        "tool_specs",
        "toolSchemas",
        "tool_schemas",
        "schemas",
        "available_tools",
        "calls",
        "tool_calls",
        "invocations",
        "executions",
        "tool_invocations",
        "events",
        "records",
        "requests",
        "responses",
        "items",
        "messages",
        "resources",
        "resource_templates",
        "server",
        "server_name",
        "serverName",
        "session_id",
        "sessionId",
        "protocol_version",
        "protocolVersion",
    ):
        value = source.get(key)
        if value not in (None, "", [], {}) and key not in payload:
            payload[key] = _plain_value(value)
    return payload


def _mcp_tool_session_payload_with_defaults(
    payload: Mapping[str, Any],
    raw: Any,
) -> Dict[str, Any]:
    normalized = _mcp_tool_session_payload_aliases(payload)
    for key, value in _plain_mapping(payload).items():
        if value not in (None, "", [], {}) and key not in normalized:
            normalized[key] = _plain_value(value)
    server_name = _mcp_tool_session_server_name(raw)
    session_id = _mcp_tool_session_session_id(raw)
    framework = _mcp_tool_session_framework(raw)
    if server_name:
        normalized.setdefault("server_name", server_name)
    if session_id:
        normalized.setdefault("session_id", session_id)
    if framework:
        normalized.setdefault("framework", framework)
    return normalized


def _mcp_tool_session_framework(raw: Any) -> str:
    value = (
        _mcp_tool_session_field(raw, "framework")
        or _mcp_tool_session_field(raw, "protocol")
        or _plain_mapping(_mcp_tool_session_field(raw, "metadata")).get("framework")
    )
    text = str(value or "")
    return "mcp" if _mcp_protocol_key(text) in {"mcp", "modelcontextprotocol"} else (text or "mcp")


def _mcp_tool_session_server_name(raw: Any) -> str:
    server = _mcp_tool_session_field(raw, "server")
    server_mapping = _plain_mapping(server)
    value = (
        _mcp_tool_session_field(raw, "mcp_server_name")
        or _mcp_tool_session_field(raw, "server_name")
        or _mcp_tool_session_field(raw, "serverName")
        or server_mapping.get("name")
        or _plain_mapping(_mcp_tool_session_field(raw, "metadata")).get("server_name")
    )
    return str(value or "")


def _mcp_tool_session_session_id(raw: Any) -> str:
    value = (
        _mcp_tool_session_field(raw, "mcp_session_id")
        or _mcp_tool_session_field(raw, "session_id")
        or _mcp_tool_session_field(raw, "sessionId")
        or _mcp_tool_session_field(raw, "thread_id")
        or _plain_mapping(_mcp_tool_session_field(raw, "metadata")).get("session_id")
    )
    return str(value or "")


def _mcp_tool_session_field(raw: Any, name: str) -> Any:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        return raw_mapping.get(name)
    return getattr(raw, name, None)


def _mcp_has_protocol_marker(value: Mapping[str, Any]) -> bool:
    for key in ("framework", "protocol", "type", "kind"):
        if _mcp_protocol_key(value.get(key)) in {"mcp", "modelcontextprotocol"}:
            return True
    metadata = _plain_mapping(value.get("metadata"))
    for key in ("framework", "protocol"):
        if _mcp_protocol_key(metadata.get(key)) in {"mcp", "modelcontextprotocol"}:
            return True
    return False


def _mcp_protocol_key(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _looks_like_mcp_session_record(value: Any) -> bool:
    record = _plain_mapping(value)
    if not record:
        return False
    method = str(record.get("method") or "").lower()
    if method.startswith("tools/") or method.startswith("resources/"):
        return True
    if record.get("jsonrpc") and (record.get("result") or record.get("error")):
        return True
    params = _plain_mapping(record.get("params"))
    if params.get("name") and ("arguments" in params or "input" in params):
        return True
    result = _plain_mapping(record.get("result"))
    return bool(result.get("tools") or result.get("resources"))


def _mcp_jsonrpc_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    return any(_looks_like_mcp_session_record(item) for item in value)


def _mcp_span_event_type(span: Mapping[str, Any]) -> str:
    span_type = str(span.get("type") or "")
    if span_type.startswith("mcp_"):
        return span_type
    if _mcp_span_has_signal(span, "mcp_tool_error"):
        return "mcp_tool_error"
    if _mcp_span_has_signal(span, "mcp_tool_result"):
        return "mcp_tool_result"
    if _mcp_span_has_signal(span, "mcp_tool_call"):
        return "mcp_tool_call"
    if _mcp_span_has_signal(span, "mcp_tool_schema"):
        return "mcp_tool_schema"
    if _mcp_span_has_signal(span, "mcp_resource"):
        return "mcp_resource"
    if _mcp_span_has_signal(span, "mcp_server"):
        return "mcp_server"
    return "mcp_tool_span"


def _mcp_span_has_signal(span: Mapping[str, Any], signal: str) -> bool:
    normalized = signal.lower()
    return normalized in {
        str(item).lower()
        for item in _plain_list(span.get("signals"))
        if str(item)
    }


def _mcp_span_has_tool_signal(span: Mapping[str, Any]) -> bool:
    return any(
        _mcp_span_has_signal(span, signal)
        for signal in (
            "mcp_tool_schema",
            "mcp_tool_call",
            "mcp_tool_result",
            "mcp_tool_error",
        )
    )


def _mcp_span_tool_name(span: Mapping[str, Any]) -> str:
    event = _plain_mapping(span.get("framework_event"))
    attributes = _plain_mapping(span.get("attributes"))
    for source in (span, event, attributes):
        for key in ("tool_name", "tool", "mcp.tool.name", "gen_ai.tool.name"):
            value = source.get(key)
            if value not in (None, "", [], {}):
                return str(value)
    name = str(span.get("name") or "")
    lowered = name.lower()
    for prefix in (
        "mcp tool result ",
        "mcp tool error ",
        "mcp tool call ",
        "mcp tool schema ",
    ):
        if lowered.startswith(prefix):
            return name[len(prefix):].strip()
    return ""


def _mcp_span_call_id(span: Mapping[str, Any], *, index: int) -> str:
    attributes = _plain_mapping(span.get("attributes"))
    return str(
        attributes.get("mcp.request.id")
        or span.get("call_id")
        or span.get("tool_call_id")
        or span.get("id")
        or span.get("span_id")
        or f"mcp_tool_call_{index}"
    )


def _mcp_span_arguments(span: Mapping[str, Any]) -> Dict[str, Any]:
    attributes = _plain_mapping(span.get("attributes"))
    return (
        _plain_mapping(span.get("arguments"))
        or _plain_mapping(span.get("input"))
        or _plain_mapping(attributes.get("arguments"))
        or _plain_mapping(attributes.get("mcp.tool.arguments"))
    )


def _mcp_span_output(span: Mapping[str, Any]) -> Any:
    attributes = _plain_mapping(span.get("attributes"))
    if span.get("output") not in (None, "", [], {}):
        return _plain_value(span.get("output"))
    for key in ("result", "mcp.tool.result"):
        if attributes.get(key) not in (None, "", [], {}):
            return _plain_value(attributes.get(key))
    return None


def _mcp_span_error(span: Mapping[str, Any]) -> Any:
    attributes = _plain_mapping(span.get("attributes"))
    return _plain_value(
        span.get("error")
        or attributes.get("error")
        or attributes.get("exception")
    )


def _a2a_protocol_state(raw: Any) -> Dict[str, Any]:
    if not _has_a2a_protocol_shape(raw):
        return {}
    agent_cards = _a2a_agent_cards(raw)
    messages = _a2a_messages(raw)
    tasks = _a2a_tasks(raw)
    artifacts = _a2a_artifacts(raw, tasks=tasks)
    protocol_events = _a2a_protocol_events_payload(raw)
    if not any((agent_cards, messages, tasks, artifacts, protocol_events)):
        return {}

    parts = [
        *[
            part
            for message in messages
            for part in _plain_list(message.get("parts"))
        ],
        *[
            part
            for artifact in artifacts
            for part in _plain_list(artifact.get("parts"))
        ],
    ]
    states = sorted(
        {
            str(task.get("state") or "")
            for task in tasks
            if task.get("state")
        }
    )
    event_types = sorted(
        {
            str(event.get("type") or "")
            for event in protocol_events
            if event.get("type")
        }
    )
    task_ids = sorted(
        {
            str(value)
            for value in [
                *[task.get("id") for task in tasks],
                *[message.get("task_id") for message in messages],
                *[event.get("task_id") for event in protocol_events],
            ]
            if value not in (None, "", [], {})
        }
    )
    context_ids = sorted(
        {
            str(value)
            for value in [
                *[task.get("context_id") for task in tasks],
                *[message.get("context_id") for message in messages],
                *[event.get("context_id") for event in protocol_events],
            ]
            if value not in (None, "", [], {})
        }
    )
    skill_names = sorted(
        {
            str(skill.get("name") or skill.get("id") or "")
            for card in agent_cards
            for skill in _plain_list(card.get("skills"))
            if _plain_mapping(skill).get("name") or _plain_mapping(skill).get("id")
        }
    )
    agent_names = sorted(
        {
            str(card.get("name") or "")
            for card in agent_cards
            if card.get("name")
        }
    )
    roles = sorted(
        {
            str(message.get("role") or "")
            for message in messages
            if message.get("role")
        }
    )
    terminal_states = {"completed", "failed", "canceled", "cancelled", "rejected"}
    input_states = {"input_required", "input-required", "auth_required", "auth-required"}
    summary = {
        "agent_card_count": len(agent_cards),
        "skill_count": len(skill_names),
        "message_count": len(messages),
        "task_count": len(tasks),
        "artifact_count": len(artifacts),
        "protocol_event_count": len(protocol_events),
        "part_count": len(parts),
        "text_part_count": sum(1 for part in parts if _plain_mapping(part).get("kind") == "text"),
        "data_part_count": sum(1 for part in parts if _plain_mapping(part).get("kind") == "data"),
        "file_part_count": sum(1 for part in parts if _plain_mapping(part).get("kind") == "file"),
        "status_update_count": sum(1 for event in protocol_events if str(event.get("type") or "") == "a2a_task_status"),
        "artifact_update_count": sum(1 for event in protocol_events if str(event.get("type") or "") == "a2a_task_artifact"),
        "terminal_task_count": sum(1 for state in states if state in terminal_states),
        "input_required_count": sum(1 for state in states if state in input_states),
        "error_count": sum(1 for event in protocol_events if event.get("error")) + sum(1 for state in states if state == "failed"),
        "task_ids": task_ids,
        "context_ids": context_ids,
        "agent_names": agent_names,
        "skill_names": skill_names,
        "roles": roles,
        "states": states,
        "event_types": event_types,
    }
    return {
        "kind": "a2a_protocol_trace",
        "framework": _a2a_protocol_framework(raw),
        "protocol": "a2a",
        **summary,
        "agent_cards": agent_cards,
        "messages": messages,
        "tasks": tasks,
        "artifacts": artifacts,
        "events": protocol_events,
        "summary": summary,
    }


def _a2a_protocol_events(raw: Any) -> List[SimulationEvent]:
    state = _a2a_protocol_state(raw)
    if not state:
        return []
    events: List[SimulationEvent] = []
    for index, card in enumerate(_plain_list(state.get("agent_cards")), start=1):
        card_dict = _plain_mapping(card)
        events.append(
            SimulationEvent(
                type="a2a_agent_card",
                name=str(card_dict.get("name") or f"agent_card_{index}"),
                payload={**card_dict, "sequence": index},
                metadata={"kind": "a2a_protocol_trace", "source": "framework_adapter_output"},
            )
        )
    for index, message in enumerate(_plain_list(state.get("messages")), start=1):
        message_dict = _plain_mapping(message)
        events.append(
            SimulationEvent(
                type="a2a_message",
                name=str(message_dict.get("message_id") or f"message_{index}"),
                payload={**message_dict, "sequence": index},
                metadata={"kind": "a2a_protocol_trace", "source": "framework_adapter_output"},
            )
        )
    for index, task in enumerate(_plain_list(state.get("tasks")), start=1):
        task_dict = _plain_mapping(task)
        events.append(
            SimulationEvent(
                type="a2a_task",
                name=str(task_dict.get("id") or f"task_{index}"),
                payload={**task_dict, "sequence": index},
                metadata={"kind": "a2a_protocol_trace", "source": "framework_adapter_output"},
            )
        )
    for index, artifact in enumerate(_plain_list(state.get("artifacts")), start=1):
        artifact_dict = _plain_mapping(artifact)
        events.append(
            SimulationEvent(
                type="a2a_artifact",
                name=str(artifact_dict.get("name") or artifact_dict.get("id") or f"artifact_{index}"),
                payload={**artifact_dict, "sequence": index},
                metadata={"kind": "a2a_protocol_trace", "source": "framework_adapter_output"},
            )
        )
    for index, protocol_event in enumerate(_plain_list(state.get("events")), start=1):
        event_dict = _plain_mapping(protocol_event)
        events.append(
            SimulationEvent(
                type=str(event_dict.get("type") or "a2a_protocol_event"),
                name=str(event_dict.get("name") or event_dict.get("method") or f"a2a_event_{index}"),
                payload={**event_dict, "sequence": index},
                metadata={"kind": "a2a_protocol_trace", "source": "framework_adapter_output"},
            )
        )
    events.append(
        SimulationEvent(
            type="a2a_protocol_trace",
            name="a2a_protocol_trace",
            payload=state,
            metadata={"kind": "a2a_protocol_trace", "source": "framework_adapter_output"},
        )
    )
    return events


def _has_a2a_protocol_shape(raw: Any) -> bool:
    if raw in (None, "", [], {}):
        return False
    if isinstance(raw, (list, tuple)):
        return any(_looks_like_a2a_record(item) for item in raw)
    raw_mapping = _object_mapping(raw)
    explicit_names = (
        "a2a_protocol_trace",
        "a2a_session",
        "a2a_trace",
        "a2a_events",
        "a2a_messages",
        "a2a_tasks",
        "a2a_artifacts",
        "a2a_agent_card",
        "a2a_agent_cards",
        "agent_card",
        "agentCard",
        "agent_cards",
        "remote_agents",
    )
    if raw_mapping is None:
        return any(
            hasattr(raw, name) and getattr(raw, name) not in (None, "", [], {})
            for name in explicit_names
        )
    if any(raw_mapping.get(name) not in (None, "", [], {}) for name in explicit_names):
        return True
    if _looks_like_a2a_record(raw_mapping):
        return True
    if not _a2a_has_protocol_marker(raw_mapping):
        return False
    protocol_fields = (
        "messages",
        "tasks",
        "task",
        "events",
        "records",
        "requests",
        "responses",
        "stream_events",
        "items",
        "artifacts",
    )
    return any(raw_mapping.get(name) not in (None, "", [], {}) for name in protocol_fields)


def _a2a_agent_cards(raw: Any) -> List[Dict[str, Any]]:
    values: List[Any] = []
    for name in (
        "a2a_agent_card",
        "a2a_agent_cards",
        "agent_card",
        "agentCard",
        "agent_cards",
        "remote_agents",
    ):
        values.extend(_a2a_values(_a2a_field(raw, name)))
    raw_mapping = _object_mapping(raw)
    if raw_mapping and _looks_like_a2a_agent_card(raw_mapping):
        values.append(raw_mapping)
    return _dedupe_a2a_items(
        _normalize_a2a_agent_card(value, index=index)
        for index, value in enumerate(values, start=1)
        if _looks_like_a2a_agent_card(value)
    )


def _a2a_messages(raw: Any) -> List[Dict[str, Any]]:
    values: List[Any] = []
    for name in ("a2a_messages", "messages", "history"):
        field_value = _a2a_field(raw, name)
        if name in {"messages", "history"} and not _a2a_has_protocol_marker(_object_mapping(raw) or {}):
            continue
        values.extend(_a2a_values(field_value))
    for record in _a2a_protocol_records(raw):
        record_dict = _plain_mapping(record)
        params = _plain_mapping(record_dict.get("params"))
        result = _plain_mapping(record_dict.get("result"))
        task = _a2a_task_payload(record_dict) or _a2a_task_payload(result)
        for candidate in (
            params.get("message"),
            result.get("message"),
            _plain_mapping(record_dict.get("status")).get("message"),
            _plain_mapping(result.get("status")).get("message"),
        ):
            if _looks_like_a2a_message(candidate):
                values.append(candidate)
        if task:
            values.extend(_plain_list(task.get("history")))
            status_message = _plain_mapping(_plain_mapping(task.get("status")).get("message"))
            if status_message:
                values.append(status_message)
    raw_mapping = _object_mapping(raw)
    if raw_mapping and _looks_like_a2a_message(raw_mapping):
        values.append(raw_mapping)
    return _dedupe_a2a_items(
        _normalize_a2a_message(value, index=index)
        for index, value in enumerate(values, start=1)
        if _looks_like_a2a_message(value)
    )


def _a2a_tasks(raw: Any) -> List[Dict[str, Any]]:
    values: List[Any] = []
    for name in ("a2a_tasks", "tasks", "task"):
        value = _a2a_field(raw, name)
        if name in {"tasks", "task"} and not _a2a_has_protocol_marker(_object_mapping(raw) or {}):
            continue
        values.extend(_a2a_values(value))
    for record in _a2a_protocol_records(raw):
        record_dict = _plain_mapping(record)
        result = _plain_mapping(record_dict.get("result"))
        candidates: List[Any] = [result, result.get("task")]
        if not record_dict.get("method") and not record_dict.get("type") and not record_dict.get("event"):
            candidates.append(record_dict)
        for candidate in candidates:
            task = _a2a_task_payload(candidate)
            if task:
                values.append(task)
    raw_mapping = _object_mapping(raw)
    if raw_mapping:
        task = _a2a_task_payload(raw_mapping)
        if task:
            values.append(task)
    return _dedupe_a2a_tasks(
        _normalize_a2a_task(value, index=index)
        for index, value in enumerate(values, start=1)
        if _a2a_task_payload(value)
    )


def _a2a_artifacts(raw: Any, *, tasks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    values: List[Any] = []
    for name in ("a2a_artifacts", "task_artifacts"):
        values.extend(_a2a_values(_a2a_field(raw, name)))
    raw_mapping = _object_mapping(raw)
    if raw_mapping and _a2a_has_protocol_marker(raw_mapping):
        values.extend(_a2a_values(raw_mapping.get("artifacts")))
    for task in tasks:
        values.extend(_a2a_values(_plain_mapping(task).get("artifacts")))
    for record in _a2a_protocol_records(raw):
        record_dict = _plain_mapping(record)
        result = _plain_mapping(record_dict.get("result"))
        for candidate in (
            record_dict.get("artifact"),
            result.get("artifact"),
            _plain_mapping(record_dict.get("params")).get("artifact"),
        ):
            if _looks_like_a2a_artifact(candidate):
                values.append(candidate)
    return _dedupe_a2a_items(
        _normalize_a2a_artifact(value, index=index)
        for index, value in enumerate(values, start=1)
        if _looks_like_a2a_artifact(value)
    )


def _a2a_protocol_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, (list, tuple)):
        return [
            _plain_mapping(item)
            for item in raw
            if _plain_mapping(item)
        ]
    raw_mapping = _object_mapping(raw)
    if not raw_mapping:
        return []
    records: List[Dict[str, Any]] = []
    for name in ("a2a_events", "events", "records", "requests", "responses", "stream_events", "items"):
        for value in _a2a_values(raw_mapping.get(name)):
            record = _plain_mapping(value)
            if record:
                records.append(record)
    if _looks_like_a2a_record(raw_mapping):
        records.append(raw_mapping)
    return _dedupe_a2a_items(records)


def _a2a_protocol_events_payload(raw: Any) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for index, record in enumerate(_a2a_protocol_records(raw), start=1):
        event = _normalize_a2a_protocol_event(record, index=index)
        if event:
            events.append(event)
    return _dedupe_a2a_items(events)


def _normalize_a2a_agent_card(value: Any, *, index: int) -> Dict[str, Any]:
    card = _plain_mapping(value)
    skills = [
        _normalize_a2a_skill(skill, index=skill_index)
        for skill_index, skill in enumerate(_a2a_values(card.get("skills")), start=1)
    ]
    input_modes = _unique_nonempty_strings(
        [
            *_plain_list(card.get("defaultInputModes") or card.get("default_input_modes")),
            *[
                mode
                for skill in skills
                for mode in _plain_list(skill.get("input_modes"))
            ],
        ]
    )
    output_modes = _unique_nonempty_strings(
        [
            *_plain_list(card.get("defaultOutputModes") or card.get("default_output_modes")),
            *[
                mode
                for skill in skills
                for mode in _plain_list(skill.get("output_modes"))
            ],
        ]
    )
    return {
        "id": str(card.get("id") or card.get("name") or f"a2a_agent_{index}"),
        "name": str(card.get("name") or f"a2a_agent_{index}"),
        "description": str(card.get("description") or ""),
        "url": str(card.get("url") or ""),
        "version": str(card.get("version") or ""),
        "protocol_version": str(card.get("protocolVersion") or card.get("protocol_version") or ""),
        "preferred_transport": str(card.get("preferredTransport") or card.get("preferred_transport") or ""),
        "capabilities": _plain_mapping(card.get("capabilities")),
        "input_modes": input_modes,
        "output_modes": output_modes,
        "skills": skills,
        "security": _plain_value(card.get("security") or {}),
        "metadata": _plain_mapping(card.get("metadata")),
    }


def _normalize_a2a_skill(value: Any, *, index: int) -> Dict[str, Any]:
    skill = _plain_mapping(value)
    skill_id = str(skill.get("id") or skill.get("name") or f"skill_{index}")
    return {
        "id": skill_id,
        "name": str(skill.get("name") or skill_id),
        "description": str(skill.get("description") or ""),
        "tags": _unique_nonempty_strings(skill.get("tags")),
        "examples": _unique_nonempty_strings(skill.get("examples")),
        "input_modes": _unique_nonempty_strings(skill.get("inputModes") or skill.get("input_modes")),
        "output_modes": _unique_nonempty_strings(skill.get("outputModes") or skill.get("output_modes")),
        "metadata": _plain_mapping(skill.get("metadata")),
    }


def _normalize_a2a_message(value: Any, *, index: int) -> Dict[str, Any]:
    message = _plain_mapping(value)
    parts = _a2a_parts(message.get("parts") or message.get("content"))
    return {
        "id": str(message.get("id") or message.get("messageId") or message.get("message_id") or f"message_{index}"),
        "message_id": str(message.get("messageId") or message.get("message_id") or message.get("id") or f"message_{index}"),
        "task_id": str(message.get("taskId") or message.get("task_id") or ""),
        "context_id": str(message.get("contextId") or message.get("context_id") or ""),
        "role": str(message.get("role") or ""),
        "parts": parts,
        "text": _a2a_parts_text(parts),
        "metadata": _plain_mapping(message.get("metadata")),
    }


def _normalize_a2a_task(value: Any, *, index: int) -> Dict[str, Any]:
    task = _a2a_task_payload(value)
    status = _plain_mapping(task.get("status"))
    artifacts = [
        _normalize_a2a_artifact(artifact, index=artifact_index)
        for artifact_index, artifact in enumerate(_a2a_values(task.get("artifacts")), start=1)
        if _looks_like_a2a_artifact(artifact)
    ]
    history = [
        _normalize_a2a_message(message, index=message_index)
        for message_index, message in enumerate(_a2a_values(task.get("history")), start=1)
        if _looks_like_a2a_message(message)
    ]
    return {
        "id": str(task.get("id") or task.get("taskId") or task.get("task_id") or f"task_{index}"),
        "context_id": str(task.get("contextId") or task.get("context_id") or ""),
        "state": _a2a_state_key(status.get("state") or task.get("state")),
        "status": status,
        "history": history,
        "artifacts": artifacts,
        "metadata": _plain_mapping(task.get("metadata")),
    }


def _normalize_a2a_artifact(value: Any, *, index: int) -> Dict[str, Any]:
    artifact = _plain_mapping(value)
    parts = _a2a_parts(artifact.get("parts") or artifact.get("content"))
    return {
        "id": str(artifact.get("artifactId") or artifact.get("artifact_id") or artifact.get("id") or f"artifact_{index}"),
        "name": str(artifact.get("name") or artifact.get("title") or f"artifact_{index}"),
        "description": str(artifact.get("description") or ""),
        "parts": parts,
        "text": _a2a_parts_text(parts),
        "metadata": _plain_mapping(artifact.get("metadata")),
    }


def _normalize_a2a_protocol_event(record: Mapping[str, Any], *, index: int) -> Dict[str, Any]:
    event = _plain_mapping(record)
    params = _plain_mapping(event.get("params"))
    result = _plain_mapping(event.get("result"))
    status = _plain_mapping(event.get("status") or result.get("status"))
    artifact = _plain_mapping(event.get("artifact") or result.get("artifact") or params.get("artifact"))
    method = str(event.get("method") or event.get("event") or event.get("type") or "")
    event_type = _a2a_event_type(event, method=method, status=status, artifact=artifact)
    task = _a2a_task_payload(event) or _a2a_task_payload(result)
    return {
        "id": str(event.get("id") or event.get("event_id") or event.get("eventId") or f"a2a_event_{index}"),
        "type": event_type,
        "name": method or event_type,
        "method": method,
        "task_id": str(
            event.get("taskId")
            or event.get("task_id")
            or params.get("taskId")
            or params.get("task_id")
            or result.get("taskId")
            or result.get("task_id")
            or task.get("id")
            or task.get("taskId")
            or ""
        ),
        "context_id": str(
            event.get("contextId")
            or event.get("context_id")
            or params.get("contextId")
            or params.get("context_id")
            or result.get("contextId")
            or result.get("context_id")
            or task.get("contextId")
            or task.get("context_id")
            or ""
        ),
        "state": _a2a_state_key(status.get("state") or task.get("state") or _plain_mapping(task.get("status")).get("state")),
        "final": bool(event.get("final", False) or result.get("final", False)),
        "error": _plain_value(event.get("error") or result.get("error")),
        "payload": _plain_value(event),
    }


def _a2a_event_type(
    event: Mapping[str, Any],
    *,
    method: str,
    status: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> str:
    key = _a2a_protocol_key(method or event.get("kind") or event.get("type"))
    if key in {"sendmessage", "messagesend", "message_send", "message_sendstream", "messagesendstream", "message_stream", "sendstreamingmessage"}:
        return "a2a_message_send"
    if key in {"gettask", "tasksget", "task_get"}:
        return "a2a_task_get"
    if key in {"canceltask", "taskscancel", "task_cancel"}:
        return "a2a_task_cancel"
    if artifact or "artifact" in key:
        return "a2a_task_artifact"
    if status or "status" in key:
        return "a2a_task_status"
    if _looks_like_a2a_message(event):
        return "a2a_message"
    if _a2a_task_payload(event):
        return "a2a_task"
    if event.get("error"):
        return "a2a_error"
    return "a2a_protocol_event"


def _a2a_parts(value: Any) -> List[Dict[str, Any]]:
    values = _a2a_values(value)
    if not values and value not in (None, "", [], {}):
        values = [value]
    parts: List[Dict[str, Any]] = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, str):
            parts.append({"kind": "text", "text": item})
            continue
        part = _plain_mapping(item)
        if not part:
            continue
        file_payload = _plain_mapping(part.get("file"))
        kind = _a2a_part_kind(part, file_payload=file_payload)
        normalized = {
            "id": str(part.get("id") or f"part_{index}"),
            "kind": kind,
            "text": str(part.get("text") or part.get("content") or "") if kind == "text" else "",
            "data": _plain_value(part.get("data")) if kind == "data" else None,
            "file": file_payload or _a2a_file_part_payload(part),
            "metadata": _plain_mapping(part.get("metadata")),
        }
        parts.append({key: value for key, value in normalized.items() if value not in (None, "", [], {})})
    return parts


def _a2a_part_kind(part: Mapping[str, Any], *, file_payload: Mapping[str, Any]) -> str:
    raw_kind = _a2a_protocol_key(part.get("kind") or part.get("type"))
    if "file" in raw_kind or file_payload or part.get("uri") or part.get("path"):
        return "file"
    if "data" in raw_kind or part.get("data") not in (None, "", [], {}):
        return "data"
    if "text" in raw_kind or part.get("text") not in (None, "", [], {}) or part.get("content") not in (None, "", [], {}):
        return "text"
    return raw_kind or "part"


def _a2a_file_part_payload(part: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {}
    for key in ("uri", "path", "name", "mimeType", "mime_type", "bytes"):
        value = part.get(key)
        if value not in (None, "", [], {}):
            payload[key] = _plain_value(value)
    return payload


def _a2a_parts_text(parts: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(
        str(_plain_mapping(part).get("text") or "")
        for part in parts
        if _plain_mapping(part).get("kind") == "text"
        and _plain_mapping(part).get("text")
    )


def _a2a_simulation_artifact_type(artifact: Mapping[str, Any]) -> str:
    parts = _plain_list(artifact.get("parts"))
    if any(_plain_mapping(part).get("kind") == "file" for part in parts):
        return "file"
    if any(_plain_mapping(part).get("kind") == "data" for part in parts):
        return "json"
    return "text"


def _a2a_task_payload(value: Any) -> Dict[str, Any]:
    task = _plain_mapping(value)
    if not task:
        return {}
    result_task = _plain_mapping(task.get("task"))
    if result_task:
        return result_task
    if _looks_like_a2a_task(task):
        return task
    return {}


def _looks_like_a2a_record(value: Any) -> bool:
    record = _plain_mapping(value)
    if not record:
        return False
    method = _a2a_protocol_key(record.get("method"))
    if method in {
        "sendmessage",
        "sendstreamingmessage",
        "gettask",
        "canceltask",
        "settaskpushnotificationconfig",
        "gettaskpushnotificationconfig",
        "message_send",
        "message_stream",
        "tasks_get",
        "tasks_cancel",
    }:
        return True
    if _looks_like_a2a_agent_card(record) or _looks_like_a2a_message(record) or _looks_like_a2a_task(record):
        return True
    result = _plain_mapping(record.get("result"))
    return bool(_looks_like_a2a_task(result) or _looks_like_a2a_message(result))


def _looks_like_a2a_agent_card(value: Any) -> bool:
    card = _plain_mapping(value)
    if not card:
        return False
    has_card_fields = bool(card.get("skills") or card.get("capabilities"))
    has_identity = bool(card.get("name") or card.get("url") or card.get("version"))
    return has_card_fields and has_identity


def _looks_like_a2a_message(value: Any) -> bool:
    message = _plain_mapping(value)
    if not message:
        return False
    has_parts = message.get("parts") not in (None, "", [], {})
    has_identity = bool(
        message.get("messageId")
        or message.get("message_id")
        or message.get("taskId")
        or message.get("task_id")
        or message.get("contextId")
        or message.get("context_id")
    )
    return has_parts and bool(message.get("role")) and has_identity


def _looks_like_a2a_task(value: Any) -> bool:
    task = _plain_mapping(value)
    if not task:
        return False
    has_id = bool(task.get("id") or task.get("taskId") or task.get("task_id"))
    return has_id and any(key in task for key in ("status", "artifacts", "history", "contextId", "context_id"))


def _looks_like_a2a_artifact(value: Any) -> bool:
    artifact = _plain_mapping(value)
    if not artifact:
        return False
    return bool(artifact.get("parts")) and bool(
        artifact.get("artifactId")
        or artifact.get("artifact_id")
        or artifact.get("id")
        or artifact.get("name")
    )


def _a2a_has_protocol_marker(value: Mapping[str, Any]) -> bool:
    for key in ("framework", "protocol", "type", "kind"):
        if _a2a_protocol_key(value.get(key)) in {"a2a", "agent2agent", "agenttoagent"}:
            return True
    metadata = _plain_mapping(value.get("metadata"))
    for key in ("framework", "protocol"):
        if _a2a_protocol_key(metadata.get(key)) in {"a2a", "agent2agent", "agenttoagent"}:
            return True
    return False


def _a2a_protocol_framework(raw: Any) -> str:
    value = (
        _a2a_field(raw, "framework")
        or _a2a_field(raw, "protocol")
        or _plain_mapping(_a2a_field(raw, "metadata")).get("framework")
    )
    return "a2a" if _a2a_protocol_key(value) in {"a2a", "agent2agent", "agenttoagent"} else str(value or "a2a")


def _a2a_field(raw: Any, name: str) -> Any:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        return raw_mapping.get(name)
    return getattr(raw, name, None)


def _a2a_values(value: Any) -> List[Any]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item) for item in value]
    return [_plain_value(value)]


def _a2a_state_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _a2a_protocol_key(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def _unique_nonempty_strings(value: Any) -> List[str]:
    return sorted({str(item) for item in _plain_list(value) if str(item)})


def _dedupe_a2a_items(values: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        item = _plain_mapping(value)
        if not item:
            continue
        signature = json.dumps(item, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def _dedupe_a2a_tasks(values: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    indexes: Dict[str, int] = {}
    for value in values:
        task = _plain_mapping(value)
        if not task:
            continue
        task_id = str(task.get("id") or "")
        if not task_id or task_id not in indexes:
            indexes[task_id] = len(deduped)
            deduped.append(task)
            continue
        existing = deduped[indexes[task_id]]
        if _a2a_task_prefer(task, existing):
            deduped[indexes[task_id]] = task
    return deduped


def _a2a_task_prefer(candidate: Mapping[str, Any], existing: Mapping[str, Any]) -> bool:
    terminal_states = {"completed", "failed", "canceled", "cancelled", "rejected"}
    candidate_state = str(candidate.get("state") or "")
    existing_state = str(existing.get("state") or "")
    if candidate_state in terminal_states and existing_state not in terminal_states:
        return True
    candidate_evidence = len(_plain_list(candidate.get("artifacts"))) + len(_plain_list(candidate.get("history")))
    existing_evidence = len(_plain_list(existing.get("artifacts"))) + len(_plain_list(existing.get("history")))
    return candidate_evidence > existing_evidence


def _orchestration_trace_state(raw: Any) -> Dict[str, Any]:
    if not _has_orchestration_trace_shape(raw):
        return {}
    framework = _orchestration_trace_framework(raw)
    metadata = _orchestration_trace_metadata(raw)
    runtime_state = _orchestration_trace_runtime_state(raw)
    nodes = _orchestration_trace_nodes(raw)
    edges = _orchestration_trace_edges(raw)
    steps = _orchestration_trace_steps(raw)
    records = _orchestration_trace_records(raw)
    export_count = 0
    for trace_export in _orchestration_trace_exports(raw):
        export_trace = normalize_orchestration_trace_export(
            trace_export,
            framework=framework,
            state=runtime_state,
            metadata=metadata,
        )
        export_count += 1
        nodes.extend(_plain_list(export_trace.get("nodes")))
        edges.extend(_plain_list(export_trace.get("edges")))
        records.extend(_plain_list(export_trace.get("steps")))
    if not any([nodes, edges, steps, records]):
        return {}
    trace = normalize_orchestration_trace_events(
        framework,
        records,
        nodes=[_plain_mapping(node) for node in nodes if _plain_mapping(node)],
        edges=[_plain_mapping(edge) for edge in edges if _plain_mapping(edge)],
        steps=[_plain_mapping(step) for step in steps if _plain_mapping(step)],
        state=runtime_state,
        metadata=metadata,
    )
    if export_count:
        trace.setdefault("metadata", {}).setdefault("trace_export", {})[
            "source"
        ] = "framework_adapter_output"
    return trace


def _orchestration_trace_events(raw: Any) -> List[SimulationEvent]:
    state = _orchestration_trace_state(raw)
    if not state:
        return []
    framework = str(state.get("framework") or "generic")
    events: List[SimulationEvent] = []
    for index, step in enumerate(_plain_list(state.get("steps")), start=1):
        step_dict = _plain_mapping(step)
        events.append(
            SimulationEvent(
                type="orchestration_step",
                name=str(
                    step_dict.get("name")
                    or step_dict.get("node")
                    or f"orchestration_step_{index}"
                ),
                payload={**step_dict, "sequence": index},
                metadata={
                    "kind": "orchestration_trace",
                    "framework": framework,
                    "source": "framework_adapter_output",
                    "signals": _plain_list(step_dict.get("signals")),
                },
            )
        )
    events.append(
        SimulationEvent(
            type="orchestration_trace",
            name="orchestration_trace",
            payload=state,
            metadata={
                "kind": "orchestration_trace",
                "framework": framework,
                "source": "framework_adapter_output",
            },
        )
    )
    return events


def _orchestration_trace_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    state = _orchestration_trace_state(raw)
    if not state:
        return []
    calls: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, step in enumerate(_plain_list(state.get("steps")), start=1):
        step_dict = _plain_mapping(step)
        if not _orchestration_trace_step_has_tool_signal(step_dict):
            continue
        name = _orchestration_trace_step_tool_name(step_dict)
        if not name:
            continue
        call_id = _orchestration_trace_step_call_id(step_dict, index=index)
        signature = f"{call_id}:{name}"
        if signature in seen:
            continue
        seen.add(signature)
        arguments = _orchestration_trace_step_arguments(step_dict)
        calls.append(
            {
                "id": call_id,
                "type": "orchestration_trace_tool_call",
                "name": name,
                "arguments": arguments,
                "function": {"name": name, "arguments": arguments},
            }
        )
    return calls


def _orchestration_trace_tool_responses(raw: Any) -> List[Dict[str, Any]]:
    state = _orchestration_trace_state(raw)
    if not state:
        return []
    responses: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, step in enumerate(_plain_list(state.get("steps")), start=1):
        step_dict = _plain_mapping(step)
        if not _orchestration_trace_step_has_tool_signal(step_dict):
            continue
        output = _orchestration_trace_step_output(step_dict)
        error = step_dict.get("error")
        if output in (None, "", [], {}) and not error:
            continue
        name = _orchestration_trace_step_tool_name(step_dict)
        if not name:
            continue
        call_id = _orchestration_trace_step_call_id(step_dict, index=index)
        signature = f"{call_id}:{name}:{bool(error)}"
        if signature in seen:
            continue
        seen.add(signature)
        responses.append(
            {
                "id": f"{call_id}_response",
                "tool_call_id": call_id,
                "name": name,
                "content": _plain_value(error if error else output),
                "success": not bool(error),
                "result": _plain_value(output),
                "error": _plain_value(error),
            }
        )
    return responses


def _has_orchestration_trace_shape(raw: Any) -> bool:
    if raw in (None, "", [], {}):
        return False
    if isinstance(raw, (list, tuple)):
        return any(_looks_like_orchestration_trace_record(item) for item in raw)
    raw_mapping = _object_mapping(raw)
    explicit_names = (
        "orchestration_trace",
        "orchestration_trace_export",
        "orchestration_export",
        "orchestration_records",
        "orchestration_events",
        "orchestration_steps",
        "orchestration_nodes",
        "orchestration_edges",
        "orchestration_state",
        "agent_orchestration_trace",
        "agent_graph_trace",
        "agent_graph_trace_export",
        "coordination_trace",
        "coordination_events",
        "multi_agent_orchestration",
        "handoff_trace",
        "delegation_trace",
        "supervisor_trace",
    )
    if raw_mapping is None:
        return any(
            hasattr(raw, name) and getattr(raw, name) not in (None, "", [], {})
            for name in explicit_names
        )
    if any(raw_mapping.get(name) not in (None, "", [], {}) for name in explicit_names):
        return True
    if _looks_like_orchestration_trace_record(raw_mapping):
        return True
    if not _orchestration_trace_has_marker(raw_mapping):
        return False
    if raw_mapping.get("trace_export") not in (None, "", [], {}):
        return True
    return any(
        raw_mapping.get(name) not in (None, "", [], {})
        for name in ("nodes", "edges", "steps", "records", "events", "items", "results")
    )


def _orchestration_trace_exports(raw: Any) -> List[Any]:
    exports: List[Any] = []
    for name in (
        "orchestration_trace_export",
        "orchestration_export",
        "agent_graph_trace_export",
        "coordination_trace_export",
    ):
        value = _plain_value(_orchestration_trace_field(raw, name))
        if value not in (None, "", [], {}):
            exports.append(value)
    explicit = _orchestration_trace_explicit_payload(raw)
    for name in ("trace_export", "export"):
        value = _plain_value(explicit.get(name))
        if value not in (None, "", [], {}):
            exports.append(value)
    raw_mapping = _object_mapping(raw)
    if (
        raw_mapping
        and _orchestration_trace_has_marker(raw_mapping)
        and raw_mapping.get("trace_export") not in (None, "", [], {})
    ):
        exports.append(raw_mapping["trace_export"])
    return _dedupe_framework_trace_values(exports)


def _orchestration_trace_records(raw: Any) -> List[Any]:
    if isinstance(raw, (list, tuple)):
        return [_plain_value(item) for item in raw if _plain_value(item) not in (None, "", [], {})]
    records: List[Any] = []
    explicit = _orchestration_trace_explicit_payload(raw)
    for key in ("records", "events", "items", "results", "spans"):
        records.extend(_plain_list(explicit.get(key)))
    for name in (
        "orchestration_records",
        "orchestration_events",
        "orchestration_spans",
        "coordination_events",
        "agent_events",
        "agent_trace_events",
        "handoff_events",
        "delegation_events",
        "supervisor_events",
    ):
        records.extend(_plain_list(_orchestration_trace_field(raw, name)))
    raw_mapping = _object_mapping(raw)
    if raw_mapping and _orchestration_trace_has_marker(raw_mapping):
        for key in ("records", "events", "items", "results", "spans"):
            records.extend(_plain_list(raw_mapping.get(key)))
    if raw_mapping and _looks_like_orchestration_trace_record(raw_mapping):
        records.append(raw_mapping)
    return [
        _plain_value(record)
        for record in records
        if _plain_value(record) not in (None, "", [], {})
    ]


def _orchestration_trace_nodes(raw: Any) -> List[Any]:
    explicit = _orchestration_trace_explicit_payload(raw)
    values: List[Any] = []
    for name in ("orchestration_nodes", "agent_nodes", "agent_graph_nodes"):
        values.extend(_plain_list(_orchestration_trace_field(raw, name)))
    for key in ("nodes", "agents"):
        values.extend(_plain_list(explicit.get(key)))
    return [
        _plain_value(value)
        for value in values
        if _plain_mapping(value) or str(value)
    ]


def _orchestration_trace_edges(raw: Any) -> List[Any]:
    explicit = _orchestration_trace_explicit_payload(raw)
    values: List[Any] = []
    for name in ("orchestration_edges", "agent_edges", "agent_graph_edges"):
        values.extend(_plain_list(_orchestration_trace_field(raw, name)))
    for key in ("edges", "routes", "handoffs", "delegations"):
        values.extend(_plain_list(explicit.get(key)))
    return [_plain_value(value) for value in values if _plain_mapping(value)]


def _orchestration_trace_steps(raw: Any) -> List[Any]:
    explicit = _orchestration_trace_explicit_payload(raw)
    values: List[Any] = []
    for name in (
        "orchestration_steps",
        "agent_steps",
        "task_steps",
        "coordination_steps",
        "handoff_steps",
        "delegation_steps",
        "supervisor_steps",
    ):
        values.extend(_plain_list(_orchestration_trace_field(raw, name)))
    for key in ("steps", "orchestration_steps", "agent_steps"):
        values.extend(_plain_list(explicit.get(key)))
    return [
        _orchestration_trace_clean_step(value)
        for value in values
        if _plain_mapping(value)
    ]


def _orchestration_trace_explicit_payload(raw: Any) -> Dict[str, Any]:
    for name in (
        "orchestration_trace",
        "agent_orchestration_trace",
        "agent_graph_trace",
        "coordination_trace",
        "multi_agent_orchestration",
        "handoff_trace",
        "delegation_trace",
        "supervisor_trace",
    ):
        value = _plain_mapping(_orchestration_trace_field(raw, name))
        if value:
            return value
    return {}


def _orchestration_trace_framework(raw: Any) -> str:
    explicit = _orchestration_trace_explicit_payload(raw)
    metadata = _plain_mapping(_orchestration_trace_field(raw, "metadata"))
    value = (
        _orchestration_trace_field(raw, "framework")
        or _orchestration_trace_field(raw, "orchestration_framework")
        or _orchestration_trace_field(raw, "trace_framework")
        or explicit.get("framework")
        or explicit.get("trace_provider")
        or metadata.get("framework")
        or metadata.get("trace_provider")
        or "generic"
    )
    return str(value or "generic")


def _orchestration_trace_metadata(raw: Any) -> Dict[str, Any]:
    explicit = _orchestration_trace_explicit_payload(raw)
    metadata = {
        **_plain_mapping(explicit.get("metadata")),
        **_plain_mapping(_orchestration_trace_field(raw, "trace_metadata")),
        **_plain_mapping(_orchestration_trace_field(raw, "orchestration_metadata")),
        **_plain_mapping(_orchestration_trace_field(raw, "metadata")),
    }
    if _orchestration_trace_exports(raw):
        metadata.setdefault("trace_export", {})["source"] = "framework_adapter_output"
    return metadata


def _orchestration_trace_runtime_state(raw: Any) -> Dict[str, Any]:
    explicit = _orchestration_trace_explicit_payload(raw)
    return (
        _plain_mapping(_orchestration_trace_field(raw, "orchestration_state"))
        or _plain_mapping(_orchestration_trace_field(raw, "agent_state"))
        or _plain_mapping(_orchestration_trace_field(raw, "coordination_state"))
        or _plain_mapping(_orchestration_trace_field(raw, "final_state"))
        or _plain_mapping(explicit.get("state"))
        or _plain_mapping(explicit.get("final_state"))
    )


def _orchestration_trace_field(raw: Any, name: str) -> Any:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        return raw_mapping.get(name)
    return getattr(raw, name, None)


def _orchestration_trace_has_marker(value: Mapping[str, Any]) -> bool:
    markers = {
        "orchestration",
        "orchestration_trace",
        "agent_orchestration",
        "agent_graph",
        "multi_agent",
        "supervisor",
        "coordination",
        "handoff",
        "delegation",
        "delegate",
    }
    for key in ("kind", "type", "protocol", "trace_provider", "provider", "telemetry"):
        marker = _orchestration_trace_key(value.get(key))
        if marker in markers:
            return True
    metadata = _plain_mapping(value.get("metadata"))
    return bool(metadata) and _orchestration_trace_has_marker(metadata)


def _looks_like_orchestration_trace_record(value: Any) -> bool:
    record = _plain_mapping(value)
    if not record:
        return False
    if any(
        record.get(key) not in (None, "", [], {})
        for key in (
            "route_from",
            "route_to",
            "handoff_to",
            "handoff_from",
            "delegate_to",
            "delegate_from",
        )
    ):
        return True
    signals = {_orchestration_trace_key(item) for item in _plain_list(record.get("signals"))}
    if signals & {"spawn", "delegate", "handoff", "communicate", "aggregate", "stop"}:
        return True
    text = " ".join(
        str(record.get(key) or "")
        for key in ("kind", "type", "event", "method", "name", "operation")
    )
    if any(
        token in text.lower()
        for token in (
            "orchestration",
            "supervisor",
            "multi_agent",
            "agent_graph",
            "handoff",
            "delegate",
            "spawn",
            "aggregate",
            "consensus",
        )
    ):
        return True
    return _orchestration_trace_has_marker(record) and any(
        record.get(key) not in (None, "", [], {})
        for key in ("node", "agent", "name", "status", "input", "output", "attributes")
    )


def _orchestration_trace_step_has_tool_signal(step: Mapping[str, Any]) -> bool:
    signals = {_orchestration_trace_key(signal) for signal in _plain_list(step.get("signals"))}
    if "tool" in signals or _orchestration_trace_key(step.get("type")) == "tool":
        return True
    attributes = _plain_mapping(step.get("attributes"))
    return any(
        source.get(key) not in (None, "", [], {})
        for source in (step, attributes)
        for key in (
            "tool",
            "gen_ai.tool.name",
            "mcp.tool.name",
            "tool.name",
        )
    )


def _orchestration_trace_clean_step(value: Any) -> Dict[str, Any]:
    step = _plain_mapping(value)
    for key in (
        "error",
        "recovered",
        "recoverable",
        "route_from",
        "route_to",
        "tool_name",
        "tool_call_id",
    ):
        if step.get(key) in (None, "", [], {}, False):
            step.pop(key, None)
    step.pop("recoverable", None)
    return step


def _orchestration_trace_step_tool_name(step: Mapping[str, Any]) -> str:
    attributes = _plain_mapping(step.get("attributes"))
    for source in (step, attributes):
        for key in (
            "tool_name",
            "tool",
            "gen_ai.tool.name",
            "mcp.tool.name",
            "tool.name",
        ):
            value = source.get(key)
            if value not in (None, "", [], {}):
                parsed = _orchestration_trace_tool_name_from_text(str(value))
                return parsed or str(value)
    return _orchestration_trace_tool_name_from_text(
        str(step.get("name") or step.get("type") or "")
    )


def _orchestration_trace_step_call_id(step: Mapping[str, Any], *, index: int) -> str:
    attributes = _plain_mapping(step.get("attributes"))
    return str(
        step.get("tool_call_id")
        or step.get("call_id")
        or attributes.get("tool_call_id")
        or attributes.get("mcp.tool.call_id")
        or step.get("id")
        or f"orchestration_tool_{index}"
    )


def _orchestration_trace_step_arguments(step: Mapping[str, Any]) -> Any:
    attributes = _plain_mapping(step.get("attributes"))
    return _plain_value(
        step.get("arguments")
        if "arguments" in step
        else step.get(
            "input",
            attributes.get(
                "arguments",
                attributes.get("gen_ai.tool.arguments", attributes.get("mcp.tool.arguments", {})),
            ),
        )
    )


def _orchestration_trace_step_output(step: Mapping[str, Any]) -> Any:
    attributes = _plain_mapping(step.get("attributes"))
    return _plain_value(
        step.get("result")
        if "result" in step
        else step.get(
            "output",
            attributes.get(
                "result",
                attributes.get("gen_ai.tool.result", attributes.get("mcp.tool.result")),
            ),
        )
    )


def _orchestration_trace_tool_name_from_text(value: str) -> str:
    lowered = value.lower()
    for prefix in ("tool result ", "tool error ", "tool call ", "function call "):
        if lowered.startswith(prefix):
            return value[len(prefix):].strip(" :")
    return ""


def _orchestration_trace_key(value: Any) -> str:
    aliases = {
        "delegation": "delegate",
        "delegated": "delegate",
        "transfer": "handoff",
        "message": "communicate",
        "communication": "communicate",
        "consensus": "aggregate",
        "vote": "aggregate",
        "finish": "stop",
        "terminate": "stop",
        "function": "tool",
        "function_call": "tool",
        "tool_call": "tool",
        "duration": "latency",
        "duration_ms": "latency",
        "tokens": "cost",
        "usage": "cost",
        "recover": "recovered",
    }
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")
    return aliases.get(normalized, normalized)


def _workflow_trace_state(raw: Any) -> Dict[str, Any]:
    if not _has_workflow_trace_shape(raw):
        return {}
    trace = _workflow_trace_payload(raw)
    nodes = _plain_list(trace.get("nodes"))
    edges = _plain_list(trace.get("edges"))
    steps = _plain_list(trace.get("steps"))
    checkpoints = _plain_list(trace.get("checkpoints"))
    routes = _plain_list(trace.get("route_decisions"))
    interrupts = _plain_list(trace.get("interrupts"))
    replay = _plain_list(trace.get("replay"))
    writes = _plain_list(trace.get("writes"))
    final_state = _plain_mapping(trace.get("final_state"))
    tool_calls = _workflow_trace_tool_calls(raw)
    step_statuses = sorted(
        {
            str(_plain_mapping(step).get("status") or "")
            for step in steps
            if _plain_mapping(step).get("status")
        }
    )
    return {
        "kind": "framework_workflow_trace",
        "workflow_id": str(trace.get("workflow_id") or ""),
        "thread_id": str(trace.get("thread_id") or ""),
        "run_id": str(trace.get("run_id") or ""),
        "framework": str(trace.get("framework") or ""),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "step_count": len(steps),
        "checkpoint_count": len(checkpoints),
        "route_decision_count": len(routes),
        "interrupt_count": len(interrupts),
        "replay_count": len(replay),
        "write_count": len(writes),
        "tool_call_count": len(tool_calls),
        "tool_names": sorted(
            {str(call.get("name") or "") for call in tool_calls if call.get("name")}
        ),
        "step_statuses": step_statuses,
        "final_state_keys": sorted(str(key) for key in final_state),
        "has_replay": bool(replay),
        "has_interrupts": bool(interrupts),
        "has_routes": bool(routes),
        "nodes": nodes,
        "edges": edges,
        "steps": steps,
        "checkpoints": checkpoints,
        "route_decisions": routes,
        "interrupts": interrupts,
        "replay": replay,
        "writes": writes,
        "topology": _workflow_trace_topology(nodes, edges),
        "final_state": final_state,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "step_count": len(steps),
            "checkpoint_count": len(checkpoints),
            "route_decision_count": len(routes),
            "interrupt_count": len(interrupts),
            "replay_count": len(replay),
            "tool_call_count": len(tool_calls),
        },
    }


def _workflow_trace_payload(raw: Any) -> Dict[str, Any]:
    explicit_trace = _workflow_explicit_trace(raw)
    nodes = _workflow_trace_nodes(raw, explicit_trace=explicit_trace)
    edges = _workflow_trace_edges(raw, explicit_trace=explicit_trace)
    steps = _workflow_trace_steps(raw, explicit_trace=explicit_trace)
    checkpoints = _workflow_trace_checkpoints(raw, explicit_trace=explicit_trace)
    routes = _workflow_trace_routes(raw, explicit_trace=explicit_trace)
    interrupts = _workflow_trace_interrupts(raw, explicit_trace=explicit_trace)
    replay = _workflow_trace_replay(raw, explicit_trace=explicit_trace)
    writes = _workflow_trace_writes(raw, explicit_trace=explicit_trace)
    final_state = (
        _plain_mapping(_workflow_trace_field(raw, "final_state"))
        or _plain_mapping(_workflow_trace_field(raw, "workflow_state"))
        or _plain_mapping(_workflow_trace_field(raw, "flow_state"))
        or _plain_mapping(explicit_trace.get("final_state"))
        or _plain_mapping(explicit_trace.get("state"))
    )
    return {
        "kind": "workflow_trace",
        "framework": str(
            _workflow_trace_field(raw, "framework")
            or explicit_trace.get("framework")
            or ""
        ),
        "workflow_id": str(
            _workflow_trace_field(raw, "workflow_id")
            or _workflow_trace_field(raw, "flow_id")
            or explicit_trace.get("workflow_id")
            or explicit_trace.get("flow_id")
            or ""
        ),
        "thread_id": str(
            _workflow_trace_field(raw, "thread_id")
            or explicit_trace.get("thread_id")
            or ""
        ),
        "run_id": str(
            _workflow_trace_field(raw, "run_id")
            or explicit_trace.get("run_id")
            or ""
        ),
        "nodes": nodes,
        "edges": edges,
        "steps": steps,
        "events": _workflow_trace_named_events(raw, explicit_trace=explicit_trace),
        "checkpoints": checkpoints,
        "route_decisions": routes,
        "interrupts": interrupts,
        "replay": replay,
        "writes": writes,
        "state_snapshots": _workflow_trace_state_snapshots(
            raw,
            explicit_trace=explicit_trace,
        ),
        "final_state": final_state,
        "topology": _workflow_trace_topology(nodes, edges),
        "trace_import": {
            "source": "framework_adapter_output",
            "provider": str(
                _workflow_trace_field(raw, "trace_provider")
                or explicit_trace.get("trace_provider")
                or "framework_workflow_trace"
            ),
        },
    }


def _workflow_trace_events(raw: Any) -> List[SimulationEvent]:
    if not _has_workflow_trace_shape(raw):
        return []
    trace = _workflow_trace_payload(raw)
    events: List[SimulationEvent] = []
    for index, step in enumerate(_plain_list(trace.get("steps")), start=1):
        step_dict = _plain_mapping(step)
        events.append(
            SimulationEvent(
                type="workflow_step",
                name=str(step_dict.get("name") or step_dict.get("node") or f"step_{index}"),
                payload={**step_dict, "sequence": index},
                metadata={"kind": "workflow_trace", "source": "framework_adapter_output"},
            )
        )
    for index, route in enumerate(_plain_list(trace.get("route_decisions")), start=1):
        route_dict = _plain_mapping(route)
        events.append(
            SimulationEvent(
                type="workflow_route",
                name=str(route_dict.get("name") or route_dict.get("source") or f"route_{index}"),
                payload={**route_dict, "sequence": index},
                metadata={"kind": "workflow_trace", "source": "framework_adapter_output"},
            )
        )
    for index, checkpoint in enumerate(_plain_list(trace.get("checkpoints")), start=1):
        checkpoint_dict = _plain_mapping(checkpoint)
        events.append(
            SimulationEvent(
                type="workflow_checkpoint",
                name=str(
                    checkpoint_dict.get("id")
                    or checkpoint_dict.get("checkpoint_id")
                    or f"checkpoint_{index}"
                ),
                payload={**checkpoint_dict, "sequence": index},
                metadata={"kind": "workflow_trace", "source": "framework_adapter_output"},
            )
        )
    for index, interrupt in enumerate(_plain_list(trace.get("interrupts")), start=1):
        interrupt_dict = _plain_mapping(interrupt)
        events.append(
            SimulationEvent(
                type="workflow_interrupt",
                name=str(interrupt_dict.get("node") or interrupt_dict.get("id") or f"interrupt_{index}"),
                payload={**interrupt_dict, "sequence": index},
                metadata={"kind": "workflow_trace", "source": "framework_adapter_output"},
            )
        )
    for index, replay in enumerate(_plain_list(trace.get("replay")), start=1):
        replay_dict = _plain_mapping(replay)
        events.append(
            SimulationEvent(
                type="workflow_replay",
                name=str(replay_dict.get("id") or f"replay_{index}"),
                payload={**replay_dict, "sequence": index},
                metadata={"kind": "workflow_trace", "source": "framework_adapter_output"},
            )
        )
    events.append(
        SimulationEvent(
            type="workflow_trace",
            name="framework_workflow_trace",
            payload=trace,
            metadata={"kind": "workflow_trace", "source": "framework_adapter_output"},
        )
    )
    return events


def _workflow_trace_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    if not _has_workflow_trace_shape(raw):
        return []
    calls: List[Dict[str, Any]] = []
    for step_index, step in enumerate(_workflow_trace_steps(raw), start=1):
        step_dict = _plain_mapping(step)
        step_call_values = [
            *_plain_list(step_dict.get("tool_calls")),
            *_plain_list(step_dict.get("tools")),
        ]
        if step_dict.get("tool_name") not in (None, "", [], {}):
            step_call_values.append(
                {
                    "id": step_dict.get("tool_call_id") or f"workflow_tool_{step_index}",
                    "name": step_dict.get("tool_name"),
                    "arguments": step_dict.get("tool_arguments") or {},
                }
            )
        for call_index, call in enumerate(step_call_values, start=1):
            call_dict = _plain_mapping(call)
            name = str(
                call_dict.get("name")
                or call_dict.get("tool")
                or _plain_mapping(call_dict.get("function")).get("name")
                or ""
            )
            if not name:
                continue
            arguments = (
                _plain_mapping(call_dict.get("arguments"))
                or _plain_mapping(call_dict.get("args"))
                or _plain_mapping(_plain_mapping(call_dict.get("function")).get("arguments"))
            )
            calls.append(
                {
                    "id": str(
                        call_dict.get("id")
                        or call_dict.get("call_id")
                        or f"{name}_{step_index}_{call_index}"
                    ),
                    "name": name,
                    "arguments": arguments,
                    "function": {"name": name, "arguments": arguments},
                }
            )
    return calls


def _has_workflow_trace_shape(raw: Any) -> bool:
    raw_mapping = _object_mapping(raw)
    names = (
        "workflow_trace",
        "graph_trace",
        "workflow_steps",
        "workflow_events",
        "workflow_nodes",
        "workflow_edges",
        "workflow_checkpoints",
        "workflow_replay",
        "graph_nodes",
        "graph_edges",
        "graph_steps",
        "graph_events",
        "graph_checkpoints",
        "state_history",
        "route_decisions",
        "router_decisions",
        "interrupts",
        "flow_state",
        "flow_id",
    )
    if raw_mapping is not None:
        return any(raw_mapping.get(name) not in (None, "", [], {}) for name in names)
    return any(
        hasattr(raw, name) and getattr(raw, name) not in (None, "", [], {})
        for name in names
    )


def _workflow_explicit_trace(raw: Any) -> Dict[str, Any]:
    for name in ("workflow_trace", "graph_trace"):
        trace = _plain_mapping(_workflow_trace_field(raw, name))
        if trace:
            return trace
    return {}


def _workflow_trace_nodes(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace) or _workflow_explicit_trace(raw)
    values = [
        *_plain_list(_workflow_trace_field(raw, "workflow_nodes")),
        *_plain_list(_workflow_trace_field(raw, "graph_nodes")),
        *_plain_list(trace.get("nodes")),
        *_plain_list(trace.get("graph_nodes")),
    ]
    return [
        _normalize_workflow_node(item, index=index)
        for index, item in enumerate(values, start=1)
        if _plain_mapping(item) or str(item)
    ]


def _workflow_trace_edges(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace) or _workflow_explicit_trace(raw)
    values = [
        *_plain_list(_workflow_trace_field(raw, "workflow_edges")),
        *_plain_list(_workflow_trace_field(raw, "graph_edges")),
        *_plain_list(trace.get("edges")),
        *_plain_list(trace.get("graph_edges")),
    ]
    return [
        _normalize_workflow_edge(item, index=index)
        for index, item in enumerate(values, start=1)
        if _plain_mapping(item) or str(item)
    ]


def _workflow_trace_steps(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace) or _workflow_explicit_trace(raw)
    values = [
        *_plain_list(_workflow_trace_field(raw, "workflow_steps")),
        *_plain_list(_workflow_trace_field(raw, "graph_steps")),
        *_plain_list(trace.get("steps")),
        *_plain_list(trace.get("workflow_steps")),
    ]
    return [
        _normalize_workflow_step(item, index=index)
        for index, item in enumerate(values, start=1)
        if _plain_mapping(item)
    ]


def _workflow_trace_checkpoints(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace) or _workflow_explicit_trace(raw)
    values = [
        *_plain_list(_workflow_trace_field(raw, "workflow_checkpoints")),
        *_plain_list(_workflow_trace_field(raw, "graph_checkpoints")),
        *_plain_list(trace.get("checkpoints")),
        *_plain_list(trace.get("workflow_checkpoints")),
    ]
    return [
        _normalize_workflow_checkpoint(item, index=index)
        for index, item in enumerate(values, start=1)
        if _plain_mapping(item)
    ]


def _workflow_trace_routes(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace) or _workflow_explicit_trace(raw)
    values = [
        *_plain_list(_workflow_trace_field(raw, "route_decisions")),
        *_plain_list(_workflow_trace_field(raw, "router_decisions")),
        *_plain_list(trace.get("route_decisions")),
        *_plain_list(trace.get("router_decisions")),
    ]
    return [
        _normalize_workflow_route(item, index=index)
        for index, item in enumerate(values, start=1)
        if _plain_mapping(item)
    ]


def _workflow_trace_interrupts(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace) or _workflow_explicit_trace(raw)
    values = [
        *_plain_list(_workflow_trace_field(raw, "interrupts")),
        *_plain_list(_workflow_trace_field(raw, "workflow_interrupts")),
        *_plain_list(trace.get("interrupts")),
    ]
    return [
        _normalize_workflow_interrupt(item, index=index)
        for index, item in enumerate(values, start=1)
        if _plain_mapping(item)
    ]


def _workflow_trace_replay(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace) or _workflow_explicit_trace(raw)
    values = [
        *_plain_list(_workflow_trace_field(raw, "workflow_replay")),
        *_plain_list(_workflow_trace_field(raw, "replay")),
        *_plain_list(trace.get("replay")),
        *_plain_list(trace.get("workflow_replay")),
    ]
    return [
        _normalize_workflow_replay(item, index=index)
        for index, item in enumerate(values, start=1)
        if _plain_mapping(item)
    ]


def _workflow_trace_writes(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace) or _workflow_explicit_trace(raw)
    values = [
        *_plain_list(_workflow_trace_field(raw, "workflow_writes")),
        *_plain_list(_workflow_trace_field(raw, "pending_writes")),
        *_plain_list(trace.get("writes")),
        *_plain_list(trace.get("pending_writes")),
    ]
    return [
        _plain_mapping(item)
        for item in values
        if _plain_mapping(item)
    ]


def _workflow_trace_named_events(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace) or _workflow_explicit_trace(raw)
    values = [
        *_plain_list(_workflow_trace_field(raw, "workflow_events")),
        *_plain_list(_workflow_trace_field(raw, "graph_events")),
        *_plain_list(trace.get("events")),
        *_plain_list(trace.get("workflow_events")),
    ]
    return [
        _plain_mapping(item)
        for item in values
        if _plain_mapping(item)
    ]


def _workflow_trace_state_snapshots(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace) or _workflow_explicit_trace(raw)
    values = [
        *_plain_list(_workflow_trace_field(raw, "state_history")),
        *_plain_list(_workflow_trace_field(raw, "state_snapshots")),
        *_plain_list(trace.get("state_history")),
        *_plain_list(trace.get("state_snapshots")),
    ]
    return [
        _plain_mapping(item)
        for item in values
        if _plain_mapping(item)
    ]


def _normalize_workflow_node(item: Any, *, index: int) -> Dict[str, Any]:
    if not isinstance(item, Mapping) and not _object_mapping(item):
        node_id = str(item)
        return {
            "id": node_id or f"node_{index}",
            "name": node_id or f"node_{index}",
            "type": "node",
            "metadata": {},
        }
    node = _plain_mapping(item)
    node_id = str(
        node.get("id")
        or node.get("node_id")
        or node.get("name")
        or f"node_{index}"
    )
    return {
        "id": node_id,
        "name": str(node.get("name") or node_id),
        "type": str(node.get("type") or node.get("kind") or "node"),
        "role": str(node.get("role") or ""),
        "input_keys": [str(key) for key in _plain_list(node.get("input_keys"))],
        "output_keys": [str(key) for key in _plain_list(node.get("output_keys"))],
        "metadata": _plain_mapping(node.get("metadata")),
    }


def _normalize_workflow_edge(item: Any, *, index: int) -> Dict[str, Any]:
    if not isinstance(item, Mapping) and not _object_mapping(item):
        parts = [part.strip() for part in str(item).replace("->", ":").split(":")]
        source = parts[0] if parts else ""
        target = parts[1] if len(parts) > 1 else ""
        return {
            "id": f"edge_{index}",
            "source": source,
            "target": target,
            "condition": "",
            "label": "",
        }
    edge = _plain_mapping(item)
    return {
        "id": str(edge.get("id") or f"edge_{index}"),
        "source": str(edge.get("source") or edge.get("from") or edge.get("start") or ""),
        "target": str(edge.get("target") or edge.get("to") or edge.get("end") or ""),
        "condition": str(edge.get("condition") or edge.get("route") or ""),
        "label": str(edge.get("label") or edge.get("name") or ""),
        "metadata": _plain_mapping(edge.get("metadata")),
    }


def _normalize_workflow_step(item: Any, *, index: int) -> Dict[str, Any]:
    step = _plain_mapping(item)
    node = str(step.get("node") or step.get("node_id") or step.get("name") or "")
    return {
        "id": str(step.get("id") or step.get("step_id") or f"step_{index}"),
        "name": str(step.get("name") or node or f"step_{index}"),
        "node": node,
        "event_type": str(step.get("event_type") or step.get("event") or ""),
        "status": str(step.get("status") or step.get("outcome") or "completed"),
        "superstep": _as_int_or_zero(step.get("superstep") or step.get("turn") or index),
        "input": _plain_value(step.get("input") or step.get("inputs") or {}),
        "output": _plain_value(step.get("output") or step.get("outputs") or {}),
        "state_delta": _plain_mapping(step.get("state_delta") or step.get("writes")),
        "tool_calls": [
            _plain_mapping(call)
            for call in _plain_list(step.get("tool_calls"))
            if _plain_mapping(call)
        ],
        "tool_name": str(step.get("tool_name") or ""),
        "duration_ms": _as_int_or_zero(step.get("duration_ms") or step.get("elapsed_ms")),
        "metadata": _plain_mapping(step.get("metadata")),
    }


def _normalize_workflow_checkpoint(item: Any, *, index: int) -> Dict[str, Any]:
    checkpoint = _plain_mapping(item)
    state = _plain_mapping(checkpoint.get("state") or checkpoint.get("values"))
    return {
        "id": str(checkpoint.get("id") or checkpoint.get("checkpoint_id") or f"checkpoint_{index}"),
        "checkpoint_id": str(
            checkpoint.get("checkpoint_id")
            or checkpoint.get("id")
            or f"checkpoint_{index}"
        ),
        "thread_id": str(checkpoint.get("thread_id") or ""),
        "namespace": str(
            checkpoint.get("namespace")
            or checkpoint.get("checkpoint_ns")
            or checkpoint.get("ns")
            or ""
        ),
        "superstep": _as_int_or_zero(checkpoint.get("superstep") or index),
        "next_nodes": [str(node) for node in _plain_list(checkpoint.get("next_nodes"))],
        "state_keys": sorted(str(key) for key in state),
        "pending_writes": _plain_list(
            checkpoint.get("pending_writes") or checkpoint.get("writes")
        ),
        "metadata": _plain_mapping(checkpoint.get("metadata")),
    }


def _normalize_workflow_route(item: Any, *, index: int) -> Dict[str, Any]:
    route = _plain_mapping(item)
    return {
        "id": str(route.get("id") or f"route_{index}"),
        "source": str(route.get("source") or route.get("from") or route.get("node") or ""),
        "target": str(route.get("target") or route.get("to") or route.get("selected") or ""),
        "condition": str(route.get("condition") or route.get("route") or ""),
        "selected": str(route.get("selected") or route.get("target") or ""),
        "reason": str(route.get("reason") or ""),
        "metadata": _plain_mapping(route.get("metadata")),
    }


def _normalize_workflow_interrupt(item: Any, *, index: int) -> Dict[str, Any]:
    interrupt = _plain_mapping(item)
    return {
        "id": str(interrupt.get("id") or f"interrupt_{index}"),
        "node": str(interrupt.get("node") or interrupt.get("node_id") or ""),
        "reason": str(interrupt.get("reason") or interrupt.get("message") or ""),
        "resumable": bool(interrupt.get("resumable", True)),
        "resolved": bool(interrupt.get("resolved", False)),
        "metadata": _plain_mapping(interrupt.get("metadata")),
    }


def _normalize_workflow_replay(item: Any, *, index: int) -> Dict[str, Any]:
    replay = _plain_mapping(item)
    return {
        "id": str(replay.get("id") or f"replay_{index}"),
        "from_checkpoint": str(replay.get("from_checkpoint") or replay.get("checkpoint_id") or ""),
        "to_checkpoint": str(replay.get("to_checkpoint") or ""),
        "skipped_nodes": [str(node) for node in _plain_list(replay.get("skipped_nodes"))],
        "rerun_nodes": [str(node) for node in _plain_list(replay.get("rerun_nodes"))],
        "reason": str(replay.get("reason") or ""),
        "metadata": _plain_mapping(replay.get("metadata")),
    }


def _workflow_trace_topology(
    nodes: Sequence[Any],
    edges: Sequence[Any],
) -> Dict[str, Any]:
    node_ids = [
        str(_plain_mapping(node).get("id") or "")
        for node in nodes
        if _plain_mapping(node).get("id")
    ]
    adjacency: Dict[str, List[str]] = {}
    inbound: set[str] = set()
    outbound: set[str] = set()
    for edge in edges:
        edge_dict = _plain_mapping(edge)
        source = str(edge_dict.get("source") or "")
        target = str(edge_dict.get("target") or "")
        if not source or not target:
            continue
        adjacency.setdefault(source, []).append(target)
        outbound.add(source)
        inbound.add(target)
    return {
        "node_ids": node_ids,
        "edge_count": len(edges),
        "entry_nodes": sorted(node for node in node_ids if node not in inbound),
        "terminal_nodes": sorted(node for node in node_ids if node not in outbound),
        "adjacency": {key: sorted(values) for key, values in adjacency.items()},
    }


def _workflow_trace_field(raw: Any, name: str) -> Any:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        return raw_mapping.get(name)
    return getattr(raw, name, None)


def _openenv_trace_state(raw: Any) -> Dict[str, Any]:
    payload = _openenv_trace_payload(raw)
    if not payload:
        return {}
    return payload


def _openenv_trace_payload(raw: Any) -> Dict[str, Any]:
    mapping = _plain_mapping(raw)
    if not mapping:
        return {}

    for key in ("openenv", "open_env", "gymnasium_env", "environment_replay"):
        candidate = _plain_mapping(mapping.get(key))
        if candidate:
            return _normalize_openenv_trace_payload(candidate)

    for container_key in ("state", "output_state", "outputState", "metadata"):
        container = _plain_mapping(mapping.get(container_key))
        for key in ("openenv", "open_env", "gymnasium_env", "environment_replay"):
            candidate = _plain_mapping(container.get(key))
            if candidate:
                return _normalize_openenv_trace_payload(candidate)

    for key in ("output", "result", "data", "payload", "trace_export"):
        candidate = _plain_mapping(mapping.get(key))
        if _has_openenv_trace_shape(candidate):
            return _normalize_openenv_trace_payload(candidate)

    if _has_openenv_trace_shape(mapping):
        return _normalize_openenv_trace_payload(mapping)
    return {}


def _has_openenv_trace_shape(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    kind = _openenv_key(
        payload.get("kind")
        or payload.get("type")
        or payload.get("framework")
        or payload.get("adapter")
    )
    if kind in {
        "openenv",
        "openenv_trace",
        "open_env",
        "gymnasium",
        "gymnasium_env",
        "environment_replay",
    }:
        return True
    if any(key in payload for key in ("openenv", "open_env", "gymnasium_env")):
        return True
    if _plain_mapping(payload.get("summary")) and (
        payload.get("runtime")
        or payload.get("transport")
        or payload.get("trajectory")
        or payload.get("action_log")
    ):
        return True
    has_reset_shape = any(
        key in payload
        for key in (
            "reset",
            "reset_info",
            "initial_observation",
            "current_observation",
            "observation_space",
            "action_space",
        )
    )
    has_step_shape = any(
        key in payload
        for key in (
            "steps",
            "trajectory",
            "action_log",
            "reward",
            "terminated",
            "truncated",
            "done",
        )
    )
    if has_reset_shape and has_step_shape:
        return True
    if _plain_list(payload.get("trajectory") or payload.get("steps")) and (
        payload.get("runtime")
        or payload.get("transport")
        or payload.get("sandbox")
        or payload.get("replay")
    ):
        return True
    if "observation" in payload and any(
        key in payload for key in ("reward", "terminated", "truncated", "done", "info")
    ):
        return bool(kind or payload.get("env") or payload.get("gymnasium"))
    return False


def _normalize_openenv_trace_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = _plain_mapping(payload)
    summary = _plain_mapping(data.get("summary"))
    reset = _plain_mapping(data.get("reset"))
    sandbox = _plain_mapping(data.get("sandbox") or data.get("isolation"))
    replay = _plain_mapping(data.get("replay"))
    metadata = _plain_mapping(data.get("metadata"))
    steps = [
        _plain_mapping(item)
        for item in _plain_list(data.get("steps"))
        if _plain_mapping(item)
    ]
    trajectory = [
        _plain_mapping(item)
        for item in _plain_list(data.get("trajectory"))
        if _plain_mapping(item)
    ]
    if not trajectory:
        trajectory = steps
    action_log = [
        _plain_mapping(item)
        for item in _plain_list(data.get("action_log") or data.get("actions"))
        if _plain_mapping(item)
    ]
    if not action_log:
        action_log = [
            {
                "id": str(step.get("id") or f"step-{index}"),
                "step_index": _openenv_int(step.get("step_index"), default=index),
                "action": _plain_value(step.get("action")),
                "reward": _openenv_float(step.get("reward"), default=0.0),
                "terminated": _openenv_bool(step.get("terminated"), default=False),
                "truncated": _openenv_bool(step.get("truncated"), default=False),
                "done": _openenv_bool(step.get("done"), default=False),
                "failure_injected": _openenv_bool(
                    step.get("failure_injected"), default=False
                ),
                "metadata": _plain_mapping(step.get("metadata")),
            }
            for index, step in enumerate(trajectory, start=1)
        ]
    error_log = [
        _plain_mapping(item)
        for item in _plain_list(data.get("error_log") or data.get("errors"))
        if _plain_mapping(item)
    ]
    failure_injections = [
        _plain_mapping(item)
        for item in _plain_list(data.get("failure_injections") or data.get("faults"))
        if _plain_mapping(item)
    ]
    runtime = _openenv_key(
        data.get("runtime")
        or data.get("mode")
        or replay.get("runtime")
        or "in_process"
    )
    transport = _openenv_key(
        data.get("transport")
        or replay.get("transport")
        or ("mcp" if runtime == "mcp" else "local")
    )
    state = _plain_mapping(
        data.get("state") or data.get("current_state") or data.get("final_state")
    )
    current_observation = _plain_value(
        data.get("current_observation")
        or data.get("observation")
        or (trajectory[-1].get("observation") if trajectory else None)
    )
    initial_observation = _plain_value(
        data.get("initial_observation")
        or reset.get("observation")
        or current_observation
        or {}
    )
    reset_count = _openenv_int(summary.get("reset_count"), default=-1)
    if reset_count < 0:
        reset_count = 1 if initial_observation not in (None, "", [], {}) else 0
    step_count = _openenv_int(summary.get("step_count"), default=-1)
    if step_count < 0:
        step_count = len(trajectory)
    action_route_count = _openenv_int(summary.get("action_route_count"), default=-1)
    if action_route_count < 0:
        action_route_count = len(action_log)
    reward_total = _openenv_float(summary.get("reward_total"), default=None)
    if reward_total is None:
        reward_total = round(
            sum(
                _openenv_float(step.get("reward"), default=0.0) or 0.0
                for step in trajectory
            ),
            4,
        )
    terminated = _openenv_bool(summary.get("terminated"), default=None)
    if terminated is None:
        terminated = any(
            _openenv_bool(step.get("terminated"), default=False)
            for step in trajectory
        )
    truncated = _openenv_bool(summary.get("truncated"), default=None)
    if truncated is None:
        truncated = any(
            _openenv_bool(step.get("truncated"), default=False)
            for step in trajectory
        )
    done = _openenv_bool(summary.get("done"), default=None)
    if done is None:
        done = terminated or truncated or any(
            _openenv_bool(step.get("done"), default=False) for step in trajectory
        )
    failure_count = _openenv_int(summary.get("failure_count"), default=-1)
    if failure_count < 0:
        failure_count = max(
            len(failure_injections),
            sum(
                1
                for step in trajectory
                if _openenv_bool(step.get("failure_injected"), default=False)
                or bool(_plain_mapping(step.get("failure")))
            ),
        )
    error_count = _openenv_int(summary.get("error_count"), default=-1)
    if error_count < 0:
        error_count = len(error_log) + sum(
            1 for step in trajectory if step.get("adapter_error") or step.get("error")
        )
    metadata_capture_count = _openenv_int(
        summary.get("metadata_capture_count"), default=-1
    )
    if metadata_capture_count < 0:
        metadata_capture_count = (
            (1 if reset.get("info") or data.get("reset_info") else 0)
            + sum(
                1
                for step in trajectory
                if step.get("info") not in (None, "", [], {})
                or step.get("metadata") not in (None, "", [], {})
            )
        )
    sandbox_enabled = _openenv_bool(summary.get("sandbox_enabled"), default=None)
    if sandbox_enabled is None:
        sandbox_enabled = _openenv_bool(
            sandbox.get("enabled"), default=bool(sandbox) or True
        )
    requires_external_service = _openenv_bool(
        summary.get("requires_external_service"),
        default=_openenv_bool(data.get("requires_external_service"), default=False),
    )
    deterministic_reset = _openenv_bool(
        summary.get("deterministic_reset"),
        default=_openenv_bool(data.get("deterministic_reset"), default=True),
    )
    merged_summary = {
        **summary,
        "reset_count": reset_count,
        "step_count": step_count,
        "action_route_count": action_route_count,
        "reward_total": reward_total,
        "terminated": terminated,
        "truncated": truncated,
        "done": done,
        "failure_count": failure_count,
        "error_count": error_count,
        "metadata_capture_count": metadata_capture_count,
        "sandbox_enabled": sandbox_enabled,
        "isolation": str(
            summary.get("isolation") or sandbox.get("isolation") or "process"
        ),
        "runtime": runtime,
        "transport": transport,
        "requires_external_service": requires_external_service,
        "deterministic_reset": deterministic_reset,
        "state_key_count": _openenv_key_count(state),
        "observation_key_count": _openenv_key_count(current_observation),
    }
    merged_summary.setdefault(
        "terminal_status",
        "success" if done and error_count == 0 else "incomplete",
    )
    normalized = {
        "kind": "openenv",
        "name": str(data.get("name") or data.get("id") or "framework-openenv"),
        "runtime": runtime,
        "transport": transport,
        "requires_external_service": requires_external_service,
        "deterministic_reset": deterministic_reset,
        "action_space": _plain_mapping(data.get("action_space")),
        "observation_space": _plain_mapping(data.get("observation_space")),
        "initial_observation": initial_observation,
        "current_observation": current_observation,
        "state": state,
        "reset_info": _plain_mapping(data.get("reset_info") or reset.get("info")),
        "last_info": _plain_mapping(data.get("last_info") or data.get("info")),
        "steps": steps,
        "trajectory": trajectory,
        "action_log": action_log,
        "error_log": error_log,
        "sandbox": {
            "enabled": sandbox_enabled,
            "isolation": merged_summary["isolation"],
            **sandbox,
        },
        "replay": {"transport": transport, "deterministic": deterministic_reset, **replay},
        "failure_injections": failure_injections,
        "tool_registry": _plain_list(data.get("tool_registry") or data.get("tools")),
        "signals": _openenv_payload_signals(merged_summary, data),
        "summary": merged_summary,
        "metadata": metadata,
    }
    return {
        key: value
        for key, value in normalized.items()
        if value not in (None, "", [], {})
    }


def _openenv_trace_events(raw: Any) -> List[SimulationEvent]:
    state = _openenv_trace_state(raw)
    if not state:
        return []
    summary = _plain_mapping(state.get("summary"))
    events: List[SimulationEvent] = []
    if _openenv_int(summary.get("reset_count"), default=0) > 0:
        events.append(
            SimulationEvent(
                type="openenv",
                name="openenv_reset",
                payload={
                    "name": state.get("name"),
                    "observation": state.get("initial_observation"),
                    "info": state.get("reset_info"),
                    "state": state.get("state"),
                    "summary": summary,
                },
                metadata={
                    "kind": "openenv_trace",
                    "source": "framework_adapter_output",
                },
            )
        )
    for index, step in enumerate(_plain_list(state.get("trajectory")), start=1):
        step_dict = _plain_mapping(step)
        if not step_dict:
            continue
        events.append(
            SimulationEvent(
                type="openenv",
                name="openenv_step",
                payload={**step_dict, "sequence": index},
                metadata={
                    "kind": "openenv_trace",
                    "source": "framework_adapter_output",
                },
            )
        )
    events.append(
        SimulationEvent(
            type="openenv",
            name="openenv_state",
            payload=state,
            metadata={"kind": "openenv_trace", "source": "framework_adapter_output"},
        )
    )
    return events


def _openenv_payload_signals(
    summary: Mapping[str, Any],
    data: Mapping[str, Any],
) -> List[str]:
    signals = {
        "openenv",
        "state" if data.get("state") or summary.get("state_key_count") else "",
        "observation" if summary.get("observation_key_count") else "",
        "action" if data.get("action_space") or summary.get("action_route_count") else "",
        "reset" if summary.get("reset_count") else "",
        "step" if summary.get("step_count") else "",
        "reward" if summary.get("step_count") else "",
        "done" if summary.get("done") else "",
        "terminated" if summary.get("terminated") else "",
        "truncated" if summary.get("truncated") else "",
        "metadata" if summary.get("metadata_capture_count") else "",
        "sandbox" if summary.get("sandbox_enabled") else "",
        "failure_injection" if summary.get("failure_count") else "",
        summary.get("runtime"),
        summary.get("transport"),
    }
    signals.update(_plain_list(data.get("signals")))
    return sorted({_openenv_key(signal) for signal in signals if _openenv_key(signal)})


def _openenv_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return default
    return default


def _openenv_float(value: Any, *, default: float | None = 0.0) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _openenv_bool(value: Any, *, default: bool | None = False) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _openenv_key(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")
    aliases = {
        "open_env": "openenv",
        "gymnasium": "openenv",
        "gymnasium_env": "openenv",
        "environment_replay": "openenv",
    }
    return aliases.get(normalized, normalized)


def _openenv_key_count(value: Any) -> int:
    if isinstance(value, Mapping):
        total = len(value)
        for item in value.values():
            total += _openenv_key_count(item)
        return total
    if isinstance(value, (list, tuple, set)):
        return sum(_openenv_key_count(item) for item in value)
    return 0


def _browser_cua_state(raw: Any) -> Dict[str, Any]:
    if not _has_browser_cua_shape(raw):
        return {}
    trace = _browser_cua_trace_payload(raw)
    snapshots = _plain_list(trace.get("snapshots"))
    actions = _plain_list(trace.get("action_replay"))
    screenshots = _browser_cua_screenshots(raw, snapshots=snapshots)
    regions = _plain_mapping(trace.get("regions"))
    network_log = _plain_list(trace.get("network_log"))
    runtime_events = _plain_list(trace.get("runtime_events"))
    performance_entries = _plain_list(trace.get("performance_entries"))
    prompt_injections = _plain_list(trace.get("prompt_injections"))
    mutation_pack = _plain_mapping(trace.get("mutation_pack"))
    mutations = _plain_list(
        mutation_pack.get("mutations") or trace.get("browser_mutations")
    )
    storage_state = _plain_mapping(trace.get("storage_state"))
    return {
        "kind": "framework_browser_cua_trace",
        "url": str(trace.get("url") or ""),
        "snapshot_count": len(snapshots),
        "action_count": len(actions),
        "successful_action_count": sum(
            1 for action in actions if _plain_mapping(action).get("success") is True
        ),
        "blocked_action_count": sum(
            1 for action in actions if _plain_mapping(action).get("blocked") is True
        ),
        "matched_action_count": sum(
            1 for action in actions if _plain_mapping(action).get("matched") is True
        ),
        "screenshot_count": len(screenshots),
        "region_count": len(regions),
        "network_request_count": len(network_log),
        "runtime_event_count": len(runtime_events),
        "performance_entry_count": len(performance_entries),
        "prompt_injection_surface_count": len(prompt_injections),
        "prompt_injection_touched_count": sum(
            1
            for action in actions
            if _plain_mapping(action).get("prompt_injection_touched") is True
        ),
        "screenshot_diff_count": len(_plain_list(trace.get("screenshot_diffs"))),
        "mutation_count": len(mutations),
        "layout_shift_present": bool(trace.get("layout_shift_distribution")),
        "storage_present": bool(
            _plain_list(storage_state.get("cookies"))
            or _plain_list(storage_state.get("origins"))
        ),
        "action_types": sorted(
            {
                str(
                    _plain_mapping(action).get("action")
                    or _plain_mapping(action).get("type")
                    or ""
                )
                for action in actions
                if _plain_mapping(action).get("action")
                or _plain_mapping(action).get("type")
            }
        ),
        "tool_names": sorted(
            {
                str(tool.get("name") or "")
                for tool in _browser_cua_tool_calls(raw)
                if tool.get("name")
            }
        ),
        "screenshots": screenshots,
        "snapshots": snapshots,
        "action_replay": actions,
        "regions": regions,
        "network_log": network_log,
        "runtime_events": runtime_events,
        "performance_entries": performance_entries,
        "prompt_injections": prompt_injections,
        "mutation_pack": mutation_pack,
        "summary": {
            "snapshot_count": len(snapshots),
            "action_count": len(actions),
            "successful_action_count": sum(
                1 for action in actions if _plain_mapping(action).get("success") is True
            ),
            "screenshot_count": len(screenshots),
            "region_count": len(regions),
            "network_request_count": len(network_log),
            "prompt_injection_surface_count": len(prompt_injections),
            "mutation_count": len(mutations),
        },
    }


def _browser_cua_trace_payload(raw: Any) -> Dict[str, Any]:
    explicit_trace = _plain_mapping(_browser_cua_field(raw, "browser_trace"))
    if not explicit_trace:
        trace_export = _plain_mapping(_browser_cua_field(raw, "trace_export"))
        if _browser_cua_trace_export_has_browser_shape(trace_export):
            explicit_trace = trace_export
    snapshots = _browser_cua_snapshots(raw, explicit_trace=explicit_trace)
    actions = _browser_cua_actions(raw, explicit_trace=explicit_trace)
    regions = (
        _plain_mapping(_browser_cua_field(raw, "regions"))
        or _plain_mapping(explicit_trace.get("regions"))
    )
    mutation_pack = (
        _plain_mapping(_browser_cua_field(raw, "mutation_pack"))
        or _plain_mapping(explicit_trace.get("mutation_pack"))
    )
    mutations = [
        _plain_mapping(item)
        for item in [
            *_plain_list(_browser_cua_field(raw, "mutations")),
            *_plain_list(explicit_trace.get("browser_mutations")),
            *_plain_list(explicit_trace.get("mutations")),
        ]
        if _plain_mapping(item)
    ]
    if mutations and not mutation_pack:
        mutation_pack = {"kind": "browser_mutation_pack", "mutations": mutations}
    elif mutations and not mutation_pack.get("mutations"):
        mutation_pack = {**mutation_pack, "mutations": mutations}
    storage_state = (
        _plain_mapping(_browser_cua_field(raw, "storage_state"))
        or _plain_mapping(_browser_cua_field(raw, "storageState"))
        or _plain_mapping(explicit_trace.get("storage_state"))
        or _plain_mapping(explicit_trace.get("storageState"))
    )
    trace = {
        "kind": "browser_trace",
        "url": str(
            _browser_cua_field(raw, "url")
            or explicit_trace.get("url")
            or _browser_cua_snapshot_url(snapshots)
            or ""
        ),
        "snapshots": snapshots,
        "action_replay": actions,
        "dom_mutations": [
            _plain_mapping(item)
            for item in _plain_list(
                _browser_cua_field(raw, "dom_mutations")
                or explicit_trace.get("dom_mutations")
            )
            if _plain_mapping(item)
        ],
        "screenshot_diffs": [
            _plain_mapping(item)
            for item in _plain_list(
                _browser_cua_field(raw, "screenshot_diffs")
                or explicit_trace.get("screenshot_diffs")
            )
            if _plain_mapping(item)
        ],
        "regions": regions,
        "console_logs": _plain_list(
            _browser_cua_field(raw, "console_logs") or explicit_trace.get("console_logs")
        ),
        "network_log": [
            _plain_mapping(item)
            for item in _plain_list(
                _browser_cua_field(raw, "network_log") or explicit_trace.get("network_log")
            )
            if _plain_mapping(item)
        ],
        "resource_bodies": _plain_list(
            _browser_cua_field(raw, "resource_bodies")
            or explicit_trace.get("resource_bodies")
        ),
        "actionability_timeline": [
            _plain_mapping(item)
            for item in _plain_list(
                _browser_cua_field(raw, "actionability_timeline")
                or explicit_trace.get("actionability_timeline")
            )
            if _plain_mapping(item)
        ],
        "storage_state": storage_state,
        "runtime_events": [
            _plain_mapping(item)
            for item in _plain_list(
                _browser_cua_field(raw, "runtime_events")
                or explicit_trace.get("runtime_events")
            )
            if _plain_mapping(item)
        ],
        "performance_entries": [
            _plain_mapping(item)
            for item in _plain_list(
                _browser_cua_field(raw, "performance_entries")
                or explicit_trace.get("performance_entries")
            )
            if _plain_mapping(item)
        ],
        "prompt_injections": [
            _plain_mapping(item)
            for item in _plain_list(
                _browser_cua_field(raw, "prompt_injections")
                or _browser_cua_field(raw, "prompt_injection_surfaces")
                or explicit_trace.get("prompt_injections")
                or explicit_trace.get("prompt_injection_surfaces")
            )
            if _plain_mapping(item)
        ],
        "video_artifacts": _plain_list(
            _browser_cua_field(raw, "video_artifacts")
            or explicit_trace.get("video_artifacts")
        ),
        "perturbations": [
            _plain_mapping(item)
            for item in _plain_list(
                _browser_cua_field(raw, "perturbations")
                or explicit_trace.get("perturbations")
            )
            if _plain_mapping(item)
        ],
        "mutation_pack": mutation_pack,
        "browser_mutations": _plain_list(mutation_pack.get("mutations")),
        "layout_shift_distribution": _plain_value(
            _browser_cua_field(raw, "layout_shift_distribution")
            or explicit_trace.get("layout_shift_distribution")
            or {}
        ),
        "trace_import": {
            "source": "framework_adapter_output",
            "provider": str(
                _browser_cua_field(raw, "trace_provider")
                or explicit_trace.get("trace_provider")
                or "framework_browser_cua"
            ),
        },
    }
    trace["final_state"] = {
        "browser": {
            "url": trace["url"],
            "snapshot": snapshots[-1] if snapshots else {},
            "action_replay": actions,
            "regions": regions,
            "storage_state": storage_state,
            "runtime_events": trace["runtime_events"],
            "performance_entries": trace["performance_entries"],
            "network_log": trace["network_log"],
            "mutation_pack": mutation_pack,
            "browser_mutations": trace["browser_mutations"],
            "layout_shift_distribution": trace["layout_shift_distribution"],
        }
    }
    return trace


def _browser_cua_events(raw: Any) -> List[SimulationEvent]:
    if not _has_browser_cua_shape(raw):
        return []
    trace = _browser_cua_trace_payload(raw)
    events: List[SimulationEvent] = []
    for index, snapshot in enumerate(_plain_list(trace.get("snapshots")), start=1):
        snapshot_dict = _plain_mapping(snapshot)
        events.append(
            SimulationEvent(
                type="browser_snapshot",
                name=str(snapshot_dict.get("id") or f"snapshot_{index}"),
                payload={**snapshot_dict, "sequence": index},
                metadata={"kind": "browser_cua", "source": "framework_adapter_output"},
            )
        )
    for index, action in enumerate(_plain_list(trace.get("action_replay")), start=1):
        action_dict = _plain_mapping(action)
        events.append(
            SimulationEvent(
                type="browser_action",
                name=str(
                    action_dict.get("tool")
                    or action_dict.get("tool_name")
                    or action_dict.get("action")
                    or f"browser_action_{index}"
                ),
                payload={**action_dict, "sequence": index},
                metadata={"kind": "browser_cua", "source": "framework_adapter_output"},
            )
        )
    if trace.get("network_log"):
        events.append(
            SimulationEvent(
                type="browser_network",
                name="network_log_loaded",
                payload={"requests": trace["network_log"]},
                metadata={"kind": "browser_cua", "source": "framework_adapter_output"},
            )
        )
    if trace.get("runtime_events") or trace.get("performance_entries"):
        events.append(
            SimulationEvent(
                type="browser_runtime",
                name="runtime_capture_loaded",
                payload={
                    "runtime_events": trace["runtime_events"],
                    "performance_entries": trace["performance_entries"],
                },
                metadata={"kind": "browser_cua", "source": "framework_adapter_output"},
            )
        )
    if trace.get("storage_state"):
        events.append(
            SimulationEvent(
                type="browser_storage",
                name="storage_state_loaded",
                payload={"storage_state": trace["storage_state"]},
                metadata={"kind": "browser_cua", "source": "framework_adapter_output"},
            )
        )
    if trace.get("mutation_pack"):
        events.append(
            SimulationEvent(
                type="browser_mutation_pack",
                name="browser_mutation_pack_loaded",
                payload=trace["mutation_pack"],
                metadata={"kind": "browser_cua", "source": "framework_adapter_output"},
            )
        )
    for index, injection in enumerate(_plain_list(trace.get("prompt_injections")), start=1):
        injection_dict = _plain_mapping(injection)
        events.append(
            SimulationEvent(
                type="environment_injection",
                name=str(injection_dict.get("id") or f"prompt_injection_{index}"),
                payload=injection_dict,
                metadata={"kind": "browser_cua", "source": "framework_adapter_output"},
            )
        )
    events.append(
        SimulationEvent(
            type="browser_trace",
            name="framework_browser_cua_trace",
            payload=trace,
            metadata={"kind": "browser_trace", "source": "framework_adapter_output"},
        )
    )
    return events


def _browser_cua_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    if not _has_browser_cua_shape(raw):
        return []
    calls: List[Dict[str, Any]] = []
    for index, action in enumerate(_browser_cua_actions(raw), start=1):
        action_dict = _plain_mapping(action)
        name = _browser_cua_tool_name(action_dict)
        arguments = _browser_cua_action_arguments(action_dict)
        calls.append(
            {
                "id": str(
                    action_dict.get("id")
                    or action_dict.get("call_id")
                    or f"{name}_{index}"
                ),
                "name": name,
                "arguments": arguments,
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return calls


def _has_browser_cua_shape(raw: Any) -> bool:
    raw_mapping = _object_mapping(raw)
    names = (
        "browser_trace",
        "browser_actions",
        "computer_actions",
        "cua_actions",
        "action_replay",
        "browser_snapshots",
        "dom_snapshots",
        "screenshots",
        "screenshot_diffs",
        "prompt_injections",
        "prompt_injection_surfaces",
        "mutation_pack",
        "browser_mutations",
    )
    if raw_mapping is not None:
        return any(raw_mapping.get(name) not in (None, "", [], {}) for name in names) or (
            _browser_cua_trace_export_has_browser_shape(
                _plain_mapping(raw_mapping.get("trace_export"))
            )
        )
    return any(
        hasattr(raw, name) and getattr(raw, name) not in (None, "", [], {})
        for name in names
    ) or _browser_cua_trace_export_has_browser_shape(
        _plain_mapping(getattr(raw, "trace_export", None))
    )


def _browser_cua_trace_export_has_browser_shape(trace_export: Mapping[str, Any]) -> bool:
    if not trace_export:
        return False
    if any(
        trace_export.get(name) not in (None, "", [], {})
        for name in (
            "browser_actions",
            "computer_actions",
            "cua_actions",
            "action_replay",
            "browser_snapshots",
            "dom_snapshots",
            "screenshots",
            "screenshot_diffs",
            "prompt_injections",
            "prompt_injection_surfaces",
            "mutation_pack",
            "browser_mutations",
            "regions",
        )
    ):
        return True
    kind = _memory_key(
        trace_export.get("kind")
        or trace_export.get("type")
        or trace_export.get("trace_provider")
    )
    if any(token in kind for token in ("browser", "computer", "cua")):
        return True
    return bool(
        (
            trace_export.get("actions")
            or trace_export.get("snapshots")
            or trace_export.get("url")
        )
        and not (
            trace_export.get("resourceSpans")
            or trace_export.get("resource_spans")
            or trace_export.get("scopeSpans")
            or trace_export.get("scope_spans")
        )
    )


def _browser_cua_actions(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace)
    values: List[Any] = []
    for name in ("browser_actions", "computer_actions", "cua_actions", "action_replay"):
        values.extend(_plain_list(_browser_cua_field(raw, name)))
    values.extend(_plain_list(trace.get("actions")))
    values.extend(_plain_list(trace.get("action_replay")))
    actions = [
        _normalize_browser_cua_action(action, index=index)
        for index, action in enumerate(values, start=1)
        if _plain_mapping(action)
    ]
    return actions


def _browser_cua_snapshots(
    raw: Any,
    *,
    explicit_trace: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = _plain_mapping(explicit_trace)
    values: List[Any] = []
    for name in ("browser_snapshots", "dom_snapshots", "snapshots"):
        values.extend(_plain_list(_browser_cua_field(raw, name)))
    values.extend(_plain_list(trace.get("snapshots")))
    screenshots = _plain_list(_browser_cua_field(raw, "screenshots"))
    snapshots = [
        _normalize_browser_cua_snapshot(snapshot, index=index)
        for index, snapshot in enumerate(values, start=1)
        if _plain_mapping(snapshot)
    ]
    if not snapshots and screenshots:
        snapshots = [
            _normalize_browser_cua_snapshot(screenshot, index=index)
            for index, screenshot in enumerate(screenshots, start=1)
            if _plain_mapping(screenshot)
        ]
    return snapshots


def _browser_cua_screenshots(
    raw: Any,
    *,
    snapshots: Sequence[Any],
) -> List[Dict[str, Any]]:
    screenshots = [
        _plain_mapping(item)
        for item in _plain_list(_browser_cua_field(raw, "screenshots"))
        if _plain_mapping(item)
    ]
    for snapshot in snapshots:
        snapshot_dict = _plain_mapping(snapshot)
        uri = snapshot_dict.get("screenshot_uri") or snapshot_dict.get("uri")
        if not uri:
            continue
        screenshots.append(
            {
                "id": str(
                    snapshot_dict.get("id")
                    or f"screenshot_{len(screenshots) + 1}"
                ),
                "uri": str(uri),
                "screenshot_uri": str(uri),
            }
        )
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for screenshot in screenshots:
        uri = str(screenshot.get("uri") or screenshot.get("screenshot_uri") or "")
        key = uri or json.dumps(screenshot, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(screenshot)
    return deduped


def _normalize_browser_cua_snapshot(
    item: Any,
    *,
    index: int,
) -> Dict[str, Any]:
    snapshot = _plain_mapping(item)
    metadata = _plain_mapping(snapshot.get("metadata"))
    dom = str(snapshot.get("dom") or snapshot.get("html") or "")
    screenshot_uri = str(snapshot.get("screenshot_uri") or snapshot.get("uri") or "")
    return {
        "id": str(
            snapshot.get("id")
            or snapshot.get("snapshot_id")
            or f"snapshot_{index}"
        ),
        "url": str(snapshot.get("url") or ""),
        "title": str(snapshot.get("title") or ""),
        "dom": dom,
        "screenshot_uri": screenshot_uri,
        "has_dom": bool(snapshot.get("has_dom", bool(dom))),
        "has_screenshot": bool(snapshot.get("has_screenshot", bool(screenshot_uri))),
        "metadata": {
            **metadata,
            "stale": bool(metadata.get("stale", snapshot.get("stale", False))),
            "stale_screenshot": bool(
                metadata.get(
                    "stale_screenshot",
                    snapshot.get("stale_screenshot", False),
                )
            ),
        },
    }


def _normalize_browser_cua_action(
    item: Any,
    *,
    index: int,
) -> Dict[str, Any]:
    action = _plain_mapping(item)
    arguments = _browser_cua_action_arguments(action)
    selector = (
        action.get("selector")
        or action.get("locator")
        or arguments.get("selector")
        or arguments.get("locator")
    )
    coordinates = (
        _plain_mapping(action.get("coordinates"))
        or _plain_mapping(arguments.get("coordinates"))
        or {
            key: action.get(key, arguments.get(key))
            for key in ("x", "y")
            if action.get(key, arguments.get(key)) is not None
        }
    )
    region = _plain_mapping(action.get("region") or action.get("observed_region"))
    return {
        "id": str(
            action.get("id")
            or action.get("call_id")
            or f"browser_action_{index}"
        ),
        "tool": _browser_cua_tool_name(action),
        "tool_name": _browser_cua_tool_name(action),
        "action": str(
            action.get("action")
            or action.get("type")
            or arguments.get("action")
            or "action"
        ),
        "selector": str(selector or ""),
        "url": str(action.get("url") or arguments.get("url") or ""),
        "coordinates": coordinates,
        "region": region,
        "observed_region": _plain_mapping(action.get("observed_region")) or region,
        "success": bool(action.get("success", True)),
        "blocked": bool(action.get("blocked", False)),
        "matched": bool(action.get("matched", True)),
        "region_matched": bool(action.get("region_matched", bool(region))),
        "prompt_injection_touched": bool(action.get("prompt_injection_touched", False)),
        "prompt_injection_surfaces": _plain_list(action.get("prompt_injection_surfaces")),
        "screenshot_diff": _plain_value(action.get("screenshot_diff") or {}),
        "mutation_id": str(action.get("mutation_id") or ""),
        "mutation_type": str(action.get("mutation_type") or ""),
        "arguments": arguments,
    }


def _browser_cua_action_arguments(action: Mapping[str, Any]) -> Dict[str, Any]:
    arguments = _plain_mapping(action.get("arguments") or action.get("args"))
    for key in ("action", "selector", "locator", "url", "x", "y"):
        value = action.get(key)
        if value not in (None, "", [], {}) and key not in arguments:
            arguments[key] = _plain_value(value)
    coordinates = _plain_mapping(action.get("coordinates"))
    if coordinates and "coordinates" not in arguments:
        arguments["coordinates"] = coordinates
    return arguments


def _browser_cua_tool_name(action: Mapping[str, Any]) -> str:
    for key in ("tool_name", "tool", "name"):
        value = action.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    action_type = str(action.get("action") or action.get("type") or "").lower()
    if action_type in {"click", "tap", "press"}:
        return "browser_click"
    if action_type in {"navigate", "goto", "open"}:
        return "browser_navigate"
    if action_type in {"type", "fill", "input"}:
        return "browser_type"
    if action_type in {"screenshot", "snapshot"}:
        return "browser_snapshot"
    if action_type in {"scroll"}:
        return "browser_scroll"
    return "browser_action"


def _browser_cua_field(raw: Any, name: str) -> Any:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        return raw_mapping.get(name)
    return getattr(raw, name, None)


def _browser_cua_snapshot_url(snapshots: Sequence[Mapping[str, Any]]) -> str:
    for snapshot in reversed(list(snapshots)):
        url = str(_plain_mapping(snapshot).get("url") or "")
        if url:
            return url
    return ""


def _framework_memory_state(raw: Any) -> Dict[str, Any]:
    if not _has_framework_memory_shape(raw):
        return {}
    operations = _framework_memory_operations(raw)
    checkpoints = _framework_memory_checkpoints(raw)
    memories = _framework_memory_records(raw)
    retrievals = _framework_memory_retrievals(raw)
    stores = _framework_memory_stores(raw)
    policies = _framework_memory_policies(raw)
    if not any((operations, checkpoints, memories, retrievals, stores, policies)):
        return {}

    operation_types = sorted(
        {
            _memory_key(operation.get("operation") or operation.get("type") or operation.get("op"))
            for operation in operations
            if _memory_key(operation.get("operation") or operation.get("type") or operation.get("op"))
        }
    )
    namespaces = sorted(
        {
            str(
                item.get("namespace")
                or item.get("tenant")
                or item.get("user_id")
                or item.get("thread_id")
                or ""
            )
            for item in [*operations, *checkpoints, *memories, *retrievals, *stores]
            if (
                item.get("namespace")
                or item.get("tenant")
                or item.get("user_id")
                or item.get("thread_id")
            )
        }
    )
    thread_ids = sorted(
        {
            str(item.get("thread_id") or "")
            for item in [*operations, *checkpoints, *retrievals]
            if item.get("thread_id")
        }
    )
    source_ids = sorted(
        {
            str(source_id)
            for memory in memories
            for source_id in _plain_list(
                memory.get("source_ids")
                or memory.get("sources")
                or memory.get("doc_ids")
            )
            if str(source_id)
        }
    )
    retrieval_doc_ids = sorted(
        {
            str(document.get("id") or document.get("doc_id") or document.get("key") or "")
            for retrieval in retrievals
            for document in _list_of_mappings(retrieval.get("documents") or retrieval.get("results"))
            if document.get("id") or document.get("doc_id") or document.get("key")
        }
    )
    policy_keys = sorted(_memory_key(key) for key in policies if _memory_key(key))
    signals = sorted(
        {
            "memory",
            "framework_memory",
            *(operation_types or []),
            *(["checkpoint"] if checkpoints else []),
            *(["retrieval"] if retrievals else []),
            *(["memory_record"] if memories else []),
            *(["store"] if stores else []),
            *(["policy"] if policies else []),
            *(["source_attribution"] if source_ids else []),
        }
    )
    return {
        "kind": "framework_memory_trace",
        "operation_count": len(operations),
        "checkpoint_count": len(checkpoints),
        "memory_count": len(memories),
        "retrieval_count": len(retrievals),
        "store_count": len(stores),
        "policy_count": len(policies),
        "operation_types": operation_types,
        "namespaces": namespaces,
        "thread_ids": thread_ids,
        "source_ids": source_ids,
        "retrieval_doc_ids": retrieval_doc_ids,
        "policy_keys": policy_keys,
        "signals": signals,
        "stores": stores,
        "memories": memories,
        "operations": operations,
        "checkpoints": checkpoints,
        "retrievals": retrievals,
        "policies": policies,
        "summary": {
            "operation_count": len(operations),
            "checkpoint_count": len(checkpoints),
            "memory_count": len(memories),
            "retrieval_count": len(retrievals),
            "store_count": len(stores),
            "has_read": "read" in operation_types or "search" in operation_types,
            "has_write": "write" in operation_types or "add" in operation_types,
            "has_recall": "recall" in operation_types or bool(retrievals),
            "has_update": "update" in operation_types,
            "has_delete": "delete" in operation_types,
            "has_checkpoint": bool(checkpoints),
            "has_source_attribution": bool(source_ids),
            "has_policy": bool(policies),
        },
    }


def _framework_memory_events(raw: Any) -> List[SimulationEvent]:
    if not _has_framework_memory_shape(raw):
        return []
    events: List[SimulationEvent] = []
    for index, operation in enumerate(_framework_memory_operations(raw), start=1):
        operation_type = _memory_key(
            operation.get("operation") or operation.get("type") or operation.get("op")
        ) or "memory_operation"
        events.append(
            SimulationEvent(
                type="framework_memory_operation",
                name=operation_type,
                payload={**operation, "sequence": index},
                metadata={"kind": "framework_memory", "operation": operation_type},
            )
        )
    for index, checkpoint in enumerate(_framework_memory_checkpoints(raw), start=1):
        name = str(
            checkpoint.get("id")
            or checkpoint.get("checkpoint_id")
            or checkpoint.get("thread_id")
            or f"checkpoint_{index}"
        )
        events.append(
            SimulationEvent(
                type="framework_memory_checkpoint",
                name=name,
                payload={**checkpoint, "sequence": index},
                metadata={"kind": "framework_memory", "memory": "checkpoint"},
            )
        )
    for index, retrieval in enumerate(_framework_memory_retrievals(raw), start=1):
        name = str(retrieval.get("query") or retrieval.get("id") or f"retrieval_{index}")
        events.append(
            SimulationEvent(
                type="framework_memory_retrieval",
                name=name,
                payload={**retrieval, "sequence": index},
                metadata={"kind": "framework_memory", "memory": "retrieval"},
            )
        )
    for index, memory in enumerate(_framework_memory_records(raw), start=1):
        name = str(memory.get("id") or memory.get("key") or f"memory_{index}")
        events.append(
            SimulationEvent(
                type="framework_memory_record",
                name=name,
                payload={**memory, "sequence": index},
                metadata={"kind": "framework_memory", "memory": "record"},
            )
        )
    return events


def _framework_memory_updates(raw: Any) -> Dict[str, Any]:
    operations = _framework_memory_operations(raw)
    if not operations:
        return {}
    writes = [
        operation
        for operation in operations
        if _memory_key(operation.get("operation") or operation.get("type") or operation.get("op"))
        in {"add", "write", "remember", "save", "put", "upsert", "set"}
    ]
    updates = [
        operation
        for operation in operations
        if _memory_key(operation.get("operation") or operation.get("type") or operation.get("op"))
        == "update"
    ]
    deletes = [
        operation
        for operation in operations
        if _memory_key(operation.get("operation") or operation.get("type") or operation.get("op"))
        in {"delete", "forget", "remove", "purge"}
    ]
    if not writes and not updates and not deletes:
        return {}
    return {
        "framework_memory": {
            "write_count": len(writes),
            "update_count": len(updates),
            "delete_count": len(deletes),
            "writes": writes,
            "updates": updates,
            "deletes": deletes,
        }
    }


def _framework_memory_retrieval_memory(raw: Any) -> Dict[str, Any]:
    retrievals = _framework_memory_retrievals(raw)
    memories = _framework_memory_records(raw)
    if not retrievals and not memories:
        return {}
    documents: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    for index, retrieval in enumerate(retrievals, start=1):
        query = str(retrieval.get("query") or retrieval.get("input") or f"memory retrieval {index}")
        docs = [
            _framework_memory_document(document, index=doc_index)
            for doc_index, document in enumerate(
                _plain_list(retrieval.get("documents") or retrieval.get("results")),
                start=1,
            )
        ]
        docs = [doc for doc in docs if doc]
        documents.extend(docs)
        queries.append(
            {
                "query": query,
                "documents": [str(doc.get("id")) for doc in docs if doc.get("id")],
            }
        )
        cited_ids = [
            str(doc_id)
            for doc_id in _plain_list(retrieval.get("doc_ids") or retrieval.get("source_ids"))
            if str(doc_id)
        ] or [str(doc.get("id")) for doc in docs if doc.get("id")]
        if cited_ids:
            citations.append(
                {
                    "claim": str(retrieval.get("claim") or query),
                    "doc_ids": cited_ids,
                    "freshness_checked": bool(retrieval.get("freshness_checked", True)),
                }
            )
    for index, memory in enumerate(memories, start=1):
        source_ids = [
            str(item)
            for item in _plain_list(
                memory.get("source_ids") or memory.get("sources") or memory.get("doc_ids")
            )
            if str(item)
        ]
        if not source_ids:
            continue
        citations.append(
            {
                "claim": str(memory.get("content") or memory.get("value") or f"memory {index}"),
                "doc_ids": source_ids,
                "freshness_checked": True,
            }
        )
    return {
        "documents": _dedupe_framework_memory_documents(documents),
        "queries": queries,
        "citations": citations,
        "memory_writes": [
            {
                "key": str(memory.get("id") or memory.get("key") or index),
                "value": str(memory.get("content") or memory.get("value") or ""),
            }
            for index, memory in enumerate(memories, start=1)
        ],
        "require_current": True,
    }


def _framework_memory_agent_lineage(raw: Any) -> Dict[str, Any]:
    state = _framework_memory_state(raw)
    if not state:
        return {}
    stores = _framework_memory_stores(raw) or [
        {
            "id": "framework_memory",
            "type": "framework",
            "tenant": next(iter(state.get("namespaces") or ["default"]), "default"),
        }
    ]
    memories = [
        {
            "id": str(memory.get("id") or memory.get("key") or index),
            "store": str(memory.get("store") or stores[0].get("id") or "framework_memory"),
            "status": str(memory.get("status") or "active"),
            "source_ids": _plain_list(
                memory.get("source_ids") or memory.get("sources") or memory.get("doc_ids")
            ),
            "tenant": str(
                memory.get("namespace")
                or memory.get("tenant")
                or stores[0].get("tenant")
                or "default"
            ),
        }
        for index, memory in enumerate(_framework_memory_records(raw), start=1)
    ]
    operations = [
        {
            "id": str(operation.get("id") or f"memory_operation_{index}"),
            "operation": _memory_key(
                operation.get("operation") or operation.get("type") or operation.get("op")
            )
            or "operation",
            "store": str(operation.get("store") or stores[0].get("id") or "framework_memory"),
            "memory_id": str(
                operation.get("memory_id")
                or operation.get("key")
                or operation.get("id")
                or f"memory_{index}"
            ),
            "status": str(operation.get("status") or "allowed"),
            "policy_decision": str(operation.get("policy_decision") or "allowed"),
            "trace_id": str(operation.get("trace_id") or operation.get("span_id") or ""),
            "evidence": _plain_value(operation.get("evidence") or {}),
        }
        for index, operation in enumerate(_framework_memory_operations(raw), start=1)
    ]
    lineage_edges = [
        {
            "from": str(source_id),
            "to": str(memory.get("id") or memory.get("key") or index),
            "type": "source_attribution",
        }
        for index, memory in enumerate(_framework_memory_records(raw), start=1)
        for source_id in _plain_list(
            memory.get("source_ids") or memory.get("sources") or memory.get("doc_ids")
        )
        if str(source_id)
    ]
    policies = _framework_memory_policies(raw)
    return {
        "target": {
            "agent": "framework-adapter",
            "tenant": next(iter(state.get("namespaces") or ["default"]), "default"),
        },
        "stores": stores,
        "memories": memories,
        "operations": operations,
        "checkpoints": _framework_memory_checkpoints(raw),
        "lineage": lineage_edges,
        "policies": policies,
        "poison_tests": _framework_memory_named_tests(raw, "poison"),
        "isolation_tests": _framework_memory_named_tests(raw, "isolation"),
        "retention_tests": _framework_memory_named_tests(raw, "retention"),
        "observability": _framework_memory_observability(raw),
        "artifacts": _framework_memory_audit_artifacts(raw),
        "required_evidence": [
            "source_attribution",
            "tenant_isolation",
            "audit",
            "retention_policy",
            "deletion_policy",
            "redaction",
            "canary",
        ],
        "required_signals": [
            "memory_lineage",
            "source_attribution",
            "tenant_isolation",
            "audit",
        ],
    }


def _has_framework_memory_shape(raw: Any) -> bool:
    raw_mapping = _object_mapping(raw)
    names = (
        "memory_trace",
        "memory_operations",
        "memoryOperations",
        "memory_ops",
        "memory_records",
        "memoryRecords",
        "memory_searches",
        "memorySearches",
        "memory_retrievals",
        "memoryRetrievals",
        "memory_stores",
        "memoryStores",
        "checkpoints",
        "checkpoint_writes",
        "thread_checkpoints",
        "graph_checkpoints",
        "retrievals",
    )
    if raw_mapping is not None:
        return any(raw_mapping.get(name) not in (None, "", [], {}) for name in names)
    return any(
        hasattr(raw, name) and getattr(raw, name) not in (None, "", [], {})
        for name in names
    )


def _framework_memory_operations(raw: Any) -> List[Dict[str, Any]]:
    explicit = _extract_list_field(
        raw,
        (
            "memory_operations",
            "memoryOperations",
            "memory_ops",
            "memoryOps",
        ),
    )
    trace = _object_mapping(_framework_memory_field(raw, "memory_trace"))
    trace_operations = _list_of_mappings(trace.get("operations")) if trace else []
    return [
        _normalize_framework_memory_operation(item, index=index)
        for index, item in enumerate([*(explicit or []), *trace_operations], start=1)
    ]


def _framework_memory_checkpoints(raw: Any) -> List[Dict[str, Any]]:
    checkpoints = _extract_list_field(
        raw,
        (
            "checkpoints",
            "checkpoint_writes",
            "thread_checkpoints",
            "graph_checkpoints",
        ),
    )
    trace = _object_mapping(_framework_memory_field(raw, "memory_trace"))
    trace_checkpoints = _list_of_mappings(trace.get("checkpoints")) if trace else []
    return [
        _normalize_framework_memory_checkpoint(item, index=index)
        for index, item in enumerate([*(checkpoints or []), *trace_checkpoints], start=1)
    ]


def _framework_memory_records(raw: Any) -> List[Dict[str, Any]]:
    memories = _extract_list_field(
        raw,
        (
            "memory_records",
            "memoryRecords",
            "memories",
        ),
    )
    trace = _object_mapping(_framework_memory_field(raw, "memory_trace"))
    trace_memories = _list_of_mappings(trace.get("memories")) if trace else []
    return [
        _normalize_framework_memory_record(item, index=index)
        for index, item in enumerate([*(memories or []), *trace_memories], start=1)
    ]


def _framework_memory_retrievals(raw: Any) -> List[Dict[str, Any]]:
    retrievals = _extract_list_field(
        raw,
        (
            "memory_searches",
            "memorySearches",
            "memory_retrievals",
            "memoryRetrievals",
            "retrievals",
        ),
    )
    trace = _object_mapping(_framework_memory_field(raw, "memory_trace"))
    trace_retrievals = _list_of_mappings(trace.get("retrievals")) if trace else []
    return [
        _normalize_framework_memory_retrieval(item, index=index)
        for index, item in enumerate([*(retrievals or []), *trace_retrievals], start=1)
    ]


def _framework_memory_stores(raw: Any) -> List[Dict[str, Any]]:
    stores = _extract_list_field(raw, ("memory_stores", "memoryStores"))
    trace = _object_mapping(_framework_memory_field(raw, "memory_trace"))
    trace_stores = _list_of_mappings(trace.get("stores")) if trace else []
    return [
        _normalize_framework_memory_store(item, index=index)
        for index, item in enumerate([*(stores or []), *trace_stores], start=1)
    ]


def _framework_memory_policies(raw: Any) -> Dict[str, Any]:
    for name in ("memory_policies", "memoryPolicies"):
        value = _object_mapping(_framework_memory_field(raw, name))
        if value:
            return value
    trace = _object_mapping(_framework_memory_field(raw, "memory_trace"))
    if trace:
        return _plain_mapping(trace.get("policies"))
    return {}


def _framework_memory_named_tests(raw: Any, family: str) -> List[Dict[str, Any]]:
    names = {
        "poison": ("poison_tests", "poisoning_tests", "memory_poison_tests"),
        "isolation": ("isolation_tests", "memory_isolation_tests"),
        "retention": ("retention_tests", "deletion_tests", "memory_retention_tests"),
    }.get(family, ())
    values: List[Dict[str, Any]] = []
    for name in names:
        values.extend(_extract_list_field(raw, (name,)) or [])
    trace = _object_mapping(_framework_memory_field(raw, "memory_trace"))
    if trace:
        for name in names:
            values.extend(_list_of_mappings(trace.get(name)))
    return values


def _framework_memory_observability(raw: Any) -> Dict[str, Any]:
    for name in ("memory_observability", "observability"):
        value = _object_mapping(_framework_memory_field(raw, name))
        if value:
            return value
    trace = _object_mapping(_framework_memory_field(raw, "memory_trace"))
    if trace:
        return _plain_mapping(trace.get("observability"))
    return {}


def _framework_memory_audit_artifacts(raw: Any) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for name in ("memory_artifacts", "audit_artifacts"):
        values.extend(_extract_list_field(raw, (name,)) or [])
    trace = _object_mapping(_framework_memory_field(raw, "memory_trace"))
    if trace:
        values.extend(_list_of_mappings(trace.get("artifacts")))
    return values


def _framework_memory_field(raw: Any, name: str) -> Any:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        return raw_mapping.get(name)
    return getattr(raw, name, None)


def _normalize_framework_memory_operation(
    item: Mapping[str, Any],
    *,
    index: int,
) -> Dict[str, Any]:
    operation = _memory_key(item.get("operation") or item.get("type") or item.get("op"))
    return {
        "id": str(item.get("id") or item.get("operation_id") or f"memory_operation_{index}"),
        "operation": operation or "operation",
        "memory_id": str(item.get("memory_id") or item.get("key") or item.get("id") or ""),
        "key": str(item.get("key") or item.get("memory_id") or item.get("id") or ""),
        "namespace": str(item.get("namespace") or item.get("tenant") or item.get("user_id") or ""),
        "thread_id": str(item.get("thread_id") or ""),
        "status": str(item.get("status") or "allowed"),
        "policy_decision": str(item.get("policy_decision") or "allowed"),
        "trace_id": str(item.get("trace_id") or item.get("span_id") or ""),
        "value": _plain_value(item.get("value") or item.get("content") or item.get("text") or ""),
        "source_ids": _plain_list(item.get("source_ids") or item.get("sources") or item.get("doc_ids")),
        "evidence": _plain_value(item.get("evidence") or {}),
    }


def _normalize_framework_memory_checkpoint(
    item: Mapping[str, Any],
    *,
    index: int,
) -> Dict[str, Any]:
    return {
        "id": str(item.get("id") or item.get("checkpoint_id") or f"checkpoint_{index}"),
        "checkpoint_id": str(item.get("checkpoint_id") or item.get("id") or f"checkpoint_{index}"),
        "thread_id": str(item.get("thread_id") or item.get("thread") or ""),
        "namespace": str(item.get("namespace") or item.get("tenant") or ""),
        "state_keys": [
            str(key)
            for key in _plain_list(item.get("state_keys") or item.get("keys"))
            if str(key)
        ],
        "status": str(item.get("status") or "saved"),
        "trace_id": str(item.get("trace_id") or item.get("span_id") or ""),
    }


def _normalize_framework_memory_record(
    item: Mapping[str, Any],
    *,
    index: int,
) -> Dict[str, Any]:
    return {
        "id": str(item.get("id") or item.get("key") or f"memory_{index}"),
        "key": str(item.get("key") or item.get("id") or f"memory_{index}"),
        "store": str(item.get("store") or item.get("store_id") or "framework_memory"),
        "namespace": str(item.get("namespace") or item.get("tenant") or item.get("user_id") or ""),
        "content": str(item.get("content") or item.get("value") or item.get("text") or ""),
        "status": str(item.get("status") or "active"),
        "source_ids": _plain_list(item.get("source_ids") or item.get("sources") or item.get("doc_ids")),
        "metadata": _plain_mapping(item.get("metadata")),
    }


def _normalize_framework_memory_retrieval(
    item: Mapping[str, Any],
    *,
    index: int,
) -> Dict[str, Any]:
    return {
        "id": str(item.get("id") or f"retrieval_{index}"),
        "query": str(item.get("query") or item.get("input") or ""),
        "namespace": str(item.get("namespace") or item.get("tenant") or item.get("user_id") or ""),
        "thread_id": str(item.get("thread_id") or ""),
        "documents": [
            _framework_memory_document(document, index=doc_index)
            for doc_index, document in enumerate(
                _plain_list(item.get("documents") or item.get("results")),
                start=1,
            )
        ],
        "doc_ids": _plain_list(item.get("doc_ids") or item.get("source_ids")),
        "freshness_checked": bool(item.get("freshness_checked", True)),
        "status": str(item.get("status") or "returned"),
    }


def _normalize_framework_memory_store(
    item: Mapping[str, Any],
    *,
    index: int,
) -> Dict[str, Any]:
    return {
        "id": str(item.get("id") or item.get("name") or f"memory_store_{index}"),
        "type": str(item.get("type") or item.get("kind") or "framework"),
        "tenant": str(item.get("tenant") or item.get("namespace") or "default"),
        "namespace": str(item.get("namespace") or item.get("tenant") or "default"),
    }


def _framework_memory_document(value: Any, *, index: int) -> Dict[str, Any]:
    item = _object_mapping(value)
    if not item:
        return {
            "id": f"doc_{index}",
            "content": str(value),
            "current": True,
        }
    return {
        "id": str(item.get("id") or item.get("doc_id") or item.get("key") or f"doc_{index}"),
        "title": str(item.get("title") or item.get("name") or ""),
        "content": str(item.get("content") or item.get("text") or item.get("value") or ""),
        "current": bool(item.get("current", True)),
    }


def _dedupe_framework_memory_documents(
    documents: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for document in documents:
        doc_id = str(document.get("id") or "")
        if doc_id and doc_id in seen:
            continue
        if doc_id:
            seen.add(doc_id)
        deduped.append(dict(document))
    return deduped


def _plain_mapping(value: Any) -> Dict[str, Any]:
    mapping = _object_mapping(value)
    return dict(mapping or {})


def _plain_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item) for item in value]
    return [_plain_value(value)]


def _memory_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _realtime_trace_state(raw: Any) -> Dict[str, Any]:
    frames = _realtime_frames(raw)
    events = _realtime_session_events(raw)
    if not frames and not events:
        return {}

    frame_entries = [
        _realtime_item_entry(frame, index=index, source="frame")
        for index, frame in enumerate(frames, start=1)
    ]
    event_entries = [
        _realtime_item_entry(event, index=index, source="event")
        for index, event in enumerate(events, start=1)
    ]
    items = [*frame_entries, *event_entries]
    frame_types = sorted(
        {
            str(item.get("item_type") or "")
            for item in frame_entries
            if item.get("item_type")
        }
    )
    event_types = sorted(
        {
            str(item.get("item_type") or "")
            for item in event_entries
            if item.get("item_type")
        }
    )
    categories = sorted(
        {
            str(item.get("category") or "")
            for item in items
            if item.get("category")
        }
    )
    directions = sorted(
        {
            str(item.get("direction") or "")
            for item in items
            if item.get("direction")
        }
    )
    modalities = sorted(
        {
            str(item.get("modality") or "")
            for item in items
            if item.get("modality")
        }
    )
    tool_names = sorted(
        {
            str(tool.get("name") or "")
            for tool in _realtime_tool_calls(raw)
            if tool.get("name")
        }
    )
    transcripts = [
        _realtime_compact_transcript(item)
        for item in items
        if _realtime_compact_transcript(item)
    ]
    signals = sorted(
        {
            signal
            for item in items
            for signal in _realtime_item_signals(item)
        }
    )
    kind_counts: Dict[str, int] = {}
    for item in items:
        kind = str(item.get("kind") or "event")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    return {
        "kind": "framework_realtime_trace",
        "frame_count": len(frame_entries),
        "event_count": len(event_entries),
        "tool_call_count": len(_realtime_tool_calls(raw)),
        "tool_response_count": len(_realtime_tool_responses(raw)),
        "transcript_count": kind_counts.get("transcript", 0),
        "audio_frame_count": kind_counts.get("audio", 0),
        "lifecycle_event_count": kind_counts.get("lifecycle", 0),
        "interruption_count": kind_counts.get("interruption", 0),
        "error_count": kind_counts.get("error", 0),
        "completion_count": kind_counts.get("completion", 0),
        "signals": signals,
        "frame_types": frame_types,
        "event_types": event_types,
        "categories": categories,
        "directions": directions,
        "modalities": modalities,
        "tool_names": tool_names,
        "transcripts": transcripts,
        "frames": frame_entries,
        "events": event_entries,
        "summary": {
            "frame_count": len(frame_entries),
            "event_count": len(event_entries),
            "tool_call_count": len(_realtime_tool_calls(raw)),
            "tool_response_count": len(_realtime_tool_responses(raw)),
            "transcript_count": kind_counts.get("transcript", 0),
            "audio_frame_count": kind_counts.get("audio", 0),
            "lifecycle_event_count": kind_counts.get("lifecycle", 0),
            "completion_count": kind_counts.get("completion", 0),
            "error_count": kind_counts.get("error", 0),
        },
    }


def _realtime_trace_events(raw: Any) -> List[SimulationEvent]:
    frames = _realtime_frames(raw)
    events = _realtime_session_events(raw)
    normalized: List[SimulationEvent] = []
    for index, frame in enumerate(frames, start=1):
        entry = _realtime_item_entry(frame, index=index, source="frame")
        normalized.append(
            SimulationEvent(
                type="realtime_frame",
                name=str(entry.get("name") or entry.get("item_type") or f"frame_{index}"),
                payload=entry,
                timestamp_ms=_realtime_timestamp_ms(frame),
                metadata={
                    "kind": "realtime_trace",
                    "source": "frame",
                    "category": str(entry.get("category") or ""),
                },
            )
        )
        specialized = _realtime_specialized_event_type(entry)
        if specialized != "realtime_frame":
            normalized.append(
                SimulationEvent(
                    type=specialized,
                    name=str(entry.get("name") or entry.get("item_type") or specialized),
                    payload=entry,
                    timestamp_ms=_realtime_timestamp_ms(frame),
                    metadata={
                        "kind": "realtime_trace",
                        "source": "frame",
                        "category": str(entry.get("category") or ""),
                    },
                )
            )
    for index, event in enumerate(events, start=1):
        entry = _realtime_item_entry(event, index=index, source="event")
        event_type = _realtime_specialized_event_type(entry)
        normalized.append(
            SimulationEvent(
                type=event_type,
                name=str(entry.get("name") or entry.get("item_type") or f"event_{index}"),
                payload=entry,
                timestamp_ms=_realtime_timestamp_ms(event),
                metadata={
                    "kind": "realtime_trace",
                    "source": "event",
                    "category": str(entry.get("category") or ""),
                },
            )
        )
    return normalized


def _realtime_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for index, item in enumerate([*_realtime_frames(raw), *_realtime_session_events(raw)], start=1):
        item_type = _realtime_item_type(item).lower()
        if not _realtime_is_tool_call(item, item_type):
            continue
        name = _realtime_tool_name(item) or f"realtime_tool_{index}"
        calls.append(
            {
                "id": str(
                    item.get("id")
                    or item.get("call_id")
                    or item.get("tool_call_id")
                    or name
                ),
                "type": "function",
                "name": name,
                "arguments": _plain_value(
                    item.get("arguments")
                    if "arguments" in item
                    else item.get("args", item.get("input", item.get("payload", {})))
                ),
                "function": {
                    "name": name,
                    "arguments": _plain_value(
                        item.get("arguments")
                        if "arguments" in item
                        else item.get("args", item.get("input", item.get("payload", {})))
                    ),
                },
            }
        )
    return calls


def _realtime_tool_responses(raw: Any) -> List[Dict[str, Any]]:
    responses: List[Dict[str, Any]] = []
    for index, item in enumerate([*_realtime_frames(raw), *_realtime_session_events(raw)], start=1):
        item_type = _realtime_item_type(item).lower()
        if not _realtime_is_tool_response(item, item_type):
            continue
        name = _realtime_tool_name(item) or f"realtime_tool_{index}"
        content = item.get("result", item.get("output", item.get("response", item.get("content", ""))))
        responses.append(
            {
                "id": str(
                    item.get("id")
                    or item.get("call_id")
                    or item.get("tool_call_id")
                    or name
                ),
                "name": name,
                "content": _plain_value(content),
                "is_error": bool(item.get("is_error") or item.get("error")),
            }
        )
    return responses


def _realtime_last_text(raw: Any) -> str:
    entries = [
        *[
            _realtime_item_entry(frame, index=index, source="frame")
            for index, frame in enumerate(_realtime_frames(raw), start=1)
        ],
        *[
            _realtime_item_entry(event, index=index, source="event")
            for index, event in enumerate(_realtime_session_events(raw), start=1)
        ],
    ]
    for entry in reversed(entries):
        text = str(entry.get("text") or "")
        if text:
            return text
    return ""


def _realtime_frames(raw: Any) -> List[Dict[str, Any]]:
    frames = _extract_list_field(
        raw,
        (
            "frames",
            "frame_trace",
            "pipeline_frames",
            "pipecat_frames",
            "media_frames",
        ),
    )
    return [dict(frame) for frame in frames or []]


def _realtime_session_events(raw: Any) -> List[Dict[str, Any]]:
    candidates = _extract_list_field(
        raw,
        (
            "session_events",
            "sessionEvents",
            "livekit_events",
            "realtime_events",
            "events",
            "trajectory",
            "spans",
        ),
    )
    return [
        dict(event)
        for event in candidates or []
        if _is_realtime_item(event)
    ]


def _is_realtime_item(item: Mapping[str, Any]) -> bool:
    keys = set(item)
    if keys & {
        "frame_type",
        "frameType",
        "direction",
        "sample_rate",
        "sample_rate_hz",
        "audio",
        "transcript",
        "utterance",
        "agent_state",
        "user_state",
        "from_state",
        "to_state",
        "tool_name",
        "function_name",
        "speech_id",
        "interrupted",
    }:
        return True
    text = " ".join(
        str(item.get(key) or "")
        for key in ("type", "event", "name", "kind", "category", "source")
    ).lower()
    return any(
        token in text
        for token in (
            "audio",
            "speech",
            "tts",
            "stt",
            "vad",
            "transcript",
            "utterance",
            "session",
            "participant",
            "agent_state",
            "user_state",
            "tool_execution",
            "function_call",
            "interruption",
            "turn_start",
            "turn_end",
        )
    )


def _realtime_item_entry(
    item: Mapping[str, Any],
    *,
    index: int,
    source: str,
) -> Dict[str, Any]:
    item_type = _realtime_item_type(item)
    text = _realtime_item_text(item)
    entry: Dict[str, Any] = {
        "index": index,
        "source": source,
        "item_type": item_type,
        "name": _realtime_item_name(item, item_type=item_type),
        "kind": _realtime_item_kind(item, item_type=item_type),
        "category": _realtime_item_category(item, item_type=item_type),
        "direction": str(item.get("direction") or item.get("frame_direction") or ""),
        "modality": str(item.get("modality") or _realtime_item_modality(item, item_type)),
        "payload": _plain_value(dict(item)),
    }
    timestamp = _realtime_timestamp_ms(item)
    if timestamp is not None:
        entry["timestamp_ms"] = timestamp
    if text:
        entry["text"] = text
        entry["text_length"] = len(text)
    for key in (
        "participant",
        "participant_id",
        "agent",
        "speaker",
        "role",
        "from_state",
        "to_state",
        "state",
        "sample_rate",
        "sample_rate_hz",
        "duration_ms",
    ):
        value = item.get(key)
        if value not in (None, "", [], {}):
            entry[key] = _plain_value(value)
    tool_name = _realtime_tool_name(item)
    if tool_name:
        entry["tool_name"] = tool_name
    return entry


def _realtime_item_type(item: Mapping[str, Any]) -> str:
    for key in ("frame_type", "frameType", "type", "event", "kind", "name"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return "realtime_item"


def _realtime_item_name(item: Mapping[str, Any], *, item_type: str) -> str:
    for key in ("name", "event", "id", "tool_name", "function_name"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return item_type


def _realtime_item_kind(item: Mapping[str, Any], *, item_type: str) -> str:
    normalized = _realtime_key(
        " ".join(
            str(item.get(key) or "")
            for key in ("type", "event", "kind", "name", "frame_type", "frameType")
        )
        or item_type
    )
    if "error" in normalized or item.get("error"):
        return "error"
    if "interrupt" in normalized or item.get("interrupted"):
        return "interruption"
    if _realtime_is_tool_response(item, normalized):
        return "tool_response"
    if _realtime_is_tool_call(item, normalized):
        return "tool_call"
    if "transcript" in normalized or "utterance" in normalized or item.get("transcript"):
        return "transcript"
    if (
        "audio" in normalized
        or "tts" in normalized
        or "stt" in normalized
        or "vad" in normalized
        or item.get("audio")
        or item.get("sample_rate")
        or item.get("sample_rate_hz")
    ):
        return "audio"
    if (
        "complete" in normalized
        or "completed" in normalized
        or "final" in normalized
        or "closed" in normalized
        or "end" in normalized
    ):
        return "completion"
    if (
        "session" in normalized
        or "state" in normalized
        or "participant" in normalized
        or "start" in normalized
        or "connect" in normalized
        or item.get("from_state")
        or item.get("to_state")
    ):
        return "lifecycle"
    return "frame" if "frame" in normalized else "event"


def _realtime_is_tool_call(item: Mapping[str, Any], item_type: str) -> bool:
    normalized = _realtime_key(item_type)
    return bool(
        item.get("tool_name")
        or item.get("function_name")
        or item.get("function")
        or "functioncall" in normalized
        or "toolcall" in normalized
        or "toolexecutionstarted" in normalized
        or "toolexecutionrequested" in normalized
    ) and not _realtime_is_tool_response(item, item_type)


def _realtime_is_tool_response(item: Mapping[str, Any], item_type: str) -> bool:
    normalized = _realtime_key(item_type)
    return bool(
        item.get("result") not in (None, "", [], {})
        or item.get("tool_result") not in (None, "", [], {})
        or "functioncallresult" in normalized
        or "toolresult" in normalized
        or "toolexecutioncompleted" in normalized
        or "toolexecutionfailed" in normalized
    )


def _realtime_tool_name(item: Mapping[str, Any]) -> str:
    function = _object_mapping(item.get("function")) or {}
    tool_call = _object_mapping(item.get("tool_call")) or {}
    for value in (
        item.get("tool_name"),
        item.get("function_name"),
        item.get("tool"),
        function.get("name"),
        tool_call.get("name"),
        item.get("name"),
    ):
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def _realtime_item_category(item: Mapping[str, Any], *, item_type: str) -> str:
    for key in ("category", "frame_category", "frameCategory"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    normalized = _realtime_key(item_type)
    if "systemframe" in normalized:
        return "system"
    if "controlframe" in normalized:
        return "control"
    if "dataframe" in normalized or "audio" in normalized or "transcript" in normalized:
        return "data"
    if "frame" in normalized:
        return "frame"
    return "event"


def _realtime_item_modality(item: Mapping[str, Any], item_type: str) -> str:
    normalized = _realtime_key(item_type)
    if (
        item.get("audio")
        or item.get("sample_rate")
        or item.get("sample_rate_hz")
        or "audio" in normalized
        or "speech" in normalized
        or "tts" in normalized
        or "stt" in normalized
        or "vad" in normalized
    ):
        return "voice"
    if "video" in normalized:
        return "video"
    return ""


def _realtime_item_text(item: Mapping[str, Any]) -> str:
    for key in ("transcript", "text", "content", "utterance", "delta"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            return _stringify(value)
    payload = _object_mapping(item.get("payload"))
    if payload:
        for key in ("transcript", "text", "content", "utterance", "delta"):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                return _stringify(value)
    return ""


def _realtime_timestamp_ms(item: Mapping[str, Any]) -> Optional[int]:
    for key in ("timestamp_ms", "time_ms", "start_ms", "elapsed_ms"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    value = item.get("timestamp")
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _realtime_specialized_event_type(entry: Mapping[str, Any]) -> str:
    kind = str(entry.get("kind") or "")
    return {
        "audio": "realtime_audio_frame",
        "completion": "realtime_completion",
        "error": "realtime_error",
        "interruption": "realtime_interruption",
        "lifecycle": "realtime_lifecycle",
        "tool_call": "realtime_tool_call",
        "tool_response": "realtime_tool_response",
        "transcript": "realtime_transcript",
    }.get(kind, "realtime_frame")


def _realtime_item_signals(entry: Mapping[str, Any]) -> set[str]:
    signals = {"realtime"}
    source = str(entry.get("source") or "")
    kind = str(entry.get("kind") or "")
    category = str(entry.get("category") or "")
    if source:
        signals.add(source)
    if kind:
        signals.add(kind)
    if category:
        signals.add(f"{category}_frame" if category != "event" else "event")
    if entry.get("direction"):
        signals.add("direction")
    if entry.get("tool_name"):
        signals.add("tool")
    if entry.get("modality"):
        signals.add(str(entry["modality"]))
    return signals


def _realtime_compact_transcript(entry: Mapping[str, Any]) -> Dict[str, Any]:
    if entry.get("kind") != "transcript" or not entry.get("text"):
        return {}
    return {
        "index": entry.get("index"),
        "source": entry.get("source"),
        "role": entry.get("role"),
        "speaker": entry.get("speaker") or entry.get("participant"),
        "text": entry.get("text"),
    }


def _realtime_key(value: Any) -> str:
    return str(value or "").lower().replace("_", "").replace("-", "").replace(" ", "")


def _message_history_state(raw: Any) -> Dict[str, Any]:
    messages = _message_history(raw)
    if not messages:
        return {}
    tool_calls = [
        call
        for message in messages
        for call in _tool_calls_from_message(message)
    ]
    tool_responses = _message_history_tool_responses(raw)
    roles = sorted(
        {
            str(message.get("role"))
            for message in messages
            if message.get("role") not in (None, "", [], {})
        }
    )
    sources = sorted(
        {
            str(message.get("source") or message.get("speaker") or message.get("name"))
            for message in messages
            if message.get("source") or message.get("speaker") or message.get("name")
        }
    )
    types = sorted(
        {
            str(message.get("type") or message.get("kind") or message.get("role") or "")
            for message in messages
            if message.get("type") or message.get("kind") or message.get("role")
        }
    )
    stop_reason = _message_history_stop_reason(raw)
    state: Dict[str, Any] = {
        "message_count": len(messages),
        "roles": roles,
        "sources": sources,
        "types": types,
        "tool_call_count": len(tool_calls),
        "tool_response_count": len(tool_responses),
        "tool_names": sorted(
            {
                str(
                    call.get("name")
                    or call.get("tool")
                    or dict(call.get("function") or {}).get("name")
                    or ""
                )
                for call in tool_calls
                if isinstance(call, Mapping)
            }
        ),
        "last_content": _message_content(messages[-1]),
        "messages": [
            {
                "index": index,
                "type": str(message.get("type") or message.get("kind") or ""),
                "role": str(message.get("role") or ""),
                "source": str(message.get("source") or message.get("speaker") or message.get("name") or ""),
                "content_length": len(_message_content(message)),
                "tool_call_count": len(_tool_calls_from_message(message)),
            }
            for index, message in enumerate(messages, start=1)
        ],
    }
    if stop_reason:
        state["stop_reason"] = stop_reason
    handoffs = [
        {
            "from": message.get("handoff_from"),
            "to": message.get("handoff_to") or message.get("recipient"),
            "task": message.get("task"),
        }
        for message in messages
        if message.get("handoff_to") or message.get("recipient")
    ]
    if handoffs:
        state["handoff_count"] = len(handoffs)
        state["handoffs"] = handoffs
    return state


def _message_history(raw: Any) -> List[Dict[str, Any]]:
    value = None
    raw_mapping = _object_mapping(raw)
    for name in ("messages", "history", "chat_history", "conversation"):
        if raw_mapping is not None and raw_mapping.get(name) is not None:
            value = raw_mapping.get(name)
            break
        if raw_mapping is None and hasattr(raw, name):
            value = getattr(raw, name)
            break
    if not isinstance(value, (list, tuple)):
        return []
    messages: List[Dict[str, Any]] = []
    for item in value:
        mapping = _message_mapping(item)
        if mapping:
            messages.append(mapping)
    return messages


def _message_mapping(message: Any) -> Dict[str, Any]:
    mapping = _object_mapping(message)
    if mapping is not None:
        return mapping
    values: Dict[str, Any] = {}
    for attr in (
        "id",
        "type",
        "kind",
        "role",
        "source",
        "speaker",
        "name",
        "content",
        "tool_calls",
        "tool_responses",
        "metadata",
        "models_usage",
        "handoff_from",
        "handoff_to",
        "recipient",
        "task",
        "call_id",
        "tool_call_id",
        "result",
        "output",
        "is_error",
    ):
        if not hasattr(message, attr):
            continue
        value = getattr(message, attr)
        if value not in (None, "", [], {}):
            values[attr] = _plain_value(value)
    if values and "type" not in values:
        values["type"] = type(message).__name__
    return values


def _message_history_stop_reason(raw: Any) -> str:
    raw_mapping = _object_mapping(raw)
    if raw_mapping is not None:
        for key in ("stop_reason", "finish_reason", "termination", "termination_reason"):
            value = raw_mapping.get(key)
            if value not in (None, "", [], {}):
                return str(value)
    for attr in ("stop_reason", "finish_reason", "termination", "termination_reason"):
        if hasattr(raw, attr):
            value = getattr(raw, attr)
            if value not in (None, "", [], {}):
                return str(value)
    return ""


def _resolve_callable_attr_path(root: Any, path: str | None) -> Callable[..., Any] | None:
    if not path:
        return root if callable(root) else None
    value = root
    for raw_part in str(path).split("."):
        part = raw_part.strip()
        if not part:
            return None
        try:
            value = getattr(value, part)
        except Exception:
            return None
    return value if callable(value) else None


def _method_leaf(method_name: str | None) -> str:
    return str(method_name or "").rsplit(".", 1)[-1]


def _invoke_method_with_payload(
    method: Callable[..., Any],
    payload: Any,
    *,
    method_name: str | None,
    input_key: str | None,
    input_kwargs: Mapping[str, Any] | None,
) -> tuple[Any, str, str | None]:
    static_kwargs = {str(key): value for key, value in dict(input_kwargs or {}).items()}
    if payload is _NO_PAYLOAD:
        if static_kwargs:
            return method(**static_kwargs), "keyword", None
        return method(), "none", None

    if input_key:
        selected_key = str(input_key)
        return method(**{**static_kwargs, selected_key: payload}), "keyword", selected_key

    selected_key = _signature_input_key(method, method_name=method_name)
    if selected_key:
        return method(**{**static_kwargs, selected_key: payload}), "keyword", selected_key

    if _signature_accepts_positional(method):
        if static_kwargs:
            return method(payload, **static_kwargs), "positional_with_kwargs", None
        return method(payload), "positional", None

    if _signature_accepts_var_keyword(method) and isinstance(payload, Mapping):
        return method(**{**dict(payload), **static_kwargs}), "expanded_kwargs", None

    if static_kwargs:
        return method(payload, **static_kwargs), "positional_with_kwargs", None
    return method(payload), "positional", None


def _signature_input_key(
    method: Callable[..., Any],
    *,
    method_name: str | None,
) -> str | None:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return None
    params = list(signature.parameters.values())
    names = {param.name: param for param in params}
    method_preferences = (
        _METHOD_INPUT_KEY_PREFERENCES.get(str(method_name or ""))
        or _METHOD_INPUT_KEY_PREFERENCES.get(_method_leaf(method_name), ())
    )
    preferred_names = method_preferences + _KEYWORD_INPUT_NAMES
    accepts_positional = _params_accept_positional(params)

    for name in preferred_names:
        param = names.get(name)
        if param is None:
            continue
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            continue
        if name == "inputs" or not accepts_positional or _keyword_only(param):
            return name
    if not accepts_positional:
        for param in params:
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                return param.name
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
            first_preference = next(iter(preferred_names), None)
            return first_preference
    return None


def _signature_accepts_positional(method: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return True
    return _params_accept_positional(list(signature.parameters.values()))


def _params_accept_positional(params: List[inspect.Parameter]) -> bool:
    return any(
        param.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        }
        for param in params
    )


def _signature_accepts_var_keyword(method: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _keyword_only(param: inspect.Parameter) -> bool:
    return param.kind == inspect.Parameter.KEYWORD_ONLY


def _is_async_stream(value: Any) -> bool:
    if isinstance(value, (AgentResponse, str, bytes, dict, list, tuple)):
        return False
    return inspect.isasyncgen(value) or hasattr(value, "__anext__") or hasattr(value, "__aiter__")


def _is_sync_stream(value: Any) -> bool:
    if isinstance(value, (AgentResponse, str, bytes, dict, list, tuple)):
        return False
    return inspect.isgenerator(value) or hasattr(value, "__next__")


def _stream_chunk_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    if isinstance(chunk, dict):
        for key in (
            "content",
            "delta",
            "text",
            "transcript",
            "output",
            "response",
            "final_output",
        ):
            value = chunk.get(key)
            if value is not None:
                return _stringify(value)
        for key in ("message", "chunk"):
            if key in chunk:
                return _message_content(chunk[key])
        if "choices" in chunk:
            return _choices_content(chunk["choices"])
        for key in ("data", "payload"):
            value = chunk.get(key)
            if isinstance(value, dict):
                text = _stream_chunk_text(value)
                if text:
                    return text
    for attr in ("content", "delta", "text", "transcript", "output", "response"):
        if hasattr(chunk, attr):
            value = getattr(chunk, attr)
            if value is not None:
                return _stringify(value)
    if hasattr(chunk, "message"):
        return _message_content(getattr(chunk, "message"))
    if hasattr(chunk, "choices"):
        return _choices_content(getattr(chunk, "choices"))
    return ""


def _stream_chunk_event(chunk: Any, *, index: int, text: str) -> SimulationEvent:
    payload = _stream_chunk_payload(chunk)
    if text:
        payload.setdefault("delta", text)
    return SimulationEvent(
        type=_stream_chunk_event_type(chunk),
        name=_stream_chunk_event_name(chunk, index=index),
        payload=payload,
        timestamp_ms=_stream_chunk_timestamp_ms(chunk),
        metadata={"stream_index": index},
    )


def _stream_chunk_event_type(chunk: Any) -> str:
    value = _stream_chunk_field(chunk, ("type", "event", "frame_type", "method"))
    if value:
        return str(value)
    return "stream_chunk"


def _stream_chunk_event_name(chunk: Any, *, index: int) -> str:
    value = _stream_chunk_field(chunk, ("name", "id", "event_id"))
    if value:
        return str(value)
    return f"stream_chunk_{index}"


def _stream_chunk_timestamp_ms(chunk: Any) -> Optional[int]:
    value = _stream_chunk_field(chunk, ("timestamp_ms", "time_ms"))
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _stream_chunk_payload(chunk: Any) -> Dict[str, Any]:
    chunk_mapping = _object_mapping(chunk)
    if chunk_mapping is not None:
        return dict(chunk_mapping)
    if isinstance(chunk, (str, bytes)):
        return {"delta": _stream_chunk_text(chunk)}
    payload: Dict[str, Any] = {}
    for key in ("id", "type", "event", "name", "content", "delta", "text", "transcript"):
        if hasattr(chunk, key):
            value = getattr(chunk, key)
            if value is not None:
                payload[key] = value
    return payload or {"value": str(chunk)}


def _stream_chunk_field(chunk: Any, names: Iterable[str]) -> Any:
    chunk_mapping = _object_mapping(chunk)
    if chunk_mapping is not None:
        for name in names:
            value = chunk_mapping.get(name)
            if value is not None:
                return value
        for key in ("data", "payload"):
            value = chunk_mapping.get(key)
            if isinstance(value, dict):
                nested = _stream_chunk_field(value, names)
                if nested is not None:
                    return nested
    for name in names:
        if hasattr(chunk, name):
            value = getattr(chunk, name)
            if value is not None:
                return value
    return None


def _streaming_trace_from_chunks(chunks: List[Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    from fi.simulate.environment import normalize_streaming_trace_events

    framework = str(metadata.get("framework") or "generic")
    trace_metadata = {
        "source": "generic_agent_wrapper",
        **dict(metadata),
    }
    return normalize_streaming_trace_events(
        framework,
        chunks,
        metadata=trace_metadata,
    )


def _framework_runtime_trace(
    *,
    framework: str,
    method: Callable[..., Any],
    method_name: str | None,
    input_mode: str,
    payload: Any,
    response: str | AgentResponse,
    duration_ms: int,
    streamed: bool,
    call_style: str,
    input_key: str | None,
    input_kwargs_keys: List[str],
    wrapper_metadata: Dict[str, Any],
    runtime_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    response_dict = _response_summary(response)
    input_shape = _shape_summary(payload)
    callable_signature = _framework_callable_signature(
        method,
        method_name=method_name,
        selected_input_key=input_key,
    )
    call_contract = {
        "kind": "agent-learning.framework-adapter-call-contract.v1",
        "method": method_name or "callable",
        "method_leaf": _method_leaf(method_name),
        "input_mode": "none" if payload is _NO_PAYLOAD else input_mode,
        "call_style": call_style,
        "input_key": input_key,
        "input_kwargs_keys": input_kwargs_keys,
        "signature": callable_signature,
        "observed_io": {
            "input": input_shape,
            "output": response_dict,
        },
    }
    signature_bound = _call_contract_signature_bound(call_contract)
    call_contract["signature_bound"] = signature_bound
    signals = {"framework", "runtime", "method", "input", "output", "latency"}
    if streamed or response_dict.get("streaming"):
        signals.add("streaming")
    if response_dict.get("tool_call_count", 0) > 0:
        signals.add("tool")
    if response_dict.get("artifact_count", 0) > 0:
        signals.add("artifact")
    if response_dict.get("event_count", 0) > 0:
        signals.add("event")
    if response_dict.get("state_keys"):
        signals.add("state")
    if "openenv" in response_dict.get("state_keys", []) or response_dict.get("openenv_summary"):
        signals.add("openenv")
    state_keys = set(response_dict.get("state_keys") or [])
    if "agent_control_plane" in state_keys:
        signals.add("control_plane")
    if "agent_trust_boundary_model" in state_keys:
        signals.add("trust_boundary")
    if response_dict.get("metadata_keys"):
        signals.add("metadata")

    invocation = {
        "id": "framework_runtime_1",
        "framework": framework or "generic",
        "method": method_name or "callable",
        "input_mode": "none" if payload is _NO_PAYLOAD else input_mode,
        "input": input_shape,
        "output": response_dict,
        "duration_ms": max(0, int(duration_ms)),
        "call_style": call_style,
        "signals": sorted(signals),
        "call_contract": call_contract,
    }
    if input_key:
        invocation["input_key"] = input_key
    if input_kwargs_keys:
        invocation["input_kwargs_keys"] = input_kwargs_keys
    summary = {
        "invocation_count": 1,
        "framework": framework or "generic",
        "methods": [invocation["method"]],
        "input_modes": [invocation["input_mode"]],
        "call_styles": [call_style],
        "input_keys": [input_key] if input_key else [],
        "input_kwargs_keys": input_kwargs_keys,
        "output_types": [response_dict["type"]],
        "call_contract_count": 1,
        "signature_inspectable": bool(callable_signature.get("inspectable")),
        "signature_bound": signature_bound,
        "tool_call_count": response_dict.get("tool_call_count", 0),
        "artifact_count": response_dict.get("artifact_count", 0),
        "event_count": response_dict.get("event_count", 0),
        "state_key_count": len(response_dict.get("state_keys", [])),
        "metadata_key_count": len(response_dict.get("metadata_keys", [])),
        "streamed": bool(streamed or response_dict.get("streaming")),
        "error_count": 0,
        "duration_ms": invocation["duration_ms"],
    }
    return {
        "kind": "framework_runtime",
        "framework": framework or "generic",
        "modality": wrapper_metadata.get("modality"),
        "invocations": [invocation],
        "summary": summary,
        "signals": sorted(signals),
        "metadata": {
            "source": "generic_agent_wrapper",
            **dict(wrapper_metadata),
            **dict(runtime_metadata),
        },
    }


def _framework_callable_signature(
    method: Callable[..., Any],
    *,
    method_name: str | None,
    selected_input_key: str | None,
) -> Dict[str, Any]:
    selected_method = method_name or getattr(method, "__name__", None) or "callable"
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return {
            "kind": "agent-learning.framework-adapter-callable-signature.v1",
            "inspectable": False,
            "method": selected_method,
            "method_leaf": _method_leaf(selected_method),
            "parameters": [],
            "parameter_names": [],
            "required_parameters": [],
            "selected_input_key": selected_input_key,
            "selection_source": "explicit" if selected_input_key else "unavailable",
        }

    params = list(signature.parameters.values())
    parameter_rows = [
        {
            "name": param.name,
            "kind": str(param.kind).rsplit(".", 1)[-1].lower(),
            "required": param.default is inspect.Parameter.empty
            and param.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            },
            "has_default": param.default is not inspect.Parameter.empty,
            "annotation": _signature_annotation_name(param.annotation),
        }
        for param in params
    ]
    inferred_key = _signature_input_key(method, method_name=method_name)
    return {
        "kind": "agent-learning.framework-adapter-callable-signature.v1",
        "inspectable": True,
        "method": selected_method,
        "method_leaf": _method_leaf(selected_method),
        "parameters": parameter_rows,
        "parameter_names": [row["name"] for row in parameter_rows],
        "required_parameters": [
            row["name"] for row in parameter_rows if row["required"]
        ],
        "required_parameter_count": sum(1 for row in parameter_rows if row["required"]),
        "accepts_positional": _params_accept_positional(params),
        "accepts_var_positional": any(
            param.kind == inspect.Parameter.VAR_POSITIONAL for param in params
        ),
        "accepts_var_keyword": any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params
        ),
        "keyword_only_parameters": [
            param.name
            for param in params
            if param.kind == inspect.Parameter.KEYWORD_ONLY
        ],
        "positional_parameters": [
            param.name
            for param in params
            if param.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ],
        "inferred_input_key": inferred_key,
        "selected_input_key": selected_input_key,
        "selection_source": (
            "signature_preference"
            if selected_input_key and selected_input_key == inferred_key
            else "explicit"
            if selected_input_key
            else "positional_or_none"
        ),
        "is_async": inspect.iscoroutinefunction(method),
        "is_generator": inspect.isgeneratorfunction(method),
        "is_async_generator": inspect.isasyncgenfunction(method),
        "return_annotation": _signature_annotation_name(signature.return_annotation),
    }


def _signature_annotation_name(annotation: Any) -> str | None:
    if annotation is inspect.Signature.empty or annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _call_contract_signature_bound(contract: Mapping[str, Any]) -> bool:
    signature = dict(contract.get("signature") or {})
    if not signature.get("inspectable"):
        return False
    call_style = str(contract.get("call_style") or "")
    input_key = contract.get("input_key")
    parameter_names = set(str(name) for name in list(signature.get("parameter_names") or []))
    required_parameters = set(
        str(name) for name in list(signature.get("required_parameters") or []) if str(name)
    )
    input_kwargs_keys = set(
        str(key) for key in list(contract.get("input_kwargs_keys") or []) if str(key)
    )
    accepts_var_keyword = bool(signature.get("accepts_var_keyword"))

    if call_style in {"keyword", "positional_with_kwargs"} and input_key:
        return str(input_key) in parameter_names or accepts_var_keyword
    if call_style == "expanded_kwargs":
        return accepts_var_keyword
    if call_style in {"positional", "positional_with_kwargs"}:
        return bool(signature.get("accepts_positional"))
    if call_style == "none":
        return required_parameters <= input_kwargs_keys
    return False


def _attach_framework_runtime_trace(
    response: str | AgentResponse,
    trace: Dict[str, Any],
) -> AgentResponse:
    artifact = SimulationArtifact(
        type="trace",
        role="assistant",
        data=trace,
        metadata={
            "kind": "framework_runtime",
            "framework": trace.get("framework", "generic"),
            "source": "generic_agent_wrapper",
        },
    )
    event = SimulationEvent(
        type="framework_runtime",
        name=str(trace["invocations"][0].get("method") or "callable"),
        payload=trace["invocations"][0],
        metadata={"kind": "framework_runtime", "framework": trace.get("framework", "generic")},
    )
    runtime_metadata = {
        "framework_runtime": {
            "framework": trace.get("framework", "generic"),
            "signals": list(trace.get("signals", [])),
            "summary": dict(trace.get("summary", {})),
        }
    }
    if not isinstance(response, AgentResponse):
        return AgentResponse(
            content=str(response),
            artifacts=[artifact],
            events=[event],
            state={"framework_runtime": trace},
            metadata=runtime_metadata,
        )

    state = dict(response.state or {})
    state["framework_runtime"] = trace
    metadata = {**dict(response.metadata or {}), **runtime_metadata}
    return AgentResponse(
        content=response.content,
        tool_calls=response.tool_calls,
        tool_responses=response.tool_responses,
        artifacts=[*response.artifacts, artifact],
        events=[*response.events, event],
        memory_updates=response.memory_updates,
        state=state,
        metadata=metadata,
    )


def _response_summary(response: str | AgentResponse) -> Dict[str, Any]:
    if not isinstance(response, AgentResponse):
        return {
            "type": type(response).__name__,
            "content_length": len(str(response)),
            "tool_call_count": 0,
            "artifact_count": 0,
            "event_count": 0,
            "state_keys": [],
            "metadata_keys": [],
            "streaming": False,
        }
    metadata = dict(response.metadata or {})
    state = dict(response.state or {})
    openenv_summary = _plain_mapping(
        _plain_mapping(state.get("openenv")).get("summary")
    )
    return {
        "type": "AgentResponse",
        "content_length": len(response.content or ""),
        "tool_call_count": len(response.tool_calls or []),
        "tool_names": sorted(
            {
                str(call.get("name") or call.get("tool") or call.get("function", {}).get("name") or "")
                for call in response.tool_calls or []
                if isinstance(call, dict)
            }
        ),
        "tool_response_count": len(response.tool_responses or []),
        "artifact_count": len(response.artifacts),
        "artifact_types": sorted({artifact.type for artifact in response.artifacts}),
        "event_count": len(response.events),
        "event_types": sorted({event.type for event in response.events}),
        "state_keys": sorted(str(key) for key in state.keys()),
        "openenv_summary": openenv_summary,
        "metadata_keys": sorted(str(key) for key in metadata.keys()),
        "streaming": bool(metadata.get("streaming") or state.get("streaming_trace")),
    }


def _shape_summary(value: Any) -> Dict[str, Any]:
    if value is _NO_PAYLOAD:
        return {"type": "none"}
    if isinstance(value, AgentInput):
        return {
            "type": "AgentInput",
            "message_count": len(value.messages),
            "tool_count": len(value.tools),
            "artifact_count": len(value.artifacts),
            "event_count": len(value.events),
            "modality": value.modality,
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(str(key) for key in value.keys()),
            "message_count": len(value.get("messages") or []),
            "tool_count": len(value.get("tools") or []),
            "artifact_count": len(value.get("artifacts") or []),
            "event_count": len(value.get("events") or []),
            "has_metadata": isinstance(value.get("metadata"), dict),
        }
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    if isinstance(value, tuple):
        return {"type": "tuple", "length": len(value)}
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return {"type": type(value).__name__, "length": len(text)}
    return {"type": type(value).__name__}


def _choices_content(choices: Any) -> str:
    if not choices:
        return ""
    first = choices[0]
    if isinstance(first, dict):
        return _message_content(first.get("message") or first.get("delta") or first)
    return _message_content(getattr(first, "message", None) or getattr(first, "delta", None) or first)


def _last_message_content(messages: Any) -> str:
    if not messages:
        return ""
    try:
        return _message_content(list(messages)[-1])
    except TypeError:
        return _message_content(messages)


def _message_content(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        if "content" in message and message["content"] is not None:
            block_text = _content_blocks_text(message["content"])
            if block_text:
                return block_text
            return _stringify(message["content"])
        if "text" in message and message["text"] is not None:
            return _stringify(message["text"])
        if "parts" in message:
            return " ".join(_stringify(part) for part in message["parts"])
    for attr in ("content", "text"):
        if hasattr(message, attr):
            value = getattr(message, attr)
            if value is not None:
                return _stringify(value)
    return str(message)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    plain = _plain_value(value)
    if isinstance(plain, (dict, list, tuple)):
        return json.dumps(plain, default=str)
    return str(value)


def _model_to_dict(value: Any) -> Dict[str, Any]:
    mapping = _object_mapping(value)
    if mapping is not None:
        return dict(mapping)
    return dict(value)


def _object_mapping(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_value(item)
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            str(key): _plain_value(item)
            for key, item in asdict(value).items()
        }
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            dumped = method()
        except TypeError:
            try:
                dumped = method(mode="json")
            except TypeError:
                continue
        if isinstance(dumped, Mapping):
            return {
                str(key): _plain_value(item)
                for key, item in dumped.items()
            }
    return None


def _plain_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {
            str(key): _plain_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item) for item in value]
    mapping = _object_mapping(value)
    if mapping is not None:
        return mapping
    return value


def _as_int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
