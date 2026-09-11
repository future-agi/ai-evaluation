from __future__ import annotations

import asyncio
import copy
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse

from fi.simulate.agent.generic import GenericAgentWrapper, InputMode
from fi.simulate.agent.wrapper import AgentInput, AgentResponse, AgentWrapper


@dataclass(frozen=True)
class FrameworkAdapterSpec:
    """Import-free adapter preset for a common agent/orchestration framework."""

    name: str
    method: Optional[str]
    input_mode: InputMode
    modality: str = "text"
    transport: str = "in_process"
    lifecycle_hooks: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    notes: str = ""


FRAMEWORK_PRESETS: Dict[str, FrameworkAdapterSpec] = {
    # Text/chat orchestration
    "custom": FrameworkAdapterSpec("custom", None, "auto", notes="Bring-your-own framework adapter."),
    "callable": FrameworkAdapterSpec("callable", None, "agent_input", notes="Plain Python callable."),
    "a2a": FrameworkAdapterSpec("a2a", "send_message", "dict", notes="Agent2Agent protocol client/server session."),
    "openenv": FrameworkAdapterSpec(
        "openenv",
        "run",
        "dict",
        capabilities=(
            "environment_replay",
            "reset_step_trace",
            "runtime_trace",
            "state",
            "artifacts",
        ),
        notes="OpenEnv/Gymnasium-style environment replay adapter.",
    ),
    "gymnasium": FrameworkAdapterSpec(
        "gymnasium",
        "run",
        "dict",
        capabilities=(
            "environment_replay",
            "reset_step_trace",
            "runtime_trace",
            "state",
            "artifacts",
        ),
        notes="Gymnasium Env reset/step replay adapter.",
    ),
    "langchain": FrameworkAdapterSpec("langchain", "ainvoke", "dict", notes="LangChain Runnable/Chain."),
    "langgraph": FrameworkAdapterSpec("langgraph", "ainvoke", "dict", notes="LangGraph compiled graph."),
    "llamaindex": FrameworkAdapterSpec("llamaindex", "achat", "text", notes="LlamaIndex chat/query engines."),
    "crewai": FrameworkAdapterSpec("crewai", "kickoff", "dict", notes="CrewAI Crew kickoff."),
    "autogen": FrameworkAdapterSpec("autogen", "run", "text", notes="AutoGen AgentChat style task run."),
    "semantic_kernel": FrameworkAdapterSpec("semantic_kernel", "invoke", "dict", notes="Semantic Kernel function/agent."),
    "openai_agents": FrameworkAdapterSpec("openai_agents", "run", "text", notes="OpenAI Agents SDK runner/agent."),
    "pydantic_ai": FrameworkAdapterSpec("pydantic_ai", "run", "text", notes="PydanticAI agent."),
    "haystack": FrameworkAdapterSpec("haystack", "run", "dict", notes="Haystack pipeline."),
    "agno": FrameworkAdapterSpec("agno", "run", "dict", notes="Agno agent/team runner."),
    "beeai": FrameworkAdapterSpec("beeai", "run", "dict", notes="BeeAI agent runner."),
    "claude_agent_sdk": FrameworkAdapterSpec("claude_agent_sdk", "query", "text", notes="Claude Agent SDK query runner."),
    "dspy": FrameworkAdapterSpec("dspy", "__call__", "dict", notes="DSPy module/program."),
    "google_adk": FrameworkAdapterSpec("google_adk", "run", "dict", notes="Google ADK runner/agent."),
    "guardrails": FrameworkAdapterSpec("guardrails", "__call__", "text", notes="Guardrails validation wrapper."),
    "litellm": FrameworkAdapterSpec("litellm", "completion", "dict", notes="LiteLLM completion shim."),
    "mcp": FrameworkAdapterSpec("mcp", "call_tool", "dict", notes="MCP client/server tool session."),
    "portkey": FrameworkAdapterSpec("portkey", "chat", "dict", notes="Portkey gateway client."),
    "smolagents": FrameworkAdapterSpec("smolagents", "run", "text", notes="SmolAgents runner."),
    "strands": FrameworkAdapterSpec("strands", "__call__", "text", notes="Strands agent callable."),
    # Voice and realtime
    "livekit": FrameworkAdapterSpec("livekit", "respond", "text", modality="voice", notes="LiveKit agent/session shim."),
    "pipecat": FrameworkAdapterSpec("pipecat", "process", "dict", modality="voice", notes="Pipecat pipeline/processor shim."),
    "vapi": FrameworkAdapterSpec("vapi", "respond", "dict", modality="voice", notes="Webhook/local adapter shim."),
    "retell": FrameworkAdapterSpec("retell", "respond", "dict", modality="voice", notes="Webhook/local adapter shim."),
    "elevenlabs": FrameworkAdapterSpec("elevenlabs", "respond", "dict", modality="voice", notes="ElevenLabs conversational agent shim."),
    "deepgram": FrameworkAdapterSpec("deepgram", "respond", "dict", modality="voice", notes="Deepgram voice agent shim."),
    "agora": FrameworkAdapterSpec("agora", "respond", "dict", modality="voice", notes="Agora conversational AI shim."),
    "twilio": FrameworkAdapterSpec("twilio", "respond", "dict", modality="voice", notes="Twilio voice/media stream webhook shim."),
    # Model/provider clients commonly instrumented by TraceAI
    "anthropic": FrameworkAdapterSpec("anthropic", "messages.create", "messages", notes="Anthropic messages client shim."),
    "bedrock": FrameworkAdapterSpec("bedrock", "invoke_model", "dict", notes="AWS Bedrock client shim."),
    "cerebras": FrameworkAdapterSpec("cerebras", "chat", "dict", notes="Cerebras client shim."),
    "cohere": FrameworkAdapterSpec("cohere", "chat", "dict", notes="Cohere client shim."),
    "deepseek": FrameworkAdapterSpec("deepseek", "chat", "dict", notes="DeepSeek OpenAI-compatible client shim."),
    "fireworks": FrameworkAdapterSpec("fireworks", "chat", "dict", notes="Fireworks client shim."),
    "google_genai": FrameworkAdapterSpec("google_genai", "generate_content", "dict", notes="Google GenAI client shim."),
    "groq": FrameworkAdapterSpec("groq", "chat", "dict", notes="Groq client shim."),
    "huggingface": FrameworkAdapterSpec("huggingface", "__call__", "dict", notes="Hugging Face pipeline/client shim."),
    "instructor": FrameworkAdapterSpec("instructor", "chat", "dict", notes="Instructor structured output client shim."),
    "mistralai": FrameworkAdapterSpec("mistralai", "chat", "dict", notes="Mistral AI client shim."),
    "ollama": FrameworkAdapterSpec("ollama", "chat", "dict", notes="Ollama client shim."),
    "openai": FrameworkAdapterSpec("openai", "chat.completions.create", "messages", notes="OpenAI chat client shim."),
    "together": FrameworkAdapterSpec("together", "chat", "dict", notes="Together AI client shim."),
    "vertexai": FrameworkAdapterSpec("vertexai", "generate_content", "dict", notes="Vertex AI client shim."),
    "vllm": FrameworkAdapterSpec("vllm", "generate", "dict", notes="vLLM server/client shim."),
    "xai": FrameworkAdapterSpec("xai", "chat", "dict", notes="xAI client shim."),
    # Computer-use / browser / multimodal
    "computer_use": FrameworkAdapterSpec("computer_use", "run", "dict", modality="cua", notes="Browser or desktop CUA runner."),
    "browser_use": FrameworkAdapterSpec("browser_use", "run", "dict", modality="cua", notes="Browser automation agent."),
    "playwright": FrameworkAdapterSpec("playwright", "run", "dict", modality="cua", notes="Playwright-backed agent harness."),
    "vision_agent": FrameworkAdapterSpec("vision_agent", "run", "dict", modality="image", notes="Image or multimodal agent."),
}


_DISCOVERY_METHOD_ORDER = (
    "ainvoke",
    "invoke",
    "astream",
    "stream",
    "stream_events",
    "execute_task",
    "process_frame",
    "send_message",
    "message_send",
    "call",
    "achat",
    "chat",
    "responses.create",
    "chat.completions.create",
    "messages.create",
    "kickoff",
    "query",
    "process",
    "respond",
    "run",
    "run_stream",
    "arun",
    "send",
    "completion",
    "call_tool",
    "invoke_model",
    "generate_content",
    "generate",
    "__call__",
)

_DISCOVERY_METHOD_INPUT_MODES: dict[str, InputMode] = {
    "ainvoke": "dict",
    "invoke": "dict",
    "astream": "dict",
    "stream": "dict",
    "stream_events": "dict",
    "execute_task": "dict",
    "process_frame": "dict",
    "send_message": "dict",
    "message_send": "dict",
    "responses.create": "text",
    "chat.completions.create": "messages",
    "messages.create": "messages",
    "kickoff": "dict",
    "process": "dict",
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
    "send": "text",
}

_STREAMING_METHODS = {"astream", "stream", "stream_events", "run_stream"}

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

_DISCOVERY_INPUT_MODE_ORDER: tuple[InputMode, ...] = (
    "dict",
    "text",
    "agent_input",
    "messages",
    "auto",
)


def supported_frameworks() -> list[str]:
    """Return built-in framework preset names.

    ``wrap_framework`` also accepts unknown framework names as custom adapters
    when the caller supplies method/input-mode overrides or the generic wrapper
    can infer a callable method.
    """

    return sorted(FRAMEWORK_PRESETS)


def framework_adapter_contract(
    framework: str,
    *,
    target: str | None = None,
    method: str | None = None,
    input_mode: InputMode | None = None,
    input_key: str | None = None,
    input_kwargs: Mapping[str, Any] | None = None,
    modality: str | None = None,
    trace_runtime: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return the native adapter contract used for framework simulation.

    The contract is import-free and local: it describes the framework shim,
    lifecycle, transport, capabilities, schemas, and replay requirements without
    pulling in LangGraph, LiveKit, Pipecat, or any other framework package.
    """

    meta = dict(metadata or {})
    key = _framework_key(framework)
    spec = FRAMEWORK_PRESETS.get(key)
    adapter_kind = "preset" if spec is not None else "custom"
    resolved_method = str(method or (spec.method if spec else "") or "auto")
    resolved_input_mode = str(input_mode or (spec.input_mode if spec else "") or "auto")
    resolved_modality = str(modality or meta.get("modality") or (spec.modality if spec else "text"))
    transport = str((spec.transport if spec else "") or _default_transport(resolved_modality))
    lifecycle_hooks = list(
        spec.lifecycle_hooks
        if spec and spec.lifecycle_hooks
        else _default_lifecycle_hooks(resolved_modality)
    )
    capabilities = list(
        spec.capabilities
        if spec and spec.capabilities
        else _default_capabilities(resolved_modality, resolved_input_mode)
    )
    target_scheme = urlparse(str(target or "")).scheme.lower()
    target_is_external = target_scheme in {"http", "https"}
    local_fixture = not target_is_external

    contract: dict[str, Any] = {
        "kind": "agent-learning.framework-adapter-contract.v1",
        "framework": key,
        "adapter": adapter_kind,
        "method": resolved_method,
        "input_mode": resolved_input_mode,
        "modality": resolved_modality,
        "transport": transport,
        "lifecycle_hooks": lifecycle_hooks,
        "capabilities": capabilities,
        "schemas": {
            "input": _input_schema(resolved_input_mode),
            "output": _output_schema(),
        },
        "trace_runtime": bool(trace_runtime),
        "requires_external_service": False,
        "local_executable_fixture": local_fixture,
        "evidence_requirements": [
            "framework_runtime",
            "framework_trace",
            "tool_calls",
            "adapter_conformance",
            "metric_evidence",
        ],
    }
    if input_key:
        contract["input_key"] = str(input_key)
    input_kwargs_keys = sorted(str(key) for key in dict(input_kwargs or {}))
    if input_kwargs_keys:
        contract["input_kwargs_keys"] = input_kwargs_keys
    if target:
        contract["target"] = str(target)
        contract["target_scheme"] = target_scheme
    if spec and spec.notes:
        contract["notes"] = spec.notes
    if key in {"openenv", "gymnasium", "gymnasium_env", "environment_replay"}:
        contract["evidence_requirements"] = [
            *contract["evidence_requirements"],
            "openenv",
        ]
    return contract


def framework_adapter_contract_matrix(
    frameworks: Sequence[str] | str | None = None,
    *,
    targets: Mapping[str, str] | None = None,
    methods: Mapping[str, str] | None = None,
    input_modes: Mapping[str, InputMode] | None = None,
    modalities: Mapping[str, str] | None = None,
    trace_runtime: bool = True,
    allow_external_targets: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a native, import-free adapter contract matrix.

    The matrix is a first-party certification artifact: it proves what the
    Agent Learning simulator can run through local adapter fixtures without
    importing or calling LangGraph, LiveKit, Pipecat, or other frameworks.
    HTTP/HTTPS targets are rejected by default so this path stays native unless
    a caller explicitly opts into external target documentation.
    """

    framework_keys = _framework_matrix_keys(frameworks)
    target_map = {
        _framework_key(key): str(value)
        for key, value in dict(targets or {}).items()
    }
    method_map = {
        _framework_key(key): str(value)
        for key, value in dict(methods or {}).items()
    }
    input_mode_map = {
        _framework_key(key): value for key, value in dict(input_modes or {}).items()
    }
    modality_map = {
        _framework_key(key): str(value)
        for key, value in dict(modalities or {}).items()
    }
    external_targets = {
        key: target
        for key, target in target_map.items()
        if _is_external_target(target)
    }
    if external_targets and not allow_external_targets:
        blocked = ", ".join(
            f"{key}={target}" for key, target in sorted(external_targets.items())
        )
        raise ValueError(
            "external targets are disabled for native framework adapter matrices: "
            f"{blocked}"
        )

    contracts = [
        framework_adapter_contract(
            key,
            target=target_map.get(key) or _local_fixture_target(key),
            method=method_map.get(key),
            input_mode=input_mode_map.get(key),
            modality=modality_map.get(key),
            trace_runtime=trace_runtime,
            metadata=copy_metadata,
        )
        for key in framework_keys
        for copy_metadata in [dict(metadata or {})]
    ]
    profiles = [
        _framework_adapter_capability_profile_from_contract(contract)
        for contract in contracts
    ]
    findings = _framework_matrix_findings(contracts)
    summary = _framework_matrix_summary(contracts)
    return {
        "kind": "agent-learning.framework-adapter-contract-matrix.v1",
        "status": "passed" if not findings else "failed",
        "requires_external_service": False,
        "runtime": "in_process",
        "allow_external_targets": bool(allow_external_targets),
        "framework_count": len(framework_keys),
        "frameworks": framework_keys,
        "contracts": contracts,
        "profiles": profiles,
        "summary": summary,
        "profile_summary": _framework_profile_collection_summary(profiles),
        "findings": findings,
        "contract_quality_gate": {
            "kind": "agent-learning.framework-adapter-contract.v1",
            "required_frameworks": framework_keys,
            "require_trace_runtime": bool(trace_runtime),
            "require_local_executable_fixture": not bool(allow_external_targets),
            "require_no_external_service": True,
            "require_target": True,
            "forbidden_target_schemes": (
                [] if allow_external_targets else ["http", "https"]
            ),
            "required_schema_sections": ["input", "output"],
            "required_lifecycle_hooks": ["setup", "teardown"],
            "required_capabilities": ["messages", "tool_calls", "runtime_trace"],
            "required_evidence_requirements": [
                "framework_runtime",
                "framework_trace",
                "tool_calls",
                "adapter_conformance",
                "metric_evidence",
            ],
        },
        "evidence_requirements": [
            "framework_runtime",
            "framework_trace",
            "adapter_conformance",
            "metric_evidence",
            "matrix_coverage",
        ],
    }


def framework_adapter_capability_profile(
    framework: str,
    *,
    target: str | None = None,
    method: str | None = None,
    input_mode: InputMode | None = None,
    input_key: str | None = None,
    input_kwargs: Mapping[str, Any] | None = None,
    modality: str | None = None,
    trace_runtime: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a portable trinity profile for one framework adapter.

    Profiles are derived from the native adapter contract and intentionally do
    not import LangChain, LiveKit, Pipecat, or any other framework. The payload
    is the shared handshake between simulate-sdk, ai-evaluation, and agent-opt:
    it tells each library which local contract, metric, and optimization layer
    represent the framework surface.
    """

    selected_contract = (
        copy.deepcopy(dict(contract))
        if contract is not None
        else framework_adapter_contract(
            framework,
            target=target,
            method=method,
            input_mode=input_mode,
            input_key=input_key,
            input_kwargs=input_kwargs,
            modality=modality,
            trace_runtime=trace_runtime,
            metadata=metadata,
        )
    )
    return _framework_adapter_capability_profile_from_contract(selected_contract)


def framework_adapter_capability_profiles(
    frameworks: Sequence[str] | str | None = None,
    *,
    matrix: Mapping[str, Any] | None = None,
    targets: Mapping[str, str] | None = None,
    methods: Mapping[str, str] | None = None,
    input_modes: Mapping[str, InputMode] | None = None,
    modalities: Mapping[str, str] | None = None,
    trace_runtime: bool = True,
    allow_external_targets: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return framework capability profiles for a matrix or framework list."""

    matrix_payload = (
        copy.deepcopy(dict(matrix))
        if matrix is not None
        else framework_adapter_contract_matrix(
            frameworks,
            targets=targets,
            methods=methods,
            input_modes=input_modes,
            modalities=modalities,
            trace_runtime=trace_runtime,
            allow_external_targets=allow_external_targets,
            metadata=metadata,
        )
    )
    contracts = [
        copy.deepcopy(dict(contract))
        for contract in matrix_payload.get("contracts", []) or []
        if isinstance(contract, Mapping)
    ]
    if not contracts:
        framework_keys = _framework_matrix_keys(
            matrix_payload.get("frameworks") or frameworks
        )
        contracts = [
            framework_adapter_contract(
                key,
                target=(dict(targets or {}).get(key) if targets else None),
                method=(dict(methods or {}).get(key) if methods else None),
                input_mode=(dict(input_modes or {}).get(key) if input_modes else None),
                modality=(dict(modalities or {}).get(key) if modalities else None),
                trace_runtime=trace_runtime,
                metadata=metadata,
            )
            for key in framework_keys
        ]

    profiles = [
        _framework_adapter_capability_profile_from_contract(contract)
        for contract in contracts
    ]
    required_frameworks = _framework_matrix_keys(
        matrix_payload.get("frameworks") or frameworks
    )
    findings = _framework_profile_collection_findings(
        profiles,
        required_frameworks=required_frameworks,
    )
    return {
        "kind": "agent-learning.framework-adapter-capability-profiles.v1",
        "status": "passed" if not findings else "failed",
        "passed": not findings,
        "requires_external_service": False,
        "framework_count": len(required_frameworks),
        "profile_count": len(profiles),
        "frameworks": required_frameworks,
        "profiles": profiles,
        "summary": _framework_profile_collection_summary(profiles),
        "findings": findings,
        "source_matrix_kind": matrix_payload.get("kind"),
        "evidence_requirements": [
            "framework_adapter_contract",
            "framework_adapter_profile",
            "framework_runtime",
            "framework_trace",
            "metric_evidence",
            "optimization_lineage",
        ],
    }


def wrap_framework(
    framework: str,
    agent: Any,
    *,
    target: str | None = None,
    method: str | None = None,
    input_mode: InputMode | None = None,
    input_key: str | None = None,
    input_kwargs: Mapping[str, Any] | None = None,
    system_prompt: str | None = None,
    output_key: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_runtime: bool = False,
    runtime_metadata: Optional[Dict[str, Any]] = None,
) -> AgentWrapper:
    """
    Wrap a known or custom framework by name without importing that framework.

    Presets are intentionally thin. They encode the most common method/payload
    shape while leaving escape hatches for custom method, input_mode, and
    output_key.
    """

    key = framework.lower().replace("-", "_")
    spec = FRAMEWORK_PRESETS.get(key)
    raw_metadata = dict(metadata or {})
    contract = raw_metadata.get("framework_adapter_contract")
    if not isinstance(contract, dict):
        contract = framework_adapter_contract(
            key,
            target=target,
            method=method,
            input_mode=input_mode,
            input_key=input_key,
            input_kwargs=input_kwargs,
            trace_runtime=trace_runtime,
            metadata=raw_metadata,
        )
    runtime = dict(runtime_metadata or {})
    runtime.setdefault("framework_adapter_contract", contract)
    if spec is None:
        return GenericAgentWrapper(
            agent,
            method=method,
            input_mode=input_mode or "auto",
            input_key=input_key,
            input_kwargs=input_kwargs,
            output_key=output_key,
            system_prompt=system_prompt,
            metadata={
                "framework": key,
                "modality": str(raw_metadata.get("modality") or "text"),
                "adapter": "custom",
                "framework_adapter_contract": contract,
                **raw_metadata,
            },
            trace_runtime=trace_runtime,
            runtime_metadata=runtime,
        )

    return GenericAgentWrapper(
        agent,
        method=method or spec.method,
        input_mode=input_mode or spec.input_mode,
        input_key=input_key,
        input_kwargs=input_kwargs,
        output_key=output_key,
        system_prompt=system_prompt,
        metadata={
            "framework": spec.name,
            "modality": spec.modality,
            "framework_adapter_contract": contract,
            **raw_metadata,
        },
        trace_runtime=trace_runtime,
        runtime_metadata=runtime,
    )


async def probe_framework_adapter(
    framework: str,
    agent: Any,
    *,
    cases: Sequence[Mapping[str, Any]] | None = None,
    target: str | None = None,
    method: str | Callable[..., Any] | None = None,
    input_mode: InputMode | None = None,
    input_key: str | None = None,
    input_kwargs: Mapping[str, Any] | None = None,
    system_prompt: str | None = None,
    output_key: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_runtime: bool = True,
    allow_external_target: bool = False,
) -> dict[str, Any]:
    """Run a local adapter conformance probe for any framework shim.

    The probe is intentionally import-free: callers pass an already-created
    LangChain/LangGraph/LiveKit/Pipecat/custom object or plain callable. The
    function wraps it with the same generic adapter used by manifests, executes
    representative cases, and returns runtime evidence that can feed evals,
    reports, optimizer proofs, or Future AGI UI cards.
    """

    if target and _is_external_target(target) and not allow_external_target:
        raise ValueError(
            "external targets are disabled for framework adapter probes; "
            "set allow_external_target=True only when the user explicitly "
            "wants to test that live workload"
        )

    key = _framework_key(framework)
    method_name = _probe_method_name(method)
    selected_metadata = dict(metadata or {})
    contract = framework_adapter_contract(
        key,
        target=target,
        method=method_name,
        input_mode=input_mode,
        input_key=input_key,
        input_kwargs=input_kwargs,
        trace_runtime=trace_runtime,
        metadata=selected_metadata,
    )
    callable_signature = _adapter_callable_signature(
        agent,
        method=method,
        method_name=str(contract.get("method") or method_name or ""),
    )
    if callable_signature:
        contract["callable_signature"] = callable_signature
    selected_metadata["framework_adapter_contract"] = contract
    wrapper = wrap_framework(
        key,
        agent,
        target=target,
        method=method,
        input_mode=input_mode,
        input_key=input_key,
        input_kwargs=input_kwargs,
        system_prompt=system_prompt,
        output_key=output_key,
        metadata=selected_metadata,
        trace_runtime=trace_runtime,
        runtime_metadata={"framework_adapter_contract": contract},
    )

    probe_cases = _probe_cases(cases)
    case_results: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    for index, case in enumerate(probe_cases, start=1):
        case_result = await _run_probe_case(
            wrapper,
            key,
            case,
            index=index,
            trace_runtime=trace_runtime,
            contract=contract,
        )
        case_results.append(case_result)
        findings.extend(case_result.get("findings", []))

    summary = _probe_summary(case_results, contract)
    status = "passed" if not findings else "failed"
    return {
        "kind": "agent-learning.framework-adapter-probe.v1",
        "status": status,
        "passed": status == "passed",
        "framework": key,
        "method": method_name or str(contract.get("method") or "auto"),
        "input_mode": str(input_mode or contract.get("input_mode") or "auto"),
        "input_key": str(input_key or contract.get("input_key") or "")
        or None,
        "input_kwargs_keys": sorted(str(key) for key in dict(input_kwargs or {})),
        "requires_external_service": bool(contract.get("requires_external_service")),
        "allow_external_target": bool(allow_external_target),
        "contract": contract,
        "summary": summary,
        "cases": case_results,
        "findings": findings,
    }


def run_framework_adapter_probe(
    framework: str,
    agent: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Synchronous wrapper for :func:`probe_framework_adapter`.

    Use ``await probe_framework_adapter(...)`` when already inside an event loop.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(probe_framework_adapter(framework, agent, **kwargs))
    raise RuntimeError(
        "run_framework_adapter_probe cannot run inside an active event loop; "
        "await probe_framework_adapter(...) instead"
    )


def discover_framework_adapter(
    framework: str,
    agent: Any = None,
    *,
    target: str | None = None,
    method_candidates: Sequence[str | None] | None = None,
    input_mode_candidates: Sequence[InputMode] | None = None,
    modality: str | None = None,
    trace_runtime: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    allow_external_target: bool = False,
    max_candidates: int | None = 24,
) -> dict[str, Any]:
    """Discover local adapter candidates for an arbitrary framework object.

    Discovery never imports optional framework packages or calls the supplied
    agent. It inspects callable attributes, combines them with the built-in
    framework presets, and returns ranked method/input-mode contracts that can
    be passed directly to ``optimize_framework_adapter_probe``.
    """

    if target and _is_external_target(target) and not allow_external_target:
        raise ValueError(
            "external targets are disabled for framework adapter discovery; "
            "set allow_external_target=True only when the user explicitly "
            "wants to document that live workload"
        )
    if max_candidates is not None and int(max_candidates) <= 0:
        raise ValueError("max_candidates must be greater than zero")

    key = _framework_key(framework)
    spec = FRAMEWORK_PRESETS.get(key)
    selected_metadata = dict(metadata or {})
    inventory = _adapter_discovery_inventory(agent)
    methods = _adapter_discovery_methods(
        agent,
        spec=spec,
        method_candidates=method_candidates,
    )
    input_modes = _adapter_discovery_input_modes(
        spec=spec,
        input_mode_candidates=input_mode_candidates,
    )

    candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    explicit_methods = {
        str(method)
        for method in method_candidates or []
        if method is not None and str(method)
    }

    for method_name in methods:
        for input_mode in _adapter_discovery_modes_for_method(
            method_name,
            input_modes,
            spec=spec,
        ):
            pair = (str(method_name or ""), str(input_mode))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            input_key = _adapter_candidate_input_key(
                agent,
                method_name,
                input_mode,
            )
            contract = framework_adapter_contract(
                key,
                target=target,
                method=method_name,
                input_mode=input_mode,
                input_key=input_key,
                modality=modality,
                trace_runtime=trace_runtime,
                metadata=selected_metadata,
            )
            scoring = _adapter_discovery_score(
                method_name,
                input_mode,
                spec=spec,
                inventory=inventory,
                explicit_methods=explicit_methods,
            )
            adapter_candidate: dict[str, Any] = {
                "input_mode": input_mode,
                "trace_runtime": bool(trace_runtime),
            }
            if method_name:
                adapter_candidate["method"] = method_name
            if input_key:
                adapter_candidate["input_key"] = input_key
            if target:
                adapter_candidate["target"] = str(target)
            candidates.append(
                {
                    "rank": 0,
                    "framework": key,
                    "method": method_name or "auto",
                    "input_mode": input_mode,
                    "score": scoring["score"],
                    "reasons": scoring["reasons"],
                    "agent_method_present": _adapter_method_present(
                        inventory,
                        method_name,
                    ),
                    "contract": contract,
                    "adapter_candidate": adapter_candidate,
                }
            )

    candidates.sort(
        key=lambda item: (
            -float(item["score"]),
            _adapter_method_rank(str(item.get("method") or "")),
            _adapter_input_mode_rank(str(item.get("input_mode") or "")),
        )
    )
    if max_candidates is not None:
        candidates = candidates[: int(max_candidates)]
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    adapter_candidates = [
        dict(candidate["adapter_candidate"]) for candidate in candidates
    ]
    findings = _adapter_discovery_findings(inventory, candidates)
    status = "passed" if candidates else "failed"
    top = candidates[0] if candidates else {}
    return {
        "kind": "agent-learning.framework-adapter-discovery.v1",
        "status": status,
        "passed": status == "passed",
        "framework": key,
        "target": str(target) if target else None,
        "requires_external_service": False,
        "allow_external_target": bool(allow_external_target),
        "trace_runtime": bool(trace_runtime),
        "agent": inventory,
        "summary": {
            "framework": key,
            "candidate_count": len(candidates),
            "adapter_candidate_count": len(adapter_candidates),
            "max_candidates": max_candidates,
            "top_method": top.get("method"),
            "top_input_mode": top.get("input_mode"),
            "top_score": top.get("score"),
            "agent_provided": bool(inventory.get("provided")),
            "agent_callable": bool(inventory.get("callable")),
            "method_count": len(inventory.get("exposed_methods", [])),
            "local_executable_fixture": not bool(target and _is_external_target(target)),
        },
        "candidates": candidates,
        "adapter_candidates": adapter_candidates,
        "findings": findings,
        "evidence_requirements": [
            "local_introspection",
            "framework_adapter_contract",
            "adapter_candidates",
            "framework_adapter_probe",
            "metric_evidence",
        ],
    }


def _framework_key(value: str) -> str:
    return str(value or "custom").strip().lower().replace("-", "_") or "custom"


def _framework_matrix_keys(frameworks: Sequence[str] | str | None) -> list[str]:
    default_frameworks = (
        "langchain",
        "langgraph",
        "llamaindex",
        "crewai",
        "autogen",
        "openai_agents",
        "livekit",
        "pipecat",
    )
    values: Sequence[str] | str = frameworks or default_frameworks
    if isinstance(values, str):
        values = [values]
    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _framework_key(value)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    if not keys:
        raise ValueError("frameworks must contain at least one framework")
    return keys


def _local_fixture_target(framework: str) -> str:
    return f"agent-learning-fixture://framework/{_framework_key(framework)}"


def _is_external_target(target: str) -> bool:
    return urlparse(str(target or "")).scheme.lower() in {"http", "https"}


def _adapter_discovery_inventory(agent: Any) -> dict[str, Any]:
    if agent is None:
        return {
            "provided": False,
            "callable": False,
            "type": None,
            "exposed_methods": [],
            "wrapper": False,
        }

    exposed_methods = _adapter_public_callable_names(agent)
    return {
        "provided": True,
        "callable": callable(agent),
        "type": _adapter_agent_type(agent),
        "exposed_methods": exposed_methods,
        "wrapper": isinstance(agent, AgentWrapper),
    }


def _adapter_agent_type(agent: Any) -> str:
    if inspect.isfunction(agent) or inspect.ismethod(agent):
        module = getattr(agent, "__module__", "")
        qualname = getattr(agent, "__qualname__", getattr(agent, "__name__", "callable"))
        return f"{module}.{qualname}".strip(".")
    cls = type(agent)
    module = getattr(cls, "__module__", "")
    qualname = getattr(cls, "__qualname__", getattr(cls, "__name__", "object"))
    return f"{module}.{qualname}".strip(".")


def _adapter_public_callable_names(agent: Any) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for name in _DISCOVERY_METHOD_ORDER:
        if name == "__call__" and not callable(agent):
            continue
        if name != "__call__" and not _adapter_has_callable_method(agent, name):
            continue
        seen.add(name)
        names.append(name)

    try:
        members = inspect.getmembers_static(agent)
    except Exception:
        members = []

    public_names: list[str] = []
    for name, value in members:
        if name in seen:
            continue
        if name.startswith("_"):
            continue
        if not (
            inspect.isroutine(value)
            or isinstance(value, (classmethod, staticmethod))
            or _adapter_has_callable_method(agent, name)
        ):
            continue
        if not _adapter_has_callable_method(agent, name):
            continue
        public_names.append(name)

    public_names.sort(key=lambda item: (_adapter_method_rank(item), item))
    names.extend(public_names[:24])
    return names


def _adapter_resolve_callable_attr_path(agent: Any, method_name: str | None) -> Callable[..., Any] | None:
    if not method_name:
        return agent if callable(agent) else None
    value = agent
    for raw_part in str(method_name).split("."):
        part = raw_part.strip()
        if not part:
            return None
        try:
            value = getattr(value, part)
        except Exception:
            return None
    return value if callable(value) else None


def _adapter_method_leaf(method_name: str | None) -> str:
    return str(method_name or "").rsplit(".", 1)[-1]


def _adapter_has_callable_method(agent: Any, method_name: str) -> bool:
    if not method_name:
        return callable(agent)
    return _adapter_resolve_callable_attr_path(agent, method_name) is not None


def _adapter_discovery_methods(
    agent: Any,
    *,
    spec: FrameworkAdapterSpec | None,
    method_candidates: Sequence[str | None] | None,
) -> list[str | None]:
    methods: list[str | None] = []

    for method in method_candidates or []:
        methods.append(str(method) if method is not None and str(method) else None)
    if spec and spec.method:
        methods.append(spec.method)
    methods.extend(_adapter_public_callable_names(agent) if agent is not None else [])
    if agent is not None and callable(agent):
        methods.append(None)
    methods.extend(_DISCOVERY_METHOD_ORDER)
    if spec is None:
        methods.append(None)

    unique: list[str | None] = []
    seen: set[str] = set()
    for method in methods:
        key = str(method or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(method)
    return unique or [None]


def _adapter_discovery_input_modes(
    *,
    spec: FrameworkAdapterSpec | None,
    input_mode_candidates: Sequence[InputMode] | None,
) -> list[InputMode]:
    modes: list[InputMode] = []
    if input_mode_candidates is not None:
        modes.extend(input_mode_candidates)
    elif spec is not None and spec.input_mode != "auto":
        modes.append(spec.input_mode)
    modes.extend(_DISCOVERY_INPUT_MODE_ORDER)

    unique: list[InputMode] = []
    seen: set[str] = set()
    for mode in modes:
        normalized = str(mode or "auto")
        if normalized not in _DISCOVERY_INPUT_MODE_ORDER:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)  # type: ignore[arg-type]
    return unique or ["auto"]


def _adapter_discovery_modes_for_method(
    method_name: str | None,
    input_modes: Sequence[InputMode],
    *,
    spec: FrameworkAdapterSpec | None,
) -> list[InputMode]:
    modes: list[InputMode] = []
    inferred = _adapter_inferred_input_mode(method_name)
    if inferred:
        modes.append(inferred)
    elif method_name is None and spec is not None:
        modes.append(spec.input_mode)
    modes.extend(input_modes)

    unique: list[InputMode] = []
    seen: set[str] = set()
    for mode in modes:
        normalized = str(mode or "auto")
        if normalized not in _DISCOVERY_INPUT_MODE_ORDER:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)  # type: ignore[arg-type]
    return unique or ["auto"]


def _adapter_inferred_input_mode(method_name: str | None) -> InputMode | None:
    if method_name is None:
        return "agent_input"
    return (
        _DISCOVERY_METHOD_INPUT_MODES.get(method_name)
        or _DISCOVERY_METHOD_INPUT_MODES.get(_adapter_method_leaf(method_name))
    )


def _adapter_candidate_input_key(
    agent: Any,
    method_name: str | None,
    input_mode: InputMode,
) -> str | None:
    if agent is None or not method_name:
        return None
    method = _adapter_resolve_callable_attr_path(agent, method_name)
    if method is None:
        return None
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return None
    params = list(signature.parameters.values())
    names = {param.name: param for param in params}
    preferred_names = (
        (
            _METHOD_INPUT_KEY_PREFERENCES.get(str(method_name or ""))
            or _METHOD_INPUT_KEY_PREFERENCES.get(_adapter_method_leaf(method_name), ())
        )
        + _KEYWORD_INPUT_NAMES
    )
    accepts_positional = any(
        param.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        }
        for param in params
    )
    for name in preferred_names:
        param = names.get(name)
        if param is None or param.kind == inspect.Parameter.POSITIONAL_ONLY:
            continue
        if name == "inputs" or not accepts_positional or param.kind == inspect.Parameter.KEYWORD_ONLY:
            return name
    if not accepts_positional:
        for param in params:
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                return param.name
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
            if input_mode == "dict":
                if method_name == "kickoff":
                    return "inputs"
                if method_name in {"process", "process_frame"}:
                    return "frame"
                return "payload"
            return "task" if method_name in {"run", "arun", "run_stream"} else "input"
    return None


def _adapter_callable_signature(
    agent: Any,
    *,
    method: str | Callable[..., Any] | None,
    method_name: str | None,
) -> dict[str, Any]:
    callable_method = method if callable(method) else None
    resolved_method = str(method_name or "").strip()
    if callable_method is None:
        if resolved_method == "auto":
            resolved_method = ""
        callable_method = _adapter_resolve_callable_attr_path(agent, resolved_method)
    if callable_method is None and callable(agent):
        callable_method = agent
    if callable_method is None:
        return {}

    try:
        signature = inspect.signature(callable_method)
    except (TypeError, ValueError):
        return {}

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
            "annotation": _adapter_annotation_name(param.annotation),
        }
        for param in params
    ]
    selected_method = (
        str(method)
        if isinstance(method, str) and str(method)
        else resolved_method
        or getattr(callable_method, "__name__", None)
        or "callable"
    )
    method_leaf = _adapter_method_leaf(selected_method)
    preferred_input_key = _adapter_candidate_input_key(
        agent,
        selected_method if isinstance(method, str) else method_leaf,
        "auto",
    )
    return {
        "kind": "agent-learning.framework-adapter-callable-signature.v1",
        "inspectable": True,
        "method": selected_method,
        "method_leaf": method_leaf,
        "callable_type": _adapter_agent_type(callable_method),
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
        "preferred_input_key": preferred_input_key,
        "is_async": inspect.iscoroutinefunction(callable_method),
        "is_generator": inspect.isgeneratorfunction(callable_method),
        "is_async_generator": inspect.isasyncgenfunction(callable_method),
        "return_annotation": _adapter_annotation_name(signature.return_annotation),
    }


def _adapter_annotation_name(annotation: Any) -> str | None:
    if annotation is inspect.Signature.empty or annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _params_accept_positional(params: Sequence[inspect.Parameter]) -> bool:
    return any(
        param.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        }
        for param in params
    )


def _adapter_discovery_score(
    method_name: str | None,
    input_mode: InputMode,
    *,
    spec: FrameworkAdapterSpec | None,
    inventory: Mapping[str, Any],
    explicit_methods: set[str],
) -> dict[str, Any]:
    score = 0.15
    reasons: list[str] = ["local_contract_candidate"]
    method_present = _adapter_method_present(inventory, method_name)
    inferred_mode = _adapter_inferred_input_mode(method_name)

    if method_name and method_name in explicit_methods:
        score += 0.15
        reasons.append("explicit_method_candidate")
    if method_present:
        score += 0.35
        reasons.append("agent_exposes_method")
    elif method_name is None and inventory.get("callable"):
        score += 0.35
        reasons.append("agent_is_direct_callable")
    elif inventory.get("provided") and method_name:
        score -= 0.15
        reasons.append("method_not_found_on_agent")

    if spec and method_name and method_name == spec.method:
        score += 0.2
        reasons.append("matches_framework_preset_method")
    if spec and input_mode == spec.input_mode:
        score += 0.15
        reasons.append("matches_framework_preset_input_mode")
    if inferred_mode and input_mode == inferred_mode:
        score += 0.15
        reasons.append("matches_inferred_input_mode")
    if method_name == "execute_task" and input_mode == "dict":
        score += 0.1
        reasons.append("task_payload_adapter")
    if method_name == "process_frame" and input_mode == "dict":
        score += 0.1
        reasons.append("frame_payload_adapter")
    if method_name in _STREAMING_METHODS:
        score += 0.1
        reasons.append("streaming_adapter_surface")
    if input_mode == "auto":
        score -= 0.05
        reasons.append("auto_input_mode_requires_runtime_inference")

    normalized = max(0.0, min(1.0, score))
    return {"score": round(normalized, 3), "reasons": reasons}


def _adapter_method_present(
    inventory: Mapping[str, Any],
    method_name: str | None,
) -> bool:
    if method_name is None:
        return bool(inventory.get("callable"))
    return method_name in set(inventory.get("exposed_methods", []) or [])


def _adapter_method_rank(method_name: str) -> int:
    normalized = str(method_name or "")
    if normalized == "auto":
        normalized = ""
    try:
        return _DISCOVERY_METHOD_ORDER.index(normalized)
    except ValueError:
        return len(_DISCOVERY_METHOD_ORDER)


def _adapter_input_mode_rank(input_mode: str) -> int:
    try:
        return _DISCOVERY_INPUT_MODE_ORDER.index(input_mode)  # type: ignore[arg-type]
    except ValueError:
        return len(_DISCOVERY_INPUT_MODE_ORDER)


def _adapter_discovery_findings(
    inventory: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not inventory.get("provided"):
        findings.append(
            {
                "level": "info",
                "type": "agent_not_provided",
                "message": "Discovery used framework presets without inspecting an agent.",
            }
        )
    elif not inventory.get("callable") and not inventory.get("exposed_methods"):
        findings.append(
            {
                "level": "warning",
                "type": "no_callable_adapter_surface",
                "message": "Agent does not expose a discovered callable adapter method.",
            }
        )
    if not candidates:
        findings.append(
            {
                "level": "error",
                "type": "adapter_candidates_missing",
                "message": "No framework adapter candidates were discovered.",
            }
        )
    return findings


def _framework_matrix_summary(contracts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frameworks: list[str] = []
    methods: dict[str, str] = {}
    input_modes: dict[str, str] = {}
    modalities: dict[str, str] = {}
    transports: dict[str, str] = {}
    capabilities: set[str] = set()
    evidence_requirements: set[str] = set()
    target_schemes: set[str] = set()

    for contract in contracts:
        framework = _framework_key(str(contract.get("framework") or "custom"))
        frameworks.append(framework)
        methods[framework] = str(contract.get("method") or "")
        input_modes[framework] = str(contract.get("input_mode") or "")
        modalities[framework] = str(contract.get("modality") or "")
        transports[framework] = str(contract.get("transport") or "")
        capabilities.update(
            str(item) for item in contract.get("capabilities", []) or []
        )
        evidence_requirements.update(
            str(item) for item in contract.get("evidence_requirements", []) or []
        )
        target_scheme = str(contract.get("target_scheme") or "")
        if target_scheme:
            target_schemes.add(target_scheme)

    return {
        "frameworks": frameworks,
        "modalities": sorted(set(modalities.values())),
        "transports": sorted(set(transports.values())),
        "methods": methods,
        "input_modes": input_modes,
        "framework_modalities": modalities,
        "framework_transports": transports,
        "capabilities": sorted(capabilities),
        "evidence_requirements": sorted(evidence_requirements),
        "target_schemes": sorted(target_schemes),
        "contract_count": len(contracts),
        "local_executable_fixture_count": sum(
            1 for contract in contracts if bool(contract.get("local_executable_fixture"))
        ),
        "trace_runtime_count": sum(
            1 for contract in contracts if bool(contract.get("trace_runtime"))
        ),
        "requires_external_service_count": sum(
            1 for contract in contracts if bool(contract.get("requires_external_service"))
        ),
        "external_target_count": sum(
            1
            for contract in contracts
            if _is_external_target(str(contract.get("target") or ""))
        ),
    }


def _framework_matrix_findings(
    contracts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for contract in contracts:
        framework = _framework_key(str(contract.get("framework") or "custom"))
        if contract.get("kind") != "agent-learning.framework-adapter-contract.v1":
            findings.append({"framework": framework, "type": "contract_kind_mismatch"})
        if bool(contract.get("requires_external_service")):
            findings.append(
                {"framework": framework, "type": "external_service_required"}
            )
        if not bool(contract.get("local_executable_fixture")):
            findings.append({"framework": framework, "type": "local_fixture_missing"})
        if _is_external_target(str(contract.get("target") or "")):
            findings.append({"framework": framework, "type": "external_target_scheme"})
    return findings


def _framework_adapter_capability_profile_from_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(contract))
    framework = _framework_key(str(payload.get("framework") or "custom"))
    capabilities = _framework_profile_capabilities(payload)
    task_surfaces = _framework_profile_task_surfaces(payload)
    bindings = _framework_profile_bindings(payload)
    findings = _framework_profile_findings(payload, bindings)
    summary = _framework_profile_summary(
        payload,
        capabilities=capabilities,
        task_surfaces=task_surfaces,
        bindings=bindings,
    )
    status = "passed" if not findings else "failed"
    return {
        "kind": "agent-learning.framework-adapter-capability-profile.v1",
        "status": status,
        "passed": status == "passed",
        "framework": framework,
        "method": str(payload.get("method") or "auto"),
        "input_mode": str(payload.get("input_mode") or "auto"),
        "modality": str(payload.get("modality") or "text"),
        "transport": str(payload.get("transport") or "in_process"),
        "requires_external_service": bool(payload.get("requires_external_service")),
        "local_executable_fixture": bool(payload.get("local_executable_fixture")),
        "trace_runtime": bool(payload.get("trace_runtime")),
        "contract": payload,
        "capabilities": capabilities,
        "task_surfaces": task_surfaces,
        "bindings": bindings,
        "summary": summary,
        "findings": findings,
        "evidence_requirements": sorted(
            {
                "framework_adapter_profile",
                "framework_adapter_contract",
                *[
                    str(item)
                    for item in payload.get("evidence_requirements", []) or []
                    if str(item)
                ],
            }
        ),
    }


def _framework_profile_capabilities(
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    names = [
        *[str(item) for item in contract.get("capabilities", []) or []],
        "adapter_contract",
        "local_fixture",
        "metric_evidence",
        "optimization_search",
    ]
    if bool(contract.get("trace_runtime")):
        names.append("trace_runtime")
    if str(contract.get("modality") or "") == "voice":
        names.extend(["voice", "realtime"])
    if str(contract.get("modality") or "") == "cua":
        names.extend(["browser", "computer_use"])

    capabilities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_name in names:
        name = _framework_key(raw_name)
        if not name or name in seen:
            continue
        seen.add(name)
        capabilities.append(
            {
                "name": name,
                "category": _framework_profile_capability_category(name),
                "status": "supported",
                "source": "framework_adapter_contract",
            }
        )
    return capabilities


def _framework_profile_capability_category(name: str) -> str:
    normalized = _framework_key(name)
    if normalized in {"tool_calls", "tools", "call_tool"}:
        return "tools"
    if normalized in {"runtime_trace", "trace_runtime", "metric_evidence"}:
        return "observability"
    if normalized in {"structured_input", "messages", "artifacts", "state"}:
        return "io"
    if normalized in {"voice", "realtime", "streaming"}:
        return "realtime"
    if normalized in {"browser", "computer_use"}:
        return "computer_use"
    if normalized in {"environment_replay", "reset_step_trace"}:
        return "world"
    if normalized in {"optimization_search"}:
        return "optimization"
    if normalized in {"adapter_contract", "local_fixture"}:
        return "adapter"
    return "framework"


def _framework_profile_task_surfaces(
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    framework = _framework_key(str(contract.get("framework") or "custom"))
    method = str(contract.get("method") or "auto")
    input_mode = str(contract.get("input_mode") or "auto")
    return [
        {
            "name": "framework_adapter_simulation",
            "library": "simulate-sdk",
            "framework": framework,
            "method": method,
            "input_mode": input_mode,
            "evidence": [
                "framework_adapter_contract",
                "framework_runtime",
                "framework_trace",
            ],
        },
        {
            "name": "framework_adapter_evaluation",
            "library": "ai-evaluation",
            "metric": "framework_adapter_contract_quality",
            "evidence": [
                "adapter_conformance",
                "metric_evidence",
                "tool_calls",
            ],
        },
        {
            "name": "framework_adapter_optimization",
            "library": "agent-opt",
            "layers": ["framework", "integration", "harness", "evaluator"],
            "search_paths": [
                "agent.method",
                "agent.input_mode",
                "simulation.environments",
            ],
        },
    ]


def _framework_profile_bindings(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    method = str(contract.get("method") or "auto")
    input_mode = str(contract.get("input_mode") or "auto")
    framework = _framework_key(str(contract.get("framework") or "custom"))
    return {
        "simulate-sdk": {
            "adapter": "wrap_framework",
            "contract": "framework_adapter_contract",
            "matrix": "framework_adapter_contract_matrix",
            "probe": "probe_framework_adapter",
            "framework": framework,
            "method": method,
            "input_mode": input_mode,
            "local_executable_fixture": bool(
                contract.get("local_executable_fixture")
            ),
        },
        "ai-evaluation": {
            "metric": "framework_adapter_contract_quality",
            "contract_kind": "agent-learning.framework-adapter-contract.v1",
            "required_frameworks": [framework],
            "required_methods": [method] if method != "auto" else [],
            "required_input_modes": [input_mode] if input_mode != "auto" else [],
            "required_capabilities": [
                str(item)
                for item in contract.get("capabilities", []) or []
                if str(item)
            ],
        },
        "agent-opt": {
            "target": "OptimizationTarget",
            "candidate": "AgentCandidate",
            "optimizer": "AgentOptimizer",
            "layers": ["framework", "integration", "harness", "evaluator"],
            "search_paths": [
                "agent.method",
                "agent.input_mode",
                "simulation.environments",
            ],
        },
    }


def _framework_profile_findings(
    contract: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    framework = _framework_key(str(contract.get("framework") or "custom"))
    if contract.get("kind") != "agent-learning.framework-adapter-contract.v1":
        findings.append(
            {
                "level": "error",
                "framework": framework,
                "type": "contract_kind_mismatch",
            }
        )
    if bool(contract.get("requires_external_service")):
        findings.append(
            {
                "level": "error",
                "framework": framework,
                "type": "external_service_required",
            }
        )
    if not bool(contract.get("local_executable_fixture")):
        findings.append(
            {
                "level": "error",
                "framework": framework,
                "type": "local_fixture_missing",
            }
        )
    if _is_external_target(str(contract.get("target") or "")):
        findings.append(
            {
                "level": "error",
                "framework": framework,
                "type": "external_target_scheme",
            }
        )
    for library in ("simulate-sdk", "ai-evaluation", "agent-opt"):
        if not isinstance(bindings.get(library), Mapping):
            findings.append(
                {
                    "level": "error",
                    "framework": framework,
                    "type": "trinity_binding_missing",
                    "library": library,
                }
            )
    return findings


def _framework_profile_summary(
    contract: Mapping[str, Any],
    *,
    capabilities: Sequence[Mapping[str, Any]],
    task_surfaces: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "framework": _framework_key(str(contract.get("framework") or "custom")),
        "method": str(contract.get("method") or "auto"),
        "input_mode": str(contract.get("input_mode") or "auto"),
        "modality": str(contract.get("modality") or "text"),
        "transport": str(contract.get("transport") or "in_process"),
        "capability_count": len(capabilities),
        "task_surface_count": len(task_surfaces),
        "binding_count": len(bindings),
        "libraries": sorted(str(key) for key in bindings),
        "local_executable_fixture": bool(contract.get("local_executable_fixture")),
        "trace_runtime": bool(contract.get("trace_runtime")),
        "requires_external_service": bool(contract.get("requires_external_service")),
    }


def _framework_profile_collection_summary(
    profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frameworks = [
        _framework_key(str(profile.get("framework") or "custom"))
        for profile in profiles
    ]
    libraries = sorted(
        {
            str(library)
            for profile in profiles
            for library in (profile.get("bindings") or {})
        }
    )
    capabilities = sorted(
        {
            _framework_key(str(capability.get("name") or ""))
            for profile in profiles
            for capability in profile.get("capabilities", []) or []
            if isinstance(capability, Mapping)
        }
    )
    return {
        "frameworks": frameworks,
        "profile_count": len(profiles),
        "passed_profile_count": sum(
            1 for profile in profiles if str(profile.get("status")) == "passed"
        ),
        "failed_profile_count": sum(
            1 for profile in profiles if str(profile.get("status")) != "passed"
        ),
        "libraries": libraries,
        "capabilities": capabilities,
        "local_executable_fixture_count": sum(
            1 for profile in profiles if bool(profile.get("local_executable_fixture"))
        ),
        "requires_external_service_count": sum(
            1 for profile in profiles if bool(profile.get("requires_external_service"))
        ),
    }


def _framework_profile_collection_findings(
    profiles: Sequence[Mapping[str, Any]],
    *,
    required_frameworks: Sequence[str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    observed = {
        _framework_key(str(profile.get("framework") or "custom"))
        for profile in profiles
    }
    for framework in required_frameworks:
        key = _framework_key(framework)
        if key not in observed:
            findings.append(
                {
                    "level": "error",
                    "framework": key,
                    "type": "framework_profile_missing",
                }
            )
    for profile in profiles:
        for finding in profile.get("findings", []) or []:
            if isinstance(finding, Mapping):
                findings.append(copy.deepcopy(dict(finding)))
    return findings


def _default_transport(modality: str) -> str:
    if modality == "voice":
        return "realtime_adapter"
    if modality == "cua":
        return "browser_adapter"
    if modality == "image":
        return "multimodal_adapter"
    return "in_process"


def _default_lifecycle_hooks(modality: str) -> tuple[str, ...]:
    if modality == "voice":
        return ("setup", "connect", "stream", "respond", "teardown")
    if modality == "cua":
        return ("setup", "observe", "act", "verify", "teardown")
    return ("setup", "invoke", "observe", "teardown")


def _default_capabilities(modality: str, input_mode: str) -> tuple[str, ...]:
    capabilities = [
        "messages",
        "tool_calls",
        "runtime_trace",
        "state",
        "artifacts",
    ]
    if input_mode == "dict":
        capabilities.append("structured_input")
    if input_mode in {"dict", "messages"}:
        capabilities.append("streaming_trace")
    if modality == "voice":
        capabilities.extend(["voice_frames", "realtime_events"])
    elif modality == "cua":
        capabilities.extend(["browser_actions", "visual_grounding"])
    elif modality == "image":
        capabilities.extend(["image_context", "multimodal_grounding"])
    return tuple(capabilities)


def _input_schema(input_mode: str) -> dict[str, Any]:
    if input_mode == "dict":
        return {
            "type": "object",
            "required": ["messages", "scenario"],
            "additionalProperties": True,
        }
    if input_mode == "messages":
        return {
            "type": "array",
            "items": {"type": "object", "required": ["role", "content"]},
        }
    if input_mode == "agent_input":
        return {"type": "object", "class": "AgentInput"}
    if input_mode == "text":
        return {"type": "string"}
    return {"type": "any"}


def _output_schema() -> dict[str, Any]:
    return {
        "oneOf": [
            {"type": "string"},
            {"type": "object", "class": "AgentResponse"},
        ],
        "required_trace_state": ["framework_runtime"],
    }


def _probe_method_name(method: str | Callable[..., Any] | None) -> str | None:
    if method is None:
        return None
    if isinstance(method, str):
        return method
    return getattr(method, "__name__", None) or "callable"


def _probe_cases(cases: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not cases:
        return [
            {
                "id": "default",
                "input": "Return a short adapter probe result.",
            }
        ]
    return [dict(case) for case in cases]


async def _run_probe_case(
    wrapper: AgentWrapper,
    framework: str,
    case: Mapping[str, Any],
    *,
    index: int,
    trace_runtime: bool,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = str(case.get("id") or f"case_{index}")
    agent_input = _probe_agent_input(framework, case, index=index)
    raw_response = await wrapper.call(agent_input)
    response = (
        raw_response
        if isinstance(raw_response, AgentResponse)
        else AgentResponse(content=str(raw_response))
    )
    response_payload = _probe_response_payload(response)
    runtime_trace = dict((response.state or {}).get("framework_runtime") or {})
    observed_io_contract = _probe_observed_io_contract(
        framework,
        case_id=case_id,
        runtime_trace=runtime_trace,
        response_payload=response_payload,
        contract=contract,
    )
    checks = _probe_case_checks(
        case,
        response_payload,
        runtime_trace=runtime_trace,
        observed_io_contract=observed_io_contract,
        trace_runtime=trace_runtime,
        contract=contract,
    )
    findings = [
        {
            "case_id": case_id,
            "check": check["id"],
            "level": "error",
            "message": check["message"],
            "expected": check.get("expected"),
            "observed": check.get("observed"),
        }
        for check in checks
        if not check["passed"]
    ]
    return {
        "id": case_id,
        "status": "passed" if not findings else "failed",
        "passed": not findings,
        "input": {
            "message_count": len(agent_input.messages),
            "tool_count": len(agent_input.tools),
            "artifact_count": len(agent_input.artifacts),
            "event_count": len(agent_input.events),
            "modality": agent_input.modality,
        },
        "response": response_payload,
        "runtime_trace": runtime_trace,
        "observed_io_contract": observed_io_contract,
        "checks": checks,
        "findings": findings,
    }


def _probe_agent_input(
    framework: str,
    case: Mapping[str, Any],
    *,
    index: int,
) -> AgentInput:
    messages = _probe_messages(case)
    new_message = dict(case.get("new_message") or messages[-1])
    metadata = {
        "framework": framework,
        "probe_case_id": str(case.get("id") or f"case_{index}"),
        **dict(case.get("metadata") or {}),
    }
    return AgentInput(
        thread_id=str(case.get("thread_id") or f"{framework}-probe-{index}"),
        messages=messages,
        new_message=new_message,
        execution_id=str(case.get("execution_id") or f"{framework}-probe"),
        turn_index=int(case.get("turn_index") or index - 1),
        scenario_name=str(case.get("scenario_name") or "framework-adapter-probe"),
        persona=dict(case.get("persona") or {}),
        situation=str(case.get("situation") or ""),
        expected_outcome=str(case.get("expected_outcome") or ""),
        modality=str(case.get("modality") or ""),
        artifacts=list(case.get("artifacts") or []),
        events=list(case.get("events") or []),
        memory=dict(case.get("memory") or {}),
        tools=[dict(tool) for tool in case.get("tools", []) if isinstance(tool, Mapping)],
        metadata=metadata,
    )


def _probe_messages(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_messages = case.get("messages")
    if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes)):
        messages = [
            dict(message)
            for message in raw_messages
            if isinstance(message, Mapping)
        ]
        if messages:
            return messages
    message = case.get("input", case.get("message", "Run the adapter probe."))
    return [{"role": "user", "content": str(message)}]


def _probe_case_checks(
    case: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    runtime_trace: Mapping[str, Any],
    observed_io_contract: Mapping[str, Any],
    trace_runtime: bool,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    content = str(response.get("content") or "")
    tool_names = set(response.get("tool_names") or [])
    event_types = set(response.get("event_types") or [])
    state_keys = set(response.get("state_keys") or [])

    for term in _probe_strings(case.get("expected_contains")):
        checks.append(
            _probe_check(
                f"content_contains_{_framework_key(term)}",
                term.lower() in content.lower(),
                f"response content should contain {term!r}",
                expected=term,
                observed=content,
            )
        )
    for tool in _probe_strings(case.get("required_tools")):
        checks.append(
            _probe_check(
                f"required_tool_{_framework_key(tool)}",
                tool in tool_names,
                f"response should emit required tool {tool!r}",
                expected=tool,
                observed=sorted(tool_names),
            )
        )
    for event_type in _probe_strings(case.get("required_events")):
        checks.append(
            _probe_check(
                f"required_event_{_framework_key(event_type)}",
                event_type in event_types,
                f"response should emit required event {event_type!r}",
                expected=event_type,
                observed=sorted(event_types),
            )
        )
    for state_key in _probe_strings(case.get("required_state_keys")):
        checks.append(
            _probe_check(
                f"required_state_{_framework_key(state_key)}",
                state_key in state_keys,
                f"response should include required state key {state_key!r}",
                expected=state_key,
                observed=sorted(state_keys),
            )
        )
    if trace_runtime:
        runtime_summary = dict(runtime_trace.get("summary") or {})
        runtime_contract = dict(
            dict(runtime_trace.get("metadata") or {}).get("framework_adapter_contract")
            or {}
        )
        signature = dict(contract.get("callable_signature") or {})
        io_summary = dict(observed_io_contract.get("summary") or {})
        checks.extend(
            [
                _probe_check(
                    "framework_runtime_trace_present",
                    bool(runtime_trace),
                    "trace_runtime=True should attach framework_runtime state",
                    expected=True,
                    observed=bool(runtime_trace),
                ),
                _probe_check(
                    "framework_runtime_contract_present",
                    runtime_contract.get("kind")
                    == "agent-learning.framework-adapter-contract.v1",
                    "runtime trace should carry the adapter contract",
                    expected="agent-learning.framework-adapter-contract.v1",
                    observed=runtime_contract.get("kind"),
                ),
                _probe_check(
                    "framework_runtime_invocation_present",
                    int(runtime_summary.get("invocation_count") or 0) >= 1,
                    "runtime trace should record at least one invocation",
                    expected=">=1",
                    observed=runtime_summary.get("invocation_count"),
                ),
                _probe_check(
                    "framework_adapter_callable_signature_present",
                    signature.get("kind")
                    == "agent-learning.framework-adapter-callable-signature.v1",
                    "probe contract should carry deterministic callable signature evidence",
                    expected="agent-learning.framework-adapter-callable-signature.v1",
                    observed=signature.get("kind"),
                ),
                _probe_check(
                    "framework_adapter_observed_io_contract_present",
                    observed_io_contract.get("kind")
                    == "agent-learning.framework-adapter-observed-io-contract.v1"
                    and int(io_summary.get("invocation_count") or 0) >= 1,
                    "probe case should carry observed input/output contract evidence",
                    expected="agent-learning.framework-adapter-observed-io-contract.v1",
                    observed={
                        "kind": observed_io_contract.get("kind"),
                        "invocation_count": io_summary.get("invocation_count"),
                    },
                ),
                _probe_check(
                    "framework_adapter_observed_io_matches_signature",
                    io_summary.get("signature_bound") is True,
                    "observed adapter invocation should bind to the callable signature",
                    expected=True,
                    observed=io_summary.get("signature_bound"),
                ),
            ]
        )
    checks.append(
        _probe_check(
            "adapter_contract_local_first",
            contract.get("requires_external_service") is False,
            "adapter contract should not require a hosted service",
            expected=False,
            observed=contract.get("requires_external_service"),
        )
    )
    return checks


def _probe_check(
    check_id: str,
    passed: bool,
    message: str,
    *,
    expected: Any = None,
    observed: Any = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "message": message,
        "expected": expected,
        "observed": observed,
    }


def _probe_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _unique_probe_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _probe_observed_io_contract(
    framework: str,
    *,
    case_id: str,
    runtime_trace: Mapping[str, Any],
    response_payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    invocations = [
        dict(invocation)
        for invocation in list(runtime_trace.get("invocations") or [])
        if isinstance(invocation, Mapping)
    ]
    signature = (
        dict(contract.get("callable_signature") or {})
        if isinstance(contract.get("callable_signature"), Mapping)
        else {}
    )
    observed_invocations: list[dict[str, Any]] = []
    for index, invocation in enumerate(invocations, start=1):
        input_shape = dict(invocation.get("input") or {})
        output_shape = dict(invocation.get("output") or {})
        observed_invocations.append(
            {
                "id": str(invocation.get("id") or f"{case_id}_invocation_{index}"),
                "framework": str(invocation.get("framework") or framework),
                "method": str(invocation.get("method") or ""),
                "input_mode": str(invocation.get("input_mode") or ""),
                "call_style": str(invocation.get("call_style") or ""),
                "input_key": invocation.get("input_key"),
                "input_kwargs_keys": _unique_probe_strings(
                    list(invocation.get("input_kwargs_keys") or [])
                ),
                "input_shape": input_shape,
                "output_shape": output_shape,
                "duration_ms": int(invocation.get("duration_ms") or 0),
                "signals": _unique_probe_strings(list(invocation.get("signals") or [])),
            }
        )
    signature_bound = bool(signature) and bool(observed_invocations) and all(
        _probe_invocation_matches_signature(invocation, signature)
        for invocation in observed_invocations
    )
    input_keys = _unique_probe_strings(
        invocation.get("input_key")
        for invocation in observed_invocations
        if invocation.get("input_key") not in (None, "", [], {})
    )
    output_shapes = [dict(item.get("output_shape") or {}) for item in observed_invocations]
    return {
        "kind": "agent-learning.framework-adapter-observed-io-contract.v1",
        "framework": framework,
        "case_id": case_id,
        "method": contract.get("method"),
        "input_mode": contract.get("input_mode"),
        "signature_method": signature.get("method"),
        "signature_bound": signature_bound,
        "invocations": observed_invocations,
        "summary": {
            "invocation_count": len(observed_invocations),
            "methods": _unique_probe_strings(
                invocation.get("method") for invocation in observed_invocations
            ),
            "input_modes": _unique_probe_strings(
                invocation.get("input_mode") for invocation in observed_invocations
            ),
            "call_styles": _unique_probe_strings(
                invocation.get("call_style") for invocation in observed_invocations
            ),
            "input_keys": input_keys,
            "input_kwargs_keys": _unique_probe_strings(
                key
                for invocation in observed_invocations
                for key in list(invocation.get("input_kwargs_keys") or [])
            ),
            "input_types": _unique_probe_strings(
                dict(invocation.get("input_shape") or {}).get("type")
                for invocation in observed_invocations
            ),
            "output_types": _unique_probe_strings(
                output.get("type") for output in output_shapes
            ),
            "output_state_keys": _unique_probe_strings(
                key for output in output_shapes for key in list(output.get("state_keys") or [])
            ),
            "output_metadata_keys": _unique_probe_strings(
                key
                for output in output_shapes
                for key in list(output.get("metadata_keys") or [])
            ),
            "output_tool_names": _unique_probe_strings(
                key for output in output_shapes for key in list(output.get("tool_names") or [])
            ),
            "output_event_types": _unique_probe_strings(
                key for output in output_shapes for key in list(output.get("event_types") or [])
            ),
            "output_artifact_types": _unique_probe_strings(
                key
                for output in output_shapes
                for key in list(output.get("artifact_types") or [])
            ),
            "content_observed": bool(response_payload.get("content")),
            "signature_bound": signature_bound,
        },
    }


def _probe_invocation_matches_signature(
    invocation: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> bool:
    if not signature:
        return False
    method = str(invocation.get("method") or "")
    signature_method = str(signature.get("method") or "")
    signature_leaf = str(signature.get("method_leaf") or "")
    method_matches = method in {signature_method, signature_leaf, "callable"} or (
        signature_method in {"", "auto"} and bool(method)
    )
    if not method_matches:
        return False

    call_style = str(invocation.get("call_style") or "")
    input_key = invocation.get("input_key")
    parameter_names = set(str(name) for name in list(signature.get("parameter_names") or []))
    input_kwargs_keys = set(
        str(key) for key in list(invocation.get("input_kwargs_keys") or []) if str(key)
    )
    required_parameters = set(
        str(name) for name in list(signature.get("required_parameters") or []) if str(name)
    )
    accepts_var_keyword = bool(signature.get("accepts_var_keyword"))

    if call_style in {"keyword", "positional_with_kwargs"} and input_key:
        return (
            str(input_key) in parameter_names
            or accepts_var_keyword
            or str(input_key) == str(signature.get("preferred_input_key") or "")
        )
    if call_style == "expanded_kwargs":
        return accepts_var_keyword
    if call_style in {"positional", "positional_with_kwargs"}:
        return bool(signature.get("accepts_positional"))
    if call_style == "none":
        return required_parameters <= input_kwargs_keys
    return False


def _probe_response_payload(response: AgentResponse) -> dict[str, Any]:
    tool_calls = [dict(call) for call in response.tool_calls or []]
    events = [event.model_dump() for event in response.events]
    artifacts = [artifact.model_dump() for artifact in response.artifacts]
    state = dict(response.state or {})
    metadata = dict(response.metadata or {})
    streaming_trace = (
        state.get("streaming_trace")
        if isinstance(state.get("streaming_trace"), Mapping)
        else {}
    )
    return {
        "content": response.content,
        "tool_call_count": len(tool_calls),
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
        "event_count": len(events),
        "event_types": sorted({str(event.get("type") or "") for event in events}),
        "artifact_count": len(artifacts),
        "artifact_types": sorted({str(artifact.get("type") or "") for artifact in artifacts}),
        "artifact_evidence": [
            _probe_artifact_evidence(artifact) for artifact in artifacts
        ],
        "state_keys": sorted(str(key) for key in state),
        "framework_lifecycle_summary": _probe_framework_lifecycle_summary(
            state.get("framework_lifecycle_trace")
        ),
        "framework_trace_summary": _probe_framework_trace_summary(
            state.get("framework_trace")
        ),
        "message_history_summary": _probe_message_history_summary(
            state.get("message_history")
        ),
        "framework_handoff_summary": _probe_framework_handoff_summary(
            state.get("framework_handoffs")
        ),
        "orchestration_trace_summary": _probe_orchestration_trace_summary(
            state.get("orchestration_trace")
        ),
        "workflow_trace_summary": _probe_workflow_trace_summary(
            state.get("workflow_trace")
        ),
        "realtime_trace_summary": _probe_realtime_trace_summary(
            state.get("realtime_trace")
        ),
        "framework_memory_summary": _probe_framework_memory_summary(
            state.get("framework_memory")
        ),
        "retrieval_memory_summary": _probe_retrieval_memory_summary(
            state.get("retrieval_memory")
        ),
        "agent_memory_lineage_summary": _probe_agent_memory_lineage_summary(
            state.get("agent_memory_lineage")
        ),
        "browser_cua_summary": _probe_browser_cua_summary(
            state.get("browser_cua")
        ),
        "mcp_tool_session_summary": _probe_mcp_tool_session_summary(
            state.get("mcp_tool_session")
        ),
        "a2a_protocol_summary": _probe_a2a_protocol_summary(
            state.get("a2a_protocol_trace")
        ),
        "openenv_summary": _probe_openenv_summary(state.get("openenv")),
        "agent_trust_boundary_summary": _probe_agent_trust_boundary_summary(
            state.get("agent_trust_boundary_model")
        ),
        "agent_control_plane_summary": _probe_agent_control_plane_summary(
            state.get("agent_control_plane")
        ),
        "metadata_keys": sorted(str(key) for key in metadata),
        "streaming": bool(streaming_trace or metadata.get("streaming")),
        "streaming_trace_signals": sorted(
            str(signal)
            for signal in (dict(streaming_trace).get("signals") or [])
            if str(signal)
        ),
        "streaming_trace_summary": dict(
            dict(streaming_trace).get("summary") or {}
        ),
    }


def _probe_framework_lifecycle_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    summary = trace.get("summary")
    return dict(summary) if isinstance(summary, Mapping) else {}


def _probe_agent_trust_boundary_summary(value: Any) -> dict[str, Any]:
    payload = dict(value or {}) if isinstance(value, Mapping) else {}
    summary = dict(payload.get("summary") or {}) if payload else {}
    if not payload and not summary:
        return {}
    return {
        "framework": payload.get("framework"),
        "control_count": summary.get("control_count"),
        "required_control_rate": summary.get("required_control_rate"),
        "high_risk_unmitigated_count": summary.get(
            "high_risk_unmitigated_count"
        ),
        "evidence_count": summary.get("evidence_count"),
        "gaps": list(summary.get("gaps") or []),
        "present_controls": list(summary.get("present_controls") or []),
        "present_categories": list(summary.get("present_categories") or []),
        "assets": [
            str(item.get("id") or item.get("name") or "")
            for item in _probe_mappings(payload.get("assets"))
            if item.get("id") or item.get("name")
        ],
        "tools": [
            str(item.get("id") or item.get("name") or "")
            for item in _probe_mappings(payload.get("tools"))
            if item.get("id") or item.get("name")
        ],
        "surfaces": [
            str(item.get("id") or item.get("name") or "")
            for item in _probe_mappings(payload.get("surfaces"))
            if item.get("id") or item.get("name")
        ],
        "threats": list(summary.get("threats") or []),
        "mitigated_threats": list(summary.get("mitigated_threats") or []),
        "signals": list(payload.get("signals") or []),
        "has_identity": bool(summary.get("has_identity")),
        "has_permissions": bool(summary.get("has_permissions")),
        "has_sandbox": bool(summary.get("has_sandbox")),
        "has_audit": bool(summary.get("has_audit")),
        "has_canaries": bool(summary.get("has_canaries")),
        "has_human_approval": bool(summary.get("has_human_approval")),
        "has_memory_isolation": bool(summary.get("has_memory_isolation")),
        "has_network_egress_controls": bool(
            summary.get("has_network_egress_controls")
        ),
        "has_tool_allowlist": bool(summary.get("has_tool_allowlist")),
        "has_data_boundary": bool(summary.get("has_data_boundary")),
        "has_secret_handling": bool(summary.get("has_secret_handling")),
    }


def _probe_agent_control_plane_summary(value: Any) -> dict[str, Any]:
    payload = dict(value or {}) if isinstance(value, Mapping) else {}
    summary = dict(payload.get("summary") or {}) if payload else {}
    if not payload and not summary:
        return {}
    return {
        "framework": payload.get("framework"),
        "control_count": summary.get("control_count"),
        "required_control_rate": summary.get("required_control_rate"),
        "exceeded_budget_count": summary.get("exceeded_budget_count"),
        "high_risk_uncontained_count": summary.get(
            "high_risk_uncontained_count"
        ),
        "approval_required_action_count": summary.get(
            "approval_required_action_count"
        ),
        "approved_action_count": summary.get("approved_action_count"),
        "blocked_action_count": summary.get("blocked_action_count"),
        "rolled_back_action_count": summary.get("rolled_back_action_count"),
        "contained_incident_count": summary.get("contained_incident_count"),
        "within_budget_count": summary.get("within_budget_count"),
        "evidence_count": summary.get("evidence_count"),
        "gaps": list(summary.get("gaps") or []),
        "present_controls": list(summary.get("present_controls") or []),
        "present_categories": list(summary.get("present_categories") or []),
        "actions": list(summary.get("actions") or []),
        "budgets": list(summary.get("budgets") or []),
        "incidents": list(summary.get("incidents") or []),
        "signals": list(payload.get("signals") or []),
        "has_risk_scoring": bool(summary.get("has_risk_scoring")),
        "has_action_policy": bool(summary.get("has_action_policy")),
        "has_approval_gates": bool(summary.get("has_approval_gates")),
        "has_rollback": bool(summary.get("has_rollback")),
        "has_kill_switch": bool(summary.get("has_kill_switch")),
        "has_circuit_breakers": bool(summary.get("has_circuit_breakers")),
        "has_rate_limits": bool(summary.get("has_rate_limits")),
        "has_budgets": bool(summary.get("has_budgets")),
        "has_audit": bool(summary.get("has_audit")),
        "has_containment": bool(summary.get("has_containment")),
        "has_drift_detection": bool(summary.get("has_drift_detection")),
    }


def _probe_framework_trace_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    if not trace:
        return {}
    summary = (
        dict(trace.get("summary") or {})
        if isinstance(trace.get("summary"), Mapping)
        else {}
    )
    for count_key, trace_key in (
        ("span_count", "spans"),
        ("event_count", "events"),
        ("checkpoint_count", "checkpoints"),
        ("session_count", "sessions"),
    ):
        if count_key not in summary and isinstance(trace.get(trace_key), list):
            summary[count_key] = len(trace.get(trace_key, []))
        elif count_key not in summary and trace.get(count_key) is not None:
            summary[count_key] = trace.get(count_key)
    for key in ("signals", "tool_names"):
        if trace.get(key):
            summary[key] = sorted(str(item) for item in trace.get(key, []) if str(item))
    adapter_conformance = (
        dict(trace.get("adapter_conformance"))
        if isinstance(trace.get("adapter_conformance"), Mapping)
        else {}
    )
    if adapter_conformance:
        summary["adapter_conformance_passed"] = bool(
            adapter_conformance.get("passed")
        )
        findings = _probe_mappings(adapter_conformance.get("findings"))
        summary["adapter_conformance_finding_count"] = len(findings)
        required_signals = adapter_conformance.get("required_signals")
        if required_signals:
            summary["adapter_required_signals"] = sorted(
                str(item) for item in _probe_list(required_signals) if str(item)
            )
    spans = _probe_mappings(trace.get("spans"))
    events = _probe_mappings(trace.get("events"))
    if spans and "span_names" not in summary:
        summary["span_names"] = sorted(
            {
                str(span.get("name") or span.get("id") or "")
                for span in spans
                if span.get("name") or span.get("id")
            }
        )
    if events and "event_names" not in summary:
        summary["event_names"] = sorted(
            {
                str(event.get("name") or event.get("id") or event.get("type") or "")
                for event in events
                if event.get("name") or event.get("id") or event.get("type")
            }
        )
    return summary


def _probe_message_history_summary(value: Any) -> dict[str, Any]:
    history = dict(value or {}) if isinstance(value, Mapping) else {}
    if not history:
        return {}
    messages = _probe_mappings(history.get("messages"))
    summary: dict[str, Any] = {}
    for key in (
        "message_count",
        "tool_call_count",
        "tool_response_count",
        "handoff_count",
    ):
        if history.get(key) not in (None, "", [], {}):
            summary[key] = history.get(key)
    for key in ("roles", "sources", "types", "tool_names"):
        values = _probe_list(history.get(key))
        if values:
            summary[key] = sorted(str(item) for item in values if str(item))
    speaker_sequence = [
        str(
            message.get("source")
            or message.get("speaker")
            or message.get("role")
            or ""
        )
        for message in messages
        if message.get("source") or message.get("speaker") or message.get("role")
    ]
    if speaker_sequence:
        summary["speaker_sequence"] = speaker_sequence
    message_types = [
        str(message.get("type") or message.get("message_type") or "")
        for message in messages
        if message.get("type") or message.get("message_type")
    ]
    if message_types:
        summary["message_types"] = sorted(set(message_types))
    stop_reason = str(history.get("stop_reason") or "")
    if stop_reason:
        summary["stop_reason"] = stop_reason
    last_content = str(history.get("last_content") or "")
    if last_content:
        summary["last_content"] = last_content
    handoffs = _probe_mappings(history.get("handoffs"))
    if handoffs:
        summary["handoffs"] = [
            {
                key: str(handoff.get(key) or "")
                for key in ("from", "to", "task")
                if handoff.get(key) not in (None, "", [], {})
            }
            for handoff in handoffs
        ]
    return summary


def _probe_framework_handoff_summary(value: Any) -> dict[str, Any]:
    coordination = dict(value or {}) if isinstance(value, Mapping) else {}
    if not coordination:
        return {}
    summary: dict[str, Any] = {}
    for key in ("handoff_count", "review_count", "reconciliation_count"):
        if coordination.get(key) not in (None, "", [], {}):
            summary[key] = coordination.get(key)
    participants = _probe_list(coordination.get("participants"))
    if participants:
        summary["participants"] = sorted(str(item) for item in participants if str(item))
    handoffs = _probe_mappings(coordination.get("handoffs"))
    if handoffs:
        summary["handoffs"] = [
            {
                key: str(handoff.get(key) or "")
                for key in ("from", "to", "task", "reason", "message_type")
                if handoff.get(key) not in (None, "", [], {})
            }
            for handoff in handoffs
        ]
    reviews = _probe_mappings(coordination.get("reviews"))
    if reviews:
        summary["reviews"] = [
            {
                key: str(review.get(key) or "")
                for key in ("reviewer", "target", "status", "message_type")
                if review.get(key) not in (None, "", [], {})
            }
            for review in reviews
        ]
    reconciliations = _probe_mappings(coordination.get("reconciliations"))
    if reconciliations:
        summary["reconciliations"] = [
            {
                key: str(reconciliation.get(key) or "")
                for key in (
                    "source",
                    "accepted_source",
                    "status",
                    "message_type",
                )
                if reconciliation.get(key) not in (None, "", [], {})
            }
            for reconciliation in reconciliations
        ]
    summary["has_handoffs"] = bool(handoffs)
    summary["has_reviews"] = bool(reviews)
    summary["has_reconciliation"] = bool(reconciliations)
    return summary


def _probe_orchestration_trace_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    summary = dict(trace.get("summary") or {}) if isinstance(trace.get("summary"), Mapping) else {}
    if trace.get("signals"):
        summary["signals"] = sorted(str(signal) for signal in trace.get("signals", []) if str(signal))
    if trace.get("nodes"):
        summary["node_names"] = sorted(
            {
                str(dict(node).get("name") or dict(node).get("id") or "")
                for node in trace.get("nodes", [])
                if isinstance(node, Mapping)
                and (dict(node).get("name") or dict(node).get("id"))
            }
        )
    for key, trace_key in (
        ("node_count", "nodes"),
        ("edge_count", "edges"),
        ("step_count", "steps"),
    ):
        if key not in summary and isinstance(trace.get(trace_key), list):
            summary[key] = len(trace.get(trace_key, []))
    return summary


def _probe_workflow_trace_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    if not trace:
        return {}
    summary = (
        dict(trace.get("summary") or {})
        if isinstance(trace.get("summary"), Mapping)
        else {}
    )
    for count_key, trace_key in (
        ("node_count", "nodes"),
        ("edge_count", "edges"),
        ("step_count", "steps"),
        ("checkpoint_count", "checkpoints"),
        ("route_decision_count", "route_decisions"),
        ("interrupt_count", "interrupts"),
        ("replay_count", "replay"),
        ("write_count", "writes"),
        ("state_snapshot_count", "state_snapshots"),
    ):
        if count_key not in summary and isinstance(trace.get(trace_key), list):
            summary[count_key] = len(trace.get(trace_key, []))
        elif count_key not in summary and trace.get(count_key) is not None:
            summary[count_key] = trace.get(count_key)
    if "tool_call_count" not in summary and trace.get("tool_call_count") is not None:
        summary["tool_call_count"] = trace.get("tool_call_count")

    nodes = _probe_mappings(trace.get("nodes"))
    steps = _probe_mappings(trace.get("steps"))
    checkpoints = _probe_mappings(trace.get("checkpoints"))
    routes = _probe_mappings(trace.get("route_decisions"))
    interrupts = _probe_mappings(trace.get("interrupts"))
    replay = _probe_mappings(trace.get("replay"))
    topology = dict(trace.get("topology") or {}) if isinstance(trace.get("topology"), Mapping) else {}
    final_state = (
        dict(trace.get("final_state"))
        if isinstance(trace.get("final_state"), Mapping)
        else {}
    )

    node_names = [
        node.get("name") or node.get("id")
        for node in nodes
        if node.get("name") or node.get("id")
    ]
    step_names = [
        step.get("name") or step.get("node") or step.get("id")
        for step in steps
        if step.get("name") or step.get("node") or step.get("id")
    ]
    tool_names = [
        *[str(item) for item in _probe_list(trace.get("tool_names")) if str(item)],
        *[
            str(call.get("name") or call.get("tool") or "")
            for step in steps
            for call in _probe_mappings(step.get("tool_calls"))
            if call.get("name") or call.get("tool")
        ],
    ]
    final_state_keys = trace.get("final_state_keys") or list(final_state.keys())

    for key, values in (
        ("node_names", node_names),
        ("step_names", step_names),
        ("checkpoint_ids", [item.get("id") or item.get("checkpoint_id") for item in checkpoints]),
        ("route_targets", [item.get("target") or item.get("selected") for item in routes]),
        ("interrupt_nodes", [item.get("node") or item.get("id") for item in interrupts]),
        ("replay_ids", [item.get("id") or item.get("replay_id") for item in replay]),
        ("tool_names", tool_names),
        ("step_statuses", trace.get("step_statuses")),
        ("final_state_keys", final_state_keys),
    ):
        cleaned = sorted({str(item) for item in _probe_list(values) if str(item)})
        if cleaned:
            summary[key] = cleaned

    for key in ("entry_nodes", "terminal_nodes"):
        values = sorted(str(item) for item in _probe_list(topology.get(key)) if str(item))
        if values:
            summary[key] = values
    for key in ("has_replay", "has_interrupts", "has_routes"):
        if trace.get(key) is not None:
            summary[key] = bool(trace.get(key))
    if topology:
        summary["has_topology"] = True
    if trace.get("framework"):
        summary["framework"] = str(trace.get("framework"))
    return summary


def _probe_realtime_trace_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    summary = (
        dict(trace.get("summary") or {})
        if isinstance(trace.get("summary"), Mapping)
        else {}
    )
    for key in (
        "signals",
        "tool_names",
        "frame_types",
        "event_types",
        "categories",
        "directions",
        "modalities",
    ):
        if trace.get(key):
            summary[key] = sorted(str(item) for item in trace.get(key, []) if str(item))
    for count_key, value_key in (
        ("frame_count", "frames"),
        ("event_count", "events"),
    ):
        if count_key not in summary and isinstance(trace.get(value_key), list):
            summary[count_key] = len(trace.get(value_key, []))
    return summary


def _probe_framework_memory_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    if not trace:
        return {}
    summary = (
        dict(trace.get("summary") or {})
        if isinstance(trace.get("summary"), Mapping)
        else {}
    )
    for count_key, value_key in (
        ("operation_count", "operations"),
        ("checkpoint_count", "checkpoints"),
        ("memory_count", "memories"),
        ("retrieval_count", "retrievals"),
        ("store_count", "stores"),
    ):
        if count_key not in summary and isinstance(trace.get(value_key), list):
            summary[count_key] = len(trace.get(value_key, []))
        elif count_key not in summary and trace.get(count_key) is not None:
            summary[count_key] = trace.get(count_key)
    policies = dict(trace.get("policies") or {}) if isinstance(trace.get("policies"), Mapping) else {}
    if "policy_count" not in summary:
        summary["policy_count"] = len(policies) or trace.get("policy_count", 0)
    for key in ("operation_types", "source_ids", "namespaces", "retrieval_doc_ids"):
        if trace.get(key):
            summary[key] = sorted(str(item) for item in trace.get(key, []) if str(item))
    policy_keys = trace.get("policy_keys") or list(policies.keys())
    if policy_keys:
        summary["policy_keys"] = sorted(str(item) for item in policy_keys if str(item))
    return summary


def _probe_retrieval_memory_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    if not trace:
        return {}
    summary = (
        dict(trace.get("summary") or {})
        if isinstance(trace.get("summary"), Mapping)
        else {}
    )
    documents = [
        dict(item)
        for item in (trace.get("documents") or [])
        if isinstance(item, Mapping)
    ]
    queries = [
        dict(item)
        for item in (trace.get("queries") or [])
        if isinstance(item, Mapping)
    ]
    citations = [
        dict(item)
        for item in (trace.get("citations") or [])
        if isinstance(item, Mapping)
    ]
    memory_writes = [
        dict(item)
        for item in (trace.get("memory_writes") or [])
        if isinstance(item, Mapping)
    ]
    summary.setdefault("document_count", len(documents))
    summary.setdefault("query_count", len(queries))
    summary.setdefault("citation_count", len(citations))
    summary.setdefault("memory_write_count", len(memory_writes))
    summary.setdefault(
        "current_document_count",
        len([document for document in documents if document.get("current") is not False]),
    )
    doc_ids = sorted(
        {
            str(
                document.get("id")
                or document.get("doc_id")
                or document.get("source")
                or ""
            )
            for document in documents
            if document.get("id") or document.get("doc_id") or document.get("source")
        }
    )
    cited_doc_ids = sorted(
        {
            str(doc_id)
            for citation in citations
            for doc_id in _probe_list(citation.get("doc_ids"))
            if str(doc_id)
        }
    )
    if doc_ids:
        summary["document_ids"] = doc_ids
    if cited_doc_ids:
        summary["citation_doc_ids"] = cited_doc_ids
    if trace.get("require_current") is not None:
        summary["require_current"] = bool(trace.get("require_current"))
    return summary


def _probe_agent_memory_lineage_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    if not trace:
        return {}
    summary = (
        dict(trace.get("summary") or {})
        if isinstance(trace.get("summary"), Mapping)
        else {}
    )
    stores = _probe_mappings(trace.get("stores"))
    memories = _probe_mappings(trace.get("memories"))
    operations = _probe_mappings(trace.get("operations"))
    lineage = _probe_mappings(trace.get("lineage"))
    policies = dict(trace.get("policies") or {}) if isinstance(trace.get("policies"), Mapping) else {}
    poison_tests = _probe_mappings(trace.get("poison_tests") or trace.get("poisoning_tests"))
    isolation_tests = _probe_mappings(trace.get("isolation_tests"))
    retention_tests = _probe_mappings(trace.get("retention_tests") or trace.get("deletion_tests"))
    observability = (
        dict(trace.get("observability"))
        if isinstance(trace.get("observability"), Mapping)
        else {}
    )
    artifacts = _probe_mappings(trace.get("artifacts"))
    operation_type_values = [
        operation_type
        for operation_type in (
            _probe_memory_key(
                operation.get("operation")
                or operation.get("type")
                or operation.get("op")
            )
            for operation in operations
        )
        if operation_type
    ]
    operation_types = sorted(set(operation_type_values))
    policy_keys = sorted(_probe_memory_key(key) for key in policies if _probe_memory_key(key))
    attributed_memories = [
        memory
        for memory in memories
        if _probe_list(
            memory.get("source_ids")
            or memory.get("sources")
            or memory.get("doc_ids")
        )
    ]
    unattributed_memories = [
        str(memory.get("id") or memory.get("key") or index)
        for index, memory in enumerate(memories, start=1)
        if not _probe_list(
            memory.get("source_ids")
            or memory.get("sources")
            or memory.get("doc_ids")
        )
        and str(memory.get("status") or "active").lower() not in {"deleted", "expired", "blocked"}
    ]
    poison_good = {"passed", "blocked", "mitigated", "contained", "accepted"}
    isolation_good = {"passed", "blocked", "mitigated", "contained"}
    retention_good = {"passed", "deleted", "expired", "purged", "mitigated"}
    poisoning_failures = [
        test
        for test in poison_tests
        if _probe_memory_key(test.get("status")) not in poison_good
    ]
    isolation_violations = [
        test
        for test in isolation_tests
        if _probe_memory_key(test.get("status")) not in isolation_good
    ]
    retention_violations = [
        test
        for test in retention_tests
        if _probe_memory_key(test.get("status")) not in retention_good
    ]
    policy_violations = [
        operation
        for operation in operations
        if _probe_memory_key(operation.get("status")) in {"policy_violation", "violation", "failed_policy"}
        or _probe_memory_key(operation.get("policy_decision")) in {"violation", "failed", "bypassed"}
    ]
    summary.setdefault("store_count", len(stores))
    summary.setdefault("memory_count", len(memories))
    summary.setdefault("operation_count", len(operations))
    summary.setdefault("lineage_count", len(lineage))
    summary.setdefault("policy_count", len(policies))
    summary.setdefault("artifact_count", len(artifacts))
    summary.setdefault("observability_hook_count", _probe_observability_count(observability))
    summary.setdefault("attributed_memory_count", len(attributed_memories))
    summary.setdefault("unattributed_memory_count", len(unattributed_memories))
    summary.setdefault("poisoned_memory_count", 0)
    summary.setdefault("open_poisoning_count", len(poisoning_failures))
    summary.setdefault("isolation_violation_count", len(isolation_violations))
    summary.setdefault("retention_violation_count", len(retention_violations))
    summary.setdefault("policy_violation_count", len(policy_violations))
    for operation_type in ("read", "write", "update", "delete", "recall"):
        summary.setdefault(
            f"{operation_type}_operation_count",
            operation_type_values.count(operation_type),
        )
    summary.setdefault("has_target", bool(trace.get("target")))
    summary.setdefault("has_stores", bool(stores))
    summary.setdefault("has_memory_records", bool(memories))
    summary.setdefault("has_operations", bool(operations))
    summary.setdefault("has_lineage", bool(lineage))
    summary.setdefault("has_source_attribution", bool(attributed_memories) and not unattributed_memories)
    summary.setdefault(
        "has_tenant_isolation",
        "tenant_isolation" in policy_keys or bool(isolation_tests),
    )
    summary.setdefault("has_audit", "audit" in policy_keys)
    summary.setdefault(
        "has_retention_policy",
        any(key in policy_keys for key in ("retention", "retention_policy")),
    )
    summary.setdefault(
        "has_deletion_policy",
        any(key in policy_keys for key in ("deletion", "deletion_policy")),
    )
    summary.setdefault("has_redaction", "redaction" in policy_keys)
    summary.setdefault("has_canaries", "canary" in policy_keys or bool(poison_tests))
    summary.setdefault("has_observability", bool(observability))
    summary.setdefault("has_artifacts", bool(artifacts))
    if operation_types:
        summary["operation_types"] = operation_types
    if policy_keys:
        summary["policy_keys"] = policy_keys
    observed_evidence = {
        signal
        for flag, signal in (
            ("has_target", "target"),
            ("has_stores", "store"),
            ("has_memory_records", "memory_record"),
            ("has_operations", "operation"),
            ("has_lineage", "lineage"),
            ("has_source_attribution", "source_attribution"),
            ("has_tenant_isolation", "tenant_isolation"),
            ("has_audit", "audit"),
            ("has_retention_policy", "retention_policy"),
            ("has_deletion_policy", "deletion_policy"),
            ("has_redaction", "redaction"),
            ("has_canaries", "canary"),
            ("has_observability", "observability"),
            ("has_artifacts", "artifact"),
        )
        if summary.get(flag)
    }
    observed_evidence.update(f"{operation_type}_operation" for operation_type in operation_types)
    observed_signals = {
        *observed_evidence,
        *operation_types,
        *policy_keys,
        "agent_memory_lineage",
        "memory_lineage",
        "memory_provenance",
        "memory",
        "provenance",
    }
    summary.setdefault("observed_evidence", sorted(observed_evidence))
    summary.setdefault("observed_signals", sorted(observed_signals))
    summary.setdefault("blocking_gap_count", 0)
    return summary


def _probe_browser_cua_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    if not trace:
        return {}
    summary = (
        dict(trace.get("summary") or {})
        if isinstance(trace.get("summary"), Mapping)
        else {}
    )
    snapshots = _probe_mappings(trace.get("snapshots"))
    actions = _probe_mappings(trace.get("action_replay") or trace.get("actions"))
    screenshots = _probe_mappings(trace.get("screenshots"))
    regions = dict(trace.get("regions") or {}) if isinstance(trace.get("regions"), Mapping) else {}
    network_log = _probe_mappings(trace.get("network_log"))
    runtime_events = _probe_mappings(trace.get("runtime_events"))
    performance_entries = _probe_mappings(trace.get("performance_entries"))
    prompt_injections = _probe_mappings(trace.get("prompt_injections"))
    mutation_pack = (
        dict(trace.get("mutation_pack"))
        if isinstance(trace.get("mutation_pack"), Mapping)
        else {}
    )
    mutations = _probe_mappings(
        mutation_pack.get("mutations")
        or trace.get("browser_mutations")
        or trace.get("mutations")
    )
    for count_key, count in (
        ("snapshot_count", len(snapshots)),
        ("action_count", len(actions)),
        ("screenshot_count", len(screenshots)),
        ("region_count", len(regions)),
        ("network_request_count", len(network_log)),
        ("runtime_event_count", len(runtime_events)),
        ("performance_entry_count", len(performance_entries)),
        ("prompt_injection_surface_count", len(prompt_injections)),
        ("mutation_count", len(mutations)),
    ):
        summary.setdefault(count_key, trace.get(count_key, count))
    for count_key in (
        "successful_action_count",
        "blocked_action_count",
        "matched_action_count",
        "prompt_injection_touched_count",
        "screenshot_diff_count",
    ):
        if trace.get(count_key) is not None:
            summary[count_key] = trace.get(count_key)
    summary.setdefault(
        "stale_action_count",
        len([action for action in actions if action.get("stale_screenshot")]),
    )
    summary.setdefault(
        "dom_snapshot_count",
        len([snapshot for snapshot in snapshots if snapshot.get("dom")]),
    )
    summary.setdefault(
        "screenshot_snapshot_count",
        len(
            [
                snapshot
                for snapshot in snapshots
                if snapshot.get("screenshot_uri") or snapshot.get("screenshot_path")
            ]
        ),
    )
    summary["layout_shift_present"] = bool(
        trace.get("layout_shift_present")
        or trace.get("layout_shift_distribution")
    )
    summary["storage_present"] = bool(trace.get("storage_present"))
    action_types = trace.get("action_types") or [
        action.get("action") or action.get("type")
        for action in actions
        if action.get("action") or action.get("type")
    ]
    tool_names = trace.get("tool_names") or [
        action.get("tool") or action.get("tool_name")
        for action in actions
        if action.get("tool") or action.get("tool_name")
    ]
    mutation_ids = [
        mutation.get("id") or mutation.get("name")
        for mutation in mutations
        if mutation.get("id") or mutation.get("name")
    ]
    mutation_types = [
        mutation.get("type") or mutation.get("kind")
        for mutation in mutations
        if mutation.get("type") or mutation.get("kind")
    ]
    region_ids = [
        *list(regions.keys()),
        *[
            region.get("id") or region.get("name")
            for region in regions.values()
            if isinstance(region, Mapping) and (region.get("id") or region.get("name"))
        ],
    ]
    prompt_injection_ids = [
        injection.get("id") or injection.get("selector")
        for injection in prompt_injections
        if injection.get("id") or injection.get("selector")
    ]
    summary["action_types"] = sorted(str(item) for item in action_types if str(item))
    summary["tool_names"] = sorted(str(item) for item in tool_names if str(item))
    summary["mutation_ids"] = sorted(str(item) for item in mutation_ids if str(item))
    summary["mutation_types"] = sorted(str(item) for item in mutation_types if str(item))
    summary["region_ids"] = sorted(str(item) for item in region_ids if str(item))
    summary["prompt_injection_ids"] = sorted(
        str(item) for item in prompt_injection_ids if str(item)
    )
    layout_distribution = (
        dict(trace.get("layout_shift_distribution"))
        if isinstance(trace.get("layout_shift_distribution"), Mapping)
        else {}
    )
    layout_values = [
        _probe_float(layout_distribution.get(key))
        for key in ("max", "p95", "score", "value")
    ]
    layout_values = [value for value in layout_values if value is not None]
    if layout_values:
        summary["max_layout_shift_score"] = max(layout_values)
    performance_durations = [
        _probe_float(entry.get("duration_ms") or entry.get("duration"))
        for entry in performance_entries
    ]
    performance_durations = [value for value in performance_durations if value is not None]
    if performance_durations:
        summary["max_performance_duration_ms"] = max(performance_durations)
    compact_actions = []
    for action in actions[:5]:
        compact_action = {
            key: action.get(key)
            for key in (
                "id",
                "tool",
                "tool_name",
                "action",
                "selector",
                "success",
                "matched",
                "blocked",
                "mutation_id",
                "mutation_type",
            )
            if action.get(key) not in (None, "", [], {})
        }
        region = action.get("region")
        if isinstance(region, Mapping):
            compact_action["region"] = {
                key: region.get(key)
                for key in ("id", "name", "selector", "x", "y", "width", "height")
                if region.get(key) not in (None, "", [], {})
            }
        if compact_action:
            compact_actions.append(compact_action)
    if compact_actions:
        summary["actions"] = compact_actions
    if runtime_events:
        summary["runtime_events"] = [
            {
                key: event.get(key)
                for key in ("id", "name", "type", "level", "message")
                if event.get(key) not in (None, "", [], {})
            }
            for event in runtime_events[:5]
        ]
    return summary


def _probe_mcp_tool_session_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    summary = trace.get("summary")
    return dict(summary) if isinstance(summary, Mapping) else {}


def _probe_a2a_protocol_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    summary = trace.get("summary")
    return dict(summary) if isinstance(summary, Mapping) else {}


def _probe_openenv_summary(value: Any) -> dict[str, Any]:
    trace = dict(value or {}) if isinstance(value, Mapping) else {}
    if not trace:
        return {}
    summary = (
        dict(trace.get("summary") or {})
        if isinstance(trace.get("summary"), Mapping)
        else {}
    )
    trajectory = _probe_mappings(trace.get("trajectory") or trace.get("steps"))
    action_log = _probe_mappings(trace.get("action_log") or trace.get("actions"))
    error_log = _probe_mappings(trace.get("error_log") or trace.get("errors"))
    sandbox = (
        dict(trace.get("sandbox"))
        if isinstance(trace.get("sandbox"), Mapping)
        else {}
    )
    for count_key, fallback in (
        ("reset_count", 1 if trace.get("initial_observation") is not None else 0),
        ("step_count", len(trajectory)),
        ("action_route_count", len(action_log) or len(trajectory)),
        (
            "failure_count",
            max(
                len(_probe_mappings(trace.get("failure_injections") or trace.get("faults"))),
                sum(
                    1
                    for step in trajectory
                    if step.get("failure_injected") or step.get("failure")
                ),
            ),
        ),
        (
            "metadata_capture_count",
            sum(
                1
                for step in trajectory
                if step.get("metadata") not in (None, "", [], {})
                or step.get("info") not in (None, "", [], {})
            )
            + (1 if trace.get("reset_info") else 0),
        ),
        ("error_count", len(error_log)),
    ):
        if summary.get(count_key) in (None, "", [], {}):
            summary[count_key] = fallback
    if summary.get("reward_total") in (None, "", [], {}):
        summary["reward_total"] = round(
            sum(_probe_float(step.get("reward")) or 0.0 for step in trajectory),
            4,
        )
    for key in ("done", "terminated", "truncated"):
        if summary.get(key) in (None, "", [], {}):
            summary[key] = any(bool(step.get(key)) for step in trajectory)
    if summary.get("sandbox_enabled") in (None, "", [], {}):
        summary["sandbox_enabled"] = bool(sandbox.get("enabled", bool(sandbox)))
    if summary.get("isolation") in (None, "", [], {}):
        summary["isolation"] = str(sandbox.get("isolation") or "process")
    for key in (
        "runtime",
        "transport",
        "requires_external_service",
        "deterministic_reset",
    ):
        if summary.get(key) in (None, "", [], {}) and trace.get(key) not in (
            None,
            "",
            [],
            {},
        ):
            summary[key] = trace.get(key)
    signals = trace.get("signals")
    if signals and summary.get("signals") in (None, "", [], {}):
        summary["signals"] = sorted(str(signal) for signal in signals if str(signal))
    return summary


def _probe_artifact_evidence(artifact: Mapping[str, Any]) -> dict[str, Any]:
    metadata = artifact.get("metadata")
    return {
        "type": str(artifact.get("type") or ""),
        "uri": str(artifact.get("uri") or ""),
        "path": str(artifact.get("path") or ""),
        "mime_type": str(artifact.get("mime_type") or ""),
        "role": str(artifact.get("role") or ""),
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }


def _probe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _probe_mappings(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in _probe_list(value) if isinstance(item, Mapping)]


def _probe_memory_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _probe_observability_count(observability: Mapping[str, Any]) -> int:
    count = 0
    for value in observability.values():
        if isinstance(value, Mapping):
            count += len(value)
        elif isinstance(value, (list, tuple, set)):
            count += len([item for item in value if item])
        elif value:
            count += 1
    return count


def _probe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _probe_summary(
    cases: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    passed = sum(1 for case in cases if case.get("passed"))
    failed = len(cases) - passed
    response_tool_count = sum(
        int(dict(case.get("response") or {}).get("tool_call_count") or 0)
        for case in cases
    )
    runtime_trace_count = sum(1 for case in cases if case.get("runtime_trace"))
    streaming_trace_count = sum(
        1 for case in cases if dict(case.get("response") or {}).get("streaming")
    )
    observed_io_contracts = [
        dict(case.get("observed_io_contract") or {})
        for case in cases
        if isinstance(case.get("observed_io_contract"), Mapping)
    ]
    observed_io_contract_count = sum(
        1
        for item in observed_io_contracts
        if item.get("kind")
        == "agent-learning.framework-adapter-observed-io-contract.v1"
        and int(dict(item.get("summary") or {}).get("invocation_count") or 0) >= 1
    )
    signature_bound_count = sum(
        1
        for item in observed_io_contracts
        if dict(item.get("summary") or {}).get("signature_bound") is True
    )
    call_contract_count = sum(
        1
        for case in cases
        for invocation in dict(case.get("runtime_trace") or {}).get("invocations", [])
        if isinstance(invocation, Mapping)
        and dict(invocation.get("call_contract") or {}).get("kind")
        == "agent-learning.framework-adapter-call-contract.v1"
    )
    input_keys = sorted(
        {
            str(invocation.get("input_key"))
            for case in cases
            for invocation in dict(case.get("runtime_trace") or {}).get("invocations", [])
            if isinstance(invocation, Mapping)
            and invocation.get("input_key") not in (None, "", [], {})
        }
    )
    call_styles = sorted(
        {
            str(invocation.get("call_style"))
            for case in cases
            for invocation in dict(case.get("runtime_trace") or {}).get("invocations", [])
            if isinstance(invocation, Mapping)
            and invocation.get("call_style") not in (None, "", [], {})
        }
    )
    input_kwargs_keys = sorted(
        {
            str(key)
            for case in cases
            for invocation in dict(case.get("runtime_trace") or {}).get("invocations", [])
            if isinstance(invocation, Mapping)
            for key in invocation.get("input_kwargs_keys", [])
            if key not in (None, "", [], {})
        }
    )
    input_types = _unique_probe_strings(
        input_type
        for item in observed_io_contracts
        for input_type in list(dict(item.get("summary") or {}).get("input_types") or [])
    )
    output_types = _unique_probe_strings(
        output_type
        for item in observed_io_contracts
        for output_type in list(dict(item.get("summary") or {}).get("output_types") or [])
    )
    return {
        "case_count": len(cases),
        "passed_case_count": passed,
        "failed_case_count": failed,
        "runtime_trace_count": runtime_trace_count,
        "streaming_trace_count": streaming_trace_count,
        "call_contract_count": call_contract_count,
        "callable_signature_present": bool(contract.get("callable_signature")),
        "observed_io_contract_count": observed_io_contract_count,
        "signature_bound_count": signature_bound_count,
        "tool_call_count": response_tool_count,
        "framework": contract.get("framework"),
        "method": contract.get("method"),
        "input_mode": contract.get("input_mode"),
        "input_keys": input_keys,
        "input_types": input_types,
        "output_types": output_types,
        "call_styles": call_styles,
        "input_kwargs_keys": input_kwargs_keys,
        "local_executable_fixture": bool(contract.get("local_executable_fixture")),
        "requires_external_service": bool(contract.get("requires_external_service")),
        "trace_runtime": bool(contract.get("trace_runtime")),
    }
