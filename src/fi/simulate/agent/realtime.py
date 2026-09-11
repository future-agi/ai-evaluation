from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse

from fi.simulate.environment import StreamingTraceEnvironment, VoiceEnvironment


def realtime_stack_contract(
    *,
    target: str | None = None,
    framework: str = "livekit",
    metadata: Optional[Dict[str, Any]] = None,
    external_sources: Sequence[str] = (),
) -> dict[str, Any]:
    """Return an import-free local contract for a realtime voice stack."""

    target_scheme = urlparse(str(target or "")).scheme.lower()
    external_source_list = _unique_strings(external_sources)
    requires_external = target_scheme in {"http", "https"} or bool(external_source_list)
    return {
        "kind": "agent-learning.realtime-stack-contract.v1",
        "runtime": "in_process",
        "framework": _scope_key(framework) or "realtime",
        "target": str(target) if target else "",
        "target_scheme": target_scheme,
        "requires_external_service": requires_external,
        "local_executable_fixture": not requires_external,
        "external_sources": external_source_list,
        "evidence_requirements": [
            "voice",
            "streaming_trace",
            "route_call",
            "transcript",
            "tts",
            "timing_distribution",
            "audio_quality",
            "stream_tool_delta",
            "completion",
            "trace_artifact",
        ],
        "metadata": _plain_mapping(metadata),
    }


def run_realtime_stack_probe(
    realtime: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility alias for the synchronous realtime stack probe."""

    return probe_realtime_stack(realtime=realtime, **kwargs)


def probe_realtime_stack(
    *,
    realtime: Mapping[str, Any],
    agent: Optional[Mapping[str, Any]] = None,
    framework: str = "livekit",
    target: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    allow_external_target: bool = False,
    expected_route: str | None = None,
    min_sample_rate_hz: int = 16000,
    min_timing_stage_count: int = 4,
) -> dict[str, Any]:
    """Probe local realtime voice + streaming coordination evidence."""

    if target and _is_external_target(target) and not allow_external_target:
        raise ValueError(
            "external targets are disabled for realtime stack probes; "
            "set allow_external_target=True only when the user explicitly "
            "wants to test that live workload"
        )
    realtime_data = _realtime_data(realtime, framework=framework)
    external_sources = _external_sources(realtime_data)
    if external_sources and not allow_external_target:
        raise ValueError(
            "external export sources are disabled for realtime stack probes; "
            "set allow_external_target=True only when the user explicitly "
            "wants to test live exports"
        )
    contract = realtime_stack_contract(
        target=target,
        framework=framework,
        metadata=metadata,
        external_sources=external_sources,
    )

    voice_state: dict[str, Any] = {}
    streaming_state: dict[str, Any] = {}
    if realtime_data.get("voice") is not None:
        voice_environment = _voice_environment(_plain_mapping(realtime_data["voice"]))
        voice_environment.reset()
    else:
        voice_environment = None
    if realtime_data.get("streaming_trace") is not None:
        streaming_environment = _streaming_environment(
            _plain_mapping(realtime_data["streaming_trace"]),
            framework=str(framework),
        )
        streaming_environment.reset()
    else:
        streaming_environment = None

    active_agent = agent or _default_realtime_probe_agent(
        realtime_data,
        expected_route=expected_route,
    )
    tool_calls = _agent_tool_calls(active_agent)
    successful_tool_calls = 0
    for tool_call in tool_calls:
        for environment in (voice_environment, streaming_environment):
            if environment is None:
                continue
            result = environment.handle_tool_call(tool_call)
            if result is not None and result.success:
                successful_tool_calls += 1
    if voice_environment is not None:
        voice_state = voice_environment._state_payload()
    if streaming_environment is not None:
        streaming_state = streaming_environment._state_payload()

    summary = _realtime_probe_summary(
        voice_state,
        streaming_state,
        contract=contract,
        tool_calls=tool_calls,
        successful_tool_calls=successful_tool_calls,
        expected_route=expected_route,
        min_sample_rate_hz=min_sample_rate_hz,
        min_timing_stage_count=min_timing_stage_count,
    )
    findings = _realtime_probe_findings(summary, contract=contract)
    summary["finding_count"] = len(findings)
    summary["passed_case_count"] = 1 if not findings else 0
    summary["failed_case_count"] = 0 if not findings else 1
    status = "passed" if not findings else "failed"
    return {
        "kind": "agent-learning.realtime-stack-probe.v1",
        "status": status,
        "passed": status == "passed",
        "requires_external_service": bool(contract["requires_external_service"]),
        "allow_external_target": bool(allow_external_target),
        "contract": contract,
        "summary": summary,
        "realtime": realtime_data,
        "environments": copy.deepcopy(realtime_data["environments"]),
        "state": {
            "voice": copy.deepcopy(voice_state),
            "streaming_trace": copy.deepcopy(streaming_state),
        },
        "findings": findings,
        "metadata": {
            "source": "fi.simulate.agent.realtime.probe_realtime_stack",
            **_plain_mapping(metadata),
        },
    }


def _realtime_data(
    realtime: Mapping[str, Any],
    *,
    framework: str,
) -> dict[str, Any]:
    source = copy.deepcopy(dict(realtime or {}))
    voice_data = source.pop("voice", source.pop("voice_trace", None))
    streaming_data = source.pop(
        "streaming_trace",
        source.pop("streaming", None),
    )
    explicit_environments = source.pop("environments", None)
    if explicit_environments is not None:
        for environment in _plain_list(explicit_environments):
            item = _plain_mapping(environment)
            env_type = _scope_key(item.get("type"))
            data = _plain_mapping(item.get("data")) or {
                key: value for key, value in item.items() if key not in {"type", "kind"}
            }
            if env_type == "voice":
                voice_data = data
            elif env_type == "streaming_trace":
                streaming_data = data
    framework_key = _scope_key(source.pop("framework", framework)) or "realtime"
    if source:
        raise ValueError(
            "realtime candidate keys must be environments, voice, voice_trace, "
            "streaming_trace, streaming, or framework"
        )
    if voice_data is None and streaming_data is None:
        raise ValueError("realtime candidate must define voice or streaming_trace")
    result: dict[str, Any] = {
        "framework": framework_key,
        "environments": [],
    }
    if voice_data is not None:
        voice = copy.deepcopy(dict(voice_data))
        voice.setdefault("framework", framework_key)
        result["voice"] = voice
        result["environments"].append({"type": "voice", "data": copy.deepcopy(voice)})
    if streaming_data is not None:
        streaming = copy.deepcopy(dict(streaming_data))
        streaming.setdefault("framework", framework_key)
        result["streaming_trace"] = streaming
        result["environments"].append(
            {"type": "streaming_trace", "data": copy.deepcopy(streaming)}
        )
    return result


def _voice_environment(data: Mapping[str, Any]) -> VoiceEnvironment:
    source = dict(data)
    return VoiceEnvironment(
        utterances=_plain_list(source.get("utterances") or source.get("transcripts")),
        audio_uris=_unique_strings(source.get("audio_uris") or source.get("audio")),
        sample_rate_hz=_as_int(source.get("sample_rate_hz") or source.get("sample_rate") or 16000),
        stt_latency_ms=_as_int(source.get("stt_latency_ms") or 180),
        tts_latency_ms=_as_int(source.get("tts_latency_ms") or 320),
        state=_plain_mapping(source.get("state")),
        event_replay=_plain_list(source.get("event_replay") or source.get("events")),
        frame_replay=_plain_list(source.get("frame_replay") or source.get("frames")),
        latency_profile=_plain_mapping(source.get("latency_profile")),
        timing_distribution=_plain_mapping(
            source.get("timing_distribution")
            or source.get("timing")
            or source.get("latency_distribution")
        ),
        noise_profile=_plain_mapping(source.get("noise_profile") or source.get("noise")),
        allow_interruptions=bool(source.get("allow_interruptions", True)),
        interruption_policy=_plain_mapping(source.get("interruption_policy")),
        routes=source.get("routes"),
        initial_route=str(source.get("initial_route") or "") or None,
        voice_export=source.get("voice_export") or source.get("export"),
        voice_export_source=source.get("voice_export_source")
        or source.get("export_source")
        or source.get("trace_source"),
        export_framework=str(source.get("export_framework") or source.get("framework") or "voice"),
        export_headers=_plain_mapping(source.get("export_headers") or source.get("headers")),
        export_auth=_plain_mapping(source.get("export_auth") or source.get("auth")),
        export_pagination=_plain_mapping(
            source.get("export_pagination") or source.get("pagination")
        ),
        export_max_pages=_as_int(source.get("export_max_pages") or source.get("max_pages") or 20),
        export_timeout=float(source.get("export_timeout") or source.get("timeout") or 30.0),
        waveforms=_plain_list(source.get("waveforms")),
        diarization=source.get("diarization") or source.get("speaker_segments"),
        perceptual_metrics=(
            source.get("perceptual_metrics")
            or source.get("audio_quality")
            or source.get("quality_profile")
        ),
    )


def _streaming_environment(
    data: Mapping[str, Any],
    *,
    framework: str,
) -> StreamingTraceEnvironment:
    source = dict(data)
    return StreamingTraceEnvironment(
        framework=str(source.get("framework") or source.get("provider") or framework),
        events=_plain_list(
            source.get("events")
            or source.get("stream_events")
            or source.get("chunks")
            or source.get("frames")
        ),
        trace_export=source.get("trace_export") or source.get("export"),
        export_source=source.get("export_source") or source.get("source"),
        export_headers=_plain_mapping(source.get("export_headers") or source.get("headers")),
        export_timeout=float(source.get("export_timeout") or source.get("timeout") or 30.0),
        state=_plain_mapping(source.get("state")),
        metadata=_plain_mapping(source.get("metadata")),
    )


def _default_realtime_probe_agent(
    realtime_data: Mapping[str, Any],
    *,
    expected_route: str | None,
) -> dict[str, Any]:
    voice = _plain_mapping(realtime_data.get("voice"))
    streaming = _plain_mapping(realtime_data.get("streaming_trace"))
    utterance_id = _first_item_id(
        _plain_list(voice.get("utterances") or voice.get("transcripts")),
        default="utt_refund",
    )
    stream_event_id = _first_stream_event_id(streaming, default="stream_tool_delta")
    route = (
        str(expected_route or "")
        or str(_plain_mapping(streaming.get("state")).get("route") or "")
        or str(voice.get("initial_route") or "")
        or "support"
    )
    responses: list[dict[str, Any]] = []
    first_turn: list[dict[str, Any]] = []
    second_turn: list[dict[str, Any]] = []
    if voice:
        first_turn.extend(
            [
                {"id": "voice_status", "name": "voice_status", "arguments": {}},
                {"id": "voice_timing", "name": "voice_timing", "arguments": {}},
                {
                    "id": "transcribe_user",
                    "name": "transcribe_audio",
                    "arguments": {"id": utterance_id},
                },
                {
                    "id": "route_support",
                    "name": "route_call",
                    "arguments": {"route": route, "reason": f"{route} route required"},
                },
            ]
        )
        second_turn.append(
            {
                "id": "speak_answer",
                "name": "speak",
                "arguments": {
                    "text": f"Your request has been routed to {route}.",
                    "latency_ms": _as_int(voice.get("tts_latency_ms") or 260),
                    "duration_ms": 1800,
                },
            }
        )
    if streaming:
        second_turn.extend(
            [
                {
                    "id": "stream_status",
                    "name": "streaming_trace_status",
                    "arguments": {},
                },
                {
                    "id": "stream_tool_events",
                    "name": "list_stream_events",
                    "arguments": {"signal": "tool_delta"},
                },
                {
                    "id": "inspect_stream_tool",
                    "name": "inspect_stream_event",
                    "arguments": {"id": stream_event_id},
                },
            ]
        )
    if first_turn:
        responses.append(
            {
                "content": "Inspecting realtime voice routing evidence.",
                "tool_calls": first_turn,
            }
        )
    if second_turn:
        responses.append(
            {
                "content": "Realtime voice and streaming evidence checked.",
                "tool_calls": second_turn,
            }
        )
    return {"type": "scripted", "responses": responses}


def _agent_tool_calls(agent: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not agent:
        return []
    calls: list[dict[str, Any]] = []
    for response in _plain_list(_plain_mapping(agent).get("responses")):
        for call in _plain_list(_plain_mapping(response).get("tool_calls")):
            item = _plain_mapping(call)
            if item:
                calls.append(item)
    return calls


def _realtime_probe_summary(
    voice_state: Mapping[str, Any],
    streaming_state: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    tool_calls: Sequence[Mapping[str, Any]],
    successful_tool_calls: int,
    expected_route: str | None,
    min_sample_rate_hz: int,
    min_timing_stage_count: int,
) -> dict[str, Any]:
    timing = _plain_mapping(voice_state.get("timing_distribution"))
    timing_stages = _plain_mapping(timing.get("stages"))
    perceptual = _plain_mapping(voice_state.get("perceptual_metrics"))
    audio = _plain_mapping(perceptual.get("overall"))
    webrtc = [_plain_mapping(item) for item in _plain_list(voice_state.get("webrtc_stats"))]
    if not audio and webrtc:
        audio = webrtc[0]
    streaming_summary = _plain_mapping(streaming_state.get("summary"))
    streaming_signals = _unique_strings(streaming_state.get("signals"))
    route = str(voice_state.get("current_route") or "")
    route_match = (
        route == str(expected_route)
        if expected_route
        else bool(route and _plain_list(voice_state.get("route_history")))
    )
    return {
        "case_count": 1,
        "passed_case_count": 0,
        "failed_case_count": 1,
        "finding_count": 0,
        "voice_present": bool(voice_state),
        "streaming_trace_present": bool(streaming_state),
        "sample_rate_hz": _as_int(voice_state.get("sample_rate_hz")),
        "min_sample_rate_hz": int(min_sample_rate_hz),
        "utterance_count": _as_int(voice_state.get("utterance_count")),
        "transcript_count": len(_plain_list(voice_state.get("transcript_history"))),
        "tts_count": len(_plain_list(voice_state.get("tts_history"))),
        "route_history_count": len(_plain_list(voice_state.get("route_history"))),
        "current_route": route,
        "expected_route": str(expected_route or ""),
        "route_match": route_match,
        "frame_count": len(_plain_list(voice_state.get("frame_replay"))),
        "speaker_count": len(
            _unique_strings(
                [
                    _plain_mapping(item).get("speaker")
                    for item in _plain_list(voice_state.get("timeline"))
                ]
            )
        ),
        "timing_stage_count": len(timing_stages),
        "min_timing_stage_count": int(min_timing_stage_count),
        "timing_sample_count": _as_int(timing.get("sample_count")),
        "snr_db": _as_float(audio.get("snr_db")),
        "mos": _as_float(audio.get("mos")),
        "clipping_ratio": _as_float(audio.get("clipping_ratio")),
        "jitter_ms": _as_float(audio.get("jitter_ms")),
        "packet_loss_pct": _as_float(audio.get("packet_loss_pct")),
        "streaming_event_count": _as_int(streaming_summary.get("event_count")),
        "streaming_chunk_count": _as_int(streaming_summary.get("chunk_count")),
        "streaming_tool_delta_count": _as_int(streaming_summary.get("tool_delta_count")),
        "streaming_dropped_event_count": _as_int(
            streaming_summary.get("dropped_event_count")
        ),
        "streaming_error_count": _as_int(streaming_summary.get("error_count")),
        "streaming_completion_status": _scope_key(
            streaming_summary.get("completion_status")
        ),
        "streaming_first_token_latency_ms": _as_float(
            streaming_summary.get("first_token_latency_ms")
        ),
        "streaming_max_gap_ms": _as_float(streaming_summary.get("max_gap_ms")),
        "streaming_signals": streaming_signals,
        "streaming_route": str(_plain_mapping(streaming_state.get("state")).get("route") or ""),
        "tool_call_count": len(tool_calls),
        "successful_tool_call_count": int(successful_tool_calls),
        "observed_tool_names": _unique_strings(
            [call.get("name") for call in tool_calls]
        ),
        "requires_external_service": bool(contract.get("requires_external_service")),
        "local_executable_fixture": bool(contract.get("local_executable_fixture")),
    }


def _realtime_probe_findings(
    summary: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    _append_finding(
        findings,
        "realtime_probe_local_contract",
        bool(summary["local_executable_fixture"])
        and not bool(summary["requires_external_service"]),
        "realtime probe target must be local and no-external-service",
        {"contract": dict(contract)},
    )
    _append_finding(
        findings,
        "realtime_probe_voice_trace",
        summary["voice_present"] is True
        and _as_int(summary.get("sample_rate_hz")) >= _as_int(summary.get("min_sample_rate_hz"))
        and _as_int(summary.get("utterance_count")) > 0
        and _as_int(summary.get("transcript_count")) > 0
        and _as_int(summary.get("tts_count")) > 0
        and _as_int(summary.get("frame_count")) > 0,
        "voice evidence must include transcript, TTS, audio frames, and sample-rate closure",
        summary,
    )
    _append_finding(
        findings,
        "realtime_probe_voice_timing_audio",
        _as_int(summary.get("timing_stage_count"))
        >= _as_int(summary.get("min_timing_stage_count"))
        and _as_float(summary.get("snr_db")) >= 20.0
        and _as_float(summary.get("mos")) >= 4.0
        and _as_float(summary.get("jitter_ms")) <= 40.0
        and _as_float(summary.get("packet_loss_pct")) <= 1.0
        and _as_float(summary.get("clipping_ratio")) <= 0.03,
        "voice timing and audio-quality evidence must meet realtime gates",
        summary,
    )
    _append_finding(
        findings,
        "realtime_probe_routing",
        summary["route_match"] is True
        and _as_int(summary.get("route_history_count")) > 0,
        "voice routing must reach the expected route",
        summary,
    )
    _append_finding(
        findings,
        "realtime_probe_streaming_trace",
        summary["streaming_trace_present"] is True
        and _as_int(summary.get("streaming_event_count")) > 0
        and _as_int(summary.get("streaming_chunk_count")) > 0
        and _as_int(summary.get("streaming_tool_delta_count")) > 0
        and _as_int(summary.get("streaming_dropped_event_count")) == 0
        and _as_int(summary.get("streaming_error_count")) == 0
        and summary.get("streaming_completion_status") in {"completed", "done"},
        "streaming evidence must include chunks, tool deltas, completion, and no drops/errors",
        summary,
    )
    required_tools = {
        "voice_status",
        "voice_timing",
        "transcribe_audio",
        "route_call",
        "speak",
        "streaming_trace_status",
        "list_stream_events",
        "inspect_stream_event",
    }
    observed_tools = set(_plain_list(summary.get("observed_tool_names")))
    _append_finding(
        findings,
        "realtime_probe_tool_evidence",
        required_tools.issubset(observed_tools)
        and _as_int(summary.get("successful_tool_call_count")) >= len(required_tools),
        "probe must exercise voice and streaming tools successfully",
        {"required_tools": sorted(required_tools), **dict(summary)},
    )
    for finding in findings:
        finding.setdefault("evidence", {})["summary"] = dict(summary)
    return findings


def _append_finding(
    findings: list[dict[str, Any]],
    check: str,
    passed: bool,
    message: str,
    evidence: Mapping[str, Any],
) -> None:
    if passed:
        return
    findings.append(
        {
            "check": check,
            "level": "error",
            "message": message,
            "evidence": dict(evidence),
        }
    )


def _first_item_id(values: Sequence[Any], *, default: str) -> str:
    for value in values:
        item = _plain_mapping(value)
        text = str(item.get("id") or item.get("name") or "")
        if text:
            return text
    return default


def _first_stream_event_id(streaming: Mapping[str, Any], *, default: str) -> str:
    for event in _plain_list(streaming.get("events")):
        item = _plain_mapping(event)
        signals = {_scope_key(signal) for signal in _plain_list(item.get("signals"))}
        event_type = _scope_key(item.get("type"))
        if "tool_delta" in signals or event_type == "tool_delta":
            return str(item.get("id") or item.get("event_id") or default)
    return _first_item_id(_plain_list(streaming.get("events")), default=default)


def _external_sources(realtime_data: Mapping[str, Any]) -> list[str]:
    sources: list[str] = []
    for key in (
        ("voice", "voice_export_source"),
        ("voice", "export_source"),
        ("voice", "trace_source"),
        ("streaming_trace", "export_source"),
        ("streaming_trace", "source"),
    ):
        section = _plain_mapping(realtime_data.get(key[0]))
        value = section.get(key[1])
        if value and _is_external_target(str(value)):
            sources.append(str(value))
    return _unique_strings(sources)


def _plain_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _scope_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in _plain_list(values):
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _as_int(value: Any) -> int:
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
            return 0
    return 0


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _is_external_target(target: str) -> bool:
    return urlparse(str(target)).scheme.lower() in {"http", "https"}


__all__ = [
    "probe_realtime_stack",
    "realtime_stack_contract",
    "run_realtime_stack_probe",
]
