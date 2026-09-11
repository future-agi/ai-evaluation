from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_REALTIME_EXAMPLE_KEY"


def weak_candidate() -> dict[str, Any]:
    return {
        "voice": {
            "sample_rate_hz": 8000,
            "stt_latency_ms": 620,
            "tts_latency_ms": 980,
            "utterances": [
                {
                    "id": "utt_refund",
                    "speaker": "user",
                    "transcript": "I need help with a refund on my order.",
                    "start_ms": 0,
                    "end_ms": 2100,
                    "latency_ms": 620,
                    "confidence": 0.82,
                }
            ],
            "frame_replay": [
                {
                    "id": "weak_frame",
                    "type": "audio_frame",
                    "frame_type": "audio_frame",
                    "speaker": "user",
                    "timestamp_ms": 100,
                    "duration_ms": 20,
                }
            ],
            "timing_distribution": {"stages": {"stt": [620], "tts": [980]}},
            "routes": {"billing": {"queue": "billing"}},
            "initial_route": "billing",
            "noise_profile": {"snr_db": 12, "noise_db": 58},
            "perceptual_metrics": {
                "overall": {
                    "snr_db": 12,
                    "mos": 3.1,
                    "clipping_ratio": 0.06,
                    "jitter_ms": 85,
                    "packet_loss_pct": 4.2,
                    "sample_rate_hz": 8000,
                    "rms_db": -36,
                    "peak_db": 1,
                }
            },
        },
        "streaming_trace": {
            "state": {"route": "billing"},
            "events": [
                {
                    "id": "weak_stream_start",
                    "type": "session_start",
                    "content": "session opened",
                    "timestamp_ms": 0,
                },
                {
                    "id": "weak_token",
                    "type": "token_delta",
                    "content": "Refund request noted.",
                    "timestamp_ms": 620,
                    "latency_ms": 620,
                    "gap_ms": 420,
                },
                {
                    "id": "weak_stream_end",
                    "type": "message_done",
                    "content": "Done.",
                    "status": "completed",
                    "timestamp_ms": 980,
                },
            ],
        },
    }


def strong_candidate() -> dict[str, Any]:
    return {
        "voice": {
            "sample_rate_hz": 16000,
            "stt_latency_ms": 132,
            "tts_latency_ms": 260,
            "utterances": [
                {
                    "id": "utt_refund",
                    "speaker": "user",
                    "transcript": "I need help with a refund on my order.",
                    "start_ms": 0,
                    "end_ms": 1720,
                    "latency_ms": 132,
                    "confidence": 0.97,
                }
            ],
            "frame_replay": [
                {
                    "id": "frame_user_audio",
                    "type": "audio_frame",
                    "frame_type": "audio_frame",
                    "speaker": "user",
                    "timestamp_ms": 80,
                    "duration_ms": 20,
                    "energy": 0.74,
                },
                {
                    "id": "frame_agent_audio",
                    "type": "audio_frame",
                    "frame_type": "audio_frame",
                    "speaker": "agent",
                    "timestamp_ms": 900,
                    "duration_ms": 20,
                    "overlap": True,
                    "overlap_ms": 20,
                    "energy": 0.42,
                },
            ],
            "timing_distribution": {
                "stage_order": ["vad", "stt", "llm", "tts"],
                "stages": {
                    "vad": [24, 29, 31],
                    "stt": [120, 132, 148],
                    "llm": [210, 224, 241],
                    "tts": [250, 260, 280],
                }
            },
            "routes": {
                "support": {"queue": "refund_support", "priority": "high"},
                "billing": {"queue": "billing"},
            },
            "initial_route": "support",
            "noise_profile": {"snr_db": 28, "noise_db": 18},
            "perceptual_metrics": {
                "overall": {
                    "snr_db": 28,
                    "mos": 4.4,
                    "clipping_ratio": 0.01,
                    "jitter_ms": 18,
                    "packet_loss_pct": 0.2,
                    "sample_rate_hz": 16000,
                    "rms_db": -18,
                    "peak_db": -3,
                }
            },
            "webrtc_stats": [
                {
                    "type": "inbound-rtp",
                    "track_id": "support-audio",
                    "codec": "opus",
                    "jitter_ms": 18,
                    "packet_loss_pct": 0.2,
                    "sample_rate_hz": 16000,
                }
            ],
        },
        "streaming_trace": {
            "state": {"route": "support"},
            "events": [
                {
                    "id": "stream_start",
                    "type": "session_start",
                    "content": "session opened",
                    "timestamp_ms": 0,
                },
                {
                    "id": "stream_token_1",
                    "type": "token_delta",
                    "content": "Your refund ",
                    "timestamp_ms": 110,
                    "latency_ms": 110,
                    "gap_ms": 110,
                },
                {
                    "id": "stream_tool_delta",
                    "type": "tool_delta",
                    "name": "route_call",
                    "tool_call": {"name": "route_call", "arguments": {"route": "support"}},
                    "timestamp_ms": 190,
                    "gap_ms": 80,
                },
                {
                    "id": "stream_token_2",
                    "type": "token_delta",
                    "content": "request has been routed to support.",
                    "timestamp_ms": 300,
                    "gap_ms": 110,
                },
                {
                    "id": "stream_end",
                    "type": "message_done",
                    "content": "Your refund request has been routed to support.",
                    "status": "completed",
                    "timestamp_ms": 420,
                    "gap_ms": 120,
                },
            ],
        },
    }


def evaluation_config() -> dict[str, Any]:
    return {
        "task_description": (
            "Optimize a realtime refund voice harness with support routing, "
            "audio quality, streaming evidence, and timing gates."
        ),
        "expected_result": (
            "Realtime voice and streaming evidence proves the support route."
        ),
        "required_tools": [
            "voice_status",
            "voice_timing",
            "transcribe_audio",
            "route_call",
            "streaming_trace_status",
            "list_stream_events",
            "inspect_stream_event",
            "speak",
        ],
        "max_voice_latency_ms": 1800,
        "success_criteria": [
            "refund request has been routed to support",
            "realtime voice and streaming evidence",
        ],
        "required_voice_trace": [
            "event",
            "vad",
            "stt",
            "tts",
            "route",
            "timing_distribution",
            "timing_stage",
            "frame",
            "audio",
            "snr",
            "mos",
            "jitter",
            "packet_loss",
            "sample_rate",
        ],
        "expected_voice_route": "support",
        "expected_voice_transcript_contains": ["refund"],
        "required_voice_frame_types": ["audio_frame"],
        "max_voice_overlap_ms": 30,
        "max_voice_noise_db": 35,
        "required_voice_speakers": ["user", "agent"],
        "min_voice_snr_db": 20,
        "min_voice_mos": 4.0,
        "max_voice_clipping_ratio": 0.03,
        "max_voice_jitter_ms": 40,
        "max_voice_packet_loss_pct": 1.0,
        "min_voice_sample_rate_hz": 16000,
        "min_voice_duration_ms": 20,
        "max_voice_duration_ms": 3000,
        "min_voice_rms_db": -30,
        "max_voice_peak_db": -1,
        "voice_timing_distribution": {
            "required_stages": ["vad", "stt", "llm", "tts"],
            "min_samples_per_stage": 2,
            "max_stage_p95_ms": {"vad": 45, "stt": 180, "llm": 260, "tts": 320},
            "required_order": ["vad", "stt", "llm", "tts"],
        },
        "required_streaming_trace": [
            "trace",
            "event",
            "chunk",
            "tool_delta",
            "final",
            "latency",
            "gap",
            "livekit",
        ],
        "streaming_trace_quality": {
            "expected_output_contains": ["refund", "support"],
            "required_chunks": ["Your refund", "support"],
            "expected_tool_deltas": [
                {"name": "route_call", "arguments": {"route": "support"}}
            ],
            "min_chunk_count": 2,
            "min_tool_delta_count": 1,
            "max_first_token_latency_ms": 180,
            "max_gap_ms": 220,
            "max_dropped_events": 0,
            "max_error_count": 0,
            "require_completion": True,
            "expected_state": {"route": "support"},
        },
        "metric_weights": {
            "voice_trace_coverage": 5.0,
            "voice_interaction_quality": 8.0,
            "voice_timing_distribution_quality": 8.0,
            "streaming_trace_coverage": 5.0,
            "streaming_interaction_quality": 8.0,
            "tool_selection_accuracy": 2.0,
            "task_completion": 2.0,
        },
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_realtime_optimization_manifest(
        name="sdk-realtime-voice-optimization",
        required_env=[REQUIRED_ENV],
        realtime_candidates=[weak_candidate(), strong_candidate()],
        evaluation_config=evaluation_config(),
        threshold=0.9,
        framework="livekit",
        modality="voice",
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    result = optimize.optimize_manifest(
        build_manifest(),
        manifest_path=Path(__file__).with_suffix(".json"),
    )
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
