from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def voice_cases():
    path = (
        Path(__file__).parents[1] / "oss" / "simulation-acceptance" / "voice_cases.py"
    )
    spec = importlib.util.spec_from_file_location("acceptance_voice_cases", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCEPTANCE_LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("LIVEKIT_TARGET_AGENT_NAME", "target-agent")
    monkeypatch.setenv(
        "LIVEKIT_TARGET_SYSTEM_PROMPT",
        "You support Swift Delivery Services.",
    )


def test_google_only_voice_stack_does_not_require_deepgram(
    monkeypatch: pytest.MonkeyPatch,
    voice_cases,
) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("SIMULATOR_LLM_PROVIDER", "google")
    monkeypatch.setenv("SIMULATOR_STT_PROVIDER", "google")
    monkeypatch.setenv("SIMULATOR_TTS_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/google.json")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    case = voice_cases.CASES["1.1.2"]
    inputs = voice_cases.build_inputs(case.case_id, "run-google")

    assert voice_cases.missing_env(case) == []
    assert "DEEPGRAM_API_KEY" not in case.required_env
    assert inputs.simulator.stt.provider == "google"
    assert inputs.simulator.tts.provider == "google"
    assert inputs.max_seconds == 210.0


def test_default_deepgram_voice_stack_keeps_web_budget(
    monkeypatch: pytest.MonkeyPatch,
    voice_cases,
) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/google.json")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram-key")
    for name in (
        "SIMULATOR_LLM_PROVIDER",
        "SIMULATOR_STT_PROVIDER",
        "SIMULATOR_TTS_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)

    inputs = voice_cases.build_inputs("1.1.2", "run-default")

    # Transactional calls need enough room for disambiguation, payment selection and OTP without
    # turning a healthy but deliberate agent into an infrastructure timeout.
    assert inputs.max_seconds == 240.0


def test_cartesia_voice_stack_uses_cartesia_models(
    monkeypatch: pytest.MonkeyPatch,
    voice_cases,
) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("SIMULATOR_LLM_PROVIDER", "google")
    monkeypatch.setenv("SIMULATOR_STT_PROVIDER", "cartesia")
    monkeypatch.setenv("SIMULATOR_TTS_PROVIDER", "cartesia")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/google.json")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project")
    monkeypatch.setenv("CARTESIA_API_KEY", "cartesia-key")
    monkeypatch.delenv("SIMULATOR_STT_MODEL", raising=False)
    monkeypatch.delenv("SIMULATOR_TTS_MODEL", raising=False)
    monkeypatch.delenv("SIMULATOR_TTS_VOICE", raising=False)

    case = voice_cases.CASES["1.1.2"]
    inputs = voice_cases.build_inputs(case.case_id, "run-cartesia")

    assert voice_cases.missing_env(case) == []
    assert "CARTESIA_API_KEY" in case.required_env
    assert inputs.simulator.stt.model == "ink-2"
    assert inputs.simulator.tts.model == "sonic-3"


def test_openai_voice_stack_uses_openai_defaults(
    monkeypatch: pytest.MonkeyPatch,
    voice_cases,
) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("SIMULATOR_LLM_PROVIDER", "openai")
    monkeypatch.setenv("SIMULATOR_STT_PROVIDER", "openai")
    monkeypatch.setenv("SIMULATOR_TTS_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    for name in (
        "SIMULATOR_LLM_MODEL",
        "SIMULATOR_STT_MODEL",
        "SIMULATOR_TTS_MODEL",
        "SIMULATOR_TTS_VOICE",
    ):
        monkeypatch.delenv(name, raising=False)

    inputs = voice_cases.build_inputs("1.1.2", "run-openai")

    assert inputs.simulator.llm.model == "gpt-4o"
    assert inputs.simulator.stt.model == "gpt-4o-mini-transcribe"
    assert inputs.simulator.tts.model == "gpt-4o-mini-tts"
    assert inputs.simulator.tts.voice == "alloy"


def test_livekit_url_fallback_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    voice_cases,
) -> None:
    _base_env(monkeypatch)
    monkeypatch.delenv("ACCEPTANCE_LIVEKIT_URL")
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("SIMULATOR_LLM_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/google.json")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram-key")

    with pytest.warns(RuntimeWarning, match="using LIVEKIT_URL"):
        inputs = voice_cases.build_inputs("1.1.2", "run-fallback")

    assert str(inputs.livekit_runtime.url) == "ws://localhost:7880/"
