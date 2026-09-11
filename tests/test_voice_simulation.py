from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from fi.simulate.agent.definition import (
    AgentDefinition,
    LiveKitSimulatorRuntime,
    SimulatorAgentDefinition,
)
from fi.simulate.simulation.models import Persona, Scenario
from fi.simulate import voice


def _agent(**updates) -> AgentDefinition:
    values = {
        "name": "vapi-agent",
        "url": "wss://livekit.example.com",
        "room_name": "sdk-{test_case_id}",
        "room_mode": "managed",
        "system_prompt": "Help the caller.",
        "transport": {"kind": "vapi_websocket"},
        "provider_evidence": {
            "provider": "vapi",
            "call_id_source": "originator_response",
        },
    }
    values.update(updates)
    return AgentDefinition(**values)


def _runtime() -> LiveKitSimulatorRuntime:
    return LiveKitSimulatorRuntime(
        url="wss://futureagi-livekit.example.com",
        room_name="sdk-{test_case_id}",
        api_key_env="FAGI_LIVEKIT_KEY",
        api_secret_env="FAGI_LIVEKIT_SECRET",
    )


def _scenario() -> Scenario:
    return Scenario(
        name="delivery",
        dataset=[
            Persona(
                persona={"name": "Priya"},
                situation="My device is late.",
                outcome="The delivery is resolved.",
            )
        ],
    )


def test_platform_voice_scenario_creates_agent_and_scenario(monkeypatch) -> None:
    captured = {}

    def generate(request, *, config):
        captured["request"] = request
        captured["config"] = config
        return "generated"

    from fi.alk import studio

    monkeypatch.setattr(studio, "generate_scenario", generate)

    result = asyncio.run(
        voice.generate_platform_voice_scenario(
            agent_definition=_agent(),
            name="platform-delivery",
            description="Generate delivery scenarios.",
            custom_instruction="Exercise delay handling.",
            no_of_rows=10,
            config="platform-config",
        )
    )

    assert result == "generated"
    assert captured["request"].agent_definition == _agent()
    assert captured["request"].name == "platform-delivery"
    assert captured["config"] == "platform-config"


def test_build_voice_run_manifest_serializes_typed_inputs_without_secrets() -> None:
    manifest = voice.build_voice_run_manifest(
        agent_definition=_agent(),
        scenario=_scenario(),
        simulator=SimulatorAgentDefinition(),
        name="direct-vapi",
        required_env=["DEEPGRAM_API_KEY"],
        simulation_run_id="run_voice",
        record_audio=True,
        max_seconds=120,
    )

    assert manifest["version"] == "agent-learning.run.v1"
    assert manifest["name"] == "direct-vapi"
    assert manifest["agent_definition"]["transport"]["kind"] == "vapi_websocket"
    assert manifest["scenario"]["name"] == "delivery"
    assert manifest["simulation"]["run_id"] == "run_voice"
    assert manifest["simulation"]["max_seconds"] == 120
    assert manifest["required_env"] == [
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "DEEPGRAM_API_KEY",
        "VAPI_API_KEY",
        "VAPI_ASSISTANT_ID",
    ]
    assert "VAPI_API_KEY" not in str(manifest["agent_definition"])
    assert "llm" not in manifest["agent_definition"]
    assert "stt" not in manifest["agent_definition"]
    assert "tts" not in manifest["agent_definition"]


def test_run_voice_simulation_delegates_typed_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    class FakeRunner:
        async def run_test(self, **kwargs):
            captured.update(kwargs)
            return "report"

    monkeypatch.setattr(voice, "TestRunner", FakeRunner)

    result = asyncio.run(
        voice.run_voice_simulation(
            agent_definition=_agent(),
            scenario=_scenario(),
            simulator=SimulatorAgentDefinition(),
            livekit_runtime=_runtime(),
            simulation_run_id="run_direct",
            recording_root=tmp_path,
            recording_case_directory=tmp_path / "case-recordings",
            record_audio=True,
            max_seconds=90,
        )
    )

    assert result == "report"
    assert isinstance(captured["agent_definition"], AgentDefinition)
    assert isinstance(captured["scenario"], Scenario)
    assert captured["livekit_runtime"] == _runtime()
    assert captured["simulation_run_id"] == "run_direct"
    assert captured["recording_root"] == tmp_path
    assert captured["recording_case_directory"] == tmp_path / "case-recordings"
    assert captured["max_seconds"] == 90


def test_direct_transport_rejects_mismatched_explicit_target() -> None:
    with pytest.raises(ValueError, match="vapi_websocket_requires_vapi_target"):
        _agent(
            target={
                "provider": "retell",
                "agent_id": "agent_healthcare",
            },
        )


def test_explicit_target_requires_matching_direct_transport() -> None:
    with pytest.raises(ValueError, match="vapi_target_requires_vapi_websocket"):
        _agent(
            transport={"kind": "webrtc"},
            target={
                "provider": "vapi",
                "assistant_id": "assistant_healthcare",
            },
        )


def test_explicit_vapi_target_manifest_keeps_runtime_and_secrets_separate() -> None:
    agent = _agent(
        target={
            "provider": "vapi",
            "assistant_id": "assistant_healthcare",
            "api_base_url": "https://vapi.example",
            "api_key_env": "ACME_VAPI_API_KEY",
        },
    )

    manifest = voice.build_voice_run_manifest(
        agent_definition=agent,
        scenario=_scenario(),
        livekit_runtime=_runtime(),
    )

    assert manifest["agent_definition"]["target"] == {
        "provider": "vapi",
        "assistant_id": "assistant_healthcare",
        "api_base_url": "https://vapi.example/",
        "api_key_env": "ACME_VAPI_API_KEY",
    }
    assert manifest["simulation"]["livekit_runtime"] == {
        "url": "wss://futureagi-livekit.example.com/",
        "room_name": "sdk-{test_case_id}",
        "room_mode": "managed",
        "room_name_verbatim": False,
        "api_key_env": "FAGI_LIVEKIT_KEY",
        "api_secret_env": "FAGI_LIVEKIT_SECRET",
    }
    assert manifest["required_env"] == [
        "FAGI_LIVEKIT_KEY",
        "FAGI_LIVEKIT_SECRET",
        "ACME_VAPI_API_KEY",
    ]
    assert '"api_key":' not in json.dumps(manifest)


@pytest.mark.parametrize(
    ("transport", "target", "provider", "target_id"),
    [
        (
            "vapi_websocket",
            {
                "provider": "vapi",
                "assistant_id": "assistant_healthcare",
                "api_key_env": "ACME_VAPI_API_KEY",
            },
            "vapi",
            "assistant_healthcare",
        ),
        (
            "retell_webcall",
            {
                "provider": "retell",
                "agent_id": "agent_healthcare",
                "api_key_env": "ACME_RETELL_API_KEY",
            },
            "retell",
            "agent_healthcare",
        ),
    ],
)
def test_platform_payload_uses_target_provider_without_credentials(
    transport, target, provider, target_id
) -> None:
    from fi.alk.studio._generate import _agent_payload

    payload, configuration_hash = _agent_payload(
        _agent(
            description="Human-readable target summary that is not part of the prompt.",
            transport={"kind": transport},
            provider_evidence={
                "provider": provider,
                "call_id_source": "originator_response",
            },
            target=target,
        )
    )

    assert payload["description"] == "Help the caller."
    assert "Human-readable target summary" not in payload["description"]
    assert payload["provider"] == provider
    assert payload["assistant_id"] == target_id
    assert "livekit_url" not in payload
    assert "api_key" not in payload
    assert configuration_hash


def test_run_voice_simulation_rejects_ambiguous_scenario_generation() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        asyncio.run(
            voice.run_voice_simulation(
                agent_definition=_agent(),
                scenario=_scenario(),
                topic="delivery support",
            )
        )


def test_run_voice_simulation_requires_scenario_or_topic() -> None:
    with pytest.raises(ValueError, match="provide scenario or topic"):
        asyncio.run(voice.run_voice_simulation(agent_definition=_agent()))
