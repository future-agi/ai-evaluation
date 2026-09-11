from __future__ import annotations

import asyncio
from pathlib import Path

from fi.alk import simulate


def build_inputs() -> tuple[
    simulate.AgentDefinition,
    simulate.LiveKitSimulatorRuntime,
    simulate.Scenario,
    simulate.SimulatorAgentDefinition,
]:
    agent_definition = simulate.AgentDefinition(
        name="vapi-support-agent",
        description="Medical-device delivery support voice assistant.",
        system_prompt=(
            "Copy the current Vapi assistant system prompt here before generating "
            "or running scenarios."
        ),
        target={
            "provider": "vapi",
            "assistant_id": "your-vapi-assistant-id",
            "api_base_url": "https://api.vapi.ai",
            "api_key_env": "VAPI_API_KEY",
        },
        transport={"kind": "vapi_websocket"},
        provider_evidence={
            "provider": "vapi",
            "call_id_source": "originator_response",
        },
    )
    livekit_runtime = simulate.LiveKitSimulatorRuntime(
        url="wss://your-futureagi-livekit-project.livekit.cloud",
        room_name="support-{test_case_id}",
    )
    scenario = simulate.Scenario(
        name="delivery-support",
        dataset=[
            simulate.Persona(
                persona={"name": "Priya", "temperament": "assertive"},
                situation="A medical-device delivery is late. Ask when it will arrive.",
                outcome="Complete a natural multi-turn conversation.",
            )
        ],
    )
    simulator = simulate.SimulatorAgentDefinition(
        llm={"provider": "google", "model": "gemini-2.5-flash-lite"},
        stt={"provider": "deepgram", "model": "nova-2-phonecall"},
        tts={
            "provider": "elevenlabs",
            "model": "eleven_flash_v2_5",
            "voice": "your-elevenlabs-voice-id",
        },
    )
    return agent_definition, livekit_runtime, scenario, simulator


async def main() -> None:
    agent_definition, livekit_runtime, scenario, simulator = build_inputs()
    report = await simulate.run_voice_simulation(
        agent_definition=agent_definition,
        livekit_runtime=livekit_runtime,
        scenario=scenario,
        simulator=simulator,
        record_audio=True,
        recording_root="artifacts/recordings",
        min_turn_messages=8,
        max_seconds=120,
    )
    manifest = simulate.build_voice_run_manifest(
        agent_definition=agent_definition,
        livekit_runtime=livekit_runtime,
        scenario=scenario,
        simulator=simulator,
        record_audio=True,
        max_seconds=120,
    )
    simulate.write_manifest_file(manifest, Path("artifacts/voice.manifest.json"))
    print(report.results[0].metadata["status"])


if __name__ == "__main__":
    asyncio.run(main())
