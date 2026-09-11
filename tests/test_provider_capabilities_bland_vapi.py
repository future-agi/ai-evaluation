"""Bland.ai and Vapi must be first-class providers in the simulation layer.

They were missing from AGENT_INTEGRATION_PROVIDER_CAPABILITIES, so the common
simulation layer didn't know their voice/phone capabilities when normalizing an
agent-integration manifest.
"""

from fi.alk.simulate import (
    AGENT_INTEGRATION_PROVIDER_CAPABILITIES,
    normalize_agent_integration_provider_name,
    normalize_agent_integration_manifest,
)
from fi.alk.evals.metrics.agents.report import (
    _normalize_agent_integration_provider as normalize_report_provider,
    evaluate_agent_report,
)
from fi.alk.optimize.components import COMPONENT_SPECS, diagnose_text
from fi.alk import optimize


def test_bland_and_vapi_present_with_voice_capabilities():
    for provider in ("vapi", "bland"):
        caps = AGENT_INTEGRATION_PROVIDER_CAPABILITIES[provider]
        assert "voice" in caps
        assert "phone" in caps
        assert "sip" in caps


def test_bland_aliases_resolve():
    # The user wrote "bland.ai"; the key-normalizer turns "." and spaces into "_".
    assert normalize_agent_integration_provider_name("bland") == "bland"
    assert normalize_agent_integration_provider_name("bland.ai") == "bland"
    assert normalize_agent_integration_provider_name("bland_ai") == "bland"
    assert normalize_agent_integration_provider_name("Bland AI") == "bland"


def test_vapi_alias_resolves():
    assert normalize_agent_integration_provider_name("vapi") == "vapi"
    assert normalize_agent_integration_provider_name("vapi_ai") == "vapi"


def test_agent_learning_builder_defaults_cover_vapi_and_bland_channels():
    manifest = optimize.build_agent_integration_optimization_manifest(
        name="builder-provider-channel-defaults"
    )
    required_channels = manifest["evaluation"]["agent_report"]["config"][
        "agent_integration_quality"
    ]["required_provider_channels"]

    assert required_channels["vapi"] == [
        "chat",
        "voice",
        "webrtc",
        "phone",
        "sip",
        "websocket",
    ]
    assert required_channels["bland"] == [
        "voice",
        "phone",
        "sip",
        "web_call",
        "websocket",
    ]


def test_report_aliases_resolve_like_simulation_layer():
    assert normalize_report_provider("Bland.ai") == "bland"
    assert normalize_report_provider("Bland AI") == "bland"
    assert normalize_report_provider("Vapi AI") == "vapi"


def test_phone_and_sip_sessions_infer_channel_from_provider_capabilities():
    manifest = normalize_agent_integration_manifest(
        sessions=[
            {
                "id": "vapi_phone",
                "provider": "Vapi AI",
                "phone_number": "+15550101011",
                "transcript": "Vapi phone session passed.",
            },
            {
                "id": "bland_sip",
                "provider": "Bland.ai",
                "sip_call_id": "sip-bland-123",
                "transcript": "Bland SIP session passed.",
            },
        ],
    )

    sessions = {session["id"]: session for session in manifest["sessions"]}
    assert sessions["vapi_phone"]["provider"] == "vapi"
    assert sessions["vapi_phone"]["channel"] == "phone"
    assert sessions["bland_sip"]["provider"] == "bland"
    assert sessions["bland_sip"]["channel"] == "sip"


def test_agent_report_quality_accepts_vapi_bland_alias_requirements():
    evaluation = evaluate_agent_report(
        {
            "results": [
                {
                    "artifacts": [
                        {
                            "type": "trace",
                            "metadata": {"kind": "agent_integration_manifest"},
                            "data": {
                                "kind": "agent_integration_manifest",
                                "agent_definition": {"name": "support-agent"},
                                "providers": [
                                    {
                                        "provider": "vapi",
                                        "channels": ["phone", "webrtc"],
                                        "credential_status": "live_verified",
                                    },
                                    {
                                        "provider": "bland",
                                        "channels": ["phone", "sip", "web_call"],
                                        "credential_status": "live_verified",
                                    },
                                ],
                                "sessions": [
                                    {
                                        "id": "vapi_phone",
                                        "provider": "vapi",
                                        "channel": "phone",
                                        "signals": ["trace", "transcript"],
                                    },
                                    {
                                        "id": "bland_sip",
                                        "provider": "bland",
                                        "channel": "sip",
                                        "signals": ["trace", "transcript"],
                                    },
                                ],
                            },
                        }
                    ],
                }
            ]
        },
        config={
            "agent_integration_quality": {
                "required_providers": ["Vapi AI", "Bland.ai"],
                "required_provider_channels": {
                    "Vapi AI": ["phone", "webrtc"],
                    "Bland.ai": ["phone", "sip", "web_call"],
                },
                "min_verified_providers": 2,
            }
        },
    )

    quality = next(
        metric
        for metric in evaluation.cases[0].metrics
        if metric.name == "agent_integration_quality"
    )
    assert quality.score == 1.0


def test_all_goal_providers_have_voice_capability():
    # Every voice provider from the goal is known to the simulation layer.
    for provider in (
        "livekit",
        "livekit_bridge",
        "vapi",
        "retell",
        "bland",
        "elevenlabs",
        "deepgram",
        "agora",
        "pipecat",
        "twilio",
    ):
        assert provider in AGENT_INTEGRATION_PROVIDER_CAPABILITIES, provider


def test_optimizer_routes_vapi_bland_provider_paths():
    integration_paths = set(COMPONENT_SPECS["integration"].config_paths)
    voice_paths = set(COMPONENT_SPECS["voice"].config_paths)

    assert "providers.vapi" in integration_paths
    assert "providers.bland" in integration_paths
    assert "integrations.vapi.phone" in integration_paths
    assert "integrations.bland.sip" in integration_paths
    assert "voice.trace.vapi" in voice_paths
    assert "voice.trace.bland" in voice_paths

    diagnoses = diagnose_text("Vapi integration failed and Bland SIP trace is missing.")
    assert any(diagnosis.component == "integration" for diagnosis in diagnoses)
