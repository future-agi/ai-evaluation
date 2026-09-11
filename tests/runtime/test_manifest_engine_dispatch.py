from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from fi.simulate import cli
from fi.simulate.manifest import ManifestError, run_manifest_file
from fi.simulate.simulation.models import (
    Persona,
    TestCaseResult as CaseResult,
    TestReport as SimulationReport,
)


def _scenario() -> dict:
    return {
        "name": "voice",
        "dataset": [
            {
                "persona": {"name": "Morgan"},
                "situation": "My delivery is late.",
                "outcome": "The delivery status is confirmed.",
            }
        ],
    }


def test_livekit_manifest_builds_typed_runtime_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    class FakeRunner:
        async def run_test(self, **kwargs):
            captured.update(kwargs)
            return "report"

    monkeypatch.setattr(cli, "TestRunner", FakeRunner)
    manifest = {
        "scenario": _scenario(),
        "agent_definition": {
            "name": "reference-agent",
            "url": "ws://127.0.0.1:7880",
            "room_name": "sdk-local-simulation",
            "room_mode": "managed",
            "system_prompt": "Help the caller.",
        },
        "simulator": {
            "llm": {"provider": "openai", "model": "gpt-5.4-mini"},
            "stt": {"provider": "elevenlabs", "model": "scribe_v2_realtime"},
            "tts": {
                "provider": "elevenlabs",
                "model": "eleven_flash_v2_5",
                "voice": "voice-id",
            },
        },
        "simulation": {
            "engine": "livekit",
            "run_id": "run_cli_livekit",
            "record_audio": True,
            "recording_root": "artifacts/audio",
            "min_turn_messages": 4,
            "max_seconds": 90,
            "connect_timeout": 10,
            "readiness_timeout": 20,
            "cleanup_timeout": 25,
        },
    }

    report = asyncio.run(cli._run_manifest(manifest, tmp_path / "manifest.json"))

    assert report == "report"
    assert captured["agent_definition"].name == "reference-agent"
    assert captured["simulator"].tts.provider == "elevenlabs"
    assert captured["scenario"].dataset[0].persona["name"] == "Morgan"
    assert captured["simulation_run_id"] == "run_cli_livekit"
    assert captured["record_audio"] is True
    assert captured["recording_root"] == tmp_path / "artifacts/audio"
    assert captured["min_turn_messages"] == 4
    assert captured["max_seconds"] == 90.0


def test_livekit_manifest_hydrates_explicit_target_and_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    class FakeRunner:
        async def run_test(self, **kwargs):
            captured.update(kwargs)
            return "report"

    monkeypatch.setattr(cli, "TestRunner", FakeRunner)
    manifest = {
        "scenario": _scenario(),
        "agent_definition": {
            "name": "healthcare-agent",
            "system_prompt": "Schedule appointments.",
            "target": {
                "provider": "vapi",
                "assistant_id": "assistant_healthcare",
                "api_key_env": "HEALTHCARE_VAPI_KEY",
            },
            "transport": {"kind": "vapi_websocket"},
        },
        "simulation": {
            "engine": "livekit",
            "livekit_runtime": {
                "url": "wss://futureagi-livekit.example.com",
                "room_name": "healthcare-{test_case_id}",
                "api_key_env": "FAGI_LIVEKIT_KEY",
                "api_secret_env": "FAGI_LIVEKIT_SECRET",
            },
        },
    }

    report = asyncio.run(cli._run_manifest(manifest, tmp_path / "manifest.json"))

    assert report == "report"
    assert captured["agent_definition"].target.assistant_id == "assistant_healthcare"
    assert captured["livekit_runtime"].room_name == "healthcare-{test_case_id}"


def test_cloud_manifest_uses_existing_test_runner_mode(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    class FakeRunner:
        async def run_test(self, **kwargs):
            captured.update(kwargs)
            return "cloud-report"

    monkeypatch.setattr(cli, "TestRunner", FakeRunner)
    manifest = {
        "simulation": {
            "engine": "cloud",
            "run_test_name": "nightly-support",
            "timeout": 45,
        }
    }

    report = asyncio.run(cli._run_manifest(manifest, tmp_path / "manifest.json"))

    assert report == "cloud-report"
    assert captured == {
        "run_id": None,
        "run_test_name": "nightly-support",
        "agent_callback": None,
        "timeout": 45.0,
    }


def test_local_text_manifest_keeps_existing_runner(monkeypatch, tmp_path: Path) -> None:
    async def fake_local(manifest, manifest_path):
        return manifest, manifest_path

    monkeypatch.setattr(cli, "_run_local_text_manifest", fake_local)
    manifest = {"simulation": {"engine": "local_text"}}

    result = asyncio.run(cli._run_manifest(manifest, tmp_path / "manifest.json"))

    assert result == (manifest, tmp_path / "manifest.json")


def test_manifest_dispatch_rejects_unknown_engine(tmp_path: Path) -> None:
    with pytest.raises(
        ManifestError, match="Supported: cloud, livekit, local, local_text"
    ):
        asyncio.run(
            cli._run_manifest(
                {"simulation": {"engine": "unknown"}},
                tmp_path / "manifest.json",
            )
        )


def test_local_text_manifest_writes_canonical_artifacts(tmp_path: Path) -> None:
    manifest = {
        "name": "canonical-text",
        "scenario": _scenario(),
        "agent": {"type": "scripted", "content": "The delivery status is confirmed."},
        "simulation": {
            "engine": "local_text",
            "run_id": "run_canonical_text",
            "result_root": "canonical",
            "max_turns": 1,
            "min_turns": 1,
        },
    }

    report = asyncio.run(
        cli._run_local_text_manifest(manifest, tmp_path / "manifest.json")
    )

    assert report.results[0].transcript
    run_directory = tmp_path / "canonical" / "run_canonical_text"
    assert (run_directory / "spec.json").exists()
    assert (run_directory / "plan.json").exists()
    assert (run_directory / "events.jsonl").exists()
    assert (run_directory / "report.json").exists()
    assert (run_directory / "artifacts.json").exists()
    canonical = json.loads((run_directory / "report.json").read_text())
    artifacts = json.loads((run_directory / "artifacts.json").read_text())
    assert canonical["schema_version"] == "futureagi.simulation-report.v1"
    assert canonical["run_id"] == "run_canonical_text"
    assert artifacts["schema_version"] == "futureagi.artifact-manifest.v1"
    assert artifacts["run_id"] == "run_canonical_text"


def test_local_text_manifest_rejects_invalid_result_root(tmp_path: Path) -> None:
    manifest = {
        "scenario": _scenario(),
        "agent": {"type": "scripted", "content": "done"},
        "simulation": {"engine": "local_text", "result_root": {}},
    }

    with pytest.raises(ManifestError, match="result_root"):
        asyncio.run(cli._run_local_text_manifest(manifest, tmp_path / "manifest.json"))


def test_scenario_source_resolves_relative_to_manifest(tmp_path: Path) -> None:
    source = tmp_path / "scenario.json"
    source.write_text(json.dumps(_scenario()))

    scenario = cli._build_scenario(
        {"scenario": {"source": source.name}},
        tmp_path,
    )

    assert scenario.name == "voice"
    assert scenario.dataset[0].persona["name"] == "Morgan"


def test_scenario_source_rejects_inline_dataset(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="cannot be combined"):
        cli._build_scenario(
            {"scenario": {"source": "scenario.json", "dataset": []}},
            tmp_path,
        )


def test_run_manifest_file_serializes_livekit_report(
    monkeypatch, tmp_path: Path
) -> None:
    persona = Persona(
        persona={"name": "Morgan"},
        situation="My delivery is late.",
        outcome="The delivery status is confirmed.",
    )
    report = SimulationReport(
        results=[
            CaseResult(
                persona=persona,
                transcript="assistant: Hello\nuser: Hi",
                messages=[
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Hi"},
                ],
                metadata={"status": "completed", "engine": "livekit"},
            )
        ]
    )

    async def fake_run_manifest(_manifest, _manifest_path):
        return report

    monkeypatch.setattr(cli, "_run_manifest", fake_run_manifest)
    manifest_path = tmp_path / "livekit.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "agent-learning.run.v1",
                "name": "livekit-cli",
                "scenario": _scenario(),
                "agent_definition": {
                    "name": "reference-agent",
                    "url": "ws://127.0.0.1:7880",
                    "room_name": "sdk-local-simulation",
                    "system_prompt": "Help the caller.",
                },
                "simulation": {"engine": "livekit"},
                "evaluation": {"enabled": False},
            }
        )
    )

    result = asyncio.run(run_manifest_file(manifest_path, no_eval=True))

    assert result["status"] == "passed"
    assert result["exit_code"] == 0
    assert result["report"]["results"][0]["metadata"]["status"] == "completed"


def test_livekit_manifest_accepts_sip_outbound_transport(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    class FakeRunner:
        async def run_test(self, **kwargs):
            captured.update(kwargs)
            return "report"

    monkeypatch.setattr(cli, "TestRunner", FakeRunner)
    manifest = {
        "scenario": _scenario(),
        "agent_definition": {
            "name": "phone-agent",
            "url": "ws://127.0.0.1:7880",
            "room_name": "sdk-{test_case_id}",
            "room_mode": "managed",
            "system_prompt": "Help the caller.",
            "transport": {
                "kind": "sip_outbound",
                "sip_trunk_id": "ST_test",
                "sip_number": "+12068956991",
                "sip_call_to": "+14155551234",
            },
        },
        "simulation": {"engine": "livekit"},
    }
    asyncio.run(cli._run_manifest(manifest, tmp_path / "manifest.json"))
    assert captured["agent_definition"].transport.kind == "sip_outbound"
    assert captured["agent_definition"].transport.sip_trunk_id == "ST_test"


def test_livekit_manifest_rejects_missing_sip_fields(tmp_path: Path) -> None:
    manifest = {
        "scenario": _scenario(),
        "agent_definition": {
            "name": "phone-agent",
            "url": "ws://127.0.0.1:7880",
            "room_name": "sdk-{test_case_id}",
            "room_mode": "managed",
            "system_prompt": "Help.",
            "transport": {"kind": "sip_outbound", "sip_trunk_id": "ST_test"},
        },
        "simulation": {"engine": "livekit"},
    }
    with pytest.raises(ManifestError, match="sip_call_to"):
        asyncio.run(cli._run_manifest(manifest, tmp_path / "m.json"))


def test_livekit_manifest_rejects_unknown_transport_kind(tmp_path: Path) -> None:
    manifest = {
        "scenario": _scenario(),
        "agent_definition": {
            "name": "phone-agent",
            "url": "ws://127.0.0.1:7880",
            "room_name": "sdk",
            "system_prompt": "Help.",
            "transport": {"kind": "carrier-pigeon"},
        },
        "simulation": {"engine": "livekit"},
    }
    with pytest.raises(ManifestError):
        asyncio.run(cli._run_manifest(manifest, tmp_path / "m.json"))


def test_livekit_manifest_accepts_sip_inbound_without_dispatch_rule(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    class FakeRunner:
        async def run_test(self, **kwargs):
            captured.update(kwargs)
            return "report"

    monkeypatch.setattr(cli, "TestRunner", FakeRunner)
    manifest = {
        "scenario": _scenario(),
        "agent_definition": {
            "name": "phone-agent",
            "url": "ws://127.0.0.1:7880",
            "room_name": "sdk-{test_case_id}",
            "room_mode": "managed",
            "system_prompt": "Help.",
            "transport": {"kind": "sip_inbound"},
        },
        "simulation": {"engine": "livekit"},
    }
    asyncio.run(cli._run_manifest(manifest, tmp_path / "m.json"))
    assert captured["agent_definition"].transport.kind == "sip_inbound"
    assert captured["agent_definition"].transport.dispatch_rule_name is None


def test_livekit_manifest_rejects_sip_inbound_empty_dispatch_rule(
    tmp_path: Path,
) -> None:
    manifest = {
        "scenario": _scenario(),
        "agent_definition": {
            "name": "phone-agent",
            "url": "ws://127.0.0.1:7880",
            "room_name": "sdk-{test_case_id}",
            "room_mode": "managed",
            "system_prompt": "Help.",
            "transport": {"kind": "sip_inbound", "dispatch_rule_name": "   "},
        },
        "simulation": {"engine": "livekit"},
    }
    with pytest.raises(ManifestError, match="dispatch_rule_name"):
        asyncio.run(cli._run_manifest(manifest, tmp_path / "m.json"))


def test_livekit_manifest_without_transport_defaults_to_webrtc(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    class FakeRunner:
        async def run_test(self, **kwargs):
            captured.update(kwargs)
            return "report"

    monkeypatch.setattr(cli, "TestRunner", FakeRunner)
    manifest = {
        "scenario": _scenario(),
        "agent_definition": {
            "name": "reference-agent",
            "url": "ws://127.0.0.1:7880",
            "room_name": "sdk",
            "system_prompt": "Help.",
        },
        "simulation": {"engine": "livekit"},
    }
    asyncio.run(cli._run_manifest(manifest, tmp_path / "m.json"))
    assert captured["agent_definition"].transport is None


def test_cli_result_fails_when_engine_case_fails() -> None:
    persona = Persona(
        persona={"name": "Morgan"},
        situation="My delivery is late.",
        outcome="The delivery status is confirmed.",
    )
    report = SimulationReport(
        results=[
            CaseResult(
                persona=persona,
                transcript="",
                metadata={"status": "timed_out", "engine": "livekit"},
            )
        ]
    )

    result = cli._run_result(
        manifest={"name": "livekit-cli"},
        report=report,
        evaluation=None,
        duration_seconds=1.0,
    )

    assert result["status"] == "failed"
    assert result["exit_code"] == 1


@pytest.mark.parametrize(
    ("transport_kind", "provider"),
    [("vapi_websocket", "vapi"), ("retell_webcall", "retell")],
)
def test_livekit_manifest_accepts_provider_web_transport(
    monkeypatch, tmp_path: Path, transport_kind, provider
) -> None:
    captured = {}

    class FakeRunner:
        async def run_test(self, **kwargs):
            captured.update(kwargs)
            return "report"

    monkeypatch.setattr(cli, "TestRunner", FakeRunner)
    manifest = {
        "scenario": _scenario(),
        "agent_definition": {
            "name": "web-agent",
            "url": "wss://livekit.example.com",
            "room_name": "sdk-web-{test_case_id}",
            "room_mode": "managed",
            "system_prompt": "Evaluate the provider assistant.",
            "transport": {"kind": transport_kind},
            "provider_evidence": {
                "provider": provider,
                "call_id_source": "originator_response",
            },
        },
        "simulation": {"engine": "livekit"},
    }

    asyncio.run(cli._run_manifest(manifest, tmp_path / "web.json"))

    definition = captured["agent_definition"]
    assert definition.transport.kind == transport_kind
    assert definition.provider_evidence.provider == provider
    assert definition.provider_evidence.call_id_source == "originator_response"


def test_provider_web_transport_rejects_sip_fields(tmp_path: Path) -> None:
    manifest = {
        "scenario": _scenario(),
        "agent_definition": {
            "name": "vapi-agent",
            "url": "wss://livekit.example.com",
            "room_name": "sdk-vapi",
            "room_mode": "managed",
            "system_prompt": "Evaluate the Vapi assistant.",
            "transport": {
                "kind": "vapi_websocket",
                "sip_call_to": "+14155551234",
            },
        },
        "simulation": {"engine": "livekit"},
    }

    with pytest.raises(ManifestError, match="cannot set SIP fields"):
        asyncio.run(cli._run_manifest(manifest, tmp_path / "vapi.json"))
