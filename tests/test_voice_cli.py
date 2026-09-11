from __future__ import annotations

import json
from pathlib import Path

from fi.simulate import cli


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_voice_cli_dry_run_builds_optional_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    agent = _write_json(
        tmp_path / "agent.json",
        {
            "name": "vapi-agent",
            "url": "wss://livekit.example.com",
            "room_name": "sdk-{test_case_id}",
            "room_mode": "managed",
            "system_prompt": "Help the caller.",
            "transport": {"kind": "vapi_websocket"},
        },
    )
    scenario = _write_json(
        tmp_path / "scenario.json",
        {
            "name": "delivery",
            "dataset": [
                {
                    "persona": {"name": "Priya"},
                    "situation": "My delivery is late.",
                    "outcome": "The delivery is resolved.",
                }
            ],
        },
    )
    manifest = tmp_path / "voice.manifest.json"
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "VAPI_API_KEY",
        "VAPI_ASSISTANT_ID",
    ):
        monkeypatch.setenv(name, "test-value")

    exit_code = cli.main(
        [
            "voice",
            "--agent-definition",
            str(agent),
            "--scenario",
            str(scenario),
            "--write-manifest",
            str(manifest),
            "--dry-run",
            "--quiet",
        ]
    )

    assert exit_code == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["agent_definition"]["transport"]["kind"] == "vapi_websocket"
    assert payload["scenario"]["name"] == "delivery"


def test_voice_cli_uses_explicit_target_and_livekit_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    agent = _write_json(
        tmp_path / "agent.json",
        {
            "name": "healthcare-vapi-agent",
            "system_prompt": "You schedule and reschedule patient appointments.",
            "transport": {"kind": "vapi_websocket"},
            "target": {
                "provider": "vapi",
                "assistant_id": "assistant_healthcare",
                "api_key_env": "HEALTHCARE_VAPI_KEY",
            },
        },
    )
    runtime = _write_json(
        tmp_path / "runtime.json",
        {
            "url": "wss://futureagi-livekit.example.com",
            "room_name": "healthcare-{test_case_id}",
            "api_key_env": "FAGI_LIVEKIT_KEY",
            "api_secret_env": "FAGI_LIVEKIT_SECRET",
        },
    )
    scenario = _write_json(
        tmp_path / "scenario.json",
        {
            "name": "appointments",
            "dataset": [
                {
                    "persona": {"name": "Priya"},
                    "situation": "My appointment was cancelled.",
                    "outcome": "Book a new appointment.",
                }
            ],
        },
    )
    manifest = tmp_path / "voice.manifest.json"
    for name in ("FAGI_LIVEKIT_KEY", "FAGI_LIVEKIT_SECRET", "HEALTHCARE_VAPI_KEY"):
        monkeypatch.setenv(name, "test-value")

    exit_code = cli.main(
        [
            "voice",
            "--agent-definition",
            str(agent),
            "--livekit-runtime",
            str(runtime),
            "--scenario",
            str(scenario),
            "--write-manifest",
            str(manifest),
            "--dry-run",
            "--quiet",
        ]
    )

    assert exit_code == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert (
        payload["agent_definition"]["target"]["assistant_id"] == "assistant_healthcare"
    )
    assert payload["simulation"]["livekit_runtime"]["room_name"] == (
        "healthcare-{test_case_id}"
    )
    assert payload["required_env"] == [
        "FAGI_LIVEKIT_KEY",
        "FAGI_LIVEKIT_SECRET",
        "HEALTHCARE_VAPI_KEY",
    ]


def test_voice_cli_rejects_manifest_export_for_generated_scenario(
    tmp_path: Path,
) -> None:
    agent = _write_json(
        tmp_path / "agent.json",
        {
            "name": "agent",
            "url": "wss://livekit.example.com",
            "room_name": "sdk",
            "room_mode": "managed",
            "system_prompt": "Help.",
        },
    )

    exit_code = cli.main(
        [
            "voice",
            "--agent-definition",
            str(agent),
            "--topic",
            "delivery support",
            "--write-manifest",
            str(tmp_path / "voice.json"),
            "--quiet",
        ]
    )

    assert exit_code == 2
