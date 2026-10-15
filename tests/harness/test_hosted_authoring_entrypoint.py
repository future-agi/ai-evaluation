from __future__ import annotations

import json
from pathlib import Path

from fi.alk.harness import hosted_authoring_entrypoint as entrypoint


def test_platform_simulator_values_ignore_customer_target_credentials() -> None:
    values = entrypoint._platform_simulator_values(
        {
            "ANTHROPIC_API_KEY": "customer-agent-key",
            "GOOGLE_APPLICATION_CREDENTIALS_JSON": "customer-agent-adc",
            "SIMULATOR_GOOGLE_APPLICATION_CREDENTIALS_JSON": "platform-adc",
            "SIMULATOR_GOOGLE_CLOUD_PROJECT": "platform-project",
        }
    )

    assert values == {
        "GOOGLE_APPLICATION_CREDENTIALS_JSON": "platform-adc",
        "GOOGLE_CLOUD_PROJECT": "platform-project",
    }
    assert "ANTHROPIC_API_KEY" not in values


def test_fallback_authoring_reads_gateway_key_without_deleting_control_channel(
    tmp_path, monkeypatch
) -> None:
    secrets = tmp_path / "secrets.json"
    gateway = tmp_path / "simulator-secrets.json"
    secrets.write_text("{}", encoding="utf-8")
    gateway.write_text(
        json.dumps(
            {
                "AGENTCC_API_KEY": "platform-key",
                "AGENTCC_BASE_URL": "https://gateway.example.test",
                "RETELL_API_KEY": "must-not-be-forwarded",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(entrypoint, "_SECRETS_PATH", secrets)
    monkeypatch.setattr(entrypoint, "_SIMULATOR_SECRETS_PATH", gateway)
    monkeypatch.setattr(entrypoint, "_ADC_PATH", tmp_path / "adc.json")
    monkeypatch.setenv("AGENTCC_API_KEY", "pre-test-key")
    monkeypatch.setenv("AGENTCC_BASE_URL", "https://pre-test.example.test")
    monkeypatch.setattr(
        entrypoint, "authoring_main", lambda argv, *, validate_runtime: 0
    )

    assert entrypoint.main([]) == 0
    assert entrypoint.os.environ["AGENTCC_API_KEY"] == "platform-key"
    assert entrypoint.os.environ["AGENTCC_BASE_URL"] == "https://gateway.example.test"
    assert gateway.is_file()
    assert "RETELL_API_KEY" not in entrypoint.os.environ


def test_authoring_hears_the_platform_noise_catalogue(tmp_path, monkeypatch) -> None:
    secrets = tmp_path / "secrets.json"
    gateway = tmp_path / "simulator-secrets.json"
    secrets.write_text("{}", encoding="utf-8")
    catalogue = json.dumps([{"environment": "street", "url": "https://clips.example.test/a.wav"}])
    gateway.write_text(
        json.dumps({"ALK_BACKGROUND_NOISE": "1", "ALK_BACKGROUND_NOISE_CATALOG": catalogue}),
        encoding="utf-8",
    )
    monkeypatch.setattr(entrypoint, "_SECRETS_PATH", secrets)
    monkeypatch.setattr(entrypoint, "_SIMULATOR_SECRETS_PATH", gateway)
    monkeypatch.setattr(entrypoint, "_ADC_PATH", tmp_path / "adc.json")
    monkeypatch.delenv("ALK_BACKGROUND_NOISE_CATALOG", raising=False)
    monkeypatch.setattr(
        entrypoint, "authoring_main", lambda argv, *, validate_runtime: 0
    )

    assert entrypoint.main([]) == 0
    assert entrypoint.os.environ["ALK_BACKGROUND_NOISE_CATALOG"] == catalogue
    assert gateway.is_file()


def test_vertex_generation_region_is_not_copied_from_google_location(
    tmp_path, monkeypatch
) -> None:
    adc = json.dumps({"type": "service_account", "project_id": "p"})
    monkeypatch.setattr(entrypoint, "_ADC_PATH", tmp_path / "google.json")
    monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
    entrypoint._configure_generation_environment(
        {
            "GOOGLE_APPLICATION_CREDENTIALS_JSON": adc,
            "GOOGLE_CLOUD_PROJECT": "p",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        }
    )
    assert entrypoint.os.environ["CLOUD_ML_REGION"] == "us-east5"
    assert entrypoint.os.environ["ANTHROPIC_VERTEX_PROJECT_ID"] == "p"


def test_explicit_claude_vertex_region_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(entrypoint, "_ADC_PATH", tmp_path / "google.json")
    monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
    entrypoint._configure_generation_environment(
        {
            "GOOGLE_APPLICATION_CREDENTIALS_JSON": json.dumps(
                {"type": "service_account"}
            ),
            "GOOGLE_CLOUD_PROJECT": "p",
            "ANTHROPIC_VERTEX_REGION": "europe-west1",
        }
    )
    assert entrypoint.os.environ["CLOUD_ML_REGION"] == "europe-west1"


def test_exhausted_runtime_repair_returns_nonretryable_exit(tmp_path, monkeypatch):
    from fi.alk.harness.authoring_runtime_validation import RuntimeValidationError

    monkeypatch.setattr(entrypoint, "_load_values", lambda path: {})
    monkeypatch.setattr(entrypoint, "_ADC_PATH", tmp_path / "adc.json")

    def failed(argv, *, validate_runtime):
        assert validate_runtime is True
        raise RuntimeValidationError("environment", "seed_dependency_unresolved")

    monkeypatch.setattr(entrypoint, "authoring_main", failed)
    assert entrypoint.main([]) == 78


def test_hosted_entrypoint_passes_one_shot_provider_secret_to_authoring(
    monkeypatch, tmp_path
) -> None:
    secrets = tmp_path / "secrets.json"
    target_secrets = tmp_path / "authoring-target-secrets.json"
    secrets.write_text(
        json.dumps(
            {
                "RETELL_API_KEY": "retell-secret",
                "SIMULATOR_GOOGLE_CLOUD_PROJECT": "project",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(entrypoint, "_SECRETS_PATH", secrets)
    monkeypatch.setattr(entrypoint, "_TARGET_SECRETS_PATH", target_secrets)
    monkeypatch.setattr(entrypoint, "_ADC_PATH", tmp_path / "adc.json")
    observed = {}

    def authoring_main(argv, *, validate_runtime):
        observed["argv"] = argv
        observed["validate_runtime"] = validate_runtime
        index = argv.index("--target-secrets")
        path = Path(argv[index + 1])
        observed["values"] = json.loads(path.read_text(encoding="utf-8"))
        return 0

    monkeypatch.setattr(entrypoint, "authoring_main", authoring_main)

    assert entrypoint.main(["job.json", "--source", "source", "--output", "out"]) == 0
    assert observed == {
        "argv": [
            "job.json",
            "--source",
            "source",
            "--output",
            "out",
            "--target-secrets",
            str(target_secrets),
        ],
        "validate_runtime": True,
        "values": {"RETELL_API_KEY": "retell-secret"},
    }
    assert not target_secrets.exists()
