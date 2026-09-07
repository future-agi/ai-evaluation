from __future__ import annotations

import json

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
