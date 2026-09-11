import pytest

from fi.alk.harness.secrets import (
    SecretResolutionError,
    resolve_worker_secrets,
    runtime_configuration_value,
    worker_environment,
)
from fi.simulate.runtime.spec import SecretRef


def _ref(manager="futureagi", key="secret-livekit"):
    return SecretRef(manager=manager, key=key, purpose="voice connection")


def test_resolves_reference_to_alias_only_at_worker_boundary():
    resolved = resolve_worker_secrets(
        {"LIVEKIT_API_KEY": _ref()},
        environment={"secret-livekit": "resolved-value"},
    )

    assert resolved == {"LIVEKIT_API_KEY": "resolved-value"}


def test_missing_or_unsupported_secret_reference_fails_closed():
    with pytest.raises(SecretResolutionError, match="unavailable"):
        resolve_worker_secrets({"LIVEKIT_API_KEY": _ref()}, environment={})
    with pytest.raises(SecretResolutionError, match="unsupported"):
        resolve_worker_secrets(
            {"LIVEKIT_API_KEY": _ref(manager="unknown")},
            environment={"secret-livekit": "value"},
        )


def test_worker_does_not_inherit_unrelated_host_secrets():
    child = worker_environment(
        {"LIVEKIT_API_KEY": "job-value"},
        host_environment={
            "PATH": "/bin",
            "AWS_SECRET_ACCESS_KEY": "host-secret",
            "UNRELATED_CUSTOMER_TOKEN": "other-secret",
        },
    )

    assert child == {"PATH": "/bin", "LIVEKIT_API_KEY": "job-value"}


def test_uploaded_runtime_values_cannot_replace_runner_model_credentials():
    child = worker_environment(
        {},
        runtime_configuration={
            "GOOGLE_APPLICATION_CREDENTIALS": "customer.json",
            "OPENAI_API_KEY": "customer-openai",
        },
        host_environment={
            "PATH": "/bin",
            "GOOGLE_APPLICATION_CREDENTIALS": "/runner/google.json",
            "GOOGLE_CLOUD_PROJECT": "runner-project",
        },
    )

    assert child["GOOGLE_APPLICATION_CREDENTIALS"] == "/runner/google.json"
    assert child["GOOGLE_CLOUD_PROJECT"] == "runner-project"
    assert "OPENAI_API_KEY" not in child
    assert (
        runtime_configuration_value("GOOGLE_APPLICATION_CREDENTIALS", environment=child)
        == "customer.json"
    )
    assert (
        runtime_configuration_value("OPENAI_API_KEY", environment=child)
        == "customer-openai"
    )


def test_worker_inherits_runner_topology_but_uploaded_values_cannot_replace_it():
    child = worker_environment(
        {
            "ALK_DOCKER_NETWORK": "customer-network",
            "ALK_RUNNER_CONTAINER": "customer-container",
            "HARNESS_WEBHOOK_HOST": "127.0.0.1",
            "HARNESS_RUNTIME_WEBHOOK_URL": "http://customer.invalid",
        },
        host_environment={
            "ALK_DOCKER_NETWORK": "runner-network",
            "ALK_RUNNER_CONTAINER": "self",
            "ALK_DOCKER_PUBLISHED_HOST": "host.docker.internal",
            "HARNESS_WEBHOOK_HOST": "0.0.0.0",
            "HARNESS_RUNTIME_WEBHOOK_URL": "http://runner.internal:8787",
        },
    )

    assert child["ALK_DOCKER_NETWORK"] == "runner-network"
    assert child["ALK_RUNNER_CONTAINER"] == "self"
    assert child["ALK_DOCKER_PUBLISHED_HOST"] == "host.docker.internal"
    assert child["HARNESS_WEBHOOK_HOST"] == "0.0.0.0"
    assert child["HARNESS_RUNTIME_WEBHOOK_URL"] == "http://runner.internal:8787"


def test_worker_never_inherits_host_voice_provider_credentials():
    child = worker_environment(
        {"LIVEKIT_API_KEY": "job-livekit"},
        host_environment={
            "PATH": "/bin",
            "LIVEKIT_API_KEY": "runner-livekit",
            "LIVEKIT_API_SECRET": "runner-secret",
            "DEEPGRAM_API_KEY": "runner-deepgram",
            "CARTESIA_API_KEY": "runner-cartesia",
        },
    )

    assert child["LIVEKIT_API_KEY"] == "job-livekit"
    assert "LIVEKIT_API_SECRET" not in child
    assert "DEEPGRAM_API_KEY" not in child
    assert "CARTESIA_API_KEY" not in child


def test_worker_inherits_runner_model_roles_but_jobs_cannot_replace_them():
    child = worker_environment(
        {
            "ALK_AGENT_MODEL": "customer-agent",
            "ALK_USER_MODEL": "customer-user",
            "ALK_JUDGE_MODEL": "customer-judge",
        },
        host_environment={
            "ALK_AGENT_MODEL": "runner-agent",
            "ALK_USER_MODEL": "runner-user",
            "ALK_JUDGE_MODEL": "runner-judge",
        },
    )

    assert child["ALK_AGENT_MODEL"] == "runner-agent"
    assert child["ALK_USER_MODEL"] == "runner-user"
    assert child["ALK_JUDGE_MODEL"] == "runner-judge"
