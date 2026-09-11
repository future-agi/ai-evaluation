from __future__ import annotations

import json
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fi.alk.harness.job import AgentConnection, ProviderExecutionMode
from fi.alk.harness.provider_lifecycle import (
    ProviderContext,
    ProviderLifecycleError,
    ProviderType,
    build_lifecycle_invocation,
    load_provider_manifest,
    validate_provision_receipt,
    run_lifecycle_invocation,
)


def _context(provider: ProviderType = ProviderType.VAPI) -> ProviderContext:
    return ProviderContext(
        attempt_id="attempt-1",
        world_id="world-2",
        provider=provider,
        public_base_url="https://opaque.example/a/w",
        event_url="https://opaque.example/a/w/events",
        tool_base_url="https://opaque.example/a/w/tools",
        provider_resource_prefix="alk-attempt-1-world-2",
        idempotency_key=f"attempt-1:world-2:{provider.value}",
        expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )


def test_connect_only_requires_provider_target_id() -> None:
    with pytest.raises(ValueError, match="connect_only_requires_assistant_id"):
        AgentConnection(
            connector="vapi",
            mode=ProviderExecutionMode.CONNECT_ONLY,
            config={},
        )


def test_environment_backed_rejects_preexisting_target() -> None:
    with pytest.raises(
        ValueError, match="environment_backed_target_is_provision_output"
    ):
        AgentConnection(
            connector="retell",
            mode=ProviderExecutionMode.ENVIRONMENT_BACKED,
            config={"agent_id": "production-agent"},
        )


def test_provider_import_requires_and_keeps_source_target_id() -> None:
    connection = AgentConnection(
        connector="vapi",
        mode=ProviderExecutionMode.PROVIDER_IMPORT,
        config={"assistant_id": "assistant-source"},
    )
    assert connection.config["assistant_id"] == "assistant-source"


def test_provider_import_rejects_unsafe_tool_path() -> None:
    with pytest.raises(ValueError, match="provider_import_tool_path_invalid"):
        AgentConnection(
            connector="retell",
            mode=ProviderExecutionMode.PROVIDER_IMPORT,
            config={"agent_id": "agent-source", "tool_path": "/tools/../admin"},
        )


def test_load_manifest_and_build_secret_scoped_invocation(tmp_path: Path) -> None:
    (tmp_path / "alk.yaml").write_text(
        """
schema_version: "1"
runtime: {}
provider:
  type: vapi
  scope: world
  provision:
    command: python scripts/provider_agent.py provision
    timeout_seconds: 90
  destroy:
    command: [python, scripts/provider_agent.py, destroy]
  output: /alk/output/provider-target.json
  required_secrets: [VAPI_API_KEY]
  public_capability: target_http
""",
        encoding="utf-8",
    )
    manifest = load_provider_manifest(tmp_path)
    context = _context()
    context_path = tmp_path / "lifecycle" / "provider-context.json"
    invocation = build_lifecycle_invocation(
        command=manifest.provider.provision,
        source_directory=tmp_path,
        lifecycle_directory=context_path.parent,
        context_path=context_path,
        context=context,
        required_secret_values={"VAPI_API_KEY": "test-key"},
        output_relative_path=manifest.provider.output,
    )

    assert invocation.argv == ("python", "scripts/provider_agent.py", "provision")
    assert invocation.timeout_seconds == 90
    assert invocation.output_path == context_path.parent / "provider-target.json"
    assert invocation.environment["PATH"].startswith(str(tmp_path / ".venv" / "bin"))
    assert invocation.environment["ALK_TOOL_BASE_URL"].endswith("/tools")
    assert invocation.environment["VAPI_API_KEY"] == "test-key"


def test_receipt_is_scope_checked_and_secret_free(tmp_path: Path) -> None:
    context = _context()
    receipt_path = tmp_path / "provider-target.json"
    body = {
        "schema_version": "1",
        "provider": "vapi",
        "attempt_id": context.attempt_id,
        "world_id": context.world_id,
        "target": {"kind": "assistant", "id": "assistant-1", "version": None},
        "resources": [
            {"kind": "tool", "id": "tool-1", "owned": True},
            {"kind": "assistant", "id": "assistant-1", "owned": True},
        ],
        "cleanup": {
            "receipt_version": "1",
            "idempotency_key": context.idempotency_key,
        },
        "metadata": {"definition_fingerprint": "sha256:abc"},
    }
    receipt_path.write_text(json.dumps(body), encoding="utf-8")

    receipt = validate_provision_receipt(
        receipt_path, context=context, secret_values={"VAPI_API_KEY": "test-key"}
    )
    assert receipt.target.id == "assistant-1"

    body["metadata"] = {"leaked": "test-key"}
    receipt_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ProviderLifecycleError, match="contains_secret"):
        validate_provision_receipt(
            receipt_path,
            context=context,
            secret_values={"VAPI_API_KEY": "test-key"},
        )


def test_attempt_scope_requires_declared_world_routing(tmp_path: Path) -> None:
    (tmp_path / "alk.yaml").write_text(
        """
schema_version: "1"
provider:
  type: retell
  scope: attempt
  provision: {command: "python provision.py"}
  destroy: {command: "python destroy.py"}
  required_secrets: [RETELL_API_KEY]
  public_capability: target_http
""",
        encoding="utf-8",
    )
    with pytest.raises(ProviderLifecycleError, match="world_routing"):
        load_provider_manifest(tmp_path)


def test_lifecycle_log_redacts_secrets_and_signed_urls(tmp_path: Path) -> None:
    context = _context()
    invocation = build_lifecycle_invocation(
        command=load_provider_manifest(
            _write_minimal_manifest(tmp_path)
        ).provider.provision,
        source_directory=tmp_path,
        lifecycle_directory=tmp_path / "lifecycle",
        context_path=tmp_path / "lifecycle" / "context.json",
        context=context,
        required_secret_values={"VAPI_API_KEY": "private-provider-key"},
        output_relative_path="provider-target.json",
    )
    log_path = tmp_path / "lifecycle.log"

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(f"key=private-provider-key callback={context.public_base_url}"),
            stderr="",
        )

    assert run_lifecycle_invocation(invocation, log_path=log_path, run=fake_run) == 0
    log = log_path.read_text(encoding="utf-8")
    assert "private-provider-key" not in log
    assert context.public_base_url not in log
    assert "[REDACTED]" in log


def _write_minimal_manifest(root: Path) -> Path:
    (root / "alk.yaml").write_text(
        """
schema_version: "1"
provider:
  type: vapi
  provision: {command: "python provision.py"}
  destroy: {command: "python destroy.py"}
  required_secrets: [VAPI_API_KEY]
  public_capability: target_http
""",
        encoding="utf-8",
    )
    return root
