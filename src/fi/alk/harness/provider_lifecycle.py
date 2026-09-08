"""Explicit repository-owned lifecycle contract for Vapi and Retell targets.

ALK does not translate provider definitions. A submitted repository either points at an existing
target (``connect_only`` on ``HarnessJob.agent``) or declares how its own code creates and removes
an attempt-owned target in ``alk.yaml``. This module owns only the latter contract and its
non-secret receipt validation; execution is deliberately injected so the hosted runtime can run
the command under the same unprivileged world identity as the customer processes.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


class ProviderLifecycleError(RuntimeError):
    """A repository lifecycle declaration or receipt is missing, unsafe, or inconsistent."""


class ProviderType(str, Enum):
    VAPI = "vapi"
    RETELL = "retell"


class ProviderScope(str, Enum):
    ATTEMPT = "attempt"
    WORLD = "world"


class LifecycleCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: list[str] = Field(min_length=1)
    timeout_seconds: float = Field(default=120.0, gt=0, le=1800)

    @field_validator("command", mode="before")
    @classmethod
    def _parse_command(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            try:
                parsed = shlex.split(value)
            except ValueError as exc:
                raise ValueError("provider_lifecycle_command_invalid") from exc
            return parsed
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return [str(part) for part in value]
        raise ValueError("provider_lifecycle_command_must_be_string_or_argv")

    @model_validator(mode="after")
    def _no_shell_indirection(self) -> "LifecycleCommand":
        if not self.command or not self.command[0].strip():
            raise ValueError("provider_lifecycle_command_empty")
        if self.command[0] in {"sh", "bash", "zsh", "cmd", "powershell"}:
            raise ValueError("provider_lifecycle_shell_wrapper_forbidden")
        return self


class ProviderLifecycleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ProviderType
    scope: ProviderScope = ProviderScope.WORLD
    provision: LifecycleCommand
    destroy: LifecycleCommand
    output: str = "provider-target.json"
    required_secrets: list[str] = Field(default_factory=list)
    process: str | None = None
    public_capability: str
    event_path: str = "/"
    tool_path: str = "/"
    world_routing: str | None = None

    @field_validator("required_secrets")
    @classmethod
    def _environment_names(cls, values: list[str]) -> list[str]:
        invalid = [
            name for name in values if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name)
        ]
        if invalid:
            raise ValueError(
                "provider_required_secret_name_invalid: " + ", ".join(invalid)
            )
        if len(values) != len(set(values)):
            raise ValueError("provider_required_secrets_duplicate")
        return values

    @field_validator("output")
    @classmethod
    def _safe_output(cls, value: str) -> str:
        # The architecture document used /alk/output as a logical path. The process guest owns
        # /work, so normalize that spelling into its job-private lifecycle directory rather than
        # granting customer code an arbitrary absolute write path.
        if value.startswith("/alk/output/"):
            value = value.removeprefix("/alk/output/")
        path = PurePosixPath(value)
        if path.is_absolute() or not value or ".." in path.parts:
            raise ValueError("provider_output_must_be_lifecycle_relative")
        return path.as_posix()

    @field_validator("public_capability")
    @classmethod
    def _capability_slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
            raise ValueError("provider_public_capability_invalid")
        return value

    @field_validator("event_path", "tool_path")
    @classmethod
    def _public_path(cls, value: str) -> str:
        if (
            not value.startswith("/")
            or value.startswith("//")
            or ".." in value.split("/")
        ):
            raise ValueError("provider_public_path_invalid")
        return value

    @model_validator(mode="after")
    def _attempt_scope_has_routing(self) -> "ProviderLifecycleSpec":
        if (
            self.scope is ProviderScope.ATTEMPT
            and not (self.world_routing or "").strip()
        ):
            raise ValueError("attempt_scoped_provider_requires_world_routing")
        return self


class ProviderRepositoryManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["1"]
    provider: ProviderLifecycleSpec


class ProviderContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    attempt_id: str = Field(min_length=1)
    world_id: str = Field(min_length=1)
    provider: ProviderType
    public_base_url: str
    event_url: str
    tool_base_url: str
    provider_resource_prefix: str
    idempotency_key: str
    expires_at: datetime


class ProviderTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["assistant", "voice_agent", "chat_agent"]
    id: str = Field(min_length=1)
    version: str | None = None


class ProviderResource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    id: str = Field(min_length=1)
    owned: bool


class ProviderCleanupReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_version: Literal["1"] = "1"
    idempotency_key: str = Field(min_length=1)


class ProviderProvisionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    provider: ProviderType
    attempt_id: str
    world_id: str
    target: ProviderTarget
    resources: list[ProviderResource] = Field(default_factory=list)
    cleanup: ProviderCleanupReceipt
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _provider_target_kind(self) -> "ProviderProvisionReceipt":
        expected = {
            ProviderType.VAPI: {"assistant"},
            ProviderType.RETELL: {"voice_agent", "chat_agent"},
        }[self.provider]
        if self.target.kind not in expected:
            raise ValueError(
                "provider_target_kind_invalid: "
                f"{self.provider.value} requires one of {', '.join(sorted(expected))}"
            )
        if not any(
            resource.owned
            and resource.id == self.target.id
            and resource.kind == self.target.kind
            for resource in self.resources
        ):
            raise ValueError("provider_target_missing_owned_resource")
        return self


def load_provider_manifest(
    source: str | Path, relative_path: str = "alk.yaml"
) -> ProviderRepositoryManifest:
    root = Path(source).resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents or not path.is_file():
        raise ProviderLifecycleError(
            f"provider_lifecycle_manifest_missing: {relative_path}"
        )
    try:
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ProviderRepositoryManifest.model_validate(body)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ProviderLifecycleError(
            f"provider_lifecycle_manifest_invalid: {exc}"
        ) from exc


def validate_provision_receipt(
    path: str | Path,
    *,
    context: ProviderContext,
    secret_values: Mapping[str, str] | None = None,
) -> ProviderProvisionReceipt:
    receipt_path = Path(path)
    try:
        raw = receipt_path.read_text(encoding="utf-8")
        body = json.loads(raw)
        receipt = ProviderProvisionReceipt.model_validate(body)
    except (OSError, ValueError) as exc:
        raise ProviderLifecycleError(
            f"provider_provision_receipt_invalid: {exc}"
        ) from exc
    if receipt.provider is not context.provider:
        raise ProviderLifecycleError("provider_provision_receipt_provider_mismatch")
    if receipt.attempt_id != context.attempt_id or receipt.world_id != context.world_id:
        raise ProviderLifecycleError("provider_provision_receipt_scope_mismatch")
    if receipt.cleanup.idempotency_key != context.idempotency_key:
        raise ProviderLifecycleError("provider_provision_receipt_idempotency_mismatch")
    forbidden_values = [
        *(secret_values or {}).values(),
        context.public_base_url,
        context.event_url,
        context.tool_base_url,
    ]
    for secret in forbidden_values:
        if secret and secret in raw:
            raise ProviderLifecycleError(
                "provider_provision_receipt_contains_secret_or_capability"
            )
    return receipt


@dataclass(frozen=True)
class LifecycleInvocation:
    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    timeout_seconds: float
    output_path: Path


LifecycleInvoker = Callable[[LifecycleInvocation], Awaitable[int]]


def build_lifecycle_invocation(
    *,
    command: LifecycleCommand,
    source_directory: Path,
    lifecycle_directory: Path,
    context_path: Path,
    context: ProviderContext,
    required_secret_values: Mapping[str, str],
    output_relative_path: str,
    receipt_path: Path | None = None,
) -> LifecycleInvocation:
    missing = sorted(
        name for name in required_secret_values if not required_secret_values.get(name)
    )
    if missing:
        raise ProviderLifecycleError(
            "provider_required_secrets_missing: " + ", ".join(missing)
        )
    output_path = lifecycle_directory / output_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    executable_path = os.pathsep.join(
        filter(
            None,
            (
                str(source_directory / ".venv" / "bin"),
                str(source_directory / "node_modules" / ".bin"),
                os.environ.get("PATH", ""),
            ),
        )
    )
    environment = {
        # Use the selected process's writable build, including its per-process venv/node
        # installation. A bare `python`/`node` in alk.yaml must not depend on the immutable
        # hosted snapshot's packages.
        "PATH": executable_path,
        "ALK_ATTEMPT_ID": context.attempt_id,
        "ALK_WORLD_ID": context.world_id,
        "ALK_PROVIDER": context.provider.value,
        "ALK_PROVIDER_CONTEXT": str(context_path),
        "ALK_PROVIDER_OUTPUT": str(output_path),
        "ALK_PUBLIC_BASE_URL": context.public_base_url,
        "ALK_EVENT_URL": context.event_url,
        "ALK_TOOL_BASE_URL": context.tool_base_url,
        **required_secret_values,
    }
    if receipt_path is not None:
        environment["ALK_PROVIDER_RECEIPT"] = str(receipt_path)
    return LifecycleInvocation(
        argv=tuple(command.command),
        cwd=source_directory,
        environment=environment,
        timeout_seconds=command.timeout_seconds,
        output_path=output_path,
    )


def run_lifecycle_invocation(
    invocation: LifecycleInvocation,
    *,
    log_path: Path,
    run: Callable[..., Any],
) -> int:
    """Execute one bounded lifecycle command without a shell and without logging its env."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    }
    environment.update(invocation.environment)
    try:
        result = run(
            list(invocation.argv),
            cwd=invocation.cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=invocation.timeout_seconds,
            check=False,
        )
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        raise ProviderLifecycleError("provider_lifecycle_command_timed_out") from exc
    except OSError as exc:
        raise ProviderLifecycleError(
            f"provider_lifecycle_command_unavailable: {exc}"
        ) from exc
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    combined = stdout + ("\n" if stdout and stderr else "") + stderr
    # Repository lifecycle code is untrusted and may accidentally print its environment. Scrub
    # every injected value (provider keys, signed callback URLs and run identifiers) before the
    # bounded diagnostic log is persisted or uploaded.
    for sensitive in sorted(
        {value for value in invocation.environment.values() if len(value) >= 4},
        key=len,
        reverse=True,
    ):
        combined = combined.replace(sensitive, "[REDACTED]")
    log_path.write_text(
        combined[-16000:],
        encoding="utf-8",
    )
    return int(getattr(result, "returncode", 1))
