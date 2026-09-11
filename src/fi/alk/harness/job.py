"""Versioned job and runtime contracts for the complete ALK harness workflow.

These contracts are control-plane neutral.  A local CLI creates the same ``HarnessJob`` as the
platform hosted-job API.  The platform may enqueue it, but only an ALK executor interprets its
stages.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field, JsonValue, model_validator

from fi.simulate.runtime.spec import RuntimeIsolation, RuntimeRequirements, SecretRef

from .github import parse_github_location

HARNESS_JOB_SCHEMA_VERSION = "futureagi.harness-job.v1"
MAX_HOSTED_SCENARIO_COUNT = 200


class ExecutionMode(str, Enum):
    LOCAL = "local"
    HOSTED = "hosted"


class SourceKind(str, Enum):
    LOCAL_REPOSITORY = "local_repository"
    GITHUB = "github"
    ARCHIVE = "archive"
    IMAGE = "image"
    REMOTE = "remote"
    PROVIDER = "provider"


class SourceVisibility(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"


class RepositorySource(BaseModel):
    kind: SourceKind
    local_path: str | None = None
    installation_id: str | None = None
    repository: str | None = None
    ref: str | None = None
    commit_sha: str | None = None
    visibility: SourceVisibility = SourceVisibility.PRIVATE
    archive_artifact_id: str | None = None
    image: str | None = None
    endpoint: str | None = None

    @model_validator(mode="after")
    def _required_locator(self) -> "RepositorySource":
        # A connected provider agent is itself the source of truth. Its ID lives
        # in AgentConnection.config and its definition is fetched with the
        # run-scoped provider credential during authoring, so no repository
        # locator is required.
        if self.kind is SourceKind.PROVIDER:
            return self
        if self.kind is SourceKind.GITHUB and self.repository:
            location = parse_github_location(self.repository)
            if self.ref and location.ref and self.ref != location.ref:
                raise ValueError("github_ref_conflicts_with_repository_url")
            object.__setattr__(self, "repository", location.repository)
            object.__setattr__(self, "ref", location.ref or self.ref)
        required = {
            SourceKind.LOCAL_REPOSITORY: self.local_path,
            SourceKind.GITHUB: self.repository
            and (self.visibility is SourceVisibility.PUBLIC or self.installation_id),
            SourceKind.ARCHIVE: self.archive_artifact_id,
            SourceKind.IMAGE: self.image,
            SourceKind.REMOTE: self.endpoint,
        }[self.kind]
        if not required:
            raise ValueError(f"source_locator_missing: {self.kind.value}")
        if self.kind is SourceKind.GITHUB and self.commit_sha:
            normalized = self.commit_sha.lower()
            if len(normalized) != 40 or any(
                char not in "0123456789abcdef" for char in normalized
            ):
                raise ValueError("github_commit_sha_invalid")
        return self


class AgentConnection(BaseModel):
    connector: str
    mode: "ProviderExecutionMode | None" = None
    config: dict[str, JsonValue] = Field(default_factory=dict)
    secret_refs: dict[str, SecretRef] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _provider_mode_is_explicit_and_safe(self) -> "AgentConnection":
        connector = self.connector.strip().lower()
        provider_connector = "retell" if connector == "retell_chat" else connector
        if self.mode is ProviderExecutionMode.ENVIRONMENT_BACKED:
            if connector == "retell_chat":
                raise ValueError("retell_chat_environment_backed_not_supported")
            if provider_connector not in {"vapi", "retell"}:
                raise ValueError("environment_backed_requires_vapi_or_retell")
            manifest = str(self.config.get("lifecycle_manifest") or "alk.yaml")
            if manifest.startswith("/") or ".." in manifest.split("/"):
                raise ValueError("provider_lifecycle_manifest_must_be_source_relative")
            forbidden = {"assistant_id", "agent_id"} & set(self.config)
            if forbidden:
                raise ValueError(
                    "environment_backed_target_is_provision_output: "
                    + ", ".join(sorted(forbidden))
                )
        elif self.mode is ProviderExecutionMode.PROVIDER_IMPORT:
            if provider_connector not in {"vapi", "retell"}:
                raise ValueError("provider_import_requires_vapi_or_retell")
            target_key = {"vapi": "assistant_id", "retell": "agent_id"}[
                provider_connector
            ]
            if not str(self.config.get(target_key) or "").strip():
                raise ValueError(f"provider_import_requires_{target_key}")
            for path_key in ("event_path", "tool_path"):
                value = str(self.config.get(path_key) or "")
                if value and (
                    not value.startswith("/")
                    or value.startswith("//")
                    or ".." in value.split("/")
                ):
                    raise ValueError(f"provider_import_{path_key}_invalid")
        elif self.mode is ProviderExecutionMode.CONNECT_ONLY:
            target_key = {"vapi": "assistant_id", "retell": "agent_id"}.get(
                provider_connector
            )
            if target_key and not str(self.config.get(target_key) or "").strip():
                raise ValueError(f"connect_only_requires_{target_key}")
        elif self.mode is not None and provider_connector not in {"vapi", "retell"}:
            raise ValueError("provider_mode_only_supported_for_vapi_or_retell")
        return self


class ProviderExecutionMode(str, Enum):
    """How a provider-hosted target enters a harness run.

    ``None`` on :class:`AgentConnection` preserves existing LiveKit/HTTP jobs and legacy
    connector-only payloads. New Vapi/Retell submissions must choose one of these modes.
    """

    CONNECT_ONLY = "connect_only"
    ENVIRONMENT_BACKED = "environment_backed"
    PROVIDER_IMPORT = "provider_import"


class ArtifactLevel(str, Enum):
    METADATA_ONLY = "metadata-only"
    TRACES = "traces"
    TRACES_AND_RECORDINGS = "traces-and-recordings"
    FULL = "full"
    LOCAL_ONLY = "local-only"


class HarnessArtifactPolicy(BaseModel):
    level: ArtifactLevel = ArtifactLevel.TRACES
    retention_days: int | None = Field(default=30, ge=1, le=3650)
    allow_bundle_download: bool = False
    max_artifact_bytes: int = Field(default=1_073_741_824, ge=0)


class SandboxSecurityPolicy(BaseModel):
    """Security invariants a provider must satisfy before executing customer code."""

    untrusted_source: bool = True
    read_only_source: bool = True
    allow_privileged: bool = False
    allow_host_runtime_control: bool = False
    allowed_egress_domains: list[str] = Field(default_factory=list)


class HarnessRetryPolicy(BaseModel):
    """Conservative job retry policy; agent behavior is intentionally absent."""

    max_infrastructure_attempts: int = Field(default=2, ge=1, le=5)
    initial_backoff_seconds: float = Field(default=1.0, ge=0, le=60)
    max_backoff_seconds: float = Field(default=15.0, ge=0, le=300)
    retryable_domains: list[str] = Field(
        default_factory=lambda: ["infrastructure", "connectivity"]
    )

    @model_validator(mode="after")
    def _valid_retry_policy(self) -> HarnessRetryPolicy:
        allowed = {"infrastructure", "connectivity", "platform_sync"}
        unsupported = set(self.retryable_domains) - allowed
        if unsupported:
            raise ValueError("retry_domain_unsafe: " + ", ".join(sorted(unsupported)))
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("retry_backoff_invalid")
        return self


class HarnessJob(BaseModel):
    schema_version: str = HARNESS_JOB_SCHEMA_VERSION
    job_id: str
    run_id: str
    execution: ExecutionMode
    source: RepositorySource
    agent: AgentConnection
    scenario_count: int = Field(default=10, ge=1, le=1000)
    seed: int | None = None
    runtime: RuntimeRequirements = Field(default_factory=RuntimeRequirements)
    security: SandboxSecurityPolicy = Field(default_factory=SandboxSecurityPolicy)
    retry: HarnessRetryPolicy = Field(default_factory=HarnessRetryPolicy)
    artifacts: HarnessArtifactPolicy = Field(default_factory=HarnessArtifactPolicy)
    platform_run_id: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_job(self) -> HarnessJob:
        if self.schema_version != HARNESS_JOB_SCHEMA_VERSION:
            raise ValueError(f"harness_job_version_unsupported: {self.schema_version}")
        if (
            self.execution is ExecutionMode.LOCAL
            and self.source.kind is SourceKind.GITHUB
            and (
                self.source.visibility is not SourceVisibility.PUBLIC
                or self.source.installation_id
            )
        ):
            # A local runner may clone public source, but it must never receive a platform
            # GitHub installation credential. Private source is acquired by a hosted provider.
            raise ValueError("local_job_cannot_use_private_platform_github_source")
        if (
            self.execution is ExecutionMode.HOSTED
            and self.source.kind is SourceKind.LOCAL_REPOSITORY
        ):
            raise ValueError("hosted_job_cannot_use_local_path")
        if self.execution is ExecutionMode.HOSTED:
            if self.source.kind is SourceKind.IMAGE:
                raise ValueError("image_source_not_hosted")
            if self.source.kind is SourceKind.GITHUB and not self.source.commit_sha:
                raise ValueError("github_commit_sha_required")
            if self.scenario_count > MAX_HOSTED_SCENARIO_COUNT:
                raise ValueError("hosted_scenario_count_out_of_range")
            if self.runtime.isolation is not RuntimeIsolation.DEDICATED_VM:
                raise ValueError("hosted_isolation_must_be_dedicated_vm")
            if self.runtime.parallelism > self.runtime.cpu_units:
                raise ValueError("hosted_parallelism_exceeds_cpu")
            if self.artifacts.level is ArtifactLevel.LOCAL_ONLY:
                raise ValueError("local_only_not_hosted")
            for alias, reference in self.agent.secret_refs.items():
                expected_manager = (
                    "platform-config"
                    if reference.purpose == "simulator_provider"
                    else "platform-vault"
                )
                if reference.manager != expected_manager:
                    raise ValueError(f"hosted_secret_manager_unsupported: {alias}")
                if reference.purpose not in {
                    "target_provider",
                    "simulator_provider",
                }:
                    raise ValueError(f"hosted_secret_purpose_invalid: {alias}")
            if self.security.allow_privileged:
                raise ValueError("hosted_privileged_execution_forbidden")
            if self.security.allow_host_runtime_control:
                raise ValueError("hosted_runtime_control_forbidden")
            if not self.security.read_only_source:
                raise ValueError("hosted_source_must_be_read_only")
        _reject_secret_fields(self.model_dump(exclude={"agent": {"secret_refs"}}))
        return self


class HarnessStage(str, Enum):
    QUEUED = "queued"
    ACQUIRING_SOURCE = "acquiring_source"
    UNDERSTANDING_AGENT = "understanding_agent"
    GENERATING_ENVIRONMENT = "generating_environment"
    BUILDING_ENVIRONMENT = "building_environment"
    VALIDATING_ENVIRONMENT = "validating_environment"
    GENERATING_DATA = "generating_data"
    GENERATING_SCENARIOS = "generating_scenarios"
    VALIDATING_SCENARIOS = "validating_scenarios"
    CONNECTING_AGENT = "connecting_agent"
    RUNNING = "running"
    GRADING = "grading"
    UPLOADING_ARTIFACTS = "uploading_artifacts"
    CLEANING_UP = "cleaning_up"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELED}


class FailureDomain(str, Enum):
    AGENT = "agent"
    SIMULATOR = "simulator"
    ENVIRONMENT = "environment"
    CONNECTIVITY = "connectivity"
    INFRASTRUCTURE = "infrastructure"
    GRADING = "grading"
    ARTIFACT = "artifact"
    PLATFORM_SYNC = "platform_sync"


class FailureOwner(str, Enum):
    CUSTOMER_AGENT = "customer_agent"
    FUTUREAGI = "futureagi"


class CustomerFailure(BaseModel):
    """Small, ownership-correct failure surface safe for customer-facing APIs."""

    category: str
    owner: FailureOwner
    code: str
    message: str
    retryable: bool = False


class HarnessFailure(BaseModel):
    domain: FailureDomain
    stage: HarnessStage
    code: str
    message: str
    retryable: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def owner(self) -> FailureOwner:
        return (
            FailureOwner.CUSTOMER_AGENT
            if self.domain is FailureDomain.AGENT
            else FailureOwner.FUTUREAGI
        )

    def for_customer(self) -> CustomerFailure:
        """Do not present a harness subsystem failure as a customer-agent failure."""
        if self.owner is FailureOwner.CUSTOMER_AGENT:
            return CustomerFailure(
                category="agent_failure",
                owner=self.owner,
                code=self.code,
                message=self.message,
                retryable=False,
            )
        return CustomerFailure(
            category="system_failure",
            owner=self.owner,
            code=self.code,
            message=(
                "FutureAGI could not complete this run. The failure has been recorded "
                "for diagnosis; the submitted agent has not been marked as failed."
            ),
            retryable=self.retryable,
        )


class HarnessJobStatus(BaseModel):
    job_id: str
    run_id: str
    stage: HarnessStage
    updated_at: datetime
    detail: str | None = None
    failure: HarnessFailure | None = None
    environment_digest: str | None = None
    scenario_set_digest: str | None = None
    completed_scenarios: int = Field(default=0, ge=0)
    total_scenarios: int = Field(default=0, ge=0)
    attempt: int = Field(default=1, ge=1)


class HostedHarnessPort(Protocol):
    """Scheduler-neutral interface implemented by the hosted ALK sandbox fleet."""

    async def submit(self, job: HarnessJob) -> HarnessJobStatus: ...

    async def status(self, job_id: str) -> HarnessJobStatus: ...

    async def cancel(self, job_id: str) -> None: ...


_SECRET_NAMES = {
    "api_key",
    "api_secret",
    "authorization",
    "credential",
    "credentials",
    "password",
    "private_key",
    "secret",
    "token",
}


def _reject_secret_fields(value: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            current = (*path, str(key))
            if normalized in _SECRET_NAMES and item not in (None, "", {}, []):
                raise ValueError("resolved_secret_forbidden: " + ".".join(current))
            _reject_secret_fields(item, current)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_fields(item, (*path, str(index)))


__all__ = [
    "HARNESS_JOB_SCHEMA_VERSION",
    "AgentConnection",
    "ArtifactLevel",
    "ExecutionMode",
    "CustomerFailure",
    "FailureDomain",
    "FailureOwner",
    "HarnessArtifactPolicy",
    "HarnessFailure",
    "HarnessJob",
    "HarnessJobStatus",
    "HarnessRetryPolicy",
    "HarnessStage",
    "HostedHarnessPort",
    "RepositorySource",
    "SandboxSecurityPolicy",
    "SourceKind",
    "SourceVisibility",
]
