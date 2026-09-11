from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum

from pydantic import BaseModel, Field, JsonValue, model_validator

from fi.simulate._hashing import content_hash
from fi.simulate.evidence import EvidenceSourceSpec
from fi.simulate.simulation.models import Scenario

SIMULATION_SPEC_SCHEMA_VERSION = "futureagi.simulation-spec.v1"

_SECRET_KEYS = {
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


class SecretRef(BaseModel):
    manager: str
    key: str
    version: str | None = None
    purpose: str


class AdapterSpec(BaseModel):
    adapter: str
    adapter_version: str = "1"
    config: dict[str, JsonValue] = Field(default_factory=dict)
    secret_refs: dict[str, SecretRef] = Field(default_factory=dict)


class EnvironmentSpec(AdapterSpec):
    world_kind: str


class AgentEndpointSpec(AdapterSpec):
    required_capabilities: list[str] = Field(default_factory=list)


class SimulatorPolicySpec(AdapterSpec):
    pass


class ConversationDirection(str, Enum):
    SIMULATOR_FIRST = "simulator_first"
    AGENT_FIRST = "agent_first"


class RuntimeIsolation(str, Enum):
    SHARED_RUNNER_PROCESS = "shared_runner_process"
    DEDICATED_POD = "dedicated_pod"
    DEDICATED_VM = "dedicated_vm"
    EXTERNAL = "external"


class RuntimeRequirements(BaseModel):
    isolation: RuntimeIsolation = RuntimeIsolation.SHARED_RUNNER_PROCESS
    cpu_units: int = Field(default=1, ge=1)
    memory_mb: int = Field(default=512, ge=128)
    parallelism: int = Field(default=1, ge=1, le=8)
    concurrency_weight: int = Field(default=1, ge=1)
    max_duration_seconds: int = Field(default=300, ge=1)
    network_policy: str = "live"
    # World count for the hosted harness (seam contract §1); the gateway caps
    # it at admission, so the model stays permissive beyond ge=1.
    parallelism: int = Field(default=1, ge=1)


class TimeoutPolicy(BaseModel):
    connect_seconds: float = Field(default=15.0, gt=0)
    readiness_seconds: float = Field(default=30.0, gt=0)
    run_seconds: float = Field(default=300.0, gt=0)
    finalize_seconds: float = Field(default=30.0, gt=0)
    cleanup_seconds: float = Field(default=30.0, gt=0)


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=1, ge=1)
    initial_backoff_seconds: float = Field(default=1.0, ge=0)
    max_backoff_seconds: float = Field(default=30.0, ge=0)

    @model_validator(mode="after")
    def _validate_backoff(self) -> RetryPolicy:
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "retry_policy_invalid: max backoff is below initial backoff"
            )
        return self


class CleanupPolicy(BaseModel):
    always: bool = True
    reconcile_before_create: bool = True
    orphan_cleanup: bool = True


class ExecutionPolicy(BaseModel):
    direction: ConversationDirection = ConversationDirection.SIMULATOR_FIRST
    runtime: RuntimeRequirements = Field(default_factory=RuntimeRequirements)
    timeout: TimeoutPolicy = Field(default_factory=TimeoutPolicy)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    cleanup: CleanupPolicy = Field(default_factory=CleanupPolicy)
    max_parallel_cases: int = Field(default=1, ge=1)


class EvidencePolicy(BaseModel):
    sources: list[EvidenceSourceSpec] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_source_ids(self) -> EvidencePolicy:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(
                "evidence_source_duplicate: source_id values must be unique"
            )
        return self


class ArtifactPolicy(BaseModel):
    enabled: bool = True
    record_audio: bool = False
    root_directory: str | None = None
    max_inline_bytes: int = Field(default=65_536, ge=0)
    required_types: list[str] = Field(default_factory=list)


class EvaluationRef(BaseModel):
    evaluation_id: str
    version: str | None = None
    config: dict[str, JsonValue] = Field(default_factory=dict)


class SimulationSpec(BaseModel):
    schema_version: str = SIMULATION_SPEC_SCHEMA_VERSION
    run_id: str
    environment: EnvironmentSpec
    target: AgentEndpointSpec
    simulator: SimulatorPolicySpec
    scenario: Scenario
    execution: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    evidence: EvidencePolicy = Field(default_factory=EvidencePolicy)
    artifacts: ArtifactPolicy = Field(default_factory=ArtifactPolicy)
    evaluation_refs: list[EvaluationRef] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    spec_hash: str | None = None

    def content_hash(self) -> str:
        return content_hash(self.model_dump(exclude={"spec_hash"}, exclude_none=True))

    @model_validator(mode="after")
    def _validate_and_stamp(self) -> SimulationSpec:
        if self.schema_version != SIMULATION_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"simulation_spec_version_unsupported: {self.schema_version}"
            )
        _reject_resolved_secrets(self.model_dump(exclude={"spec_hash"}))
        expected = self.content_hash()
        if self.spec_hash is not None and self.spec_hash != expected:
            raise ValueError("simulation_spec_hash_mismatch")
        object.__setattr__(self, "spec_hash", expected)
        return self


def _reject_resolved_secrets(value: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower().replace("-", "_")
            current_path = (*path, str(key))
            if name == "secret_refs":
                continue
            if name in _SECRET_KEYS and item not in (None, "", {}, []):
                raise ValueError("resolved_secret_forbidden: " + ".".join(current_path))
            _reject_resolved_secrets(item, current_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_resolved_secrets(item, (*path, str(index)))
