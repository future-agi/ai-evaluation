from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .capabilities import CapabilitySet
from fi.simulate._hashing import content_hash
from .spec import CleanupPolicy, RetryPolicy, RuntimeRequirements, TimeoutPolicy

SIMULATION_PLAN_VERSION = "futureagi.simulation-plan.v1"


class AdapterRef(BaseModel):
    name: str
    version: str
    config: dict[str, JsonValue] = Field(default_factory=dict)


class EvidencePlan(BaseModel):
    source_ids: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)


class ArtifactPlan(BaseModel):
    enabled: bool = True
    root_directory: str
    record_audio: bool = False
    required_types: list[str] = Field(default_factory=list)


class SimulationPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: str = SIMULATION_PLAN_VERSION
    plan_id: str
    run_id: str
    spec_hash: str
    environment_adapter: AdapterRef
    target_adapter: AdapterRef
    simulator_adapter: AdapterRef
    transport_adapter: AdapterRef | None = None
    negotiated_capabilities: CapabilitySet = Field(default_factory=CapabilitySet)
    runtime_requirements: RuntimeRequirements = Field(default_factory=RuntimeRequirements)
    timeout_policy: TimeoutPolicy = Field(default_factory=TimeoutPolicy)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    cleanup_policy: CleanupPolicy = Field(default_factory=CleanupPolicy)
    evidence_plan: EvidencePlan = Field(default_factory=EvidencePlan)
    artifact_plan: ArtifactPlan
    plan_hash: str | None = None

    def content_hash(self) -> str:
        return content_hash(
            self.model_dump(exclude={"plan_hash"}, exclude_none=True)
        )

    @model_validator(mode="after")
    def _validate_and_stamp(self) -> "SimulationPlan":
        if self.plan_version != SIMULATION_PLAN_VERSION:
            raise ValueError(
                f"simulation_plan_version_unsupported: {self.plan_version}"
            )
        missing = self.negotiated_capabilities.missing()
        if missing:
            raise ValueError(
                "simulation_capabilities_missing: " + ", ".join(missing)
            )
        expected = self.content_hash()
        if self.plan_hash is not None and self.plan_hash != expected:
            raise ValueError("simulation_plan_hash_mismatch")
        object.__setattr__(self, "plan_hash", expected)
        return self
