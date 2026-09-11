"""Hosted-runner job contracts (plan §9.1).

A ``StartRunnerJob`` is the serializable unit the platform hands to a
``simulation-runner`` worker. It embeds an immutable ``SimulationSpec`` plus the
result-sink target; it carries only ``SecretRef``s, never resolved secrets (the
runner resolves those into the child process environment). The child process
(``fi.simulate.hosted.child_entrypoint``) consumes exactly this model.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field, JsonValue, model_validator

from fi.simulate.runtime.spec import SecretRef, SimulationSpec

RUNNER_JOB_SCHEMA_VERSION = "futureagi.runner-job.v1"


class RunnerMode(str, Enum):
    """Execution mode the child selects an engine for. Only the SIP mode
    leases a phone-number slot; chat and WebRTC never touch the pool."""

    CHAT = "chat"
    VOICE_WEBRTC = "voice_webrtc"
    VOICE_SIP = "voice_sip"

    @property
    def needs_phone(self) -> bool:
        return self is RunnerMode.VOICE_SIP

    @property
    def is_voice(self) -> bool:
        return self in {RunnerMode.VOICE_WEBRTC, RunnerMode.VOICE_SIP}


class ResultSinkConfig(BaseModel):
    """Where the child submits results. ``test_execution_id`` is set for hosted
    runs (the platform pre-creates the execution); leaving it unset preserves
    the local create-then-submit behavior."""

    api_url: str | None = None
    run_test_id: str | None = None
    test_execution_id: str | None = None
    root_directory: str | None = None
    secret_refs: dict[str, SecretRef] = Field(default_factory=dict)


class VoiceRunConfig(BaseModel):
    """Voice runs use ``run_voice_simulation`` (LiveKit), not the chat
    ``SimulationRunner``. This carries the typed inputs as JSON-round-trippable
    dicts the child hydrates into ``AgentDefinition`` / ``LiveKitSimulatorRuntime``
    / ``Scenario`` / ``SimulatorAgentDefinition``. ``transport.kind`` on the
    agent definition selects webrtc vs sip."""

    agent_definition: dict[str, JsonValue]
    scenario: dict[str, JsonValue]
    livekit_runtime: dict[str, JsonValue] | None = None
    simulator: dict[str, JsonValue] | None = None
    params: dict[str, JsonValue] = Field(default_factory=dict)


class StartRunnerJob(BaseModel):
    schema_version: str = RUNNER_JOB_SCHEMA_VERSION
    job_id: str
    mode: RunnerMode = RunnerMode.CHAT
    spec: SimulationSpec | None = None
    voice: VoiceRunConfig | None = None
    sink: ResultSinkConfig = Field(default_factory=ResultSinkConfig)
    job_token_env: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "StartRunnerJob":
        if self.schema_version != RUNNER_JOB_SCHEMA_VERSION:
            raise ValueError(f"runner_job_version_unsupported: {self.schema_version}")
        if self.mode is RunnerMode.CHAT and self.spec is None:
            raise ValueError("chat runner job requires a spec")
        if self.mode.is_voice and self.voice is None:
            raise ValueError(f"{self.mode.value} runner job requires a voice config")
        return self


class RunnerJobPhase(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def terminal(self) -> bool:
        return self in {
            RunnerJobPhase.COMPLETED,
            RunnerJobPhase.FAILED,
            RunnerJobPhase.CANCELED,
        }


class RunnerJobHandle(BaseModel):
    job_id: str
    run_id: str
    pid: int | None = None
    run_directory: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RunnerJobStatus(BaseModel):
    job_id: str
    phase: RunnerJobPhase
    detail: str | None = None
    report_hash: str | None = None
    submission_status: str | None = None
    updated_at: datetime


class RunnerReconcileResult(BaseModel):
    reconciled: bool
    orphan_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class HostedRunnerPort(Protocol):
    """Scheduler-neutral port Temporal invokes (plan §9.1)."""

    async def start(self, request: StartRunnerJob) -> RunnerJobHandle: ...

    async def status(self, handle: RunnerJobHandle) -> RunnerJobStatus: ...

    async def cancel(self, handle: RunnerJobHandle) -> None: ...

    async def reconcile(self, handle: RunnerJobHandle) -> RunnerReconcileResult: ...


__all__ = [
    "RUNNER_JOB_SCHEMA_VERSION",
    "HostedRunnerPort",
    "ResultSinkConfig",
    "RunnerJobHandle",
    "RunnerJobPhase",
    "RunnerJobStatus",
    "RunnerMode",
    "RunnerReconcileResult",
    "StartRunnerJob",
]
