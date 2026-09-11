from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, JsonValue


class FailureStage(str, Enum):
    PLANNING = "planning"
    PREPARING = "preparing"
    READINESS = "readiness"
    RUNNING = "running"
    FINALIZING = "finalizing"
    SUBMITTING = "submitting"
    CLEANUP = "cleanup"


class SimulationFailure(BaseModel):
    stage: FailureStage
    code: str
    message: str
    retryable: bool = False
    provider: str | None = None
    external_ref: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)
