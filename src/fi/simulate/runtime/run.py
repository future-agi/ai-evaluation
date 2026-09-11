from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, JsonValue

from .failures import SimulationFailure


class RunStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    FINALIZING = "finalizing"
    SUBMITTING = "submitting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        return self in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELED,
            RunStatus.TIMED_OUT,
        }


class TestCaseStatus(str, Enum):
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"
    AGENT_UNAVAILABLE = "agent_unavailable"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"

    @property
    def terminal(self) -> bool:
        return self not in {
            TestCaseStatus.CREATED,
            TestCaseStatus.PREPARING,
            TestCaseStatus.READY,
            TestCaseStatus.RUNNING,
        }


class CleanupStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SimulationRun(BaseModel):
    run_id: str
    plan_id: str | None = None
    spec_hash: str
    status: RunStatus = RunStatus.CREATED
    cleanup_status: CleanupStatus = CleanupStatus.PENDING
    created_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    failure: SimulationFailure | None = None
    cleanup_failure: SimulationFailure | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
