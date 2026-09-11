"""Hosted-runner surface: the platform-triggered execution path for the SDK.

The platform dispatches a ``StartRunnerJob`` to a ``simulation-runner`` worker,
which spawns ``child_entrypoint`` to run the released SDK and submit results
through the existing ingestion API. See the plan doc §9.
"""

from __future__ import annotations

from .job import (
    RUNNER_JOB_SCHEMA_VERSION,
    HostedRunnerPort,
    ResultSinkConfig,
    RunnerJobHandle,
    RunnerJobPhase,
    RunnerJobStatus,
    RunnerMode,
    RunnerReconcileResult,
    StartRunnerJob,
)

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
