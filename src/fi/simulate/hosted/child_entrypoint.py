"""Child process a ``simulation-runner`` worker spawns per hosted job.

    python -m fi.simulate.hosted.child_entrypoint <job.json> [--status-file PATH]

It runs the released SDK for the job's mode and submits results through
``FutureAGIResultSink``. It is the only place hosted execution differs from a
local run — the simulation itself is the same ``SimulationRunner``/engine code.

Lifecycle is reported as newline-delimited JSON ``RunnerJobStatus`` objects, both
to stdout (the worker tails these for Temporal heartbeats) and to an optional
status file. SIGTERM triggers a graceful cancel + cleanup.

Slice 1 wires the chat mode only; the voice modes raise until their slices land.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fi.simulate.results.futureagi import FutureAGIResultSink
from fi.simulate.runtime.report import SimulationReport
from fi.simulate.runtime.run import RunStatus
from fi.simulate.runtime.runner import SimulationRunner

from .job import RunnerJobPhase, RunnerJobStatus, RunnerMode, StartRunnerJob
from .targets import resolve_chat_target

if TYPE_CHECKING:
    from fi.simulate.runtime.spec import SimulationSpec

_HEARTBEAT_INTERVAL_SECONDS = 10.0
_CANCEL_GRACE_SECONDS = 30.0

logger = logging.getLogger("fi.simulate.hosted.runner")


def _job_log_fields(job: StartRunnerJob) -> dict[str, Any]:
    fields: dict[str, Any] = {"job_id": job.job_id, "mode": job.mode.value}
    if job.voice is not None:
        target = dict(job.voice.agent_definition or {}).get("target") or {}
        fields["provider"] = target.get("provider")
        dataset = dict(job.voice.scenario or {}).get("dataset") or []
        fields["cases"] = len(dataset)
    if job.sink is not None:
        fields["run_test_id"] = job.sink.run_test_id
        fields["test_execution_id"] = job.sink.test_execution_id
    return fields


class _StatusReporter:
    def __init__(self, job_id: str, status_file: Path | None) -> None:
        self._job_id = job_id
        self._status_file = status_file

    def emit(
        self,
        phase: RunnerJobPhase,
        *,
        detail: str | None = None,
        report_hash: str | None = None,
        submission_status: str | None = None,
    ) -> None:
        status = RunnerJobStatus(
            job_id=self._job_id,
            phase=phase,
            detail=detail,
            report_hash=report_hash,
            submission_status=submission_status,
            updated_at=datetime.now(timezone.utc),
        )
        line = status.model_dump_json()
        print(line, flush=True)
        if self._status_file is not None:
            with self._status_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _load_job(path: Path) -> StartRunnerJob:
    return StartRunnerJob.model_validate_json(path.read_text(encoding="utf-8"))


def _build_sink(job: StartRunnerJob) -> FutureAGIResultSink:
    root = job.sink.root_directory or os.environ.get("FI_RUN_ROOT") or ".fagi/runs"
    return FutureAGIResultSink(
        root=root,
        api_url=job.sink.api_url,
        run_test_id=job.sink.run_test_id,
        test_execution_id=job.sink.test_execution_id,
    )


def _read_submission(run_directory: Path | None) -> dict[str, Any]:
    if run_directory is None:
        return {}
    submission_path = run_directory / "submission.json"
    if not submission_path.exists():
        return {}
    try:
        return json.loads(submission_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _build_voice_spec(job: StartRunnerJob) -> "SimulationSpec":
    """Translate a voice job into a ``SimulationSpec`` so the voice run flows
    through the same ``SimulationRunner`` spine as chat (plan §3). The typed voice
    inputs ride in ``environment.config`` — secret-free, since providers are
    referenced by ``*_env`` name, never raw values. ``transport.kind`` selects the
    target adapter. The DID pool is leased by the runner activity (telephone
    only), not here — the leased number arrives via the agent definition / params.
    """
    from fi.simulate.runtime import new_run_id
    from fi.simulate.runtime.spec import (
        AgentEndpointSpec,
        EnvironmentSpec,
        EvidencePolicy,
        ExecutionPolicy,
        SimulationSpec,
        SimulatorPolicySpec,
        TimeoutPolicy,
    )
    from fi.simulate.simulation.models import Scenario

    cfg = job.voice
    run_id = str((job.spec.run_id if job.spec else None) or new_run_id())
    params = dict(cfg.params or {})
    transport = (dict(cfg.agent_definition or {}).get("transport") or {})
    transport_kind = transport.get("kind") or "livekit"

    # The runner's outer deadline must clear the voice call's own budget.
    run_seconds = max(
        300.0,
        float(params.get("max_seconds", 45.0))
        + float(params.get("connect_timeout", 15.0))
        + float(params.get("readiness_timeout", 30.0))
        + float(params.get("cleanup_timeout", 30.0))
        + 60.0,
    )

    return SimulationSpec(
        run_id=run_id,
        environment=EnvironmentSpec(
            adapter="voice",
            world_kind="voice",
            config={
                "agent_definition": cfg.agent_definition,
                "livekit_runtime": cfg.livekit_runtime,
                "simulator": cfg.simulator,
                "params": cfg.params,
            },
        ),
        target=AgentEndpointSpec(adapter=transport_kind),
        simulator=SimulatorPolicySpec(adapter="livekit_simulator"),
        scenario=Scenario.model_validate(cfg.scenario),
        execution=ExecutionPolicy(timeout=TimeoutPolicy(run_seconds=run_seconds)),
        evidence=EvidencePolicy(),
    )


async def _heartbeat(reporter: _StatusReporter) -> None:
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        reporter.emit(RunnerJobPhase.RUNNING, detail="heartbeat")


async def _execute(job: StartRunnerJob, reporter: _StatusReporter) -> int:
    reporter.emit(RunnerJobPhase.PREPARING)
    logger.info("hosted job start", extra=_job_log_fields(job))
    sink = _build_sink(job)

    if job.mode is RunnerMode.CHAT:
        target = resolve_chat_target(job.spec)
        run_coro = SimulationRunner().run(job.spec, target=target, result_sink=sink)
    elif job.mode.is_voice:
        run_coro = SimulationRunner().run(_build_voice_spec(job), result_sink=sink)
    else:
        raise NotImplementedError(f"runner mode not wired: {job.mode.value}")

    run_task = asyncio.ensure_future(run_coro)
    heartbeat_task = asyncio.ensure_future(_heartbeat(reporter))
    reporter.emit(RunnerJobPhase.RUNNING)
    try:
        report: SimulationReport = await run_task
    except asyncio.CancelledError:
        reporter.emit(RunnerJobPhase.CANCELED, detail="cancelled")
        logger.warning("hosted job cancelled", extra={"job_id": job.job_id})
        # Cancelling this coroutine does not cancel ``run_task``; without an
        # explicit cancel ``asyncio.run`` shutdown waits on it forever and the
        # child leaks past SIGTERM.
        run_task.cancel()
        try:
            await asyncio.wait({run_task}, timeout=_CANCEL_GRACE_SECONDS)
        except asyncio.CancelledError:
            pass
        if not run_task.done():
            os._exit(2)
        raise
    finally:
        heartbeat_task.cancel()

    reporter.emit(RunnerJobPhase.FINALIZING)
    submission = _read_submission(sink.run_directory)
    submission_status = submission.get("status")
    run_completed = report.status is RunStatus.COMPLETED
    # When a submission target is configured (a hosted run), a failed or omitted
    # submission is a job failure — otherwise a broken upload reports as green.
    submission_expected = bool(job.sink.run_test_id)
    submission_ok = (not submission_expected) or submission_status == "submitted"
    completed = run_completed and submission_ok
    if completed:
        detail = None
    elif not run_completed:
        detail = report.failure.code if report.failure else "run_failed"
    else:
        detail = f"submission_{submission_status or 'missing'}"
    outcome_fields = {
        **_job_log_fields(job),
        "run_status": getattr(report.status, "value", str(report.status)),
        "submission_status": submission_status,
        "report_hash": report.report_hash,
        "detail": detail,
    }
    if completed:
        logger.info("hosted job completed", extra=outcome_fields)
    else:
        logger.error("hosted job failed", extra=outcome_fields)
    reporter.emit(
        RunnerJobPhase.COMPLETED if completed else RunnerJobPhase.FAILED,
        detail=detail,
        report_hash=report.report_hash,
        submission_status=submission_status,
    )
    return 0 if completed else 1


def _install_cancellation(run_task_holder: dict[str, asyncio.Task[int]]) -> None:
    loop = asyncio.get_running_loop()

    def _cancel() -> None:
        task = run_task_holder.get("task")
        if task is not None and not task.done():
            task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _cancel)
        except (NotImplementedError, ValueError):
            pass


async def _main_async(job: StartRunnerJob, reporter: _StatusReporter) -> int:
    holder: dict[str, asyncio.Task[int]] = {}
    _install_cancellation(holder)
    task = asyncio.ensure_future(_execute(job, reporter))
    holder["task"] = task
    try:
        return await task
    except asyncio.CancelledError:
        return 2


def _configure_logging() -> None:
    """The child runs with no logging config, so INFO seams (job start/outcome,
    engine dispatch/join/stop_reason) were silently dropped by the WARNING-level
    lastResort handler and never reached the runner's log capture."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.WARNING)
    logging.getLogger("fi.simulate").setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(prog="fi.simulate.hosted.child_entrypoint")
    parser.add_argument("job", help="path to the StartRunnerJob JSON file")
    parser.add_argument("--status-file", default=None)
    args = parser.parse_args(argv)

    job = _load_job(Path(args.job))
    status_file = Path(args.status_file) if args.status_file else None
    reporter = _StatusReporter(job.job_id, status_file)

    try:
        return asyncio.run(_main_async(job, reporter))
    except Exception as exc:  # noqa: BLE001
        logger.exception("hosted job crashed", extra={"job_id": job.job_id})
        reporter.emit(
            RunnerJobPhase.FAILED, detail=f"{type(exc).__name__}: {exc}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
