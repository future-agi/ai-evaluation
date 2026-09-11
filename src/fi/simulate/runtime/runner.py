from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from fi.simulate._logging import redacted_exc_info
from fi.simulate.agent.wrapper import AgentWrapper, SimulationArtifact, SimulationEvent
from fi.simulate.artifacts import ArtifactManifest
from fi.simulate.environment import EnvironmentAdapter
import fi.simulate.environments  # noqa: F401  (registers builtin environment plugins)
from fi.simulate.registry import environment_registry
from fi.simulate.evidence import EvidenceSourceSummary
from fi.simulate.results.base import ResultSink
from fi.simulate.simulation.models import Persona

from .events import CanonicalEvent
from .failures import FailureStage, SimulationFailure
from .plan import SimulationPlan
from .planner import build_plan
from .report import SimulationReport, SimulationTestCaseResult
from .run import CleanupStatus, RunStatus
from .spec import SimulationSpec

logger = logging.getLogger(__name__)


class SimulationRunner:
    async def run(
        self,
        spec: SimulationSpec,
        *,
        target: Callable[..., Any] | AgentWrapper | Any = None,
        result_sink: ResultSink | None = None,
        artifacts: list[SimulationArtifact | dict[str, Any]] | None = None,
        events: list[SimulationEvent | dict[str, Any]] | None = None,
        environment: EnvironmentAdapter | Iterable[EnvironmentAdapter] | None = None,
        auto_execute_tools: bool = True,
        stop_when: Callable[[list[dict[str, Any]], Persona], bool] | None = None,
        agent_wrapper_kwargs: dict[str, Any] | None = None,
    ) -> SimulationReport:
        started_at = datetime.now(timezone.utc)
        plan: SimulationPlan | None = None
        try:
            plan = build_plan(spec)
            if result_sink is not None:
                result_sink.prepare(spec, plan)
            # Hosted runs stream each case to the platform the moment it finishes
            # (allocate CallExecution rows up front, PATCH by index). The sink
            # decides eligibility; a non-streaming sink (local/chat, or missing
            # config) returns False and we fall back to the batch-at-end path.
            on_case_start, on_case_complete = self._begin_streaming(
                result_sink, spec, plan
            )
            # Keep every case that finished. A deadline that fires while the plugin is still
            # tidying up used to discard cases whose conversation had already completed, and the
            # run then reported no test case at all: no turns, no transcript, and an infrastructure
            # failure for a call that worked. Held here because this is the only layer that sees
            # both the cases and the timeout.
            finished: list[SimulationTestCaseResult] = []
            # Bound to its own name before the wrapper is installed. Closing over
            # ``on_case_complete`` and then rebinding it makes the wrapper call itself: the
            # closure reads the name at call time, not the value it had at definition. It cost a
            # whole run to find, and only after cases started completing at all, because a
            # wrapper that never runs never recurses.
            report_case = on_case_complete

            async def _remember(index: int, legacy_case: Any) -> None:
                try:
                    finished.append(
                        SimulationTestCaseResult.from_legacy_case(
                            legacy_case, index=index, run_id=spec.run_id
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - never fail a case over bookkeeping
                    logger.warning(
                        "could not keep a finished case for the timeout path: %s", exc
                    )
                if report_case is not None:
                    await report_case(index, legacy_case)

            on_case_complete = _remember
            self._write_event(
                result_sink,
                CanonicalEvent.create(
                    run_id=spec.run_id,
                    test_case_id="run",
                    event_type="session.started",
                    source="runtime",
                    sequence=0,
                ),
            )
            plugin = environment_registry.create(spec.environment.adapter)
            legacy_report = await asyncio.wait_for(
                plugin.run(
                    spec,
                    target=target,
                    artifacts=artifacts,
                    events=events,
                    environment=environment,
                    auto_execute_tools=auto_execute_tools,
                    stop_when=stop_when,
                    agent_wrapper_kwargs=agent_wrapper_kwargs,
                    on_case_start=on_case_start,
                    on_case_complete=on_case_complete,
                ),
                timeout=spec.execution.timeout.run_seconds,
            )
        except asyncio.TimeoutError:
            report = self._failure_report(
                spec,
                plan=plan,
                started_at=started_at,
                status=RunStatus.TIMED_OUT,
                failure=SimulationFailure(
                    stage=FailureStage.RUNNING,
                    code="simulation_timeout",
                    message="Simulation exceeded its run deadline",
                    retryable=True,
                ),
                cases=finished,
            )
        except Exception as exc:
            stage = FailureStage.PLANNING if plan is None else FailureStage.RUNNING
            report = self._failure_report(
                spec,
                plan=plan,
                started_at=started_at,
                status=RunStatus.FAILED,
                failure=SimulationFailure(
                    stage=stage,
                    code="simulation_failed",
                    message="Simulation execution failed",
                    retryable=False,
                    details={"exception_type": type(exc).__name__},
                ),
            )
            logger.error(
                "Simulation run failed",
                exc_info=redacted_exc_info(exc),
                extra={
                    "run_id": spec.run_id,
                    "exception_type": type(exc).__name__,
                },
            )
        else:
            ended_at = datetime.now(timezone.utc)
            report = SimulationReport.from_legacy(
                legacy_report,
                run_id=spec.run_id,
                plan_id=plan.plan_id,
                spec_hash=spec.spec_hash or spec.content_hash(),
                status=RunStatus.COMPLETED,
                started_at=started_at,
                ended_at=ended_at,
                artifacts=ArtifactManifest(run_id=spec.run_id),
                evidence=[
                    EvidenceSourceSummary(
                        source_id=source.source_id,
                        adapter=source.adapter,
                        evidence_class=source.evidence_class,
                        capabilities=source.capabilities,
                    )
                    for source in spec.evidence.sources
                ],
            )
            report.cleanup_status = CleanupStatus.COMPLETED
            report = SimulationReport.model_validate(
                report.model_dump(exclude={"report_hash"})
            )
            # Environments enforce terminal conditions (plan §3): let the plugin
            # override the run status from per-case results (e.g. voice marks an
            # all-cases-failed run FAILED). Absent hook -> COMPLETED, as before.
            finalize = getattr(plugin, "finalize_run_status", None)
            if finalize is not None:
                report = finalize(report)
            self._write_event(
                result_sink,
                CanonicalEvent.create(
                    run_id=spec.run_id,
                    test_case_id="run",
                    event_type="session.ended",
                    source="runtime",
                    sequence=1,
                    payload={"status": report.status.value},
                ),
            )
        self._write_report(result_sink, report)
        return report

    def _begin_streaming(
        self,
        result_sink: ResultSink | None,
        spec: SimulationSpec,
        plan: SimulationPlan | None,
    ) -> tuple[Callable[[int], Any] | None, Callable[[int, Any], Any] | None]:
        """Open the sink's streaming session; return ``(on_case_start,
        on_case_complete)`` — either may be ``None``.

        ``on_case_complete`` converts a legacy ``TestCaseResult`` to its canonical
        form (identical to the finalized report) and submits it off the event loop
        so a slow result PATCH never blocks case concurrency. ``on_case_start``
        fires a best-effort ONGOING status ping the moment a case begins, and is
        present only when the sink exposes a ``case_started`` method. Both are
        engine-agnostic — any engine that invokes the hooks gets the behaviour.
        A streaming error is swallowed here — the case is left un-streamed and
        ``finalize`` reconciles it; a broken upload must never fail a case.
        """
        if result_sink is None:
            return None, None
        begin = getattr(result_sink, "begin_stream", None)
        submit_case = getattr(result_sink, "submit_case", None)
        if begin is None or submit_case is None:
            return None, None
        try:
            streaming = bool(begin(spec, plan))
        except Exception as exc:
            logger.error(
                "Simulation stream begin failed",
                exc_info=redacted_exc_info(exc),
                extra={"run_id": spec.run_id},
            )
            return None, None
        if not streaming:
            return None, None

        evidence = [
            EvidenceSourceSummary(
                source_id=source.source_id,
                adapter=source.adapter,
                evidence_class=source.evidence_class,
                capabilities=source.capabilities,
            )
            for source in spec.evidence.sources
        ]
        run_id = spec.run_id

        async def _on_case_complete(index: int, legacy_case: Any) -> None:
            try:
                canonical = SimulationTestCaseResult.from_legacy_case(
                    legacy_case, index=index, run_id=run_id, evidence=evidence
                )
                await asyncio.to_thread(submit_case, index, canonical)
            except Exception as exc:
                logger.error(
                    "Simulation stream case submit failed",
                    exc_info=redacted_exc_info(exc),
                    extra={"run_id": run_id, "case_index": index},
                )

        # Optional per-case start hook — present only if the sink supports it.
        # Marks the row ONGOING when its case begins; never fails the case.
        case_started = getattr(result_sink, "case_started", None)
        on_case_start: Callable[[int], Any] | None = None
        if case_started is not None:

            async def _on_case_start(index: int) -> None:
                try:
                    await asyncio.to_thread(case_started, index)
                except Exception as exc:
                    logger.error(
                        "Simulation stream case start ping failed",
                        exc_info=redacted_exc_info(exc),
                        extra={"run_id": run_id, "case_index": index},
                    )

            on_case_start = _on_case_start

        return on_case_start, _on_case_complete

    def _failure_report(
        self,
        spec: SimulationSpec,
        *,
        plan: SimulationPlan | None,
        started_at: datetime,
        status: RunStatus,
        failure: SimulationFailure,
        cases: list[SimulationTestCaseResult] | None = None,
    ) -> SimulationReport:
        return SimulationReport(
            run_id=spec.run_id,
            plan_id=plan.plan_id if plan is not None else None,
            spec_hash=spec.spec_hash or spec.content_hash(),
            status=status,
            cleanup_status=CleanupStatus.COMPLETED,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
            artifacts=ArtifactManifest(run_id=spec.run_id),
            failure=failure,
            test_cases=list(cases or []),
        )

    def _write_event(
        self,
        result_sink: ResultSink | None,
        event: CanonicalEvent,
    ) -> None:
        if result_sink is None:
            return
        try:
            result_sink.write_event(event)
        except Exception as exc:
            logger.error(
                "Simulation event sink failed",
                exc_info=redacted_exc_info(exc),
                extra={
                    "run_id": event.run_id,
                    "event_id": event.event_id,
                    "exception_type": type(exc).__name__,
                },
            )

    def _write_report(
        self,
        result_sink: ResultSink | None,
        report: SimulationReport,
    ) -> None:
        if result_sink is None:
            return
        try:
            result_sink.write_report(report)
        except Exception as exc:
            logger.error(
                "Simulation report sink failed",
                exc_info=redacted_exc_info(exc),
                extra={
                    "run_id": report.run_id,
                    "exception_type": type(exc).__name__,
                },
            )
