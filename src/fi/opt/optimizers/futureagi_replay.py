from __future__ import annotations

import copy
import json
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from ..base.base_optimizer import BaseOptimizer
from ..deployment import AgentDeploymentExport, export_agent_deployment
from ..observability import (
    AgentRegressionDataset,
    AgentObservabilityWindow,
    AgentRegistryReplayPackLineageReport,
    AgentRegistryReplayPackTriageReport,
    load_futureagi_experiment_history,
    load_futureagi_regression_dataset,
)
from ..targets import AgentCandidate, CandidateEvaluation, OptimizationTarget
from ..types import EvaluationResult
from .agent_feedback import (
    AgentFeedbackOptimizationResult,
    AgentFeedbackOptimizer,
    AgentMultiInteractionOptimizationResult,
    AgentMultiInteractionOptimizer,
    DEFAULT_MULTI_INTERACTION_BACKENDS,
    _normalize_optimizer_name,
)


CandidateScorer = Callable[
    [AgentCandidate],
    CandidateEvaluation | EvaluationResult | float,
]
FUTUREAGI_REPLAY_OPTIMIZER_SCHEDULE_SCHEMA_VERSION = (
    "agent-opt.futureagi-replay-optimizer-schedule.v1"
)
_REPLAY_EVIDENCE_BLOCKERS = {
    "futureagi_readback_signature_mismatch",
    "futureagi_readback_case_count_mismatch",
    "futureagi_replay_record_count_mismatch",
    "latest_promotion_failed",
    "latest_promotion_failures",
    "missing_latest_promotion_check",
    "missing_latest_optimizer_score",
}
_REPLAY_PORTFOLIO_TRIGGERS = {
    "optimizer_score_regression",
    "latest_below_best_optimizer_score",
    "selected_patch_changed",
    "optimizer_backend_changed",
    "case_signature_changed",
}
_REPLAY_CURRICULUM_TRIGGERS = {
    "coverage_regression",
    "coverage_drift",
    "required_contract_expanded",
    "required_presets_removed",
    "required_invariant_families_removed",
}


class FutureAGIReplayOptimizerSchedule(BaseModel):
    """Optimizer replay plan derived from Future AGI registry replay-pack triage."""

    schema_version: str = FUTUREAGI_REPLAY_OPTIMIZER_SCHEDULE_SCHEMA_VERSION
    provider: str = "futureagi"
    should_optimize: bool
    selected_optimizer: str = "none"
    reason: str
    replay_dataset_id: Optional[str] = None
    replay_dataset_name: Optional[str] = None
    registry_version: Optional[str] = None
    baseline_registry_version: Optional[str] = None
    triage_decision: str
    triage_severity: str
    triage_block_rollout: bool
    triggers: list[str] = Field(default_factory=list)
    optimizer_pool: list[str] = Field(default_factory=list)
    max_backends: Optional[int] = None
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    optimizer_kwargs_by_backend: dict[str, dict[str, Any]] = Field(default_factory=dict)
    rollback_kwargs: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)

    def to_optimizer_kwargs(self) -> dict[str, Any]:
        """Arguments that can be passed to FutureAGIRegressionReplayOptimizer."""

        payload: dict[str, Any] = {
            "optimizer": self.selected_optimizer,
            "optimizer_kwargs": copy.deepcopy(self.optimizer_kwargs),
            "optimizer_kwargs_by_backend": copy.deepcopy(
                self.optimizer_kwargs_by_backend
            ),
            "rollback_kwargs": copy.deepcopy(self.rollback_kwargs),
            "optimizer_schedule": self,
        }
        if self.optimizer_pool:
            payload["optimizer_pool"] = list(self.optimizer_pool)
        if self.max_backends is not None:
            payload["max_backends"] = self.max_backends
        return payload


def schedule_futureagi_registry_replay_optimization(
    triage: AgentRegistryReplayPackTriageReport | Mapping[str, Any],
    *,
    lineage: Optional[AgentRegistryReplayPackLineageReport | Mapping[str, Any]] = None,
    schedule_on_review: bool = True,
    force: bool = False,
    optimizer_pool: Optional[Sequence[str]] = None,
    max_backends: Optional[int] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    optimizer_kwargs_by_backend: Optional[Mapping[str, Mapping[str, Any]]] = None,
    rollback_kwargs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> FutureAGIReplayOptimizerSchedule:
    """
    Convert replay-pack triage into Future AGI replay optimizer arguments.

    Coverage and required-contract drift schedules curriculum replay. Optimizer
    score, selected-patch, backend, or case-signature drift schedules the
    multi-interaction backend portfolio. Pure Future AGI readback/promotion
    evidence failures do not schedule optimizer work until the retained pack can
    be read back cleanly.
    """

    active_triage = _coerce_futureagi_replay_triage(triage)
    active_lineage = _coerce_futureagi_replay_lineage(lineage)
    latest_entry = _lineage_entry_for_triage(active_lineage, active_triage)
    trigger_candidates = list(active_triage.blocking_reasons)
    if active_triage.block_rollout or schedule_on_review or force:
        trigger_candidates.extend(active_triage.warnings)
    triggers = _unique_schedule_items(trigger_candidates)
    optimizer_triggers = [
        trigger
        for trigger in triggers
        if trigger in _REPLAY_PORTFOLIO_TRIGGERS
        or trigger in _REPLAY_CURRICULUM_TRIGGERS
    ]
    evidence_only_block = (
        active_triage.block_rollout
        and not optimizer_triggers
        and all(trigger in _REPLAY_EVIDENCE_BLOCKERS for trigger in triggers)
    )
    should_optimize = bool(force or optimizer_triggers) and not evidence_only_block
    selected_optimizer = "none"
    reason = "no replay optimization scheduled"
    schedule_optimizer_pool: list[str] = []
    schedule_optimizer_kwargs = {"target_score": 1.0}
    if should_optimize:
        if any(trigger in _REPLAY_PORTFOLIO_TRIGGERS for trigger in optimizer_triggers):
            selected_optimizer = "multi_interaction"
            schedule_optimizer_pool = _unique_schedule_items(
                optimizer_pool or DEFAULT_MULTI_INTERACTION_BACKENDS
            )
            reason = "portfolio replay for optimizer-score or selected-patch drift"
        else:
            selected_optimizer = "curriculum"
            reason = "curriculum replay for coverage or required-contract drift"
    elif evidence_only_block:
        reason = "fix Future AGI readback or promotion evidence before optimizer replay"

    schedule_optimizer_kwargs.update(dict(optimizer_kwargs or {}))
    schedule_kwargs_by_backend = {
        _normalize_optimizer_name(key): dict(value)
        for key, value in dict(optimizer_kwargs_by_backend or {}).items()
    }
    schedule_rollback_kwargs = {
        "consecutive_failures": 1,
        "min_evaluations": 1,
        **dict(rollback_kwargs or {}),
    }
    recommendations = list(active_triage.recommendations)
    if should_optimize:
        recommendations.append(
            "Run FutureAGIRegressionReplayOptimizer with this optimizer schedule "
            "against the latest retained Future AGI replay pack."
        )
    return FutureAGIReplayOptimizerSchedule(
        should_optimize=should_optimize,
        selected_optimizer=selected_optimizer,
        reason=reason,
        replay_dataset_id=active_triage.latest_dataset_id,
        replay_dataset_name=latest_entry.dataset_name if latest_entry else None,
        registry_version=active_triage.latest_registry_version,
        baseline_registry_version=active_triage.baseline_registry_version,
        triage_decision=active_triage.decision,
        triage_severity=active_triage.severity,
        triage_block_rollout=active_triage.block_rollout,
        triggers=triggers,
        optimizer_pool=schedule_optimizer_pool,
        max_backends=max_backends,
        optimizer_kwargs=schedule_optimizer_kwargs,
        optimizer_kwargs_by_backend=schedule_kwargs_by_backend,
        rollback_kwargs=schedule_rollback_kwargs,
        recommendations=_unique_schedule_items(recommendations),
        evidence={
            "triage": active_triage.to_manifest(),
            "lineage": active_lineage.to_manifest() if active_lineage else None,
        },
        metadata={
            "kind": "futureagi_registry_replay_optimizer_schedule",
            **dict(metadata or {}),
        },
    )


class FutureAGIRegressionReplayOptimizer(BaseOptimizer):
    """
    Re-optimize an agent from a Future AGI regression dataset.

    The optimizer accepts either an already-loaded `AgentRegressionDataset` or a
    Future AGI `dataset_id`, turns the regression cases into an observability
    replay window, and delegates the repair search to `AgentFeedbackOptimizer`.
    Use `optimizer="society"`, `"social_memory"`, `"curriculum"`,
    `"council"`, `"evolution"`, `"bandit"`, `"tpe"`, `"pareto"`, or
    `"agent"` to select the backend.
    """

    def __init__(
        self,
        target: Optional[OptimizationTarget] = None,
        *,
        dataset: Optional[AgentRegressionDataset] = None,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        fi_api_key: Optional[str] = None,
        fi_secret_key: Optional[str] = None,
        fi_base_url: Optional[str] = None,
        client: Any = None,
        page_size: int = 100,
        max_pages: int = 100,
        timeout: float = 30.0,
        candidate: Optional[AgentCandidate] = None,
        deployment: Optional[AgentDeploymentExport] = None,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        optimizer: str = "society",
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        optimizer_pool: Optional[Sequence[str]] = None,
        max_backends: Optional[int] = None,
        optimizer_kwargs_by_backend: Optional[Mapping[str, Mapping[str, Any]]] = None,
        optimizer_schedule: Optional[
            FutureAGIReplayOptimizerSchedule | Mapping[str, Any]
        ] = None,
        rollback_kwargs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.target = target
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.fi_api_key = fi_api_key
        self.fi_secret_key = fi_secret_key
        self.fi_base_url = fi_base_url
        self.client = client
        self.page_size = page_size
        self.max_pages = max_pages
        self.timeout = timeout
        self.candidate = candidate
        self.deployment = deployment
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.optimizer = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.optimizer_pool = list(optimizer_pool) if optimizer_pool is not None else None
        self.max_backends = max_backends
        self.optimizer_kwargs_by_backend = {
            str(key): dict(value)
            for key, value in dict(optimizer_kwargs_by_backend or {}).items()
        }
        self.optimizer_schedule = _coerce_futureagi_replay_optimizer_schedule(
            optimizer_schedule
        )
        self.rollback_kwargs = dict(rollback_kwargs or {})
        self.metadata = dict(metadata or {})
        super().__init__()

    def optimize(
        self,
        evaluator: Any = None,
        data_mapper: Any = None,
        dataset_records: Optional[list[dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        *,
        target: Optional[OptimizationTarget] = None,
        dataset: Optional[AgentRegressionDataset] = None,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        fi_api_key: Optional[str] = None,
        fi_secret_key: Optional[str] = None,
        fi_base_url: Optional[str] = None,
        client: Any = None,
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
        timeout: Optional[float] = None,
        candidate: Optional[AgentCandidate] = None,
        deployment: Optional[AgentDeploymentExport] = None,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        optimizer: Optional[str] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        optimizer_pool: Optional[Sequence[str]] = None,
        max_backends: Optional[int] = None,
        optimizer_kwargs_by_backend: Optional[Mapping[str, Mapping[str, Any]]] = None,
        optimizer_schedule: Optional[
            FutureAGIReplayOptimizerSchedule | Mapping[str, Any]
        ] = None,
        rollback_kwargs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **backend_kwargs: Any,
    ) -> AgentFeedbackOptimizationResult | AgentMultiInteractionOptimizationResult:
        active_target = target or self.target
        if active_target is None:
            raise ValueError("FutureAGIRegressionReplayOptimizer requires a target.")

        active_dataset = dataset or self.dataset
        active_dataset_id = dataset_id or self.dataset_id
        if active_dataset is None:
            if not active_dataset_id:
                raise ValueError(
                    "FutureAGIRegressionReplayOptimizer requires dataset or dataset_id."
                )
            active_dataset = load_futureagi_regression_dataset(
                dataset_id=active_dataset_id,
                dataset_name=dataset_name or self.dataset_name,
                fi_api_key=fi_api_key or self.fi_api_key,
                fi_secret_key=fi_secret_key or self.fi_secret_key,
                fi_base_url=fi_base_url or self.fi_base_url,
                client=client or self.client,
                page_size=page_size if page_size is not None else self.page_size,
                max_pages=max_pages if max_pages is not None else self.max_pages,
                timeout=timeout if timeout is not None else self.timeout,
                metadata={
                    "optimizer": "FutureAGIRegressionReplayOptimizer",
                    **self.metadata,
                    **dict(metadata or {}),
                },
            )

        active_candidate = candidate or self.candidate or active_target.seed_candidate()
        active_deployment = (
            deployment
            or self.deployment
            or export_agent_deployment(
                active_candidate,
                metadata={
                    "futureagi_regression_dataset_name": active_dataset.name,
                    "futureagi_regression_dataset_id": active_dataset_id,
                },
            )
        )
        replay_window = active_dataset.to_observability_window(
            candidate=active_candidate,
            metadata={
                "futureagi_regression_replay_optimizer": True,
                "futureagi_regression_dataset_id": active_dataset_id,
            },
        )
        live_evaluations = replay_window.to_live_evaluations()
        active_rollback_kwargs = {
            "required_metrics": replay_window.required_metrics,
            "consecutive_failures": 1,
            "min_evaluations": 1,
            **self.rollback_kwargs,
            **dict(rollback_kwargs or {}),
        }
        active_schedule = (
            _coerce_futureagi_replay_optimizer_schedule(optimizer_schedule)
            if optimizer_schedule is not None
            else self.optimizer_schedule
        )
        if (
            active_schedule is not None
            and not active_schedule.should_optimize
            and optimizer is None
        ):
            raise ValueError(
                "Future AGI replay optimizer schedule does not require "
                "re-optimization."
            )
        schedule_optimizer_kwargs = (
            copy.deepcopy(active_schedule.optimizer_kwargs)
            if active_schedule is not None
            else {}
        )
        active_optimizer_kwargs = {
            **schedule_optimizer_kwargs,
            **self.optimizer_kwargs,
            **dict(optimizer_kwargs or {}),
            **backend_kwargs,
        }
        active_optimizer_pool = (
            list(optimizer_pool)
            if optimizer_pool is not None
            else (
                self.optimizer_pool
                if self.optimizer_pool is not None
                else (
                    list(active_schedule.optimizer_pool)
                    if active_schedule is not None and active_schedule.optimizer_pool
                    else None
                )
            )
        )
        active_max_backends = (
            max_backends
            if max_backends is not None
            else (
                self.max_backends
                if self.max_backends is not None
                else (active_schedule.max_backends if active_schedule is not None else None)
            )
        )
        schedule_kwargs_by_backend = (
            copy.deepcopy(active_schedule.optimizer_kwargs_by_backend)
            if active_schedule is not None
            else {}
        )
        active_optimizer_kwargs_by_backend = {
            **schedule_kwargs_by_backend,
            **self.optimizer_kwargs_by_backend,
            **{
                str(key): dict(value)
                for key, value in dict(optimizer_kwargs_by_backend or {}).items()
            },
        }
        schedule_backend = (
            active_schedule.selected_optimizer
            if active_schedule is not None and active_schedule.selected_optimizer != "none"
            else None
        )
        resolved_backend = optimizer or schedule_backend or self.optimizer
        normalized_backend = _normalize_optimizer_name(resolved_backend)
        if normalized_backend == "social_memory":
            active_optimizer_kwargs.setdefault("experiment_history", replay_window)
        if normalized_backend == "curriculum":
            active_optimizer_kwargs.setdefault("curriculum_history", replay_window)
        result_metadata = {
            **self.metadata,
            **dict(metadata or {}),
            "futureagi_regression_dataset_id": active_dataset_id,
            "futureagi_regression_dataset_name": active_dataset.name,
            "futureagi_regression_case_count": len(active_dataset.cases),
            "futureagi_replay_record_count": len(replay_window.records),
            "futureagi_replay_required_metrics": dict(replay_window.required_metrics),
            "futureagi_replay_required_trace_signals": list(
                replay_window.required_trace_signals
            ),
        }
        if active_schedule is not None:
            result_metadata.update(
                {
                    "futureagi_replay_optimizer_schedule": active_schedule.to_manifest(),
                    "futureagi_replay_scheduled_optimizer": active_schedule.selected_optimizer,
                    "futureagi_replay_schedule_triggers": list(active_schedule.triggers),
                }
            )
        if normalized_backend == "multi_interaction":
            active_optimizer_kwargs_by_backend.setdefault(
                "social_memory",
                {},
            ).setdefault("experiment_history", replay_window)
            active_optimizer_kwargs_by_backend.setdefault(
                "curriculum",
                {},
            ).setdefault("curriculum_history", replay_window)
            return AgentMultiInteractionOptimizer(
                target=active_target,
                deployment=active_deployment,
                live_evaluations=live_evaluations,
                evaluate_candidate=evaluate_candidate or self.evaluate_candidate or evaluator,
                simulation_evaluator=simulation_evaluator or self.simulation_evaluator,
                optimizer_pool=active_optimizer_pool,
                max_backends=active_max_backends,
                optimizer_kwargs=active_optimizer_kwargs,
                optimizer_kwargs_by_backend=active_optimizer_kwargs_by_backend,
                rollback_kwargs=active_rollback_kwargs,
                metadata=result_metadata,
            ).optimize()
        return AgentFeedbackOptimizer(
            target=active_target,
            deployment=active_deployment,
            live_evaluations=live_evaluations,
            evaluate_candidate=evaluate_candidate or self.evaluate_candidate or evaluator,
            simulation_evaluator=simulation_evaluator or self.simulation_evaluator,
            optimizer=resolved_backend,
            optimizer_kwargs=active_optimizer_kwargs,
            rollback_kwargs=active_rollback_kwargs,
            metadata=result_metadata,
        ).optimize()


def _coerce_futureagi_replay_optimizer_schedule(
    value: Optional[FutureAGIReplayOptimizerSchedule | Mapping[str, Any]],
) -> Optional[FutureAGIReplayOptimizerSchedule]:
    if value is None:
        return None
    if isinstance(value, FutureAGIReplayOptimizerSchedule):
        return value
    if isinstance(value, Mapping):
        return FutureAGIReplayOptimizerSchedule(**dict(value))
    raise TypeError(
        "optimizer_schedule must be FutureAGIReplayOptimizerSchedule or mapping."
    )


def _coerce_futureagi_replay_triage(
    value: AgentRegistryReplayPackTriageReport | Mapping[str, Any],
) -> AgentRegistryReplayPackTriageReport:
    if isinstance(value, AgentRegistryReplayPackTriageReport):
        return value
    if isinstance(value, Mapping):
        return AgentRegistryReplayPackTriageReport(**dict(value))
    raise TypeError("triage must be AgentRegistryReplayPackTriageReport or mapping.")


def _coerce_futureagi_replay_lineage(
    value: Optional[AgentRegistryReplayPackLineageReport | Mapping[str, Any]],
) -> Optional[AgentRegistryReplayPackLineageReport]:
    if value is None:
        return None
    if isinstance(value, AgentRegistryReplayPackLineageReport):
        return value
    if isinstance(value, Mapping):
        return AgentRegistryReplayPackLineageReport(**dict(value))
    raise TypeError("lineage must be AgentRegistryReplayPackLineageReport or mapping.")


def _lineage_entry_for_triage(
    lineage: Optional[AgentRegistryReplayPackLineageReport],
    triage: AgentRegistryReplayPackTriageReport,
) -> Any:
    if lineage is None:
        return None
    for entry in reversed(lineage.entries):
        if triage.latest_dataset_id and entry.dataset_id == triage.latest_dataset_id:
            return entry
        if (
            triage.latest_registry_version
            and entry.registry_version == triage.latest_registry_version
        ):
            return entry
    return lineage.entries[-1] if lineage.entries else None


def _unique_schedule_items(values: Sequence[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


class FutureAGIExperimentHistoryOptimizer(BaseOptimizer):
    """
    Re-optimize an agent from native Future AGI experiment history.

    This backend reads Future AGI experiment detail/stats/row payloads, converts
    variant and row-level scores into observability feedback, and delegates the
    repair search to `AgentFeedbackOptimizer`. Use this when the production
    signal lives in Future AGI experiments rather than regression-pack datasets.
    """

    def __init__(
        self,
        target: Optional[OptimizationTarget] = None,
        *,
        experiment_history: Optional[AgentObservabilityWindow] = None,
        experiment_id: Optional[str] = None,
        fi_api_key: Optional[str] = None,
        fi_secret_key: Optional[str] = None,
        fi_base_url: Optional[str] = None,
        client: Any = None,
        page_size: int = 100,
        max_pages: int = 20,
        timeout: float = 30.0,
        include_rows: bool = True,
        include_stats: bool = True,
        prefer_v2: bool = True,
        required_metrics: Optional[Mapping[str, float]] = None,
        required_trace_signals: Optional[list[str]] = None,
        candidate: Optional[AgentCandidate] = None,
        deployment: Optional[AgentDeploymentExport] = None,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        optimizer: str = "society",
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        optimizer_pool: Optional[Sequence[str]] = None,
        max_backends: Optional[int] = None,
        optimizer_kwargs_by_backend: Optional[Mapping[str, Mapping[str, Any]]] = None,
        rollback_kwargs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.target = target
        self.experiment_history = experiment_history
        self.experiment_id = experiment_id
        self.fi_api_key = fi_api_key
        self.fi_secret_key = fi_secret_key
        self.fi_base_url = fi_base_url
        self.client = client
        self.page_size = page_size
        self.max_pages = max_pages
        self.timeout = timeout
        self.include_rows = include_rows
        self.include_stats = include_stats
        self.prefer_v2 = prefer_v2
        self.required_metrics = dict(required_metrics or {})
        self.required_trace_signals = list(required_trace_signals or [])
        self.candidate = candidate
        self.deployment = deployment
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.optimizer = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.optimizer_pool = list(optimizer_pool) if optimizer_pool is not None else None
        self.max_backends = max_backends
        self.optimizer_kwargs_by_backend = {
            str(key): dict(value)
            for key, value in dict(optimizer_kwargs_by_backend or {}).items()
        }
        self.rollback_kwargs = dict(rollback_kwargs or {})
        self.metadata = dict(metadata or {})
        super().__init__()

    def optimize(
        self,
        evaluator: Any = None,
        data_mapper: Any = None,
        dataset_records: Optional[list[dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        *,
        target: Optional[OptimizationTarget] = None,
        experiment_history: Optional[AgentObservabilityWindow] = None,
        experiment_id: Optional[str] = None,
        fi_api_key: Optional[str] = None,
        fi_secret_key: Optional[str] = None,
        fi_base_url: Optional[str] = None,
        client: Any = None,
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
        timeout: Optional[float] = None,
        include_rows: Optional[bool] = None,
        include_stats: Optional[bool] = None,
        prefer_v2: Optional[bool] = None,
        required_metrics: Optional[Mapping[str, float]] = None,
        required_trace_signals: Optional[list[str]] = None,
        candidate: Optional[AgentCandidate] = None,
        deployment: Optional[AgentDeploymentExport] = None,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        optimizer: Optional[str] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        optimizer_pool: Optional[Sequence[str]] = None,
        max_backends: Optional[int] = None,
        optimizer_kwargs_by_backend: Optional[Mapping[str, Mapping[str, Any]]] = None,
        rollback_kwargs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **backend_kwargs: Any,
    ) -> AgentFeedbackOptimizationResult | AgentMultiInteractionOptimizationResult:
        active_target = target or self.target
        if active_target is None:
            raise ValueError("FutureAGIExperimentHistoryOptimizer requires a target.")

        active_candidate = candidate or self.candidate or active_target.seed_candidate()
        active_history = experiment_history or self.experiment_history
        active_experiment_id = experiment_id or self.experiment_id
        active_required_metrics = {
            **self.required_metrics,
            **dict(required_metrics or {}),
        }
        active_required_signals = list(
            required_trace_signals
            if required_trace_signals is not None
            else self.required_trace_signals
        )
        if active_history is None:
            if not active_experiment_id:
                raise ValueError(
                    "FutureAGIExperimentHistoryOptimizer requires "
                    "experiment_history or experiment_id."
                )
            active_history = load_futureagi_experiment_history(
                experiment_id=active_experiment_id,
                candidate=active_candidate,
                required_metrics=active_required_metrics,
                required_trace_signals=active_required_signals,
                fi_api_key=fi_api_key or self.fi_api_key,
                fi_secret_key=fi_secret_key or self.fi_secret_key,
                fi_base_url=fi_base_url or self.fi_base_url,
                client=client or self.client,
                page_size=page_size if page_size is not None else self.page_size,
                max_pages=max_pages if max_pages is not None else self.max_pages,
                timeout=timeout if timeout is not None else self.timeout,
                include_rows=include_rows if include_rows is not None else self.include_rows,
                include_stats=include_stats if include_stats is not None else self.include_stats,
                prefer_v2=prefer_v2 if prefer_v2 is not None else self.prefer_v2,
                metadata={
                    "optimizer": "FutureAGIExperimentHistoryOptimizer",
                    **self.metadata,
                    **dict(metadata or {}),
                },
            )

        resolved_experiment_id = active_experiment_id or str(
            active_history.metadata.get("experiment_id") or ""
        )
        active_deployment = (
            deployment
            or self.deployment
            or export_agent_deployment(
                active_candidate,
                metadata={
                    "futureagi_experiment_id": resolved_experiment_id,
                    "futureagi_experiment_name": active_history.metadata.get(
                        "experiment_name"
                    ),
                },
            )
        )
        live_evaluations = active_history.to_live_evaluations(candidate=active_candidate)
        active_rollback_kwargs = {
            "required_metrics": active_history.required_metrics,
            "consecutive_failures": 1,
            "min_evaluations": 1,
            **self.rollback_kwargs,
            **dict(rollback_kwargs or {}),
        }
        active_optimizer_kwargs = {
            **self.optimizer_kwargs,
            **dict(optimizer_kwargs or {}),
            **backend_kwargs,
        }
        active_optimizer_pool = (
            list(optimizer_pool) if optimizer_pool is not None else self.optimizer_pool
        )
        active_max_backends = self.max_backends if max_backends is None else max_backends
        active_optimizer_kwargs_by_backend = {
            **self.optimizer_kwargs_by_backend,
            **{
                str(key): dict(value)
                for key, value in dict(optimizer_kwargs_by_backend or {}).items()
            },
        }
        resolved_backend = optimizer or self.optimizer
        normalized_backend = _normalize_optimizer_name(resolved_backend)
        if normalized_backend == "social_memory":
            active_optimizer_kwargs.setdefault("experiment_history", active_history)
        if normalized_backend == "curriculum":
            active_optimizer_kwargs.setdefault("curriculum_history", active_history)
        result_metadata = {
            **self.metadata,
            **dict(metadata or {}),
            "futureagi_experiment_id": resolved_experiment_id,
            "futureagi_experiment_name": active_history.metadata.get("experiment_name"),
            "futureagi_experiment_record_count": len(active_history.records),
            "futureagi_experiment_required_metrics": dict(active_history.required_metrics),
            "futureagi_experiment_required_trace_signals": list(
                active_history.required_trace_signals
            ),
        }
        if normalized_backend == "multi_interaction":
            active_optimizer_kwargs_by_backend.setdefault(
                "social_memory",
                {},
            ).setdefault("experiment_history", active_history)
            active_optimizer_kwargs_by_backend.setdefault(
                "curriculum",
                {},
            ).setdefault("curriculum_history", active_history)
            return AgentMultiInteractionOptimizer(
                target=active_target,
                deployment=active_deployment,
                live_evaluations=live_evaluations,
                evaluate_candidate=evaluate_candidate or self.evaluate_candidate or evaluator,
                simulation_evaluator=simulation_evaluator or self.simulation_evaluator,
                optimizer_pool=active_optimizer_pool,
                max_backends=active_max_backends,
                optimizer_kwargs=active_optimizer_kwargs,
                optimizer_kwargs_by_backend=active_optimizer_kwargs_by_backend,
                rollback_kwargs=active_rollback_kwargs,
                metadata=result_metadata,
            ).optimize()
        return AgentFeedbackOptimizer(
            target=active_target,
            deployment=active_deployment,
            live_evaluations=live_evaluations,
            evaluate_candidate=evaluate_candidate or self.evaluate_candidate or evaluator,
            simulation_evaluator=simulation_evaluator or self.simulation_evaluator,
            optimizer=resolved_backend,
            optimizer_kwargs=active_optimizer_kwargs,
            rollback_kwargs=active_rollback_kwargs,
            metadata=result_metadata,
        ).optimize()
