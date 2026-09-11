from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

from ..base.base_optimizer import BaseOptimizer
from ..components import (
    ComponentDiagnosis,
    diagnose_agent_report_evaluation,
    relevant_search_paths,
)
from ..observability import AgentObservabilityWindow
from ..targets import AgentCandidate, CandidateEvaluation, OptimizationTarget
from ..types import EvaluationResult, IterationHistory, OptimizationResult
from .agent import (
    _dedupe_diagnoses,
    _diagnose_candidate_evaluation,
    _dump_model,
    _history_from_candidate,
    _normalize_candidate_evaluation,
    _normalize_diagnoses,
)

logger = logging.getLogger(__name__)


CandidateScorer = Callable[
    [AgentCandidate],
    CandidateEvaluation | EvaluationResult | float,
]


@dataclass
class AgentCurriculumStage:
    """
    One metric-focused optimization stage.

    Stages let a multi-interaction workflow practice one failure family at a
    time, then promote the best candidate into the next stage. Names and
    inspiration labels are metadata only; numeric evaluator evidence controls
    promotion.
    """

    name: str
    search_paths: Sequence[str] = field(default_factory=tuple)
    diagnoses: Sequence[ComponentDiagnosis | Mapping[str, Any]] = field(default_factory=tuple)
    metric_weights: Mapping[str, float] = field(default_factory=dict)
    target_score: Optional[float] = None
    max_candidates: Optional[int] = None
    evaluator: Optional[CandidateScorer] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AgentCurriculumOptimizer(BaseOptimizer):
    """
    Deterministic staged optimizer for multi-interaction agents.

    `AgentOptimizer` is a flat candidate search. `AgentCurriculumOptimizer`
    instead runs a sequence of metric-focused drills: memory first, handoff
    next, policy next, etc. Each stage searches only its relevant config paths,
    scores candidates with the stage's metric weights, promotes the best
    candidate, and carries that config into the next stage.
    """

    def __init__(
        self,
        target: Optional[OptimizationTarget] = None,
        *,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        stages: Optional[Iterable[AgentCurriculumStage | Mapping[str, Any]]] = None,
        curriculum_history: Optional[AgentObservabilityWindow] = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        max_candidates_per_stage: int = 32,
        include_seed: bool = True,
        auto_diagnose: bool = True,
        diagnostic_score_threshold: float = 0.85,
        target_score: float = 1.0,
        carry_forward: bool = True,
    ) -> None:
        if max_candidates_per_stage < 1:
            raise ValueError("max_candidates_per_stage must be at least 1.")
        if target_score < 0:
            raise ValueError("target_score must be non-negative.")

        self.target = target
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.stages = _normalize_curriculum_stages(stages)
        self.curriculum_history = curriculum_history
        self.diagnoses = _normalize_diagnoses(diagnoses)
        self.max_candidates_per_stage = max_candidates_per_stage
        self.include_seed = include_seed
        self.auto_diagnose = auto_diagnose
        self.diagnostic_score_threshold = diagnostic_score_threshold
        self.target_score = target_score
        self.carry_forward = carry_forward
        super().__init__()

    def optimize(
        self,
        evaluator: Any = None,
        data_mapper: Any = None,
        dataset: Optional[List[dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        *,
        target: Optional[OptimizationTarget] = None,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        stages: Optional[Iterable[AgentCurriculumStage | Mapping[str, Any]]] = None,
        curriculum_history: Optional[AgentObservabilityWindow] = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        max_candidates_per_stage: Optional[int] = None,
        include_seed: Optional[bool] = None,
        auto_diagnose: Optional[bool] = None,
        diagnostic_score_threshold: Optional[float] = None,
        target_score: Optional[float] = None,
        carry_forward: Optional[bool] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        active_target = target or self.target
        if active_target is None:
            raise ValueError("AgentCurriculumOptimizer requires a target.")
        if not active_target.search_space:
            raise ValueError(
                "AgentCurriculumOptimizer target search space cannot be empty."
            )

        active_evaluator = (
            evaluate_candidate
            or self.evaluate_candidate
            or getattr(simulation_evaluator, "evaluate_candidate", None)
            or getattr(self.simulation_evaluator, "evaluate_candidate", None)
            or evaluator
        )
        active_stages = (
            _normalize_curriculum_stages(stages)
            if stages is not None
            else list(self.stages)
        )
        active_history = curriculum_history or self.curriculum_history
        active_max_candidates = (
            self.max_candidates_per_stage
            if max_candidates_per_stage is None
            else max_candidates_per_stage
        )
        if active_max_candidates < 1:
            raise ValueError("max_candidates_per_stage must be at least 1.")
        use_include_seed = self.include_seed if include_seed is None else include_seed
        use_auto_diagnose = self.auto_diagnose if auto_diagnose is None else auto_diagnose
        active_diagnostic_threshold = (
            self.diagnostic_score_threshold
            if diagnostic_score_threshold is None
            else diagnostic_score_threshold
        )
        active_target_score = self.target_score if target_score is None else target_score
        use_carry_forward = self.carry_forward if carry_forward is None else carry_forward
        active_diagnoses = _normalize_diagnoses(diagnoses)
        if diagnoses is None:
            active_diagnoses = list(self.diagnoses)

        if not active_stages:
            active_stages = _stages_from_history(
                active_history,
                target=active_target,
                failing_threshold=active_diagnostic_threshold,
            )
        if not active_stages:
            active_stages = [
                AgentCurriculumStage(
                    name="diagnosed",
                    metadata={"source": "default_single_stage"},
                )
            ]

        if active_evaluator is None and not all(stage.evaluator for stage in active_stages):
            raise ValueError(
                "AgentCurriculumOptimizer requires evaluate_candidate, "
                "simulation_evaluator, or evaluators on every stage."
            )

        anchor = active_target.seed_candidate()
        best: CandidateEvaluation | None = None
        history: List[IterationHistory] = []
        stage_summaries: list[dict[str, Any]] = []
        stage_audit: list[dict[str, Any]] = []

        if use_include_seed and active_evaluator is not None:
            seed_stage = AgentCurriculumStage(
                name="seed",
                metadata={"source": "deployed_seed"},
            )
            seed_evaluation = self._evaluate(
                anchor,
                active_evaluator,
                seed_stage,
                stage_index=0,
                history=history,
                reason="evaluate_deployed_seed",
            )
            best = seed_evaluation
            if use_auto_diagnose and not active_diagnoses:
                active_diagnoses = _diagnose_candidate_evaluation(
                    seed_evaluation,
                    failing_threshold=active_diagnostic_threshold,
                )

        for stage_index, stage in enumerate(active_stages, start=1):
            stage_evaluator = stage.evaluator or active_evaluator
            if stage_evaluator is None:
                raise ValueError(f"Curriculum stage '{stage.name}' has no evaluator.")

            stage_paths = _stage_search_paths(
                active_target,
                stage,
                active_diagnoses,
            )
            if not stage_paths:
                stage_paths = list(active_target.search_space)

            stage_best: CandidateEvaluation | None = None
            stage_best_score = float("-inf")
            evaluated_in_stage = 0
            stage_target = (
                stage.target_score
                if stage.target_score is not None
                else active_target_score
            )

            for candidate in _iter_stage_candidates(
                anchor,
                target=active_target,
                stage=stage,
                stage_index=stage_index,
                search_paths=stage_paths,
                include_seed=True,
                max_candidates=stage.max_candidates or active_max_candidates,
            ):
                evaluation = self._evaluate(
                    candidate,
                    stage_evaluator,
                    stage,
                    stage_index=stage_index,
                    history=history,
                    reason=f"curriculum_stage:{stage.name}",
                )
                evaluated_in_stage += 1
                candidate_stage_score = _stage_score(evaluation, stage)
                stage_audit.append(
                    {
                        "stage": stage.name,
                        "stage_index": stage_index,
                        "candidate_id": candidate.id,
                        "patch": candidate.patch,
                        "score": evaluation.score,
                        "stage_score": candidate_stage_score,
                    }
                )
                if _is_better_stage_candidate(
                    evaluation,
                    candidate_stage_score,
                    stage_best,
                    stage_best_score,
                ):
                    stage_best = evaluation
                    stage_best_score = candidate_stage_score
                if best is None or evaluation.score > best.score:
                    best = evaluation
                    logger.info(
                        "New best curriculum candidate %s score=%.4f",
                        candidate.id,
                        evaluation.score,
                    )
                if candidate_stage_score >= stage_target:
                    break

            if stage_best is not None and use_carry_forward:
                anchor = stage_best.candidate
            if stage_best is not None and use_auto_diagnose:
                stage_diagnoses = _diagnose_candidate_evaluation(
                    stage_best,
                    failing_threshold=active_diagnostic_threshold,
                )
                if stage_diagnoses:
                    active_diagnoses = _dedupe_diagnoses(
                        [*active_diagnoses, *stage_diagnoses]
                    )

            stage_summaries.append(
                {
                    "stage": stage.name,
                    "stage_index": stage_index,
                    "search_paths": list(stage_paths),
                    "metric_weights": dict(stage.metric_weights),
                    "target_score": stage_target,
                    "evaluated": evaluated_in_stage,
                    "best_stage_score": None
                    if stage_best is None
                    else round(stage_best_score, 4),
                    "best_candidate_id": None if stage_best is None else stage_best.candidate.id,
                    "promoted": bool(stage_best is not None and use_carry_forward),
                    "target_met": bool(stage_best is not None and stage_best_score >= stage_target),
                    "metadata": dict(stage.metadata),
                }
            )
            if best is not None and best.score >= active_target_score:
                break

        if best is None:
            raise ValueError("AgentCurriculumOptimizer did not evaluate any candidates.")

        metadata = {
            "optimizer": "AgentCurriculumOptimizer",
            "strategy": "deliberate_practice_curriculum",
            "strategy_inspiration": (
                "curriculum learning, deliberate practice, metacognitive "
                "stage gates, and dharma-style stewardship; names are metadata only"
            ),
            "roles": ["viveka", "abhyasa", "satsanga", "dharma_steward"],
            "target_name": best.candidate.target_name,
            "best_candidate_id": best.candidate.id,
            "stages": stage_summaries,
            "stage_audit": stage_audit,
            "search_paths": _unique_paths(
                path
                for summary in stage_summaries
                for path in summary["search_paths"]
            ),
            "max_candidates_per_stage": active_max_candidates,
            "carry_forward": use_carry_forward,
            "history_source": active_history.source if active_history else None,
            "history_record_count": len(active_history.records) if active_history else 0,
        }
        if active_diagnoses:
            metadata["diagnostics"] = [_dump_model(item) for item in active_diagnoses]
            metadata["auto_diagnosed"] = use_auto_diagnose

        return OptimizationResult(
            best_generator=best.candidate,
            best_candidate=best.candidate,
            history=history,
            final_score=best.score,
            total_iterations=len(history),
            total_evaluations=len(history),
            metadata=metadata,
        )

    def _evaluate(
        self,
        candidate: AgentCandidate,
        evaluator: CandidateScorer,
        stage: AgentCurriculumStage,
        *,
        stage_index: int,
        history: List[IterationHistory],
        reason: str,
    ) -> CandidateEvaluation:
        value = evaluator(candidate)
        evaluation = _normalize_candidate_evaluation(value, candidate)
        stage_score = _stage_score(evaluation, stage)
        evaluation.metadata = {
            **candidate.metadata,
            **evaluation.metadata,
            "optimizer": "AgentCurriculumOptimizer",
            "curriculum_stage": stage.name,
            "curriculum_stage_index": stage_index,
            "curriculum_stage_score": stage_score,
            "curriculum_stage_metric_weights": dict(stage.metric_weights),
            "curriculum_stage_reason": reason,
            "curriculum_stage_metadata": dict(stage.metadata),
        }
        history.append(_history_from_candidate(evaluation))
        return evaluation


def _normalize_curriculum_stages(
    stages: Optional[Iterable[AgentCurriculumStage | Mapping[str, Any]]],
) -> list[AgentCurriculumStage]:
    if stages is None:
        return []
    normalized: list[AgentCurriculumStage] = []
    for index, raw in enumerate(stages, start=1):
        if isinstance(raw, AgentCurriculumStage):
            normalized.append(raw)
            continue
        item = dict(raw)
        metrics = item.get("metric_weights") or {}
        if not metrics:
            metric_names = _string_list(item.get("metrics") or item.get("metric"))
            metrics = {name: 1.0 for name in metric_names}
        normalized.append(
            AgentCurriculumStage(
                name=str(item.get("name") or f"stage_{index}"),
                search_paths=tuple(_string_list(item.get("search_paths") or item.get("paths"))),
                diagnoses=tuple(_normalize_diagnoses(item.get("diagnoses") or [])),
                metric_weights={
                    str(key): float(value)
                    for key, value in dict(metrics).items()
                    if float(value) > 0
                },
                target_score=_optional_float(item.get("target_score")),
                max_candidates=_optional_int(item.get("max_candidates")),
                evaluator=item.get("evaluator"),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return normalized


def _stages_from_history(
    history: Optional[AgentObservabilityWindow],
    *,
    target: OptimizationTarget,
    failing_threshold: float,
) -> list[AgentCurriculumStage]:
    if history is None:
        return []
    thresholds = dict(history.required_metrics)
    if not thresholds:
        for record in history.records:
            for metric_name in record.metrics:
                thresholds.setdefault(metric_name, failing_threshold)
    stages: list[AgentCurriculumStage] = []
    for metric_name, threshold in thresholds.items():
        observed = [
            float(record.metrics.get(metric_name, 0.0))
            for record in history.records
        ]
        if observed and min(observed) >= threshold:
            continue
        diagnoses = diagnose_agent_report_evaluation(
            {
                "summary": {"metric_averages": {metric_name: min(observed or [0.0])}},
                "cases": [
                    {
                        "metrics": [
                            {
                                "name": metric_name,
                                "score": min(observed or [0.0]),
                                "reason": f"Curriculum history failed {metric_name}.",
                            }
                        ]
                    }
                ],
            },
            failing_threshold=threshold,
            confidence=0.9,
        )
        paths = relevant_search_paths(target.search_space, diagnoses)
        stages.append(
            AgentCurriculumStage(
                name=metric_name,
                search_paths=tuple(path for path in target.search_space if path in paths),
                diagnoses=tuple(diagnoses),
                metric_weights={metric_name: 1.0},
                target_score=float(threshold),
                metadata={
                    "source": "curriculum_history",
                    "history_source": history.source,
                    "observed_min": min(observed or [0.0]),
                },
            )
        )
    return stages


def _stage_search_paths(
    target: OptimizationTarget,
    stage: AgentCurriculumStage,
    active_diagnoses: Sequence[ComponentDiagnosis],
) -> list[str]:
    explicit = [path for path in stage.search_paths if path in target.search_space]
    if explicit:
        return [path for path in target.search_space if path in set(explicit)]
    stage_diagnoses = _normalize_diagnoses(stage.diagnoses)
    diagnoses = stage_diagnoses or list(active_diagnoses)
    if diagnoses:
        allowed = relevant_search_paths(target.search_space, diagnoses)
        return [path for path in target.search_space if path in allowed]
    return list(target.search_space)


def _iter_stage_candidates(
    anchor: AgentCandidate,
    *,
    target: OptimizationTarget,
    stage: AgentCurriculumStage,
    stage_index: int,
    search_paths: Sequence[str],
    include_seed: bool,
    max_candidates: int,
) -> Iterable[AgentCandidate]:
    count = 0
    if include_seed:
        yield AgentCandidate.from_config(
            anchor.config,
            target_name=target.name,
            layers=target.layers,
            parent_id=anchor.parent_id,
            patch=anchor.patch,
            metadata={
                **anchor.metadata,
                "kind": "curriculum_anchor",
                "curriculum_stage": stage.name,
                "curriculum_stage_index": stage_index,
            },
        )
        count += 1
        if count >= max_candidates:
            return

    paths = [path for path in search_paths if path in target.search_space]
    value_lists = [
        _values_with_anchor_first(target.search_space[path], anchor.get_path(path))
        for path in paths
    ]
    for values in itertools.product(*value_lists):
        patch = dict(zip(paths, values))
        if all(anchor.get_path(path) == value for path, value in patch.items()):
            continue
        yield anchor.with_patch(
            patch,
            layers=target.layers,
            metadata={
                "kind": "curriculum_stage_candidate",
                "optimizer": "AgentCurriculumOptimizer",
                "curriculum_stage": stage.name,
                "curriculum_stage_index": stage_index,
                "curriculum_search_paths": list(paths),
                "curriculum_stage_metadata": dict(stage.metadata),
            },
        )
        count += 1
        if count >= max_candidates:
            return


def _stage_score(
    evaluation: CandidateEvaluation,
    stage: AgentCurriculumStage,
) -> float:
    weights = {key: float(value) for key, value in stage.metric_weights.items() if value > 0}
    if not weights:
        return float(evaluation.score)
    metrics = _metric_averages(evaluation)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return float(evaluation.score)
    return sum(float(metrics.get(name, 0.0)) * weight for name, weight in weights.items()) / total_weight


def _metric_averages(evaluation: CandidateEvaluation) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ("metric_averages", "metrics"):
        raw = evaluation.metadata.get(key)
        if isinstance(raw, Mapping):
            for name, value in raw.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[str(name)] = float(value)
    payload = evaluation.metadata.get("agent_report_evaluation")
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if isinstance(payload, Mapping):
        summary = payload.get("summary")
        if isinstance(summary, Mapping):
            raw_metrics = summary.get("metric_averages")
            if isinstance(raw_metrics, Mapping):
                for name, value in raw_metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[str(name)] = float(value)
    return metrics


def _is_better_stage_candidate(
    evaluation: CandidateEvaluation,
    stage_score: float,
    current: Optional[CandidateEvaluation],
    current_stage_score: float,
) -> bool:
    if current is None:
        return True
    return (
        stage_score,
        evaluation.score,
        -len(evaluation.candidate.patch),
        evaluation.candidate.id,
    ) > (
        current_stage_score,
        current.score,
        -len(current.candidate.patch),
        current.candidate.id,
    )


def _string_list(value: Any) -> list[str]:
    if value in (None, "", [], ()):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (Mapping, bytes)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _unique_paths(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _values_with_anchor_first(values: Sequence[Any], anchor_value: Any) -> list[Any]:
    ordered = list(values)
    for index, value in enumerate(ordered):
        if value == anchor_value:
            return [value, *ordered[:index], *ordered[index + 1:]]
    return ordered
