from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Optional, Sequence

from ..base.base_optimizer import BaseOptimizer
from ..components import (
    ComponentDiagnosis,
    diagnose_agent_report_evaluation,
    diagnose_report,
    relevant_search_paths,
)
from ..targets import AgentCandidate, CandidateEvaluation, OptimizationTarget
from ..types import EvaluationResult, IterationHistory, OptimizationResult

logger = logging.getLogger(__name__)


class AgentOptimizer(BaseOptimizer):
    """
    Optimizes framework-neutral agent/workflow configurations.

    This is the bridge from prompt-only optimization to agent optimization. It
    can search across prompt text, policy rules, tool schemas, memory strategy,
    routers, graph/handoff settings, retriever config, voice settings, browser
    policy, CUA config, or any custom JSON-like layer.
    """

    def __init__(
        self,
        target: Optional[OptimizationTarget] = None,
        *,
        candidates: Optional[Iterable[AgentCandidate]] = None,
        evaluate_candidate: Optional[
            Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float]
        ] = None,
        simulation_evaluator: Any = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        max_candidates: Optional[int] = None,
        include_seed: bool = True,
        auto_diagnose: bool = True,
        diagnostic_score_threshold: float = 0.85,
    ) -> None:
        self.target = target
        self.candidates = list(candidates) if candidates is not None else None
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.diagnoses = _normalize_diagnoses(diagnoses)
        self.max_candidates = max_candidates
        self.include_seed = include_seed
        self.auto_diagnose = auto_diagnose
        self.diagnostic_score_threshold = diagnostic_score_threshold
        super().__init__()

    def optimize(
        self,
        evaluator: Any = None,
        data_mapper: Any = None,
        dataset: Optional[List[dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        *,
        target: Optional[OptimizationTarget] = None,
        candidates: Optional[Iterable[AgentCandidate]] = None,
        evaluate_candidate: Optional[
            Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float]
        ] = None,
        simulation_evaluator: Any = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        auto_diagnose: Optional[bool] = None,
        diagnostic_score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        active_target = target or self.target
        active_candidates = list(candidates) if candidates is not None else self.candidates
        active_diagnoses = _normalize_diagnoses(diagnoses)
        if diagnoses is None:
            active_diagnoses = list(self.diagnoses)
        use_auto_diagnose = self.auto_diagnose if auto_diagnose is None else auto_diagnose
        active_diagnostic_threshold = (
            self.diagnostic_score_threshold
            if diagnostic_score_threshold is None
            else diagnostic_score_threshold
        )
        active_evaluator = (
            evaluate_candidate
            or self.evaluate_candidate
            or getattr(simulation_evaluator, "evaluate_candidate", None)
            or getattr(self.simulation_evaluator, "evaluate_candidate", None)
        )

        pre_evaluations: List[CandidateEvaluation] = []

        if active_candidates is None:
            if active_target is None:
                raise ValueError("AgentOptimizer requires a target or candidates.")
            if active_evaluator is None:
                raise ValueError(
                    "AgentOptimizer requires evaluate_candidate or simulation_evaluator."
                )

            if use_auto_diagnose and not active_diagnoses and self.include_seed:
                seed_candidate = active_target.seed_candidate()
                seed_evaluation = _normalize_candidate_evaluation(
                    active_evaluator(seed_candidate),
                    seed_candidate,
                )
                seed_diagnoses = _diagnose_candidate_evaluation(
                    seed_evaluation,
                    failing_threshold=active_diagnostic_threshold,
                )
                if seed_diagnoses:
                    active_diagnoses = seed_diagnoses
                pre_evaluations.append(seed_evaluation)

            active_target = _target_for_diagnoses(active_target, active_diagnoses)
            include_seed = self.include_seed and not pre_evaluations
            remaining_candidates = _remaining_candidate_budget(
                self.max_candidates,
                already_evaluated=len(pre_evaluations),
            )
            if remaining_candidates == 0:
                active_candidates = []
            else:
                active_candidates = list(
                    active_target.iter_candidates(
                        include_seed=include_seed,
                        max_candidates=remaining_candidates,
                    )
                )

        if not active_candidates and not pre_evaluations:
            raise ValueError("AgentOptimizer candidate list cannot be empty.")

        if active_evaluator is None:
            raise ValueError(
                "AgentOptimizer requires evaluate_candidate or simulation_evaluator."
            )

        best: CandidateEvaluation | None = None
        history: List[IterationHistory] = []

        for evaluation in pre_evaluations:
            history.append(_history_from_candidate(evaluation))
            best = evaluation if best is None or evaluation.score > best.score else best

        for index, candidate in enumerate(active_candidates):
            logger.info(
                "Evaluating agent candidate %s/%s: %s",
                index + 1,
                len(active_candidates),
                candidate.id,
            )
            evaluation = _normalize_candidate_evaluation(
                active_evaluator(candidate),
                candidate,
            )
            history.append(_history_from_candidate(evaluation))

            if best is None or evaluation.score > best.score:
                best = evaluation
                logger.info(
                    "New best agent candidate %s score=%.4f",
                    candidate.id,
                    evaluation.score,
                )

        assert best is not None
        metadata = {
            "optimizer": "AgentOptimizer",
            "target_name": best.candidate.target_name,
            "best_candidate_id": best.candidate.id,
            "search_paths": list(active_target.search_space.keys()) if active_target else [],
        }
        if active_diagnoses:
            metadata["diagnostics"] = [_dump_model(item) for item in active_diagnoses]
            metadata["auto_diagnosed"] = use_auto_diagnose and bool(pre_evaluations)
        return OptimizationResult(
            best_generator=best.candidate,
            best_candidate=best.candidate,
            history=history,
            final_score=best.score,
            total_iterations=len(history),
            total_evaluations=len(history),
            metadata=metadata,
        )


def _normalize_candidate_evaluation(
    value: CandidateEvaluation | EvaluationResult | float,
    candidate: AgentCandidate,
) -> CandidateEvaluation:
    if isinstance(value, CandidateEvaluation):
        return value
    if isinstance(value, EvaluationResult):
        return CandidateEvaluation(
            candidate=candidate,
            score=value.score,
            reason=value.reason,
            individual_results=[value],
            metadata=value.metadata,
        )
    if isinstance(value, (int, float)):
        return CandidateEvaluation(candidate=candidate, score=float(value))
    raise TypeError(
        "evaluate_candidate must return CandidateEvaluation, EvaluationResult, int, or float."
    )


def _history_from_candidate(evaluation: CandidateEvaluation) -> IterationHistory:
    result = EvaluationResult(
        score=evaluation.score,
        reason=evaluation.reason,
        metadata={
            "candidate_id": evaluation.candidate.id,
            "patch": evaluation.candidate.patch,
            **evaluation.metadata,
        },
    )
    individual_results = evaluation.individual_results or [result]
    return IterationHistory(
        prompt=evaluation.candidate.id,
        average_score=evaluation.score,
        individual_results=individual_results,
        candidate_id=evaluation.candidate.id,
        candidate_config=evaluation.candidate.config,
        layers=evaluation.candidate.layers,
        metadata={
            "reason": evaluation.reason,
            "patch": evaluation.candidate.patch,
            "report": evaluation.report,
            **evaluation.metadata,
        },
    )


def _target_for_diagnoses(
    target: OptimizationTarget,
    diagnoses: Sequence[ComponentDiagnosis],
) -> OptimizationTarget:
    if not diagnoses or not target.search_space:
        return target

    allowed_paths = relevant_search_paths(target.search_space, diagnoses)
    filtered_search_space = {
        path: values
        for path, values in target.search_space.items()
        if path in allowed_paths
    }
    if filtered_search_space == target.search_space:
        return target

    return OptimizationTarget(
        name=target.name,
        base_config=target.base_config,
        layers=target.layers,
        search_space=filtered_search_space,
        metadata={
            **target.metadata,
            "diagnostic_search_paths": sorted(filtered_search_space.keys()),
        },
    )


def _diagnose_candidate_evaluation(
    evaluation: CandidateEvaluation,
    *,
    failing_threshold: float,
) -> List[ComponentDiagnosis]:
    diagnostics = _normalize_diagnoses(evaluation.metadata.get("diagnostics"))
    if diagnostics:
        return diagnostics

    agent_report = evaluation.metadata.get("agent_report_evaluation")
    if agent_report is not None:
        diagnostics.extend(
            diagnose_agent_report_evaluation(
                agent_report,
                failing_threshold=failing_threshold,
            )
        )
        return _dedupe_diagnoses(diagnostics)

    diagnostics.extend(diagnose_report(evaluation.report))
    return _dedupe_diagnoses(diagnostics)


def _dedupe_diagnoses(
    diagnoses: Iterable[ComponentDiagnosis],
) -> List[ComponentDiagnosis]:
    best: dict[tuple[str, str], ComponentDiagnosis] = {}
    for diagnosis in diagnoses:
        key = (diagnosis.component, diagnosis.failure_mode)
        if key not in best or diagnosis.confidence > best[key].confidence:
            best[key] = diagnosis
    return sorted(
        best.values(),
        key=lambda item: (item.confidence, item.component, item.failure_mode),
        reverse=True,
    )


def _remaining_candidate_budget(
    max_candidates: Optional[int],
    *,
    already_evaluated: int,
) -> Optional[int]:
    if max_candidates is None:
        return None
    return max(0, max_candidates - already_evaluated)


def _normalize_diagnoses(
    diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]],
) -> List[ComponentDiagnosis]:
    if diagnoses is None:
        return []
    normalized: List[ComponentDiagnosis] = []
    for item in diagnoses:
        if isinstance(item, ComponentDiagnosis):
            normalized.append(item)
        else:
            normalized.append(ComponentDiagnosis(**item))
    return normalized


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value
