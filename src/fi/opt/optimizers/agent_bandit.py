from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional

from ..base.base_optimizer import BaseOptimizer
from ..components import ComponentDiagnosis
from ..targets import AgentCandidate, CandidateEvaluation, OptimizationTarget
from ..types import EvaluationResult, IterationHistory, OptimizationResult
from .agent import (
    _diagnose_candidate_evaluation,
    _dump_model,
    _history_from_candidate,
    _normalize_candidate_evaluation,
    _normalize_diagnoses,
    _target_for_diagnoses,
)

logger = logging.getLogger(__name__)


@dataclass
class _ArmStats:
    candidate: AgentCandidate
    pulls: int = 0
    total_score: float = 0.0
    best_score: float = float("-inf")
    best_evaluation: Optional[CandidateEvaluation] = None
    evaluations: List[CandidateEvaluation] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_score / self.pulls

    @property
    def score_std(self) -> float:
        if self.pulls < 2:
            return 0.0
        mean = self.mean_score
        variance = sum(
            (evaluation.score - mean) ** 2 for evaluation in self.evaluations
        ) / (self.pulls - 1)
        return math.sqrt(max(0.0, variance))


class AgentBanditOptimizer(BaseOptimizer):
    """
    Optimizes agent configs with UCB-style adaptive evaluation allocation.

    Candidate configs are treated as bandit arms. This is useful when
    simulation/evaluation is noisy or expensive and promising configs should
    receive more repeated trials than weak configs.
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
        max_candidates: int = 64,
        total_budget: int = 32,
        min_pulls_per_candidate: int = 1,
        exploration: float = math.sqrt(2.0),
        include_seed: bool = True,
        auto_diagnose: bool = True,
        diagnostic_score_threshold: float = 0.85,
        target_score: Optional[float] = None,
        selection: str = "mean",
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1.")
        if total_budget < 1:
            raise ValueError("total_budget must be at least 1.")
        if min_pulls_per_candidate < 1:
            raise ValueError("min_pulls_per_candidate must be at least 1.")
        if exploration < 0:
            raise ValueError("exploration must be non-negative.")
        if selection not in {"mean", "best"}:
            raise ValueError("selection must be 'mean' or 'best'.")

        self.target = target
        self.candidates = list(candidates) if candidates is not None else None
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.diagnoses = _normalize_diagnoses(diagnoses)
        self.max_candidates = max_candidates
        self.total_budget = total_budget
        self.min_pulls_per_candidate = min_pulls_per_candidate
        self.exploration = exploration
        self.include_seed = include_seed
        self.auto_diagnose = auto_diagnose
        self.diagnostic_score_threshold = diagnostic_score_threshold
        self.target_score = target_score
        self.selection = selection
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
        max_candidates: Optional[int] = None,
        total_budget: Optional[int] = None,
        min_pulls_per_candidate: Optional[int] = None,
        exploration: Optional[float] = None,
        include_seed: Optional[bool] = None,
        auto_diagnose: Optional[bool] = None,
        diagnostic_score_threshold: Optional[float] = None,
        target_score: Optional[float] = None,
        selection: Optional[str] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        active_target = target or self.target
        active_candidates = (
            list(candidates) if candidates is not None else self.candidates
        )
        active_evaluator = (
            evaluate_candidate
            or self.evaluate_candidate
            or getattr(simulation_evaluator, "evaluate_candidate", None)
            or getattr(self.simulation_evaluator, "evaluate_candidate", None)
        )
        if active_evaluator is None:
            raise ValueError(
                "AgentBanditOptimizer requires evaluate_candidate or simulation_evaluator."
            )

        active_max_candidates = (
            self.max_candidates if max_candidates is None else max_candidates
        )
        active_total_budget = self.total_budget if total_budget is None else total_budget
        active_min_pulls = (
            self.min_pulls_per_candidate
            if min_pulls_per_candidate is None
            else min_pulls_per_candidate
        )
        active_exploration = self.exploration if exploration is None else exploration
        use_include_seed = self.include_seed if include_seed is None else include_seed
        use_auto_diagnose = self.auto_diagnose if auto_diagnose is None else auto_diagnose
        active_diagnostic_threshold = (
            self.diagnostic_score_threshold
            if diagnostic_score_threshold is None
            else diagnostic_score_threshold
        )
        active_target_score = self.target_score if target_score is None else target_score
        active_selection = self.selection if selection is None else selection
        if active_max_candidates < 1:
            raise ValueError("max_candidates must be at least 1.")
        if active_total_budget < 1:
            raise ValueError("total_budget must be at least 1.")
        if active_min_pulls < 1:
            raise ValueError("min_pulls_per_candidate must be at least 1.")
        if active_exploration < 0:
            raise ValueError("exploration must be non-negative.")
        if active_selection not in {"mean", "best"}:
            raise ValueError("selection must be 'mean' or 'best'.")

        active_diagnoses = _normalize_diagnoses(diagnoses)
        if diagnoses is None:
            active_diagnoses = list(self.diagnoses)

        pre_evaluation: Optional[CandidateEvaluation] = None
        if active_candidates is None:
            if active_target is None:
                raise ValueError("AgentBanditOptimizer requires a target or candidates.")
            if use_auto_diagnose and not active_diagnoses and use_include_seed:
                seed_candidate = active_target.seed_candidate()
                pre_evaluation = self._evaluate(
                    seed_candidate,
                    active_evaluator,
                    pull_number=1,
                    arm_pull_number=1,
                    role="seed",
                )
                active_diagnoses = _diagnose_candidate_evaluation(
                    pre_evaluation,
                    failing_threshold=active_diagnostic_threshold,
                )
            active_target = _target_for_diagnoses(active_target, active_diagnoses)
            active_candidates = list(
                active_target.iter_candidates(
                    include_seed=use_include_seed,
                    max_candidates=active_max_candidates,
                )
            )

        if not active_candidates:
            raise ValueError("AgentBanditOptimizer candidate list cannot be empty.")

        arms = {
            candidate.id: _ArmStats(candidate=candidate)
            for candidate in active_candidates
        }
        history: List[IterationHistory] = []
        total_pulls = 0

        if pre_evaluation is not None and pre_evaluation.candidate.id in arms:
            self._record_pull(
                arms[pre_evaluation.candidate.id],
                pre_evaluation,
                history,
                pull_number=1,
                arm_pull_number=1,
            )
            total_pulls = 1

        while total_pulls < active_total_budget:
            arm = _select_arm(
                list(arms.values()),
                total_pulls=max(1, total_pulls),
                min_pulls=active_min_pulls,
                exploration=active_exploration,
            )
            evaluation = self._evaluate(
                arm.candidate,
                active_evaluator,
                pull_number=total_pulls + 1,
                arm_pull_number=arm.pulls + 1,
                role="bandit",
            )
            self._record_pull(
                arm,
                evaluation,
                history,
                pull_number=total_pulls + 1,
                arm_pull_number=arm.pulls + 1,
            )
            total_pulls += 1
            if (
                active_target_score is not None
                and _arm_rank_score(arm, active_selection) >= active_target_score
            ):
                break

        if not history:
            raise RuntimeError("AgentBanditOptimizer did not evaluate any candidates.")

        best_arm = max(
            arms.values(),
            key=lambda item: (
                _arm_rank_score(item, active_selection),
                item.mean_score,
                item.best_score,
                -len(item.candidate.patch),
                item.candidate.id,
            ),
        )
        assert best_arm.best_evaluation is not None
        metadata = {
            "optimizer": "AgentBanditOptimizer",
            "strategy": "ucb1",
            "target_name": best_arm.candidate.target_name,
            "best_candidate_id": best_arm.candidate.id,
            "selection": active_selection,
            "exploration": active_exploration,
            "total_budget": active_total_budget,
            "total_pulls": total_pulls,
            "min_pulls_per_candidate": active_min_pulls,
            "candidate_count": len(arms),
            "search_paths": list(active_target.search_space.keys())
            if active_target is not None
            else [],
            "arms": [_arm_summary(arm) for arm in arms.values()],
        }
        if active_diagnoses:
            metadata["diagnostics"] = [_dump_model(item) for item in active_diagnoses]
            metadata["auto_diagnosed"] = use_auto_diagnose and pre_evaluation is not None

        return OptimizationResult(
            best_generator=best_arm.candidate,
            best_candidate=best_arm.candidate,
            history=history,
            final_score=_arm_rank_score(best_arm, active_selection),
            total_iterations=len(history),
            total_evaluations=len(history),
            metadata=metadata,
        )

    def _evaluate(
        self,
        candidate: AgentCandidate,
        evaluator: Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float],
        *,
        pull_number: int,
        arm_pull_number: int,
        role: str,
    ) -> CandidateEvaluation:
        value = evaluator(candidate)
        evaluation = _normalize_candidate_evaluation(value, candidate)
        evaluation.metadata = {
            **candidate.metadata,
            **evaluation.metadata,
            "optimizer": "AgentBanditOptimizer",
            "bandit_pull_number": pull_number,
            "bandit_arm_pull_number": arm_pull_number,
            "bandit_role": role,
        }
        return evaluation

    def _record_pull(
        self,
        arm: _ArmStats,
        evaluation: CandidateEvaluation,
        history: List[IterationHistory],
        *,
        pull_number: int,
        arm_pull_number: int,
    ) -> None:
        arm.pulls += 1
        arm.total_score += evaluation.score
        arm.evaluations.append(evaluation)
        if evaluation.score > arm.best_score or arm.best_evaluation is None:
            arm.best_score = evaluation.score
            arm.best_evaluation = evaluation
        evaluation.metadata = {
            **evaluation.metadata,
            "bandit_pull_number": pull_number,
            "bandit_arm_pull_number": arm_pull_number,
            "bandit_running_mean": arm.mean_score,
            "bandit_running_best": arm.best_score,
            "bandit_running_std": arm.score_std,
        }
        history.append(_history_from_candidate(evaluation))


def _select_arm(
    arms: List[_ArmStats],
    *,
    total_pulls: int,
    min_pulls: int,
    exploration: float,
) -> _ArmStats:
    under_sampled = [arm for arm in arms if arm.pulls < min_pulls]
    if under_sampled:
        return min(under_sampled, key=lambda item: (item.pulls, item.candidate.id))
    return max(
        arms,
        key=lambda item: (
            _ucb_score(item, total_pulls=total_pulls, exploration=exploration),
            item.mean_score,
            item.best_score,
            item.candidate.id,
        ),
    )


def _ucb_score(
    arm: _ArmStats,
    *,
    total_pulls: int,
    exploration: float,
) -> float:
    if arm.pulls == 0:
        return float("inf")
    return arm.mean_score + exploration * math.sqrt(
        math.log(max(total_pulls, 2)) / arm.pulls
    )


def _arm_rank_score(arm: _ArmStats, selection: str) -> float:
    if selection == "best":
        return arm.best_score
    return arm.mean_score


def _arm_summary(arm: _ArmStats) -> dict[str, Any]:
    return {
        "candidate_id": arm.candidate.id,
        "patch": arm.candidate.patch,
        "pulls": arm.pulls,
        "mean_score": arm.mean_score,
        "best_score": arm.best_score,
        "score_std": arm.score_std,
    }
