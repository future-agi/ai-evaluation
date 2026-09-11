from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Optional

import optuna

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


class AgentTPEOptimizer(BaseOptimizer):
    """
    Optimizes agent configs with Optuna's TPE sampler.

    This is the algorithm-backed counterpart to deterministic
    `AgentOptimizer`. It keeps the same `OptimizationTarget` and evaluator
    contracts while using a trial sampler to choose categorical config patches.
    """

    def __init__(
        self,
        target: Optional[OptimizationTarget] = None,
        *,
        evaluate_candidate: Optional[
            Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float]
        ] = None,
        simulation_evaluator: Any = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        n_trials: int = 24,
        seed: int = 42,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        include_seed: bool = True,
        auto_diagnose: bool = True,
        diagnostic_score_threshold: float = 0.85,
        target_score: float = 1.0,
    ) -> None:
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")

        self.target = target
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.diagnoses = _normalize_diagnoses(diagnoses)
        self.n_trials = n_trials
        self.seed = seed
        self.sampler = sampler or optuna.samplers.TPESampler(seed=seed)
        self.pruner = pruner
        self.storage = storage
        self.study_name = study_name
        self.include_seed = include_seed
        self.auto_diagnose = auto_diagnose
        self.diagnostic_score_threshold = diagnostic_score_threshold
        self.target_score = target_score
        super().__init__()

    def optimize(
        self,
        evaluator: Any = None,
        data_mapper: Any = None,
        dataset: Optional[List[dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        *,
        target: Optional[OptimizationTarget] = None,
        evaluate_candidate: Optional[
            Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float]
        ] = None,
        simulation_evaluator: Any = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        n_trials: Optional[int] = None,
        include_seed: Optional[bool] = None,
        auto_diagnose: Optional[bool] = None,
        diagnostic_score_threshold: Optional[float] = None,
        target_score: Optional[float] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        active_target = target or self.target
        if active_target is None:
            raise ValueError("AgentTPEOptimizer requires a target.")

        active_evaluator = (
            evaluate_candidate
            or self.evaluate_candidate
            or getattr(simulation_evaluator, "evaluate_candidate", None)
            or getattr(self.simulation_evaluator, "evaluate_candidate", None)
        )
        if active_evaluator is None:
            raise ValueError(
                "AgentTPEOptimizer requires evaluate_candidate or simulation_evaluator."
            )

        active_n_trials = self.n_trials if n_trials is None else n_trials
        if active_n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        active_diagnoses = _normalize_diagnoses(diagnoses)
        if diagnoses is None:
            active_diagnoses = list(self.diagnoses)
        use_include_seed = self.include_seed if include_seed is None else include_seed
        use_auto_diagnose = self.auto_diagnose if auto_diagnose is None else auto_diagnose
        active_diagnostic_threshold = (
            self.diagnostic_score_threshold
            if diagnostic_score_threshold is None
            else diagnostic_score_threshold
        )
        active_target_score = self.target_score if target_score is None else target_score

        seed_candidate = active_target.seed_candidate()
        evaluated: dict[str, CandidateEvaluation] = {}
        history: List[IterationHistory] = []
        duplicate_trials = 0

        if use_include_seed:
            seed_evaluation = self._evaluate(
                seed_candidate,
                active_evaluator,
                evaluated,
                history,
                trial_number=None,
                trial_params={},
                duplicate=False,
                is_seed=True,
            )
            if use_auto_diagnose and not active_diagnoses:
                active_diagnoses = _diagnose_candidate_evaluation(
                    seed_evaluation,
                    failing_threshold=active_diagnostic_threshold,
                )

        active_target = _target_for_diagnoses(active_target, active_diagnoses)
        search_paths = [path for path in active_target.search_space if active_target.search_space[path]]
        if not search_paths:
            if evaluated:
                return self._result(
                    evaluated=evaluated,
                    history=history,
                    search_paths=search_paths,
                    active_diagnoses=active_diagnoses,
                    auto_diagnosed=use_auto_diagnose and use_include_seed,
                    duplicate_trials=duplicate_trials,
                    study=None,
                    n_trials=active_n_trials,
                )
            raise ValueError("AgentTPEOptimizer target search space cannot be empty.")

        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=bool(self.storage and self.study_name),
        )

        def objective(trial: optuna.Trial) -> float:
            nonlocal duplicate_trials
            patch, trial_params = _trial_patch(
                trial,
                seed_candidate=seed_candidate,
                search_space=active_target.search_space,
                search_paths=search_paths,
            )
            candidate = seed_candidate.with_patch(
                patch,
                metadata={
                    "kind": "tpe_trial",
                    "optimizer": "AgentTPEOptimizer",
                    "tpe_trial_number": trial.number,
                    "tpe_params": trial_params,
                },
            )
            duplicate = candidate.id in evaluated
            if duplicate:
                duplicate_trials += 1
            evaluation = self._evaluate(
                candidate,
                active_evaluator,
                evaluated,
                history,
                trial_number=trial.number,
                trial_params=trial_params,
                duplicate=duplicate,
                is_seed=False,
            )
            trial.set_user_attr("candidate_id", candidate.id)
            trial.set_user_attr("patch", patch)
            trial.set_user_attr("duplicate_candidate", duplicate)
            if evaluation.score >= active_target_score:
                trial.study.stop()
            return evaluation.score

        study.optimize(objective, n_trials=active_n_trials)

        if not evaluated:
            raise RuntimeError("AgentTPEOptimizer did not evaluate any candidates.")

        return self._result(
            evaluated=evaluated,
            history=history,
            search_paths=search_paths,
            active_diagnoses=active_diagnoses,
            auto_diagnosed=use_auto_diagnose and use_include_seed,
            duplicate_trials=duplicate_trials,
            study=study,
            n_trials=active_n_trials,
        )

    def _evaluate(
        self,
        candidate: AgentCandidate,
        evaluator: Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float],
        evaluated: dict[str, CandidateEvaluation],
        history: List[IterationHistory],
        *,
        trial_number: Optional[int],
        trial_params: dict[str, int],
        duplicate: bool,
        is_seed: bool,
    ) -> CandidateEvaluation:
        if candidate.id in evaluated:
            return evaluated[candidate.id]

        value = evaluator(candidate)
        evaluation = _normalize_candidate_evaluation(value, candidate)
        evaluation.metadata = {
            **candidate.metadata,
            **evaluation.metadata,
            "optimizer": "AgentTPEOptimizer",
            "tpe_trial_number": trial_number,
            "tpe_params": trial_params,
            "duplicate_candidate": duplicate,
            "seed_candidate": is_seed,
        }
        evaluated[candidate.id] = evaluation
        history.append(_history_from_candidate(evaluation))
        logger.info(
            "Evaluated TPE agent candidate %s score=%.4f",
            candidate.id,
            evaluation.score,
        )
        return evaluation

    def _result(
        self,
        *,
        evaluated: dict[str, CandidateEvaluation],
        history: List[IterationHistory],
        search_paths: List[str],
        active_diagnoses: List[ComponentDiagnosis],
        auto_diagnosed: bool,
        duplicate_trials: int,
        study: Optional[optuna.Study],
        n_trials: int,
    ) -> OptimizationResult:
        best = max(
            evaluated.values(),
            key=lambda item: (item.score, -len(item.candidate.patch), item.candidate.id),
        )
        completed_trials = (
            [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
            if study is not None
            else []
        )
        metadata = {
            "optimizer": "AgentTPEOptimizer",
            "strategy": "optuna_tpe",
            "sampler": self.sampler.__class__.__name__,
            "target_name": best.candidate.target_name,
            "best_candidate_id": best.candidate.id,
            "search_paths": list(search_paths),
            "n_trials": n_trials,
            "completed_trials": len(completed_trials),
            "duplicate_trials": duplicate_trials,
        }
        if study is not None and completed_trials:
            metadata["best_trial_number"] = study.best_trial.number
        if active_diagnoses:
            metadata["diagnostics"] = [_dump_model(item) for item in active_diagnoses]
            metadata["auto_diagnosed"] = auto_diagnosed
        return OptimizationResult(
            best_generator=best.candidate,
            best_candidate=best.candidate,
            history=history,
            final_score=best.score,
            total_iterations=len(history),
            total_evaluations=len(history),
            metadata=metadata,
        )


def _trial_patch(
    trial: optuna.Trial,
    *,
    seed_candidate: AgentCandidate,
    search_space: dict[str, List[Any]],
    search_paths: List[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    patch: dict[str, Any] = {}
    trial_params: dict[str, int] = {}
    for path in search_paths:
        values = list(search_space[path])
        choice_index = trial.suggest_categorical(path, list(range(len(values))))
        trial_params[path] = int(choice_index)
        value = values[int(choice_index)]
        if value != seed_candidate.get_path(path):
            patch[path] = value
    return patch, trial_params
