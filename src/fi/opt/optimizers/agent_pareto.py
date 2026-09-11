from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

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


Direction = str


class AgentParetoOptimizer(BaseOptimizer):
    """
    Optimizes agent configs with Optuna NSGA-II multi-objective search.

    Objective values are read from candidate-evaluation metadata. Direct
    scorers can return `metadata={"objectives": {"safety": 1.0}}`; simulation
    runs can use ai-evaluation agent-report `summary.metric_averages`.
    """

    def __init__(
        self,
        target: Optional[OptimizationTarget] = None,
        *,
        objective_names: Sequence[str],
        objective_directions: Optional[Sequence[Direction]] = None,
        evaluate_candidate: Optional[
            Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float]
        ] = None,
        simulation_evaluator: Any = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        n_trials: int = 32,
        seed: int = 42,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        include_seed: bool = True,
        auto_diagnose: bool = True,
        diagnostic_score_threshold: float = 0.85,
        target_score: Optional[float] = None,
        target_objectives: Optional[Mapping[str, float]] = None,
    ) -> None:
        if not objective_names:
            raise ValueError("objective_names cannot be empty.")
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")

        self.target = target
        self.objective_names = list(objective_names)
        self.objective_directions = _normalize_directions(
            objective_directions,
            objective_count=len(self.objective_names),
        )
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.diagnoses = _normalize_diagnoses(diagnoses)
        self.n_trials = n_trials
        self.seed = seed
        self.sampler = sampler or optuna.samplers.NSGAIISampler(seed=seed)
        self.storage = storage
        self.study_name = study_name
        self.include_seed = include_seed
        self.auto_diagnose = auto_diagnose
        self.diagnostic_score_threshold = diagnostic_score_threshold
        self.target_score = target_score
        self.target_objectives = dict(target_objectives or {})
        super().__init__()

    def optimize(
        self,
        evaluator: Any = None,
        data_mapper: Any = None,
        dataset: Optional[List[dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        *,
        target: Optional[OptimizationTarget] = None,
        objective_names: Optional[Sequence[str]] = None,
        objective_directions: Optional[Sequence[Direction]] = None,
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
        target_objectives: Optional[Mapping[str, float]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        active_target = target or self.target
        if active_target is None:
            raise ValueError("AgentParetoOptimizer requires a target.")

        active_objective_names = list(objective_names or self.objective_names)
        if not active_objective_names:
            raise ValueError("objective_names cannot be empty.")
        if objective_directions is not None:
            active_directions = _normalize_directions(
                objective_directions,
                objective_count=len(active_objective_names),
            )
        elif objective_names is not None:
            active_directions = ["maximize"] * len(active_objective_names)
        else:
            active_directions = list(self.objective_directions)
        if len(active_directions) != len(active_objective_names):
            raise ValueError("objective_directions must match objective_names length.")

        active_evaluator = (
            evaluate_candidate
            or self.evaluate_candidate
            or getattr(simulation_evaluator, "evaluate_candidate", None)
            or getattr(self.simulation_evaluator, "evaluate_candidate", None)
        )
        if active_evaluator is None:
            raise ValueError(
                "AgentParetoOptimizer requires evaluate_candidate or simulation_evaluator."
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
        active_target_objectives = (
            dict(self.target_objectives)
            if target_objectives is None
            else dict(target_objectives)
        )

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
                objective_names=active_objective_names,
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
        search_paths = [
            path
            for path in active_target.search_space
            if active_target.search_space[path]
        ]
        if not search_paths:
            if evaluated:
                return self._result(
                    evaluated=evaluated,
                    history=history,
                    search_paths=search_paths,
                    objective_names=active_objective_names,
                    objective_directions=active_directions,
                    active_diagnoses=active_diagnoses,
                    auto_diagnosed=use_auto_diagnose and use_include_seed,
                    duplicate_trials=duplicate_trials,
                    study=None,
                    n_trials=active_n_trials,
                )
            raise ValueError("AgentParetoOptimizer target search space cannot be empty.")

        study = optuna.create_study(
            directions=active_directions,
            sampler=self.sampler,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=bool(self.storage and self.study_name),
        )

        def objective(trial: optuna.Trial) -> List[float]:
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
                    "kind": "pareto_trial",
                    "optimizer": "AgentParetoOptimizer",
                    "pareto_trial_number": trial.number,
                    "pareto_params": trial_params,
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
                objective_names=active_objective_names,
                trial_number=trial.number,
                trial_params=trial_params,
                duplicate=duplicate,
                is_seed=False,
            )
            trial.set_user_attr("candidate_id", candidate.id)
            trial.set_user_attr("patch", patch)
            trial.set_user_attr("duplicate_candidate", duplicate)
            trial.set_user_attr(
                "objective_values",
                evaluation.metadata["objective_values"],
            )
            if _target_met(
                evaluation,
                objective_names=active_objective_names,
                objective_directions=active_directions,
                target_score=active_target_score,
                target_objectives=active_target_objectives,
            ):
                trial.study.stop()
            return [
                evaluation.metadata["objective_values"][name]
                for name in active_objective_names
            ]

        study.optimize(objective, n_trials=active_n_trials)

        if not evaluated:
            raise RuntimeError("AgentParetoOptimizer did not evaluate any candidates.")

        return self._result(
            evaluated=evaluated,
            history=history,
            search_paths=search_paths,
            objective_names=active_objective_names,
            objective_directions=active_directions,
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
        objective_names: Sequence[str],
        trial_number: Optional[int],
        trial_params: dict[str, int],
        duplicate: bool,
        is_seed: bool,
    ) -> CandidateEvaluation:
        if candidate.id in evaluated:
            return evaluated[candidate.id]

        value = evaluator(candidate)
        evaluation = _normalize_candidate_evaluation(value, candidate)
        objective_values = _objective_values(evaluation, objective_names)
        evaluation.metadata = {
            **candidate.metadata,
            **evaluation.metadata,
            "optimizer": "AgentParetoOptimizer",
            "pareto_trial_number": trial_number,
            "pareto_params": trial_params,
            "duplicate_candidate": duplicate,
            "seed_candidate": is_seed,
            "objective_values": objective_values,
        }
        evaluated[candidate.id] = evaluation
        history.append(_history_from_candidate(evaluation))
        logger.info(
            "Evaluated Pareto agent candidate %s score=%.4f objectives=%s",
            candidate.id,
            evaluation.score,
            objective_values,
        )
        return evaluation

    def _result(
        self,
        *,
        evaluated: dict[str, CandidateEvaluation],
        history: List[IterationHistory],
        search_paths: List[str],
        objective_names: List[str],
        objective_directions: List[Direction],
        active_diagnoses: List[ComponentDiagnosis],
        auto_diagnosed: bool,
        duplicate_trials: int,
        study: Optional[optuna.Study],
        n_trials: int,
    ) -> OptimizationResult:
        evaluations = list(evaluated.values())
        best = max(
            evaluations,
            key=lambda item: (item.score, -len(item.candidate.patch), item.candidate.id),
        )
        pareto_front = _pareto_front(
            evaluations,
            objective_names=objective_names,
            objective_directions=objective_directions,
        )
        completed_trials = (
            [
                trial
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
            if study is not None
            else []
        )
        metadata = {
            "optimizer": "AgentParetoOptimizer",
            "strategy": "optuna_nsga_ii",
            "sampler": self.sampler.__class__.__name__,
            "target_name": best.candidate.target_name,
            "best_candidate_id": best.candidate.id,
            "search_paths": list(search_paths),
            "objective_names": list(objective_names),
            "objective_directions": list(objective_directions),
            "pareto_front": [_pareto_entry(item) for item in pareto_front],
            "pareto_front_candidate_ids": [item.candidate.id for item in pareto_front],
            "n_trials": n_trials,
            "completed_trials": len(completed_trials),
            "duplicate_trials": duplicate_trials,
        }
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


def _normalize_directions(
    directions: Optional[Sequence[Direction]],
    *,
    objective_count: int,
) -> List[Direction]:
    if directions is None:
        return ["maximize"] * objective_count
    normalized = [item.lower() for item in directions]
    if len(normalized) != objective_count:
        raise ValueError("objective_directions must match objective_names length.")
    invalid = [item for item in normalized if item not in {"maximize", "minimize"}]
    if invalid:
        raise ValueError("objective_directions must be 'maximize' or 'minimize'.")
    return normalized


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


def _objective_values(
    evaluation: CandidateEvaluation,
    objective_names: Sequence[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in objective_names:
        value = _find_objective_value(evaluation, name)
        if value is None:
            raise ValueError(f"Missing objective value for '{name}'.")
        values[name] = float(value)
    return values


def _find_objective_value(
    evaluation: CandidateEvaluation,
    name: str,
) -> Optional[float]:
    if name in {"score", "final_score", "overall"}:
        return float(evaluation.score)

    for source_key in ("objectives", "metric_averages"):
        source = evaluation.metadata.get(source_key)
        value = _get_mapping_value(source, name)
        if value is not None:
            return float(value)

    agent_report = evaluation.metadata.get("agent_report_evaluation")
    if isinstance(agent_report, Mapping):
        summary = agent_report.get("summary")
        if isinstance(summary, Mapping):
            value = _get_mapping_value(summary.get("metric_averages"), name)
            if value is not None:
                return float(value)
    return None


def _get_mapping_value(source: Any, path: str) -> Optional[float]:
    if not isinstance(source, Mapping):
        return None
    current: Any = source
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool):
        return 1.0 if current else 0.0
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _target_met(
    evaluation: CandidateEvaluation,
    *,
    objective_names: Sequence[str],
    objective_directions: Sequence[Direction],
    target_score: Optional[float],
    target_objectives: Mapping[str, float],
) -> bool:
    if target_score is not None and evaluation.score < target_score:
        return False
    for name, direction in zip(objective_names, objective_directions):
        if name not in target_objectives:
            continue
        value = evaluation.metadata["objective_values"][name]
        threshold = target_objectives[name]
        if direction == "maximize" and value < threshold:
            return False
        if direction == "minimize" and value > threshold:
            return False
    return target_score is not None or bool(target_objectives)


def _pareto_front(
    evaluations: Sequence[CandidateEvaluation],
    *,
    objective_names: Sequence[str],
    objective_directions: Sequence[Direction],
) -> List[CandidateEvaluation]:
    front: List[CandidateEvaluation] = []
    for candidate in evaluations:
        if any(
            _dominates(
                other,
                candidate,
                objective_names=objective_names,
                objective_directions=objective_directions,
            )
            for other in evaluations
            if other.candidate.id != candidate.candidate.id
        ):
            continue
        front.append(candidate)
    return sorted(
        front,
        key=lambda item: (item.score, -len(item.candidate.patch), item.candidate.id),
        reverse=True,
    )


def _dominates(
    left: CandidateEvaluation,
    right: CandidateEvaluation,
    *,
    objective_names: Sequence[str],
    objective_directions: Sequence[Direction],
) -> bool:
    left_values = left.metadata["objective_values"]
    right_values = right.metadata["objective_values"]
    strictly_better = False
    for name, direction in zip(objective_names, objective_directions):
        left_value = left_values[name]
        right_value = right_values[name]
        if direction == "maximize":
            if left_value < right_value:
                return False
            if left_value > right_value:
                strictly_better = True
        else:
            if left_value > right_value:
                return False
            if left_value < right_value:
                strictly_better = True
    return strictly_better


def _pareto_entry(evaluation: CandidateEvaluation) -> dict[str, Any]:
    return {
        "candidate_id": evaluation.candidate.id,
        "score": evaluation.score,
        "patch": evaluation.candidate.patch,
        "objectives": evaluation.metadata["objective_values"],
    }
