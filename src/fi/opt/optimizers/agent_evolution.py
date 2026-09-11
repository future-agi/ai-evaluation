from __future__ import annotations

import logging
import random
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

from ..base.base_optimizer import BaseOptimizer
from ..components import ComponentDiagnosis
from ..mutations import (
    AgentMutationBundle,
    AgentMutationLibrary,
    dump_mutation_bundle,
    resolve_agent_mutation_library,
)
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


DEFAULT_LAYER_PATH_BIAS: Mapping[str, Sequence[str]] = {
    "prompt": ("prompt", "instructions", "system"),
    "policy": ("policy", "guardrail", "security", "safety"),
    "tools": ("tools", "tool", "function"),
    "memory": ("memory", "session", "checkpoint"),
    "router": ("router", "routing", "model"),
    "retrieval": ("retrieval", "retriever", "rag", "knowledge"),
    "retriever": ("retrieval", "retriever", "rag", "knowledge"),
    "model": ("model", "router"),
    "voice": ("voice", "audio", "vad", "stt", "tts"),
    "browser": ("browser", "cua", "selectors", "screenshot"),
    "cua": ("browser", "cua", "action"),
    "multi_agent": ("multi_agent", "handoff", "review", "reconciliation"),
    "orchestration": ("orchestration", "graph", "workflow"),
    "streaming": ("streaming", "chunk", "interruption"),
    "world": ("world", "state", "contract"),
    "framework": ("framework", "trace", "adapter"),
    "security": ("security", "policy", "trust"),
}


class AgentEvolutionOptimizer(BaseOptimizer):
    """
    Optimizes agent configs with deterministic evolutionary mutation.

    Mutation is domain-aware: config paths tied to active target layers or
    component diagnoses receive higher mutation probability than unrelated
    paths. This is useful when interacting config changes must be discovered
    without exhaustively enumerating the whole search space.
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
        population_size: int = 12,
        generations: int = 4,
        elite_count: int = 2,
        mutation_rate: float = 0.65,
        crossover_rate: float = 0.75,
        max_mutations_per_candidate: int = 2,
        tournament_size: int = 3,
        selection: str = "tournament",        # "tournament" (legacy) | "elo" — explicit opt-in
        eval_budget: Optional[int] = None,    # declared budget; None = unbounded (legacy)
        elo_k_factor: float = 32.0,
        elo_initial_rating: float = 1500.0,   # ARCH Decision 6
        seed: int = 42,
        include_seed: bool = True,
        auto_diagnose: bool = True,
        diagnostic_score_threshold: float = 0.85,
        target_score: Optional[float] = None,
        layer_path_bias: Optional[Mapping[str, Sequence[str]]] = None,
        mutation_library: Optional[
            AgentMutationLibrary | Iterable[AgentMutationBundle] | bool
        ] = True,
        max_library_candidates: int = 8,
    ) -> None:
        _validate_evolution_params(
            population_size=population_size,
            generations=generations,
            elite_count=elite_count,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            max_mutations_per_candidate=max_mutations_per_candidate,
            tournament_size=tournament_size,
        )
        _validate_selection_params(
            selection=selection,
            eval_budget=eval_budget,
            elo_k_factor=elo_k_factor,
            elo_initial_rating=elo_initial_rating,
        )
        self.target = target
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.diagnoses = _normalize_diagnoses(diagnoses)
        self.population_size = population_size
        self.generations = generations
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_mutations_per_candidate = max_mutations_per_candidate
        self.tournament_size = tournament_size
        self.selection = selection
        self.eval_budget = eval_budget
        self.elo_k_factor = elo_k_factor
        self.elo_initial_rating = elo_initial_rating
        self.seed = seed
        self.include_seed = include_seed
        self.auto_diagnose = auto_diagnose
        self.diagnostic_score_threshold = diagnostic_score_threshold
        self.target_score = target_score
        self.layer_path_bias = _merged_layer_path_bias(layer_path_bias)
        self.mutation_library = mutation_library
        self.max_library_candidates = max_library_candidates
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
        population_size: Optional[int] = None,
        generations: Optional[int] = None,
        elite_count: Optional[int] = None,
        mutation_rate: Optional[float] = None,
        crossover_rate: Optional[float] = None,
        max_mutations_per_candidate: Optional[int] = None,
        tournament_size: Optional[int] = None,
        selection: Optional[str] = None,
        eval_budget: Optional[int] = None,
        elo_k_factor: Optional[float] = None,
        elo_initial_rating: Optional[float] = None,
        seed: Optional[int] = None,
        include_seed: Optional[bool] = None,
        auto_diagnose: Optional[bool] = None,
        diagnostic_score_threshold: Optional[float] = None,
        target_score: Optional[float] = None,
        layer_path_bias: Optional[Mapping[str, Sequence[str]]] = None,
        mutation_library: Optional[
            AgentMutationLibrary | Iterable[AgentMutationBundle] | bool
        ] = None,
        max_library_candidates: Optional[int] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        active_target = target or self.target
        if active_target is None:
            raise ValueError("AgentEvolutionOptimizer requires a target.")

        active_evaluator = (
            evaluate_candidate
            or self.evaluate_candidate
            or getattr(simulation_evaluator, "evaluate_candidate", None)
            or getattr(self.simulation_evaluator, "evaluate_candidate", None)
        )
        if active_evaluator is None:
            raise ValueError(
                "AgentEvolutionOptimizer requires evaluate_candidate or simulation_evaluator."
            )

        active_population_size = (
            self.population_size if population_size is None else population_size
        )
        active_generations = self.generations if generations is None else generations
        active_elite_count = self.elite_count if elite_count is None else elite_count
        active_mutation_rate = self.mutation_rate if mutation_rate is None else mutation_rate
        active_crossover_rate = (
            self.crossover_rate if crossover_rate is None else crossover_rate
        )
        active_max_mutations = (
            self.max_mutations_per_candidate
            if max_mutations_per_candidate is None
            else max_mutations_per_candidate
        )
        active_tournament_size = (
            self.tournament_size if tournament_size is None else tournament_size
        )
        active_selection = self.selection if selection is None else selection
        active_eval_budget = self.eval_budget if eval_budget is None else eval_budget
        active_elo_k_factor = (
            self.elo_k_factor if elo_k_factor is None else elo_k_factor
        )
        active_elo_initial_rating = (
            self.elo_initial_rating
            if elo_initial_rating is None
            else elo_initial_rating
        )
        active_seed = self.seed if seed is None else seed
        use_include_seed = self.include_seed if include_seed is None else include_seed
        use_auto_diagnose = self.auto_diagnose if auto_diagnose is None else auto_diagnose
        active_diagnostic_threshold = (
            self.diagnostic_score_threshold
            if diagnostic_score_threshold is None
            else diagnostic_score_threshold
        )
        active_target_score = self.target_score if target_score is None else target_score
        active_layer_path_bias = (
            self.layer_path_bias
            if layer_path_bias is None
            else _merged_layer_path_bias(layer_path_bias)
        )
        active_mutation_library = resolve_agent_mutation_library(
            self.mutation_library if mutation_library is None else mutation_library
        )
        active_max_library_candidates = (
            self.max_library_candidates
            if max_library_candidates is None
            else max_library_candidates
        )
        _validate_evolution_params(
            population_size=active_population_size,
            generations=active_generations,
            elite_count=active_elite_count,
            mutation_rate=active_mutation_rate,
            crossover_rate=active_crossover_rate,
            max_mutations_per_candidate=active_max_mutations,
            tournament_size=active_tournament_size,
        )
        _validate_selection_params(
            selection=active_selection,
            eval_budget=active_eval_budget,
            elo_k_factor=active_elo_k_factor,
            elo_initial_rating=active_elo_initial_rating,
        )
        if active_max_library_candidates < 0:
            raise ValueError("max_library_candidates must be non-negative.")

        active_diagnoses = _normalize_diagnoses(diagnoses)
        if diagnoses is None:
            active_diagnoses = list(self.diagnoses)

        rng = random.Random(active_seed)
        original_target = active_target
        seed_candidate = original_target.seed_candidate()
        evaluated: dict[str, CandidateEvaluation] = {}
        history: List[IterationHistory] = []
        generation_summaries: List[dict[str, Any]] = []

        if use_auto_diagnose and not active_diagnoses and use_include_seed:
            seed_evaluation = self._evaluate(
                seed_candidate,
                active_evaluator,
                evaluated,
                history,
                generation=0,
                role="seed",
            )
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
        library_bundles: List[AgentMutationBundle] = []
        if active_mutation_library is not None and active_max_library_candidates:
            library_bundles = active_mutation_library.propose(
                original_target,
                diagnoses=active_diagnoses,
                search_paths=search_paths,
                max_bundles=active_max_library_candidates,
            )
            for bundle in library_bundles:
                for path in bundle.patch:
                    if path not in active_target.search_space and path in original_target.search_space:
                        active_target.search_space[path] = original_target.search_space[path]
                    if path not in search_paths and path in active_target.search_space:
                        search_paths.append(path)
        if not search_paths:
            raise ValueError("AgentEvolutionOptimizer target search space cannot be empty.")

        path_weights = _mutation_path_weights(
            search_paths,
            target=active_target,
            diagnoses=active_diagnoses,
            layer_path_bias=active_layer_path_bias,
        )
        population = _initial_population(
            seed_candidate=seed_candidate,
            search_space=active_target.search_space,
            search_paths=search_paths,
            population_size=active_population_size,
            include_seed=use_include_seed,
            rng=rng,
            path_weights=path_weights,
            library_bundles=library_bundles,
        )

        best: Optional[CandidateEvaluation] = None
        budget_exhausted = False
        for generation in range(active_generations + 1):
            generation_evaluations: List[CandidateEvaluation] = []
            for candidate in population:
                if (
                    active_eval_budget is not None
                    and candidate.id not in evaluated
                    and len(evaluated) >= active_eval_budget
                ):
                    budget_exhausted = True
                    break
                evaluation = self._evaluate(
                    candidate,
                    active_evaluator,
                    evaluated,
                    history,
                    generation=generation,
                    role=candidate.metadata.get("evolution_role", "population"),
                )
                generation_evaluations.append(evaluation)
                if best is None or evaluation.score > best.score:
                    best = evaluation

            if generation_evaluations:
                generation_evaluations = sorted(
                    generation_evaluations,
                    key=lambda item: (
                        item.score,
                        -len(item.candidate.patch),
                        item.candidate.id,
                    ),
                    reverse=True,
                )
                generation_summaries.append(
                    {
                        "generation": generation,
                        "population": len(population),
                        "best_score": generation_evaluations[0].score,
                        "best_candidate_id": generation_evaluations[0].candidate.id,
                    }
                )
            if budget_exhausted:
                break
            if (
                active_target_score is not None
                and best is not None
                and best.score >= active_target_score
            ):
                break
            if generation >= active_generations:
                break

            elo_rankings: Optional[List[tuple[CandidateEvaluation, float]]] = None
            if active_selection == "elo":
                # Deterministic round-robin Elo over already-evaluated candidates
                # (RoboPhD discipline): selection pressure changes under a fixed
                # budget; no extra rollouts, no LLM ranking.
                elo_rankings = _elo_tournament_ranking(
                    generation_evaluations,
                    k_factor=active_elo_k_factor,
                    initial_rating=active_elo_initial_rating,
                    rng=random.Random(active_seed * 1000003 + generation + 1),
                )

            elites = [item.candidate for item in generation_evaluations[:active_elite_count]]
            population = _next_population(
                seed_candidate=seed_candidate,
                current_evaluations=generation_evaluations,
                elites=elites,
                search_space=active_target.search_space,
                search_paths=search_paths,
                population_size=active_population_size,
                mutation_rate=active_mutation_rate,
                crossover_rate=active_crossover_rate,
                max_mutations_per_candidate=active_max_mutations,
                tournament_size=active_tournament_size,
                rng=rng,
                path_weights=path_weights,
                generation=generation + 1,
                selection=active_selection,
                elo_rankings=elo_rankings,
            )

        if best is None:
            raise RuntimeError("AgentEvolutionOptimizer did not evaluate any candidates.")

        final_elo_rankings: Optional[List[tuple[CandidateEvaluation, float]]] = None
        if active_selection == "elo" and evaluated:
            final_elo_rankings = _elo_tournament_ranking(
                list(evaluated.values()),
                k_factor=active_elo_k_factor,
                initial_rating=active_elo_initial_rating,
                rng=random.Random(active_seed * 1000003),
            )
            # Final-winner selection uses the Elo order instead of raw
            # best-score order (explicit opt-in mode only).
            best = final_elo_rankings[0][0]

        metadata = {
            "optimizer": "AgentEvolutionOptimizer",
            "strategy": "domain_aware_evolution",
            "target_name": best.candidate.target_name,
            "best_candidate_id": best.candidate.id,
            "search_paths": list(search_paths),
            "population_size": active_population_size,
            "generations": active_generations,
            "elite_count": active_elite_count,
            "mutation_rate": active_mutation_rate,
            "crossover_rate": active_crossover_rate,
            "max_mutations_per_candidate": active_max_mutations,
            "tournament_size": active_tournament_size,
            "selection": active_selection,
            "eval_budget": active_eval_budget,
            "evaluations_used": len(evaluated),
            "seed": active_seed,
            "mutation_path_weights": path_weights,
            "path_weights": path_weights,
            "mutation_library": getattr(active_mutation_library, "name", None)
            if active_mutation_library is not None
            else None,
            "mutation_library_bundles": [
                dump_mutation_bundle(bundle) for bundle in library_bundles
            ],
            "generation_summaries": generation_summaries,
            "evaluated_candidates": len(evaluated),
        }
        if final_elo_rankings is not None:
            metadata["elo_ratings"] = {
                evaluation.candidate.id: rating
                for evaluation, rating in final_elo_rankings
            }
            metadata["elo_k_factor"] = active_elo_k_factor
            metadata["elo_initial_rating"] = active_elo_initial_rating
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
            early_stopped=budget_exhausted,
            stop_reason="eval_budget_exhausted" if budget_exhausted else None,
        )

    def _evaluate(
        self,
        candidate: AgentCandidate,
        evaluator: Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float],
        evaluated: dict[str, CandidateEvaluation],
        history: List[IterationHistory],
        *,
        generation: int,
        role: str,
    ) -> CandidateEvaluation:
        if candidate.id in evaluated:
            return evaluated[candidate.id]
        value = evaluator(candidate)
        evaluation = _normalize_candidate_evaluation(value, candidate)
        evaluation.metadata = {
            **candidate.metadata,
            **evaluation.metadata,
            "optimizer": "AgentEvolutionOptimizer",
            "evolution_generation": generation,
            "evolution_role": role,
        }
        evaluated[candidate.id] = evaluation
        history.append(_history_from_candidate(evaluation))
        logger.info(
            "Evaluated evolution candidate %s score=%.4f generation=%s",
            candidate.id,
            evaluation.score,
            generation,
        )
        return evaluation


def _validate_evolution_params(
    *,
    population_size: int,
    generations: int,
    elite_count: int,
    mutation_rate: float,
    crossover_rate: float,
    max_mutations_per_candidate: int,
    tournament_size: int,
) -> None:
    if population_size < 2:
        raise ValueError("population_size must be at least 2.")
    if generations < 0:
        raise ValueError("generations must be non-negative.")
    if elite_count < 1 or elite_count >= population_size:
        raise ValueError("elite_count must be at least 1 and less than population_size.")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1.")
    if not 0 <= crossover_rate <= 1:
        raise ValueError("crossover_rate must be between 0 and 1.")
    if max_mutations_per_candidate < 1:
        raise ValueError("max_mutations_per_candidate must be at least 1.")
    if tournament_size < 1:
        raise ValueError("tournament_size must be at least 1.")


def _merged_layer_path_bias(
    overrides: Optional[Mapping[str, Sequence[str]]],
) -> dict[str, tuple[str, ...]]:
    merged = {
        layer: tuple(paths)
        for layer, paths in DEFAULT_LAYER_PATH_BIAS.items()
    }
    for layer, paths in dict(overrides or {}).items():
        merged[layer] = tuple(paths)
    return merged


def _mutation_path_weights(
    search_paths: Sequence[str],
    *,
    target: OptimizationTarget,
    diagnoses: Sequence[ComponentDiagnosis],
    layer_path_bias: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for path in search_paths:
        weight = 1.0
        for layer in target.layers:
            for prefix in layer_path_bias.get(layer, ()):
                if path == prefix or path.startswith(f"{prefix}."):
                    weight += 2.0
        for diagnosis in diagnoses:
            if path == diagnosis.component or path.startswith(f"{diagnosis.component}."):
                weight += 3.0 * diagnosis.confidence
            for suggested_path in diagnosis.suggested_paths:
                if path == suggested_path or path.startswith(f"{suggested_path}."):
                    weight += 4.0 * diagnosis.confidence
        weights[path] = round(weight, 4)
    return weights


def _initial_population(
    *,
    seed_candidate: AgentCandidate,
    search_space: Mapping[str, List[Any]],
    search_paths: Sequence[str],
    population_size: int,
    include_seed: bool,
    rng: random.Random,
    path_weights: Mapping[str, float],
    library_bundles: Sequence[AgentMutationBundle],
) -> List[AgentCandidate]:
    population: List[AgentCandidate] = []
    seen: set[str] = set()
    if include_seed:
        _append_candidate(population, seen, seed_candidate)

    for bundle in library_bundles:
        candidate = seed_candidate.with_patch(
            dict(bundle.patch),
            metadata={
                "kind": "evolution_library",
                "evolution_role": "library",
                "mutation_bundle": bundle.name,
                "mutation_framework": bundle.framework,
                "mutation_component": bundle.component,
                "mutation_reason": bundle.reason,
                "mutation_tags": list(bundle.tags),
            },
        )
        _append_candidate(population, seen, candidate)
        if len(population) >= population_size:
            return population

    weighted_paths = sorted(
        search_paths,
        key=lambda path: (path_weights.get(path, 1.0), path),
        reverse=True,
    )
    for path in weighted_paths:
        for value in search_space[path]:
            if seed_candidate.get_path(path) == value:
                continue
            candidate = seed_candidate.with_patch(
                {path: value},
                metadata={
                    "kind": "evolution_initial",
                    "evolution_role": "mutant",
                },
            )
            _append_candidate(population, seen, candidate)
            if len(population) >= population_size:
                return population

    attempts = 0
    while len(population) < population_size and attempts < population_size * 20:
        attempts += 1
        patch = _mutated_patch(
            {},
            seed_candidate=seed_candidate,
            search_space=search_space,
            search_paths=search_paths,
            rng=rng,
            path_weights=path_weights,
            max_mutations=2,
        )
        candidate = seed_candidate.with_patch(
            patch,
            metadata={
                "kind": "evolution_initial",
                "evolution_role": "mutant",
            },
        )
        _append_candidate(population, seen, candidate)
    return population


def _next_population(
    *,
    seed_candidate: AgentCandidate,
    current_evaluations: Sequence[CandidateEvaluation],
    elites: Sequence[AgentCandidate],
    search_space: Mapping[str, List[Any]],
    search_paths: Sequence[str],
    population_size: int,
    mutation_rate: float,
    crossover_rate: float,
    max_mutations_per_candidate: int,
    tournament_size: int,
    rng: random.Random,
    path_weights: Mapping[str, float],
    generation: int,
    selection: str = "tournament",
    elo_rankings: Optional[Sequence[tuple[CandidateEvaluation, float]]] = None,
) -> List[AgentCandidate]:
    population: List[AgentCandidate] = []
    seen: set[str] = set()
    for elite in elites:
        _append_candidate(population, seen, elite)

    use_elo = selection == "elo" and bool(elo_rankings)

    attempts = 0
    while len(population) < population_size and attempts < population_size * 30:
        attempts += 1
        if use_elo:
            parent = _elo_weighted_select(elo_rankings, rng)
        else:
            parent = _tournament_select(current_evaluations, tournament_size, rng)
        patch = dict(parent.candidate.patch)
        role = "mutant"
        parent_ids = [parent.candidate.id]
        if rng.random() < crossover_rate and len(current_evaluations) > 1:
            if use_elo:
                other = _elo_weighted_select(elo_rankings, rng)
            else:
                other = _tournament_select(current_evaluations, tournament_size, rng)
            patch = _crossover_patch(
                patch,
                dict(other.candidate.patch),
                rng,
            )
            parent_ids.append(other.candidate.id)
            role = "crossover"
        if rng.random() < mutation_rate or not patch:
            patch = _mutated_patch(
                patch,
                seed_candidate=seed_candidate,
                search_space=search_space,
                search_paths=search_paths,
                rng=rng,
                path_weights=path_weights,
                max_mutations=max_mutations_per_candidate,
            )
            role = "mutant" if role != "crossover" else "crossover_mutant"
        candidate = seed_candidate.with_patch(
            patch,
            metadata={
                "kind": "evolution_candidate",
                "evolution_role": role,
                "evolution_generation": generation,
                "evolution_parent_ids": parent_ids,
            },
        )
        _append_candidate(population, seen, candidate)

    if len(population) < population_size:
        for path in search_paths:
            for value in search_space[path]:
                if len(population) >= population_size:
                    return population
                patch = {path: value}
                candidate = seed_candidate.with_patch(
                    patch,
                    metadata={
                        "kind": "evolution_backfill",
                        "evolution_role": "backfill",
                        "evolution_generation": generation,
                    },
                )
                _append_candidate(population, seen, candidate)
    return population


def _append_candidate(
    population: List[AgentCandidate],
    seen: set[str],
    candidate: AgentCandidate,
) -> None:
    if candidate.id in seen:
        return
    seen.add(candidate.id)
    population.append(candidate)


def _tournament_select(
    evaluations: Sequence[CandidateEvaluation],
    tournament_size: int,
    rng: random.Random,
) -> CandidateEvaluation:
    sample_size = min(tournament_size, len(evaluations))
    sample = rng.sample(list(evaluations), sample_size)
    return max(
        sample,
        key=lambda item: (
            item.score,
            -len(item.candidate.patch),
            item.candidate.id,
        ),
    )


def _validate_selection_params(
    *,
    selection: str,
    eval_budget: Optional[int],
    elo_k_factor: float,
    elo_initial_rating: float,
) -> None:
    if selection not in {"tournament", "elo"}:
        raise ValueError("selection must be 'tournament' or 'elo'.")
    if eval_budget is not None and eval_budget < 1:
        raise ValueError("eval_budget must be at least 1 when declared.")
    if elo_k_factor <= 0:
        raise ValueError("elo_k_factor must be positive.")
    if elo_initial_rating <= 0:
        raise ValueError("elo_initial_rating must be positive.")


def _elo_tournament_ranking(
    evaluations: Sequence[CandidateEvaluation],
    *,
    k_factor: float,
    initial_rating: float,
    rng: random.Random,
) -> List[tuple[CandidateEvaluation, float]]:
    """Deterministic round-robin Elo over already-evaluated candidates.

    Pairings are seeded-shuffled once; each pair plays one 'match' decided by
    the existing scalar scores (win/draw/loss); ratings update with fixed K.
    Returns (evaluation, rating) sorted by rating desc, candidate.id asc.
    The ranking consumes scores the eval suite already produced — it changes
    selection pressure under a fixed budget, it never adds rollouts and never
    asks any LLM to rank (external-verification rule).
    """

    unique: dict[str, CandidateEvaluation] = {}
    for evaluation in evaluations:
        unique.setdefault(evaluation.candidate.id, evaluation)
    entries = sorted(unique.values(), key=lambda item: item.candidate.id)
    ratings = {entry.candidate.id: float(initial_rating) for entry in entries}
    pairs = [
        (left_index, right_index)
        for left_index in range(len(entries))
        for right_index in range(left_index + 1, len(entries))
    ]
    rng.shuffle(pairs)
    for left_index, right_index in pairs:
        left = entries[left_index]
        right = entries[right_index]
        left_rating = ratings[left.candidate.id]
        right_rating = ratings[right.candidate.id]
        expected_left = 1.0 / (1.0 + 10 ** ((right_rating - left_rating) / 400.0))
        if left.score > right.score:
            actual_left = 1.0
        elif left.score < right.score:
            actual_left = 0.0
        else:
            actual_left = 0.5
        delta = k_factor * (actual_left - expected_left)
        ratings[left.candidate.id] = left_rating + delta
        ratings[right.candidate.id] = right_rating - delta
    ranked = sorted(
        entries,
        key=lambda item: (-ratings[item.candidate.id], item.candidate.id),
    )
    return [(entry, round(ratings[entry.candidate.id], 4)) for entry in ranked]


def _elo_weighted_select(
    rankings: Sequence[tuple[CandidateEvaluation, float]],
    rng: random.Random,
) -> CandidateEvaluation:
    total = sum(max(1.0, rating) for _, rating in rankings)
    threshold = rng.random() * total if total > 0 else 0.0
    running = 0.0
    for evaluation, rating in rankings:
        running += max(1.0, rating)
        if running >= threshold:
            return evaluation
    return rankings[-1][0]


def _crossover_patch(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for path in sorted(set(left) | set(right)):
        if path in left and path in right:
            patch[path] = left[path] if rng.random() < 0.5 else right[path]
        elif path in left:
            patch[path] = left[path]
        else:
            patch[path] = right[path]
    return patch


def _mutated_patch(
    patch: Mapping[str, Any],
    *,
    seed_candidate: AgentCandidate,
    search_space: Mapping[str, List[Any]],
    search_paths: Sequence[str],
    rng: random.Random,
    path_weights: Mapping[str, float],
    max_mutations: int,
) -> dict[str, Any]:
    mutated = dict(patch)
    mutation_count = rng.randint(1, min(max_mutations, len(search_paths)))
    for path in _weighted_sample_paths(search_paths, path_weights, mutation_count, rng):
        values = [
            value
            for value in search_space[path]
            if value != mutated.get(path, seed_candidate.get_path(path))
        ]
        if not values:
            continue
        value = rng.choice(values)
        if value == seed_candidate.get_path(path):
            mutated.pop(path, None)
        else:
            mutated[path] = value
    return mutated


def _weighted_sample_paths(
    search_paths: Sequence[str],
    weights: Mapping[str, float],
    count: int,
    rng: random.Random,
) -> List[str]:
    remaining = list(search_paths)
    selected: List[str] = []
    for _ in range(min(count, len(remaining))):
        total = sum(max(0.0, weights.get(path, 1.0)) for path in remaining)
        threshold = rng.random() * total if total > 0 else 0.0
        running = 0.0
        chosen = remaining[-1]
        for path in remaining:
            running += max(0.0, weights.get(path, 1.0))
            if running >= threshold:
                chosen = path
                break
        selected.append(chosen)
        remaining.remove(chosen)
    return selected
