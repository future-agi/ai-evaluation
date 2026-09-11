from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from ..base.base_optimizer import BaseOptimizer
from ..components import (
    ComponentDiagnosis,
    diagnose_agent_report_evaluation,
    diagnose_text,
    relevant_search_paths,
)
from ..deployment import (
    AgentDeploymentExport,
    AgentPromotionCheck,
    AgentRollbackDecision,
    check_agent_deployment_rollback,
    export_agent_deployment,
)
from ..targets import AgentCandidate, CandidateEvaluation, OptimizationTarget
from ..types import EvaluationResult, OptimizationResult
from .agent import AgentOptimizer, _dedupe_diagnoses, _normalize_diagnoses
from .agent_bandit import AgentBanditOptimizer
from .agent_curriculum import AgentCurriculumOptimizer
from .agent_evolution import AgentEvolutionOptimizer
from .agent_pareto import AgentParetoOptimizer
from .agent_social_memory import AgentSocialMemoryOptimizer
from .agent_tpe import AgentTPEOptimizer
from .council import CouncilAgentOptimizer, SocietyAgentOptimizer


FEEDBACK_SCHEMA_VERSION = "agent-opt.feedback.v1"
MULTI_INTERACTION_SCHEMA_VERSION = "agent-opt.multi-interaction.v1"
DEFAULT_MULTI_INTERACTION_BACKENDS = (
    "curriculum",
    "council",
    "society",
    "social_memory",
    "evolution",
    "pareto",
    "tpe",
    "bandit",
    "agent",
)
MULTI_INTERACTION_BACKEND_PROFILES: dict[str, dict[str, Any]] = {
    "society": {
        "allocation_kind": "role_graph_society_search",
        "roles": (
            "sutradhara",
            "smriti",
            "arjuna",
            "hanuman",
            "vidura",
            "krishna",
            "sangha",
            "dharma_steward",
        ),
        "role_archetypes": (
            "orchestrator",
            "working_memory",
            "focused_action",
            "bridge_builder",
            "prudent_critic",
            "charioteer_counsel",
            "collective_synthesis",
            "minimal_process_guardian",
        ),
        "path_prefixes": (
            "multi_agent",
            "memory",
            "policy",
            "security",
            "orchestration",
            "framework",
        ),
        "role_path_prefixes": {
            "sutradhara": ("multi_agent", "orchestration", "framework"),
            "smriti": (
                "memory",
                "framework.memory",
                "framework.checkpoints",
                "framework.sessions",
            ),
            "arjuna": ("tools", "action", "policy"),
            "hanuman": ("multi_agent", "framework", "orchestration"),
            "vidura": ("policy", "security", "adversarial"),
            "krishna": ("multi_agent", "memory", "policy"),
            "sangha": (),
            "dharma_steward": ("policy", "security", "reliability"),
        },
    },
    "council": {
        "allocation_kind": "council_deliberation",
        "roles": ("explorer", "critic", "synthesizer", "steward"),
        "role_archetypes": (
            "exploration",
            "critique",
            "synthesis",
            "process_guardian",
        ),
        "path_prefixes": ("multi_agent", "memory", "policy", "tools", "framework"),
        "role_path_prefixes": {
            "explorer": (),
            "critic": ("policy", "security", "adversarial"),
            "synthesizer": (),
            "steward": ("policy", "reliability", "framework"),
        },
    },
    "social_memory": {
        "allocation_kind": "social_memory_credit_ledger",
        "roles": ("smriti", "arjuna", "vidura", "sangha", "dharma_steward"),
        "role_archetypes": (
            "working_memory",
            "focused_action",
            "prudent_critic",
            "collective_synthesis",
            "minimal_process_guardian",
        ),
        "path_prefixes": ("memory", "multi_agent", "policy", "framework"),
        "role_path_prefixes": {
            "smriti": (
                "memory",
                "framework.memory",
                "framework.checkpoints",
                "framework.sessions",
            ),
            "arjuna": ("multi_agent", "tools", "action"),
            "vidura": ("policy", "security", "adversarial"),
            "sangha": (),
            "dharma_steward": ("policy", "reliability", "framework"),
        },
    },
    "curriculum": {
        "allocation_kind": "deliberate_practice_curriculum",
        "roles": ("teacher", "student", "coach"),
        "role_archetypes": (
            "staged_practice",
            "metric_drill",
            "remediation_coach",
        ),
        "path_prefixes": ("objective", "planner", "memory", "policy", "framework"),
        "role_path_prefixes": {
            "teacher": ("objective", "evaluation", "framework"),
            "student": (),
            "coach": ("memory", "policy", "planner"),
        },
    },
    "evolution": {
        "allocation_kind": "evolutionary_exploration",
        "roles": ("population_explorer", "mutation_stressor", "fitness_selector"),
        "role_archetypes": ("variation", "stress", "selection"),
        "path_prefixes": (),
        "role_path_prefixes": {
            "population_explorer": (),
            "mutation_stressor": ("security", "policy", "tools", "framework"),
            "fitness_selector": (),
        },
    },
    "pareto": {
        "allocation_kind": "pareto_tradeoff_search",
        "roles": ("tradeoff_arbiter", "frontier_keeper"),
        "role_archetypes": ("multi_objective_balance", "frontier_selection"),
        "path_prefixes": (),
        "role_path_prefixes": {
            "tradeoff_arbiter": (),
            "frontier_keeper": (),
        },
    },
    "tpe": {
        "allocation_kind": "tpe_prior_sampling",
        "roles": ("prior_sampler", "density_estimator"),
        "role_archetypes": ("probabilistic_prior", "expected_improvement"),
        "path_prefixes": (),
        "role_path_prefixes": {
            "prior_sampler": (),
            "density_estimator": (),
        },
    },
    "bandit": {
        "allocation_kind": "bandit_budget_allocation",
        "roles": ("allocation_arbiter", "exploit_explore_allocator"),
        "role_archetypes": ("budget_allocator", "adaptive_selection"),
        "path_prefixes": (),
        "role_path_prefixes": {
            "allocation_arbiter": (),
            "exploit_explore_allocator": (),
        },
    },
    "agent": {
        "allocation_kind": "deterministic_candidate_search",
        "roles": ("deterministic_engineer",),
        "role_archetypes": ("metric_patch_search",),
        "path_prefixes": (),
        "role_path_prefixes": {"deterministic_engineer": ()},
    },
}
DeploymentLike = (
    AgentPromotionCheck
    | AgentDeploymentExport
    | OptimizationResult
    | AgentCandidate
    | Mapping[str, Any]
)
CandidateScorer = Callable[
    [AgentCandidate],
    CandidateEvaluation | EvaluationResult | float,
]


class AgentFeedbackCase(BaseModel):
    """One production or replayed feedback observation used for re-optimization."""

    index: int
    source: str = "rollback_observation"
    candidate_id: Optional[str] = None
    score: float
    passed: bool
    failures: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentFeedbackOptimizationResult(BaseModel):
    """Audit record for a live-feedback-triggered optimization round."""

    schema_version: str = FEEDBACK_SCHEMA_VERSION
    optimizer: str
    feedback_source: str
    rollback_decision: AgentRollbackDecision
    feedback_cases: list[AgentFeedbackCase] = Field(default_factory=list)
    diagnoses: list[ComponentDiagnosis] = Field(default_factory=list)
    search_paths: list[str] = Field(default_factory=list)
    reoptimization_result: OptimizationResult
    baseline_score: Optional[float] = None
    feedback_score: Optional[float] = None
    final_score: float
    baseline_delta: Optional[float] = None
    feedback_delta: Optional[float] = None
    improved: bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


class AgentMultiInteractionBackendPlan(BaseModel):
    """One deterministic backend allocation in a multi-interaction round."""

    optimizer: str
    rank: int
    weight: float
    reason: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AgentMultiInteractionBackendRun(BaseModel):
    """Result from running one allocated optimizer backend."""

    optimizer: str
    rank: int
    status: str
    final_score: Optional[float] = None
    improved: bool = False
    total_evaluations: int = 0
    failure: Optional[str] = None
    result: Optional[AgentFeedbackOptimizationResult] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMultiInteractionBackendLineage(BaseModel):
    """Candidate contribution summary for one backend in a portfolio run."""

    optimizer: str
    rank: int
    allocation_weight: float = 0.0
    allocation_reason: str = ""
    status: str
    final_score: Optional[float] = None
    improved: bool = False
    total_evaluations: int = 0
    candidate_id: Optional[str] = None
    parent_candidate_id: Optional[str] = None
    candidate_patch: dict[str, Any] = Field(default_factory=dict)
    patch_paths: list[str] = Field(default_factory=list)
    unique_candidate_patch: dict[str, Any] = Field(default_factory=dict)
    unique_patch_paths: list[str] = Field(default_factory=list)
    shared_candidate_patch: dict[str, Any] = Field(default_factory=dict)
    shared_patch_paths: list[str] = Field(default_factory=list)
    equivalent_backends: list[str] = Field(default_factory=list)
    equivalent_backend_count: int = 0
    selection_relation: str = "unclassified"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMultiInteractionAblationReport(BaseModel):
    """Leave-one-backend-out summary for the selected portfolio result."""

    selected_optimizer: str
    selected_candidate_id: Optional[str] = None
    selected_patch: dict[str, Any] = Field(default_factory=dict)
    selected_patch_paths: list[str] = Field(default_factory=list)
    final_score: float
    best_without_selected_optimizer: Optional[str] = None
    best_without_selected_score: Optional[float] = None
    score_delta_without_selected: Optional[float] = None
    selected_backend_required: bool
    dependency: str
    dependency_reason: str
    consensus_backends: list[str] = Field(default_factory=list)
    consensus_backend_count: int = 0
    shared_selected_patch_paths: list[str] = Field(default_factory=list)
    unique_selected_patch_paths: list[str] = Field(default_factory=list)
    selected_patch_support: dict[str, list[str]] = Field(default_factory=dict)
    backend_scoreboard: list[dict[str, Any]] = Field(default_factory=list)


class AgentMultiInteractionOptimizationResult(BaseModel):
    """Audit record for automatic multi-backend agent re-optimization."""

    schema_version: str = MULTI_INTERACTION_SCHEMA_VERSION
    selected_optimizer: str
    feedback_source: str
    rollback_decision: AgentRollbackDecision
    feedback_cases: list[AgentFeedbackCase] = Field(default_factory=list)
    diagnoses: list[ComponentDiagnosis] = Field(default_factory=list)
    search_paths: list[str] = Field(default_factory=list)
    backend_plan: list[AgentMultiInteractionBackendPlan] = Field(default_factory=list)
    backend_runs: list[AgentMultiInteractionBackendRun] = Field(default_factory=list)
    backend_lineage: list[AgentMultiInteractionBackendLineage] = Field(default_factory=list)
    ablation_report: AgentMultiInteractionAblationReport
    best_result: AgentFeedbackOptimizationResult
    final_score: float
    improved: bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


class AgentFeedbackOptimizer(BaseOptimizer):
    """
    Re-optimize an agent from live trace/evaluation feedback.

    The optimizer first turns post-deployment rollback evidence into component
    diagnoses and search paths, then delegates the actual search to one of the
    existing agent optimizers (`society`, `social_memory`, `curriculum`,
    `council`, `evolution`, `tpe`, `pareto`, `bandit`, or deterministic
    `agent`).
    """

    def __init__(
        self,
        target: Optional[OptimizationTarget] = None,
        *,
        deployment: Optional[DeploymentLike] = None,
        rollback_decision: Optional[AgentRollbackDecision] = None,
        live_evaluations: Optional[Sequence[Any]] = None,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        optimizer: str = "society",
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        diagnostic_score_threshold: float = 0.85,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        rollback_kwargs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.target = target
        self.deployment = deployment
        self.rollback_decision = rollback_decision
        self.live_evaluations = (
            list(live_evaluations) if live_evaluations is not None else None
        )
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.optimizer = optimizer
        self.diagnoses = _normalize_diagnoses(diagnoses)
        self.diagnostic_score_threshold = diagnostic_score_threshold
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.rollback_kwargs = dict(rollback_kwargs or {})
        self.metadata = dict(metadata or {})
        super().__init__()

    def optimize(
        self,
        evaluator: Any = None,
        data_mapper: Any = None,
        dataset: Optional[List[dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        *,
        target: Optional[OptimizationTarget] = None,
        deployment: Optional[DeploymentLike] = None,
        rollback_decision: Optional[AgentRollbackDecision] = None,
        live_evaluations: Optional[Sequence[Any]] = None,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        optimizer: Optional[str] = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        diagnostic_score_threshold: Optional[float] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        rollback_kwargs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **backend_kwargs: Any,
    ) -> AgentFeedbackOptimizationResult:
        active_target = target or self.target
        if active_target is None:
            raise ValueError("AgentFeedbackOptimizer requires a target.")

        active_evaluator = evaluate_candidate or self.evaluate_candidate
        active_simulation = simulation_evaluator or self.simulation_evaluator
        if (
            active_evaluator is None
            and getattr(active_simulation, "evaluate_candidate", None) is None
        ):
            raise ValueError(
                "AgentFeedbackOptimizer requires evaluate_candidate or simulation_evaluator."
            )

        active_diagnostic_threshold = (
            self.diagnostic_score_threshold
            if diagnostic_score_threshold is None
            else diagnostic_score_threshold
        )
        explicit_diagnoses = _normalize_diagnoses(diagnoses)
        if diagnoses is None:
            explicit_diagnoses = list(self.diagnoses)

        active_rollback_decision = rollback_decision or self.rollback_decision
        active_live_evaluations = (
            list(live_evaluations)
            if live_evaluations is not None
            else self.live_evaluations
        )
        active_deployment = deployment or self.deployment
        active_deployment, auto_seed_deployment = _auto_seed_deployment_for_replay(
            target=active_target,
            deployment=active_deployment,
            rollback_decision=active_rollback_decision,
            live_evaluations=active_live_evaluations,
            simulation_evaluator=active_simulation,
            metadata={**self.metadata, **dict(metadata or {})},
        )
        decision, feedback_source = _resolve_rollback_decision(
            rollback_decision=active_rollback_decision,
            deployment=active_deployment,
            live_evaluations=active_live_evaluations,
            simulation_evaluator=active_simulation,
            rollback_kwargs={
                **self.rollback_kwargs,
                **dict(rollback_kwargs or {}),
            },
        )
        feedback_cases = _feedback_cases_from_rollback(decision)
        feedback_diagnoses = _diagnose_feedback_cases(
            feedback_cases,
            target=active_target,
            failing_threshold=active_diagnostic_threshold,
        )
        active_diagnoses = _dedupe_diagnoses([*explicit_diagnoses, *feedback_diagnoses])
        search_paths = _search_paths_for_feedback(active_target, active_diagnoses)

        backend_name = optimizer or self.optimizer
        resolved_optimizer = _resolve_feedback_optimizer(backend_name)
        combined_backend_kwargs = {
            **self.optimizer_kwargs,
            **dict(optimizer_kwargs or {}),
            **backend_kwargs,
        }
        backend = resolved_optimizer(
            target=active_target,
            evaluate_candidate=active_evaluator,
            simulation_evaluator=active_simulation,
            diagnoses=active_diagnoses,
            diagnostic_score_threshold=active_diagnostic_threshold,
            **combined_backend_kwargs,
        )
        reoptimization = backend.optimize()
        baseline_score = decision.baseline_score
        feedback_score = decision.latest_score
        baseline_delta = (
            reoptimization.final_score - baseline_score
            if baseline_score is not None
            else None
        )
        feedback_delta = (
            reoptimization.final_score - feedback_score
            if feedback_score is not None
            else None
        )
        improved = (
            reoptimization.final_score >= decision.min_score
            and (feedback_delta is None or feedback_delta > 0)
        )
        result_metadata = {
            **self.metadata,
            **dict(metadata or {}),
            "rollback_required": decision.rollback_required,
            "failure_count": decision.failure_count,
            "consecutive_failure_count": decision.consecutive_failure_count,
            "auto_seed_deployment": auto_seed_deployment,
            "backend_optimizer": reoptimization.metadata.get("optimizer"),
        }
        return AgentFeedbackOptimizationResult(
            optimizer=_normalize_optimizer_name(backend_name),
            feedback_source=feedback_source,
            rollback_decision=decision,
            feedback_cases=feedback_cases,
            diagnoses=active_diagnoses,
            search_paths=search_paths,
            reoptimization_result=reoptimization,
            baseline_score=baseline_score,
            feedback_score=feedback_score,
            final_score=reoptimization.final_score,
            baseline_delta=baseline_delta,
            feedback_delta=feedback_delta,
            improved=improved,
            metadata=result_metadata,
        )


class AgentMultiInteractionOptimizer(BaseOptimizer):
    """
    Diagnose feedback, allocate deterministic optimizer backends, and select the best.

    This is the Future AGI-native portfolio layer above `AgentFeedbackOptimizer`:
    every backend receives the same rollback/replay evidence and metric-derived
    diagnoses, while the allocator chooses backend priority from feedback
    metrics, target layers, and search-space shape. Social/psychological
    inspiration stays metadata-only; candidate acceptance is numeric.
    """

    def __init__(
        self,
        target: Optional[OptimizationTarget] = None,
        *,
        deployment: Optional[DeploymentLike] = None,
        rollback_decision: Optional[AgentRollbackDecision] = None,
        live_evaluations: Optional[Sequence[Any]] = None,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        optimizer_pool: Optional[Sequence[str]] = None,
        max_backends: Optional[int] = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        diagnostic_score_threshold: float = 0.85,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        optimizer_kwargs_by_backend: Optional[Mapping[str, Mapping[str, Any]]] = None,
        rollback_kwargs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.target = target
        self.deployment = deployment
        self.rollback_decision = rollback_decision
        self.live_evaluations = (
            list(live_evaluations) if live_evaluations is not None else None
        )
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.optimizer_pool = list(optimizer_pool) if optimizer_pool is not None else None
        self.max_backends = max_backends
        self.diagnoses = _normalize_diagnoses(diagnoses)
        self.diagnostic_score_threshold = diagnostic_score_threshold
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.optimizer_kwargs_by_backend = {
            _normalize_optimizer_name(key): dict(value)
            for key, value in dict(optimizer_kwargs_by_backend or {}).items()
        }
        self.rollback_kwargs = dict(rollback_kwargs or {})
        self.metadata = dict(metadata or {})
        super().__init__()

    def optimize(
        self,
        evaluator: Any = None,
        data_mapper: Any = None,
        dataset: Optional[List[dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        *,
        target: Optional[OptimizationTarget] = None,
        deployment: Optional[DeploymentLike] = None,
        rollback_decision: Optional[AgentRollbackDecision] = None,
        live_evaluations: Optional[Sequence[Any]] = None,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        optimizer_pool: Optional[Sequence[str]] = None,
        max_backends: Optional[int] = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        diagnostic_score_threshold: Optional[float] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        optimizer_kwargs_by_backend: Optional[Mapping[str, Mapping[str, Any]]] = None,
        rollback_kwargs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **backend_kwargs: Any,
    ) -> AgentMultiInteractionOptimizationResult:
        active_target = target or self.target
        if active_target is None:
            raise ValueError("AgentMultiInteractionOptimizer requires a target.")

        active_evaluator = evaluate_candidate or self.evaluate_candidate or evaluator
        active_simulation = simulation_evaluator or self.simulation_evaluator
        if (
            active_evaluator is None
            and getattr(active_simulation, "evaluate_candidate", None) is None
        ):
            raise ValueError(
                "AgentMultiInteractionOptimizer requires evaluate_candidate or simulation_evaluator."
            )

        active_diagnostic_threshold = (
            self.diagnostic_score_threshold
            if diagnostic_score_threshold is None
            else diagnostic_score_threshold
        )
        explicit_diagnoses = _normalize_diagnoses(diagnoses)
        if diagnoses is None:
            explicit_diagnoses = list(self.diagnoses)

        active_rollback_decision = rollback_decision or self.rollback_decision
        active_live_evaluations = (
            list(live_evaluations)
            if live_evaluations is not None
            else self.live_evaluations
        )
        active_deployment = deployment or self.deployment
        active_deployment, auto_seed_deployment = _auto_seed_deployment_for_replay(
            target=active_target,
            deployment=active_deployment,
            rollback_decision=active_rollback_decision,
            live_evaluations=active_live_evaluations,
            simulation_evaluator=active_simulation,
            metadata={**self.metadata, **dict(metadata or {})},
        )
        decision, feedback_source = _resolve_rollback_decision(
            rollback_decision=active_rollback_decision,
            deployment=active_deployment,
            live_evaluations=active_live_evaluations,
            simulation_evaluator=active_simulation,
            rollback_kwargs={
                **self.rollback_kwargs,
                **dict(rollback_kwargs or {}),
            },
        )
        feedback_cases = _feedback_cases_from_rollback(decision)
        feedback_diagnoses = _diagnose_feedback_cases(
            feedback_cases,
            target=active_target,
            failing_threshold=active_diagnostic_threshold,
        )
        active_diagnoses = _dedupe_diagnoses([*explicit_diagnoses, *feedback_diagnoses])
        search_paths = _search_paths_for_feedback(active_target, active_diagnoses)

        base_optimizer_kwargs = {
            **self.optimizer_kwargs,
            **dict(optimizer_kwargs or {}),
            **backend_kwargs,
        }
        per_backend_kwargs = dict(self.optimizer_kwargs_by_backend)
        for key, value in dict(optimizer_kwargs_by_backend or {}).items():
            per_backend_kwargs[_normalize_optimizer_name(key)] = dict(value)

        plan = _multi_interaction_backend_plan(
            target=active_target,
            feedback_cases=feedback_cases,
            diagnoses=active_diagnoses,
            search_paths=search_paths,
            optimizer_pool=optimizer_pool or self.optimizer_pool,
            max_backends=self.max_backends if max_backends is None else max_backends,
            optimizer_kwargs=base_optimizer_kwargs,
            optimizer_kwargs_by_backend=per_backend_kwargs,
        )
        if not plan:
            raise ValueError("AgentMultiInteractionOptimizer backend plan cannot be empty.")

        runs: list[AgentMultiInteractionBackendRun] = []
        for allocation in plan:
            try:
                result = AgentFeedbackOptimizer(
                    target=active_target,
                    rollback_decision=decision,
                    evaluate_candidate=active_evaluator,
                    simulation_evaluator=active_simulation,
                    optimizer=allocation.optimizer,
                    diagnoses=active_diagnoses,
                    diagnostic_score_threshold=active_diagnostic_threshold,
                    optimizer_kwargs=allocation.kwargs,
                    metadata={
                        "multi_interaction_optimizer": True,
                        "backend_rank": allocation.rank,
                        "backend_weight": allocation.weight,
                        "backend_reason": allocation.reason,
                    },
                ).optimize()
                runs.append(
                    AgentMultiInteractionBackendRun(
                        optimizer=allocation.optimizer,
                        rank=allocation.rank,
                        status="completed",
                        final_score=result.final_score,
                        improved=result.improved,
                        total_evaluations=result.reoptimization_result.total_evaluations,
                        result=result,
                        metadata={
                            "backend_optimizer": result.metadata.get("backend_optimizer"),
                        },
                    )
                )
            except Exception as exc:
                runs.append(
                    AgentMultiInteractionBackendRun(
                        optimizer=allocation.optimizer,
                        rank=allocation.rank,
                        status="failed",
                        failure=str(exc),
                    )
                )

        successful_runs = [run for run in runs if run.result is not None]
        if not successful_runs:
            failures = "; ".join(
                f"{run.optimizer}: {run.failure}" for run in runs if run.failure
            )
            raise RuntimeError(
                "AgentMultiInteractionOptimizer did not complete any backend"
                + (f": {failures}" if failures else ".")
            )
        best_run = max(
            successful_runs,
            key=lambda run: (
                run.final_score if run.final_score is not None else float("-inf"),
                1 if run.improved else 0,
                -run.rank,
                -run.total_evaluations,
            ),
        )
        assert best_run.result is not None
        backend_lineage = _multi_interaction_backend_lineage(
            target=active_target,
            plan=plan,
            runs=runs,
            selected_run=best_run,
        )
        ablation_report = _multi_interaction_ablation_report(
            lineage=backend_lineage,
            selected_run=best_run,
        )
        allocation_metadata = _multi_interaction_allocation_metadata(
            target=active_target,
            plan=plan,
            feedback_cases=feedback_cases,
            diagnoses=active_diagnoses,
            search_paths=search_paths,
        )
        result_metadata = {
            **self.metadata,
            **dict(metadata or {}),
            "allocator": "metric_diagnosis_backend_portfolio",
            **allocation_metadata,
            "auto_seed_deployment": auto_seed_deployment,
            "backend_count": len(plan),
            "completed_backend_count": len(successful_runs),
            "failed_backend_count": len(runs) - len(successful_runs),
            "optimizer_pool": [allocation.optimizer for allocation in plan],
            "selection_rule": "highest_final_score_then_improved_then_rank",
            "ablation_dependency": ablation_report.dependency,
            "selected_backend_required": ablation_report.selected_backend_required,
            "consensus_backend_count": ablation_report.consensus_backend_count,
            "selected_patch_paths": list(ablation_report.selected_patch_paths),
            "strategy_inspiration": (
                "diagnostic triage, deliberate practice, council synthesis, "
                "social memory, evolutionary exploration, Pareto tradeoff, "
                "TPE sampling, bandit allocation, human team roles, and "
                "Hindu-mythology-inspired society labels; labels are metadata only"
            ),
        }
        return AgentMultiInteractionOptimizationResult(
            selected_optimizer=best_run.optimizer,
            feedback_source=feedback_source,
            rollback_decision=decision,
            feedback_cases=feedback_cases,
            diagnoses=active_diagnoses,
            search_paths=search_paths,
            backend_plan=plan,
            backend_runs=runs,
            backend_lineage=backend_lineage,
            ablation_report=ablation_report,
            best_result=best_run.result,
            final_score=best_run.result.final_score,
            improved=best_run.result.improved,
            metadata=result_metadata,
        )


def _multi_interaction_backend_plan(
    *,
    target: OptimizationTarget,
    feedback_cases: Sequence[AgentFeedbackCase],
    diagnoses: Sequence[ComponentDiagnosis],
    search_paths: Sequence[str],
    optimizer_pool: Optional[Sequence[str]],
    max_backends: Optional[int],
    optimizer_kwargs: Mapping[str, Any],
    optimizer_kwargs_by_backend: Mapping[str, Mapping[str, Any]],
) -> list[AgentMultiInteractionBackendPlan]:
    if max_backends is not None and max_backends < 1:
        raise ValueError("max_backends must be at least 1.")

    metric_names = _failed_feedback_metric_names(feedback_cases) or _feedback_metric_names(
        feedback_cases
    )
    normalized_pool = _dedupe_optimizer_pool(optimizer_pool or DEFAULT_MULTI_INTERACTION_BACKENDS)
    scored: list[tuple[float, int, str, str, dict[str, Any]]] = []
    default_order = {
        name: index for index, name in enumerate(DEFAULT_MULTI_INTERACTION_BACKENDS)
    }
    for optimizer_name in normalized_pool:
        _resolve_feedback_optimizer(optimizer_name)
        backend_kwargs = _backend_kwargs_for_multi_interaction(
            optimizer_name,
            target=target,
            metric_names=metric_names,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_kwargs_by_backend=optimizer_kwargs_by_backend,
        )
        if optimizer_name == "pareto" and not backend_kwargs.get("objective_names"):
            continue
        weight, reason = _backend_allocation_weight(
            optimizer_name,
            target=target,
            feedback_cases=feedback_cases,
            diagnoses=diagnoses,
            search_paths=search_paths,
            metric_names=metric_names,
        )
        scored.append(
            (
                weight,
                -default_order.get(optimizer_name, len(DEFAULT_MULTI_INTERACTION_BACKENDS)),
                optimizer_name,
                reason,
                backend_kwargs,
            )
        )

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    if max_backends is not None:
        scored = scored[:max_backends]
    return [
        AgentMultiInteractionBackendPlan(
            optimizer=optimizer_name,
            rank=index,
            weight=round(weight, 4),
            reason=reason,
            kwargs=backend_kwargs,
        )
        for index, (weight, _, optimizer_name, reason, backend_kwargs) in enumerate(
            scored,
            start=1,
        )
    ]


def _auto_seed_deployment_for_replay(
    *,
    target: OptimizationTarget,
    deployment: Optional[DeploymentLike],
    rollback_decision: Optional[AgentRollbackDecision],
    live_evaluations: Optional[Sequence[Any]],
    simulation_evaluator: Any,
    metadata: Mapping[str, Any],
) -> tuple[Optional[DeploymentLike], bool]:
    if deployment is not None or rollback_decision is not None:
        return deployment, False
    if live_evaluations is not None:
        return deployment, False
    if getattr(simulation_evaluator, "evaluate_candidate", None) is None:
        return deployment, False

    seed = target.seed_candidate()
    return (
        export_agent_deployment(
            seed,
            framework="auto",
            metadata={
                **dict(metadata),
                "auto_seed_deployment": True,
                "auto_seed_deployment_source": "simulation_replay",
            },
        ),
        True,
    )


def _multi_interaction_allocation_metadata(
    *,
    target: OptimizationTarget,
    plan: Sequence[AgentMultiInteractionBackendPlan],
    feedback_cases: Sequence[AgentFeedbackCase],
    diagnoses: Sequence[ComponentDiagnosis],
    search_paths: Sequence[str],
) -> dict[str, Any]:
    metric_coverage = _diagnostic_metric_coverage(
        diagnoses,
        metric_names=_failed_feedback_metric_names(feedback_cases)
        or _feedback_metric_names(feedback_cases),
    )
    active_paths = list(search_paths or target.search_space)
    ledger: list[dict[str, Any]] = []
    role_coverage: dict[str, int] = {}
    archetype_coverage: dict[str, int] = {}

    for allocation in plan:
        profile = _multi_interaction_backend_profile(allocation.optimizer)
        path_focus = _allocation_profile_path_focus(profile, active_paths)
        role_path_focus = _allocation_role_path_focus(profile, path_focus)
        diagnosis_focus = _allocation_diagnosis_focus(
            profile=profile,
            diagnoses=diagnoses,
            active_paths=active_paths,
            path_focus=path_focus,
        )
        for role in profile["roles"]:
            role_coverage[role] = role_coverage.get(role, 0) + 1
        for archetype in profile["role_archetypes"]:
            archetype_coverage[archetype] = archetype_coverage.get(archetype, 0) + 1

        focused_metrics = _diagnosis_focus_metric_coverage(diagnosis_focus)
        ledger.append(
            {
                "optimizer": allocation.optimizer,
                "rank": allocation.rank,
                "weight": allocation.weight,
                "reason": allocation.reason,
                "allocation_kind": profile["allocation_kind"],
                "roles": list(profile["roles"]),
                "role_archetypes": list(profile["role_archetypes"]),
                "path_focus": path_focus,
                "role_path_focus": role_path_focus,
                "diagnostic_components": _diagnosis_focus_values(
                    diagnosis_focus,
                    "component",
                ),
                "diagnostic_failure_modes": _diagnosis_focus_values(
                    diagnosis_focus,
                    "failure_mode",
                ),
                "diagnostic_metrics": focused_metrics or metric_coverage,
                "diagnosis_focus": diagnosis_focus,
            }
        )

    path_coverage = _ordered_patch_paths_for_keys(
        _flatten_ledger_path_focus(ledger),
        list(target.search_space),
    )
    return {
        "allocation_algorithm": "deterministic_metric_diagnosis_society_agent_anchor_allocator",
        "allocation_inspiration": (
            "Human-team and society-role labels guide audit metadata only; "
            "candidate acceptance remains metric-based."
        ),
        "deterministic_agent_anchor": any(
            allocation.optimizer == "agent"
            and "focused deterministic diagnosis search" in allocation.reason
            for allocation in plan
        ),
        "society_allocation_ledger": ledger,
        "allocation_role_coverage": dict(sorted(role_coverage.items())),
        "allocation_archetype_coverage": dict(sorted(archetype_coverage.items())),
        "allocation_metric_coverage": metric_coverage,
        "allocation_search_path_coverage": path_coverage,
        "allocation_diagnosis_coverage": _diagnosis_coverage_keys(diagnoses),
    }


def _multi_interaction_backend_profile(optimizer_name: str) -> dict[str, Any]:
    profile = MULTI_INTERACTION_BACKEND_PROFILES.get(optimizer_name)
    if profile is not None:
        return profile
    return {
        "allocation_kind": "custom_backend_search",
        "roles": (optimizer_name,),
        "role_archetypes": ("custom_optimizer",),
        "path_prefixes": (),
        "role_path_prefixes": {optimizer_name: ()},
    }


def _allocation_profile_path_focus(
    profile: Mapping[str, Any],
    active_paths: Sequence[str],
) -> list[str]:
    path_focus = _path_prefix_focus(active_paths, profile.get("path_prefixes", ()))
    return path_focus or list(dict.fromkeys(active_paths))


def _allocation_role_path_focus(
    profile: Mapping[str, Any],
    path_focus: Sequence[str],
) -> dict[str, list[str]]:
    role_path_prefixes = dict(profile.get("role_path_prefixes", {}))
    role_focus: dict[str, list[str]] = {}
    for role in profile.get("roles", ()):
        prefixes = role_path_prefixes.get(role, ())
        focused = _path_prefix_focus(path_focus, prefixes)
        role_focus[str(role)] = focused or list(path_focus)
    return role_focus


def _allocation_diagnosis_focus(
    *,
    profile: Mapping[str, Any],
    diagnoses: Sequence[ComponentDiagnosis],
    active_paths: Sequence[str],
    path_focus: Sequence[str],
) -> list[dict[str, Any]]:
    path_focus_set = set(path_focus)
    profile_prefixes = tuple(str(prefix) for prefix in profile.get("path_prefixes", ()))
    rows: list[dict[str, Any]] = []
    for diagnosis in diagnoses:
        diagnosis_paths = _diagnosis_search_path_focus(diagnosis, active_paths)
        if (
            profile_prefixes
            and diagnosis_paths
            and path_focus_set
            and not path_focus_set.intersection(diagnosis_paths)
        ):
            continue
        metrics = _diagnosis_metric_names(diagnosis)
        row: dict[str, Any] = {
            "component": diagnosis.component,
            "failure_mode": diagnosis.failure_mode,
            "confidence": round(float(diagnosis.confidence), 4),
        }
        if metrics:
            row["metrics"] = metrics
        if diagnosis_paths:
            row["paths"] = diagnosis_paths
        if diagnosis.patch_strategy:
            row["patch_strategy"] = diagnosis.patch_strategy
        if diagnosis.evidence:
            row["evidence"] = diagnosis.evidence
        rows.append(row)
    return rows


def _diagnosis_search_path_focus(
    diagnosis: ComponentDiagnosis,
    active_paths: Sequence[str],
) -> list[str]:
    prefixes = [str(path) for path in diagnosis.suggested_paths]
    prefixes.append(str(diagnosis.component))
    return _path_prefix_focus(active_paths, prefixes)


def _path_prefix_focus(
    paths: Sequence[str],
    prefixes: Sequence[Any],
) -> list[str]:
    unique_paths = list(dict.fromkeys(str(path) for path in paths))
    unique_prefixes = [str(prefix) for prefix in prefixes if str(prefix)]
    if not unique_prefixes:
        return unique_paths
    return [
        path
        for path in unique_paths
        if any(
            path == prefix or path.startswith(f"{prefix}.")
            for prefix in unique_prefixes
        )
    ]


def _diagnostic_metric_coverage(
    diagnoses: Sequence[ComponentDiagnosis],
    *,
    metric_names: Sequence[str],
) -> list[str]:
    metrics = {str(metric) for metric in metric_names}
    for diagnosis in diagnoses:
        metrics.update(_diagnosis_metric_names(diagnosis))
    return sorted(metrics)


def _diagnosis_metric_names(diagnosis: ComponentDiagnosis) -> list[str]:
    metrics: set[str] = set()
    metadata = dict(diagnosis.metadata or {})
    for key in ("metric", "metric_name", "name"):
        value = metadata.get(key)
        if value:
            metrics.add(str(value))
    for key in ("metric_result", "finding"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            for nested_key in ("metric", "metric_name", "name"):
                nested_value = value.get(nested_key)
                if nested_value:
                    metrics.add(str(nested_value))
    return sorted(metrics)


def _diagnosis_focus_metric_coverage(
    diagnosis_focus: Sequence[Mapping[str, Any]],
) -> list[str]:
    metrics: set[str] = set()
    for row in diagnosis_focus:
        metrics.update(str(metric) for metric in row.get("metrics", ()))
    return sorted(metrics)


def _diagnosis_focus_values(
    diagnosis_focus: Sequence[Mapping[str, Any]],
    key: str,
) -> list[str]:
    return sorted({str(row[key]) for row in diagnosis_focus if key in row})


def _flatten_ledger_path_focus(ledger: Sequence[Mapping[str, Any]]) -> list[str]:
    paths: list[str] = []
    for entry in ledger:
        paths.extend(str(path) for path in entry.get("path_focus", ()))
        for role_paths in dict(entry.get("role_path_focus", {})).values():
            paths.extend(str(path) for path in role_paths)
    return list(dict.fromkeys(paths))


def _diagnosis_coverage_keys(diagnoses: Sequence[ComponentDiagnosis]) -> list[str]:
    return sorted(
        {
            f"{diagnosis.component}:{diagnosis.failure_mode}"
            for diagnosis in diagnoses
        }
    )


def _multi_interaction_backend_lineage(
    *,
    target: OptimizationTarget,
    plan: Sequence[AgentMultiInteractionBackendPlan],
    runs: Sequence[AgentMultiInteractionBackendRun],
    selected_run: AgentMultiInteractionBackendRun,
) -> list[AgentMultiInteractionBackendLineage]:
    plan_by_optimizer = {allocation.optimizer: allocation for allocation in plan}
    selected_patch_signature = _patch_signature(
        _backend_run_candidate_patch(selected_run, target)
    )
    rows: list[AgentMultiInteractionBackendLineage] = []
    for run in runs:
        allocation = plan_by_optimizer.get(run.optimizer)
        reoptimization_result = (
            run.result.reoptimization_result if run.result is not None else None
        )
        candidate = (
            getattr(reoptimization_result, "best_candidate", None)
            if reoptimization_result is not None
            else None
        )
        candidate_patch = _candidate_contribution_patch(candidate, target)
        rows.append(
            AgentMultiInteractionBackendLineage(
                optimizer=run.optimizer,
                rank=run.rank,
                allocation_weight=allocation.weight if allocation else 0.0,
                allocation_reason=allocation.reason if allocation else "",
                status=run.status,
                final_score=run.final_score,
                improved=run.improved,
                total_evaluations=run.total_evaluations,
                candidate_id=getattr(candidate, "id", None),
                parent_candidate_id=getattr(candidate, "parent_id", None),
                candidate_patch=candidate_patch,
                patch_paths=_ordered_patch_paths(target, candidate_patch),
                metadata={
                    "backend_strategy": (
                        reoptimization_result.metadata.get("strategy")
                        if reoptimization_result is not None
                        else None
                    ),
                    "backend_optimizer": (
                        run.result.metadata.get("backend_optimizer")
                        if run.result is not None
                        else None
                    ),
                },
            )
        )

    completed_rows = [row for row in rows if row.status == "completed"]
    patch_backends: dict[str, list[str]] = {}
    patch_value_backends: dict[tuple[str, str], list[str]] = {}
    for row in completed_rows:
        patch_backends.setdefault(_patch_signature(row.candidate_patch), []).append(
            row.optimizer
        )
        for path, value in row.candidate_patch.items():
            patch_value_backends.setdefault(
                (path, _value_signature(value)),
                [],
            ).append(row.optimizer)

    selected_patch = _backend_run_candidate_patch(selected_run, target)
    for row in rows:
        if row.status != "completed":
            row.selection_relation = "failed"
            continue

        patch_signature = _patch_signature(row.candidate_patch)
        equivalent_backends = patch_backends.get(patch_signature, [])
        unique_patch: dict[str, Any] = {}
        shared_patch: dict[str, Any] = {}
        for path, value in row.candidate_patch.items():
            supporters = patch_value_backends.get((path, _value_signature(value)), [])
            if len(supporters) == 1:
                unique_patch[path] = value
            else:
                shared_patch[path] = value

        row.equivalent_backends = list(equivalent_backends)
        row.equivalent_backend_count = len(equivalent_backends)
        row.unique_candidate_patch = unique_patch
        row.unique_patch_paths = _ordered_patch_paths(target, unique_patch)
        row.shared_candidate_patch = shared_patch
        row.shared_patch_paths = _ordered_patch_paths(target, shared_patch)
        if row.optimizer == selected_run.optimizer:
            row.selection_relation = "selected"
        elif patch_signature == selected_patch_signature:
            row.selection_relation = "consensus_peer"
        elif _patches_share_values(row.candidate_patch, selected_patch):
            row.selection_relation = "partial_support"
        else:
            row.selection_relation = "divergent"

    return rows


def _multi_interaction_ablation_report(
    *,
    lineage: Sequence[AgentMultiInteractionBackendLineage],
    selected_run: AgentMultiInteractionBackendRun,
) -> AgentMultiInteractionAblationReport:
    selected_lineage = next(
        (row for row in lineage if row.optimizer == selected_run.optimizer),
        None,
    )
    selected_patch = selected_lineage.candidate_patch if selected_lineage else {}
    selected_signature = _patch_signature(selected_patch)
    final_score = float(selected_run.final_score or 0.0)
    completed = [row for row in lineage if row.status == "completed"]
    peers = [row for row in completed if row.optimizer != selected_run.optimizer]
    best_without_selected = (
        max(peers, key=_lineage_selection_key) if peers else None
    )
    score_delta_without_selected: Optional[float] = None
    if best_without_selected and best_without_selected.final_score is not None:
        score_delta_without_selected = round(
            final_score - best_without_selected.final_score,
            8,
        )

    score_tolerance = 1e-9
    consensus_backends = [
        row.optimizer
        for row in completed
        if _patch_signature(row.candidate_patch) == selected_signature
        and row.final_score is not None
        and abs(row.final_score - final_score) <= score_tolerance
    ]
    peer_reproduced_selected = any(
        optimizer != selected_run.optimizer for optimizer in consensus_backends
    )
    selected_backend_required = not peer_reproduced_selected

    selected_patch_support: dict[str, list[str]] = {}
    for path, value in selected_patch.items():
        selected_patch_support[path] = [
            row.optimizer
            for row in completed
            if path in row.candidate_patch
            and _value_signature(row.candidate_patch[path]) == _value_signature(value)
        ]
    shared_selected_patch_paths = [
        path for path, supporters in selected_patch_support.items() if len(supporters) > 1
    ]
    unique_selected_patch_paths = [
        path for path, supporters in selected_patch_support.items() if len(supporters) == 1
    ]

    if best_without_selected is None:
        dependency = "single_backend_only"
        dependency_reason = "No other backend completed, so no leave-one-out comparison exists."
    elif peer_reproduced_selected:
        dependency = "backend_consensus"
        dependency_reason = (
            "At least one other backend reproduced the selected patch at the same score."
        )
    elif (
        best_without_selected.final_score is not None
        and abs(final_score - best_without_selected.final_score) <= score_tolerance
    ):
        dependency = "score_tie_different_patch"
        dependency_reason = (
            "Removing the selected backend preserves the score, but the best peer "
            "uses a different patch."
        )
    else:
        dependency = "selected_backend_dependent"
        dependency_reason = (
            "Removing the selected backend lowers the best observed portfolio score."
        )

    return AgentMultiInteractionAblationReport(
        selected_optimizer=selected_run.optimizer,
        selected_candidate_id=selected_lineage.candidate_id if selected_lineage else None,
        selected_patch=selected_patch,
        selected_patch_paths=(
            list(selected_lineage.patch_paths) if selected_lineage else []
        ),
        final_score=final_score,
        best_without_selected_optimizer=(
            best_without_selected.optimizer if best_without_selected else None
        ),
        best_without_selected_score=(
            best_without_selected.final_score if best_without_selected else None
        ),
        score_delta_without_selected=score_delta_without_selected,
        selected_backend_required=selected_backend_required,
        dependency=dependency,
        dependency_reason=dependency_reason,
        consensus_backends=consensus_backends,
        consensus_backend_count=len(consensus_backends),
        shared_selected_patch_paths=_ordered_patch_paths_for_keys(
            shared_selected_patch_paths,
            list(selected_patch),
        ),
        unique_selected_patch_paths=_ordered_patch_paths_for_keys(
            unique_selected_patch_paths,
            list(selected_patch),
        ),
        selected_patch_support={
            path: selected_patch_support[path]
            for path in _ordered_patch_paths_for_keys(
                selected_patch_support,
                list(selected_patch),
            )
        },
        backend_scoreboard=[
            {
                "optimizer": row.optimizer,
                "rank": row.rank,
                "status": row.status,
                "final_score": row.final_score,
                "improved": row.improved,
                "candidate_id": row.candidate_id,
                "patch_paths": list(row.patch_paths),
                "selection_relation": row.selection_relation,
            }
            for row in sorted(completed, key=_lineage_selection_key, reverse=True)
        ],
    )


def _backend_run_candidate_patch(
    run: AgentMultiInteractionBackendRun,
    target: OptimizationTarget,
) -> dict[str, Any]:
    if run.result is None:
        return {}
    return _candidate_contribution_patch(
        run.result.reoptimization_result.best_candidate,
        target,
    )


def _candidate_contribution_patch(
    candidate: Any,
    target: OptimizationTarget,
) -> dict[str, Any]:
    if candidate is None:
        return {}
    base_candidate = target.seed_candidate()
    changed = {
        path: candidate.get_path(path)
        for path in target.search_space
        if candidate.get_path(path) != base_candidate.get_path(path)
    }
    if changed:
        return changed
    raw_patch = getattr(candidate, "patch", None)
    return dict(raw_patch or {})


def _patch_signature(patch: Mapping[str, Any]) -> str:
    return json.dumps(dict(patch), sort_keys=True, default=str, separators=(",", ":"))


def _value_signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _patches_share_values(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    return any(
        path in second and _value_signature(value) == _value_signature(second[path])
        for path, value in first.items()
    )


def _ordered_patch_paths(
    target: OptimizationTarget,
    patch: Mapping[str, Any],
) -> list[str]:
    ordered = [path for path in target.search_space if path in patch]
    ordered.extend(path for path in patch if path not in target.search_space)
    return ordered


def _ordered_patch_paths_for_keys(
    paths: Iterable[str],
    order: Sequence[str],
) -> list[str]:
    path_set = set(paths)
    ordered = [path for path in order if path in path_set]
    ordered.extend(sorted(path for path in path_set if path not in order))
    return ordered


def _lineage_selection_key(
    row: AgentMultiInteractionBackendLineage,
) -> tuple[float, int, int, int]:
    return (
        row.final_score if row.final_score is not None else float("-inf"),
        1 if row.improved else 0,
        -row.rank,
        -row.total_evaluations,
    )


def _dedupe_optimizer_pool(pool: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for item in pool:
        normalized = _normalize_optimizer_name(str(item))
        if normalized in seen:
            continue
        seen.add(normalized)
        names.append(normalized)
    return names


def _backend_kwargs_for_multi_interaction(
    optimizer_name: str,
    *,
    target: OptimizationTarget,
    metric_names: Sequence[str],
    optimizer_kwargs: Mapping[str, Any],
    optimizer_kwargs_by_backend: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    target_score = float(optimizer_kwargs.get("target_score", 0.99))
    if optimizer_name == "agent":
        defaults.update({"max_candidates": 16})
    elif optimizer_name in {"council", "society"}:
        defaults.update(
            {
                "max_rounds": 2,
                "beam_width": 4,
                "max_proposals_per_round": 16,
                "target_score": target_score,
            }
        )
    elif optimizer_name == "social_memory":
        defaults.update(
            {
                "max_rounds": 2,
                "beam_width": 4,
                "max_proposals_per_round": 16,
                "target_score": target_score,
            }
        )
    elif optimizer_name == "curriculum":
        defaults.update({"max_candidates_per_stage": 8, "target_score": target_score})
    elif optimizer_name == "evolution":
        defaults.update(
            {
                "population_size": min(10, max(4, len(target.search_space) * 2)),
                "generations": 2,
                "elite_count": 2,
                "seed": 42,
                "target_score": target_score,
            }
        )
    elif optimizer_name == "tpe":
        defaults.update({"n_trials": 8, "seed": 42, "target_score": target_score})
    elif optimizer_name == "pareto":
        defaults.update({"n_trials": 8, "seed": 42, "target_score": target_score})
        if metric_names:
            defaults["objective_names"] = list(metric_names[:4])
    elif optimizer_name == "bandit":
        defaults.update(
            {
                "max_candidates": 8,
                "total_budget": 12,
                "selection": "best",
                "target_score": target_score,
            }
        )
    shared_keys = {"include_seed", "auto_diagnose", "diagnostic_score_threshold"}
    shared_kwargs = {
        key: value
        for key, value in dict(optimizer_kwargs).items()
        if key in shared_keys
    }
    if optimizer_name != "agent" and "target_score" in optimizer_kwargs:
        shared_kwargs["target_score"] = optimizer_kwargs["target_score"]
    combined = {
        **defaults,
        **shared_kwargs,
        **dict(optimizer_kwargs_by_backend.get(optimizer_name, {})),
    }
    return combined


def _backend_allocation_weight(
    optimizer_name: str,
    *,
    target: OptimizationTarget,
    feedback_cases: Sequence[AgentFeedbackCase],
    diagnoses: Sequence[ComponentDiagnosis],
    search_paths: Sequence[str],
    metric_names: Sequence[str],
) -> tuple[float, str]:
    layers = set(target.layers)
    text = " ".join(
        [
            " ".join(layers),
            " ".join(search_paths),
            " ".join(metric_names),
            " ".join(diagnosis.component for diagnosis in diagnoses),
            " ".join(diagnosis.failure_mode for diagnosis in diagnoses),
        ]
    ).lower()
    path_count = len(search_paths) if search_paths else len(target.search_space)
    metric_count = len(metric_names)
    failed_count = sum(1 for case in feedback_cases if not case.passed)
    candidate_space_size = _target_search_space_cardinality(target)
    architecture_config_signal = _architecture_config_signal(text)

    weights = {
        "agent": 0.25,
        "curriculum": 0.55,
        "council": 0.6,
        "society": 0.65,
        "social_memory": 0.55,
        "evolution": 0.5,
        "pareto": 0.45,
        "tpe": 0.4,
        "bandit": 0.4,
    }
    reasons: list[str] = []
    weight = weights.get(optimizer_name, 0.1)
    if failed_count:
        weight += 0.1
        reasons.append(f"{failed_count} failing feedback case(s)")
    if optimizer_name == "agent":
        if 0 < path_count <= 3:
            weight += 0.45
            reasons.append("focused deterministic diagnosis search")
        if target.search_space and candidate_space_size <= 32:
            weight += 0.25
            reasons.append(f"exact categorical search space size {candidate_space_size}")
        if architecture_config_signal:
            weight += 0.35
            reasons.append("architecture/config signal")
    if path_count > 1:
        if optimizer_name in {"council", "society", "evolution"}:
            weight += 0.3
        if optimizer_name in {"curriculum", "social_memory"}:
            weight += 0.15
        reasons.append(f"{path_count} diagnosed search paths")
    if metric_count > 1:
        if optimizer_name in {"curriculum", "pareto"}:
            weight += 0.35
        if optimizer_name in {"society", "council", "social_memory"}:
            weight += 0.15
        reasons.append(f"{metric_count} failed metrics")
    if any(token in text for token in ("multi_agent", "handoff", "coordination", "review")):
        if optimizer_name in {"society", "council"}:
            weight += 0.45
        if optimizer_name == "social_memory":
            weight += 0.15
        reasons.append("multi-agent coordination signal")
    if "memory" in text or "cross_trial" in text:
        if optimizer_name == "social_memory":
            weight += 0.45
        if optimizer_name in {"society", "council"}:
            weight += 0.1
        reasons.append("memory/history signal")
    if "policy" in text or "security" in text:
        if optimizer_name in {"agent", "evolution", "bandit"}:
            weight += 0.15
        reasons.append("policy/security signal")
    if len(target.search_space) >= 6:
        if optimizer_name in {"tpe", "evolution"}:
            weight += 0.3
        if optimizer_name == "bandit":
            weight += 0.15
        reasons.append("larger categorical search space")
    if len(feedback_cases) > 1:
        if optimizer_name in {"bandit", "social_memory", "curriculum"}:
            weight += 0.2
        reasons.append("multi-observation replay window")
    if not reasons:
        reasons.append("deterministic fallback allocation")
    return weight, "; ".join(dict.fromkeys(reasons))


def _target_search_space_cardinality(target: OptimizationTarget) -> int:
    total = 1
    for values in target.search_space.values():
        if isinstance(values, (list, tuple, set)):
            total *= max(1, len(values))
        else:
            total *= 1
        if total > 1_000_000:
            return total
    return total


def _architecture_config_signal(text: str) -> bool:
    return any(
        token in text
        for token in (
            "architecture",
            "config",
            "framework",
            "adapter",
            "trace",
            "event_stream",
            "streaming",
            "orchestration",
            "workflow",
            "runtime",
            "instrumentation",
            "otel",
            "langchain",
            "langgraph",
            "openai_agents",
            "pipecat",
            "livekit",
        )
    )


def _feedback_metric_names(feedback_cases: Sequence[AgentFeedbackCase]) -> list[str]:
    names: set[str] = set()
    for case in feedback_cases:
        names.update(str(key) for key in case.metrics.keys())
        for failure in case.failures:
            names.update(_METRIC_NAME_RE.findall(failure))
    return sorted(names)


def _failed_feedback_metric_names(feedback_cases: Sequence[AgentFeedbackCase]) -> list[str]:
    names: set[str] = set()
    for case in feedback_cases:
        for failure in case.failures:
            names.update(_METRIC_NAME_RE.findall(failure))
    return sorted(names)


def _resolve_rollback_decision(
    *,
    rollback_decision: Optional[AgentRollbackDecision],
    deployment: Optional[DeploymentLike],
    live_evaluations: Optional[Sequence[Any]],
    simulation_evaluator: Any,
    rollback_kwargs: Mapping[str, Any],
) -> tuple[AgentRollbackDecision, str]:
    if rollback_decision is not None:
        return rollback_decision, "rollback_decision"
    if deployment is None:
        raise ValueError(
            "AgentFeedbackOptimizer requires deployment or rollback_decision."
    )
    decision = check_agent_deployment_rollback(
        deployment,
        live_evaluations=(
            list(live_evaluations) if live_evaluations is not None else None
        ),
        simulation_evaluator=simulation_evaluator,
        **dict(rollback_kwargs),
    )
    source = "live_evaluations" if live_evaluations is not None else "simulation_replay"
    return decision, source


def _feedback_cases_from_rollback(
    decision: AgentRollbackDecision,
) -> list[AgentFeedbackCase]:
    return [
        AgentFeedbackCase(
            index=observation.index,
            candidate_id=observation.candidate_id,
            score=observation.score,
            passed=observation.passed,
            failures=list(observation.failures),
            metrics=dict(observation.metrics),
            metadata={
                **dict(observation.metadata),
                "rollback_required": decision.rollback_required,
            },
        )
        for observation in decision.observations
    ]


def _diagnose_feedback_cases(
    feedback_cases: Sequence[AgentFeedbackCase],
    *,
    target: OptimizationTarget,
    failing_threshold: float,
) -> list[ComponentDiagnosis]:
    diagnostics: list[ComponentDiagnosis] = []
    failed_cases = [case for case in feedback_cases if not case.passed]
    for case in failed_cases:
        diagnostics.extend(
            diagnose_agent_report_evaluation(
                _agent_report_from_feedback_case(case),
                failing_threshold=failing_threshold,
                confidence=0.9,
            )
        )
        for failure in case.failures:
            diagnostics.extend(diagnose_text(failure, confidence=0.75))

    if not diagnostics and failed_cases:
        diagnostics.append(
            ComponentDiagnosis(
                component="custom",
                failure_mode="unknown",
                confidence=0.5,
                evidence="Live feedback score regression without metric-specific diagnosis.",
                suggested_paths=list(target.search_space),
                metadata={"failed_feedback_cases": len(failed_cases)},
            )
        )
    return _dedupe_diagnoses(diagnostics)


def _agent_report_from_feedback_case(case: AgentFeedbackCase) -> dict[str, Any]:
    metrics = dict(case.metrics)
    for failure in case.failures:
        for metric_name in _METRIC_NAME_RE.findall(failure):
            metrics.setdefault(metric_name, 0.0)
    return {
        "summary": {"metric_averages": metrics},
        "cases": [
            {
                "id": f"feedback-{case.index}",
                "metrics": [
                    {
                        "name": name,
                        "score": score,
                        "reason": "; ".join(case.failures),
                    }
                    for name, score in metrics.items()
                ],
                "findings": [
                    {
                        "metric": name,
                        "score": score,
                        "evidence": "; ".join(case.failures),
                    }
                    for name, score in metrics.items()
                ],
            }
        ],
    }


_METRIC_NAME_RE = re.compile(r"metric '([^']+)'")


def _search_paths_for_feedback(
    target: OptimizationTarget,
    diagnoses: Sequence[ComponentDiagnosis],
) -> list[str]:
    allowed_paths = relevant_search_paths(target.search_space, diagnoses)
    return [path for path in target.search_space if path in allowed_paths]


def _resolve_feedback_optimizer(name: str) -> type:
    normalized = _normalize_optimizer_name(name)
    optimizers = {
        "agent": AgentOptimizer,
        "deterministic": AgentOptimizer,
        "council": CouncilAgentOptimizer,
        "society": SocietyAgentOptimizer,
        "social_memory": AgentSocialMemoryOptimizer,
        "curriculum": AgentCurriculumOptimizer,
        "evolution": AgentEvolutionOptimizer,
        "tpe": AgentTPEOptimizer,
        "pareto": AgentParetoOptimizer,
        "bandit": AgentBanditOptimizer,
    }
    if normalized not in optimizers:
        raise ValueError(
            "optimizer must be one of: agent, deterministic, council, society, "
            "social_memory, curriculum, evolution, tpe, pareto, or bandit."
        )
    return optimizers[normalized]


def _normalize_optimizer_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "agentoptimizer": "agent",
        "agent_optimizer": "agent",
        "deterministic_agent_optimizer": "deterministic",
        "councilagentoptimizer": "council",
        "council_agent_optimizer": "council",
        "societyagentoptimizer": "society",
        "society_agent_optimizer": "society",
        "agentsocialmemoryoptimizer": "social_memory",
        "agent_social_memory_optimizer": "social_memory",
        "socialmemory": "social_memory",
        "social_memory_optimizer": "social_memory",
        "futureagi_social_memory": "social_memory",
        "futureagi_social_memory_optimizer": "social_memory",
        "agentcurriculumoptimizer": "curriculum",
        "agent_curriculum_optimizer": "curriculum",
        "curriculum_optimizer": "curriculum",
        "deliberate_practice": "curriculum",
        "deliberate_practice_curriculum": "curriculum",
        "agentevolutionoptimizer": "evolution",
        "agent_evolution_optimizer": "evolution",
        "agenttpeoptimizer": "tpe",
        "agent_tpe_optimizer": "tpe",
        "agentparetooptimizer": "pareto",
        "agent_pareto_optimizer": "pareto",
        "agentbanditoptimizer": "bandit",
        "agent_bandit_optimizer": "bandit",
        "agentmultiinteractionoptimizer": "multi_interaction",
        "agent_multi_interaction_optimizer": "multi_interaction",
        "multiinteraction": "multi_interaction",
        "multi_interaction_optimizer": "multi_interaction",
        "portfolio": "multi_interaction",
        "portfolio_optimizer": "multi_interaction",
        "auto": "multi_interaction",
        "auto_backend": "multi_interaction",
    }
    return aliases.get(
        normalized,
        normalized.replace("agent_", "").replace("_optimizer", "").replace("optimizer", ""),
    )
