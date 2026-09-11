from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Type

from ..optimizers.agent import AgentOptimizer
from ..optimizers.agent_evolution import AgentEvolutionOptimizer
from ..optimizers.agent_social_memory import AgentSocialMemoryOptimizer
from ..evidence import score_simulation_evidence
from ..simulation import _coerce_score, _iter_report_scores, _run_sync
from ..targets import (
    AgentCandidate,
    CandidateEvaluation,
    OptimizationLayer,
    OptimizationTarget,
    set_path,
)
from ..types import EvaluationResult, OptimizationResult

ManifestRunner = Callable[[Mapping[str, Any], AgentCandidate], Any]
ManifestScorer = Callable[[Mapping[str, Any], Any, AgentCandidate], Any]


@dataclass
class SimulateManifestOptimizationProblem:
    """
    Bridge portable simulation manifests into AgentOptimizer-style config search.

    `base_manifest` is the runnable manifest without its `optimization` block.
    Candidate configs are deep-merged into it, then `evaluate_manifest` runs the
    real simulator/world/eval stack. The returned report, or the optional
    `score_manifest` result, is normalized into `CandidateEvaluation`.
    """

    base_manifest: Mapping[str, Any]
    target: OptimizationTarget
    evaluate_manifest: ManifestRunner
    score_manifest: Optional[ManifestScorer] = None
    evidence_scorer_config: Optional[Mapping[str, Any]] = None
    threshold: float = 0.7
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    optimizer_cls: Type[Any] = AgentOptimizer
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        evaluate_manifest: ManifestRunner,
        score_manifest: Optional[ManifestScorer] = None,
        name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> "SimulateManifestOptimizationProblem":
        optimization = _require_mapping(
            manifest.get("optimization"),
            "manifest.optimization",
        )
        target_config = _target_config(optimization)
        optimizer_kwargs = _optimizer_kwargs(
            _optional_mapping(optimization.get("optimizer"))
        )
        optimizer_cls = _optimizer_cls(_optional_mapping(optimization.get("optimizer")))
        evidence_scorer_config = _evidence_scorer_config(
            optimization,
            target_config,
            base_manifest=manifest,
        )

        base_manifest = copy.deepcopy(dict(manifest))
        base_manifest.pop("optimization", None)

        manifest_name = str(name or manifest.get("name") or "agent-simulate-manifest")
        target_metadata = copy.deepcopy(dict(target_config.get("metadata") or {}))
        target_metadata.setdefault("source", "simulate_manifest")
        target_metadata.setdefault("manifest_name", manifest_name)

        target = OptimizationTarget(
            name=str(target_config.get("name") or manifest_name),
            layers=_layers(target_config.get("layers")),
            base_config=copy.deepcopy(dict(target_config["base_config"])),
            search_space=_search_space(target_config["search_space"]),
            metadata=target_metadata,
        )
        return cls(
            base_manifest=base_manifest,
            target=target,
            evaluate_manifest=evaluate_manifest,
            score_manifest=score_manifest,
            evidence_scorer_config=evidence_scorer_config,
            threshold=float(
                threshold
                if threshold is not None
                else optimization.get("threshold", 0.7)
            ),
            optimizer_kwargs=optimizer_kwargs,
            optimizer_cls=optimizer_cls,
            metadata={
                "source": "simulate_manifest",
                "manifest_name": manifest_name,
                "optimizer_algorithm": _optimizer_algorithm_name(optimizer_cls),
            },
        )

    def candidate_manifest(self, candidate: AgentCandidate) -> dict[str, Any]:
        merged = deep_merge(
            copy.deepcopy(dict(self.base_manifest)),
            copy.deepcopy(candidate.config),
        )
        _apply_candidate_patch_replacements(merged, candidate)
        return merged

    def evaluate_candidate(self, candidate: AgentCandidate) -> CandidateEvaluation:
        candidate_manifest = self.candidate_manifest(candidate)
        report = _run_sync(self.evaluate_manifest(candidate_manifest, candidate))
        score_source = report
        if self.score_manifest is not None:
            score_source = _run_sync(
                self.score_manifest(candidate_manifest, report, candidate)
            )
        elif self.evidence_scorer_config is not None and (
            not self.evidence_scorer_config.get("_auto")
            or not _report_has_score(report)
        ):
            score_source = score_simulation_evidence(
                report,
                manifest=candidate_manifest,
                candidate=candidate,
                config=self.evidence_scorer_config,
            )

        metadata = {
            **dict(self.metadata),
            "candidate_manifest": copy.deepcopy(candidate_manifest),
            "candidate_patch": copy.deepcopy(candidate.patch),
            "patch": copy.deepcopy(candidate.patch),
            "search_paths": list(candidate.metadata.get("search_paths", [])),
        }
        evaluation = _candidate_evaluation_from_value(
            score_source,
            candidate,
            report=report,
            metadata=metadata,
        )
        return evaluation

    def build_optimizer(
        self,
        optimizer_cls: Optional[Type[Any]] = None,
        **optimizer_kwargs: Any,
    ) -> Any:
        optimizer_cls = optimizer_cls or self.optimizer_cls
        kwargs = {**dict(self.optimizer_kwargs), **optimizer_kwargs}
        kwargs = _filter_optimizer_kwargs(optimizer_cls, kwargs)
        return optimizer_cls(
            target=self.target,
            evaluate_candidate=self.evaluate_candidate,
            **kwargs,
        )

    def optimize(
        self,
        optimizer_cls: Optional[Type[Any]] = None,
        **optimizer_kwargs: Any,
    ) -> OptimizationResult:
        return _as_optimization_result(
            self.build_optimizer(optimizer_cls, **optimizer_kwargs).optimize()
        )


ManifestOptimizationProblem = SimulateManifestOptimizationProblem


@dataclass
class SimulateEvalSuiteOptimizationProblem:
    """
    Bridge promptfoo-style simulate-sdk eval suites into AgentOptimizer search.

    Candidate configs are deep-merged into the eval-suite JSON/YAML contract,
    then scored by simulate-sdk's public `run_eval_suite` API. This gives
    optimizer users a local prompt/provider/test/assertion loop without writing
    adapter glue.
    """

    base_suite: Mapping[str, Any]
    target: OptimizationTarget
    run_suite: Callable[[Mapping[str, Any], AgentCandidate], Any]
    threshold: float = 1.0
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    optimizer_cls: Type[Any] = AgentOptimizer
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_suite(
        cls,
        suite: Mapping[str, Any],
        *,
        run_suite: Optional[Callable[[Mapping[str, Any], AgentCandidate], Any]] = None,
        name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> "SimulateEvalSuiteOptimizationProblem":
        optimization = _require_mapping(
            suite.get("optimization"),
            "suite.optimization",
        )
        target_config = _target_config(optimization)
        optimizer_kwargs = _optimizer_kwargs(
            _optional_mapping(optimization.get("optimizer"))
        )
        optimizer_cls = _optimizer_cls(_optional_mapping(optimization.get("optimizer")))
        base_suite = copy.deepcopy(dict(suite))
        base_suite.pop("optimization", None)
        suite_name = str(name or suite.get("name") or "agent-simulate-eval-suite")
        target_metadata = copy.deepcopy(dict(target_config.get("metadata") or {}))
        target_metadata.setdefault("source", "simulate_eval_suite")
        target_metadata.setdefault("suite_name", suite_name)
        return cls(
            base_suite=base_suite,
            target=OptimizationTarget(
                name=str(target_config.get("name") or suite_name),
                layers=_layers(target_config.get("layers")),
                base_config=copy.deepcopy(dict(target_config["base_config"])),
                search_space=_search_space(target_config["search_space"]),
                metadata=target_metadata,
            ),
            run_suite=run_suite or _public_eval_suite_runner(),
            threshold=float(
                threshold
                if threshold is not None
                else optimization.get("threshold", 1.0)
            ),
            optimizer_kwargs=optimizer_kwargs,
            optimizer_cls=optimizer_cls,
            metadata={
                "source": "simulate_eval_suite",
                "suite_name": suite_name,
                "optimizer_algorithm": _optimizer_algorithm_name(optimizer_cls),
            },
        )

    def candidate_suite(self, candidate: AgentCandidate) -> dict[str, Any]:
        merged = deep_merge(
            copy.deepcopy(dict(self.base_suite)),
            copy.deepcopy(candidate.config),
        )
        _apply_candidate_patch_replacements(merged, candidate)
        return merged

    def evaluate_candidate(self, candidate: AgentCandidate) -> CandidateEvaluation:
        candidate_suite = self.candidate_suite(candidate)
        result = _run_sync(self.run_suite(candidate_suite, candidate))
        metadata = {
            **dict(self.metadata),
            "candidate_suite": copy.deepcopy(candidate_suite),
            "candidate_patch": copy.deepcopy(candidate.patch),
            "patch": copy.deepcopy(candidate.patch),
            "report": copy.deepcopy(result),
            "search_paths": list(candidate.metadata.get("search_paths", [])),
        }
        return _candidate_evaluation_from_value(
            result,
            candidate,
            report=result,
            metadata=metadata,
        )

    def build_optimizer(
        self,
        optimizer_cls: Optional[Type[Any]] = None,
        **optimizer_kwargs: Any,
    ) -> Any:
        optimizer_cls = optimizer_cls or self.optimizer_cls
        kwargs = {**dict(self.optimizer_kwargs), **optimizer_kwargs}
        kwargs = _filter_optimizer_kwargs(optimizer_cls, kwargs)
        return optimizer_cls(
            target=self.target,
            evaluate_candidate=self.evaluate_candidate,
            **kwargs,
        )

    def optimize(
        self,
        optimizer_cls: Optional[Type[Any]] = None,
        **optimizer_kwargs: Any,
    ) -> OptimizationResult:
        return _as_optimization_result(
            self.build_optimizer(optimizer_cls, **optimizer_kwargs).optimize()
        )


EvalSuiteOptimizationProblem = SimulateEvalSuiteOptimizationProblem


@dataclass
class SimulateSuiteOptimizationProblem:
    """
    Bridge promptfoo-style Agent Learning suites into AgentOptimizer search.

    This is the suite-level counterpart to ``SimulateManifestOptimizationProblem``
    and ``SimulateEvalSuiteOptimizationProblem``: candidate configs are merged
    into a full Agent Learning suite, then the whole mixed workflow can be
    scored across simulation, eval, red-team, nested suites, and optimization
    children. It is the optimizer primitive for trajectory-level trinity gates,
    not isolated prompt/provider edits.
    """

    base_suite: Mapping[str, Any]
    target: OptimizationTarget
    run_suite: Callable[[Mapping[str, Any], AgentCandidate], Any]
    score_suite: Optional[ManifestScorer] = None
    threshold: float = 1.0
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    optimizer_cls: Type[Any] = AgentOptimizer
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_suite(
        cls,
        suite: Mapping[str, Any],
        *,
        run_suite: Callable[[Mapping[str, Any], AgentCandidate], Any],
        score_suite: Optional[ManifestScorer] = None,
        name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> "SimulateSuiteOptimizationProblem":
        optimization = _require_mapping(
            suite.get("optimization"),
            "suite.optimization",
        )
        target_config = _target_config(optimization)
        optimizer_kwargs = _optimizer_kwargs(
            _optional_mapping(optimization.get("optimizer"))
        )
        optimizer_cls = _optimizer_cls(_optional_mapping(optimization.get("optimizer")))
        base_suite = copy.deepcopy(dict(suite))
        base_suite.pop("optimization", None)
        suite_name = str(name or suite.get("name") or "agent-learning-suite")
        target_metadata = copy.deepcopy(dict(target_config.get("metadata") or {}))
        target_metadata.setdefault("source", "agent_learning_suite")
        target_metadata.setdefault("suite_name", suite_name)
        return cls(
            base_suite=base_suite,
            target=OptimizationTarget(
                name=str(target_config.get("name") or suite_name),
                layers=_layers(target_config.get("layers")),
                base_config=copy.deepcopy(dict(target_config["base_config"])),
                search_space=_search_space(target_config["search_space"]),
                metadata=target_metadata,
            ),
            run_suite=run_suite,
            score_suite=score_suite or _score_agent_learning_suite,
            threshold=float(
                threshold
                if threshold is not None
                else optimization.get("threshold", 1.0)
            ),
            optimizer_kwargs=optimizer_kwargs,
            optimizer_cls=optimizer_cls,
            metadata={
                "source": "agent_learning_suite",
                "suite_name": suite_name,
                "optimizer_algorithm": _optimizer_algorithm_name(optimizer_cls),
            },
        )

    def candidate_suite(self, candidate: AgentCandidate) -> dict[str, Any]:
        merged = deep_merge(
            copy.deepcopy(dict(self.base_suite)),
            copy.deepcopy(candidate.config),
        )
        _apply_candidate_patch_replacements(merged, candidate)
        return merged

    def evaluate_candidate(self, candidate: AgentCandidate) -> CandidateEvaluation:
        candidate_suite = self.candidate_suite(candidate)
        result = _run_sync(self.run_suite(candidate_suite, candidate))
        score_source = result
        if self.score_suite is not None:
            score_source = _run_sync(
                self.score_suite(candidate_suite, result, candidate)
            )

        metadata = {
            **dict(self.metadata),
            "candidate_suite": copy.deepcopy(candidate_suite),
            "candidate_patch": copy.deepcopy(candidate.patch),
            "patch": copy.deepcopy(candidate.patch),
            "report": copy.deepcopy(result),
            "report_summary": copy.deepcopy(_mapping_summary(result)),
            "search_paths": list(candidate.metadata.get("search_paths", [])),
        }
        return _candidate_evaluation_from_value(
            score_source,
            candidate,
            report=result,
            metadata=metadata,
        )

    def build_optimizer(
        self,
        optimizer_cls: Optional[Type[Any]] = None,
        **optimizer_kwargs: Any,
    ) -> Any:
        optimizer_cls = optimizer_cls or self.optimizer_cls
        kwargs = {**dict(self.optimizer_kwargs), **optimizer_kwargs}
        kwargs = _filter_optimizer_kwargs(optimizer_cls, kwargs)
        return optimizer_cls(
            target=self.target,
            evaluate_candidate=self.evaluate_candidate,
            **kwargs,
        )

    def optimize(
        self,
        optimizer_cls: Optional[Type[Any]] = None,
        **optimizer_kwargs: Any,
    ) -> OptimizationResult:
        return _as_optimization_result(
            self.build_optimizer(optimizer_cls, **optimizer_kwargs).optimize()
        )


SuiteOptimizationProblem = SimulateSuiteOptimizationProblem


def problem_from_simulate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
    name: Optional[str] = None,
) -> SimulateManifestOptimizationProblem:
    """Build a manifest optimization problem using simulate-sdk's public runtime."""

    build_problem = _simulate_sdk_attr("build_manifest_optimization_problem")
    return build_problem(
        manifest,
        manifest_path=Path(manifest_path).expanduser().resolve(),
        name=name,
    )


def problem_from_simulate_manifest_file(
    path: str | Path,
    *,
    name: Optional[str] = None,
) -> SimulateManifestOptimizationProblem:
    """Load an agent-simulate manifest file and build an optimization problem."""

    load_manifest = _simulate_sdk_attr("load_manifest")
    manifest_path = Path(path).expanduser().resolve()
    return problem_from_simulate_manifest(
        load_manifest(manifest_path),
        manifest_path=manifest_path,
        name=name,
    )


def optimize_simulate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
    name: Optional[str] = None,
    optimizer_cls: Optional[Type[Any]] = None,
    **optimizer_kwargs: Any,
) -> OptimizationResult:
    """Optimize an in-memory agent-simulate manifest through simulate-sdk."""

    return problem_from_simulate_manifest(
        manifest,
        manifest_path=manifest_path,
        name=name,
    ).optimize(optimizer_cls=optimizer_cls, **optimizer_kwargs)


def optimize_simulate_manifest_file(
    path: str | Path,
    *,
    name: Optional[str] = None,
    optimizer_cls: Optional[Type[Any]] = None,
    **optimizer_kwargs: Any,
) -> OptimizationResult:
    """Optimize an agent-simulate manifest file through simulate-sdk."""

    return problem_from_simulate_manifest_file(path, name=name).optimize(
        optimizer_cls=optimizer_cls,
        **optimizer_kwargs,
    )


def problem_from_eval_suite(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
    name: Optional[str] = None,
) -> SimulateEvalSuiteOptimizationProblem:
    """Build an eval-suite optimization problem using simulate-sdk's runtime."""

    run_eval_suite = _simulate_sdk_attr("run_eval_suite")
    suite_path = _suite_file_like_path(suite_path)

    def run_suite(candidate_suite: Mapping[str, Any], candidate: AgentCandidate) -> Any:
        return run_eval_suite(candidate_suite, suite_path=suite_path)

    return SimulateEvalSuiteOptimizationProblem.from_suite(
        suite,
        run_suite=run_suite,
        name=name,
    )


def problem_from_eval_suite_file(
    path: str | Path,
    *,
    name: Optional[str] = None,
) -> SimulateEvalSuiteOptimizationProblem:
    """Load a simulate-sdk eval suite file and build an optimization problem."""

    load_eval_suite_file = _simulate_sdk_attr("load_eval_suite_file")
    suite_path = Path(path).expanduser().resolve()
    return problem_from_eval_suite(
        load_eval_suite_file(suite_path),
        suite_path=suite_path,
        name=name,
    )


def optimize_eval_suite(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
    name: Optional[str] = None,
    optimizer_cls: Optional[Type[Any]] = None,
    **optimizer_kwargs: Any,
) -> OptimizationResult:
    """Optimize an in-memory simulate-sdk eval suite."""

    return problem_from_eval_suite(
        suite,
        suite_path=suite_path,
        name=name,
    ).optimize(optimizer_cls=optimizer_cls, **optimizer_kwargs)


def optimize_eval_suite_file(
    path: str | Path,
    *,
    name: Optional[str] = None,
    optimizer_cls: Optional[Type[Any]] = None,
    **optimizer_kwargs: Any,
) -> OptimizationResult:
    """Optimize a simulate-sdk eval suite file."""

    return problem_from_eval_suite_file(path, name=name).optimize(
        optimizer_cls=optimizer_cls,
        **optimizer_kwargs,
    )


def problem_from_agent_learning_suite(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
    name: Optional[str] = None,
) -> SimulateSuiteOptimizationProblem:
    """Build a full Agent Learning suite optimization problem."""

    run_agent_learning_suite = _agent_learning_suite_attr("run_suite")
    suite_path = _agent_learning_suite_file_like_path(suite_path)

    def run_suite(candidate_suite: Mapping[str, Any], candidate: AgentCandidate) -> Any:
        return run_agent_learning_suite(candidate_suite, suite_path=suite_path)

    return SimulateSuiteOptimizationProblem.from_suite(
        suite,
        run_suite=run_suite,
        score_suite=_score_agent_learning_suite,
        name=name,
    )


def problem_from_agent_learning_suite_file(
    path: str | Path,
    *,
    name: Optional[str] = None,
) -> SimulateSuiteOptimizationProblem:
    """Load an Agent Learning suite file and build an optimization problem."""

    load_suite_file = _agent_learning_suite_attr("load_suite_file")
    suite_path = Path(path).expanduser().resolve()
    return problem_from_agent_learning_suite(
        load_suite_file(suite_path),
        suite_path=suite_path,
        name=name,
    )


def optimize_agent_learning_suite(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
    name: Optional[str] = None,
    optimizer_cls: Optional[Type[Any]] = None,
    **optimizer_kwargs: Any,
) -> OptimizationResult:
    """Optimize an in-memory Agent Learning suite."""

    return problem_from_agent_learning_suite(
        suite,
        suite_path=suite_path,
        name=name,
    ).optimize(optimizer_cls=optimizer_cls, **optimizer_kwargs)


def optimize_agent_learning_suite_file(
    path: str | Path,
    *,
    name: Optional[str] = None,
    optimizer_cls: Optional[Type[Any]] = None,
    **optimizer_kwargs: Any,
) -> OptimizationResult:
    """Optimize an Agent Learning suite file."""

    return problem_from_agent_learning_suite_file(path, name=name).optimize(
        optimizer_cls=optimizer_cls,
        **optimizer_kwargs,
    )


def deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, Mapping):
        for key, value in patch.items():
            base[key] = deep_merge(base.get(key), value)
        return base
    if isinstance(base, list) and isinstance(patch, list):
        merged = list(base)
        for index, value in enumerate(patch):
            if index < len(merged):
                merged[index] = deep_merge(merged[index], value)
            else:
                merged.append(copy.deepcopy(value))
        return merged
    return copy.deepcopy(patch)


def _apply_candidate_patch_replacements(
    payload: dict[str, Any],
    candidate: AgentCandidate,
) -> None:
    """Reapply exact search-path patches after deep-merge candidate assembly."""

    for path, value in candidate.patch.items():
        set_path(payload, str(path), copy.deepcopy(value))


def _candidate_evaluation_from_value(
    value: Any,
    candidate: AgentCandidate,
    *,
    report: Any,
    metadata: Mapping[str, Any],
) -> CandidateEvaluation:
    if isinstance(value, CandidateEvaluation):
        return CandidateEvaluation(
            candidate=candidate,
            score=float(value.score),
            reason=value.reason,
            individual_results=list(value.individual_results or []),
            report=value.report if value.report is not None else report,
            metadata={**dict(metadata), **dict(value.metadata or {})},
        )
    if isinstance(value, EvaluationResult):
        return CandidateEvaluation(
            candidate=candidate,
            score=float(value.score),
            reason=value.reason,
            individual_results=[value],
            report=report,
            metadata={**dict(metadata), **dict(value.metadata or {})},
        )

    score = _score_from_value(value)
    reason = _reason_from_value(value)
    individual_results = _individual_results_from_value(value)
    report_value = _report_from_value(value, report)
    extra_metadata = _metadata_from_value(value)
    if score is None:
        score = _score_from_value(report)
    if score is None:
        scores = list(_iter_report_scores(report))
        if scores:
            score = sum(scores) / len(scores)
    if score is None:
        raise ValueError(
            "Manifest evaluation returned no score. Return a numeric score, "
            "EvaluationResult, CandidateEvaluation, score-bearing mapping/object, "
            "or provide score_manifest."
        )
    return CandidateEvaluation(
        candidate=candidate,
        score=score,
        reason=reason,
        individual_results=individual_results,
        report=report_value,
        metadata={**dict(metadata), **extra_metadata},
    )


def _declared_anchor_objective(value: Any) -> Optional[Mapping[str, Any]]:
    """Return a REAL declared objective (with ``evals`` carrying >=1 ``anchor``
    term) if the candidate value carries one — searched in the manifest/result
    locations only. NEVER synthesized from config (that over-reaches and regresses
    structural/hook manifests). Bug #2: only opted-in declared-anchor objectives
    get objective-anchored scoring; everything else keeps the engine score."""
    from fi.opt._objective_scoring import has_declared_anchor_objective

    if not isinstance(value, Mapping):
        return None
    candidates = [
        value.get("objective"),
        (value.get("evaluation") or {}).get("objective") if isinstance(value.get("evaluation"), Mapping) else None,
        ((value.get("simulation") or {}).get("inline") or {}).get("objective")
        if isinstance(value.get("simulation"), Mapping) else None,
        (value.get("scenario") or {}).get("objective") if isinstance(value.get("scenario"), Mapping) else None,
    ]
    for obj in candidates:
        if has_declared_anchor_objective(obj):
            return obj
    return None


def _candidate_metric_averages(value: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    if isinstance(value.get("metric_averages"), Mapping):
        return value["metric_averages"]
    summary = value.get("summary")
    if isinstance(summary, Mapping) and isinstance(summary.get("metric_averages"), Mapping):
        return summary["metric_averages"]
    return None


def _objective_anchored_score(value: Any) -> Optional[float]:
    """Bug #2: score a candidate on its DECLARED anchor objective (real dynamic
    range) instead of the all-metrics-mean ``evaluation_score``. Returns None
    unless BOTH a declared-anchor objective and metric_averages are present, so
    legacy/structural manifests fall through to the existing score unchanged."""
    objective = _declared_anchor_objective(value)
    if objective is None:
        return None
    metrics = _candidate_metric_averages(value)
    if not metrics:
        return None
    from fi.opt._objective_scoring import objective_score

    return _coerce_score(objective_score(metrics, objective).get("score"))


def _score_from_value(value: Any) -> Optional[float]:
    anchored = _objective_anchored_score(value)
    if anchored is not None:
        return anchored
    direct = _coerce_score(value)
    if direct is not None:
        return direct
    if isinstance(value, Mapping):
        for key in ("score", "final_score", "average_score", "optimization_score"):
            score = _coerce_score(value.get(key))
            if score is not None:
                return score
        summary = value.get("summary")
        if isinstance(summary, Mapping):
            for key in ("score", "final_score", "optimization_score"):
                score = _coerce_score(summary.get(key))
                if score is not None:
                    return score
    for key in ("score", "final_score", "average_score", "optimization_score"):
        score = _coerce_score(getattr(value, key, None))
        if score is not None:
            return score
    return None


def _reason_from_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("reason") or value.get("status") or "")
    return str(getattr(value, "reason", "") or "")


def _individual_results_from_value(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        results = value.get("individual_results")
        return list(results or [])
    return list(getattr(value, "individual_results", []) or [])


def _report_from_value(value: Any, fallback: Any) -> Any:
    if isinstance(value, Mapping) and "report" in value:
        return value["report"]
    report = getattr(value, "report", None)
    return fallback if report is None else report


def _metadata_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value.get("metadata") or {}))
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, Mapping):
        return copy.deepcopy(dict(metadata))
    return {}


def _mapping_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        summary = value.get("summary")
        if isinstance(summary, Mapping):
            return copy.deepcopy(dict(summary))
    return {}


def _score_agent_learning_suite(
    candidate_suite: Mapping[str, Any],
    result: Any,
    candidate: AgentCandidate,
) -> dict[str, Any]:
    summary = _mapping_summary(result)
    raw_score = _score_from_value(result)
    score = float(raw_score if raw_score is not None else 0.0)
    action_run_score: Optional[float] = None
    if isinstance(result, Mapping):
        status = str(result.get("status") or "")
        exit_code = int(result.get("exit_code", 1) or 0)
        capability_gate = bool(
            summary.get("capability_gate_passed")
            if "capability_gate_passed" in summary
            else True
        )
        executed = float(summary.get("executed_count") or 0.0)
        job_count = float(summary.get("job_count") or executed or 1.0)
        execution_score = executed / job_count if job_count else score
        if status != "passed" or exit_code != 0:
            score = min(score, execution_score)
        if not capability_gate:
            score = min(score, 0.5)
        action_run_score = _action_run_suite_score(result)
        if action_run_score is not None:
            score = min(score, action_run_score)
    return {
        "score": round(score, 4),
        "reason": str(result.get("status") if isinstance(result, Mapping) else ""),
        "metadata": {
            "suite_summary": summary,
            "action_run_score": action_run_score,
            "candidate_suite_name": candidate_suite.get("name"),
            "candidate_id": candidate.id,
        },
    }


def _action_run_suite_score(result: Mapping[str, Any]) -> Optional[float]:
    children = [
        child
        for child in result.get("children") or result.get("jobs") or []
        if isinstance(child, Mapping)
    ]
    action_children = [
        child
        for child in children
        if str(child.get("command") or "").replace("-", "_") == "action_run"
    ]
    if not action_children:
        return None
    scores: list[float] = []
    for child in action_children:
        exit_code = child.get("exit_code", 1)
        if int(exit_code if exit_code is not None else 1) != 0:
            scores.append(0.0)
            continue
        child_summary = _mapping_summary(child.get("result"))
        output_count = float(child_summary.get("output_count") or 0.0)
        written_count = float(child_summary.get("outputs_written_count") or 0.0)
        completion = (
            float(child_summary.get("output_completion_rate"))
            if child_summary.get("output_completion_rate") is not None
            else (written_count / output_count if output_count else 1.0)
        )
        evidence_depth = min(written_count / 4.0, 1.0)
        scores.append((0.8 * completion) + (0.2 * evidence_depth))
    return round(sum(scores) / len(scores), 4) if scores else None


def _target_config(optimization: Mapping[str, Any]) -> Mapping[str, Any]:
    target = _require_mapping(optimization.get("target"), "optimization.target")
    _require_mapping(target.get("base_config"), "optimization.target.base_config")
    search_space = _require_mapping(
        target.get("search_space"),
        "optimization.target.search_space",
    )
    if not search_space:
        raise ValueError("optimization.target.search_space must not be empty.")
    return target


def _optimizer_kwargs(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not config:
        return {}
    allowed = {
        "max_candidates",
        "max_rounds",
        "beam_width",
        "max_proposals_per_round",
        "include_seed",
        "auto_diagnose",
        "diagnoses",
        "diagnostic_score_threshold",
        "total_budget",
        "min_pulls_per_candidate",
        "exploration",
        "target_score",
        "selection",
        "population_size",
        "generations",
        "elite_count",
        "mutation_rate",
        "crossover_rate",
        "max_mutations_per_candidate",
        "tournament_size",
        "seed",
        "layer_path_bias",
        "mutation_library",
        "max_library_candidates",
        # Phase 4 (extend-only): declared budgets, Elo selection knobs,
        # two-chamber budgets, society ledger, strategy declaration, TPE trials.
        "eval_budget",
        "elo_k_factor",
        "elo_initial_rating",
        "samiti_budget",
        "sabha_budget",
        "society_ledger",
        "search_strategy",
        "n_trials",
        # Phase 4 (extend-only): regression-replay backend inputs — a local
        # AgentRegressionDataset (mapping coerced below) plus the delegated
        # repair backend selector consumed by FutureAGIRegressionReplayOptimizer.
        "dataset",
        "optimizer",
    }
    kwargs = {key: copy.deepcopy(config[key]) for key in allowed if key in config}
    dataset = kwargs.get("dataset")
    if isinstance(dataset, Mapping) and "cases" in dataset:
        from ..observability import AgentRegressionDataset

        kwargs["dataset"] = AgentRegressionDataset.model_validate(dict(dataset))
    return kwargs


def _optimizer_cls(config: Optional[Mapping[str, Any]]) -> Type[Any]:
    if not config:
        return AgentOptimizer
    raw = (
        config.get("algorithm")
        or config.get("type")
        or config.get("name")
        or config.get("strategy")
        or "agent"
    )
    normalized = str(raw or "agent").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "agent",
        "agent_optimizer",
        "deterministic",
        "candidate_search",
        "deterministic_candidate_search",
        "grid",
    }:
        return AgentOptimizer
    if normalized in {
        "evolution",
        "agent_evolution",
        "agent_evolution_optimizer",
        "domain_aware_evolution",
        "mutation",
        "mutation_library",
    }:
        return AgentEvolutionOptimizer
    if normalized in {
        "social_memory",
        "society",
        "agent_social_memory",
        "agent_social_memory_optimizer",
        "futureagi_social_memory",
        "futureagi_social_memory_optimizer",
        "multi_interaction",
        "multi_interaction_social_memory",
    }:
        return AgentSocialMemoryOptimizer
    # Phase 4 (extend-only): additional target-contract backends for the
    # optimizer profile matrix; legacy tokens above are untouched.
    if normalized in {"council", "council_agent", "council_agent_optimizer"}:
        from ..optimizers.council import CouncilAgentOptimizer

        return CouncilAgentOptimizer
    if normalized in {
        "society_agent",
        "society_agent_optimizer",
        "role_graph_society",
        "society_role_graph",
    }:
        from ..optimizers.council import SocietyAgentOptimizer

        return SocietyAgentOptimizer
    if normalized in {"tpe", "agent_tpe", "agent_tpe_optimizer"}:
        from ..optimizers.agent_tpe import AgentTPEOptimizer

        return AgentTPEOptimizer
    if normalized in {"bandit", "agent_bandit", "agent_bandit_optimizer", "ucb"}:
        from ..optimizers.agent_bandit import AgentBanditOptimizer

        return AgentBanditOptimizer
    if normalized in {
        "regression_replay",
        "futureagi_regression_replay",
        "futureagi_replay",
        "regression_replay_optimizer",
    }:
        from ..optimizers.futureagi_replay import FutureAGIRegressionReplayOptimizer

        return FutureAGIRegressionReplayOptimizer
    if normalized in {
        "curriculum",
        "agent_curriculum",
        "agent_curriculum_optimizer",
        "staged",
    }:
        from ..optimizers.agent_curriculum import AgentCurriculumOptimizer

        return AgentCurriculumOptimizer
    if normalized in {
        "pareto",
        "agent_pareto",
        "agent_pareto_optimizer",
        "multi_objective",
        "multi_objective_pareto",
    }:
        from ..optimizers.agent_pareto import AgentParetoOptimizer

        return AgentParetoOptimizer
    if normalized in {
        "feedback",
        "agent_feedback",
        "agent_feedback_optimizer",
        "diagnostic_feedback",
    }:
        from ..optimizers.agent_feedback import AgentFeedbackOptimizer

        return AgentFeedbackOptimizer
    raise ValueError(
        "optimization.optimizer.algorithm must be one of: agent, evolution, "
        "social_memory, council, society_role_graph, tpe, bandit, "
        "regression_replay, curriculum, pareto, feedback (deterministic agent "
        "backends), or a generative token routed through the generative "
        "eval-suite bridge: gepa, protegi, metaprompt, promptwizard, "
        "random_search, bayesian_search"
    )


def _optimizer_algorithm_name(optimizer_cls: Type[Any]) -> str:
    if optimizer_cls is AgentEvolutionOptimizer:
        return "evolution"
    if optimizer_cls is AgentSocialMemoryOptimizer:
        return "social_memory"
    name = getattr(optimizer_cls, "__name__", "")
    if name == "CouncilAgentOptimizer":
        return "council"
    if name == "SocietyAgentOptimizer":
        return "society_role_graph"
    if name == "AgentTPEOptimizer":
        return "tpe"
    if name == "AgentBanditOptimizer":
        return "bandit"
    if name == "FutureAGIRegressionReplayOptimizer":
        return "regression_replay"
    if name == "AgentCurriculumOptimizer":
        return "curriculum"
    if name == "AgentParetoOptimizer":
        return "pareto"
    if name == "AgentFeedbackOptimizer":
        return "feedback"
    return "agent"


def _as_optimization_result(result: Any) -> OptimizationResult:
    """Coerce backend audit records onto the OptimizationResult contract.

    Phase 4 (extend-only): ``FutureAGIRegressionReplayOptimizer`` returns an
    ``AgentFeedbackOptimizationResult`` audit record wrapping the inner
    ``reoptimization_result``; the manifest pipeline consumes the inner
    result with the replay audit carried in its metadata. Every existing
    backend already returns ``OptimizationResult`` and passes through.
    """

    inner = getattr(result, "reoptimization_result", None)
    if inner is None:
        return result
    metadata = dict(getattr(inner, "metadata", {}) or {})
    metadata.setdefault(
        "regression_replay",
        {
            "optimizer": getattr(result, "optimizer", None),
            "feedback_source": getattr(result, "feedback_source", None),
            "baseline_score": getattr(result, "baseline_score", None),
            "final_score": getattr(result, "final_score", None),
            "improved": getattr(result, "improved", None),
            "feedback_case_count": len(getattr(result, "feedback_cases", []) or []),
        },
    )
    inner.metadata = metadata
    return inner


def _evidence_scorer_config(
    optimization: Mapping[str, Any],
    target_config: Mapping[str, Any],
    *,
    base_manifest: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    raw = (
        optimization.get("simulation_evidence")
        or optimization.get("evidence_scorer")
        or optimization.get("scoring")
    )
    if raw is False:
        return None
    if isinstance(raw, str):
        normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"simulation_evidence", "evidence", "environment_evidence"}:
            return {"enabled": True, "method": "simulation_evidence"}
        return None
    if isinstance(raw, Mapping):
        method = str(
            raw.get("method")
            or raw.get("type")
            or raw.get("name")
            or raw.get("strategy")
            or "simulation_evidence"
        ).strip().lower().replace("-", "_").replace(" ", "_")
        enabled = bool(raw.get("enabled", True))
        if enabled and method in {
            "simulation_evidence",
            "evidence",
            "environment_evidence",
            "trace_evidence",
        }:
            config = copy.deepcopy(dict(raw))
            config["method"] = "simulation_evidence"
            return config
        return None

    if raw is True:
        return {"enabled": True, "method": "simulation_evidence"}

    layers = {str(layer).lower() for layer in target_config.get("layers", [])}
    should_auto_score = bool(layers & {"framework", "world", "orchestration"}) and not (
        _optional_mapping(
            _optional_mapping(base_manifest.get("evaluation")) or {}
        )
        and _optional_mapping(
            (_optional_mapping(base_manifest.get("evaluation")) or {}).get(
                "agent_report"
            )
        )
    )
    if should_auto_score:
        return {"enabled": True, "method": "simulation_evidence", "_auto": True}
    return None


def _report_has_score(report: Any) -> bool:
    if _score_from_value(report) is not None:
        return True
    return bool(list(_iter_report_scores(report)))


def _filter_optimizer_kwargs(
    optimizer_cls: Type[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(optimizer_cls)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = signature.parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return dict(kwargs)
    allowed = set(parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def _layers(value: Any) -> list[OptimizationLayer]:
    return list(value or ["harness", "evaluator"])


def _search_space(value: Mapping[str, Any]) -> dict[str, list[Any]]:
    search_space: dict[str, list[Any]] = {}
    for path, choices in value.items():
        if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
            raise ValueError(
                f"optimization.target.search_space.{path} must be a sequence."
            )
        if not choices:
            raise ValueError(
                f"optimization.target.search_space.{path} must not be empty."
            )
        search_space[str(path)] = copy.deepcopy(list(choices))
    return search_space


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _optional_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    return _require_mapping(value, "optimization.optimizer")


def _simulate_sdk_attr(name: str) -> Any:
    try:
        from fi import simulate as simulate_sdk
    except Exception as exc:  # pragma: no cover - optional dependency clarity
        raise RuntimeError(
            "agent-simulate is required for simulate-sdk manifest helpers. "
            "Install simulate-sdk or call ManifestOptimizationProblem.from_manifest "
            "with explicit evaluate_manifest/score_manifest callbacks."
        ) from exc
    try:
        return getattr(simulate_sdk, name)
    except AttributeError as exc:  # pragma: no cover - version clarity
        raise RuntimeError(
            f"agent-simulate with `{name}` is required; upgrade simulate-sdk."
        ) from exc


def _suite_file_like_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        return resolved / "eval_suite.json"
    return resolved


def _agent_learning_suite_file_like_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        return resolved / "agent_learning_suite.json"
    return resolved


def _public_eval_suite_runner() -> Callable[[Mapping[str, Any], AgentCandidate], Any]:
    run_eval_suite = _simulate_sdk_attr("run_eval_suite")
    suite_path = Path.cwd() / "eval_suite.json"

    def run_suite(candidate_suite: Mapping[str, Any], candidate: AgentCandidate) -> Any:
        return run_eval_suite(candidate_suite, suite_path=suite_path)

    return run_suite


def _agent_learning_suite_attr(name: str) -> Any:
    try:
        from fi.alk import suite as agent_learning_suite
    except Exception as exc:  # pragma: no cover - optional dependency clarity
        raise RuntimeError(
            "agent-learning-kit is required for Agent Learning suite optimization."
        ) from exc
    try:
        return getattr(agent_learning_suite, name)
    except AttributeError as exc:  # pragma: no cover - version clarity
        raise RuntimeError(
            f"agent-learning-kit with `fi.alk.suite.{name}` is required."
        ) from exc


__all__ = [
    "EvalSuiteOptimizationProblem",
    "ManifestOptimizationProblem",
    "ManifestRunner",
    "ManifestScorer",
    "SimulateEvalSuiteOptimizationProblem",
    "SimulateManifestOptimizationProblem",
    "SimulateSuiteOptimizationProblem",
    "SuiteOptimizationProblem",
    "deep_merge",
    "optimize_agent_learning_suite",
    "optimize_agent_learning_suite_file",
    "optimize_eval_suite",
    "optimize_eval_suite_file",
    "optimize_simulate_manifest",
    "optimize_simulate_manifest_file",
    "problem_from_agent_learning_suite",
    "problem_from_agent_learning_suite_file",
    "problem_from_eval_suite",
    "problem_from_eval_suite_file",
    "problem_from_simulate_manifest",
    "problem_from_simulate_manifest_file",
]
