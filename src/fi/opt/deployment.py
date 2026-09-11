from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Optional

from pydantic import BaseModel, Field

from .targets import AgentCandidate
from .types import EvaluationResult, OptimizationResult


SCHEMA_VERSION = "agent-opt.deployment.v1"
PROMOTION_SCHEMA_VERSION = "agent-opt.promotion.v1"
ROLLBACK_SCHEMA_VERSION = "agent-opt.rollback.v1"
SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)


class AgentDeploymentExport(BaseModel):
    """Framework-specific deployment manifest for an optimized agent config."""

    schema_version: str = SCHEMA_VERSION
    framework: str
    target_name: Optional[str] = None
    candidate_id: Optional[str] = None
    layers: list[str] = Field(default_factory=list)
    final_score: Optional[float] = None
    config: dict[str, Any] = Field(default_factory=dict)
    patch: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    files: dict[str, Any] = Field(default_factory=dict)
    apply_steps: list[str] = Field(default_factory=list)
    redactions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


class PromotionMetricCheck(BaseModel):
    """One metric threshold check for a deployment promotion gate."""

    name: str
    observed: Optional[float] = None
    threshold: float
    passed: bool


class AgentPromotionCheck(BaseModel):
    """Result of evaluating whether a deployment manifest can be promoted."""

    schema_version: str = PROMOTION_SCHEMA_VERSION
    promotable: bool
    framework: str
    candidate_id: Optional[str] = None
    staging_score: float
    optimized_score: Optional[float] = None
    min_score: float
    max_score_drop: float
    score_delta: Optional[float] = None
    metric_checks: list[PromotionMetricCheck] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    deployment: AgentDeploymentExport
    evaluation_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


class RollbackObservation(BaseModel):
    """One live or replayed post-deployment evaluation observation."""

    index: int
    candidate_id: Optional[str] = None
    score: float
    passed: bool
    failures: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRollbackDecision(BaseModel):
    """Decision record for online rollback monitoring."""

    schema_version: str = ROLLBACK_SCHEMA_VERSION
    rollback_required: bool
    framework: str
    candidate_id: Optional[str] = None
    baseline_score: Optional[float] = None
    min_score: float
    max_score_drop: float
    window_size: int
    min_evaluations: int
    required_consecutive_failures: int
    failure_count: int
    consecutive_failure_count: int
    average_score: Optional[float] = None
    latest_score: Optional[float] = None
    score_delta: Optional[float] = None
    observations: list[RollbackObservation] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    rollback_steps: list[str] = Field(default_factory=list)
    deployment: AgentDeploymentExport
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_manifest(), sort_keys=True, indent=indent, default=str)


def export_agent_deployment(
    value: OptimizationResult | AgentCandidate | Mapping[str, Any],
    *,
    framework: str = "auto",
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentDeploymentExport:
    """
    Export an optimized agent candidate as a framework-specific deployment manifest.

    The export keeps the optimized config as the source of truth, redacts common
    secret-bearing fields, and adds framework apply hints without inventing
    values outside the candidate configuration.
    """

    candidate, final_score, result_metadata = _coerce_candidate(value)
    redactions: list[str] = []
    config = _redact_secrets(candidate.config, redactions=redactions)
    patch = _redact_secrets(candidate.patch, redactions=redactions)
    resolved_framework = _resolve_framework(
        framework,
        candidate=candidate,
        result_metadata=result_metadata,
    )
    runtime, files, apply_steps = _framework_export(
        resolved_framework,
        config=config,
        patch=patch,
    )
    export_metadata = {
        **result_metadata,
        **candidate.metadata,
        **dict(metadata or {}),
    }
    redacted_metadata = _redact_secrets(export_metadata, redactions=redactions)
    return AgentDeploymentExport(
        framework=resolved_framework,
        target_name=candidate.target_name,
        candidate_id=candidate.id,
        layers=list(candidate.layers),
        final_score=final_score,
        config=config,
        patch=patch,
        runtime=runtime,
        files=files,
        apply_steps=apply_steps,
        redactions=sorted(set(redactions)),
        metadata=redacted_metadata,
    )


def check_agent_deployment_rollback(
    value: AgentPromotionCheck | AgentDeploymentExport | OptimizationResult | AgentCandidate | Mapping[str, Any],
    *,
    live_evaluations: Optional[list[Any]] = None,
    simulation_evaluator: Any = None,
    evaluation_count: int = 1,
    baseline_score: Optional[float] = None,
    min_score: Optional[float] = None,
    max_score_drop: float = 0.05,
    required_metrics: Optional[Mapping[str, float]] = None,
    consecutive_failures: int = 2,
    min_evaluations: int = 2,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentRollbackDecision:
    """
    Monitor post-deployment evaluations and decide whether to roll back.

    Use explicit `live_evaluations` for production trace/eval streams, or pass a
    `simulation_evaluator` to replay the exported deployment config locally.
    Rollback is recommended only after enough observations have been collected
    and the trailing failure count reaches `consecutive_failures`.
    """

    if max_score_drop < 0:
        raise ValueError("max_score_drop must be non-negative.")
    if consecutive_failures < 1:
        raise ValueError("consecutive_failures must be at least 1.")
    if min_evaluations < 1:
        raise ValueError("min_evaluations must be at least 1.")
    if evaluation_count < 1:
        raise ValueError("evaluation_count must be at least 1.")

    deployment, promotion_baseline, promotion_min_score = _deployment_for_monitor(value)
    active_baseline = (
        baseline_score
        if baseline_score is not None
        else promotion_baseline
        if promotion_baseline is not None
        else deployment.final_score
    )
    active_min_score = (
        min_score
        if min_score is not None
        else promotion_min_score
        if promotion_min_score is not None
        else max(0.0, active_baseline - max_score_drop)
        if active_baseline is not None
        else 0.0
    )
    evaluations = _rollback_evaluations(
        deployment,
        live_evaluations=live_evaluations,
        simulation_evaluator=simulation_evaluator,
        evaluation_count=evaluation_count,
    )
    observations: list[RollbackObservation] = []
    active_required_metrics = {
        str(name): float(threshold)
        for name, threshold in dict(required_metrics or {}).items()
    }
    for index, evaluation in enumerate(evaluations, start=1):
        all_metrics = _extract_metric_scores(evaluation.metadata)
        metrics = _scope_observation_metrics(
            all_metrics,
            required_metrics=active_required_metrics,
        )
        observation_failures = _observation_failures(
            evaluation_score=evaluation.score,
            min_score=active_min_score,
            metrics=metrics,
            required_metrics=active_required_metrics,
        )
        observation_metadata: dict[str, Any] = {
            "diagnostic_metric_scope": (
                "required_metrics" if active_required_metrics else "all_metrics"
            )
        }
        if active_required_metrics:
            observation_metadata["all_metrics"] = all_metrics
            observation_metadata["required_metrics"] = active_required_metrics
        observations.append(
            RollbackObservation(
                index=index,
                candidate_id=evaluation.candidate.id,
                score=evaluation.score,
                passed=not observation_failures,
                failures=observation_failures,
                metrics=metrics,
                metadata=observation_metadata,
            )
        )

    scores = [item.score for item in observations]
    failure_count = sum(1 for item in observations if not item.passed)
    trailing_failures = 0
    for item in reversed(observations):
        if item.passed:
            break
        trailing_failures += 1

    failures: list[str] = []
    if len(observations) < min_evaluations:
        failures.append(
            f"only {len(observations)} observation(s), need at least {min_evaluations}"
        )
    if trailing_failures >= consecutive_failures and len(observations) >= min_evaluations:
        failures.append(
            f"{trailing_failures} consecutive failed observation(s) reached rollback threshold {consecutive_failures}"
        )
    rollback_required = (
        len(observations) >= min_evaluations
        and trailing_failures >= consecutive_failures
    )
    latest_score = scores[-1] if scores else None
    average_score = sum(scores) / len(scores) if scores else None
    score_delta = (
        latest_score - active_baseline
        if latest_score is not None and active_baseline is not None
        else None
    )
    redactions: list[str] = []
    check_metadata = _redact_secrets(dict(metadata or {}), redactions=redactions)
    if redactions:
        check_metadata["redactions"] = sorted(set(redactions))
    return AgentRollbackDecision(
        rollback_required=rollback_required,
        framework=deployment.framework,
        candidate_id=deployment.candidate_id,
        baseline_score=active_baseline,
        min_score=active_min_score,
        max_score_drop=max_score_drop,
        window_size=len(observations),
        min_evaluations=min_evaluations,
        required_consecutive_failures=consecutive_failures,
        failure_count=failure_count,
        consecutive_failure_count=trailing_failures,
        average_score=average_score,
        latest_score=latest_score,
        score_delta=score_delta,
        observations=observations,
        failures=failures,
        rollback_steps=_rollback_steps(deployment),
        deployment=deployment,
        metadata=check_metadata,
    )


def check_agent_deployment_promotion(
    value: AgentDeploymentExport | OptimizationResult | AgentCandidate | Mapping[str, Any],
    *,
    simulation_evaluator: Any = None,
    staging_evaluation: Any = None,
    staging_candidate: Optional[AgentCandidate] = None,
    min_score: Optional[float] = None,
    max_score_drop: float = 0.0,
    required_metrics: Optional[Mapping[str, float]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentPromotionCheck:
    """
    Run a staging promotion gate for an optimized deployment manifest.

    A promotion passes only when the staging score meets the requested minimum
    or optimized-score delta and all required metric thresholds are present and
    satisfied. If `simulation_evaluator` is supplied, it is run against the
    exported config using the same `SimulationEvaluator` contract used during
    optimization.
    """

    if max_score_drop < 0:
        raise ValueError("max_score_drop must be non-negative.")
    deployment = (
        value
        if isinstance(value, AgentDeploymentExport)
        else export_agent_deployment(value)
    )
    evaluation = _promotion_evaluation(
        deployment,
        simulation_evaluator=simulation_evaluator,
        staging_evaluation=staging_evaluation,
        staging_candidate=staging_candidate,
    )
    optimized_score = deployment.final_score
    active_min_score = (
        min_score
        if min_score is not None
        else max(0.0, optimized_score - max_score_drop)
        if optimized_score is not None
        else 0.0
    )
    failures: list[str] = []
    if evaluation.score < active_min_score:
        failures.append(
            f"staging score {evaluation.score:.4f} below required {active_min_score:.4f}"
        )

    observed_metrics = _extract_metric_scores(evaluation.metadata)
    metric_checks: list[PromotionMetricCheck] = []
    for name, threshold in dict(required_metrics or {}).items():
        observed = observed_metrics.get(name)
        passed = observed is not None and observed >= threshold
        if not passed:
            if observed is None:
                failures.append(f"required metric '{name}' missing from staging evidence")
            else:
                failures.append(
                    f"required metric '{name}' score {observed:.4f} below {threshold:.4f}"
                )
        metric_checks.append(
            PromotionMetricCheck(
                name=name,
                observed=observed,
                threshold=float(threshold),
                passed=passed,
            )
        )

    score_delta = (
        evaluation.score - optimized_score
        if optimized_score is not None
        else None
    )
    redactions: list[str] = []
    evaluation_metadata = _redact_secrets(evaluation.metadata, redactions=redactions)
    check_metadata = _redact_secrets(dict(metadata or {}), redactions=redactions)
    if redactions:
        check_metadata["redactions"] = sorted(set(redactions))
    return AgentPromotionCheck(
        promotable=not failures,
        framework=deployment.framework,
        candidate_id=evaluation.candidate.id,
        staging_score=evaluation.score,
        optimized_score=optimized_score,
        min_score=active_min_score,
        max_score_drop=max_score_drop,
        score_delta=score_delta,
        metric_checks=metric_checks,
        failures=failures,
        deployment=deployment,
        evaluation_metadata=evaluation_metadata,
        metadata=check_metadata,
    )


def _deployment_for_monitor(
    value: AgentPromotionCheck | AgentDeploymentExport | OptimizationResult | AgentCandidate | Mapping[str, Any],
) -> tuple[AgentDeploymentExport, Optional[float], Optional[float]]:
    if isinstance(value, AgentPromotionCheck):
        return value.deployment, value.staging_score, value.min_score
    deployment = (
        value
        if isinstance(value, AgentDeploymentExport)
        else export_agent_deployment(value)
    )
    return deployment, deployment.final_score, None


def _rollback_evaluations(
    deployment: AgentDeploymentExport,
    *,
    live_evaluations: Optional[list[Any]],
    simulation_evaluator: Any,
    evaluation_count: int,
) -> list[Any]:
    candidate = _candidate_from_deployment(deployment)
    if live_evaluations is not None:
        return [
            _coerce_staging_evaluation(item, candidate=candidate)
            for item in live_evaluations
        ]
    if simulation_evaluator is None:
        raise ValueError(
            "check_agent_deployment_rollback requires live_evaluations or simulation_evaluator."
        )
    evaluator = getattr(simulation_evaluator, "evaluate_candidate", None)
    if evaluator is None:
        raise ValueError("simulation_evaluator must expose evaluate_candidate(candidate).")
    return [
        _coerce_staging_evaluation(evaluator(candidate), candidate=candidate)
        for _ in range(evaluation_count)
    ]


def _observation_failures(
    *,
    evaluation_score: float,
    min_score: float,
    metrics: Mapping[str, float],
    required_metrics: Optional[Mapping[str, float]],
) -> list[str]:
    failures: list[str] = []
    if evaluation_score < min_score:
        failures.append(
            f"score {evaluation_score:.4f} below required {min_score:.4f}"
        )
    for name, threshold in dict(required_metrics or {}).items():
        observed = metrics.get(name)
        if observed is None:
            failures.append(f"metric '{name}' missing")
        elif observed < threshold:
            failures.append(
                f"metric '{name}' score {observed:.4f} below {threshold:.4f}"
            )
    return failures


def _scope_observation_metrics(
    metrics: Mapping[str, float],
    *,
    required_metrics: Mapping[str, float],
) -> dict[str, float]:
    if not required_metrics:
        return dict(metrics)
    required = set(required_metrics)
    return {
        name: value
        for name, value in dict(metrics).items()
        if name in required
    }


def _rollback_steps(deployment: AgentDeploymentExport) -> list[str]:
    framework_step = {
        "livekit": "Use LiveKit deployment rollback or redeploy the last promotable agent version.",
        "langgraph": "Restore the previous LangGraph app config and checkpointer/store wiring.",
        "langchain": "Restore the previous LangChain runnable/agent config and callback setup.",
        "openai_agents": "Restore the previous OpenAI Agents SDK RunConfig, session, handoff, guardrail, and tool setup.",
        "pipecat": "Restore the previous Pipecat pipeline processor and frame-capture config.",
        "browser_cua": "Restore the previous browser/CUA policy and trace-capture config.",
        "rag": "Restore the previous retriever, grounding, citation, and memory-write config.",
        "multi_agent": "Restore the previous multi-agent role, handoff, review, and reconciliation config.",
    }.get(
        deployment.framework,
        "Restore the previous application config or last promotable deployment manifest.",
    )
    return [
        "Stop promotion or remove traffic from the monitored candidate.",
        framework_step,
        "Replay the failing live traces through simulate-sdk and ai-evaluation to confirm the regression.",
        "Keep the rollback active until a new candidate passes staging promotion and live monitoring.",
    ]


def _coerce_candidate(
    value: OptimizationResult | AgentCandidate | Mapping[str, Any],
) -> tuple[AgentCandidate, Optional[float], dict[str, Any]]:
    if isinstance(value, OptimizationResult):
        candidate = value.best_candidate
        if not isinstance(candidate, AgentCandidate):
            raise TypeError("OptimizationResult.best_candidate must be an AgentCandidate.")
        return candidate, value.final_score, dict(value.metadata)
    if isinstance(value, AgentCandidate):
        return value, None, {}
    if isinstance(value, Mapping):
        candidate = AgentCandidate.from_config(
            dict(value),
            target_name="deployment-export",
            metadata={"kind": "deployment_export"},
        )
        return candidate, None, {}
    raise TypeError(
        "export_agent_deployment expects OptimizationResult, AgentCandidate, or config mapping."
    )


def _promotion_evaluation(
    deployment: AgentDeploymentExport,
    *,
    simulation_evaluator: Any,
    staging_evaluation: Any,
    staging_candidate: Optional[AgentCandidate],
) -> Any:
    if staging_evaluation is not None:
        return _coerce_staging_evaluation(
            staging_evaluation,
            candidate=staging_candidate or _candidate_from_deployment(deployment),
        )
    if simulation_evaluator is None:
        raise ValueError(
            "check_agent_deployment_promotion requires simulation_evaluator or staging_evaluation."
        )
    candidate = staging_candidate or _candidate_from_deployment(deployment)
    evaluator = getattr(simulation_evaluator, "evaluate_candidate", None)
    if evaluator is None:
        raise ValueError("simulation_evaluator must expose evaluate_candidate(candidate).")
    return _coerce_staging_evaluation(evaluator(candidate), candidate=candidate)


def _candidate_from_deployment(deployment: AgentDeploymentExport) -> AgentCandidate:
    return AgentCandidate.from_config(
        deployment.config,
        target_name=deployment.target_name,
        layers=deployment.layers,
        patch=deployment.patch,
        metadata={
            "kind": "deployment_staging",
            "deployment_framework": deployment.framework,
            "deployment_candidate_id": deployment.candidate_id,
        },
    )


def _coerce_staging_evaluation(value: Any, *, candidate: AgentCandidate) -> Any:
    from .targets import CandidateEvaluation

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
        "staging_evaluation must be CandidateEvaluation, EvaluationResult, int, or float."
    )


def _extract_metric_scores(metadata: Mapping[str, Any]) -> dict[str, float]:
    report = metadata.get("agent_report_evaluation")
    if report is None:
        return {}
    if hasattr(report, "model_dump"):
        report = report.model_dump()
    elif hasattr(report, "dict"):
        report = report.dict()
    if not isinstance(report, Mapping):
        return {}

    scores: dict[str, list[float]] = {}
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        averages = summary.get("metric_averages")
        if isinstance(averages, Mapping):
            for name, score in averages.items():
                coerced = _coerce_metric_score(score)
                if coerced is not None:
                    scores.setdefault(str(name), []).append(coerced)

    for case in report.get("cases", []) or []:
        if not isinstance(case, Mapping):
            continue
        for metric in case.get("metrics", []) or []:
            if not isinstance(metric, Mapping):
                continue
            name = metric.get("name")
            score = _coerce_metric_score(metric.get("score"))
            if name and score is not None:
                scores.setdefault(str(name), []).append(score)

    return {
        name: sum(values) / len(values)
        for name, values in scores.items()
        if values
    }


def _coerce_metric_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_framework(
    framework: str,
    *,
    candidate: AgentCandidate,
    result_metadata: Mapping[str, Any],
) -> str:
    if framework and framework != "auto":
        return _normalize_framework(framework)

    hints = _normalize_text(
        {
            "target": candidate.target_name,
            "layers": candidate.layers,
            "metadata": {**candidate.metadata, **dict(result_metadata)},
            "config_keys": list(candidate.config),
            "patch_keys": list(candidate.patch),
        }
    )
    checks = (
        ("langgraph", ("langgraph", "langgraph_stream_events", "checkpointer")),
        ("langchain", ("langchain", "stream_events")),
        ("openai_agents", ("openai_agents", "handoff", "guardrail_span")),
        ("livekit", ("livekit", "agent_session", "room_options")),
        ("pipecat", ("pipecat", "frame_pipeline", "frame_source")),
        ("browser_cua", ("browser", "cua", "playwright", "browser_use")),
        ("rag", ("retrieval", "retriever", "citation", "grounded")),
        ("multi_agent", ("multi_agent", "handoff", "reconciliation")),
    )
    for name, tokens in checks:
        if any(token in hints for token in tokens):
            return name
    return "generic"


def _normalize_framework(framework: str) -> str:
    normalized = framework.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "openai": "openai_agents",
        "openai_agent": "openai_agents",
        "browser": "browser_cua",
        "cua": "browser_cua",
        "retrieval": "rag",
        "retriever": "rag",
    }
    return aliases.get(normalized, normalized)


def _framework_export(
    framework: str,
    *,
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    if framework == "langgraph":
        return _langgraph_export(config, patch)
    if framework == "langchain":
        return _langchain_export(config, patch)
    if framework == "openai_agents":
        return _openai_agents_export(config, patch)
    if framework == "livekit":
        return _livekit_export(config, patch)
    if framework == "pipecat":
        return _pipecat_export(config, patch)
    if framework == "browser_cua":
        return _browser_cua_export(config, patch)
    if framework == "rag":
        return _rag_export(config, patch)
    if framework == "multi_agent":
        return _multi_agent_export(config, patch)
    return _generic_export(config, patch)


def _langgraph_export(
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    state_persistence = _get_path(config, "memory.state_persistence")
    runtime = {
        "stream_events": _get_path(config, "framework.events.source") == "langgraph_stream_events",
        "event_source": _get_path(config, "framework.events.source"),
        "nodes": _get_path(config, "langgraph.nodes", {}),
        "planner": _get_path(config, "planner", {}),
        "memory": {
            "state_persistence": state_persistence,
            "requires_checkpointer": state_persistence not in {None, "none", False},
        },
        "trace": _get_path(config, "framework.trace", {}),
    }
    files = _base_files(config, patch, runtime)
    files["langgraph.apply.json"] = {
        "compile": {
            "checkpointer": "required" if runtime["memory"]["requires_checkpointer"] else "optional",
            "store": "configure if long-term memory is enabled",
        },
        "stream": {
            "source": runtime["event_source"],
            "include_nodes": runtime["nodes"],
        },
    }
    steps = [
        "Apply config_patch.json to the application config used to build the LangGraph graph.",
        "Compile the graph with a checkpointer/store when runtime.memory.requires_checkpointer is true.",
        "Enable stream/event capture according to runtime.event_source before replaying or deploying.",
        "Run the same ai-evaluation agent-report metrics against a staging trace before promotion.",
    ]
    return runtime, files, steps


def _langchain_export(
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    runtime = {
        "stream_events": _get_path(config, "framework.events.source"),
        "callbacks": _get_path(config, "framework.callbacks", {}),
        "tools": _get_path(config, "tools", {}),
        "retrieval": _get_path(config, "retrieval", _get_path(config, "retriever", {})),
        "memory": _get_path(config, "memory", {}),
    }
    files = _base_files(config, patch, runtime)
    files["langchain.apply.json"] = {
        "streaming": runtime["stream_events"],
        "callbacks": runtime["callbacks"],
        "tool_config": runtime["tools"],
    }
    steps = [
        "Apply config_patch.json to the LangChain/LangGraph app config.",
        "Enable stream events or callbacks for the selected runtime evidence path.",
        "Wire optimized tool, retrieval, and memory settings into the runnable/agent factory.",
        "Replay a staging run through simulate-sdk and ai-evaluation before deployment.",
    ]
    return runtime, files, steps


def _openai_agents_export(
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    runtime = {
        "tracing": _get_path(config, "openai_agents.tracing", _get_path(config, "framework.trace", {})),
        "sessions": _get_path(config, "openai_agents.sessions", _get_path(config, "framework.sessions", {})),
        "handoffs": _get_path(config, "multi_agent.handoff", _get_path(config, "handoffs", {})),
        "guardrails": _get_path(config, "policy.guardrails", _get_path(config, "guardrails", {})),
        "tools": _get_path(config, "tools", {}),
    }
    files = _base_files(config, patch, runtime)
    files["openai_agents.apply.json"] = {
        "run_config": {
            "tracing": runtime["tracing"],
            "sessions": runtime["sessions"],
        },
        "agent": {
            "handoffs": runtime["handoffs"],
            "guardrails": runtime["guardrails"],
            "tools": runtime["tools"],
        },
    }
    steps = [
        "Apply config_patch.json to the OpenAI Agents SDK app config.",
        "Map runtime.tracing into RunConfig or trace processor setup without embedding secrets.",
        "Map runtime.sessions into the session implementation used by Runner runs.",
        "Update handoff, guardrail, and tool definitions, then replay a staging trace.",
    ]
    return runtime, files, steps


def _livekit_export(
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    runtime = {
        "agent_session": _get_path(config, "livekit.session", _get_path(config, "voice.session", {})),
        "room_options": _get_path(config, "livekit.room_options", _get_path(config, "voice.room_options", {})),
        "voice_pipeline": _get_path(config, "voice.pipeline", _get_path(config, "voice", {})),
        "turn_handling": _get_path(config, "voice.turn_handling", _get_path(config, "voice.endpointing", {})),
        "events": _get_path(config, "livekit.session_events", _get_path(config, "voice.trace", {})),
    }
    files = _base_files(config, patch, runtime)
    files["livekit.apply.json"] = {
        "AgentSession": runtime["agent_session"],
        "RoomOptions": runtime["room_options"],
        "voice_pipeline": runtime["voice_pipeline"],
    }
    steps = [
        "Apply config_patch.json to the LiveKit agent service config.",
        "Map runtime.agent_session and runtime.voice_pipeline into AgentSession construction.",
        "Map runtime.room_options into session.start room options.",
        "Configure secrets through LiveKit/deployment secret management, not this manifest.",
    ]
    return runtime, files, steps


def _pipecat_export(
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    runtime = {
        "pipeline": _get_path(config, "voice.pipeline", _get_path(config, "pipecat.pipeline", {})),
        "frames": _get_path(config, "pipecat.frames", _get_path(config, "voice.frames", {})),
        "audio": _get_path(config, "voice.audio", _get_path(config, "voice.media", {})),
        "timing": _get_path(config, "voice.timing", _get_path(config, "voice.timing_distribution", {})),
        "interruption": _get_path(config, "voice.interruption", _get_path(config, "voice.overlap", {})),
    }
    files = _base_files(config, patch, runtime)
    files["pipecat.apply.json"] = {
        "Pipeline": runtime["pipeline"],
        "frames": runtime["frames"],
        "audio": runtime["audio"],
    }
    steps = [
        "Apply config_patch.json to the Pipecat service config.",
        "Map runtime.pipeline to the Pipeline processor order and frame source.",
        "Enable frame/audio/timing capture before replaying the next call.",
        "Run the local Pipecat frame replay cookbook against the exported capture.",
    ]
    return runtime, files, steps


def _browser_cua_export(
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    runtime = {
        "trace": _get_path(config, "browser.trace", _get_path(config, "cua.trace", {})),
        "policy": _get_path(config, "browser.policy", _get_path(config, "policy", {})),
        "selectors": _get_path(config, "browser.selectors", {}),
        "actionability": _get_path(config, "browser.actionability", {}),
        "storage": {
            "storage_state": _get_path(config, "browser.storage_state"),
            "cookies": _get_path(config, "browser.cookies"),
            "local_storage": _get_path(config, "browser.local_storage"),
        },
    }
    files = _base_files(config, patch, runtime)
    files["browser_cua.apply.json"] = {
        "trace_capture": runtime["trace"],
        "policy": runtime["policy"],
        "selectors": runtime["selectors"],
    }
    steps = [
        "Apply config_patch.json to the browser/CUA runtime policy and trace-capture config.",
        "Enable screenshot, actionability, network, and storage evidence selected by runtime.trace.",
        "Keep domain and cross-origin policy enforcement in production, then replay a staging trace.",
    ]
    return runtime, files, steps


def _rag_export(
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    runtime = {
        "retrieval": _get_path(config, "retrieval", _get_path(config, "retriever", {})),
        "generation": _get_path(config, "generation", {}),
        "memory": _get_path(config, "memory", {}),
        "policy": _get_path(config, "policy", {}),
        "evaluation": _get_path(config, "evaluation", {}),
    }
    files = _base_files(config, patch, runtime)
    files["rag.apply.json"] = {
        "retrieval": runtime["retrieval"],
        "generation": runtime["generation"],
        "memory": runtime["memory"],
    }
    steps = [
        "Apply config_patch.json to retriever, generation, and memory-write config.",
        "Confirm citation, freshness, and grounded-generation settings in staging.",
        "Replay retrieved documents and memory traces through ai-evaluation before deployment.",
    ]
    return runtime, files, steps


def _multi_agent_export(
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    runtime = {
        "roles": _get_path(config, "multi_agent.roles", {}),
        "handoff": _get_path(config, "multi_agent.handoff", {}),
        "review": _get_path(config, "multi_agent.review", {}),
        "memory": _get_path(config, "memory", {}),
        "reconciliation": _get_path(config, "policy.reconciliation", {}),
    }
    files = _base_files(config, patch, runtime)
    files["multi_agent.apply.json"] = {
        "roles": runtime["roles"],
        "handoff": runtime["handoff"],
        "review": runtime["review"],
        "reconciliation": runtime["reconciliation"],
    }
    steps = [
        "Apply config_patch.json to the multi-agent orchestration config.",
        "Map handoff contracts, review gates, shared memory, and reconciliation into the runtime.",
        "Replay a captured multi-agent transcript before production promotion.",
    ]
    return runtime, files, steps


def _generic_export(
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    runtime = {"config": dict(config)}
    files = _base_files(config, patch, runtime)
    steps = [
        "Apply config_patch.json to the application config.",
        "Run the same SimulationEvaluator and ai-evaluation metrics against a staging run.",
        "Promote the manifest only after the staging score matches or exceeds the optimized score.",
    ]
    return runtime, files, steps


def _base_files(
    config: Mapping[str, Any],
    patch: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "config_patch.json": dict(patch),
        "optimized_config.json": dict(config),
        "runtime_config.json": dict(runtime),
    }


def _redact_secrets(value: Any, *, redactions: list[str], path: str = "") -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if _is_secret_key(str(key)):
                redactions.append(child_path)
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = _redact_secrets(
                    item,
                    redactions=redactions,
                    path=child_path,
                )
        return redacted
    if isinstance(value, list):
        return [
            _redact_secrets(item, redactions=redactions, path=f"{path}.{index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _redact_secrets(item, redactions=redactions, path=f"{path}.{index}")
            for index, item in enumerate(value)
        )
    return copy.deepcopy(value)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(marker in lowered for marker in SECRET_MARKERS)


def _get_path(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return copy.deepcopy(default)
    return copy.deepcopy(current)


def _normalize_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str).lower().replace("-", "_")
