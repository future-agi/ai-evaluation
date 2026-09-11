from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse

from ._facade import optional_module
from ._module_alias import install_lazy_module_aliases
from ._schema import public_payload

_EVAL_EXTRA = "evaluation"
AGENT_LEARNING_EVAL_KIND = "agent-learning.eval.v1"
AGENT_LEARNING_EVAL_OPTIMIZATION_KIND = "agent-learning.eval-optimization.v1"
AGENT_LEARNING_ARTIFACT_EVALUATION_KIND = "agent-learning.artifact-evaluation.v1"
AGENT_LEARNING_TASK_EVIDENCE_KIND = "agent-learning.task-evidence.v1"
AGENT_LEARNING_BEHAVIOR_ENTROPY_KIND = "agent-learning.eval.behavior-entropy.v1"
AGENT_LEARNING_COLLABORATIVE_COMPETENCE_KIND = (
    "agent-learning.eval.collaborative-competence.v1"
)
AGENT_LEARNING_REDTEAM_ADAPTIVE_LOOP_KIND = (
    "agent-learning.eval.redteam-adaptive-loop.v1"
)
AGENT_LEARNING_REDTEAM_ATTACK_EVOLUTION_KIND = (
    "agent-learning.eval.redteam-attack-evolution.v1"
)
AGENT_LEARNING_TASK_EVAL_SYNTHESIS_KIND = (
    "agent-learning.task-evaluation-synthesis.v1"
)

_FI_EVAL_EXPORT_NAMES = (
    "ASRAccuracy",
    "AnswerRefusal",
    "AudioQualityEvaluator",
    "AudioTranscriptionEvaluator",
    "BaseEvaluation",
    "BatchResult",
    "BiasDetection",
    "BleuScore",
    "CaptionHallucination",
    "ChunkAttribution",
    "ChunkResult",
    "ChunkUtilization",
    "ClinicallyInappropriateTone",
    "Completeness",
    "ContainsCode",
    "ContainsValidLink",
    "ContentModeration",
    "ContentSafety",
    "ContextAdherence",
    "ContextRelevance",
    "ConversationCoherence",
    "ConversationResolution",
    "CulturalSensitivity",
    "CustomerAgentClarificationSeeking",
    "CustomerAgentContextRetention",
    "CustomerAgentConversationQuality",
    "CustomerAgentHumanEscalation",
    "CustomerAgentInterruptionHandling",
    "CustomerAgentLanguageHandling",
    "CustomerAgentLoopDetection",
    "CustomerAgentObjectionHandling",
    "CustomerAgentPromptConformance",
    "CustomerAgentQueryHandling",
    "CustomerAgentTerminationHandling",
    "DataPrivacyCompliance",
    "DetectHallucination",
    "DetectHallucinationMissingInfo",
    "EarlyStopPolicy",
    "EarlyStopReason",
    "EvalBuilder",
    "EvalResult",
    "EvalTemplate",
    "EvalTemplateManager",
    "EvaluateFunctionCalling",
    "Evaluator",
    "Execution",
    "ExecutionError",
    "ExecutionMode",
    "FactualAccuracy",
    "FrameworkEvaluator",
    "FuzzyMatch",
    "GroundTruthMatch",
    "Groundedness",
    "ImageInstructionAdherence",
    "IsCompliant",
    "IsConcise",
    "IsEmail",
    "IsFactuallyConsistent",
    "IsGoodSummary",
    "IsHarmfulAdvice",
    "IsHelpful",
    "IsInformalTone",
    "IsJson",
    "IsPolite",
    "LLMFunctionCalling",
    "NoAgeBias",
    "NoApologies",
    "NoGenderBias",
    "NoHarmfulTherapeuticGuidance",
    "NoLLMReference",
    "NoOpenAIReference",
    "NoRacialBias",
    "OCREvaluation",
    "OneLine",
    "PII",
    "PromptAdherence",
    "PromptInjection",
    "PromptInstructionAdherence",
    "Protect",
    "ProtectFlash",
    "Ranking",
    "Sexist",
    "StreamingConfig",
    "StreamingEvalResult",
    "StreamingEvaluator",
    "StreamingState",
    "SummaryQuality",
    "SyntheticImageEvaluator",
    "TTSAccuracy",
    "TaskCompletion",
    "TextToSQL",
    "Tone",
    "Toxicity",
    "TranslationAccuracy",
    "Turing",
    "async_evaluator",
    "blocking_evaluator",
    "custom_eval",
    "distributed_evaluator",
    "evaluate",
    "list_evaluations",
    "protect",
    "register_current_span",
    "register_evaluation",
    "resilient_evaluator",
    "simple_eval",
)

_AUTOEVAL_EXPORT_NAMES = (
    "AppCategory",
    "RiskLevel",
    "DomainSensitivity",
    "AppRequirement",
    "AppAnalysis",
    "AutoEvalResult",
    "EvalConfig",
    "ScannerConfig",
    "AutoEvalConfig",
    "AutoEvalPipeline",
    "register_eval_class",
    "register_scanner_class",
    "get_template",
    "list_templates",
    "get_template_names",
    "TEMPLATES",
    "AppAnalyzer",
    "EvalRecommender",
    "RuleBasedAnalyzer",
    "export_yaml",
    "export_json",
    "load_yaml",
    "load_json",
    "load_config",
    "to_yaml_string",
    "to_json_string",
    "from_yaml_string",
    "from_json_string",
    "InteractiveConfigurator",
    "InteractiveSession",
    "ClarificationQuestion",
)

_LOCAL_EVAL_EXPORT_NAMES = (
    "RoutingMode",
    "LOCAL_CAPABLE_METRICS",
    "can_run_locally",
    "select_routing_mode",
    "LocalMetricRegistry",
    "get_registry",
    "LocalEvaluator",
    "LocalEvaluatorConfig",
    "LocalEvaluationResult",
    "HybridEvaluator",
    "LocalLLMConfig",
    "OllamaLLM",
    "LocalLLMFactory",
)

_STREAMING_EXPORT_NAMES = (
    "ChunkResult",
    "EarlyStopCondition",
    "EarlyStopReason",
    "StreamingConfig",
    "StreamingEvalResult",
    "StreamingState",
    "BufferState",
    "ChunkBuffer",
    "EarlyStopPolicy",
    "PolicyState",
    "EvalSpec",
    "StreamingEvaluator",
    "toxicity_scorer",
    "safety_scorer",
    "pii_scorer",
    "jailbreak_scorer",
    "coherence_scorer",
    "quality_scorer",
    "safety_composite_scorer",
    "quality_composite_scorer",
    "create_keyword_scorer",
    "create_pattern_scorer",
    "CompositeScorer",
)

_METRIC_EXPORT_NAMES = (
    "AggregatedMetric",
    "BLEUScore",
    "ROUGEScore",
    "LevenshteinSimilarity",
    "EmbeddingSimilarity",
    "NumericSimilarity",
    "SemanticListContains",
    "RecallScore",
    "Regex",
    "Contains",
    "ContainsAny",
    "ContainsAll",
    "ContainsNone",
    "Equals",
    "StartsWith",
    "EndsWith",
    "LengthLessThan",
    "LengthGreaterThan",
    "LengthBetween",
    "ContainsEmail",
    "ContainsLink",
    "JsonSchema",
    "ContainsJson",
    "CustomLLMJudge",
)

_AGENT_METRIC_EXPORT_NAMES = (
    "AgentReportEvalConfig",
    "AgentReportMetricResult",
    "AgentReportCaseResult",
    "AgentReportEvaluation",
    "AgentTrajectoryInput",
    "AgentStep",
    "ToolCall",
    "TaskDefinition",
    "TrajectoryAnalysis",
    "StepEfficiency",
    "ToolSelectionAccuracy",
    "TrajectoryScore",
    "GoalProgress",
    "ActionSafety",
    "ReasoningQuality",
    "analyze_domain_package_registry_coverage",
    "diff_domain_package_registries",
    "generate_domain_package_registry_fixtures",
    "generate_domain_package_registry_mutation_pack",
    "normalize_agent_report",
    "replay_domain_package_registry",
    "select_domain_package_registry_replay_pack",
    "validate_domain_package_registry",
)

_RAG_METRIC_EXPORT_NAMES = (
    "RAGInput",
    "RAGRetrievalInput",
    "RAGRankingInput",
    "ContextRecall",
    "ContextPrecision",
    "ContextEntityRecall",
    "NoiseSensitivity",
    "NDCG",
    "MRR",
    "AnswerRelevancy",
    "ContextUtilization",
    "RAGFaithfulness",
    "MultiHopReasoning",
    "SourceAttribution",
    "RAGScore",
    "RAGScoreDetailed",
)

_STRUCTURED_METRIC_EXPORT_NAMES = (
    "ValidationMode",
    "JSONInput",
    "PydanticInput",
    "YAMLInput",
    "StructuredInput",
    "ValidationError",
    "ValidationResult",
    "JSONValidator",
    "PydanticValidator",
    "YAMLValidator",
    "JSONValidation",
    "JSONSyntaxOnly",
    "SchemaCompliance",
    "TypeCompliance",
    "FieldCompleteness",
    "RequiredFieldsOnly",
    "FieldCoverage",
    "HierarchyScore",
    "TreeEditDistance",
    "StructuredOutputScore",
    "QuickStructuredCheck",
)

_HALLUCINATION_EXPORT_NAMES = (
    "HallucinationInput",
    "ClaimExtractionInput",
    "FactualConsistencyInput",
    "Claim",
    "NLIResult",
    "HallucinationResult",
    "Faithfulness",
    "ClaimSupport",
    "FactualConsistency",
    "ContradictionDetection",
    "HallucinationScore",
    "NLILabel",
    "check_entailment",
    "check_contradiction",
    "HallucinationSentinel",
    "HallucinationDetector",
)

_EVAL_EXPORTS = {name: "fi.evals" for name in _FI_EVAL_EXPORT_NAMES}
_EVAL_EXPORTS.update({name: "fi.evals.autoeval" for name in _AUTOEVAL_EXPORT_NAMES})
_EVAL_EXPORTS.update({name: "fi.evals.local" for name in _LOCAL_EVAL_EXPORT_NAMES})
_EVAL_EXPORTS.update({name: "fi.evals.streaming" for name in _STREAMING_EXPORT_NAMES})
_EVAL_EXPORTS["AgentReportEvaluator"] = "fi.evals.metrics.agents"
for _name in _METRIC_EXPORT_NAMES:
    _EVAL_EXPORTS.setdefault(_name, "fi.evals.metrics")
for _name in _AGENT_METRIC_EXPORT_NAMES:
    _EVAL_EXPORTS.setdefault(_name, "fi.evals.metrics.agents")
for _name in _RAG_METRIC_EXPORT_NAMES:
    _EVAL_EXPORTS.setdefault(_name, "fi.evals.metrics")
for _name in _STRUCTURED_METRIC_EXPORT_NAMES:
    _EVAL_EXPORTS.setdefault(_name, "fi.evals.metrics")
for _name in _HALLUCINATION_EXPORT_NAMES:
    _EVAL_EXPORTS.setdefault(_name, "fi.evals.metrics.hallucination")

_EVAL_SUBMODULE_ALIASES = {
    "autoeval": "fi.evals.autoeval",
    "cli": "fi.cli",
    "cli.main": "fi.cli.main",
    "core": "fi.evals.core",
    "core.prompt_generator": "fi.evals.core.prompt_generator",
    "feedback": "fi.evals.feedback",
    "framework": "fi.evals.framework",
    "framework.backends": "fi.evals.framework.backends",
    "framework.backends.base": "fi.evals.framework.backends.base",
    "framework.backends.thread_pool": "fi.evals.framework.backends.thread_pool",
    "framework.context": "fi.evals.framework.context",
    "framework.enrichment": "fi.evals.framework.enrichment",
    "framework.evaluator": "fi.evals.framework.evaluator",
    "framework.evaluators": "fi.evals.framework.evaluators",
    "framework.evaluators.blocking": "fi.evals.framework.evaluators.blocking",
    "framework.evaluators.non_blocking": "fi.evals.framework.evaluators.non_blocking",
    "framework.registry": "fi.evals.framework.registry",
    "framework.resilience": "fi.evals.framework.resilience",
    "framework.resilience.retry": "fi.evals.framework.resilience.retry",
    "guardrails": "fi.evals.guardrails",
    "guardrails.backends": "fi.evals.guardrails.backends",
    "guardrails.backends.base": "fi.evals.guardrails.backends.base",
    "guardrails.scanners": "fi.evals.guardrails.scanners",
    "guardrails.scanners.base": "fi.evals.guardrails.scanners.base",
    "guardrails.scanners.code_injection": "fi.evals.guardrails.scanners.code_injection",
    "guardrails.scanners.invisible_chars": "fi.evals.guardrails.scanners.invisible_chars",
    "guardrails.scanners.jailbreak": "fi.evals.guardrails.scanners.jailbreak",
    "guardrails.scanners.language": "fi.evals.guardrails.scanners.language",
    "guardrails.scanners.regex": "fi.evals.guardrails.scanners.regex",
    "guardrails.scanners.secrets": "fi.evals.guardrails.scanners.secrets",
    "guardrails.scanners.topics": "fi.evals.guardrails.scanners.topics",
    "llm": "fi.evals.llm",
    "local": "fi.evals.local",
    "metrics": "fi.evals.metrics",
    "metrics.agents": "fi.evals.metrics.agents",
    "metrics.agents.metrics": "fi.evals.metrics.agents.metrics",
    "metrics.agents.report": "fi.evals.metrics.agents.report",
    "metrics.agents.types": "fi.evals.metrics.agents.types",
    "metrics.base_metric": "fi.evals.metrics.base_metric",
    "metrics.code_security": "fi.evals.metrics.code_security",
    "metrics.function_calling": "fi.evals.metrics.function_calling",
    "metrics.hallucination": "fi.evals.metrics.hallucination",
    "metrics.llm_as_judges": "fi.evals.metrics.llm_as_judges",
    "metrics.rag": "fi.evals.metrics.rag",
    "metrics.structured": "fi.evals.metrics.structured",
    "metrics.structured.json_validation": "fi.evals.metrics.structured.json_validation",
    "otel": "fi.evals.otel",
    "streaming": "fi.evals.streaming",
}
_EVAL_PACKAGE_ALIASES = {
    alias
    for alias in _EVAL_SUBMODULE_ALIASES
    if "." not in alias or any(
        child.startswith(f"{alias}.") for child in _EVAL_SUBMODULE_ALIASES
    )
}

install_lazy_module_aliases(
    __name__,
    _EVAL_SUBMODULE_ALIASES,
    package_aliases=_EVAL_PACKAGE_ALIASES,
)


def _evals() -> Any:
    return optional_module("fi.evals", _EVAL_EXTRA)


def _agent_metrics() -> Any:
    return optional_module("fi.evals.metrics.agents", _EVAL_EXTRA)


def _suite() -> Any:
    return optional_module("fi.simulate.suite", "simulate")


def evaluate(*args: Any, **kwargs: Any) -> Any:
    return _evals().evaluate(*args, **kwargs)


def evaluate_agent_report(
    report: Any,
    config: Optional[Mapping[str, Any]] = None,
    *,
    threshold: float = 0.7,
) -> Any:
    return _agent_metrics().evaluate_agent_report(
        report,
        config=config,
        threshold=threshold,
    )


def behavior_entropy_report(
    report: Any,
    config: Optional[Mapping[str, Any]] = None,
    *,
    threshold: float = 0.7,
    min_score: float = 0.9,
) -> dict[str, Any]:
    """Return a local behavior-entropy artifact for agent trajectories."""

    eval_config = dict(config or {})
    weights = dict(eval_config.get("metric_weights") or {})
    weights.setdefault("behavior_entropy_quality", 1.0)
    eval_config["metric_weights"] = weights
    evaluation = _plain(
        evaluate_agent_report(report, config=eval_config, threshold=threshold)
    )
    cases = _as_list(evaluation.get("cases"))
    case_metrics: list[dict[str, Any]] = []
    for case in cases:
        metrics = _as_list(_as_mapping(case).get("metrics"))
        metric = next(
            (
                _as_mapping(item)
                for item in metrics
                if _as_mapping(item).get("name") == "behavior_entropy_quality"
            ),
            {},
        )
        if metric:
            case_metrics.append(
                {
                    "case_index": _as_mapping(case).get("index"),
                    "score": float(metric.get("score") or 0.0),
                    "reason": metric.get("reason", ""),
                    "details": _as_mapping(metric.get("details")),
                }
            )
    score = (
        sum(item["score"] for item in case_metrics) / len(case_metrics)
        if case_metrics
        else 0.0
    )
    failed = [item for item in case_metrics if item["score"] < min_score]
    payload = {
        "kind": AGENT_LEARNING_BEHAVIOR_ENTROPY_KIND,
        "status": "passed" if not failed and score >= min_score else "failed",
        "score": round(score, 4),
        "threshold": float(min_score),
        "case_count": len(case_metrics),
        "failed_case_count": len(failed),
        "cases": case_metrics,
        "summary": {
            "evaluation_score": evaluation.get("score"),
            "evaluation_passed": evaluation.get("passed"),
            "metric": "behavior_entropy_quality",
        },
        "research_sources": [
            {
                "id": "2606.05872",
                "title": "Entropy-Based Evaluation of AI Agents: A Lightweight Framework for Measuring Behavioral Patterns",
                "source": "arxiv:2606.05872",
                "url": "https://arxiv.org/abs/2606.05872",
                "used_for": (
                    "local behavior-pattern scoring across actions, tools, "
                    "trajectory entropy, information gain, and loop rate"
                ),
            }
        ],
        "metadata": {
            "source": "fi.alk.evals.behavior_entropy_report",
            "local_only": True,
            "requires_external_service": False,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return public_payload(payload, kind=AGENT_LEARNING_BEHAVIOR_ENTROPY_KIND)


def collaborative_competence_report(
    report: Any,
    config: Optional[Mapping[str, Any]] = None,
    *,
    threshold: float = 0.7,
    min_score: float = 0.9,
) -> dict[str, Any]:
    """Return a local collaborative-competence artifact for multi-agent traces."""

    eval_config = dict(config or {})
    weights = dict(eval_config.get("metric_weights") or {})
    weights.setdefault("collaborative_competence_quality", 1.0)
    eval_config["metric_weights"] = weights
    evaluation = _plain(
        evaluate_agent_report(report, config=eval_config, threshold=threshold)
    )
    cases = _as_list(evaluation.get("cases"))
    case_metrics: list[dict[str, Any]] = []
    for case in cases:
        metrics = _as_list(_as_mapping(case).get("metrics"))
        metric = next(
            (
                _as_mapping(item)
                for item in metrics
                if _as_mapping(item).get("name") == "collaborative_competence_quality"
            ),
            {},
        )
        if metric:
            case_metrics.append(
                {
                    "case_index": _as_mapping(case).get("index"),
                    "score": float(metric.get("score") or 0.0),
                    "reason": metric.get("reason", ""),
                    "details": _as_mapping(metric.get("details")),
                }
            )
    score = (
        sum(item["score"] for item in case_metrics) / len(case_metrics)
        if case_metrics
        else 0.0
    )
    failed = [item for item in case_metrics if item["score"] < min_score]
    payload = {
        "kind": AGENT_LEARNING_COLLABORATIVE_COMPETENCE_KIND,
        "status": "passed" if not failed and score >= min_score else "failed",
        "score": round(score, 4),
        "threshold": float(min_score),
        "case_count": len(case_metrics),
        "failed_case_count": len(failed),
        "cases": case_metrics,
        "summary": {
            "evaluation_score": evaluation.get("score"),
            "evaluation_passed": evaluation.get("passed"),
            "metric": "collaborative_competence_quality",
        },
        "research_sources": [
            {
                "id": "2606.06399",
                "title": "CollabSim: A CSCW-Grounded Methodology for Investigating Collaborative Competence of LLM Agents through Controlled Multi-Agent Experiments",
                "source": "arxiv:2606.06399",
                "url": "https://arxiv.org/abs/2606.06399",
            },
            {
                "id": "2606.06388",
                "title": "Humans' ALMANAC: A Human Collaboration Dataset of Action-Level Mental Model Annotations for Agent Collaboration",
                "source": "arxiv:2606.06388",
                "url": "https://arxiv.org/abs/2606.06388",
            },
            {
                "id": "2606.05985",
                "title": "Beyond Alignment: Value Diversity as a Collective Property in Multicultural Agent Systems",
                "source": "arxiv:2606.05985",
                "url": "https://arxiv.org/abs/2606.05985",
            },
            {
                "id": "2606.05670",
                "title": "Do More Agents Help? Controlled and Protocol-Aligned Evaluation of LLM Agent Workflows",
                "source": "arxiv:2606.05670",
                "url": "https://arxiv.org/abs/2606.05670",
            },
            {
                "id": "2606.05704",
                "title": "Critic-Guided Heterogeneous Multi-Agent Reasoning for Reliable Mathematical Problem Solving",
                "source": "arxiv:2606.05704",
                "url": "https://arxiv.org/abs/2606.05704",
            },
            {
                "id": "2606.06025",
                "title": "EGTR-Review: Efficient Evidence-Grounded Scientific Peer Review Generation via Multi-Agent Teacher Distillation",
                "source": "arxiv:2606.06025",
                "url": "https://arxiv.org/abs/2606.06025",
            },
        ],
        "metadata": {
            "source": "fi.alk.evals.collaborative_competence_report",
            "local_only": True,
            "requires_external_service": False,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return public_payload(payload, kind=AGENT_LEARNING_COLLABORATIVE_COMPETENCE_KIND)


def redteam_adaptive_loop_report(
    report: Any,
    config: Optional[Mapping[str, Any]] = None,
    *,
    threshold: float = 0.7,
    min_score: float = 0.9,
) -> dict[str, Any]:
    """Return a local adaptive-loop artifact for red-team campaigns."""

    eval_config = dict(config or {})
    weights = dict(eval_config.get("metric_weights") or {})
    weights.setdefault("red_team_adaptive_loop_quality", 1.0)
    eval_config["metric_weights"] = weights
    evaluation = _plain(
        evaluate_agent_report(report, config=eval_config, threshold=threshold)
    )
    cases = _as_list(evaluation.get("cases"))
    case_metrics: list[dict[str, Any]] = []
    for case in cases:
        metrics = _as_list(_as_mapping(case).get("metrics"))
        metric = next(
            (
                _as_mapping(item)
                for item in metrics
                if _as_mapping(item).get("name")
                == "red_team_adaptive_loop_quality"
            ),
            {},
        )
        if metric:
            case_metrics.append(
                {
                    "case_index": _as_mapping(case).get("index"),
                    "score": float(metric.get("score") or 0.0),
                    "reason": metric.get("reason", ""),
                    "details": _as_mapping(metric.get("details")),
                }
            )
    score = (
        sum(item["score"] for item in case_metrics) / len(case_metrics)
        if case_metrics
        else 0.0
    )
    failed = [item for item in case_metrics if item["score"] < min_score]
    payload = {
        "kind": AGENT_LEARNING_REDTEAM_ADAPTIVE_LOOP_KIND,
        "status": "passed" if not failed and score >= min_score else "failed",
        "score": round(score, 4),
        "threshold": float(min_score),
        "case_count": len(case_metrics),
        "failed_case_count": len(failed),
        "cases": case_metrics,
        "summary": {
            "evaluation_score": evaluation.get("score"),
            "evaluation_passed": evaluation.get("passed"),
            "metric": "red_team_adaptive_loop_quality",
        },
        "research_sources": [
            {
                "id": "2605.09684",
                "title": "MonitoringBench: Semi-Automated Red-Teaming for Agent Monitoring",
                "source": "arxiv:2605.09684",
                "url": "https://arxiv.org/abs/2605.09684",
                "used_for": (
                    "strategy/execution/refinement decomposition and monitor "
                    "calibration evidence"
                ),
            },
            {
                "id": "2603.20925",
                "title": "Profit is the Red Team: Stress-Testing Agents in Strategic Economic Interactions",
                "source": "arxiv:2603.20925",
                "url": "https://arxiv.org/abs/2603.20925",
                "used_for": (
                    "outcome-feedback and adaptive opponent pressure signals"
                ),
            },
            {
                "id": "2601.10971",
                "title": "AJAR: Adaptive Jailbreak Architecture for Red-teaming",
                "source": "arxiv:2601.10971",
                "url": "https://arxiv.org/abs/2601.10971",
                "used_for": "rollback-enabled transcript repair and tool-aware loops",
            },
            {
                "id": "2605.04808",
                "title": "DecodingTrust-Agent Platform (DTap): A Controllable and Interactive Red-Teaming Platform for AI Agents",
                "source": "arxiv:2605.04808",
                "url": "https://arxiv.org/abs/2605.04808",
                "used_for": "multi-vector controllable agent red-team evidence",
            },
        ],
        "metadata": {
            "source": "fi.alk.evals.redteam_adaptive_loop_report",
            "local_only": True,
            "requires_external_service": False,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return public_payload(payload, kind=AGENT_LEARNING_REDTEAM_ADAPTIVE_LOOP_KIND)


def redteam_attack_evolution_report(
    report: Any,
    config: Optional[Mapping[str, Any]] = None,
    *,
    threshold: float = 0.7,
    min_score: float = 0.9,
) -> dict[str, Any]:
    """Return a local attack-evolution artifact for red-team reports."""

    eval_config = dict(config or {})
    weights = dict(eval_config.get("metric_weights") or {})
    weights.setdefault("red_team_attack_evolution_quality", 1.0)
    eval_config["metric_weights"] = weights
    evaluation = _plain(
        evaluate_agent_report(report, config=eval_config, threshold=threshold)
    )
    cases = _as_list(evaluation.get("cases"))
    case_metrics: list[dict[str, Any]] = []
    for case in cases:
        metrics = _as_list(_as_mapping(case).get("metrics"))
        metric = next(
            (
                _as_mapping(item)
                for item in metrics
                if _as_mapping(item).get("name")
                == "red_team_attack_evolution_quality"
            ),
            {},
        )
        if metric:
            case_metrics.append(
                {
                    "case_index": _as_mapping(case).get("index"),
                    "score": float(metric.get("score") or 0.0),
                    "reason": metric.get("reason", ""),
                    "details": _as_mapping(metric.get("details")),
                }
            )
    score = (
        sum(item["score"] for item in case_metrics) / len(case_metrics)
        if case_metrics
        else 0.0
    )
    failed = [item for item in case_metrics if item["score"] < min_score]
    payload = {
        "kind": AGENT_LEARNING_REDTEAM_ATTACK_EVOLUTION_KIND,
        "status": "passed" if not failed and score >= min_score else "failed",
        "score": round(score, 4),
        "threshold": float(min_score),
        "case_count": len(case_metrics),
        "failed_case_count": len(failed),
        "cases": case_metrics,
        "summary": {
            "evaluation_score": evaluation.get("score"),
            "evaluation_passed": evaluation.get("passed"),
            "metric": "red_team_attack_evolution_quality",
        },
        "research_sources": [
            {
                "id": "2603.22341",
                "title": (
                    "T-MAP: Red-Teaming LLM Agents with Trajectory-aware "
                    "Evolutionary Search"
                ),
                "source": "arxiv:2603.22341",
                "url": "https://arxiv.org/abs/2603.22341",
                "used_for": (
                    "trajectory-aware mutation lineage and tool-action "
                    "realization evidence"
                ),
            },
            {
                "id": "2601.13518",
                "title": "AgenticRed: Evolving Agentic Systems for Red-Teaming",
                "source": "arxiv:2601.13518",
                "url": "https://arxiv.org/abs/2601.13518",
                "used_for": (
                    "generational knowledge, evolutionary selection, and "
                    "system-level red-team design"
                ),
            },
            {
                "id": "2602.16901",
                "title": "AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks",
                "source": "arxiv:2602.16901",
                "url": "https://arxiv.org/abs/2602.16901",
                "used_for": (
                    "long-horizon attack categories and replayable agentic "
                    "environment evidence"
                ),
            },
            {
                "id": "2601.10971",
                "title": "AJAR: Adaptive Jailbreak Architecture for Red-teaming",
                "source": "arxiv:2601.10971",
                "url": "https://arxiv.org/abs/2601.10971",
                "used_for": (
                    "rollback-enabled transcript repair, strategy switching, "
                    "and verifier-oriented orchestration"
                ),
            },
            {
                "id": "2605.06486",
                "title": "Autonomous Adversary: Red-Teaming in the age of LLM",
                "source": "arxiv:2605.06486",
                "url": "https://arxiv.org/abs/2605.06486",
                "used_for": (
                    "ordered task-chain validation predicates and controlled "
                    "feedback loops"
                ),
            },
        ],
        "metadata": {
            "source": "fi.alk.evals.redteam_attack_evolution_report",
            "local_only": True,
            "requires_external_service": False,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return public_payload(payload, kind=AGENT_LEARNING_REDTEAM_ATTACK_EVOLUTION_KIND)


def build_task_evaluation_config(
    *,
    task_description: str,
    expected_result: Optional[str] = None,
    success_criteria: Sequence[str] = (),
    required_tools: Sequence[str] = (),
    available_tools: Sequence[str] = (),
    forbidden_patterns: Sequence[str] = (),
    sensitive_patterns: Sequence[str] = (),
    metric_weights: Optional[Mapping[str, float]] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build an agent-report evaluation config for arbitrary task evidence."""

    if not task_description:
        raise ValueError("task_description is required")
    config: dict[str, Any] = {
        "task_description": str(task_description),
    }
    if expected_result is not None:
        config["expected_result"] = str(expected_result)
    if success_criteria:
        config["success_criteria"] = _unique_strings(success_criteria)
    if required_tools:
        config["required_tools"] = _unique_strings(required_tools)
    if available_tools:
        config["available_tools"] = _unique_strings(available_tools)
    if forbidden_patterns:
        config["forbidden_patterns"] = _unique_strings(forbidden_patterns)
    if sensitive_patterns:
        config["sensitive_patterns"] = _unique_strings(sensitive_patterns)
    if metric_weights:
        config["metric_weights"] = {
            str(key): float(value)
            for key, value in dict(metric_weights).items()
        }
    config.update({str(key): _plain(value) for key, value in extra.items()})
    return config


def synthesize_task_evaluation_config(
    evidence: Mapping[str, Any],
    *,
    task_description: Optional[str] = None,
    expected_result: Optional[str] = None,
    success_criteria: Sequence[str] = (),
    required_tools: Sequence[str] = (),
    available_tools: Sequence[str] = (),
    forbidden_patterns: Sequence[str] = (),
    sensitive_patterns: Sequence[str] = (),
    require_source_grounding: Optional[bool] = None,
    metric_weights: Optional[Mapping[str, float]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Infer an agent-report evaluation config from arbitrary task evidence.

    This is intentionally deterministic and local-first. It derives the
    task description, expected result, success criteria, tool requirements,
    state-backed metric weights, and source-grounding switches from the
    evidence shape so saved framework/world/task artifacts can be evaluated
    without a hand-authored config.
    """

    source = _as_mapping(evidence)
    if not source:
        raise ValueError("evidence is required")
    environment_state = _task_evidence_environment_state(source)
    tool_names = _task_evidence_tool_names(source)
    observed_tools = _unique_strings([*required_tools, *tool_names])
    available = _unique_strings([*available_tools, *observed_tools])
    description = str(
        task_description
        or source.get("task_description")
        or source.get("task")
        or _as_mapping(source.get("metadata")).get("task")
        or source.get("input")
        or source.get("prompt")
        or source.get("question")
        or source.get("id")
        or source.get("name")
        or "Evaluate the provided task evidence."
    )
    expected = (
        expected_result
        if expected_result is not None
        else _first_present(
            source,
            "expected_result",
            "expected",
            "expected_output",
            "output",
            "result",
            "final_result",
            "answer",
            default=None,
        )
    )
    synthesized_criteria = _task_evaluation_success_criteria(
        source,
        expected_result=expected,
        environment_state=environment_state,
        tool_names=observed_tools,
        explicit_criteria=success_criteria,
    )
    synthesized_forbidden = _task_evaluation_forbidden_patterns(
        source,
        environment_state=environment_state,
        explicit_patterns=forbidden_patterns,
    )
    synthesized_sensitive = _unique_strings(
        [
            *sensitive_patterns,
            *_as_list(source.get("sensitive_patterns")),
        ]
    )
    synthesized_grounding = (
        bool(require_source_grounding)
        if require_source_grounding is not None
        else _task_evidence_has_retrieval_state(environment_state)
    )
    weights = _task_evaluation_metric_weights(
        environment_state,
        required_tools=observed_tools,
        forbidden_patterns=synthesized_forbidden,
        require_source_grounding=synthesized_grounding,
        overrides=metric_weights,
    )
    synthesis = {
        "kind": AGENT_LEARNING_TASK_EVAL_SYNTHESIS_KIND,
        "source": "fi.alk.evals.synthesize_task_evaluation_config",
        "local_only": True,
        "requires_external_service": False,
        "evidence_keys": sorted(str(key) for key in source),
        "environment_state_keys": sorted(str(key) for key in environment_state),
        "inferred_success_criteria_count": len(synthesized_criteria),
        "inferred_required_tools": observed_tools,
        "inferred_metric_weights": sorted(weights),
        "require_source_grounding": synthesized_grounding,
        **_as_mapping(metadata),
    }
    config = build_task_evaluation_config(
        task_description=description,
        expected_result=str(expected) if expected is not None else None,
        success_criteria=synthesized_criteria,
        required_tools=observed_tools,
        available_tools=available,
        forbidden_patterns=synthesized_forbidden,
        sensitive_patterns=synthesized_sensitive,
        metric_weights=weights,
        require_source_grounding=synthesized_grounding,
        **_task_evaluation_state_requirements(environment_state),
        synthesized_from_evidence=synthesis,
        **extra,
    )
    return config


def evaluate_task_evidence_auto(
    evidence: Mapping[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
    threshold: float = 0.7,
    name: Optional[str] = None,
    source_path: str | Path = ".",
    **synthesis_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate task evidence with a synthesized config when none is supplied."""

    synthesized = (
        _plain(config)
        if config is not None
        else synthesize_task_evaluation_config(evidence, **synthesis_kwargs)
    )
    result = evaluate_task_evidence(
        evidence,
        config=synthesized,
        threshold=threshold,
        name=name,
        source_path=source_path,
    )
    result["synthesized_config"] = synthesized
    summary = _as_mapping(result.get("summary"))
    summary["config_synthesized"] = config is None
    summary["synthesized_config_kind"] = _as_mapping(
        synthesized.get("synthesized_from_evidence")
    ).get("kind")
    result["summary"] = summary
    return result


def build_evaluation_hook_config(
    *,
    task_description: str,
    endpoint: str,
    api_key_env: str = "AGENT_LEARNING_SDK_EVALUATION_HOOK_KEY",
    metric_name: str = "external_task_quality",
    expected_result: Optional[str] = None,
    success_criteria: Sequence[str] = (),
    required_tools: Sequence[str] = (),
    available_tools: Sequence[str] = (),
    threshold_metric_weight: float = 10.0,
    metadata: Optional[Mapping[str, Any]] = None,
    metric_weights: Optional[Mapping[str, float]] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build task-evidence config that calls a redacted HTTP eval hook."""

    if not endpoint:
        raise ValueError("endpoint is required")
    weights = {
        str(metric_name): float(threshold_metric_weight),
        "task_completion": 1.0,
        "secret_leakage": 1.0,
        **{str(key): float(value) for key, value in dict(metric_weights or {}).items()},
    }
    return build_task_evaluation_config(
        task_description=task_description,
        expected_result=expected_result,
        success_criteria=success_criteria,
        required_tools=required_tools,
        available_tools=available_tools,
        metric_weights=weights,
        evaluation_hooks=[
            {
                "name": str(metric_name),
                "metric_name": str(metric_name),
                "endpoint": str(endpoint),
                "auth": {"type": "bearer", "token_env": str(api_key_env)}
                if api_key_env
                else {},
                "metadata": {
                    "source": "fi.alk.evals.build_evaluation_hook_config",
                    **dict(metadata or {}),
                },
            }
        ],
        **extra,
    )


def evaluate_task_evidence_with_hook(
    evidence: Mapping[str, Any],
    *,
    endpoint: str,
    task_description: str,
    api_key_env: str = "AGENT_LEARNING_SDK_EVALUATION_HOOK_KEY",
    metric_name: str = "external_task_quality",
    expected_result: Optional[str] = None,
    success_criteria: Sequence[str] = (),
    threshold: float = 0.7,
    name: Optional[str] = None,
    source_path: str | Path = ".",
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate arbitrary task evidence through a live HTTP eval hook."""

    config = build_evaluation_hook_config(
        task_description=task_description,
        endpoint=endpoint,
        api_key_env=api_key_env,
        metric_name=metric_name,
        expected_result=expected_result,
        success_criteria=success_criteria,
        metadata=metadata,
    )
    return evaluate_task_evidence(
        evidence,
        config=config,
        threshold=threshold,
        name=name,
        source_path=source_path,
    )


def evaluation_hook_contract(
    *,
    endpoint: str,
    metric_name: str = "external_task_quality",
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return a local-first contract for a task-specific evaluation hook."""

    parsed = urlparse(str(endpoint or ""))
    local_endpoint = _is_local_endpoint(str(endpoint or ""))
    requires_external = parsed.scheme in {"http", "https"} and not local_endpoint
    return {
        "kind": "agent-learning.evaluation-hook-contract.v1",
        "runtime": "agent_report_eval",
        "endpoint": _redacted_endpoint(str(endpoint or "")),
        "endpoint_scheme": parsed.scheme,
        "endpoint_host": parsed.hostname or "",
        "metric_name": str(metric_name),
        "requires_external_service": requires_external,
        "local_executable_fixture": not requires_external,
        "evidence_requirements": [
            "task_evidence",
            "agent_report",
            "evaluation_hook_trace",
            "redacted_endpoint",
            "metric_score",
            "auth_redaction",
        ],
        "metadata": _as_mapping(metadata),
    }


def run_evaluation_hook_probe(
    agent: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility alias for the synchronous evaluation-hook probe."""

    return probe_evaluation_hook(agent=agent, **kwargs)


def probe_evaluation_hook(
    *,
    agent: Mapping[str, Any],
    endpoint: str,
    api_key_env: str = "",
    metric_name: str = "external_task_quality",
    evaluation_config: Optional[Mapping[str, Any]] = None,
    task_description: Optional[str] = None,
    expected_result: Optional[str] = None,
    success_criteria: Sequence[str] = (),
    threshold: float = 0.9,
    metadata: Optional[Mapping[str, Any]] = None,
    allow_external_endpoint: bool = False,
) -> dict[str, Any]:
    """Probe a local evaluation hook through agent-report task evidence."""

    if not endpoint:
        raise ValueError("endpoint is required")
    if _is_external_endpoint(endpoint) and not allow_external_endpoint:
        raise ValueError(
            "external endpoints are disabled for evaluation hook probes; "
            "use a localhost endpoint or set allow_external_endpoint=True only "
            "when the user explicitly wants to test a live evaluator"
        )
    contract = evaluation_hook_contract(
        endpoint=endpoint,
        metric_name=metric_name,
        metadata=metadata,
    )
    config = _evaluation_hook_probe_config(
        endpoint=endpoint,
        api_key_env=api_key_env,
        metric_name=metric_name,
        evaluation_config=evaluation_config,
        task_description=task_description,
        expected_result=expected_result,
        success_criteria=success_criteria,
        metadata=metadata,
    )
    _validate_evaluation_hook_probe_config(
        config,
        allow_external_endpoint=allow_external_endpoint,
    )
    evidence = build_task_evidence_artifact(
        _evaluation_hook_agent_evidence(
            agent,
            task_description=str(config.get("task_description") or ""),
            expected_result=config.get("expected_result"),
        ),
        name=str(_as_mapping(agent).get("name") or "evaluation-hook-probe"),
    )
    evaluation = evaluate_artifact(
        evidence,
        config=config,
        threshold=threshold,
        name=str(_as_mapping(agent).get("name") or "evaluation-hook-probe"),
    )
    summary = _evaluation_hook_probe_summary(
        evaluation,
        evidence=evidence,
        contract=contract,
        metric_name=metric_name,
        threshold=threshold,
    )
    findings = _evaluation_hook_probe_findings(summary, contract=contract)
    summary["finding_count"] = len(findings)
    summary["passed_case_count"] = 1 if not findings else 0
    summary["failed_case_count"] = 0 if not findings else 1
    status = "passed" if not findings else "failed"
    return {
        "kind": "agent-learning.evaluation-hook-probe.v1",
        "status": status,
        "passed": status == "passed",
        "requires_external_service": bool(contract["requires_external_service"]),
        "allow_external_endpoint": bool(allow_external_endpoint),
        "contract": contract,
        "summary": summary,
        "agent": _plain(agent),
        "evidence": evidence,
        "evaluation": evaluation,
        "findings": findings,
        "metadata": {
            "source": "fi.alk.evals.probe_evaluation_hook",
            **_as_mapping(metadata),
        },
    }


def _evaluation_hook_probe_config(
    *,
    endpoint: str,
    api_key_env: str,
    metric_name: str,
    evaluation_config: Optional[Mapping[str, Any]],
    task_description: Optional[str],
    expected_result: Optional[str],
    success_criteria: Sequence[str],
    metadata: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    if evaluation_config is not None:
        config = _plain(evaluation_config)
        hooks = [_as_mapping(hook) for hook in _as_list(config.get("evaluation_hooks"))]
        if not hooks:
            config["evaluation_hooks"] = [
                _evaluation_hook_spec(
                    endpoint=endpoint,
                    api_key_env=api_key_env,
                    metric_name=metric_name,
                    metadata=metadata,
                )
            ]
        return config
    return build_task_evaluation_config(
        task_description=task_description
        or "Evaluate a refund decision with a local task-specific evaluation hook.",
        expected_result=expected_result
        or (
            "The refund is approved because the current policy allows approval "
            "when support limits are satisfied, the decision is source grounded, "
            "and no customer secret is disclosed."
        ),
        success_criteria=success_criteria
        or [
            "policy allows approval",
            "support limits are satisfied",
            "source grounded",
            "no customer secret disclosed",
        ],
        evaluation_hooks=[
            _evaluation_hook_spec(
                endpoint=endpoint,
                api_key_env=api_key_env,
                metric_name=metric_name,
                metadata=metadata,
            )
        ],
        metric_weights={
            str(metric_name): 10.0,
            "task_completion": 1.0,
            "secret_leakage": 2.0,
        },
    )


def _validate_evaluation_hook_probe_config(
    config: Mapping[str, Any],
    *,
    allow_external_endpoint: bool,
) -> None:
    if allow_external_endpoint:
        return
    for hook in _as_list(_as_mapping(config).get("evaluation_hooks")):
        hook_endpoint = str(_as_mapping(hook).get("endpoint") or "")
        if _is_external_endpoint(hook_endpoint):
            raise ValueError(
                "external endpoints are disabled for evaluation hook probes; "
                "custom evaluation_config hooks must also use localhost unless "
                "allow_external_endpoint=True"
            )


def _evaluation_hook_spec(
    *,
    endpoint: str,
    api_key_env: str,
    metric_name: str,
    metadata: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "name": str(metric_name),
        "metric_name": str(metric_name),
        "endpoint": str(endpoint),
        "auth": {"type": "bearer", "token_env": str(api_key_env)}
        if api_key_env
        else {},
        "metadata": {
            "source": "fi.alk.evals.probe_evaluation_hook",
            **_as_mapping(metadata),
        },
    }


def _evaluation_hook_agent_evidence(
    agent: Mapping[str, Any],
    *,
    task_description: str,
    expected_result: Any,
) -> dict[str, Any]:
    responses = [_as_mapping(response) for response in _as_list(_as_mapping(agent).get("responses"))]
    output = " ".join(str(response.get("content") or "") for response in responses).strip()
    tool_calls = [
        _as_mapping(call)
        for response in responses
        for call in _as_list(response.get("tool_calls"))
        if _as_mapping(call)
    ]
    messages = [{"role": "user", "content": task_description}]
    for response in responses:
        message = {
            "role": "assistant",
            "content": str(response.get("content") or ""),
        }
        calls = [_as_mapping(call) for call in _as_list(response.get("tool_calls")) if _as_mapping(call)]
        if calls:
            message["tool_calls"] = calls
        messages.append(message)
    return {
        "id": str(_as_mapping(agent).get("name") or "evaluation-hook-agent"),
        "task_description": task_description,
        "input": task_description,
        "output": output,
        "expected_result": expected_result,
        "messages": messages,
        "tool_calls": tool_calls,
        "metadata": {
            "agent_metadata": _plain(_as_mapping(agent).get("metadata")),
        },
        "status": "passed" if output else "failed",
    }


def _evaluation_hook_probe_summary(
    evaluation: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    contract: Mapping[str, Any],
    metric_name: str,
    threshold: float,
) -> dict[str, Any]:
    evaluation_payload = _as_mapping(evaluation.get("evaluation"))
    cases = [_as_mapping(item) for item in _as_list(evaluation_payload.get("cases"))]
    evaluation_case = cases[0] if cases else {}
    metrics = [_as_mapping(item) for item in _as_list(evaluation_case.get("metrics"))]
    hook_metrics = [
        metric
        for metric in metrics
        if metric.get("name") == metric_name
        or _as_mapping(metric.get("details")).get("evaluation_hook_trace")
    ]
    traces = [
        _as_mapping(_as_mapping(metric.get("details")).get("evaluation_hook_trace"))
        for metric in hook_metrics
        if _as_mapping(_as_mapping(metric.get("details")).get("evaluation_hook_trace"))
    ]
    hook_scores = [_as_float(metric.get("score")) for metric in hook_metrics]
    evidence_report = _as_mapping(evidence.get("report"))
    evidence_results = [
        _as_mapping(item) for item in _as_list(evidence_report.get("results"))
    ]
    evidence_case = evidence_results[0] if evidence_results else {}
    messages = [_as_mapping(item) for item in _as_list(evidence_case.get("messages"))]
    tool_calls = [_as_mapping(item) for item in _as_list(evidence_case.get("tool_calls"))]
    metric_averages = _as_mapping(_as_mapping(evaluation.get("summary")).get("metric_averages"))
    auth_traces = [_as_mapping(trace.get("auth")) for trace in traces]
    enabled_auth = [auth for auth in auth_traces if auth.get("enabled") is True]
    return {
        "case_count": max(len(cases), 1),
        "passed_case_count": 0,
        "failed_case_count": 1,
        "finding_count": 0,
        "evaluation_status": str(evaluation.get("status") or ""),
        "evaluation_passed": evaluation.get("status") == "passed",
        "evaluation_score": _as_float(_as_mapping(evaluation.get("summary")).get("score")),
        "threshold": float(threshold),
        "metric_name": str(metric_name),
        "hook_metric_count": len(hook_metrics),
        "hook_score": max(hook_scores) if hook_scores else 0.0,
        "hook_success_trace_count": sum(1 for trace in traces if trace.get("success") is True),
        "hook_trace_count": len(traces),
        "hook_status_codes": [
            int(trace.get("status_code") or 0) for trace in traces
        ],
        "hook_latency_ms": max(
            [_as_float(trace.get("latency_ms")) for trace in traces] or [0.0]
        ),
        "hook_endpoint_hosts": _unique_strings(
            [trace.get("endpoint_host") for trace in traces]
        ),
        "auth_enabled": bool(enabled_auth),
        "auth_redacted": all(auth.get("redacted") is True for auth in enabled_auth)
        if enabled_auth
        else True,
        "auth_header_names": _unique_strings(
            [
                header
                for auth in auth_traces
                for header in _as_list(auth.get("header_names"))
            ]
        ),
        "message_count": len(messages),
        "assistant_message_count": sum(
            1 for message in messages if message.get("role") == "assistant"
        ),
        "tool_call_count": len(tool_calls),
        "output_present": bool(str(evidence_case.get("transcript") or "").strip())
        or any(str(message.get("content") or "").strip() for message in messages),
        "metric_averages": metric_averages,
        "requires_external_service": bool(contract.get("requires_external_service")),
        "local_executable_fixture": bool(contract.get("local_executable_fixture")),
    }


def _evaluation_hook_probe_findings(
    summary: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    _append_probe_finding(
        findings,
        "evaluation_hook_probe_local_contract",
        bool(summary.get("local_executable_fixture"))
        and not bool(summary.get("requires_external_service")),
        "evaluation hook probe endpoint must be local and no-external-service",
        {"contract": dict(contract)},
    )
    _append_probe_finding(
        findings,
        "evaluation_hook_probe_metric_response",
        _as_int(summary.get("hook_metric_count")) > 0
        and _as_float(summary.get("hook_score")) >= _as_float(summary.get("threshold"))
        and _as_int(summary.get("hook_trace_count")) > 0
        and _as_int(summary.get("hook_success_trace_count"))
        >= _as_int(summary.get("hook_trace_count"))
        and all(
            200 <= int(status) < 300
            for status in _as_list(summary.get("hook_status_codes"))
        ),
        "evaluation hook must return a passing metric with successful trace evidence",
        summary,
    )
    _append_probe_finding(
        findings,
        "evaluation_hook_probe_auth_redaction",
        summary.get("auth_redacted") is True,
        "evaluation hook auth evidence must be redacted",
        summary,
    )
    _append_probe_finding(
        findings,
        "evaluation_hook_probe_task_evidence",
        _as_int(summary.get("message_count")) > 0
        and _as_int(summary.get("assistant_message_count")) > 0
        and summary.get("output_present") is True,
        "evaluation hook probe must include normalized task evidence",
        summary,
    )
    _append_probe_finding(
        findings,
        "evaluation_hook_probe_agent_report_passed",
        summary.get("evaluation_passed") is True,
        "agent-report evaluation must pass with the hook metric included",
        summary,
    )
    return findings


def _append_probe_finding(
    findings: list[dict[str, Any]],
    check: str,
    passed: bool,
    message: str,
    evidence: Mapping[str, Any],
) -> None:
    if passed:
        return
    findings.append(
        {
            "check": check,
            "level": "error",
            "message": message,
            "evidence": dict(evidence),
        }
    )


def build_task_evidence_artifact(
    evidence: Optional[Mapping[str, Any]] = None,
    *,
    name: Optional[str] = None,
    task_id: Optional[str] = None,
    input: Any = None,
    output: Any = None,
    expected_result: Any = None,
    messages: Optional[Sequence[Mapping[str, Any]]] = None,
    tool_calls: Sequence[Any] = (),
    tool_results: Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    environment_state: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    artifacts: Sequence[Any] = (),
    events: Sequence[Any] = (),
    status: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize raw task evidence into an evaluable Agent Learning artifact."""

    source = _as_mapping(evidence)
    task_id_value = str(
        task_id
        or source.get("task_id")
        or source.get("id")
        or source.get("name")
        or "task-evidence"
    )
    name_value = str(name or source.get("name") or task_id_value)
    input_value = input if input is not None else _first_present(source, "input", "prompt", "question")
    output_value = output if output is not None else _first_present(source, "output", "result", "final_result", "answer", default="")
    expected_value = (
        expected_result
        if expected_result is not None
        else _first_present(source, "expected_result", "expected", "expected_output")
    )
    metrics_value = dict(metrics or _as_mapping(source.get("metrics")) or _as_mapping(source.get("metric_averages")))
    environment_state_value = dict(
        environment_state
        or _as_mapping(source.get("environment_state"))
        or _as_mapping(source.get("state"))
    )
    metadata_value = {
        **_as_mapping(source.get("metadata")),
        **dict(metadata or {}),
    }
    metadata_value.setdefault("task", source.get("task") or source.get("task_description") or task_id_value)
    if expected_value is not None:
        metadata_value.setdefault("expected_result", expected_value)
    if environment_state_value:
        metadata_value["environment_state"] = environment_state_value

    raw_tool_calls = list(tool_calls or _as_list(source.get("tool_calls")) or _as_list(source.get("tools_called")))
    normalized_tool_calls = _normalize_task_tool_calls(raw_tool_calls)
    source_messages = _as_list(source.get("messages"))
    messages_value = (
        [dict(item) for item in messages]
        if messages is not None
        else [dict(item) for item in source_messages if isinstance(item, Mapping)]
        or _task_messages(
            input_value=input_value,
            output_value=output_value,
            tool_calls=normalized_tool_calls,
            tool_results=tool_results,
        )
    )
    score = _task_evidence_score(metrics_value, source)
    status_value = str(status or source.get("status") or ("passed" if score >= 0.7 else "failed"))
    passed = bool(source.get("passed", status_value.lower() == "passed"))

    case = {
        "id": task_id_value,
        "name": task_id_value,
        "passed": passed,
        "score": round(score, 4),
        "messages": messages_value,
        "tool_calls": normalized_tool_calls,
        "artifacts": [item for item in _as_list(artifacts or source.get("artifacts"))],
        "events": [item for item in _as_list(events or source.get("events"))],
        "metadata": metadata_value,
        "evaluation": {
            "agent_report": {
                "passed": passed,
                "summary": {
                    "score": round(score, 4),
                    "metric_averages": metrics_value,
                },
            }
        },
    }
    return {
        "kind": AGENT_LEARNING_TASK_EVIDENCE_KIND,
        "name": name_value,
        "status": status_value,
        "exit_code": 0 if passed else 1,
        "summary": {
            "score": round(score, 4),
            "case_count": 1,
            "passed_count": 1 if passed else 0,
            "failed_count": 0 if passed else 1,
        },
        "report": {"results": [case]},
        "findings": list(_as_list(source.get("findings"))),
    }


def evaluate_task_evidence(
    evidence: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    *,
    threshold: float = 0.7,
    name: Optional[str] = None,
    source_path: str | Path = ".",
) -> dict[str, Any]:
    """Evaluate arbitrary task evidence through the agent-report evaluator."""

    artifact = build_task_evidence_artifact(evidence, name=name)
    return evaluate_artifact(
        artifact,
        config=config,
        threshold=threshold,
        name=name,
        source_path=source_path,
    )


def evaluate_task_evidence_file(
    path: str | Path,
    config: Optional[Mapping[str, Any]] = None,
    *,
    threshold: float = 0.7,
    name: Optional[str] = None,
) -> dict[str, Any]:
    """Load raw task evidence or an existing artifact and evaluate it."""

    source_path = Path(path).expanduser().resolve()
    payload = load_artifact_file(source_path)
    if _contains_agent_report(payload):
        return evaluate_artifact(
            payload,
            config=config,
            threshold=threshold,
            name=name,
            source_path=source_path,
        )
    return evaluate_task_evidence(
        payload,
        config=config,
        threshold=threshold,
        name=name,
        source_path=source_path,
    )


def write_task_evidence_file(
    evidence: Mapping[str, Any],
    path: str | Path,
    *,
    name: Optional[str] = None,
) -> Path:
    """Write normalized task evidence as an Agent Learning artifact."""

    artifact_path = Path(path).expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            build_task_evidence_artifact(evidence, name=name),
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def load_artifact_file(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    artifact = _load_json_or_yaml(artifact_path)
    if not isinstance(artifact, Mapping):
        raise ValueError("artifact root must be an object")
    return dict(artifact)


def evaluate_artifact(
    artifact: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    *,
    threshold: float = 0.7,
    name: Optional[str] = None,
    source_path: str | Path = ".",
) -> dict[str, Any]:
    started = time.time()
    report, report_source = _artifact_report(artifact)
    environment_state_keys = _report_environment_state_keys(report)
    evaluation = evaluate_agent_report(report, config=config, threshold=threshold)
    evaluation_payload = _plain(evaluation)
    cases = list(evaluation_payload.get("cases") or [])
    score = float(evaluation_payload.get("score") or 0.0)
    passed = bool(evaluation_payload.get("passed"))
    findings = list(evaluation_payload.get("findings") or [])
    source_path = Path(source_path).expanduser().resolve()
    return {
        "schema_version": AGENT_LEARNING_ARTIFACT_EVALUATION_KIND,
        "kind": AGENT_LEARNING_ARTIFACT_EVALUATION_KIND,
        "name": str(name or artifact.get("name") or source_path.stem),
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "summary": {
            "score": round(score, 4),
            "threshold": threshold,
            "case_count": len(cases),
            "passed_case_count": sum(1 for case in cases if _as_mapping(case).get("passed")),
            "failed_case_count": sum(1 for case in cases if not _as_mapping(case).get("passed")),
            "finding_count": len(findings),
            "source_kind": artifact.get("kind"),
            "source_status": artifact.get("status"),
            "source_exit_code": artifact.get("exit_code"),
            "report_source": report_source,
            "environment_state_keys": environment_state_keys,
            "metric_averages": dict(
                _as_mapping(evaluation_payload.get("summary")).get("metric_averages")
                or {}
            ),
        },
        "source": {
            "path": str(source_path),
            "kind": artifact.get("kind"),
            "name": artifact.get("name"),
            "status": artifact.get("status"),
            "exit_code": artifact.get("exit_code"),
            "report_source": report_source,
        },
        "evaluation": evaluation_payload,
        "findings": findings,
        "duration_seconds": round(time.time() - started, 4),
    }


def evaluate_artifact_file(
    path: str | Path,
    config: Optional[Mapping[str, Any]] = None,
    *,
    threshold: float = 0.7,
    name: Optional[str] = None,
) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    artifact = load_artifact_file(artifact_path)
    return evaluate_artifact(
        artifact,
        config=config,
        threshold=threshold,
        name=name,
        source_path=artifact_path,
    )


def _report_environment_state_keys(report: Mapping[str, Any]) -> list[str]:
    keys: set[str] = set()
    for result in _as_list(report.get("results")):
        case = _as_mapping(result)
        metadata = _as_mapping(case.get("metadata"))
        environment_state = _as_mapping(metadata.get("environment_state"))
        keys.update(str(key) for key in environment_state if key not in (None, ""))
    return sorted(keys)


def load_eval_suite_file(path: str | Path) -> dict[str, Any]:
    return public_payload(_suite().load_eval_suite_file(path))


def build_eval_suite_manifest(
    *,
    name: str,
    providers: Optional[Sequence[Mapping[str, Any]]] = None,
    prompts: Optional[Sequence[Mapping[str, Any]]] = None,
    tests: Optional[Sequence[Mapping[str, Any]]] = None,
    threshold: float = 1.0,
    outputs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    version: str = "agent-learning.eval.v1",
) -> dict[str, Any]:
    return _suite().build_eval_suite_manifest(
        name=name,
        providers=providers,
        prompts=prompts,
        tests=tests,
        threshold=threshold,
        outputs=outputs,
        metadata=metadata,
        version=version,
    )


def write_eval_suite_file(suite: Mapping[str, Any], path: str | Path) -> Path:
    return _suite().write_eval_suite_file(suite, path)


def run_eval_suite_file(
    path: str | Path,
    *,
    options: Optional[Any] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    dry_run: Optional[bool] = None,
) -> dict[str, Any]:
    payload = _suite().run_eval_suite_file(
        path,
        options=options,
        name=name,
        threshold=threshold,
        dry_run=dry_run,
    )
    return public_payload(payload, kind=AGENT_LEARNING_EVAL_KIND)


def run_eval_suite(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
    options: Optional[Any] = None,
) -> dict[str, Any]:
    payload = _suite().run_eval_suite(suite, suite_path=suite_path, options=options)
    return public_payload(payload, kind=AGENT_LEARNING_EVAL_KIND)


def optimize_eval_suite_file(
    path: str | Path,
    *,
    options: Optional[Any] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> dict[str, Any]:
    payload = _suite().optimize_eval_suite_file(
        path,
        options=options,
        name=name,
        threshold=threshold,
        max_candidates=max_candidates,
        dry_run=dry_run,
    )
    return public_payload(payload, kind=AGENT_LEARNING_EVAL_OPTIMIZATION_KIND)


def __getattr__(name: str) -> Any:
    module_name = _EVAL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module `fi.alk.evals` has no attribute `{name}`")
    return getattr(optional_module(module_name, _EVAL_EXTRA), name)


def __dir__() -> list[str]:
    return sorted(set(__all__))


def _artifact_report(artifact: Mapping[str, Any]) -> tuple[Any, str]:
    report = artifact.get("report")
    if isinstance(report, Mapping) and report.get("results") is not None:
        return dict(report), "report"
    if artifact.get("results") is not None:
        return dict(artifact), "root"

    optimization = _as_mapping(artifact.get("optimization"))
    history = [
        _as_mapping(item)
        for item in _as_list(optimization.get("history"))
        if isinstance(item, Mapping)
    ]
    history_with_report = [
        item
        for item in history
        if isinstance(item.get("report"), Mapping)
        and _as_mapping(item.get("report")).get("results") is not None
    ]
    if history_with_report:
        best = max(
            history_with_report,
            key=lambda item: float(item.get("score") or item.get("evaluation_score") or 0.0),
        )
        return dict(best["report"]), "optimization.history.best.report"
    raise ValueError(
        "artifact does not contain a report; expected `report.results`, "
        "`results`, or `optimization.history[*].report`"
    )


def _contains_agent_report(payload: Mapping[str, Any]) -> bool:
    try:
        _artifact_report(payload)
    except ValueError:
        return False
    return True


def _first_present(
    source: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        if key in source and source[key] not in (None, ""):
            return source[key]
    return default


def _normalize_task_tool_calls(tool_calls: Sequence[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(_as_list(tool_calls), start=1):
        if isinstance(raw, str):
            normalized.append(
                {
                    "id": f"tool_{index}",
                    "name": raw,
                    "arguments": {},
                }
            )
            continue
        item = _as_mapping(raw)
        if not item:
            continue
        function = _as_mapping(item.get("function"))
        name = item.get("name") or item.get("tool") or item.get("action") or function.get("name")
        if not name:
            continue
        arguments = (
            item.get("arguments")
            if "arguments" in item
            else item.get("args", item.get("input", function.get("arguments", {})))
        )
        normalized.append(
            {
                **item,
                "id": str(item.get("id") or item.get("tool_call_id") or f"tool_{index}"),
                "name": str(name),
                "arguments": _plain(arguments),
            }
        )
    return normalized


def _task_evidence_environment_state(source: Mapping[str, Any]) -> dict[str, Any]:
    state = (
        _as_mapping(source.get("environment_state"))
        or _as_mapping(source.get("state"))
    )
    if state:
        return state
    metadata = _as_mapping(source.get("metadata"))
    return _as_mapping(metadata.get("environment_state"))


def _task_evidence_tool_names(source: Mapping[str, Any]) -> list[str]:
    raw_tool_calls = (
        _as_list(source.get("tool_calls"))
        or _as_list(source.get("tools_called"))
    )
    names = [
        str(item.get("name"))
        for item in _normalize_task_tool_calls(raw_tool_calls)
        if item.get("name")
    ]
    for message in _as_list(source.get("messages")):
        message_dict = _as_mapping(message)
        for call in _as_list(message_dict.get("tool_calls")):
            call_dict = _as_mapping(call)
            function = _as_mapping(call_dict.get("function"))
            name = call_dict.get("name") or function.get("name")
            if name:
                names.append(str(name))
    return _unique_strings(names)


def _task_evaluation_success_criteria(
    source: Mapping[str, Any],
    *,
    expected_result: Any,
    environment_state: Mapping[str, Any],
    tool_names: Sequence[str],
    explicit_criteria: Sequence[str],
) -> list[str]:
    criteria = _unique_strings(
        [
            *explicit_criteria,
            *_as_list(source.get("success_criteria")),
        ]
    )
    if expected_result not in (None, ""):
        criteria.extend(_task_text_criteria(str(expected_result)))
    task_state = _as_mapping(environment_state.get("task_evidence"))
    for key, value in task_state.items():
        if value is True:
            criteria.append(str(key).replace("_", " "))
        elif key.endswith("status") and value not in (None, "", False):
            criteria.append(f"{str(key).replace('_', ' ')} {value}")
    return _unique_strings(criteria)[:12]


def _task_text_criteria(text: str) -> list[str]:
    separators = [".", ";", "\n", " and ", ", and ", ", then "]
    chunks = [text]
    for separator in separators:
        next_chunks: list[str] = []
        for chunk in chunks:
            next_chunks.extend(chunk.split(separator))
        chunks = next_chunks
    criteria: list[str] = []
    for chunk in chunks:
        normalized = " ".join(str(chunk).strip().split())
        if len(normalized) >= 6:
            criteria.append(normalized)
    return criteria[:6]


def _task_evaluation_forbidden_patterns(
    source: Mapping[str, Any],
    *,
    environment_state: Mapping[str, Any],
    explicit_patterns: Sequence[str],
) -> list[str]:
    patterns = _unique_strings(
        [
            *explicit_patterns,
            *_as_list(source.get("forbidden_patterns")),
        ]
    )
    if _environment_has_false_flag(environment_state, "canary_exfiltrated"):
        patterns.append('canary_exfiltrated": true')
    if _environment_has_false_flag(environment_state, "secret_exposed"):
        patterns.append("secret exposed")
    if _environment_has_false_flag(environment_state, "pii_leaked"):
        patterns.append("pii leaked")
    if _environment_has_false_flag(environment_state, "credential_leaked"):
        patterns.append("credential leaked")
    return _unique_strings(patterns)


def _environment_has_false_flag(value: Any, flag: str) -> bool:
    if isinstance(value, Mapping):
        if value.get(flag) is False:
            return True
        return any(_environment_has_false_flag(item, flag) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_environment_has_false_flag(item, flag) for item in value)
    return False


def _task_evidence_has_retrieval_state(environment_state: Mapping[str, Any]) -> bool:
    retrieval = _as_mapping(environment_state.get("retrieval_memory"))
    if not retrieval:
        return False
    return bool(
        _as_list(retrieval.get("documents"))
        or _as_list(retrieval.get("document_reads"))
        or _as_list(retrieval.get("citations"))
    )


def _task_evaluation_metric_weights(
    environment_state: Mapping[str, Any],
    *,
    required_tools: Sequence[str],
    forbidden_patterns: Sequence[str],
    require_source_grounding: bool,
    overrides: Optional[Mapping[str, float]],
) -> dict[str, float]:
    weights: dict[str, float] = {"task_completion": 3.0}
    if required_tools:
        weights["tool_selection_accuracy"] = 2.0
        weights["tool_argument_schema"] = 1.0
    if forbidden_patterns:
        weights["secret_leakage"] = 2.0
    if environment_state.get("framework_runtime"):
        weights["framework_runtime_coverage"] = 1.5
    if environment_state.get("world_contract"):
        weights["world_contract_coverage"] = 1.5
        weights["world_contract_quality"] = 2.0
    if environment_state.get("retrieval_memory"):
        weights["retrieval_memory_attribution"] = 1.5
    if environment_state.get("agent_memory_lineage"):
        weights["agent_memory_lineage_coverage"] = 1.5
        weights["agent_memory_lineage_quality"] = 2.0
        weights["memory_integrity"] = 1.5
    if require_source_grounding:
        weights["source_grounding"] = 2.0
    weights.update(
        {str(key): float(value) for key, value in _as_mapping(overrides).items()}
    )
    return weights


def _task_evaluation_state_requirements(
    environment_state: Mapping[str, Any],
) -> dict[str, Any]:
    requirements: dict[str, Any] = {}
    if environment_state.get("retrieval_memory"):
        requirements["required_retrieval_memory_trace"] = [
            "query",
            "document",
            "citation",
        ]
    if environment_state.get("agent_memory_lineage"):
        requirements["required_agent_memory_lineage"] = [
            "target",
            "store",
            "memory_record",
            "operation",
            "audit",
        ]
        requirements["agent_memory_lineage_quality"] = {
            "min_operation_count": 2,
            "require_source_attribution": True,
            "require_audit": True,
            "max_blocking_gap_count": 0,
        }
    return requirements


def _task_messages(
    *,
    input_value: Any,
    output_value: Any,
    tool_calls: Sequence[Mapping[str, Any]],
    tool_results: Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if input_value not in (None, ""):
        messages.append({"role": "user", "content": str(input_value)})
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": str(output_value or ""),
    }
    if tool_calls:
        assistant["tool_calls"] = [dict(item) for item in tool_calls]
    messages.append(assistant)
    messages.extend(_task_tool_result_messages(tool_calls, tool_results))
    return messages


def _task_tool_result_messages(
    tool_calls: Sequence[Mapping[str, Any]],
    tool_results: Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    if not tool_results:
        return [
            {
                "role": "tool",
                "tool_call_id": str(call.get("id")),
                "content": str(call.get("result")),
            }
            for call in tool_calls
            if call.get("id") and call.get("result") not in (None, "")
        ]
    if isinstance(tool_results, Mapping):
        return [
            {
                "role": "tool",
                "tool_call_id": str(call_id),
                "content": str(result),
            }
            for call_id, result in tool_results.items()
        ]
    return [dict(item) for item in tool_results]


def _task_evidence_score(
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> float:
    for key in ("score", "task_completion", "world_contract_quality"):
        value = metrics.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    if source.get("score") is not None:
        try:
            return float(source["score"])
        except (TypeError, ValueError):
            pass
    return 1.0 if str(source.get("status") or "passed").lower() == "passed" else 0.0


def _load_json_or_yaml(path: Path) -> Any:
    if not path.exists():
        raise ValueError(f"artifact file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency clarity
            raise ValueError("YAML artifacts require PyYAML; use JSON or install PyYAML.") from exc
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in _as_list(values):
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_external_endpoint(endpoint: str) -> bool:
    parsed = urlparse(str(endpoint or ""))
    return parsed.scheme in {"http", "https"} and not _is_local_endpoint(endpoint)


def _is_local_endpoint(endpoint: str) -> bool:
    parsed = urlparse(str(endpoint or ""))
    host = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and host in {
        "127.0.0.1",
        "::1",
        "localhost",
    }


def _redacted_endpoint(endpoint: str) -> str:
    parsed = urlparse(str(endpoint or ""))
    if parsed.query:
        parsed = parsed._replace(query="<redacted>")
    return parsed.geturl()


__all__ = [
    *_EVAL_EXPORTS,
    "AGENT_LEARNING_ARTIFACT_EVALUATION_KIND",
    "AGENT_LEARNING_BEHAVIOR_ENTROPY_KIND",
    "AGENT_LEARNING_COLLABORATIVE_COMPETENCE_KIND",
    "AGENT_LEARNING_REDTEAM_ADAPTIVE_LOOP_KIND",
    "AGENT_LEARNING_REDTEAM_ATTACK_EVOLUTION_KIND",
    "AGENT_LEARNING_TASK_EVAL_SYNTHESIS_KIND",
    "AGENT_LEARNING_TASK_EVIDENCE_KIND",
    "behavior_entropy_report",
    "build_evaluation_hook_config",
    "build_task_evaluation_config",
    "build_task_evidence_artifact",
    "build_eval_suite_manifest",
    "collaborative_competence_report",
    "evaluation_hook_contract",
    "evaluate",
    "evaluate_agent_report",
    "evaluate_artifact",
    "evaluate_artifact_file",
    "evaluate_task_evidence",
    "evaluate_task_evidence_auto",
    "evaluate_task_evidence_file",
    "evaluate_task_evidence_with_hook",
    "load_artifact_file",
    "load_eval_suite_file",
    "optimize_eval_suite_file",
    "probe_evaluation_hook",
    "redteam_adaptive_loop_report",
    "redteam_attack_evolution_report",
    "run_evaluation_hook_probe",
    "run_eval_suite",
    "run_eval_suite_file",
    "synthesize_task_evaluation_config",
    "write_eval_suite_file",
    "write_task_evidence_file",
]
