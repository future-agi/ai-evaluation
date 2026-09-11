"""Type definitions for AutoEval."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class AppCategory(Enum):
    """Application category classifications."""

    CUSTOMER_SUPPORT = "customer_support"
    RAG_SYSTEM = "rag_system"
    CODE_ASSISTANT = "code_assistant"
    CONTENT_MODERATION = "content_moderation"
    AGENT_WORKFLOW = "agent_workflow"
    CHATBOT = "chatbot"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    DATA_EXTRACTION = "data_extraction"
    SEARCH = "search"
    QUESTION_ANSWERING = "question_answering"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk level for the application.

    Determines threshold strictness:
    - LOW: 0.6 threshold (internal tools, development)
    - MEDIUM: 0.7 threshold (general public-facing)
    - HIGH: 0.8 threshold (healthcare, finance, legal)
    - CRITICAL: 0.9 threshold (safety-critical systems)
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DomainSensitivity(Enum):
    """Domain sensitivity classification."""

    GENERAL = "general"
    PII_SENSITIVE = "pii_sensitive"
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    CHILDREN = "children"  # COPPA compliance
    GOVERNMENT = "government"


class ScannerAction(Enum):
    """Action to take when scanner detects an issue."""

    BLOCK = "block"
    FLAG = "flag"
    WARN = "warn"
    REDACT = "redact"


@dataclass
class AppRequirement:
    """A detected requirement from app analysis."""

    category: str
    importance: str  # "required", "recommended", "optional"
    reason: str
    suggested_evals: List[str] = field(default_factory=list)
    suggested_scanners: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "importance": self.importance,
            "reason": self.reason,
            "suggested_evals": self.suggested_evals,
            "suggested_scanners": self.suggested_scanners,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppRequirement":
        """Create from dictionary."""
        return cls(
            category=data.get("category", ""),
            importance=data.get("importance", "recommended"),
            reason=data.get("reason", ""),
            suggested_evals=data.get("suggested_evals", []),
            suggested_scanners=data.get("suggested_scanners", []),
        )


@dataclass
class AppAnalysis:
    """Result of analyzing an application description."""

    category: AppCategory
    risk_level: RiskLevel
    domain_sensitivity: DomainSensitivity
    requirements: List[AppRequirement]
    detected_features: List[str]  # e.g., ["tool_use", "rag", "multi_turn"]
    confidence: float  # 0.0 to 1.0
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "risk_level": self.risk_level.value,
            "domain_sensitivity": self.domain_sensitivity.value,
            "requirements": [r.to_dict() for r in self.requirements],
            "detected_features": self.detected_features,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppAnalysis":
        """Create from dictionary."""
        return cls(
            category=AppCategory(data.get("category", "unknown")),
            risk_level=RiskLevel(data.get("risk_level", "medium")),
            domain_sensitivity=DomainSensitivity(
                data.get("domain_sensitivity", "general")
            ),
            requirements=[
                AppRequirement.from_dict(r) for r in data.get("requirements", [])
            ],
            detected_features=data.get("detected_features", []),
            confidence=data.get("confidence", 0.5),
            explanation=data.get("explanation", ""),
        )


@dataclass
class AutoEvalResult:
    """Result of running an AutoEval pipeline."""

    passed: bool
    scan_result: Optional[Any] = None  # PipelineResult from scanners
    eval_result: Optional[Any] = None  # EvaluatorResult from evaluator
    metric_results: List[Any] = field(default_factory=list)  # Core EvalResults
    blocked_by_scanner: bool = False
    total_latency_ms: float = 0.0

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the results."""
        summary = {
            "passed": self.passed,
            "blocked_by_scanner": self.blocked_by_scanner,
            "total_latency_ms": self.total_latency_ms,
        }

        if self.scan_result:
            summary["scan_passed"] = self.scan_result.passed
            summary["scanners_triggered"] = (
                self.scan_result.blocked_by + self.scan_result.flagged_by
                if hasattr(self.scan_result, "blocked_by")
                else []
            )

        if self.metric_results:
            summary["metric_results"] = {
                getattr(r, "eval_name", "unknown"): {
                    "score": getattr(r, "score", None),
                    "passed": getattr(r, "passed", None),
                }
                for r in self.metric_results
            }

        if self.eval_result:
            batch = (
                self.eval_result.wait()
                if hasattr(self.eval_result, "wait") and self.eval_result.is_future
                else getattr(self.eval_result, "batch", None)
            )
            if batch:
                summary["eval_success_rate"] = batch.success_rate
                summary["eval_results"] = {
                    r.eval_name: {
                        "score": getattr(r.value, "score", None),
                        "passed": getattr(r.value, "passed", None),
                    }
                    for r in batch.results
                }

        return summary

    def explain(self) -> str:
        """Return a human-readable explanation of the results."""
        lines = [f"AutoEval Result: {'PASSED' if self.passed else 'FAILED'}"]
        lines.append(f"Total Latency: {self.total_latency_ms:.2f}ms")

        if self.blocked_by_scanner:
            lines.append("\nBlocked by scanner before evaluation.")

        if self.scan_result:
            lines.append(f"\nScanner Result: {'PASSED' if self.scan_result.passed else 'FAILED'}")
            if hasattr(self.scan_result, "blocked_by") and self.scan_result.blocked_by:
                lines.append(f"  Blocked by: {', '.join(self.scan_result.blocked_by)}")
            if hasattr(self.scan_result, "flagged_by") and self.scan_result.flagged_by:
                lines.append(f"  Flagged by: {', '.join(self.scan_result.flagged_by)}")

        if self.metric_results:
            lines.append(f"\nMetric Results ({len(self.metric_results)}):")
            for r in self.metric_results:
                status = "PASSED" if getattr(r, "passed", False) else "FAILED"
                score = getattr(r, "score", "N/A")
                name = getattr(r, "eval_name", "unknown")
                if isinstance(score, float):
                    score = f"{score:.2f}"
                lines.append(f"  {name}: {status} (score: {score})")

        if self.eval_result:
            batch = (
                self.eval_result.wait()
                if hasattr(self.eval_result, "wait") and self.eval_result.is_future
                else getattr(self.eval_result, "batch", None)
            )
            if batch:
                lines.append(f"\nFramework Evaluation Result: {batch.success_rate:.0%} passed")
                for r in batch.results:
                    status = "PASSED" if getattr(r.value, "passed", False) else "FAILED"
                    score = getattr(r.value, "score", "N/A")
                    if isinstance(score, float):
                        score = f"{score:.2f}"
                    lines.append(f"  {r.eval_name}: {status} (score: {score})")

        return "\n".join(lines)
