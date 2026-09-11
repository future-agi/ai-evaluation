"""
Base classes for the Dual-Judge System.

Provides foundational types and interfaces for pattern-based
and LLM-based security judges.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from ..types import (
    Severity,
    VulnerabilityCategory,
    SecurityFinding,
    CodeLocation,
    CWE_CATEGORIES,
    SEVERITY_WEIGHTS,
)


class ConsensusMode(str, Enum):
    """Consensus mode for dual-judge system."""

    # Flag if either judge flags (high recall, may have false positives)
    ANY = "any"

    # Flag only if both judges agree (high precision, may miss some)
    BOTH = "both"

    # Weighted combination of confidences (balanced)
    WEIGHTED = "weighted"

    # Pattern judge first, LLM only for uncertain cases
    CASCADE = "cascade"


class JudgeFinding(BaseModel):
    """A finding from a judge."""
    model_config = ConfigDict(extra="allow")

    cwe_id: str = Field(..., description="CWE identifier")
    vulnerability_type: str = Field(..., description="Type of vulnerability")
    description: str = Field(..., description="Description of the finding")
    severity: Severity = Field(default=Severity.MEDIUM)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Judge's confidence in this finding",
    )
    location: Optional[CodeLocation] = Field(
        default=None,
        description="Location in code",
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        description="Suggested remediation",
    )

    # Judge metadata
    judge_type: str = Field(
        default="unknown",
        description="Type of judge that found this (pattern/llm)",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Judge's reasoning (for LLM judge)",
    )

    def to_security_finding(self) -> SecurityFinding:
        """Convert to standard SecurityFinding."""
        return SecurityFinding(
            cwe_id=self.cwe_id,
            vulnerability_type=self.vulnerability_type,
            category=CWE_CATEGORIES.get(self.cwe_id, VulnerabilityCategory.INPUT_VALIDATION),
            severity=self.severity,
            confidence=self.confidence,
            description=self.description,
            location=self.location,
            suggested_fix=self.suggested_fix,
        )


class JudgeResult(BaseModel):
    """Result from a judge evaluation."""
    model_config = ConfigDict(extra="allow")

    # Overall assessment
    is_secure: bool = Field(
        default=True,
        description="Whether the code is considered secure",
    )
    security_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Security score (0 = insecure, 1 = secure)",
    )

    # Findings
    findings: List[JudgeFinding] = Field(
        default_factory=list,
        description="List of findings",
    )

    # Metadata
    judge_type: str = Field(
        default="unknown",
        description="Type of judge (pattern/llm/dual)",
    )
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="Execution time in milliseconds",
    )
    language: str = Field(default="python")

    # For dual judge
    pattern_result: Optional["JudgeResult"] = Field(
        default=None,
        description="Pattern judge result (for dual)",
    )
    llm_result: Optional["JudgeResult"] = Field(
        default=None,
        description="LLM judge result (for dual)",
    )
    consensus_mode: Optional[ConsensusMode] = Field(
        default=None,
        description="Consensus mode used (for dual)",
    )

    @property
    def finding_count(self) -> int:
        """Number of findings."""
        return len(self.findings)

    @property
    def high_confidence_findings(self) -> List[JudgeFinding]:
        """Findings with confidence >= 0.8."""
        return [f for f in self.findings if f.confidence >= 0.8]

    def get_severity_counts(self) -> Dict[str, int]:
        """Count findings by severity."""
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts

    def get_cwe_counts(self) -> Dict[str, int]:
        """Count findings by CWE."""
        counts = {}
        for f in self.findings:
            counts[f.cwe_id] = counts.get(f.cwe_id, 0) + 1
        return counts

    def to_security_findings(self) -> List[SecurityFinding]:
        """Convert all findings to SecurityFindings."""
        return [f.to_security_finding() for f in self.findings]


class BaseJudge(ABC):
    """Base class for security judges."""

    judge_type: str = "unknown"

    def __init__(
        self,
        severity_threshold: Severity = Severity.HIGH,
        min_confidence: float = 0.7,
    ):
        """
        Initialize the judge.

        Args:
            severity_threshold: Minimum severity to flag as insecure
            min_confidence: Minimum confidence to include findings
        """
        self.severity_threshold = severity_threshold
        self.min_confidence = min_confidence
        self._severity_order = [
            Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM,
            Severity.LOW, Severity.INFO
        ]

    @abstractmethod
    def judge(
        self,
        code: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
    ) -> JudgeResult:
        """
        Judge code for security vulnerabilities.

        Args:
            code: Source code to analyze
            language: Programming language
            context: Optional context (instruction, etc.)

        Returns:
            JudgeResult with findings and assessment
        """
        pass

    def _is_secure(self, findings: List[JudgeFinding]) -> bool:
        """Determine if code is secure based on findings."""
        threshold_idx = self._severity_order.index(self.severity_threshold)

        for finding in findings:
            if finding.confidence >= self.min_confidence:
                finding_idx = self._severity_order.index(finding.severity)
                if finding_idx <= threshold_idx:
                    return False
        return True

    def _compute_score(self, findings: List[JudgeFinding]) -> float:
        """Compute security score from findings."""
        if not findings:
            return 1.0

        total_penalty = sum(
            SEVERITY_WEIGHTS.get(f.severity, 0.1) * f.confidence
            for f in findings
            if f.confidence >= self.min_confidence
        )

        return max(0.0, 1.0 - min(1.0, total_penalty))

    def _filter_findings(
        self,
        findings: List[JudgeFinding],
    ) -> List[JudgeFinding]:
        """Filter findings by confidence."""
        return [
            f for f in findings
            if f.confidence >= self.min_confidence
        ]
