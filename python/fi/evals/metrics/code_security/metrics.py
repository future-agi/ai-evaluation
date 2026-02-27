"""
Code Security Metrics for AI Evaluation.

Provides the main evaluation metrics for code security:
- CodeSecurityScore: Comprehensive security evaluation
- QuickSecurityCheck: Fast pattern-only check (<10ms)
- InjectionSecurityScore: Focus on injection vulnerabilities
- CryptographySecurityScore: Focus on crypto issues
- SecretsSecurityScore: Focus on hardcoded credentials

Usage:
    from fi.evals.metrics.code_security.metrics import (
        CodeSecurityScore,
        QuickSecurityCheck,
    )

    # Comprehensive evaluation
    metric = CodeSecurityScore()
    result = metric.compute(CodeSecurityInput(
        response=generated_code,
        language="python",
    ))
    print(f"Score: {result.score}")
    print(f"Findings: {len(result.findings)}")

    # Fast check
    metric = QuickSecurityCheck()
    result = metric.compute(code, "python")
    print(f"Passed: {result.passed}")
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from .types import (
    Severity,
    VulnerabilityCategory,
    SecurityFinding,
    CodeSecurityInput,
    CodeSecurityOutput,
    EvaluationMode,
    SEVERITY_WEIGHTS,
)
from .detectors import (
    scan_code,
    get_all_detectors,
    get_detectors_by_category,
)
from .joint_metrics import JointSecurityMetrics, JointMetricsResult


class CodeSecurityScore:
    """
    Comprehensive AI code security evaluation.

    The main metric for evaluating security of AI-generated code.
    Combines pattern-based detection with optional functional testing
    for the full func-sec@k picture.

    Usage:
        metric = CodeSecurityScore()

        # Simple usage
        result = metric.compute(CodeSecurityInput(
            response=generated_code,
            language="python",
        ))

        # With instruction context
        result = metric.compute(CodeSecurityInput(
            response=generated_code,
            instruction="Write a database query function",
            mode=EvaluationMode.INSTRUCT,
            language="python",
        ))

        # Check results
        print(f"Score: {result.score}")
        print(f"Passed: {result.passed}")
        for finding in result.findings:
            print(f"  {finding.cwe_id}: {finding.description}")
    """

    def __init__(
        self,
        threshold: float = 0.7,
        severity_threshold: Severity = Severity.HIGH,
        min_confidence: float = 0.7,
        include_info: bool = False,
    ):
        """
        Initialize the metric.

        Args:
            threshold: Score threshold for passing (0.0 to 1.0)
            severity_threshold: Minimum severity to fail
            min_confidence: Minimum confidence to count findings
            include_info: Include INFO-level findings in output
        """
        self.threshold = threshold
        self.severity_threshold = severity_threshold
        self.min_confidence = min_confidence
        self.include_info = include_info

    def compute(self, input: CodeSecurityInput) -> CodeSecurityOutput:
        """
        Compute security score for code.

        Args:
            input: CodeSecurityInput with response and metadata

        Returns:
            CodeSecurityOutput with score, findings, and breakdown
        """
        # Scan for vulnerabilities
        findings = scan_code(input.response, input.language)

        # Filter by confidence
        confident_findings = [
            f for f in findings
            if f.confidence >= self.min_confidence
        ]

        # Optionally filter out INFO
        if not self.include_info:
            confident_findings = [
                f for f in confident_findings
                if f.severity != Severity.INFO
            ]

        # Compute score
        score = self._compute_score(confident_findings)
        passed = score >= self.threshold

        # Severity counts
        severity_counts = self._count_by_severity(confident_findings)

        # CWE breakdown
        cwe_counts = self._count_by_cwe(confident_findings)

        return CodeSecurityOutput(
            score=score,
            passed=passed,
            findings=confident_findings,
            severity_counts=severity_counts,
            cwe_counts=cwe_counts,
            language=input.language,
            mode=input.mode,
            total_findings=len(confident_findings),
            critical_count=severity_counts.get("critical", 0),
            high_count=severity_counts.get("high", 0),
        )

    def compute_one(self, input: CodeSecurityInput) -> Dict[str, Any]:
        """
        Compute and return as dictionary.

        Args:
            input: CodeSecurityInput

        Returns:
            Dictionary with score and details
        """
        result = self.compute(input)
        return {
            "output": result.score,
            "passed": result.passed,
            "findings": [f.model_dump() for f in result.findings],
            "severity_counts": result.severity_counts,
            "cwe_counts": result.cwe_counts,
        }

    def _compute_score(self, findings: List[SecurityFinding]) -> float:
        """Compute score from findings."""
        if not findings:
            return 1.0

        total_penalty = 0.0
        for finding in findings:
            weight = SEVERITY_WEIGHTS.get(finding.severity, 0.1)
            penalty = weight * finding.confidence
            total_penalty += penalty

        # Cap at 1.0 penalty
        return max(0.0, 1.0 - min(1.0, total_penalty))

    def _count_by_severity(
        self,
        findings: List[SecurityFinding],
    ) -> Dict[str, int]:
        """Count findings by severity."""
        counts = {s.value: 0 for s in Severity}
        for f in findings:
            counts[f.severity.value] += 1
        return counts

    def _count_by_cwe(self, findings: List[SecurityFinding]) -> Dict[str, int]:
        """Count findings by CWE."""
        counts = {}
        for f in findings:
            counts[f.cwe_id] = counts.get(f.cwe_id, 0) + 1
        return counts


class QuickSecurityCheck:
    """
    Fast pattern-only security check.

    Optimized for speed (<10ms), useful for:
    - Real-time IDE integration
    - Pre-screening before full analysis
    - High-volume batch processing

    Usage:
        check = QuickSecurityCheck()
        result = check.check(code, "python")

        if not result["passed"]:
            print(f"Found {result['finding_count']} issues")
    """

    def __init__(
        self,
        severity_threshold: Severity = Severity.HIGH,
        min_confidence: float = 0.8,
    ):
        """
        Initialize quick check.

        Args:
            severity_threshold: Minimum severity to fail
            min_confidence: Minimum confidence to count
        """
        self.severity_threshold = severity_threshold
        self.min_confidence = min_confidence
        self._severity_order = [
            Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM,
            Severity.LOW, Severity.INFO
        ]

    def check(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Quick security check.

        Args:
            code: Code to check
            language: Programming language

        Returns:
            Dictionary with passed status and finding count
        """
        findings = scan_code(code, language)

        threshold_idx = self._severity_order.index(self.severity_threshold)

        # Count by severity
        counts = {s.value: 0 for s in Severity}
        failed = False

        for f in findings:
            if f.confidence >= self.min_confidence:
                counts[f.severity.value] += 1
                if self._severity_order.index(f.severity) <= threshold_idx:
                    failed = True

        return {
            "passed": not failed,
            "finding_count": sum(counts.values()),
            "severity_counts": counts,
            "has_critical": counts["critical"] > 0,
            "has_high": counts["high"] > 0,
        }

    def is_secure(self, code: str, language: str = "python") -> bool:
        """
        Simple boolean check.

        Args:
            code: Code to check
            language: Programming language

        Returns:
            True if code passes security check
        """
        return self.check(code, language)["passed"]


class CategorySecurityScore:
    """
    Base class for category-specific security scores.

    Focuses on a specific vulnerability category.
    """

    category: VulnerabilityCategory

    def __init__(
        self,
        threshold: float = 0.7,
        min_confidence: float = 0.7,
    ):
        self.threshold = threshold
        self.min_confidence = min_confidence

    def compute(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Compute category-specific score.

        Args:
            code: Code to check
            language: Programming language

        Returns:
            Dictionary with score and findings
        """
        detectors = get_detectors_by_category(self.category.value)

        all_findings = []
        for detector in detectors:
            findings = detector.detect(code, language)
            all_findings.extend(findings)

        confident_findings = [
            f for f in all_findings
            if f.confidence >= self.min_confidence
        ]

        score = self._compute_score(confident_findings)
        is_secure = len(confident_findings) == 0

        return {
            "score": score,
            "output": score,
            "passed": score >= self.threshold,
            "is_secure": is_secure,
            "findings": confident_findings,
            "finding_count": len(confident_findings),
        }

    def compute_one(self, input: CodeSecurityInput) -> Dict[str, Any]:
        """
        Compute category-specific score from CodeSecurityInput.

        Args:
            input: CodeSecurityInput with response and language

        Returns:
            Dictionary with score and findings
        """
        return self.compute(input.response, input.language)

    def _compute_score(self, findings: List[SecurityFinding]) -> float:
        """Compute score from findings."""
        if not findings:
            return 1.0

        total_penalty = sum(
            SEVERITY_WEIGHTS.get(f.severity, 0.1) * f.confidence
            for f in findings
        )
        return max(0.0, 1.0 - min(1.0, total_penalty))


class InjectionSecurityScore(CategorySecurityScore):
    """Security score focused on injection vulnerabilities."""
    category = VulnerabilityCategory.INJECTION


class CryptographySecurityScore(CategorySecurityScore):
    """Security score focused on cryptography issues."""
    category = VulnerabilityCategory.CRYPTOGRAPHY


class SecretsSecurityScore(CategorySecurityScore):
    """Security score focused on hardcoded credentials."""
    category = VulnerabilityCategory.SECRETS


class SerializationSecurityScore(CategorySecurityScore):
    """Security score focused on serialization issues."""
    category = VulnerabilityCategory.SERIALIZATION


__all__ = [
    "CodeSecurityScore",
    "QuickSecurityCheck",
    "CategorySecurityScore",
    "InjectionSecurityScore",
    "CryptographySecurityScore",
    "SecretsSecurityScore",
    "SerializationSecurityScore",
]
