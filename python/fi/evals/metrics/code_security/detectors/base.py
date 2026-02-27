"""
Base Detector for Code Security Evaluation.

Provides the abstract base class for all vulnerability detectors.
Each detector focuses on a specific CWE or category of vulnerabilities.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass

from ..types import (
    SecurityFinding,
    Severity,
    VulnerabilityCategory,
    CodeLocation,
    CWE_METADATA,
    get_cwe_severity,
    get_cwe_category,
)
from ..analyzer import AnalysisResult


# Registry of all detectors
_DETECTOR_REGISTRY: Dict[str, type] = {}


def register_detector(name: str):
    """Decorator to register a detector class."""
    def decorator(cls):
        _DETECTOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_detector(name: str) -> Optional[type]:
    """Get a detector class by name."""
    return _DETECTOR_REGISTRY.get(name)


def list_detectors() -> List[str]:
    """List all registered detector names."""
    return list(_DETECTOR_REGISTRY.keys())


class BaseDetector(ABC):
    """
    Abstract base class for vulnerability detectors.

    Each detector should:
    - Focus on one or more related CWEs
    - Implement detect() to find vulnerabilities
    - Return SecurityFinding objects with location and confidence

    Example:
        @register_detector("sql_injection")
        class SQLInjectionDetector(BaseDetector):
            name = "sql_injection"
            cwe_ids = ["CWE-89"]
            category = VulnerabilityCategory.INJECTION

            def detect(self, code, language, analysis):
                findings = []
                # ... detection logic ...
                return findings
    """

    # Detector metadata (must be set by subclasses)
    name: str = "base"
    cwe_ids: List[str] = []
    category: VulnerabilityCategory = VulnerabilityCategory.INPUT_VALIDATION
    description: str = "Base detector"

    # Languages this detector supports (empty = all)
    supported_languages: Set[str] = set()

    # Default severity (can be overridden per-finding)
    default_severity: Severity = Severity.MEDIUM

    def __init__(self, enabled: bool = True, min_confidence: float = 0.5):
        """
        Initialize detector.

        Args:
            enabled: Whether this detector is active
            min_confidence: Minimum confidence threshold for findings
        """
        self.enabled = enabled
        self.min_confidence = min_confidence

    @abstractmethod
    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """
        Detect vulnerabilities in code.

        Args:
            code: Source code to analyze
            language: Programming language
            analysis: Pre-computed analysis result (optional)

        Returns:
            List of SecurityFinding objects
        """
        pass

    def supports_language(self, language: str) -> bool:
        """Check if detector supports the given language."""
        if not self.supported_languages:
            return True
        return language.lower() in self.supported_languages

    def create_finding(
        self,
        vulnerability_type: str,
        description: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        end_line: Optional[int] = None,
        snippet: Optional[str] = None,
        function: Optional[str] = None,
        severity: Optional[Severity] = None,
        confidence: float = 0.8,
        cwe_id: Optional[str] = None,
        suggested_fix: Optional[str] = None,
        references: Optional[List[str]] = None,
    ) -> SecurityFinding:
        """
        Create a SecurityFinding with sensible defaults.

        Args:
            vulnerability_type: Human-readable type (e.g., "SQL Injection")
            description: Detailed description of the issue
            line: Line number where vulnerability was found
            column: Column number
            end_line: End line for multi-line issues
            snippet: Code snippet showing the issue
            function: Function name where issue was found
            severity: Override default severity
            confidence: Detection confidence (0.0-1.0)
            cwe_id: Specific CWE ID (uses first from cwe_ids if not provided)
            suggested_fix: Recommended fix
            references: Reference URLs

        Returns:
            SecurityFinding object
        """
        # Use provided CWE or first from detector's list
        actual_cwe = cwe_id or (self.cwe_ids[0] if self.cwe_ids else "CWE-unknown")

        # Get severity from CWE metadata or use default
        if severity is None:
            severity = get_cwe_severity(actual_cwe) if actual_cwe else self.default_severity

        # Build location if we have line info
        location = None
        if line is not None:
            location = CodeLocation(
                line=line,
                column=column,
                end_line=end_line,
                function=function,
                snippet=snippet,
            )

        # Build references
        if references is None:
            references = []
            if actual_cwe and actual_cwe.startswith("CWE-"):
                cwe_num = actual_cwe.replace("CWE-", "")
                references.append(f"https://cwe.mitre.org/data/definitions/{cwe_num}.html")

        return SecurityFinding(
            cwe_id=actual_cwe,
            vulnerability_type=vulnerability_type,
            category=self.category,
            severity=severity,
            confidence=confidence,
            description=description,
            location=location,
            suggested_fix=suggested_fix,
            references=references,
        )

    def filter_findings(
        self,
        findings: List[SecurityFinding],
        min_severity: Optional[Severity] = None,
        min_confidence: Optional[float] = None,
    ) -> List[SecurityFinding]:
        """
        Filter findings by severity and confidence.

        Args:
            findings: List of findings to filter
            min_severity: Minimum severity to include
            min_confidence: Minimum confidence to include

        Returns:
            Filtered list of findings
        """
        severity_order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]

        filtered = findings

        if min_severity:
            min_idx = severity_order.index(min_severity)
            filtered = [
                f for f in filtered
                if severity_order.index(f.severity) >= min_idx
            ]

        if min_confidence:
            filtered = [
                f for f in filtered
                if f.confidence >= min_confidence
            ]

        return filtered

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"


class PatternBasedDetector(BaseDetector):
    """
    Base class for pattern-based detectors.

    Uses regex patterns to detect vulnerabilities.
    Subclasses should define PATTERNS dict mapping pattern names to regex.
    """

    # Patterns to match (name -> regex pattern)
    PATTERNS: Dict[str, str] = {}

    # Context patterns that increase confidence
    CONTEXT_PATTERNS: Dict[str, str] = {}

    # Safe patterns that negate findings
    SAFE_PATTERNS: List[str] = []

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect vulnerabilities using regex patterns."""
        import re

        if not self.enabled:
            return []

        if not self.supports_language(language):
            return []

        findings = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Skip if line matches safe pattern
            if any(re.search(p, line, re.IGNORECASE) for p in self.SAFE_PATTERNS):
                continue

            for pattern_name, pattern in self.PATTERNS.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Calculate confidence based on context
                    confidence = 0.7
                    for ctx_name, ctx_pattern in self.CONTEXT_PATTERNS.items():
                        if re.search(ctx_pattern, line, re.IGNORECASE):
                            confidence = min(1.0, confidence + 0.1)

                    # Get snippet
                    snippet = line.strip()[:100]

                    # Determine function context
                    function = self._find_enclosing_function(lines, i, analysis)

                    finding = self.create_finding(
                        vulnerability_type=self._get_vulnerability_type(pattern_name),
                        description=self._get_description(pattern_name, match),
                        line=i,
                        column=match.start(),
                        snippet=snippet,
                        function=function,
                        confidence=confidence,
                        suggested_fix=self._get_suggested_fix(pattern_name),
                    )

                    if finding.confidence >= self.min_confidence:
                        findings.append(finding)

        return findings

    def _get_vulnerability_type(self, pattern_name: str) -> str:
        """Get vulnerability type from pattern name."""
        # Override in subclasses for specific types
        return pattern_name.replace("_", " ").title()

    def _get_description(self, pattern_name: str, match) -> str:
        """Get description for finding."""
        return f"Potential {self._get_vulnerability_type(pattern_name)} detected"

    def _get_suggested_fix(self, pattern_name: str) -> Optional[str]:
        """Get suggested fix for pattern."""
        return None

    def _find_enclosing_function(
        self,
        lines: List[str],
        line_num: int,
        analysis: Optional[AnalysisResult],
    ) -> Optional[str]:
        """Find the function containing the given line."""
        if analysis and analysis.functions:
            for func in analysis.functions:
                if func.line <= line_num:
                    if func.end_line is None or func.end_line >= line_num:
                        return func.name
        return None


class CompositeDetector(BaseDetector):
    """
    Detector that combines multiple sub-detectors.

    Useful for grouping related detectors together.
    """

    def __init__(
        self,
        detectors: List[BaseDetector],
        enabled: bool = True,
        min_confidence: float = 0.5,
    ):
        super().__init__(enabled, min_confidence)
        self.detectors = detectors

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Run all sub-detectors and combine findings."""
        if not self.enabled:
            return []

        all_findings = []
        for detector in self.detectors:
            if detector.supports_language(language):
                findings = detector.detect(code, language, analysis)
                all_findings.extend(findings)

        # Deduplicate findings at same location
        return self._deduplicate_findings(all_findings)

    def _deduplicate_findings(
        self, findings: List[SecurityFinding]
    ) -> List[SecurityFinding]:
        """Remove duplicate findings at same location."""
        seen = set()
        unique = []

        for finding in findings:
            key = (
                finding.cwe_id,
                finding.location.line if finding.location else None,
                finding.vulnerability_type,
            )
            if key not in seen:
                seen.add(key)
                unique.append(finding)

        return unique


# Export commonly used items
__all__ = [
    "BaseDetector",
    "PatternBasedDetector",
    "CompositeDetector",
    "register_detector",
    "get_detector",
    "list_detectors",
]
