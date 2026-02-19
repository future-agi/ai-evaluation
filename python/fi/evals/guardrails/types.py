"""
Guardrails Types Module.

Defines response types for the guardrails system:
- GuardrailResult: Result from a single model
- GuardrailsResponse: Aggregated response from all models
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GuardrailResult:
    """Result from a single guardrail model check."""

    passed: bool
    category: str
    score: float
    model: str
    reason: Optional[str] = None
    action: str = "pass"  # "block", "flag", "redact", "warn", "pass"
    latency_ms: float = 0.0

    def __post_init__(self):
        """Validate result."""
        if not 0.0 <= self.score <= 1.0:
            # Clamp score to valid range
            self.score = max(0.0, min(1.0, self.score))


@dataclass
class GuardrailsResponse:
    """Aggregated response from all guardrails."""

    passed: bool
    results: List[GuardrailResult] = field(default_factory=list)
    blocked_categories: List[str] = field(default_factory=list)
    flagged_categories: List[str] = field(default_factory=list)
    redacted_content: Optional[str] = None
    original_content: str = ""
    total_latency_ms: float = 0.0
    models_used: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @classmethod
    def create_passed(
        cls,
        content: str,
        latency_ms: float = 0.0,
        models_used: Optional[List[str]] = None,
        results: Optional[List[GuardrailResult]] = None,
    ) -> "GuardrailsResponse":
        """Create a passed response."""
        return cls(
            passed=True,
            original_content=content,
            total_latency_ms=latency_ms,
            models_used=models_used or [],
            results=results or [],
        )

    @classmethod
    def create_blocked(
        cls,
        content: str,
        blocked_categories: List[str],
        latency_ms: float = 0.0,
        models_used: Optional[List[str]] = None,
        results: Optional[List[GuardrailResult]] = None,
        reason: Optional[str] = None,
    ) -> "GuardrailsResponse":
        """Create a blocked response."""
        return cls(
            passed=False,
            original_content=content,
            blocked_categories=blocked_categories,
            total_latency_ms=latency_ms,
            models_used=models_used or [],
            results=results or [],
        )

    @classmethod
    def create_error(
        cls,
        content: str,
        error: str,
        fail_open: bool = False,
    ) -> "GuardrailsResponse":
        """Create an error response."""
        return cls(
            passed=fail_open,  # If fail_open, allow content
            original_content=content,
            error=error,
        )
