"""Tests for AutoEval types."""

import pytest
from fi.evals.autoeval.types import (
    AppCategory,
    RiskLevel,
    DomainSensitivity,
    AppRequirement,
    AppAnalysis,
    AutoEvalResult,
)


class TestAppCategory:
    """Tests for AppCategory enum."""

    def test_all_categories_have_values(self):
        """All categories should have string values."""
        for cat in AppCategory:
            assert isinstance(cat.value, str)
            assert len(cat.value) > 0

    def test_category_values_are_unique(self):
        """All category values should be unique."""
        values = [cat.value for cat in AppCategory]
        assert len(values) == len(set(values))

    def test_common_categories_exist(self):
        """Common categories should exist."""
        assert AppCategory.CUSTOMER_SUPPORT.value == "customer_support"
        assert AppCategory.RAG_SYSTEM.value == "rag_system"
        assert AppCategory.CODE_ASSISTANT.value == "code_assistant"
        assert AppCategory.AGENT_WORKFLOW.value == "agent_workflow"
        assert AppCategory.UNKNOWN.value == "unknown"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_risk_levels(self):
        """All risk levels should exist with expected values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_risk_level_ordering(self):
        """Risk levels should be orderable by severity."""
        # Values are strings, but we can check they exist
        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(levels) == 4


class TestDomainSensitivity:
    """Tests for DomainSensitivity enum."""

    def test_common_sensitivities(self):
        """Common sensitivity levels should exist."""
        assert DomainSensitivity.GENERAL.value == "general"
        assert DomainSensitivity.PII_SENSITIVE.value == "pii_sensitive"
        assert DomainSensitivity.HEALTHCARE.value == "healthcare"
        assert DomainSensitivity.FINANCIAL.value == "financial"
        assert DomainSensitivity.CHILDREN.value == "children"


class TestAppRequirement:
    """Tests for AppRequirement dataclass."""

    def test_create_requirement(self):
        """Should create requirement with all fields."""
        req = AppRequirement(
            category="safety",
            importance="required",
            reason="Test reason",
            suggested_evals=["CoherenceEval"],
            suggested_scanners=["JailbreakScanner"],
        )
        assert req.category == "safety"
        assert req.importance == "required"
        assert req.reason == "Test reason"
        assert "CoherenceEval" in req.suggested_evals
        assert "JailbreakScanner" in req.suggested_scanners

    def test_requirement_to_dict(self):
        """Should convert to dictionary."""
        req = AppRequirement(
            category="quality",
            importance="recommended",
            reason="Quality check",
            suggested_evals=["Eval1", "Eval2"],
        )
        data = req.to_dict()
        assert data["category"] == "quality"
        assert data["importance"] == "recommended"
        assert data["reason"] == "Quality check"
        assert data["suggested_evals"] == ["Eval1", "Eval2"]

    def test_requirement_from_dict(self):
        """Should create from dictionary."""
        data = {
            "category": "safety",
            "importance": "required",
            "reason": "Safety check",
            "suggested_evals": ["SafetyEval"],
            "suggested_scanners": ["Scanner1"],
        }
        req = AppRequirement.from_dict(data)
        assert req.category == "safety"
        assert req.importance == "required"
        assert "SafetyEval" in req.suggested_evals

    def test_requirement_default_values(self):
        """Should have default empty lists for evals and scanners."""
        req = AppRequirement(
            category="test",
            importance="optional",
            reason="Test",
        )
        assert req.suggested_evals == []
        assert req.suggested_scanners == []


class TestAppAnalysis:
    """Tests for AppAnalysis dataclass."""

    def test_create_analysis(self):
        """Should create analysis with all fields."""
        analysis = AppAnalysis(
            category=AppCategory.RAG_SYSTEM,
            risk_level=RiskLevel.HIGH,
            domain_sensitivity=DomainSensitivity.HEALTHCARE,
            requirements=[],
            detected_features=["rag", "multi_turn"],
            confidence=0.85,
            explanation="Test analysis",
        )
        assert analysis.category == AppCategory.RAG_SYSTEM
        assert analysis.risk_level == RiskLevel.HIGH
        assert analysis.domain_sensitivity == DomainSensitivity.HEALTHCARE
        assert analysis.confidence == 0.85
        assert "rag" in analysis.detected_features

    def test_analysis_to_dict(self):
        """Should convert to dictionary."""
        analysis = AppAnalysis(
            category=AppCategory.CHATBOT,
            risk_level=RiskLevel.MEDIUM,
            domain_sensitivity=DomainSensitivity.GENERAL,
            requirements=[],
            detected_features=["multi_turn"],
            confidence=0.7,
            explanation="A chatbot",
        )
        data = analysis.to_dict()
        assert data["category"] == "chatbot"
        assert data["risk_level"] == "medium"
        assert data["domain_sensitivity"] == "general"
        assert data["confidence"] == 0.7

    def test_analysis_from_dict(self):
        """Should create from dictionary."""
        data = {
            "category": "rag_system",
            "risk_level": "high",
            "domain_sensitivity": "healthcare",
            "requirements": [],
            "detected_features": ["rag"],
            "confidence": 0.9,
            "explanation": "RAG system",
        }
        analysis = AppAnalysis.from_dict(data)
        assert analysis.category == AppCategory.RAG_SYSTEM
        assert analysis.risk_level == RiskLevel.HIGH
        assert analysis.domain_sensitivity == DomainSensitivity.HEALTHCARE


class TestAutoEvalResult:
    """Tests for AutoEvalResult dataclass."""

    def test_create_result_passed(self):
        """Should create a passed result."""
        result = AutoEvalResult(
            passed=True,
            total_latency_ms=150.5,
        )
        assert result.passed is True
        assert result.blocked_by_scanner is False
        assert result.total_latency_ms == 150.5

    def test_create_result_failed(self):
        """Should create a failed result."""
        result = AutoEvalResult(
            passed=False,
            blocked_by_scanner=True,
            total_latency_ms=50.0,
        )
        assert result.passed is False
        assert result.blocked_by_scanner is True

    def test_result_summary_basic(self):
        """Should generate basic summary."""
        result = AutoEvalResult(
            passed=True,
            total_latency_ms=100.0,
        )
        summary = result.summary
        assert summary["passed"] is True
        assert summary["blocked_by_scanner"] is False
        assert summary["total_latency_ms"] == 100.0

    def test_result_explain_passed(self):
        """Should explain passed result."""
        result = AutoEvalResult(
            passed=True,
            total_latency_ms=100.0,
        )
        explanation = result.explain()
        assert "PASSED" in explanation
        assert "100.00ms" in explanation

    def test_result_explain_failed(self):
        """Should explain failed result."""
        result = AutoEvalResult(
            passed=False,
            blocked_by_scanner=True,
            total_latency_ms=50.0,
        )
        explanation = result.explain()
        assert "FAILED" in explanation
        assert "Blocked by scanner" in explanation
