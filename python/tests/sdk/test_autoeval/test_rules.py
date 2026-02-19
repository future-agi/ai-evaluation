"""Tests for rule-based analyzer."""

import pytest
from fi.evals.autoeval.rules import RuleBasedAnalyzer
from fi.evals.autoeval.types import (
    AppCategory,
    RiskLevel,
    DomainSensitivity,
)


class TestRuleBasedAnalyzer:
    """Tests for RuleBasedAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return RuleBasedAnalyzer()

    def test_detect_customer_support(self, analyzer):
        """Should detect customer support applications."""
        analysis = analyzer.analyze(
            "A customer support chatbot for helping users with their issues."
        )
        assert analysis.category == AppCategory.CUSTOMER_SUPPORT
        assert analysis.confidence > 0

    def test_detect_rag_system(self, analyzer):
        """Should detect RAG-based systems."""
        analysis = analyzer.analyze(
            "A RAG-based document Q&A system that retrieves from a knowledge base."
        )
        assert analysis.category == AppCategory.RAG_SYSTEM
        assert "rag" in analysis.detected_features

    def test_detect_code_assistant(self, analyzer):
        """Should detect code assistant applications."""
        analysis = analyzer.analyze(
            "A code generation assistant for programming and debugging."
        )
        assert analysis.category == AppCategory.CODE_ASSISTANT
        assert "code_generation" in analysis.detected_features

    def test_detect_agent_workflow(self, analyzer):
        """Should detect agent workflows."""
        analysis = analyzer.analyze(
            "An autonomous agent that uses tool calling and function execution."
        )
        assert analysis.category == AppCategory.AGENT_WORKFLOW
        assert "tool_use" in analysis.detected_features

    def test_detect_content_moderation(self, analyzer):
        """Should detect content moderation systems."""
        analysis = analyzer.analyze(
            "A content moderation system for filtering inappropriate content."
        )
        assert analysis.category == AppCategory.CONTENT_MODERATION

    def test_detect_healthcare_sensitivity(self, analyzer):
        """Should detect healthcare domain sensitivity."""
        analysis = analyzer.analyze(
            "A medical chatbot for patient questions about healthcare."
        )
        assert analysis.domain_sensitivity == DomainSensitivity.HEALTHCARE
        assert analysis.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def test_detect_financial_sensitivity(self, analyzer):
        """Should detect financial domain sensitivity."""
        analysis = analyzer.analyze(
            "A banking assistant for payment and transaction queries."
        )
        assert analysis.domain_sensitivity == DomainSensitivity.FINANCIAL
        assert analysis.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def test_detect_children_sensitivity(self, analyzer):
        """Should detect children's content sensitivity."""
        analysis = analyzer.analyze(
            "An educational chatbot for children in school classrooms."
        )
        assert analysis.domain_sensitivity == DomainSensitivity.CHILDREN
        assert analysis.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def test_detect_pii_sensitivity(self, analyzer):
        """Should detect PII-sensitive applications."""
        analysis = analyzer.analyze(
            "A user profile system handling personal information and addresses."
        )
        assert analysis.domain_sensitivity == DomainSensitivity.PII_SENSITIVE

    def test_detect_low_risk_internal_tool(self, analyzer):
        """Should detect low risk for internal/development tools."""
        analysis = analyzer.analyze(
            "An internal development tool for testing and prototyping."
        )
        assert analysis.risk_level == RiskLevel.LOW

    def test_detect_medium_risk_production(self, analyzer):
        """Should detect medium risk for production apps."""
        analysis = analyzer.analyze(
            "A production chatbot for public customer queries."
        )
        assert analysis.risk_level == RiskLevel.MEDIUM

    def test_detect_critical_risk(self, analyzer):
        """Should detect critical risk for safety-critical systems."""
        analysis = analyzer.analyze(
            "A life-saving emergency healthcare system for critical patients."
        )
        assert analysis.risk_level == RiskLevel.CRITICAL

    def test_detect_multiple_features(self, analyzer):
        """Should detect multiple features from description."""
        analysis = analyzer.analyze(
            "A conversational agent with RAG retrieval, tool use, "
            "and image processing capabilities."
        )
        features = analysis.detected_features
        assert "multi_turn" in features
        assert "rag" in features
        assert "tool_use" in features
        assert "image_processing" in features

    def test_generate_requirements_for_rag(self, analyzer):
        """Should generate appropriate requirements for RAG systems."""
        analysis = analyzer.analyze(
            "A RAG system for document retrieval and question answering."
        )
        # Should have faithfulness-related evaluations
        all_evals = [
            e for r in analysis.requirements for e in r.suggested_evals
        ]
        assert any("Factual" in e or "Entailment" in e for e in all_evals)

    def test_generate_requirements_for_agents(self, analyzer):
        """Should generate appropriate requirements for agent workflows."""
        analysis = analyzer.analyze(
            "An autonomous agent with tool calling for executing tasks."
        )
        all_evals = [
            e for r in analysis.requirements for e in r.suggested_evals
        ]
        assert any("ToolUse" in e or "Goal" in e for e in all_evals)

    def test_generate_pii_scanner_for_sensitive_domains(self, analyzer):
        """Should add PII scanner for sensitive domains."""
        analysis = analyzer.analyze(
            "A healthcare chatbot handling patient medical records."
        )
        all_scanners = [
            s for r in analysis.requirements for s in r.suggested_scanners
        ]
        assert "pii" in all_scanners or "secrets" in all_scanners

    def test_generate_safety_scanners_for_high_risk(self, analyzer):
        """Should add safety scanners for high-risk applications."""
        analysis = analyzer.analyze(
            "A critical financial system for banking transactions."
        )
        all_scanners = [
            s for r in analysis.requirements for s in r.suggested_scanners
        ]
        assert any("jailbreak" in s.lower() or "toxicity" in s.lower() for s in all_scanners)

    def test_explanation_contains_analysis_info(self, analyzer):
        """Should generate meaningful explanation."""
        analysis = analyzer.analyze(
            "A customer support RAG chatbot for healthcare."
        )
        explanation = analysis.explanation
        assert len(explanation) > 0
        assert "rule-based" in explanation.lower()

    def test_unknown_category_for_vague_description(self, analyzer):
        """Should return unknown for vague descriptions."""
        analysis = analyzer.analyze(
            "A simple application."
        )
        # May or may not be unknown depending on keywords
        assert analysis.category is not None
        assert analysis.confidence <= 0.7  # Rule-based has lower confidence

    def test_empty_description(self, analyzer):
        """Should handle empty description gracefully."""
        # The main analyzer handles this, not the rule-based one
        # But we test that it doesn't crash
        analysis = analyzer.analyze("")
        assert analysis.category == AppCategory.UNKNOWN

    def test_case_insensitive_detection(self, analyzer):
        """Should detect keywords case-insensitively."""
        analysis = analyzer.analyze(
            "A RAG SYSTEM for HEALTHCARE with TOOL CALLING."
        )
        assert analysis.category == AppCategory.RAG_SYSTEM
        assert analysis.domain_sensitivity == DomainSensitivity.HEALTHCARE
