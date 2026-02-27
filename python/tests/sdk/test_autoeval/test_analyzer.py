"""Tests for AppAnalyzer (LLM-powered and rule-based)."""

import pytest
from unittest.mock import Mock, patch
import json

from fi.evals.autoeval.analyzer import AppAnalyzer
from fi.evals.autoeval.types import (
    AppCategory,
    RiskLevel,
    DomainSensitivity,
)


class TestAppAnalyzerRuleBased:
    """Tests for rule-based analysis (no LLM)."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer without LLM (rule-based only)."""
        return AppAnalyzer(use_llm=False)

    def test_empty_description(self, analyzer):
        """Should handle empty description."""
        analysis = analyzer.analyze("")
        assert analysis.category == AppCategory.UNKNOWN
        assert analysis.confidence == 0.0

    def test_whitespace_description(self, analyzer):
        """Should handle whitespace-only description."""
        analysis = analyzer.analyze("   \n\t  ")
        assert analysis.category == AppCategory.UNKNOWN
        assert analysis.confidence == 0.0

    def test_rule_based_confidence(self, analyzer):
        """Rule-based analysis should have lower confidence than LLM."""
        analysis = analyzer.analyze("A customer support chatbot.")
        assert analysis.confidence <= 0.7

    def test_detects_multiple_keywords(self, analyzer):
        """Should score based on multiple keyword matches."""
        # More specific description should still work
        analysis = analyzer.analyze(
            "A customer support help desk chatbot for customer service queries."
        )
        assert analysis.category == AppCategory.CUSTOMER_SUPPORT


class TestAppAnalyzerLLM:
    """Tests for LLM-powered analysis with mocked provider."""

    def test_llm_provider_called(self):
        """Should call LLM provider when available."""
        mock_provider = Mock()
        mock_provider.complete.return_value = json.dumps({
            "category": "rag_system",
            "risk_level": "high",
            "domain_sensitivity": "healthcare",
            "requirements": [],
            "detected_features": ["rag"],
            "explanation": "LLM detected RAG system",
        })

        analyzer = AppAnalyzer(llm_provider=mock_provider, use_llm=True)
        analysis = analyzer.analyze("A document Q&A system.")

        mock_provider.complete.assert_called_once()
        assert analysis.category == AppCategory.RAG_SYSTEM
        assert analysis.confidence == 0.85  # LLM confidence

    def test_llm_fallback_on_error(self):
        """Should fall back to rule-based when LLM fails."""
        mock_provider = Mock()
        mock_provider.complete.side_effect = Exception("API Error")

        analyzer = AppAnalyzer(llm_provider=mock_provider, use_llm=True)
        analysis = analyzer.analyze("A customer support chatbot.")

        # Should still work via fallback
        assert analysis.category == AppCategory.CUSTOMER_SUPPORT
        assert analysis.confidence <= 0.7  # Rule-based confidence

    def test_llm_fallback_on_invalid_json(self):
        """Should fall back when LLM returns invalid JSON."""
        mock_provider = Mock()
        mock_provider.complete.return_value = "This is not valid JSON!"

        analyzer = AppAnalyzer(llm_provider=mock_provider, use_llm=True)
        analysis = analyzer.analyze("A RAG system for documents.")

        # Should fall back to rule-based
        assert analysis.category == AppCategory.RAG_SYSTEM
        assert analysis.confidence <= 0.7

    def test_llm_with_markdown_code_block(self):
        """Should handle LLM response wrapped in markdown code block."""
        mock_provider = Mock()
        mock_provider.complete.return_value = """```json
{
    "category": "agent_workflow",
    "risk_level": "high",
    "domain_sensitivity": "general",
    "requirements": [],
    "detected_features": ["tool_use"],
    "explanation": "Agent with tools"
}
```"""

        analyzer = AppAnalyzer(llm_provider=mock_provider, use_llm=True)
        analysis = analyzer.analyze("An agent with tool calling.")

        assert analysis.category == AppCategory.AGENT_WORKFLOW

    def test_llm_partial_response(self):
        """Should use defaults for missing fields in LLM response."""
        mock_provider = Mock()
        mock_provider.complete.return_value = json.dumps({
            "category": "chatbot",
            # Missing other fields
        })

        analyzer = AppAnalyzer(llm_provider=mock_provider, use_llm=True)
        analysis = analyzer.analyze("A chatbot.")

        assert analysis.category == AppCategory.CHATBOT
        assert analysis.risk_level == RiskLevel.MEDIUM  # Default
        assert analysis.domain_sensitivity == DomainSensitivity.GENERAL  # Default

    def test_llm_invalid_enum_value(self):
        """Should use defaults for invalid enum values."""
        mock_provider = Mock()
        mock_provider.complete.return_value = json.dumps({
            "category": "invalid_category",
            "risk_level": "super_high",
            "domain_sensitivity": "unknown_domain",
            "requirements": [],
            "detected_features": [],
            "explanation": "Test",
        })

        analyzer = AppAnalyzer(llm_provider=mock_provider, use_llm=True)
        analysis = analyzer.analyze("Test app.")

        assert analysis.category == AppCategory.UNKNOWN
        assert analysis.risk_level == RiskLevel.MEDIUM
        assert analysis.domain_sensitivity == DomainSensitivity.GENERAL


class TestAppAnalyzerProviderInterfaces:
    """Tests for different LLM provider interfaces."""

    def test_provider_with_complete_method(self):
        """Should work with providers using complete()."""
        mock_provider = Mock()
        mock_provider.complete.return_value = json.dumps({
            "category": "chatbot",
            "risk_level": "medium",
            "domain_sensitivity": "general",
            "requirements": [],
            "detected_features": [],
            "explanation": "Test",
        })

        analyzer = AppAnalyzer(llm_provider=mock_provider)
        analysis = analyzer.analyze("A chatbot.")
        assert analysis.category == AppCategory.CHATBOT

    def test_provider_with_generate_method(self):
        """Should work with providers using generate()."""
        mock_provider = Mock(spec=["generate"])
        mock_provider.generate.return_value = json.dumps({
            "category": "rag_system",
            "risk_level": "medium",
            "domain_sensitivity": "general",
            "requirements": [],
            "detected_features": [],
            "explanation": "Test",
        })

        analyzer = AppAnalyzer(llm_provider=mock_provider)
        analysis = analyzer.analyze("A RAG system.")
        assert analysis.category == AppCategory.RAG_SYSTEM

    def test_callable_provider(self):
        """Should work with callable providers."""
        def mock_callable(prompt, system_prompt):
            return json.dumps({
                "category": "code_assistant",
                "risk_level": "medium",
                "domain_sensitivity": "general",
                "requirements": [],
                "detected_features": [],
                "explanation": "Test",
            })

        analyzer = AppAnalyzer(llm_provider=mock_callable)
        analysis = analyzer.analyze("A code assistant.")
        assert analysis.category == AppCategory.CODE_ASSISTANT


class TestAppAnalyzerConfiguration:
    """Tests for analyzer configuration options."""

    def test_use_llm_false_ignores_provider(self):
        """Should use rule-based when use_llm=False even with provider."""
        mock_provider = Mock()
        analyzer = AppAnalyzer(llm_provider=mock_provider, use_llm=False)
        analysis = analyzer.analyze("A chatbot.")

        # Should not call provider
        mock_provider.complete.assert_not_called()
        mock_provider.generate.assert_not_called()

        # Should still work via rules
        assert analysis.category == AppCategory.CHATBOT

    def test_no_provider_uses_rules(self):
        """Should use rule-based when no provider given."""
        analyzer = AppAnalyzer(llm_provider=None, use_llm=True)
        analysis = analyzer.analyze("A RAG system with retrieval.")

        assert analysis.category == AppCategory.RAG_SYSTEM
        assert analysis.confidence <= 0.7

    def test_custom_model(self):
        """Should pass custom model to provider."""
        mock_provider = Mock()
        mock_provider.complete.return_value = json.dumps({
            "category": "chatbot",
            "risk_level": "medium",
            "domain_sensitivity": "general",
            "requirements": [],
            "detected_features": [],
            "explanation": "Test",
        })

        analyzer = AppAnalyzer(
            llm_provider=mock_provider,
            model="gpt-4-turbo",
            temperature=0.0,
        )
        analyzer.analyze("A chatbot.")

        # Check that custom settings were passed
        call_kwargs = mock_provider.complete.call_args.kwargs
        assert call_kwargs.get("model") == "gpt-4-turbo"
        assert call_kwargs.get("temperature") == 0.0


class TestAppAnalyzerRequirementsParsing:
    """Tests for parsing requirements from LLM response."""

    def test_parse_requirements(self):
        """Should parse requirements from LLM response."""
        mock_provider = Mock()
        mock_provider.complete.return_value = json.dumps({
            "category": "rag_system",
            "risk_level": "high",
            "domain_sensitivity": "healthcare",
            "requirements": [
                {
                    "category": "faithfulness",
                    "importance": "required",
                    "reason": "Medical accuracy is critical",
                    "suggested_evals": ["answer_relevancy"],
                    "suggested_scanners": ["PIIScanner"],
                },
            ],
            "detected_features": ["rag"],
            "explanation": "Healthcare RAG system",
        })

        analyzer = AppAnalyzer(llm_provider=mock_provider)
        analysis = analyzer.analyze("Healthcare document system.")

        assert len(analysis.requirements) == 1
        req = analysis.requirements[0]
        assert req.category == "faithfulness"
        assert req.importance == "required"
        assert "answer_relevancy" in req.suggested_evals
        assert "PIIScanner" in req.suggested_scanners

    def test_parse_multiple_requirements(self):
        """Should parse multiple requirements."""
        mock_provider = Mock()
        mock_provider.complete.return_value = json.dumps({
            "category": "agent_workflow",
            "risk_level": "high",
            "domain_sensitivity": "general",
            "requirements": [
                {
                    "category": "tool_use",
                    "importance": "required",
                    "reason": "Agent needs correct tool use",
                    "suggested_evals": ["action_safety"],
                },
                {
                    "category": "safety",
                    "importance": "required",
                    "reason": "Agent actions must be safe",
                    "suggested_evals": ["action_safety"],
                    "suggested_scanners": ["JailbreakScanner"],
                },
            ],
            "detected_features": ["tool_use"],
            "explanation": "Agent workflow",
        })

        analyzer = AppAnalyzer(llm_provider=mock_provider)
        analysis = analyzer.analyze("An autonomous agent.")

        assert len(analysis.requirements) == 2
