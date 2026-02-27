"""Tests for fi.evals.framework.evals.semantic module."""

import pytest
from fi.evals.framework.evals.semantic import (
    CoherenceEval,
    SemanticEvalResult,
)
from fi.evals.framework.protocols import EvalRegistry


class TestSemanticEvalResult:
    """Tests for SemanticEvalResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = SemanticEvalResult(
            score=0.85,
            passed=True,
        )

        assert result.score == 0.85
        assert result.passed is True
        assert result.confidence == 1.0
        assert result.details == {}

    def test_with_details(self):
        """Test result with details."""
        result = SemanticEvalResult(
            score=0.7,
            passed=True,
            confidence=0.9,
            details={"method": "embedding"},
        )

        assert result.details == {"method": "embedding"}
        assert result.confidence == 0.9


class TestCoherenceEval:
    """Tests for CoherenceEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = CoherenceEval()

        assert eval.name == "coherence"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = CoherenceEval()
        errors = eval.validate_inputs({})

        assert "response" in str(errors)

    def test_evaluate_single_sentence(self):
        """Test evaluation with single sentence."""
        eval = CoherenceEval()
        result = eval.evaluate({
            "response": "This is a single sentence.",
        })

        assert isinstance(result, SemanticEvalResult)
        assert result.score == 1.0  # Single sentence is coherent
        assert result.passed is True

    def test_evaluate_coherent_text(self):
        """Test evaluation with coherent text."""
        eval = CoherenceEval()
        result = eval.evaluate({
            "response": "The sun rises in the east. It sets in the west. This daily cycle creates day and night.",
        })

        assert isinstance(result, SemanticEvalResult)
        assert 0.0 <= result.score <= 1.0

    def test_evaluate_with_context(self):
        """Test evaluation with context."""
        eval = CoherenceEval()
        result = eval.evaluate({
            "response": "Paris is beautiful. The Eiffel Tower is iconic.",
            "context": "We are discussing French landmarks.",
        })

        assert isinstance(result, SemanticEvalResult)

    def test_get_span_attributes(self):
        """Test span attributes generation."""
        eval = CoherenceEval()
        result = SemanticEvalResult(score=0.85, passed=True)

        attrs = eval.get_span_attributes(result)

        assert attrs["score"] == 0.85
        assert attrs["passed"] is True
        assert attrs["threshold"] == 0.6


class TestIntegrationWithFramework:
    """Tests for integration with the evaluation framework."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_use_with_evaluator(self):
        """Test using semantic evals with Evaluator."""
        from fi.evals.framework import Evaluator, ExecutionMode

        evaluator = Evaluator(
            evaluations=[
                CoherenceEval(),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "response": "The quick brown fox jumps over the lazy dog. It was a sunny day.",
        })

        assert len(result.results) == 1
        assert hasattr(result.results[0].value, "score")
        assert hasattr(result.results[0].value, "passed")

    def test_span_attributes_format(self):
        """Test span attributes are OTEL-compatible."""
        eval = CoherenceEval()
        result = SemanticEvalResult(
            score=0.85,
            passed=True,
            confidence=0.95,
        )

        attrs = eval.get_span_attributes(result)

        # All values should be OTEL-compatible types
        for key, value in attrs.items():
            assert isinstance(value, (str, int, float, bool))
