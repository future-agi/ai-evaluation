"""Tests for fi.evals.framework.evals.semantic module."""

import pytest
from fi.evals.framework.evals.semantic import (
    SemanticSimilarityEval,
    CoherenceEval,
    EntailmentEval,
    ContradictionEval,
    FactualConsistencyEval,
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


class TestSemanticSimilarityEval:
    """Tests for SemanticSimilarityEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = SemanticSimilarityEval()

        assert eval.name == "semantic_similarity"
        assert eval.version == "1.0.0"

    def test_default_threshold(self):
        """Test default threshold."""
        eval = SemanticSimilarityEval()
        assert eval.threshold == 0.7

    def test_custom_threshold(self):
        """Test custom threshold."""
        eval = SemanticSimilarityEval(threshold=0.9)
        assert eval.threshold == 0.9

    def test_required_fields(self):
        """Test required field validation."""
        eval = SemanticSimilarityEval()
        errors = eval.validate_inputs({})

        assert "response" in str(errors)
        assert "reference" in str(errors)

    def test_evaluate_identical_text(self):
        """Test evaluation with identical texts."""
        eval = SemanticSimilarityEval()
        result = eval.evaluate({
            "response": "The quick brown fox.",
            "reference": "The quick brown fox.",
        })

        assert isinstance(result, SemanticEvalResult)
        assert result.score == 1.0  # Word overlap = 1.0 for identical
        assert result.passed is True

    def test_evaluate_similar_text(self):
        """Test evaluation with similar texts."""
        eval = SemanticSimilarityEval()
        result = eval.evaluate({
            "response": "The quick brown fox jumps.",
            "reference": "The fast brown fox leaps.",
        })

        assert isinstance(result, SemanticEvalResult)
        assert 0.0 <= result.score <= 1.0

    def test_evaluate_different_text(self):
        """Test evaluation with different texts."""
        eval = SemanticSimilarityEval()
        result = eval.evaluate({
            "response": "Hello world",
            "reference": "Goodbye universe",
        })

        assert isinstance(result, SemanticEvalResult)
        assert result.score < 0.5  # Low overlap

    def test_get_span_attributes(self):
        """Test span attributes generation."""
        eval = SemanticSimilarityEval()
        result = SemanticEvalResult(score=0.85, passed=True)

        attrs = eval.get_span_attributes(result)

        assert attrs["score"] == 0.85
        assert attrs["passed"] is True
        assert attrs["threshold"] == 0.7

    def test_registered_in_registry(self):
        """Test evaluation is registered."""
        from fi.evals.framework.evals.semantic import SemanticSimilarityEval
        from fi.evals.framework.protocols import register_evaluation

        # Register the class (import doesn't re-register since module was already loaded)
        register_evaluation(SemanticSimilarityEval)

        assert EvalRegistry.is_registered("semantic_similarity")


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


class TestEntailmentEval:
    """Tests for EntailmentEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = EntailmentEval()

        assert eval.name == "entailment"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = EntailmentEval()
        errors = eval.validate_inputs({})

        assert "response" in str(errors)
        assert "context" in str(errors)

    def test_evaluate_entailed(self):
        """Test evaluation with entailed response."""
        eval = EntailmentEval()
        result = eval.evaluate({
            "context": "Paris is the capital of France. France is in Europe.",
            "response": "Paris is in Europe.",
        })

        assert isinstance(result, SemanticEvalResult)
        assert 0.0 <= result.score <= 1.0

    def test_evaluate_not_entailed(self):
        """Test evaluation with non-entailed response."""
        eval = EntailmentEval()
        result = eval.evaluate({
            "context": "The sky is blue.",
            "response": "Elephants can fly.",
        })

        assert isinstance(result, SemanticEvalResult)


class TestContradictionEval:
    """Tests for ContradictionEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = ContradictionEval()

        assert eval.name == "contradiction"
        assert eval.version == "1.0.0"

    def test_default_threshold(self):
        """Test default threshold (lower for contradiction)."""
        eval = ContradictionEval()
        assert eval.threshold == 0.3

    def test_required_fields(self):
        """Test required field validation."""
        eval = ContradictionEval()
        errors = eval.validate_inputs({})

        assert "response" in str(errors)
        assert "context" in str(errors)

    def test_evaluate_no_contradiction(self):
        """Test evaluation with no contradiction."""
        eval = ContradictionEval()
        result = eval.evaluate({
            "context": "The meeting is at 3 PM.",
            "response": "The meeting starts at 3 PM.",
        })

        assert isinstance(result, SemanticEvalResult)
        # Low contradiction score should pass
        # passed = contradiction_score < threshold

    def test_evaluate_with_contradiction(self):
        """Test evaluation with contradiction."""
        eval = ContradictionEval()
        result = eval.evaluate({
            "context": "The car is red.",
            "response": "The car is not red.",
        })

        assert isinstance(result, SemanticEvalResult)
        # Higher contradiction expected due to negation

    def test_passed_logic(self):
        """Test that passed means LOW contradiction."""
        eval = ContradictionEval(threshold=0.5)

        # Low contradiction should pass
        result = SemanticEvalResult(score=0.2, passed=True)
        assert result.score < eval.threshold

        # High contradiction should fail
        result = SemanticEvalResult(score=0.7, passed=False)
        assert result.score > eval.threshold


class TestFactualConsistencyEval:
    """Tests for FactualConsistencyEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = FactualConsistencyEval()

        assert eval.name == "factual_consistency"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = FactualConsistencyEval()
        errors = eval.validate_inputs({})

        assert "response" in str(errors)
        assert "context" in str(errors)

    def test_evaluate_consistent(self):
        """Test evaluation with consistent response."""
        eval = FactualConsistencyEval()
        result = eval.evaluate({
            "context": "John was born in 1990 in New York City.",
            "response": "John was born in New York in 1990.",
        })

        assert isinstance(result, SemanticEvalResult)
        assert 0.0 <= result.score <= 1.0

    def test_evaluate_empty_response(self):
        """Test evaluation with very short response."""
        eval = FactualConsistencyEval()
        result = eval.evaluate({
            "context": "Some context here.",
            "response": "OK.",  # Too short to extract claims
        })

        assert isinstance(result, SemanticEvalResult)
        # No claims extracted, should return 1.0

    def test_evaluate_multiple_claims(self):
        """Test evaluation with multiple claims."""
        eval = FactualConsistencyEval()
        result = eval.evaluate({
            "context": "Paris is the capital of France. It has the Eiffel Tower. The population is about 2 million.",
            "response": "Paris is the French capital. The Eiffel Tower is located there.",
        })

        assert isinstance(result, SemanticEvalResult)
        assert "claim_count" in result.details


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
                SemanticSimilarityEval(),
                CoherenceEval(),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "response": "The quick brown fox jumps over the lazy dog.",
            "reference": "A fast brown fox leaps over a sleepy dog.",
        })

        assert len(result.results) == 2
        # Both should return SemanticEvalResult
        for r in result.results:
            assert hasattr(r.value, "score")
            assert hasattr(r.value, "passed")

    def test_use_with_async_evaluator(self):
        """Test using semantic evals with async evaluator."""
        from fi.evals.framework import async_evaluator

        evaluator = async_evaluator(
            SemanticSimilarityEval(),
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "response": "Hello world",
            "reference": "Hello world",
        })

        batch = result.wait()
        assert len(batch.results) == 1
        assert batch.results[0].value.score == 1.0

        evaluator.shutdown()

    def test_span_attributes_format(self):
        """Test span attributes are OTEL-compatible."""
        eval = SemanticSimilarityEval()
        result = SemanticEvalResult(
            score=0.85,
            passed=True,
            confidence=0.95,
        )

        attrs = eval.get_span_attributes(result)

        # All values should be OTEL-compatible types
        for key, value in attrs.items():
            assert isinstance(value, (str, int, float, bool))
