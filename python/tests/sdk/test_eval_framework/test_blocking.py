"""Tests for fi.evals.framework.evaluators.blocking module."""

import pytest
import time
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch
from fi.evals.framework.evaluators.blocking import BlockingEvaluator, blocking_evaluate
from fi.evals.framework.types import EvalResult, EvalStatus, BatchEvalResult


# Test evaluation implementations
class SimpleScoreEval:
    """Simple evaluation that returns a score."""
    name = "simple_score"
    version = "1.0.0"

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs.get("text", "")
        score = min(len(text) / 100, 1.0)
        return {"score": score, "passed": score > 0.5}

    def get_span_attributes(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return result


class FailingEval:
    """Evaluation that always fails."""
    name = "failing_eval"
    version = "1.0.0"

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("Intentional failure")

    def get_span_attributes(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class SlowEval:
    """Evaluation that takes some time."""
    name = "slow_eval"
    version = "1.0.0"

    def __init__(self, delay_ms: float = 10):
        self.delay_ms = delay_ms

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(self.delay_ms / 1000)
        return {"completed": True}

    def get_span_attributes(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return result


class ValidatingEval:
    """Evaluation with input validation."""
    name = "validating_eval"
    version = "1.0.0"

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": inputs["required_field"] * 2}

    def get_span_attributes(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return result

    def validate_inputs(self, inputs: Dict[str, Any]) -> Optional[str]:
        if "required_field" not in inputs:
            return "Missing required_field"
        return None


class TestBlockingEvaluator:
    """Tests for BlockingEvaluator class."""

    def test_init_with_evaluations(self):
        """Test initialization with evaluations."""
        evaluator = BlockingEvaluator([SimpleScoreEval()])
        assert len(evaluator) == 1

    def test_init_empty(self):
        """Test initialization without evaluations."""
        evaluator = BlockingEvaluator()
        assert len(evaluator) == 0

    def test_add_evaluation(self):
        """Test adding evaluations."""
        evaluator = BlockingEvaluator()
        evaluator.add_evaluation(SimpleScoreEval())
        evaluator.add_evaluation(SlowEval())

        assert len(evaluator) == 2

    def test_add_evaluation_chaining(self):
        """Test add_evaluation returns self for chaining."""
        evaluator = (
            BlockingEvaluator()
            .add_evaluation(SimpleScoreEval())
            .add_evaluation(SlowEval())
        )
        assert len(evaluator) == 2

    def test_evaluate_basic(self):
        """Test basic evaluation."""
        evaluator = BlockingEvaluator(
            [SimpleScoreEval()],
            auto_enrich_span=False,
        )
        results = evaluator.evaluate({"text": "Hello world" * 10})

        assert len(results) == 1
        assert results[0].eval_name == "simple_score"
        assert results[0].status == EvalStatus.COMPLETED
        assert results[0].value["score"] > 0

    def test_evaluate_multiple(self):
        """Test evaluating with multiple evaluations."""
        evaluator = BlockingEvaluator(
            [SimpleScoreEval(), SlowEval(delay_ms=5)],
            auto_enrich_span=False,
        )
        results = evaluator.evaluate({"text": "test"})

        assert len(results) == 2
        assert results[0].eval_name == "simple_score"
        assert results[1].eval_name == "slow_eval"

    def test_evaluate_no_evaluations_raises(self):
        """Test that evaluating without evaluations raises."""
        evaluator = BlockingEvaluator(auto_enrich_span=False)

        with pytest.raises(ValueError, match="No evaluations"):
            evaluator.evaluate({"text": "test"})

    def test_evaluate_with_passed_evaluations(self):
        """Test passing evaluations to evaluate()."""
        evaluator = BlockingEvaluator(auto_enrich_span=False)
        results = evaluator.evaluate(
            {"text": "test"},
            evaluations=[SimpleScoreEval()],
        )

        assert len(results) == 1

    def test_evaluate_handles_failure(self):
        """Test that failures are caught and recorded."""
        evaluator = BlockingEvaluator(
            [FailingEval()],
            auto_enrich_span=False,
        )
        results = evaluator.evaluate({})

        assert len(results) == 1
        assert results[0].status == EvalStatus.FAILED
        assert "Intentional failure" in results[0].error

    def test_evaluate_fail_fast_true(self):
        """Test fail_fast=True stops on first failure."""
        evaluator = BlockingEvaluator(
            [FailingEval(), SimpleScoreEval(), SlowEval()],
            auto_enrich_span=False,
            fail_fast=True,
        )
        results = evaluator.evaluate({"text": "test"})

        assert len(results) == 3
        assert results[0].status == EvalStatus.FAILED
        assert results[1].status == EvalStatus.CANCELLED
        assert results[2].status == EvalStatus.CANCELLED

    def test_evaluate_fail_fast_false(self):
        """Test fail_fast=False continues after failure."""
        evaluator = BlockingEvaluator(
            [FailingEval(), SimpleScoreEval()],
            auto_enrich_span=False,
            fail_fast=False,
        )
        results = evaluator.evaluate({"text": "test"})

        assert len(results) == 2
        assert results[0].status == EvalStatus.FAILED
        assert results[1].status == EvalStatus.COMPLETED

    def test_evaluate_records_latency(self):
        """Test that latency is recorded."""
        evaluator = BlockingEvaluator(
            [SlowEval(delay_ms=20)],
            auto_enrich_span=False,
        )
        results = evaluator.evaluate({})

        assert results[0].latency_ms >= 15  # Allow some variance

    def test_evaluate_input_validation_pass(self):
        """Test input validation passes."""
        evaluator = BlockingEvaluator(
            [ValidatingEval()],
            auto_enrich_span=False,
            validate_inputs=True,
        )
        results = evaluator.evaluate({"required_field": 5})

        assert results[0].status == EvalStatus.COMPLETED
        assert results[0].value["value"] == 10

    def test_evaluate_input_validation_fail(self):
        """Test input validation failure."""
        evaluator = BlockingEvaluator(
            [ValidatingEval()],
            auto_enrich_span=False,
            validate_inputs=True,
        )
        results = evaluator.evaluate({})

        assert results[0].status == EvalStatus.FAILED
        assert "Validation error" in results[0].error
        assert "required_field" in results[0].error

    def test_evaluate_input_validation_disabled(self):
        """Test that validation can be disabled."""
        evaluator = BlockingEvaluator(
            [ValidatingEval()],
            auto_enrich_span=False,
            validate_inputs=False,
        )
        results = evaluator.evaluate({})

        # Should fail during evaluate, not validation
        assert results[0].status == EvalStatus.FAILED
        assert "Validation error" not in results[0].error

    def test_evaluate_single(self):
        """Test evaluate_single method."""
        evaluator = BlockingEvaluator(auto_enrich_span=False)
        result = evaluator.evaluate_single(
            SimpleScoreEval(),
            {"text": "Hello world"},
        )

        assert isinstance(result, EvalResult)
        assert result.eval_name == "simple_score"
        assert result.status == EvalStatus.COMPLETED

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = BlockingEvaluator(
            [SimpleScoreEval()],
            auto_enrich_span=False,
        )
        inputs_batch = [
            {"text": "short"},
            {"text": "medium length text here"},
            {"text": "this is a much longer text " * 10},
        ]
        batch_result = evaluator.evaluate_batch(inputs_batch)

        assert isinstance(batch_result, BatchEvalResult)
        assert batch_result.total_count == 3
        assert batch_result.success_count == 3

    def test_evaluate_batch_with_failures(self):
        """Test batch evaluation with some failures."""
        evaluator = BlockingEvaluator(
            [FailingEval()],
            auto_enrich_span=False,
        )
        batch_result = evaluator.evaluate_batch([{}, {}, {}])

        assert batch_result.total_count == 3
        assert batch_result.failure_count == 3
        assert batch_result.success_rate == 0.0

    def test_iter(self):
        """Test iterating over evaluator."""
        evals = [SimpleScoreEval(), SlowEval()]
        evaluator = BlockingEvaluator(evals)

        iterated = list(evaluator)
        assert len(iterated) == 2

    def test_auto_enrich_span(self):
        """Test auto span enrichment."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch('fi.evals.framework.evaluators.blocking.enrich_current_span') as mock_enrich:
            evaluator = BlockingEvaluator(
                [SimpleScoreEval()],
                auto_enrich_span=True,
            )
            evaluator.evaluate({"text": "test"})

            assert mock_enrich.called

    def test_auto_enrich_span_disabled(self):
        """Test disabling auto span enrichment."""
        with patch('fi.evals.framework.evaluators.blocking.enrich_current_span') as mock_enrich:
            evaluator = BlockingEvaluator(
                [SimpleScoreEval()],
                auto_enrich_span=False,
            )
            evaluator.evaluate({"text": "test"})

            assert not mock_enrich.called

    def test_result_includes_version(self):
        """Test that result includes eval version."""
        evaluator = BlockingEvaluator(
            [SimpleScoreEval()],
            auto_enrich_span=False,
        )
        results = evaluator.evaluate({"text": "test"})

        assert results[0].eval_version == "1.0.0"


class TestBlockingEvaluate:
    """Tests for blocking_evaluate convenience function."""

    def test_basic_usage(self):
        """Test basic usage of blocking_evaluate."""
        results = blocking_evaluate(
            {"text": "Hello world"},
            SimpleScoreEval(),
            auto_enrich_span=False,
        )

        assert len(results) == 1
        assert results[0].status == EvalStatus.COMPLETED

    def test_multiple_evaluations(self):
        """Test with multiple evaluations."""
        results = blocking_evaluate(
            {"text": "test"},
            SimpleScoreEval(),
            SlowEval(delay_ms=5),
            auto_enrich_span=False,
        )

        assert len(results) == 2

    def test_auto_enrich_default(self):
        """Test that auto_enrich_span defaults to True."""
        with patch('fi.evals.framework.evaluators.blocking.enrich_current_span') as mock_enrich:
            blocking_evaluate(
                {"text": "test"},
                SimpleScoreEval(),
            )
            # Should be called by default
            assert mock_enrich.called


class TestEvalNameExtraction:
    """Tests for evaluation name extraction."""

    def test_uses_name_attribute(self):
        """Test that name attribute is used."""
        evaluator = BlockingEvaluator(
            [SimpleScoreEval()],
            auto_enrich_span=False,
        )
        results = evaluator.evaluate({"text": "test"})

        assert results[0].eval_name == "simple_score"

    def test_fallback_to_class_name(self):
        """Test fallback to class name when no name attribute."""
        class NoNameEval:
            version = "1.0.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        evaluator = BlockingEvaluator(
            [NoNameEval()],
            auto_enrich_span=False,
        )
        results = evaluator.evaluate({})

        assert results[0].eval_name == "NoNameEval"


class TestSpanEnrichmentContent:
    """Tests for what gets added to spans."""

    def test_enrichment_includes_latency(self):
        """Test that latency is included in enrichment."""
        captured_attrs = {}

        def capture_enrich(name, attrs, **kwargs):
            captured_attrs.update(attrs)
            return True

        with patch('fi.evals.framework.evaluators.blocking.enrich_current_span', side_effect=capture_enrich):
            evaluator = BlockingEvaluator(
                [SlowEval(delay_ms=10)],
                auto_enrich_span=True,
            )
            evaluator.evaluate({})

        assert "latency_ms" in captured_attrs
        assert captured_attrs["latency_ms"] >= 5

    def test_enrichment_includes_status(self):
        """Test that status is included in enrichment."""
        captured_attrs = {}

        def capture_enrich(name, attrs, **kwargs):
            captured_attrs.update(attrs)
            return True

        with patch('fi.evals.framework.evaluators.blocking.enrich_current_span', side_effect=capture_enrich):
            evaluator = BlockingEvaluator(
                [SimpleScoreEval()],
                auto_enrich_span=True,
            )
            evaluator.evaluate({"text": "test"})

        assert captured_attrs["status"] == "completed"

    def test_enrichment_includes_error_on_failure(self):
        """Test that error is included on failure."""
        captured_attrs = {}

        def capture_enrich(name, attrs, **kwargs):
            captured_attrs.update(attrs)
            return True

        with patch('fi.evals.framework.evaluators.blocking.enrich_current_span', side_effect=capture_enrich):
            evaluator = BlockingEvaluator(
                [FailingEval()],
                auto_enrich_span=True,
            )
            evaluator.evaluate({})

        assert "error" in captured_attrs
        assert "Intentional failure" in captured_attrs["error"]

    def test_enrichment_includes_eval_attributes(self):
        """Test that evaluation-specific attributes are included."""
        captured_attrs = {}

        def capture_enrich(name, attrs, **kwargs):
            captured_attrs.update(attrs)
            return True

        with patch('fi.evals.framework.evaluators.blocking.enrich_current_span', side_effect=capture_enrich):
            evaluator = BlockingEvaluator(
                [SimpleScoreEval()],
                auto_enrich_span=True,
            )
            evaluator.evaluate({"text": "x" * 100})

        assert "score" in captured_attrs
        assert "passed" in captured_attrs
