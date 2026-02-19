"""Tests for fi.evals.framework.evaluator module."""

import pytest
import time
from unittest.mock import MagicMock

from fi.evals.framework.evaluator import (
    Evaluator,
    EvaluatorResult,
    blocking_evaluator,
    async_evaluator,
    distributed_evaluator,
    evaluate,
)
from fi.evals.framework.types import ExecutionMode, EvalResult, EvalStatus, BatchEvalResult
from fi.evals.framework.context import EvalContext
from fi.evals.framework.protocols import EvalRegistry
from fi.evals.framework.registry import SpanRegistry


class MockEvaluation:
    """Mock evaluation for testing."""

    name = "mock_eval"
    version = "1.0.0"

    def __init__(self, result=None, error=None):
        self._result = result if result is not None else {"score": 0.95}
        self._error = error

    def evaluate(self, inputs):
        if self._error:
            raise ValueError(self._error)
        return self._result

    def get_span_attributes(self, result):
        return {"score": result.get("score", 0)}


class MockEvaluationWithValidation(MockEvaluation):
    """Mock evaluation with input validation."""

    def validate_inputs(self, inputs):
        errors = []
        if "response" not in inputs:
            errors.append("Missing 'response' field")
        return errors


class TestEvaluatorResult:
    """Tests for EvaluatorResult class."""

    def test_blocking_result(self):
        """Test EvaluatorResult with immediate batch."""
        result = EvalResult(
            value={"score": 0.95},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10.0,
        )
        batch = BatchEvalResult.from_results([result])

        eval_result = EvaluatorResult(
            batch=batch,
            mode=ExecutionMode.BLOCKING,
        )

        assert eval_result.is_future is False
        assert eval_result.is_ready is True
        assert eval_result.wait() is batch

    def test_is_future_true(self):
        """Test is_future when future is set."""
        future = MagicMock()
        eval_result = EvaluatorResult(
            future=future,
            mode=ExecutionMode.NON_BLOCKING,
        )

        assert eval_result.is_future is True

    def test_is_ready_with_future(self):
        """Test is_ready with future."""
        future = MagicMock()
        future.done.return_value = False

        eval_result = EvaluatorResult(
            future=future,
            mode=ExecutionMode.NON_BLOCKING,
        )

        assert eval_result.is_ready is False

        future.done.return_value = True
        assert eval_result.is_ready is True

    def test_wait_blocking(self):
        """Test wait() with blocking result."""
        result = EvalResult(
            value={"score": 0.95},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10.0,
        )
        batch = BatchEvalResult.from_results([result])

        eval_result = EvaluatorResult(batch=batch)
        assert eval_result.wait() is batch

    def test_wait_non_blocking(self):
        """Test wait() with future."""
        result = EvalResult(
            value={"score": 0.95},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10.0,
        )
        batch = BatchEvalResult.from_results([result])

        future = MagicMock()
        future.results.return_value = batch

        eval_result = EvaluatorResult(
            future=future,
            mode=ExecutionMode.NON_BLOCKING,
        )

        assert eval_result.wait() is batch
        future.results.assert_called_once()

    def test_results_property(self):
        """Test results property."""
        result = EvalResult(
            value={"score": 0.95},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10.0,
        )
        batch = BatchEvalResult.from_results([result])

        eval_result = EvaluatorResult(batch=batch)
        assert eval_result.results == [result]

    def test_success_rate_property(self):
        """Test success_rate property."""
        result = EvalResult(
            value={"score": 0.95},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10.0,
        )
        batch = BatchEvalResult.from_results([result])

        eval_result = EvaluatorResult(batch=batch)
        assert eval_result.success_rate == 1.0


class TestEvaluator:
    """Tests for Evaluator class."""

    def setup_method(self):
        SpanRegistry.reset_instance()
        EvalRegistry.clear()

    def teardown_method(self):
        SpanRegistry.reset_instance()
        EvalRegistry.clear()

    def test_init_with_evaluations(self):
        """Test initialization with evaluations."""
        eval1 = MockEvaluation()
        evaluator = Evaluator(evaluations=[eval1])

        assert len(evaluator.evaluations) == 1
        assert evaluator.mode == ExecutionMode.BLOCKING

    def test_init_non_blocking(self):
        """Test initialization in non-blocking mode."""
        evaluator = Evaluator(
            evaluations=[MockEvaluation()],
            mode=ExecutionMode.NON_BLOCKING,
        )

        assert evaluator.mode == ExecutionMode.NON_BLOCKING

    def test_add_evaluation(self):
        """Test add() method."""
        evaluator = Evaluator()
        evaluator.add(MockEvaluation())

        assert len(evaluator.evaluations) == 1

    def test_add_chaining(self):
        """Test add() returns self for chaining."""
        evaluator = Evaluator()
        result = evaluator.add(MockEvaluation()).add(MockEvaluation())

        assert result is evaluator
        assert len(evaluator.evaluations) == 2

    def test_run_blocking(self):
        """Test run() in blocking mode."""
        evaluator = Evaluator(
            evaluations=[MockEvaluation(result={"score": 0.95})],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({"response": "test"})

        assert result.is_future is False
        assert result.is_ready is True
        assert len(result.results) == 1
        assert result.results[0].value == {"score": 0.95}

    def test_run_non_blocking(self):
        """Test run() in non-blocking mode."""
        evaluator = Evaluator(
            evaluations=[MockEvaluation(result={"score": 0.95})],
            mode=ExecutionMode.NON_BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({"response": "test"})

        assert result.is_future is True
        batch = result.wait()
        assert len(batch.results) == 1
        assert batch.results[0].value == {"score": 0.95}

        evaluator.shutdown()

    def test_run_no_evaluations_raises(self):
        """Test run() raises when no evaluations configured."""
        evaluator = Evaluator()

        with pytest.raises(ValueError, match="No evaluations"):
            evaluator.run({"response": "test"})

    def test_run_multiple_evaluations(self):
        """Test run() with multiple evaluations."""
        evaluator = Evaluator(
            evaluations=[
                MockEvaluation(result={"score": 0.9}),
                MockEvaluation(result={"score": 0.8}),
            ],
            auto_enrich_span=False,
        )

        result = evaluator.run({"response": "test"})

        assert len(result.results) == 2

    def test_run_handles_failure(self):
        """Test run() handles evaluation failures."""
        evaluator = Evaluator(
            evaluations=[MockEvaluation(error="Test error")],
            auto_enrich_span=False,
        )

        result = evaluator.run({"response": "test"})

        assert result.results[0].status == EvalStatus.FAILED

    def test_run_single(self):
        """Test run_single() method."""
        evaluator = Evaluator(auto_enrich_span=False)

        result = evaluator.run_single(
            MockEvaluation(result={"score": 0.95}),
            {"response": "test"},
        )

        assert isinstance(result, EvalResult)
        assert result.value == {"score": 0.95}

    def test_context_manager(self):
        """Test context manager protocol."""
        with Evaluator(
            evaluations=[MockEvaluation()],
            mode=ExecutionMode.NON_BLOCKING,
            auto_enrich_span=False,
        ) as evaluator:
            result = evaluator.run({"response": "test"})
            result.wait()

        # Should be shutdown

    def test_distributed_mode(self):
        """Test distributed mode falls back to non-blocking."""
        evaluator = Evaluator(
            evaluations=[MockEvaluation(result={"score": 0.95})],
            mode=ExecutionMode.DISTRIBUTED,
            auto_enrich_span=False,
        )

        result = evaluator.run({"response": "test"})

        # Should work like non-blocking
        assert result.is_future is True
        batch = result.wait()
        assert len(batch.results) == 1

        evaluator.shutdown()


class TestFactoryFunctions:
    """Tests for factory functions."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_blocking_evaluator(self):
        """Test blocking_evaluator factory."""
        evaluator = blocking_evaluator(
            MockEvaluation(),
            auto_enrich_span=False,
        )

        assert evaluator.mode == ExecutionMode.BLOCKING
        result = evaluator.run({"response": "test"})
        assert result.is_future is False

    def test_blocking_evaluator_multiple(self):
        """Test blocking_evaluator with multiple evals."""
        evaluator = blocking_evaluator(
            MockEvaluation(result={"a": 1}),
            MockEvaluation(result={"b": 2}),
            auto_enrich_span=False,
        )

        result = evaluator.run({"response": "test"})
        assert len(result.results) == 2

    def test_async_evaluator(self):
        """Test async_evaluator factory."""
        evaluator = async_evaluator(
            MockEvaluation(),
            auto_enrich_span=False,
        )

        assert evaluator.mode == ExecutionMode.NON_BLOCKING
        result = evaluator.run({"response": "test"})
        assert result.is_future is True
        result.wait()

        evaluator.shutdown()

    def test_async_evaluator_custom_workers(self):
        """Test async_evaluator with custom max_workers."""
        evaluator = async_evaluator(
            MockEvaluation(),
            max_workers=8,
            auto_enrich_span=False,
        )

        assert evaluator.max_workers == 8
        evaluator.shutdown()

    def test_distributed_evaluator(self):
        """Test distributed_evaluator factory."""
        from fi.evals.framework.backends import ThreadPoolBackend

        backend = ThreadPoolBackend()
        evaluator = distributed_evaluator(
            MockEvaluation(),
            backend=backend,
            auto_enrich_span=False,
        )

        assert evaluator.mode == ExecutionMode.DISTRIBUTED
        assert evaluator._backend is backend

        evaluator.shutdown()


class TestEvaluateFunction:
    """Tests for evaluate() convenience function."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_evaluate_blocking(self):
        """Test evaluate() in blocking mode."""
        result = evaluate(
            {"response": "test"},
            MockEvaluation(result={"score": 0.95}),
            auto_enrich_span=False,
        )

        assert result.is_future is False
        assert result.results[0].value == {"score": 0.95}

    def test_evaluate_multiple(self):
        """Test evaluate() with multiple evaluations."""
        result = evaluate(
            {"response": "test"},
            MockEvaluation(result={"a": 1}),
            MockEvaluation(result={"b": 2}),
            auto_enrich_span=False,
        )

        assert len(result.results) == 2

    def test_evaluate_non_blocking(self):
        """Test evaluate() in non-blocking mode."""
        result = evaluate(
            {"response": "test"},
            MockEvaluation(result={"score": 0.95}),
            mode=ExecutionMode.NON_BLOCKING,
            auto_enrich_span=False,
        )

        assert result.is_future is True
        batch = result.wait()
        assert batch.results[0].value == {"score": 0.95}


class TestInputValidation:
    """Tests for input validation."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_validation_enabled(self):
        """Test validation when enabled."""
        evaluator = Evaluator(
            evaluations=[MockEvaluationWithValidation()],
            validate_inputs=True,
            auto_enrich_span=False,
        )

        # Missing required field
        result = evaluator.run({})

        assert result.results[0].status == EvalStatus.FAILED
        assert "response" in result.results[0].error.lower()

    def test_validation_disabled(self):
        """Test validation when disabled."""
        evaluator = Evaluator(
            evaluations=[MockEvaluationWithValidation()],
            validate_inputs=False,
            auto_enrich_span=False,
        )

        # Missing required field but validation disabled
        result = evaluator.run({})

        # Should try to run (may fail differently)
        assert len(result.results) == 1


class TestReturnsImmediately:
    """Tests for non-blocking immediate return."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_non_blocking_returns_fast(self):
        """Test non-blocking mode returns immediately."""

        class SlowEval:
            name = "slow"
            version = "1.0.0"

            def evaluate(self, inputs):
                time.sleep(0.1)
                return {"score": 1.0}

            def get_span_attributes(self, result):
                return {}

        evaluator = Evaluator(
            evaluations=[SlowEval()],
            mode=ExecutionMode.NON_BLOCKING,
            auto_enrich_span=False,
        )

        start = time.perf_counter()
        result = evaluator.run({"response": "test"})
        elapsed = time.perf_counter() - start

        # Should return almost immediately
        assert elapsed < 0.05
        assert result.is_future is True

        # Wait for actual result
        batch = result.wait()
        assert batch.results[0].value == {"score": 1.0}

        evaluator.shutdown()
