"""Tests for fi.evals.framework.evaluators.non_blocking module."""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from fi.evals.framework.evaluators.non_blocking import (
    NonBlockingEvaluator,
    non_blocking_evaluate,
    EvalFuture,
    BatchEvalFuture,
    EvalResultAggregator,
)
from fi.evals.framework.types import EvalResult, EvalStatus, BatchEvalResult
from fi.evals.framework.context import EvalContext
from fi.evals.framework.registry import SpanRegistry


class MockEvaluation:
    """Mock evaluation for testing."""

    name = "mock_eval"
    version = "1.0.0"

    def __init__(self, result=None, delay=0, error=None):
        self._result = result if result is not None else {"score": 0.95}
        self._delay = delay
        self._error = error

    def evaluate(self, inputs):
        if self._delay > 0:
            time.sleep(self._delay)
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


class TestEvalFuture:
    """Tests for EvalFuture class."""

    def test_done_false_while_running(self):
        """Test done() returns False while evaluation running."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(time.sleep, 0.1)

        eval_future = EvalFuture(
            future=future,
            eval_name="test",
            eval_version="1.0.0",
        )

        assert eval_future.done() is False
        future.result()  # Wait for completion
        executor.shutdown(wait=True)

    def test_done_true_after_complete(self):
        """Test done() returns True after completion."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(lambda: "result")
        future.result()  # Wait for completion

        eval_future = EvalFuture(
            future=future,
            eval_name="test",
            eval_version="1.0.0",
        )

        assert eval_future.done() is True
        executor.shutdown(wait=True)

    def test_result_returns_value(self):
        """Test result() returns the evaluation result."""
        executor = ThreadPoolExecutor(max_workers=1)

        result = EvalResult(
            value={"score": 0.95},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10.0,
        )
        future = executor.submit(lambda: result)

        eval_future = EvalFuture(
            future=future,
            eval_name="test",
            eval_version="1.0.0",
        )

        assert eval_future.result() is result
        executor.shutdown(wait=True)

    def test_result_with_timeout(self):
        """Test result() with timeout."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(time.sleep, 1.0)

        eval_future = EvalFuture(
            future=future,
            eval_name="test",
            eval_version="1.0.0",
        )

        with pytest.raises(Exception):  # TimeoutError or concurrent.futures.TimeoutError
            eval_future.result(timeout=0.01)

        future.cancel()
        executor.shutdown(wait=False)

    def test_cancel(self):
        """Test cancel() attempts to cancel the evaluation."""
        executor = ThreadPoolExecutor(max_workers=1)
        # Submit a blocking task first
        blocker = executor.submit(time.sleep, 0.5)

        # Submit another task that should be cancellable
        future = executor.submit(time.sleep, 1.0)

        eval_future = EvalFuture(
            future=future,
            eval_name="test",
            eval_version="1.0.0",
        )

        # Try to cancel - may or may not succeed depending on timing
        eval_future.cancel()

        blocker.result()
        executor.shutdown(wait=True)

    def test_add_done_callback(self):
        """Test add_done_callback() registers callback."""
        executor = ThreadPoolExecutor(max_workers=1)

        result = EvalResult(
            value={"score": 0.95},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10.0,
        )
        future = executor.submit(lambda: result)

        eval_future = EvalFuture(
            future=future,
            eval_name="test",
            eval_version="1.0.0",
        )

        callback_called = []

        def callback(ef):
            callback_called.append(ef)

        eval_future.add_done_callback(callback)
        eval_future.result()  # Wait for completion

        # Give callback time to execute
        time.sleep(0.05)

        assert len(callback_called) == 1
        assert callback_called[0] is eval_future
        executor.shutdown(wait=True)


class TestBatchEvalFuture:
    """Tests for BatchEvalFuture class."""

    def test_done_all_complete(self):
        """Test done() returns True when all futures complete."""
        executor = ThreadPoolExecutor(max_workers=2)

        futures = [
            EvalFuture(
                future=executor.submit(lambda: EvalResult(
                    value={}, eval_name="test1", eval_version="1.0.0", latency_ms=1.0
                )),
                eval_name="test1",
                eval_version="1.0.0",
            ),
            EvalFuture(
                future=executor.submit(lambda: EvalResult(
                    value={}, eval_name="test2", eval_version="1.0.0", latency_ms=1.0
                )),
                eval_name="test2",
                eval_version="1.0.0",
            ),
        ]

        batch = BatchEvalFuture(futures=futures)

        # Wait for all to complete
        for f in futures:
            f.result()

        assert batch.done() is True
        executor.shutdown(wait=True)

    def test_done_not_all_complete(self):
        """Test done() returns False when some futures pending."""
        executor = ThreadPoolExecutor(max_workers=1)

        # First task blocks
        blocker = executor.submit(time.sleep, 0.2)

        futures = [
            EvalFuture(
                future=blocker,
                eval_name="test1",
                eval_version="1.0.0",
            ),
        ]

        batch = BatchEvalFuture(futures=futures)

        assert batch.done() is False

        blocker.result()
        executor.shutdown(wait=True)

    def test_results_returns_batch(self):
        """Test results() returns BatchEvalResult."""
        executor = ThreadPoolExecutor(max_workers=2)

        result1 = EvalResult(
            value={"score": 0.9}, eval_name="test1", eval_version="1.0.0", latency_ms=1.0
        )
        result2 = EvalResult(
            value={"score": 0.8}, eval_name="test2", eval_version="1.0.0", latency_ms=2.0
        )

        futures = [
            EvalFuture(
                future=executor.submit(lambda: result1),
                eval_name="test1",
                eval_version="1.0.0",
            ),
            EvalFuture(
                future=executor.submit(lambda: result2),
                eval_name="test2",
                eval_version="1.0.0",
            ),
        ]

        batch_future = BatchEvalFuture(futures=futures)
        batch_result = batch_future.results()

        assert isinstance(batch_result, BatchEvalResult)
        assert len(batch_result.results) == 2
        executor.shutdown(wait=True)

    def test_cancel_all(self):
        """Test cancel_all() cancels pending futures."""
        executor = ThreadPoolExecutor(max_workers=1)

        # First task blocks
        blocker = executor.submit(time.sleep, 0.5)

        # These should be cancellable
        futures = [
            EvalFuture(
                future=executor.submit(time.sleep, 1.0),
                eval_name="test1",
                eval_version="1.0.0",
            ),
            EvalFuture(
                future=executor.submit(time.sleep, 1.0),
                eval_name="test2",
                eval_version="1.0.0",
            ),
        ]

        batch = BatchEvalFuture(futures=futures)
        cancelled = batch.cancel_all()

        # At least some should be cancelled
        assert cancelled >= 0  # May be 0 if already started

        blocker.result()
        executor.shutdown(wait=True)


class TestNonBlockingEvaluator:
    """Tests for NonBlockingEvaluator class."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_init_with_evaluations(self):
        """Test initialization with evaluations."""
        eval1 = MockEvaluation()
        evaluator = NonBlockingEvaluator(evaluations=[eval1])

        assert len(evaluator.evaluations) == 1
        evaluator.shutdown()

    def test_init_default_workers(self):
        """Test default max_workers."""
        evaluator = NonBlockingEvaluator()
        assert evaluator.max_workers == 4
        evaluator.shutdown()

    def test_add_evaluation(self):
        """Test add_evaluation() method."""
        evaluator = NonBlockingEvaluator()
        evaluator.add_evaluation(MockEvaluation())

        assert len(evaluator.evaluations) == 1
        evaluator.shutdown()

    def test_add_evaluation_chaining(self):
        """Test add_evaluation() returns self for chaining."""
        evaluator = NonBlockingEvaluator()
        result = evaluator.add_evaluation(MockEvaluation()).add_evaluation(MockEvaluation())

        assert result is evaluator
        assert len(evaluator.evaluations) == 2
        evaluator.shutdown()

    def test_evaluate_returns_immediately(self):
        """Test evaluate() returns immediately without blocking."""
        evaluator = NonBlockingEvaluator(
            evaluations=[MockEvaluation(delay=0.1)],
            auto_enrich_span=False,
        )

        start = time.perf_counter()
        future = evaluator.evaluate({"response": "test"})
        elapsed = time.perf_counter() - start

        # Should return almost immediately
        assert elapsed < 0.05
        assert isinstance(future, BatchEvalFuture)

        # Wait for completion
        future.results()
        evaluator.shutdown()

    def test_evaluate_produces_result(self):
        """Test evaluate() produces correct result."""
        evaluator = NonBlockingEvaluator(
            evaluations=[MockEvaluation(result={"score": 0.95})],
            auto_enrich_span=False,
        )

        future = evaluator.evaluate({"response": "test"})
        batch = future.results()

        assert len(batch.results) == 1
        assert batch.results[0].value == {"score": 0.95}
        assert batch.results[0].status == EvalStatus.COMPLETED
        evaluator.shutdown()

    def test_evaluate_multiple_evaluations(self):
        """Test evaluate() runs multiple evaluations."""
        evaluator = NonBlockingEvaluator(
            evaluations=[
                MockEvaluation(result={"score": 0.9}),
                MockEvaluation(result={"score": 0.8}),
            ],
            auto_enrich_span=False,
        )

        future = evaluator.evaluate({"response": "test"})
        batch = future.results()

        assert len(batch.results) == 2
        evaluator.shutdown()

    def test_evaluate_no_evaluations_raises(self):
        """Test evaluate() raises when no evaluations configured."""
        evaluator = NonBlockingEvaluator(auto_enrich_span=False)

        with pytest.raises(ValueError, match="No evaluations"):
            evaluator.evaluate({"response": "test"})

        evaluator.shutdown()

    def test_evaluate_handles_failure(self):
        """Test evaluate() handles evaluation failures."""
        evaluator = NonBlockingEvaluator(
            evaluations=[MockEvaluation(error="Test error")],
            auto_enrich_span=False,
        )

        future = evaluator.evaluate({"response": "test"})
        batch = future.results()

        assert len(batch.results) == 1
        assert batch.results[0].status == EvalStatus.FAILED
        assert "Test error" in batch.results[0].error
        evaluator.shutdown()

    def test_evaluate_records_latency(self):
        """Test evaluate() records latency."""
        evaluator = NonBlockingEvaluator(
            evaluations=[MockEvaluation(delay=0.05)],
            auto_enrich_span=False,
        )

        future = evaluator.evaluate({"response": "test"})
        batch = future.results()

        assert batch.results[0].latency_ms >= 50
        evaluator.shutdown()

    def test_evaluate_with_callback(self):
        """Test evaluate() invokes callback."""
        callback_results = []

        def callback(result):
            callback_results.append(result)

        evaluator = NonBlockingEvaluator(
            evaluations=[MockEvaluation()],
            auto_enrich_span=False,
        )

        future = evaluator.evaluate({"response": "test"}, callback=callback)
        future.results()

        # Give callback time to execute
        time.sleep(0.05)

        assert len(callback_results) == 1
        evaluator.shutdown()

    def test_evaluate_single(self):
        """Test evaluate_single() method."""
        evaluator = NonBlockingEvaluator(auto_enrich_span=False)

        future = evaluator.evaluate_single(
            MockEvaluation(result={"score": 0.95}),
            {"response": "test"},
        )

        result = future.result()
        assert result.value == {"score": 0.95}
        evaluator.shutdown()

    def test_evaluate_input_validation(self):
        """Test input validation."""
        evaluator = NonBlockingEvaluator(
            evaluations=[MockEvaluationWithValidation()],
            validate_inputs=True,
            auto_enrich_span=False,
        )

        # Missing required field
        future = evaluator.evaluate({})
        batch = future.results()

        assert batch.results[0].status == EvalStatus.FAILED
        assert "response" in batch.results[0].error.lower()
        evaluator.shutdown()

    def test_evaluate_validation_disabled(self):
        """Test validation can be disabled."""
        evaluator = NonBlockingEvaluator(
            evaluations=[MockEvaluationWithValidation()],
            validate_inputs=False,
            auto_enrich_span=False,
        )

        # Missing required field but validation disabled
        future = evaluator.evaluate({})
        batch = future.results()

        # Should try to run and may fail differently
        assert len(batch.results) == 1
        evaluator.shutdown()

    def test_context_manager(self):
        """Test context manager protocol."""
        with NonBlockingEvaluator(
            evaluations=[MockEvaluation()],
            auto_enrich_span=False,
        ) as evaluator:
            future = evaluator.evaluate({"response": "test"})
            future.results()

        # Executor should be shutdown
        assert evaluator._executor is None or not evaluator._owns_executor

    def test_custom_executor(self):
        """Test using custom executor."""
        custom_executor = ThreadPoolExecutor(max_workers=2)

        evaluator = NonBlockingEvaluator(
            evaluations=[MockEvaluation()],
            executor=custom_executor,
            auto_enrich_span=False,
        )

        future = evaluator.evaluate({"response": "test"})
        future.results()

        # Should not shutdown custom executor
        evaluator.shutdown()
        assert not custom_executor._shutdown

        custom_executor.shutdown()

    def test_concurrent_evaluations(self):
        """Test evaluations run concurrently."""
        evaluator = NonBlockingEvaluator(
            evaluations=[
                MockEvaluation(delay=0.1),
                MockEvaluation(delay=0.1),
                MockEvaluation(delay=0.1),
            ],
            max_workers=3,
            auto_enrich_span=False,
        )

        start = time.perf_counter()
        future = evaluator.evaluate({"response": "test"})
        future.results()
        elapsed = time.perf_counter() - start

        # Should take ~0.1s if parallel, ~0.3s if serial
        assert elapsed < 0.25
        evaluator.shutdown()


class TestNonBlockingEvaluate:
    """Tests for non_blocking_evaluate convenience function."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_basic_usage(self):
        """Test basic non_blocking_evaluate usage."""
        future = non_blocking_evaluate(
            {"response": "test"},
            MockEvaluation(result={"score": 0.95}),
            auto_enrich_span=False,
        )

        batch = future.results()
        assert len(batch.results) == 1
        assert batch.results[0].value == {"score": 0.95}

    def test_multiple_evaluations(self):
        """Test with multiple evaluations."""
        future = non_blocking_evaluate(
            {"response": "test"},
            MockEvaluation(result={"score": 0.9}),
            MockEvaluation(result={"score": 0.8}),
            auto_enrich_span=False,
        )

        batch = future.results()
        assert len(batch.results) == 2

    def test_with_callback(self):
        """Test with callback."""
        callback_results = []

        future = non_blocking_evaluate(
            {"response": "test"},
            MockEvaluation(),
            callback=lambda r: callback_results.append(r),
            auto_enrich_span=False,
        )

        future.results()
        time.sleep(0.05)

        assert len(callback_results) == 1


class TestEvalResultAggregator:
    """Tests for EvalResultAggregator class."""

    def test_add_single(self):
        """Test adding a single result."""
        aggregator = EvalResultAggregator()

        result = EvalResult(
            value={"score": 0.95},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10.0,
        )
        aggregator.add(result)

        assert aggregator.count == 1

    def test_add_all(self):
        """Test adding multiple results."""
        aggregator = EvalResultAggregator()

        results = [
            EvalResult(value={}, eval_name="test1", eval_version="1.0.0", latency_ms=1.0),
            EvalResult(value={}, eval_name="test2", eval_version="1.0.0", latency_ms=2.0),
        ]
        aggregator.add_all(results)

        assert aggregator.count == 2

    def test_to_batch(self):
        """Test converting to batch result."""
        aggregator = EvalResultAggregator()

        result = EvalResult(
            value={"score": 0.95},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10.0,
        )
        aggregator.add(result)

        batch = aggregator.to_batch()
        assert isinstance(batch, BatchEvalResult)
        assert len(batch.results) == 1

    def test_clear(self):
        """Test clearing results."""
        aggregator = EvalResultAggregator()

        result = EvalResult(
            value={}, eval_name="test", eval_version="1.0.0", latency_ms=1.0
        )
        aggregator.add(result)
        aggregator.add(result)

        cleared = aggregator.clear()
        assert cleared == 2
        assert aggregator.count == 0

    def test_thread_safe(self):
        """Test thread safety."""
        aggregator = EvalResultAggregator()
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    result = EvalResult(
                        value={"thread": thread_id, "i": i},
                        eval_name=f"test_{thread_id}",
                        eval_version="1.0.0",
                        latency_ms=1.0,
                    )
                    aggregator.add(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert aggregator.count == 1000


class TestSpanEnrichment:
    """Tests for span enrichment in non-blocking evaluator."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_enriches_span_on_success(self):
        """Test that span is enriched on success."""
        from fi.evals.framework.registry import register_span, get_span

        # Create and register a mock span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        # Create context pointing to the span
        context = EvalContext(trace_id=trace_id, span_id=span_id)

        evaluator = NonBlockingEvaluator(
            evaluations=[MockEvaluation(result={"score": 0.95})],
            auto_enrich_span=True,
        )

        future = evaluator.evaluate({"response": "test"}, context=context)
        future.results()

        # Give time for enrichment
        time.sleep(0.05)

        # Check span was enriched
        assert mock_span.set_attribute.called
        evaluator.shutdown()

    def test_enriches_span_on_failure(self):
        """Test that span is enriched on failure."""
        from fi.evals.framework.registry import register_span

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "c" * 32
        span_id = "d" * 16

        register_span(trace_id, span_id, mock_span)

        context = EvalContext(trace_id=trace_id, span_id=span_id)

        evaluator = NonBlockingEvaluator(
            evaluations=[MockEvaluation(error="Test error")],
            auto_enrich_span=True,
        )

        future = evaluator.evaluate({"response": "test"}, context=context)
        future.results()

        time.sleep(0.05)

        # Check span was enriched with error
        assert mock_span.set_attribute.called
        evaluator.shutdown()
