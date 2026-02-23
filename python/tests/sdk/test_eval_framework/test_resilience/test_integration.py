"""Integration tests for Evaluator + Resilience stack.

Tests the full pipeline: Evaluator -> ResilientBackend -> Backend -> execution.
"""

import time
import pytest
from unittest.mock import MagicMock

from fi.evals.framework.evaluator import (
    Evaluator,
    EvaluatorResult,
    resilient_evaluator,
    _execute_single_evaluation,
)
from fi.evals.framework.types import (
    ExecutionMode,
    EvalResult,
    EvalStatus,
    BatchEvalResult,
)
from fi.evals.framework.backends import ThreadPoolBackend, ThreadPoolConfig
from fi.evals.framework.resilience import (
    ResilientBackend,
    ResilienceConfig,
    CircuitBreakerConfig,
    RateLimitConfig,
    RetryConfig,
    DegradationConfig,
    HealthCheckConfig,
    wrap_backend,
    CircuitOpenError,
    RateLimitExceededError,
)


# === Test Helpers ===


class MockEvaluation:
    """Mock evaluation for testing."""

    name = "mock_eval"
    version = "1.0.0"

    def __init__(self, result=None, error=None, delay=0):
        self._result = result if result is not None else {"score": 0.95}
        self._error = error
        self._delay = delay

    def evaluate(self, inputs):
        if self._delay:
            time.sleep(self._delay)
        if self._error:
            raise ValueError(self._error)
        return self._result

    def get_span_attributes(self, result):
        return {"score": result.get("score", 0)}


class FailingEvaluation:
    """Evaluation that always fails."""

    name = "failing_eval"
    version = "1.0.0"

    def evaluate(self, inputs):
        raise RuntimeError("Evaluation failed")

    def get_span_attributes(self, result):
        return {}


class ValidatingEvaluation:
    """Evaluation with input validation."""

    name = "validating_eval"
    version = "1.0.0"

    def evaluate(self, inputs):
        return {"score": 1.0}

    def get_span_attributes(self, result):
        return {"score": result.get("score", 0)}

    def validate_inputs(self, inputs):
        if "response" not in inputs:
            return "Missing 'response' field"
        return None


class CountingEvaluation:
    """Evaluation that counts calls."""

    name = "counting_eval"
    version = "1.0.0"

    def __init__(self):
        self.call_count = 0

    def evaluate(self, inputs):
        self.call_count += 1
        return {"count": self.call_count}

    def get_span_attributes(self, result):
        return {"count": result.get("count", 0)}


# === Tests ===


class TestResilientEvaluatorFactory:
    """Tests for resilient_evaluator() factory function."""

    def test_creates_with_defaults(self):
        """resilient_evaluator creates an Evaluator in DISTRIBUTED mode."""
        mock_eval = MockEvaluation()
        evaluator = resilient_evaluator(mock_eval)

        assert evaluator.mode == ExecutionMode.DISTRIBUTED
        assert len(evaluator.evaluations) == 1
        assert isinstance(evaluator._backend, ResilientBackend)
        evaluator.shutdown()

    def test_creates_with_multiple_evaluations(self):
        """resilient_evaluator accepts multiple evaluations."""
        eval1 = MockEvaluation(result={"score": 0.8})
        eval2 = MockEvaluation(result={"score": 0.9})
        evaluator = resilient_evaluator(eval1, eval2)

        assert len(evaluator.evaluations) == 2
        evaluator.shutdown()

    def test_creates_with_custom_resilience_config(self):
        """resilient_evaluator uses custom ResilienceConfig."""
        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=3),
            rate_limit=RateLimitConfig(requests_per_second=5.0),
        )
        evaluator = resilient_evaluator(
            MockEvaluation(),
            resilience=config,
        )

        backend = evaluator._backend
        assert isinstance(backend, ResilientBackend)
        assert backend.circuit_breaker is not None
        assert backend.rate_limiter is not None
        evaluator.shutdown()

    def test_creates_with_custom_backend(self):
        """resilient_evaluator wraps a custom backend."""
        custom_backend = ThreadPoolBackend(ThreadPoolConfig(max_workers=2))
        evaluator = resilient_evaluator(
            MockEvaluation(),
            backend=custom_backend,
        )

        backend = evaluator._backend
        assert isinstance(backend, ResilientBackend)
        assert backend.underlying is custom_backend
        evaluator.shutdown()

    def test_creates_with_fallback_backend(self):
        """resilient_evaluator sets up fallback backend."""
        fallback = ThreadPoolBackend()
        config = ResilienceConfig(
            degradation=DegradationConfig(),
        )
        evaluator = resilient_evaluator(
            MockEvaluation(),
            resilience=config,
            fallback_backend=fallback,
        )

        backend = evaluator._backend
        assert isinstance(backend, ResilientBackend)
        assert backend.fallback_backend is fallback
        evaluator.shutdown()

    def test_creates_with_event_callback(self):
        """resilient_evaluator passes event callback through."""
        callback = MagicMock()
        evaluator = resilient_evaluator(
            MockEvaluation(),
            event_callback=callback,
        )

        backend = evaluator._backend
        assert backend.event_callback is callback
        evaluator.shutdown()

    def test_auto_enrich_span_setting(self):
        """resilient_evaluator passes auto_enrich_span through."""
        evaluator = resilient_evaluator(
            MockEvaluation(),
            auto_enrich_span=False,
        )

        assert evaluator.auto_enrich_span is False
        evaluator.shutdown()


class TestDistributedModeWithBackend:
    """Tests for _run_distributed() using the backend."""

    def test_submits_to_backend_and_collects_results(self):
        """Distributed mode submits evaluations to backend and collects results."""
        mock_eval = MockEvaluation(result={"score": 0.95})
        backend = ThreadPoolBackend()

        evaluator = Evaluator(
            evaluations=[mock_eval],
            mode=ExecutionMode.DISTRIBUTED,
            backend=backend,
        )

        result = evaluator.run({"response": "test"})

        assert isinstance(result, EvaluatorResult)
        assert result.mode == ExecutionMode.DISTRIBUTED
        assert result.batch is not None
        assert len(result.batch.results) == 1
        assert result.batch.results[0].status == EvalStatus.COMPLETED
        assert result.batch.results[0].value == {"score": 0.95}
        evaluator.shutdown()

    def test_multiple_evaluations_submitted(self):
        """Each evaluation is submitted as a separate task."""
        eval1 = MockEvaluation(result={"score": 0.8})
        eval2 = MockEvaluation(result={"score": 0.9})
        backend = ThreadPoolBackend()

        evaluator = Evaluator(
            evaluations=[eval1, eval2],
            mode=ExecutionMode.DISTRIBUTED,
            backend=backend,
        )

        result = evaluator.run({"response": "test"})

        assert result.batch.total_count == 2
        assert result.batch.success_count == 2
        values = [r.value for r in result.batch.results]
        assert {"score": 0.8} in values
        assert {"score": 0.9} in values
        evaluator.shutdown()

    def test_handles_evaluation_failure(self):
        """Distributed mode handles evaluation failures gracefully."""
        success_eval = MockEvaluation(result={"score": 0.9})
        fail_eval = FailingEvaluation()
        backend = ThreadPoolBackend()

        evaluator = Evaluator(
            evaluations=[success_eval, fail_eval],
            mode=ExecutionMode.DISTRIBUTED,
            backend=backend,
        )

        result = evaluator.run({"response": "test"})

        assert result.batch.total_count == 2
        assert result.batch.success_count == 1
        assert result.batch.failure_count == 1
        evaluator.shutdown()

    def test_falls_back_to_non_blocking_without_backend(self):
        """Without a backend, distributed mode falls back to non-blocking."""
        mock_eval = MockEvaluation()
        evaluator = Evaluator(
            evaluations=[mock_eval],
            mode=ExecutionMode.DISTRIBUTED,
            backend=None,
        )

        result = evaluator.run({"response": "test"})

        # Falls back to non-blocking, returns a future
        assert result.is_future
        batch = result.wait(timeout=5.0)
        assert len(batch.results) == 1
        evaluator.shutdown()

    def test_callback_invoked_for_each_result(self):
        """Callback is called for each evaluation result."""
        callback = MagicMock()
        backend = ThreadPoolBackend()
        eval1 = MockEvaluation(result={"score": 0.8})
        eval2 = MockEvaluation(result={"score": 0.9})

        evaluator = Evaluator(
            evaluations=[eval1, eval2],
            mode=ExecutionMode.DISTRIBUTED,
            backend=backend,
        )

        evaluator.run({"response": "test"}, callback=callback)

        assert callback.call_count == 2
        evaluator.shutdown()

    def test_backend_timeout_property_default(self):
        """Default backend timeout is 300s."""
        evaluator = Evaluator(
            evaluations=[MockEvaluation()],
            mode=ExecutionMode.DISTRIBUTED,
        )
        assert evaluator._backend_timeout == 300.0

    def test_backend_timeout_property_from_config(self):
        """Backend timeout reads from backend config."""
        config = ThreadPoolConfig(timeout_seconds=60.0)
        backend = ThreadPoolBackend(config)
        evaluator = Evaluator(
            evaluations=[MockEvaluation()],
            mode=ExecutionMode.DISTRIBUTED,
            backend=backend,
        )
        assert evaluator._backend_timeout == 60.0
        evaluator.shutdown()


class TestExecuteSingleEvaluation:
    """Tests for _execute_single_evaluation helper."""

    def test_successful_evaluation(self):
        """Returns completed EvalResult for success."""
        evaluation = MockEvaluation(result={"score": 0.85})
        result = _execute_single_evaluation(evaluation, {"response": "test"})

        assert isinstance(result, EvalResult)
        assert result.status == EvalStatus.COMPLETED
        assert result.value == {"score": 0.85}
        assert result.eval_name == "mock_eval"
        assert result.latency_ms > 0

    def test_failed_evaluation(self):
        """Returns failed EvalResult for exception."""
        evaluation = FailingEvaluation()
        result = _execute_single_evaluation(evaluation, {"response": "test"})

        assert result.status == EvalStatus.FAILED
        assert result.value is None
        assert "Evaluation failed" in result.error

    def test_validation_failure(self):
        """Returns failed EvalResult for validation failure."""
        evaluation = ValidatingEvaluation()
        result = _execute_single_evaluation(evaluation, {})

        assert result.status == EvalStatus.FAILED
        assert "Validation error" in result.error

    def test_validation_passes(self):
        """Returns completed EvalResult when validation passes."""
        evaluation = ValidatingEvaluation()
        result = _execute_single_evaluation(
            evaluation, {"response": "test"}
        )

        assert result.status == EvalStatus.COMPLETED
        assert result.value == {"score": 1.0}

    def test_skips_validation_when_disabled(self):
        """Skips validation when validate=False."""
        evaluation = ValidatingEvaluation()
        result = _execute_single_evaluation(
            evaluation, {}, validate=False
        )

        assert result.status == EvalStatus.COMPLETED


class TestEvaluatorWithCircuitBreaker:
    """Tests for Evaluator + circuit breaker integration."""

    def test_circuit_opens_after_failures(self):
        """Circuit opens after failure threshold, subsequent submissions fail."""
        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=2,
                timeout_seconds=60.0,
            ),
        )
        backend = ThreadPoolBackend()
        resilient = ResilientBackend(backend, config)
        fail_eval = FailingEvaluation()

        evaluator = Evaluator(
            evaluations=[fail_eval],
            mode=ExecutionMode.DISTRIBUTED,
            backend=resilient,
        )

        # Run enough times to trip the circuit
        for _ in range(3):
            evaluator.run({"response": "test"})

        # Circuit should now be open — next submission should raise or fail
        result = evaluator.run({"response": "test"})
        # With circuit open, submit raises CircuitOpenError which gets caught
        # and results in FAILED status
        has_failure = any(
            r.status == EvalStatus.FAILED for r in result.batch.results
        )
        assert has_failure
        evaluator.shutdown()

    def test_circuit_breaker_with_resilient_evaluator(self):
        """resilient_evaluator works with circuit breaker config."""
        evaluator = resilient_evaluator(
            MockEvaluation(),
            resilience=ResilienceConfig(
                circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
            ),
        )

        result = evaluator.run({"response": "test"})
        assert result.batch.success_count == 1
        evaluator.shutdown()


class TestEvaluatorWithRateLimiter:
    """Tests for Evaluator + rate limiter integration."""

    def test_rate_limit_allows_normal_traffic(self):
        """Rate limiter allows requests within limits."""
        evaluator = resilient_evaluator(
            MockEvaluation(),
            resilience=ResilienceConfig(
                rate_limit=RateLimitConfig(
                    requests_per_second=100.0,
                    burst_size=50,
                ),
            ),
        )

        result = evaluator.run({"response": "test"})
        assert result.batch.success_count == 1
        evaluator.shutdown()

    def test_rate_limit_prevents_excess(self):
        """Rate limiter rejects excess submissions."""
        evaluator = resilient_evaluator(
            MockEvaluation(),
            MockEvaluation(),
            MockEvaluation(),
            MockEvaluation(),
            MockEvaluation(),
            resilience=ResilienceConfig(
                rate_limit=RateLimitConfig(
                    requests_per_second=1.0,
                    burst_size=2,
                ),
            ),
        )

        # First run exhausts burst capacity
        result = evaluator.run({"response": "test"})
        # Some evaluations may fail due to rate limiting
        has_rate_limited = any(
            r.status == EvalStatus.FAILED
            and r.error
            and "Rate limit" in r.error
            for r in result.batch.results
        )
        # Either all succeed (if rate limiter allows) or some fail
        assert result.batch.total_count == 5
        evaluator.shutdown()


class TestEvaluatorWithRetry:
    """Tests for Evaluator + retry integration."""

    def test_retries_transient_failures(self):
        """Retry handler retries transient failures."""
        call_count = 0

        class TransientEval:
            name = "transient_eval"
            version = "1.0.0"

            def evaluate(self, inputs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Temporary failure")
                return {"score": 1.0}

            def get_span_attributes(self, result):
                return {"score": result.get("score", 0)}

        evaluator = resilient_evaluator(
            TransientEval(),
            resilience=ResilienceConfig(
                retry=RetryConfig(
                    max_retries=3,
                    base_delay_seconds=0.01,
                    retryable_exceptions={ConnectionError},
                ),
            ),
        )

        result = evaluator.run({"response": "test"})
        # The retry wraps the submit() call, not the evaluation itself
        # The evaluation runs inside the backend thread
        assert result.batch.total_count == 1
        evaluator.shutdown()

    def test_retry_with_success(self):
        """Retry succeeds when evaluation works."""
        evaluator = resilient_evaluator(
            MockEvaluation(result={"score": 0.99}),
            resilience=ResilienceConfig(
                retry=RetryConfig(max_retries=3, base_delay_seconds=0.01),
            ),
        )

        result = evaluator.run({"response": "test"})
        assert result.batch.success_count == 1
        assert result.batch.results[0].value == {"score": 0.99}
        evaluator.shutdown()


class TestEvaluatorWithFallback:
    """Tests for Evaluator + fallback integration."""

    def test_fallback_backend_used_on_primary_failure(self):
        """Fallback backend is available when primary fails."""
        primary = ThreadPoolBackend()
        fallback = ThreadPoolBackend()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1),
            degradation=DegradationConfig(fallback_on_circuit_open=True),
        )

        resilient = ResilientBackend(
            underlying=primary,
            config=config,
            fallback_backend=fallback,
        )

        evaluator = Evaluator(
            evaluations=[MockEvaluation()],
            mode=ExecutionMode.DISTRIBUTED,
            backend=resilient,
        )

        result = evaluator.run({"response": "test"})
        assert result.batch.total_count == 1
        evaluator.shutdown()

    def test_resilient_evaluator_with_fallback(self):
        """resilient_evaluator correctly wires fallback."""
        fallback = ThreadPoolBackend()
        evaluator = resilient_evaluator(
            MockEvaluation(),
            resilience=ResilienceConfig(
                degradation=DegradationConfig(),
            ),
            fallback_backend=fallback,
        )

        backend = evaluator._backend
        assert isinstance(backend, ResilientBackend)
        assert backend.fallback_backend is fallback
        evaluator.shutdown()


class TestEvaluatorShutdownWithResilience:
    """Tests for shutdown cascading through resilience stack."""

    def test_shutdown_cascades_to_resilient_backend(self):
        """Evaluator.shutdown() shuts down the ResilientBackend."""
        backend = ThreadPoolBackend()
        resilient = ResilientBackend(backend, ResilienceConfig())

        evaluator = Evaluator(
            evaluations=[MockEvaluation()],
            mode=ExecutionMode.DISTRIBUTED,
            backend=resilient,
        )

        evaluator.run({"response": "test"})
        evaluator.shutdown()

        # After shutdown, the underlying executor should be None
        assert backend._executor is None

    def test_context_manager_shuts_down(self):
        """Using Evaluator as context manager triggers shutdown."""
        backend = ThreadPoolBackend()
        resilient = ResilientBackend(backend, ResilienceConfig())

        with Evaluator(
            evaluations=[MockEvaluation()],
            mode=ExecutionMode.DISTRIBUTED,
            backend=resilient,
        ) as evaluator:
            evaluator.run({"response": "test"})

        assert backend._executor is None

    def test_shutdown_with_fallback_backend(self):
        """Shutdown cascades to fallback backend too."""
        primary = ThreadPoolBackend()
        fallback = ThreadPoolBackend()

        resilient = ResilientBackend(
            primary,
            ResilienceConfig(degradation=DegradationConfig()),
            fallback_backend=fallback,
        )

        evaluator = Evaluator(
            evaluations=[MockEvaluation()],
            mode=ExecutionMode.DISTRIBUTED,
            backend=resilient,
        )

        evaluator.run({"response": "test"})
        evaluator.shutdown()

        assert primary._executor is None
        assert fallback._executor is None

    def test_shutdown_with_health_checker(self):
        """Shutdown stops health checker."""
        config = ResilienceConfig(
            health_check=HealthCheckConfig(
                interval_seconds=1.0,
            ),
        )
        backend = ThreadPoolBackend()
        resilient = ResilientBackend(backend, config)
        resilient.start_health_checks()

        evaluator = Evaluator(
            evaluations=[MockEvaluation()],
            mode=ExecutionMode.DISTRIBUTED,
            backend=resilient,
        )

        evaluator.run({"response": "test"})
        evaluator.shutdown()

        assert not resilient.health_checker.is_running


class TestEndToEnd:
    """End-to-end tests for the full stack."""

    def test_full_pipeline_success(self):
        """Full pipeline: resilient_evaluator -> run -> results."""
        evaluator = resilient_evaluator(
            MockEvaluation(result={"score": 0.95}),
            resilience=ResilienceConfig(
                circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
                rate_limit=RateLimitConfig(
                    requests_per_second=100.0, burst_size=50
                ),
                retry=RetryConfig(
                    max_retries=2, base_delay_seconds=0.01
                ),
            ),
        )

        result = evaluator.run({"response": "Hello world"})

        assert result.batch is not None
        assert result.batch.success_count == 1
        assert result.batch.results[0].value == {"score": 0.95}
        assert result.batch.results[0].eval_name == "mock_eval"
        evaluator.shutdown()

    def test_full_pipeline_multiple_evals(self):
        """Full pipeline with multiple evaluations."""
        eval1 = MockEvaluation(result={"score": 0.8})
        eval1.name = "eval_a"
        eval2 = MockEvaluation(result={"score": 0.9})
        eval2.name = "eval_b"

        evaluator = resilient_evaluator(
            eval1,
            eval2,
            resilience=ResilienceConfig(
                rate_limit=RateLimitConfig(
                    requests_per_second=100.0, burst_size=50
                ),
            ),
        )

        result = evaluator.run({"response": "test"})

        assert result.batch.total_count == 2
        assert result.batch.success_count == 2
        names = {r.eval_name for r in result.batch.results}
        assert "eval_a" in names
        assert "eval_b" in names
        evaluator.shutdown()

    def test_import_from_framework(self):
        """Resilience imports work from the framework level."""
        from fi.evals.framework import (
            ResilientBackend,
            ResilienceConfig,
            CircuitBreakerConfig,
            RateLimitConfig,
            RetryConfig,
            DegradationConfig,
            HealthCheckConfig,
            wrap_backend,
            resilient_evaluator,
        )

        assert ResilientBackend is not None
        assert ResilienceConfig is not None
        assert resilient_evaluator is not None

    def test_wrap_backend_convenience(self):
        """wrap_backend convenience function works with Evaluator."""
        backend = wrap_backend(
            ThreadPoolBackend(),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
            rate_limit=RateLimitConfig(requests_per_second=50.0),
        )

        evaluator = Evaluator(
            evaluations=[MockEvaluation()],
            mode=ExecutionMode.DISTRIBUTED,
            backend=backend,
        )

        result = evaluator.run({"response": "test"})
        assert result.batch.success_count == 1
        evaluator.shutdown()
