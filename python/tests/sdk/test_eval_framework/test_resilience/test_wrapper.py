"""Tests for ResilientBackend wrapper."""

import threading
import time
from typing import Any, Dict, Optional
from unittest.mock import Mock

import pytest

from fi.evals.framework.backends.base import (
    Backend,
    BackendConfig,
    TaskHandle,
    TaskStatus,
)
from fi.evals.framework.resilience.wrapper import (
    ResilientBackend,
    wrap_backend,
)
from fi.evals.framework.resilience.types import (
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    HealthCheckConfig,
    RateLimitConfig,
    RateLimitExceededError,
    ResilienceConfig,
    ResilienceEventType,
    RetryConfig,
    RetryExhaustedError,
    DegradationConfig,
)


class MockBackend(Backend):
    """Mock backend for testing."""

    name = "mock"

    def __init__(self, fail_count: int = 0, fail_on_get: bool = False):
        self.fail_count = fail_count
        self.fail_on_get = fail_on_get
        self.call_count = 0
        self.submitted_tasks = []
        self._task_counter = 0
        self._lock = threading.Lock()

    def submit(
        self,
        fn,
        args=(),
        kwargs=None,
        context=None,
    ) -> TaskHandle:
        with self._lock:
            self.call_count += 1
            self._task_counter += 1

            if self.call_count <= self.fail_count:
                raise ConnectionError(f"Mock failure {self.call_count}")

            task_id = f"task-{self._task_counter}"
            handle = TaskHandle(task_id=task_id, backend_name=self.name)

            # Execute synchronously for testing
            try:
                result = fn(*args, **(kwargs or {}))
                handle._status = TaskStatus.COMPLETED
                handle._result = result
            except Exception as e:
                handle._status = TaskStatus.FAILED
                handle._error = str(e)

            self.submitted_tasks.append((fn, args, kwargs, context, handle))
            return handle

    def get_result(self, handle: TaskHandle, timeout=None):
        if self.fail_on_get:
            raise TimeoutError("Mock timeout")

        if handle._status == TaskStatus.FAILED:
            raise RuntimeError(handle._error)

        return handle._result

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        return handle._status

    def cancel(self, handle: TaskHandle) -> bool:
        if handle._status == TaskStatus.PENDING:
            handle._status = TaskStatus.CANCELLED
            return True
        return False


class TestResilientBackendBasic:
    """Basic functionality tests."""

    def test_wraps_backend(self):
        """ResilientBackend wraps another backend."""
        mock = MockBackend()
        resilient = ResilientBackend(mock)

        assert resilient.underlying is mock
        assert "mock" in resilient.name

    def test_submit_success(self):
        """Successful submission passes through."""
        mock = MockBackend()
        resilient = ResilientBackend(mock)

        handle = resilient.submit(lambda x: x * 2, args=(21,))

        assert handle.succeeded
        assert handle.result == 42
        assert mock.call_count == 1

    def test_get_result(self):
        """Get result passes through."""
        mock = MockBackend()
        resilient = ResilientBackend(mock)

        handle = resilient.submit(lambda: "hello")
        result = resilient.get_result(handle)

        assert result == "hello"

    def test_get_status(self):
        """Get status passes through."""
        mock = MockBackend()
        resilient = ResilientBackend(mock)

        handle = resilient.submit(lambda: 1)
        status = resilient.get_status(handle)

        assert status == TaskStatus.COMPLETED

    def test_cancel(self):
        """Cancel passes through."""
        mock = MockBackend()
        resilient = ResilientBackend(mock)

        # Create a pending task
        handle = TaskHandle(task_id="test", backend_name="mock")
        result = resilient.cancel(handle)

        assert result is True


class TestResilientBackendRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_allows_within_limit(self):
        """Requests within limit succeed."""
        mock = MockBackend()
        config = ResilienceConfig(
            rate_limit=RateLimitConfig(burst_size=10, requests_per_second=100)
        )
        resilient = ResilientBackend(mock, config)

        # Submit several tasks
        for i in range(5):
            handle = resilient.submit(lambda: i)
            assert handle.succeeded

        assert mock.call_count == 5

    def test_rate_limit_rejects_excess(self):
        """Requests exceeding limit are rejected."""
        mock = MockBackend()
        config = ResilienceConfig(
            rate_limit=RateLimitConfig(
                burst_size=2,
                requests_per_second=0.1,  # Very slow refill
                wait_for_token=False,
            )
        )
        resilient = ResilientBackend(mock, config)

        # First two should succeed
        resilient.submit(lambda: 1)
        resilient.submit(lambda: 2)

        # Third should be rejected
        with pytest.raises(RateLimitExceededError):
            resilient.submit(lambda: 3)

        assert mock.call_count == 2


class TestResilientBackendCircuitBreaker:
    """Tests for circuit breaker."""

    def test_circuit_opens_on_failures(self):
        """Circuit opens after failure threshold."""
        mock = MockBackend(fail_count=100)  # Always fail
        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=2)
        )
        resilient = ResilientBackend(mock, config)

        # First two failures trip the circuit
        with pytest.raises(ConnectionError):
            resilient.submit(lambda: 1)
        with pytest.raises(ConnectionError):
            resilient.submit(lambda: 2)

        # Third should be rejected by circuit breaker
        with pytest.raises(CircuitOpenError):
            resilient.submit(lambda: 3)

        assert resilient.circuit_breaker.state == CircuitState.OPEN

    def test_circuit_allows_when_closed(self):
        """Circuit allows requests when closed."""
        mock = MockBackend()
        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=5)
        )
        resilient = ResilientBackend(mock, config)

        handle = resilient.submit(lambda: "success")

        assert handle.succeeded
        assert resilient.circuit_breaker.is_closed


class TestResilientBackendRetry:
    """Tests for retry."""

    def test_retry_on_transient_failure(self):
        """Retries on transient failures."""
        mock = MockBackend(fail_count=2)  # Fail first 2 times
        config = ResilienceConfig(
            retry=RetryConfig(
                max_retries=3,
                base_delay_seconds=0.01,
                retryable_exceptions={ConnectionError},
            )
        )
        resilient = ResilientBackend(mock, config)

        handle = resilient.submit(lambda: "success")

        assert handle.succeeded
        assert handle.result == "success"
        assert mock.call_count == 3  # 2 failures + 1 success

    def test_retry_exhausted(self):
        """RetryExhaustedError when all retries fail."""
        mock = MockBackend(fail_count=100)  # Always fail
        config = ResilienceConfig(
            retry=RetryConfig(
                max_retries=2,
                base_delay_seconds=0.01,
                retryable_exceptions={ConnectionError},
            )
        )
        resilient = ResilientBackend(mock, config)

        with pytest.raises(RetryExhaustedError):
            resilient.submit(lambda: 1)

        assert mock.call_count == 3  # 1 initial + 2 retries


class TestResilientBackendFallback:
    """Tests for fallback."""

    def test_fallback_on_circuit_open(self):
        """Falls back when circuit is open."""
        primary = MockBackend(fail_count=100)
        fallback = MockBackend()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1),
            degradation=DegradationConfig(fallback_on_circuit_open=True),
        )
        resilient = ResilientBackend(
            primary, config, fallback_backend=fallback
        )

        # First failure trips circuit
        with pytest.raises(ConnectionError):
            resilient.submit(lambda: 1)

        # Second should fallback
        handle = resilient.submit(lambda: "fallback")

        assert handle.succeeded
        assert handle.result == "fallback"
        assert handle.backend_name == "mock"  # From fallback
        assert fallback.call_count == 1

    def test_get_result_from_fallback(self):
        """Get result works for fallback tasks."""
        primary = MockBackend(fail_count=100)
        fallback = MockBackend()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1),
            degradation=DegradationConfig(fallback_on_circuit_open=True),
        )
        resilient = ResilientBackend(
            primary, config, fallback_backend=fallback
        )

        # Trip circuit
        with pytest.raises(ConnectionError):
            resilient.submit(lambda: 1)

        # Fallback
        handle = resilient.submit(lambda: 42)
        result = resilient.get_result(handle)

        assert result == 42


class TestResilientBackendEvents:
    """Tests for event callbacks."""

    def test_events_from_components(self):
        """Events are emitted from components."""
        events = []
        mock = MockBackend(fail_count=100)

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1)
        )
        resilient = ResilientBackend(
            mock, config, event_callback=lambda e: events.append(e)
        )

        # Trip circuit
        with pytest.raises(ConnectionError):
            resilient.submit(lambda: 1)

        # Should have circuit opened event
        circuit_events = [
            e for e in events if e.event_type == ResilienceEventType.CIRCUIT_OPENED
        ]
        assert len(circuit_events) == 1


class TestResilientBackendStats:
    """Tests for statistics."""

    def test_get_stats(self):
        """Get stats returns combined statistics."""
        mock = MockBackend()
        config = ResilienceConfig(
            rate_limit=RateLimitConfig(burst_size=10),
            circuit_breaker=CircuitBreakerConfig(),
        )
        resilient = ResilientBackend(mock, config)

        resilient.submit(lambda: 1)

        stats = resilient.get_stats()

        assert "backend" in stats
        assert "underlying" in stats
        assert "rate_limiter" in stats
        assert "circuit_breaker" in stats


class TestResilientBackendHealthCheck:
    """Tests for health checking."""

    def test_health_checker_created(self):
        """Health checker is created when configured."""
        mock = MockBackend()
        config = ResilienceConfig(
            health_check=HealthCheckConfig(interval_seconds=1)
        )
        resilient = ResilientBackend(mock, config)

        assert resilient.health_checker is not None

    def test_start_stop_health_checks(self):
        """Can start and stop health checks."""
        mock = MockBackend()
        config = ResilienceConfig(
            health_check=HealthCheckConfig(interval_seconds=0.1)
        )
        resilient = ResilientBackend(mock, config)

        resilient.start_health_checks()
        assert resilient.health_checker.is_running

        resilient.stop_health_checks()
        assert not resilient.health_checker.is_running


class TestResilientBackendLifecycle:
    """Tests for lifecycle management."""

    def test_shutdown(self):
        """Shutdown stops all components."""
        mock = MockBackend()
        config = ResilienceConfig(
            health_check=HealthCheckConfig(interval_seconds=0.1)
        )
        resilient = ResilientBackend(mock, config)

        resilient.start_health_checks()
        resilient.shutdown()

        assert not resilient.health_checker.is_running

    def test_reset(self):
        """Reset clears all component states."""
        mock = MockBackend(fail_count=100)
        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1),
            rate_limit=RateLimitConfig(burst_size=2),
        )
        resilient = ResilientBackend(mock, config)

        # Trip circuit
        with pytest.raises(ConnectionError):
            resilient.submit(lambda: 1)

        assert resilient.circuit_breaker.is_open

        # Reset
        resilient.reset()

        assert resilient.circuit_breaker.is_closed


class TestResilientBackendSubmitBatch:
    """Tests for batch submission."""

    def test_submit_batch(self):
        """Submit batch applies resilience to each task."""
        mock = MockBackend()
        config = ResilienceConfig(
            rate_limit=RateLimitConfig(burst_size=100)
        )
        resilient = ResilientBackend(mock, config)

        tasks = [
            (lambda x: x * 2, (1,), {}, None),
            (lambda x: x * 2, (2,), {}, None),
            (lambda x: x * 2, (3,), {}, None),
        ]

        handles = resilient.submit_batch(tasks)

        assert len(handles) == 3
        assert all(h.succeeded for h in handles)


class TestWrapBackendConvenience:
    """Tests for wrap_backend convenience function."""

    def test_wrap_with_circuit_breaker(self):
        """wrap_backend creates resilient backend with circuit breaker."""
        mock = MockBackend()
        resilient = wrap_backend(
            mock,
            circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
        )

        assert resilient.circuit_breaker is not None
        assert resilient.rate_limiter is None

    def test_wrap_with_multiple_features(self):
        """wrap_backend supports multiple features."""
        mock = MockBackend()
        resilient = wrap_backend(
            mock,
            circuit_breaker=CircuitBreakerConfig(),
            rate_limit=RateLimitConfig(),
            retry=RetryConfig(),
        )

        assert resilient.circuit_breaker is not None
        assert resilient.rate_limiter is not None
        assert resilient.retry_handler is not None

    def test_wrap_with_fallback(self):
        """wrap_backend supports fallback backend."""
        primary = MockBackend()
        fallback = MockBackend()

        resilient = wrap_backend(
            primary,
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1),
            degradation=DegradationConfig(fallback_on_circuit_open=True),
            fallback_backend=fallback,
        )

        assert resilient.fallback_backend is fallback


class TestResilientBackendCombined:
    """Tests for combined resilience features."""

    def test_rate_limit_before_circuit_breaker(self):
        """Rate limiting is checked before circuit breaker."""
        mock = MockBackend()
        config = ResilienceConfig(
            rate_limit=RateLimitConfig(burst_size=1, wait_for_token=False),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=10),
        )
        resilient = ResilientBackend(mock, config)

        # First succeeds
        resilient.submit(lambda: 1)

        # Second rate limited before hitting circuit breaker
        with pytest.raises(RateLimitExceededError):
            resilient.submit(lambda: 2)

        # Circuit should still be closed
        assert resilient.circuit_breaker.is_closed

    def test_circuit_breaker_prevents_retries_when_open(self):
        """Circuit breaker prevents further retries when it opens."""
        mock = MockBackend(fail_count=100)
        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1),
            retry=RetryConfig(
                max_retries=5,
                base_delay_seconds=0.01,
                retryable_exceptions={ConnectionError},  # Don't retry CircuitOpenError
            ),
        )
        resilient = ResilientBackend(mock, config)

        # First try: ConnectionError (trips circuit), retry gets CircuitOpenError
        # CircuitOpenError is not retryable, so it raises immediately
        with pytest.raises(CircuitOpenError):
            resilient.submit(lambda: 1)

        # Circuit is now open, next request fails immediately with CircuitOpenError
        with pytest.raises(CircuitOpenError):
            resilient.submit(lambda: 2)


class TestResilientBackendEdgeCases:
    """Edge case tests."""

    def test_no_config(self):
        """Works with no resilience config."""
        mock = MockBackend()
        resilient = ResilientBackend(mock)

        handle = resilient.submit(lambda: 42)

        assert handle.result == 42
        assert resilient.rate_limiter is None
        assert resilient.circuit_breaker is None
        assert resilient.retry_handler is None

    def test_context_manager(self):
        """Works as context manager."""
        mock = MockBackend()

        with ResilientBackend(mock) as resilient:
            handle = resilient.submit(lambda: 1)
            assert handle.succeeded

    def test_thread_safety(self):
        """Thread-safe for concurrent submissions."""
        mock = MockBackend()
        config = ResilienceConfig(
            rate_limit=RateLimitConfig(burst_size=100, requests_per_second=1000)
        )
        resilient = ResilientBackend(mock, config)
        results = []
        errors = []

        def worker(n):
            try:
                handle = resilient.submit(lambda x: x * 2, args=(n,))
                results.append(handle.result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
