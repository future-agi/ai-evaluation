"""Tests for circuit breaker implementation."""

import threading
import time
from unittest.mock import Mock

import pytest

from fi.evals.framework.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitStats,
)
from fi.evals.framework.resilience.types import (
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    ResilienceEventType,
)


class TestCircuitBreakerBasic:
    """Basic functionality tests."""

    def test_initial_state_closed(self):
        """Circuit starts in closed state."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open

    def test_execute_success(self):
        """Successful execution in closed state."""
        cb = CircuitBreaker("test")
        result = cb.execute(lambda: 42)
        assert result == 42
        assert cb.stats.total_requests == 1
        assert cb.stats.successful_requests == 1
        assert cb.stats.failed_requests == 0

    def test_execute_failure(self):
        """Failed execution records failure."""
        cb = CircuitBreaker("test")

        with pytest.raises(ValueError):
            cb.execute(lambda: (_ for _ in ()).throw(ValueError("test")))

        assert cb.stats.total_requests == 1
        assert cb.stats.failed_requests == 1
        assert cb.is_closed  # Still closed after one failure

    def test_execute_passes_through_result(self):
        """Execute returns function result."""
        cb = CircuitBreaker("test")
        assert cb.execute(lambda: "hello") == "hello"
        assert cb.execute(lambda: [1, 2, 3]) == [1, 2, 3]
        assert cb.execute(lambda: {"key": "value"}) == {"key": "value"}


class TestCircuitBreakerStateTransitions:
    """Tests for state transitions."""

    def test_trips_on_failure_threshold(self):
        """Circuit trips to open after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        def fail():
            raise RuntimeError("failure")

        # Trigger failures up to threshold
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.execute(fail)

        assert cb.state == CircuitState.OPEN
        assert cb.is_open

    def test_rejects_when_open(self):
        """Open circuit rejects new requests."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        # Now should reject
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.execute(lambda: "should not run")

        assert exc_info.value.backend_name == "test"
        assert cb.stats.rejected_requests == 1

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        cb = CircuitBreaker("test", config)

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_on_success_in_half_open(self):
        """Circuit closes after successful requests in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1, timeout_seconds=0.1, success_threshold=2
        )
        cb = CircuitBreaker("test", config)

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        # Wait for timeout
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Successful requests should close it
        cb.execute(lambda: 1)
        cb.execute(lambda: 2)
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Circuit reopens on failure in half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        cb = CircuitBreaker("test", config)

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        # Wait for timeout
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Failure should reopen
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))
        assert cb.state == CircuitState.OPEN

    def test_half_open_request_limit(self):
        """Half-open state limits number of requests."""
        config = CircuitBreakerConfig(
            failure_threshold=1, timeout_seconds=0.1, half_open_max_requests=2
        )
        cb = CircuitBreaker("test", config)

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        # Wait for timeout
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # First two requests should pass (no actual success recorded yet)
        cb.execute(lambda: 1)

        # Force back to half-open for testing
        cb._state = CircuitState.HALF_OPEN
        cb._half_open_requests = 2

        # Third should be rejected
        with pytest.raises(CircuitOpenError):
            cb.execute(lambda: 3)


class TestCircuitBreakerFailureRate:
    """Tests for failure rate calculation."""

    def test_trips_on_failure_rate(self):
        """Circuit trips when failure rate exceeds threshold."""
        config = CircuitBreakerConfig(
            failure_threshold=100,  # High absolute threshold (won't trigger)
            failure_rate_threshold=0.5,  # 50% failure rate will trigger
            window_size=10,
        )
        cb = CircuitBreaker("test", config)

        # Alternate failures and successes to fill window
        # Then add more failures to exceed 50% rate
        # Start with 4 successes, then 6 failures = 60% failure rate
        for _ in range(4):
            cb.execute(lambda: True)

        for _ in range(6):
            try:
                cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))
            except (RuntimeError, CircuitOpenError):
                pass

        # Should have tripped due to failure rate (6/10 = 60% > 50%)
        assert cb.state == CircuitState.OPEN

    def test_get_failure_rate(self):
        """Get failure rate from sliding window."""
        config = CircuitBreakerConfig(window_size=4)
        cb = CircuitBreaker("test", config)

        # 2 successes, 2 failures = 50% failure rate
        cb.execute(lambda: 1)
        cb.execute(lambda: 2)
        try:
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        try:
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass

        assert cb.get_failure_rate() == 0.5


class TestCircuitBreakerExcludedExceptions:
    """Tests for excluded exceptions."""

    def test_excluded_exceptions_not_counted(self):
        """Excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2, excluded_exceptions={ValueError}
        )
        cb = CircuitBreaker("test", config)

        # ValueErrors shouldn't count
        for _ in range(5):
            with pytest.raises(ValueError):
                cb.execute(lambda: (_ for _ in ()).throw(ValueError()))

        assert cb.is_closed
        assert cb.stats.failed_requests == 0

    def test_non_excluded_exceptions_counted(self):
        """Non-excluded exceptions count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2, excluded_exceptions={ValueError}
        )
        cb = CircuitBreaker("test", config)

        # RuntimeErrors should count
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        assert cb.is_open


class TestCircuitBreakerCallbacks:
    """Tests for callbacks and events."""

    def test_state_change_callback(self):
        """State change callback is invoked."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        callback = Mock()
        cb = CircuitBreaker("test", config, on_state_change=callback)

        # Trip to open
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        callback.assert_called_with(CircuitState.CLOSED, CircuitState.OPEN)

        # Wait for half-open
        time.sleep(0.15)
        _ = cb.state  # Trigger transition check

        callback.assert_called_with(CircuitState.OPEN, CircuitState.HALF_OPEN)

    def test_event_callback(self):
        """Event callback is invoked."""
        config = CircuitBreakerConfig(failure_threshold=1)
        events = []
        cb = CircuitBreaker("test", config, event_callback=lambda e: events.append(e))

        # Trip to open
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        assert len(events) == 1
        assert events[0].event_type == ResilienceEventType.CIRCUIT_OPENED
        assert events[0].backend_name == "test"
        assert events[0].metadata["old_state"] == "closed"
        assert events[0].metadata["new_state"] == "open"

    def test_callback_exception_handled(self):
        """Callback exceptions don't break circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)

        def bad_callback(old, new):
            raise RuntimeError("callback error")

        cb = CircuitBreaker("test", config, on_state_change=bad_callback)

        # Should not raise callback exception
        with pytest.raises(RuntimeError, match="failure"):  # Original exception
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError("failure")))

        assert cb.is_open


class TestCircuitBreakerControl:
    """Tests for manual control methods."""

    def test_reset(self):
        """Reset returns circuit to closed state."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))
        assert cb.is_open

        # Reset
        cb.reset()
        assert cb.is_closed
        assert cb.stats.state_changes == 2  # open + reset

    def test_force_open(self):
        """Force open trips the circuit."""
        cb = CircuitBreaker("test")
        assert cb.is_closed

        cb.force_open()
        assert cb.is_open

    def test_force_open_already_open(self):
        """Force open when already open is no-op."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        state_changes = cb.stats.state_changes
        cb.force_open()
        assert cb.stats.state_changes == state_changes  # No change


class TestCircuitBreakerStats:
    """Tests for statistics."""

    def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        # Mix of successes and failures
        cb.execute(lambda: 1)
        cb.execute(lambda: 2)
        try:
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass

        assert cb.stats.total_requests == 3
        assert cb.stats.successful_requests == 2
        assert cb.stats.failed_requests == 1

    def test_get_stats_dict(self):
        """Get stats as dictionary."""
        cb = CircuitBreaker("test")
        cb.execute(lambda: 1)

        stats = cb.get_stats()
        assert stats["state"] == "closed"
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert "failure_rate" in stats


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_execution(self):
        """Circuit breaker handles concurrent execution."""
        config = CircuitBreakerConfig(failure_threshold=100)
        cb = CircuitBreaker("test", config)
        results = []
        errors = []

        def worker(n):
            try:
                result = cb.execute(lambda: n * 2)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        assert cb.stats.total_requests == 20

    def test_concurrent_failures(self):
        """Circuit breaker handles concurrent failures."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)
        errors = []

        def worker():
            try:
                cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))
            except (RuntimeError, CircuitOpenError) as e:
                errors.append(type(e))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some should fail with RuntimeError, some with CircuitOpenError
        assert RuntimeError in errors or CircuitOpenError in errors
        assert cb.is_open


class TestCircuitBreakerEdgeCases:
    """Edge case tests."""

    def test_time_until_retry(self):
        """Time until retry calculation."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=10)
        cb = CircuitBreaker("test", config)

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.execute(lambda: 1)

        # Should be close to 10 seconds
        assert 9 < exc_info.value.time_until_retry <= 10

    def test_empty_window_failure_rate(self):
        """Failure rate with empty window returns 0."""
        cb = CircuitBreaker("test")
        assert cb.get_failure_rate() == 0.0

    def test_successful_after_many_failures(self):
        """Successful execution after many failures."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)

        # Almost at threshold
        for _ in range(4):
            try:
                cb.execute(lambda: (_ for _ in ()).throw(RuntimeError()))
            except RuntimeError:
                pass

        # This success doesn't trip it
        result = cb.execute(lambda: "success")
        assert result == "success"
        assert cb.is_closed
