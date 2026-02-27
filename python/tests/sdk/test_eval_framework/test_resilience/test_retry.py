"""Tests for retry handler implementation."""

import time
from unittest.mock import Mock

import pytest

from fi.evals.framework.resilience.retry import (
    RetryHandler,
    RetryStats,
    with_retry,
    retry_on,
)
from fi.evals.framework.resilience.types import (
    RetryConfig,
    RetryExhaustedError,
    ResilienceEventType,
)


class TestRetryHandlerBasic:
    """Basic functionality tests."""

    def test_success_on_first_attempt(self):
        """Successful function returns immediately."""
        handler = RetryHandler("test")
        result = handler.execute(lambda: 42)

        assert result == 42
        assert handler.stats.total_calls == 1
        assert handler.stats.successful_first_attempt == 1
        assert handler.stats.total_retries == 0

    def test_success_after_retry(self):
        """Function succeeds after retries."""
        config = RetryConfig(max_retries=3, base_delay_seconds=0.01)
        handler = RetryHandler("test", config)

        attempts = [0]

        def flaky():
            attempts[0] += 1
            if attempts[0] < 3:
                raise TimeoutError("temporary failure")
            return "success"

        result = handler.execute(flaky)

        assert result == "success"
        assert handler.stats.successful_after_retry == 1
        assert handler.stats.total_retries == 2

    def test_failure_exhausts_retries(self):
        """Function fails all retries."""
        config = RetryConfig(max_retries=2, base_delay_seconds=0.01)
        handler = RetryHandler("test", config)

        def always_fail():
            raise TimeoutError("always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            handler.execute(always_fail)

        assert exc_info.value.backend_name == "test"
        assert exc_info.value.attempts == 3  # 1 initial + 2 retries
        assert isinstance(exc_info.value.last_error, TimeoutError)
        assert handler.stats.failed_all_retries == 1


class TestRetryHandlerRetryableExceptions:
    """Tests for retryable exception handling."""

    def test_retryable_exception_triggers_retry(self):
        """Retryable exceptions cause retry."""
        config = RetryConfig(
            max_retries=3,
            base_delay_seconds=0.01,
            retryable_exceptions={TimeoutError},
        )
        handler = RetryHandler("test", config)

        attempts = [0]

        def fail_twice():
            attempts[0] += 1
            if attempts[0] <= 2:
                raise TimeoutError()
            return "success"

        result = handler.execute(fail_twice)
        assert result == "success"
        assert handler.stats.total_retries == 2

    def test_non_retryable_exception_raises_immediately(self):
        """Non-retryable exceptions raise immediately."""
        config = RetryConfig(
            max_retries=3,
            base_delay_seconds=0.01,
            retryable_exceptions={TimeoutError},
        )
        handler = RetryHandler("test", config)

        def raise_value_error():
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            handler.execute(raise_value_error)

        assert handler.stats.total_retries == 0
        assert handler.stats.failed_all_retries == 0

    def test_status_code_exception(self):
        """Exception with status_code attribute triggers retry."""
        config = RetryConfig(
            max_retries=2,
            base_delay_seconds=0.01,
            retryable_status_codes={503},
        )
        handler = RetryHandler("test", config)

        class HTTPError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        attempts = [0]

        def service_unavailable():
            attempts[0] += 1
            if attempts[0] == 1:
                raise HTTPError(503)
            return "recovered"

        result = handler.execute(service_unavailable)
        assert result == "recovered"


class TestRetryHandlerBackoff:
    """Tests for exponential backoff."""

    def test_exponential_backoff_delays(self):
        """Delays increase exponentially."""
        config = RetryConfig(
            max_retries=3,
            base_delay_seconds=0.05,
            exponential_base=2.0,
            jitter=False,  # Disable for predictable timing
        )
        handler = RetryHandler("test", config)

        timestamps = []

        def track_time():
            timestamps.append(time.monotonic())
            raise TimeoutError()

        with pytest.raises(RetryExhaustedError):
            handler.execute(track_time)

        # Check delays are increasing
        delays = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        assert len(delays) == 3
        # First delay ~0.05s, second ~0.1s, third ~0.2s
        assert 0.04 < delays[0] < 0.08
        assert 0.08 < delays[1] < 0.15
        assert 0.15 < delays[2] < 0.3

    def test_max_delay_caps_backoff(self):
        """Max delay caps the backoff."""
        config = RetryConfig(
            max_retries=5,
            base_delay_seconds=0.05,
            max_delay_seconds=0.1,
            exponential_base=4.0,
            jitter=False,
        )
        handler = RetryHandler("test", config)

        timestamps = []

        def track_time():
            timestamps.append(time.monotonic())
            raise TimeoutError()

        with pytest.raises(RetryExhaustedError):
            handler.execute(track_time)

        # All delays should be capped at 0.1s after first few
        delays = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        for delay in delays[2:]:  # Skip first two
            assert delay < 0.15  # Capped at max_delay

    def test_jitter_adds_randomness(self):
        """Jitter adds randomness to delays."""
        config = RetryConfig(
            max_retries=5,
            base_delay_seconds=0.1,
            jitter=True,
            jitter_factor=0.5,  # High jitter for visibility
        )
        handler = RetryHandler("test", config)

        # Run multiple times and collect delays
        all_delays = []

        for _ in range(3):
            timestamps = []

            def track_time():
                timestamps.append(time.monotonic())
                raise TimeoutError()

            with pytest.raises(RetryExhaustedError):
                handler.execute(track_time)

            delays = [
                timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
            ]
            all_delays.append(delays[0])

        # With high jitter, delays should vary
        # (this is probabilistic but very unlikely to all be same)
        assert len(set(round(d * 100) for d in all_delays)) > 1


class TestRetryHandlerEvents:
    """Tests for event callbacks."""

    def test_retry_event_emitted(self):
        """Retry events are emitted."""
        config = RetryConfig(max_retries=2, base_delay_seconds=0.01)
        events = []
        handler = RetryHandler("test", config, event_callback=lambda e: events.append(e))

        attempts = [0]

        def fail_once():
            attempts[0] += 1
            if attempts[0] == 1:
                raise TimeoutError("first")
            return "ok"

        handler.execute(fail_once)

        assert len(events) == 1
        assert events[0].event_type == ResilienceEventType.RETRY_ATTEMPT
        assert events[0].metadata["attempt"] == 1
        assert "delay_seconds" in events[0].metadata

    def test_exhausted_event_emitted(self):
        """Exhausted event emitted when retries fail."""
        config = RetryConfig(max_retries=1, base_delay_seconds=0.01)
        events = []
        handler = RetryHandler("test", config, event_callback=lambda e: events.append(e))

        with pytest.raises(RetryExhaustedError):
            handler.execute(lambda: (_ for _ in ()).throw(TimeoutError()))

        # Should have retry event + exhausted event
        exhausted_events = [
            e for e in events if e.event_type == ResilienceEventType.RETRY_EXHAUSTED
        ]
        assert len(exhausted_events) == 1
        assert exhausted_events[0].metadata["total_attempts"] == 2

    def test_no_events_on_success(self):
        """No events on immediate success."""
        config = RetryConfig(max_retries=2)
        events = []
        handler = RetryHandler("test", config, event_callback=lambda e: events.append(e))

        handler.execute(lambda: "success")

        assert len(events) == 0

    def test_callback_exception_handled(self):
        """Callback exceptions don't break retry."""
        config = RetryConfig(max_retries=2, base_delay_seconds=0.01)

        def bad_callback(e):
            raise RuntimeError("callback error")

        handler = RetryHandler("test", config, event_callback=bad_callback)

        attempts = [0]

        def fail_once():
            attempts[0] += 1
            if attempts[0] == 1:
                raise TimeoutError()
            return "ok"

        result = handler.execute(fail_once)
        assert result == "ok"


class TestRetryHandlerStats:
    """Tests for statistics."""

    def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        config = RetryConfig(max_retries=2, base_delay_seconds=0.01)
        handler = RetryHandler("test", config)

        # Immediate success
        handler.execute(lambda: 1)

        # Success after retry
        attempts = [0]

        def fail_once():
            attempts[0] += 1
            if attempts[0] == 1:
                raise TimeoutError()
            return 2

        handler.execute(fail_once)

        assert handler.stats.total_calls == 2
        assert handler.stats.successful_first_attempt == 1
        assert handler.stats.successful_after_retry == 1
        assert handler.stats.total_retries == 1

    def test_get_stats(self):
        """Get stats returns correct values."""
        config = RetryConfig(max_retries=2, base_delay_seconds=0.01)
        handler = RetryHandler("test", config)

        handler.execute(lambda: 1)

        stats = handler.get_stats()
        assert stats["total_calls"] == 1
        assert stats["success_rate"] == 1.0


class TestRetryHandlerConfigOverride:
    """Tests for config override."""

    def test_config_override(self):
        """Can override config per call."""
        default_config = RetryConfig(max_retries=1, base_delay_seconds=0.01)
        handler = RetryHandler("test", default_config)

        attempts = [0]

        def fail_twice():
            attempts[0] += 1
            if attempts[0] <= 2:
                raise TimeoutError()
            return "ok"

        # Default config would fail
        # Override with more retries
        override = RetryConfig(max_retries=3, base_delay_seconds=0.01)
        result = handler.execute(fail_twice, config_override=override)

        assert result == "ok"
        assert handler.stats.total_retries == 2


class TestWithRetryDecorator:
    """Tests for the @with_retry decorator."""

    def test_decorator_success(self):
        """Decorated function works on success."""

        @with_retry(RetryConfig(max_retries=2, base_delay_seconds=0.01))
        def success():
            return 42

        assert success() == 42

    def test_decorator_retry(self):
        """Decorated function retries on failure."""
        attempts = [0]

        @with_retry(RetryConfig(max_retries=2, base_delay_seconds=0.01))
        def flaky():
            attempts[0] += 1
            if attempts[0] == 1:
                raise TimeoutError()
            return "ok"

        result = flaky()
        assert result == "ok"
        assert attempts[0] == 2

    def test_decorator_preserves_name(self):
        """Decorator preserves function name."""

        @with_retry()
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_with_args(self):
        """Decorated function accepts arguments."""

        @with_retry()
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_decorator_handler_accessible(self):
        """Handler is accessible on decorated function."""

        @with_retry()
        def func():
            return 1

        func()
        assert hasattr(func, "_retry_handler")
        assert func._retry_handler.stats.total_calls == 1


class TestRetryOnDecorator:
    """Tests for the @retry_on decorator."""

    def test_retry_on_specific_exceptions(self):
        """Retries only on specified exceptions."""
        attempts = [0]

        @retry_on(TimeoutError, max_retries=2, base_delay=0.01)
        def flaky():
            attempts[0] += 1
            if attempts[0] == 1:
                raise TimeoutError()
            return "ok"

        result = flaky()
        assert result == "ok"

    def test_retry_on_different_exception(self):
        """Doesn't retry on non-specified exceptions."""

        @retry_on(TimeoutError, max_retries=2, base_delay=0.01)
        def raise_value():
            raise ValueError()

        with pytest.raises(ValueError):
            raise_value()

    def test_retry_on_multiple_exceptions(self):
        """Can specify multiple exception types."""
        attempts = [0]

        @retry_on(TimeoutError, ConnectionError, max_retries=3, base_delay=0.01)
        def flaky():
            attempts[0] += 1
            if attempts[0] == 1:
                raise TimeoutError()
            if attempts[0] == 2:
                raise ConnectionError()
            return "ok"

        result = flaky()
        assert result == "ok"
        assert attempts[0] == 3
