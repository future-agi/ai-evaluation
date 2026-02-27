"""Tests for token bucket rate limiter implementation."""

import threading
import time

import pytest

from fi.evals.framework.resilience.rate_limiter import (
    TokenBucketRateLimiter,
    RateLimitStats,
)
from fi.evals.framework.resilience.types import (
    RateLimitConfig,
    RateLimitExceededError,
    ResilienceEventType,
)


class TestRateLimiterBasic:
    """Basic functionality tests."""

    def test_initial_state(self):
        """Rate limiter starts with full bucket."""
        config = RateLimitConfig(burst_size=10)
        limiter = TokenBucketRateLimiter("test", config)
        assert limiter.available_tokens == 10
        assert not limiter.is_limited

    def test_acquire_success(self):
        """Successfully acquire tokens."""
        config = RateLimitConfig(burst_size=10)
        limiter = TokenBucketRateLimiter("test", config)

        assert limiter.acquire() is True
        assert limiter.stats.allowed_requests == 1
        # Use approximate comparison due to time-based refill
        assert 8.9 < limiter.available_tokens <= 9.1

    def test_acquire_multiple_tokens(self):
        """Acquire multiple tokens at once."""
        config = RateLimitConfig(burst_size=10)
        limiter = TokenBucketRateLimiter("test", config)

        assert limiter.acquire(tokens=5) is True
        # Use approximate comparison due to time-based refill
        assert 4.9 < limiter.available_tokens <= 5.1

    def test_acquire_rejected_when_empty(self):
        """Acquire fails when bucket is empty."""
        config = RateLimitConfig(burst_size=2, requests_per_second=0.1)
        limiter = TokenBucketRateLimiter("test", config)

        # Drain the bucket
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False

        assert limiter.stats.rejected_requests == 1
        assert limiter.is_limited

    def test_try_acquire(self):
        """try_acquire never blocks."""
        config = RateLimitConfig(burst_size=1, wait_for_token=True)
        limiter = TokenBucketRateLimiter("test", config)

        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False  # Doesn't block even with wait_for_token=True


class TestRateLimiterRefill:
    """Tests for token refill behavior."""

    def test_tokens_refill_over_time(self):
        """Tokens refill based on elapsed time."""
        config = RateLimitConfig(
            burst_size=10, requests_per_second=100  # Fast refill for testing
        )
        limiter = TokenBucketRateLimiter("test", config)

        # Drain some tokens
        for _ in range(5):
            limiter.acquire()
        # Use approximate comparison due to time-based refill
        assert 4.9 < limiter.available_tokens <= 5.5

        # Wait for refill
        time.sleep(0.05)  # Should add ~5 tokens at 100/s

        # Should have refilled
        assert limiter.available_tokens >= 9

    def test_tokens_cap_at_burst_size(self):
        """Tokens don't exceed burst size."""
        config = RateLimitConfig(
            burst_size=10, requests_per_second=1000  # Very fast refill
        )
        limiter = TokenBucketRateLimiter("test", config)

        # Wait for potential overfill
        time.sleep(0.1)

        # Should still be capped at 10
        assert limiter.available_tokens == 10

    def test_refill_after_drain(self):
        """Bucket refills after being completely drained."""
        config = RateLimitConfig(burst_size=2, requests_per_second=100)
        limiter = TokenBucketRateLimiter("test", config)

        # Drain completely
        limiter.acquire()
        limiter.acquire()
        assert limiter.available_tokens < 1

        # Wait for refill
        time.sleep(0.03)  # Should add ~3 tokens

        # Should have some tokens now
        assert limiter.available_tokens >= 2


class TestRateLimiterBlocking:
    """Tests for blocking mode."""

    def test_blocking_acquire(self):
        """Blocking acquire waits for tokens."""
        config = RateLimitConfig(
            burst_size=1, requests_per_second=100, wait_for_token=True, max_wait_seconds=1
        )
        limiter = TokenBucketRateLimiter("test", config)

        # First acquire uses the token
        assert limiter.acquire() is True

        # Second should wait and succeed
        start = time.monotonic()
        assert limiter.acquire() is True
        elapsed = time.monotonic() - start

        # Should have waited for refill
        assert elapsed > 0.005  # At least some wait
        assert limiter.stats.waited_requests == 1

    def test_blocking_timeout(self):
        """Blocking acquire times out."""
        config = RateLimitConfig(
            burst_size=1,
            requests_per_second=0.1,  # Very slow refill
            wait_for_token=True,
            max_wait_seconds=0.1,
        )
        limiter = TokenBucketRateLimiter("test", config)

        # First acquire
        assert limiter.acquire() is True

        # Second should timeout
        with pytest.raises(RateLimitExceededError) as exc_info:
            limiter.acquire()

        assert exc_info.value.backend_name == "test"
        assert limiter.stats.rejected_requests == 1

    def test_blocking_override(self):
        """Can override blocking behavior per call."""
        config = RateLimitConfig(burst_size=1, wait_for_token=True)
        limiter = TokenBucketRateLimiter("test", config)

        limiter.acquire()
        # Override to non-blocking
        assert limiter.acquire(blocking=False) is False


class TestRateLimiterEvents:
    """Tests for event callbacks."""

    def test_event_on_rate_limit(self):
        """Event emitted when rate limited."""
        config = RateLimitConfig(burst_size=1)
        events = []
        limiter = TokenBucketRateLimiter(
            "test", config, event_callback=lambda e: events.append(e)
        )

        limiter.acquire()
        limiter.acquire()  # This should trigger event

        assert len(events) == 1
        assert events[0].event_type == ResilienceEventType.RATE_LIMITED
        assert events[0].backend_name == "test"
        assert "burst_size" in events[0].metadata

    def test_no_event_on_success(self):
        """No event on successful acquire."""
        config = RateLimitConfig(burst_size=10)
        events = []
        limiter = TokenBucketRateLimiter(
            "test", config, event_callback=lambda e: events.append(e)
        )

        limiter.acquire()
        limiter.acquire()

        assert len(events) == 0

    def test_event_callback_exception_handled(self):
        """Event callback exceptions don't break limiter."""
        config = RateLimitConfig(burst_size=1)

        def bad_callback(e):
            raise RuntimeError("callback error")

        limiter = TokenBucketRateLimiter("test", config, event_callback=bad_callback)

        limiter.acquire()
        # Should not raise
        result = limiter.acquire()
        assert result is False


class TestRateLimiterControl:
    """Tests for control methods."""

    def test_reset(self):
        """Reset fills bucket to capacity."""
        config = RateLimitConfig(burst_size=10)
        limiter = TokenBucketRateLimiter("test", config)

        # Drain some tokens
        for _ in range(8):
            limiter.acquire()
        # Use approximate comparison due to time-based refill
        assert 1.9 < limiter.available_tokens <= 2.5

        # Reset
        limiter.reset()
        assert limiter.available_tokens == 10

    def test_get_stats(self):
        """Get stats returns correct values."""
        config = RateLimitConfig(burst_size=5, requests_per_second=10)
        limiter = TokenBucketRateLimiter("test", config)

        limiter.acquire()
        limiter.acquire()
        limiter.try_acquire()

        stats = limiter.get_stats()
        assert stats["burst_size"] == 5
        assert stats["requests_per_second"] == 10
        assert stats["total_requests"] == 3
        assert stats["allowed_requests"] == 3

    def test_get_wait_time(self):
        """Get estimated wait time."""
        config = RateLimitConfig(burst_size=2, requests_per_second=10)
        limiter = TokenBucketRateLimiter("test", config)

        # Full bucket - no wait
        assert limiter.get_wait_time() == 0.0

        # Drain bucket
        limiter.acquire()
        limiter.acquire()

        # Should need to wait for 1 token at 10/s = 0.1s
        wait = limiter.get_wait_time()
        assert 0.05 < wait < 0.15


class TestRateLimiterThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_acquire(self):
        """Concurrent acquires are handled correctly."""
        config = RateLimitConfig(burst_size=100, requests_per_second=1000)
        limiter = TokenBucketRateLimiter("test", config)

        results = []

        def worker():
            result = limiter.acquire()
            results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed (burst_size = 100)
        assert all(results)
        assert len(results) == 50
        assert limiter.stats.total_requests == 50

    def test_concurrent_drain(self):
        """Concurrent acquires properly drain bucket."""
        config = RateLimitConfig(burst_size=10, requests_per_second=0.1)
        limiter = TokenBucketRateLimiter("test", config)

        results = []

        def worker():
            result = limiter.try_acquire()
            results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only 10 should succeed
        success_count = sum(1 for r in results if r)
        assert success_count == 10
        assert limiter.stats.rejected_requests == 10


class TestRateLimiterEdgeCases:
    """Edge case tests."""

    def test_acquire_more_than_burst(self):
        """Cannot acquire more tokens than burst size allows."""
        config = RateLimitConfig(burst_size=5)
        limiter = TokenBucketRateLimiter("test", config)

        # Try to acquire more than burst size
        assert limiter.acquire(tokens=10) is False

    def test_fractional_tokens(self):
        """Fractional tokens work correctly."""
        config = RateLimitConfig(burst_size=10, requests_per_second=100)
        limiter = TokenBucketRateLimiter("test", config)

        # Drain to near empty
        for _ in range(10):
            limiter.acquire()

        # Wait for partial refill
        time.sleep(0.005)  # 0.5 tokens at 100/s

        # Should have fractional tokens
        tokens = limiter.available_tokens
        assert 0 < tokens < 1

    def test_zero_initial_tokens(self):
        """Limiter works with very small initial state."""
        config = RateLimitConfig(burst_size=1, requests_per_second=100)
        limiter = TokenBucketRateLimiter("test", config)

        # Drain
        limiter.acquire()

        # Wait very short time
        time.sleep(0.001)

        # Should have some fractional tokens
        assert 0 < limiter.available_tokens < 1

    def test_high_rate_limiter(self):
        """High rate limiter allows many requests."""
        config = RateLimitConfig(burst_size=1000, requests_per_second=10000)
        limiter = TokenBucketRateLimiter("test", config)

        success = 0
        for _ in range(500):
            if limiter.acquire():
                success += 1

        assert success == 500
