"""
Token bucket rate limiter implementation.

Controls request throughput to prevent overwhelming backends.
"""

import threading
import time
from dataclasses import dataclass

from .types import (
    EventCallback,
    RateLimitConfig,
    RateLimitExceededError,
    ResilienceEvent,
    ResilienceEventType,
)


@dataclass
class RateLimitStats:
    """Statistics for rate limiter."""

    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    waited_requests: int = 0
    total_wait_time_ms: float = 0


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter implementation.

    Tokens are added at a fixed rate up to a maximum bucket size.
    Each request consumes one token. If no tokens available,
    request is rejected or waits (based on config).

    Example:
        config = RateLimitConfig(requests_per_second=10, burst_size=20)
        limiter = TokenBucketRateLimiter("my_backend", config)

        if limiter.acquire():
            # Proceed with request
            pass
        else:
            # Rate limited
            pass
    """

    def __init__(
        self,
        name: str,
        config: RateLimitConfig | None = None,
        event_callback: EventCallback | None = None,
    ):
        """
        Initialize rate limiter.

        Args:
            name: Name for this rate limiter (typically backend name)
            config: Rate limit configuration
            event_callback: Callback for rate limit events
        """
        self.name = name
        self.config = config or RateLimitConfig()
        self.event_callback = event_callback

        self._tokens = float(self.config.burst_size)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

        self.stats = RateLimitStats()

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (may be fractional)."""
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def is_limited(self) -> bool:
        """Check if currently rate limited (no tokens available)."""
        return self.available_tokens < 1

    def acquire(self, tokens: int = 1, blocking: bool | None = None) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire (default 1)
            blocking: Override config.wait_for_token

        Returns:
            True if tokens acquired, False if rejected

        Raises:
            RateLimitExceededError: If blocking and max wait exceeded
        """
        should_block = blocking if blocking is not None else self.config.wait_for_token

        with self._lock:
            self.stats.total_requests += 1
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self.stats.allowed_requests += 1
                return True

            if not should_block:
                self.stats.rejected_requests += 1
                self._emit_rate_limited()
                return False

        # Blocking mode - wait for tokens
        return self._wait_for_tokens(tokens)

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if not available
        """
        return self.acquire(tokens, blocking=False)

    def _wait_for_tokens(self, tokens: int) -> bool:
        """Wait for tokens to become available."""
        start = time.monotonic()
        max_wait = self.config.max_wait_seconds

        while True:
            elapsed = time.monotonic() - start
            if elapsed >= max_wait:
                with self._lock:
                    self.stats.rejected_requests += 1
                self._emit_rate_limited()
                raise RateLimitExceededError(self.name, max_wait - elapsed)

            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self.stats.allowed_requests += 1
                    self.stats.waited_requests += 1
                    self.stats.total_wait_time_ms += elapsed * 1000
                    return True

                # Calculate wait time for needed tokens
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self.config.requests_per_second

            # Wait outside lock, but cap at reasonable intervals
            sleep_time = min(wait_time, max_wait - elapsed, 0.1)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time (called under lock)."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        tokens_to_add = elapsed * self.config.requests_per_second
        self._tokens = min(self._tokens + tokens_to_add, float(self.config.burst_size))

    def _emit_rate_limited(self) -> None:
        """Emit rate limited event."""
        if not self.event_callback:
            return

        event = ResilienceEvent(
            event_type=ResilienceEventType.RATE_LIMITED,
            backend_name=self.name,
            metadata={
                "available_tokens": self._tokens,
                "burst_size": self.config.burst_size,
                "requests_per_second": self.config.requests_per_second,
            },
        )

        try:
            self.event_callback(event)
        except Exception:
            pass

    def reset(self) -> None:
        """Reset bucket to full capacity."""
        with self._lock:
            self._tokens = float(self.config.burst_size)
            self._last_update = time.monotonic()

    def get_stats(self) -> dict:
        """Get rate limiter statistics as dictionary."""
        with self._lock:
            self._refill()
            return {
                "available_tokens": self._tokens,
                "burst_size": self.config.burst_size,
                "requests_per_second": self.config.requests_per_second,
                "total_requests": self.stats.total_requests,
                "allowed_requests": self.stats.allowed_requests,
                "rejected_requests": self.stats.rejected_requests,
                "waited_requests": self.stats.waited_requests,
                "total_wait_time_ms": self.stats.total_wait_time_ms,
            }

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get estimated wait time for acquiring tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Estimated seconds to wait (0 if tokens available)
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            tokens_needed = tokens - self._tokens
            return tokens_needed / self.config.requests_per_second
