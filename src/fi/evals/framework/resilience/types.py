"""
Core types for the resilience module.

Provides enums, configuration dataclasses, and exception classes for
circuit breaker, rate limiting, retry, graceful degradation, and health checks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Type


# === Enums ===


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Circuit tripped, requests rejected
    HALF_OPEN = "half_open"  # Testing if backend has recovered


class ResilienceEventType(Enum):
    """Types of resilience events for observability."""

    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    CIRCUIT_HALF_OPEN = "circuit_half_open"
    RATE_LIMITED = "rate_limited"
    RETRY_ATTEMPT = "retry_attempt"
    RETRY_EXHAUSTED = "retry_exhausted"
    FALLBACK_INVOKED = "fallback_invoked"
    FALLBACK_USED = "fallback_used"
    DEGRADATION_ACTIVE = "degradation_active"
    HEALTH_CHECK_PASSED = "health_check_passed"
    HEALTH_CHECK_FAILED = "health_check_failed"


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# === Exceptions ===


class ResilienceError(Exception):
    """Base exception for resilience errors."""

    pass


class CircuitOpenError(ResilienceError):
    """Raised when circuit breaker is open."""

    def __init__(self, backend_name: str, time_until_retry: float):
        self.backend_name = backend_name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker open for '{backend_name}'. "
            f"Retry in {time_until_retry:.1f}s"
        )


class RateLimitExceededError(ResilienceError):
    """Raised when rate limit is exceeded."""

    def __init__(self, backend_name: str, wait_time: float):
        self.backend_name = backend_name
        self.wait_time = wait_time
        super().__init__(
            f"Rate limit exceeded for '{backend_name}'. " f"Wait {wait_time:.2f}s"
        )


class RetryExhaustedError(ResilienceError):
    """Raised when all retries are exhausted."""

    def __init__(self, backend_name: str, attempts: int, last_error: Exception):
        self.backend_name = backend_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Retry exhausted for '{backend_name}' after {attempts} attempts. "
            f"Last error: {last_error}"
        )


class DegradedServiceError(ResilienceError):
    """Raised when service is in degraded mode and all fallbacks failed."""

    pass


# === Configuration Dataclasses ===


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes in half-open to close
    timeout_seconds: float = 30.0  # Time in open state before half-open
    half_open_max_requests: int = 3  # Max requests in half-open state
    failure_rate_threshold: float = 0.5  # Alternative: failure rate trigger
    window_size: int = 10  # Sliding window for failure rate
    excluded_exceptions: Set[Type[Exception]] = field(default_factory=set)

    def __post_init__(self):
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0 < self.failure_rate_threshold <= 1:
            raise ValueError("failure_rate_threshold must be between 0 and 1")


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter (token bucket)."""

    requests_per_second: float = 10.0  # Token refill rate
    burst_size: int = 20  # Maximum bucket capacity
    wait_for_token: bool = False  # Block until token available
    max_wait_seconds: float = 5.0  # Max wait time if blocking

    def __post_init__(self):
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.burst_size < 1:
            raise ValueError("burst_size must be at least 1")
        if self.max_wait_seconds < 0:
            raise ValueError("max_wait_seconds cannot be negative")


@dataclass
class RetryConfig:
    """Configuration for retry with exponential backoff."""

    max_retries: int = 3  # Maximum retry attempts
    base_delay_seconds: float = 1.0  # Initial delay
    max_delay_seconds: float = 60.0  # Maximum delay cap
    exponential_base: float = 2.0  # Multiplier for exponential backoff
    jitter: bool = True  # Add random jitter
    jitter_factor: float = 0.25  # Max jitter as fraction of delay
    retryable_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {TimeoutError, ConnectionError, IOError}
    )
    retryable_status_codes: Set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )

    def __post_init__(self):
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds cannot be negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if self.exponential_base < 1:
            raise ValueError("exponential_base must be at least 1")
        if not 0 <= self.jitter_factor <= 1:
            raise ValueError("jitter_factor must be between 0 and 1")


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""

    enable_fallback: bool = True
    fallback_timeout_seconds: float = 5.0
    fallback_on_circuit_open: bool = True
    fallback_on_rate_limit: bool = False
    fallback_on_timeout: bool = True
    fallback_on_exceptions: Set[Type[Exception]] = field(
        default_factory=set  # Empty set = fallback on all exceptions
    )

    def __post_init__(self):
        if self.fallback_timeout_seconds <= 0:
            raise ValueError("fallback_timeout_seconds must be positive")


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""

    enabled: bool = True
    interval_seconds: float = 30.0  # Check interval
    timeout_seconds: float = 5.0  # Health check timeout
    healthy_threshold: int = 2  # Consecutive passes for healthy
    unhealthy_threshold: int = 3  # Consecutive fails for unhealthy
    include_in_metrics: bool = True

    def __post_init__(self):
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.healthy_threshold < 1:
            raise ValueError("healthy_threshold must be at least 1")
        if self.unhealthy_threshold < 1:
            raise ValueError("unhealthy_threshold must be at least 1")


@dataclass
class ResilienceConfig:
    """Combined resilience configuration."""

    circuit_breaker: Optional[CircuitBreakerConfig] = None
    rate_limit: Optional[RateLimitConfig] = None
    retry: Optional[RetryConfig] = None
    degradation: Optional[DegradationConfig] = None
    health_check: Optional[HealthCheckConfig] = None

    @classmethod
    def default(cls) -> "ResilienceConfig":
        """Create default configuration with all features enabled."""
        return cls(
            circuit_breaker=CircuitBreakerConfig(),
            rate_limit=RateLimitConfig(),
            retry=RetryConfig(),
            degradation=DegradationConfig(),
            health_check=HealthCheckConfig(),
        )

    @classmethod
    def minimal(cls) -> "ResilienceConfig":
        """Create minimal configuration with only retry."""
        return cls(retry=RetryConfig())

    @classmethod
    def strict(cls) -> "ResilienceConfig":
        """Create strict configuration for critical systems."""
        return cls(
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=60.0,
            ),
            rate_limit=RateLimitConfig(
                requests_per_second=5.0,
                burst_size=10,
            ),
            retry=RetryConfig(
                max_retries=5,
                base_delay_seconds=2.0,
            ),
            degradation=DegradationConfig(),
            health_check=HealthCheckConfig(
                interval_seconds=15.0,
                unhealthy_threshold=2,
            ),
        )


# === Event Dataclass ===


@dataclass
class ResilienceEvent:
    """Event emitted by resilience components."""

    event_type: ResilienceEventType
    backend_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "backend_name": self.backend_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# Type alias for event callbacks
EventCallback = Callable[[ResilienceEvent], None]
