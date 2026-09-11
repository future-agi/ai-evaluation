"""
Resilience module for production hardening.

Provides circuit breaker, rate limiting, retry, graceful degradation,
and health check capabilities for evaluation backends.

Example:
    from fi.evals.framework.backends import ThreadPoolBackend
    from fi.evals.framework.resilience import (
        ResilientBackend,
        wrap_backend,
        CircuitBreakerConfig,
        RateLimitConfig,
        RetryConfig,
        ResilienceConfig,
    )

    # Option 1: Full configuration
    config = ResilienceConfig(
        circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
        rate_limit=RateLimitConfig(requests_per_second=10),
        retry=RetryConfig(max_retries=3),
    )
    backend = ResilientBackend(ThreadPoolBackend(), config)

    # Option 2: Convenience function
    backend = wrap_backend(
        ThreadPoolBackend(),
        circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
        rate_limit=RateLimitConfig(requests_per_second=10),
    )
"""

# Types and configurations
from .types import (
    # Enums
    CircuitState,
    HealthStatus,
    ResilienceEventType,
    # Exceptions
    ResilienceError,
    CircuitOpenError,
    RateLimitExceededError,
    RetryExhaustedError,
    DegradedServiceError,
    # Configuration dataclasses
    CircuitBreakerConfig,
    RateLimitConfig,
    RetryConfig,
    DegradationConfig,
    HealthCheckConfig,
    ResilienceConfig,
    # Event
    ResilienceEvent,
    EventCallback,
)

# Circuit breaker
from .circuit_breaker import (
    CircuitBreaker,
    CircuitStats,
)

# Rate limiter
from .rate_limiter import (
    TokenBucketRateLimiter,
    RateLimitStats,
)

# Retry handler
from .retry import (
    RetryHandler,
    RetryStats,
    with_retry,
    retry_on,
)

# Graceful degradation
from .degradation import (
    FallbackChain,
    FallbackStats,
    DegradationHandler,
    with_fallback,
    with_fallback_func,
)

# Health checks
from .health import (
    HealthChecker,
    HealthCheckResult,
    HealthStats,
    HealthRegistry,
)

# Wrapper
from .wrapper import (
    ResilientBackend,
    ResilientBackendConfig,
    wrap_backend,
)


__all__ = [
    # Types - Enums
    "CircuitState",
    "HealthStatus",
    "ResilienceEventType",
    # Types - Exceptions
    "ResilienceError",
    "CircuitOpenError",
    "RateLimitExceededError",
    "RetryExhaustedError",
    "DegradedServiceError",
    # Types - Configuration
    "CircuitBreakerConfig",
    "RateLimitConfig",
    "RetryConfig",
    "DegradationConfig",
    "HealthCheckConfig",
    "ResilienceConfig",
    # Types - Event
    "ResilienceEvent",
    "EventCallback",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitStats",
    # Rate limiter
    "TokenBucketRateLimiter",
    "RateLimitStats",
    # Retry
    "RetryHandler",
    "RetryStats",
    "with_retry",
    "retry_on",
    # Degradation
    "FallbackChain",
    "FallbackStats",
    "DegradationHandler",
    "with_fallback",
    "with_fallback_func",
    # Health
    "HealthChecker",
    "HealthCheckResult",
    "HealthStats",
    "HealthRegistry",
    # Wrapper
    "ResilientBackend",
    "ResilientBackendConfig",
    "wrap_backend",
]
