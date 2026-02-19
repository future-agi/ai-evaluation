"""Tests for resilience types module."""

import pytest
from datetime import datetime, timezone

from fi.evals.framework.resilience.types import (
    # Enums
    CircuitState,
    ResilienceEventType,
    HealthStatus,
    # Exceptions
    ResilienceError,
    CircuitOpenError,
    RateLimitExceededError,
    RetryExhaustedError,
    DegradedServiceError,
    # Config
    CircuitBreakerConfig,
    RateLimitConfig,
    RetryConfig,
    DegradationConfig,
    HealthCheckConfig,
    ResilienceConfig,
    # Event
    ResilienceEvent,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_circuit_states(self):
        """Test all circuit states exist."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_state_count(self):
        """Test there are exactly 3 states."""
        assert len(CircuitState) == 3


class TestResilienceEventType:
    """Tests for ResilienceEventType enum."""

    def test_event_types(self):
        """Test all event types exist."""
        assert ResilienceEventType.CIRCUIT_OPENED.value == "circuit_opened"
        assert ResilienceEventType.CIRCUIT_CLOSED.value == "circuit_closed"
        assert ResilienceEventType.RATE_LIMITED.value == "rate_limited"
        assert ResilienceEventType.RETRY_ATTEMPT.value == "retry_attempt"
        assert ResilienceEventType.FALLBACK_INVOKED.value == "fallback_invoked"

    def test_event_count(self):
        """Test there are exactly 11 event types."""
        assert len(ResilienceEventType) == 11


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_statuses(self):
        """Test all health statuses exist."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestExceptions:
    """Tests for resilience exceptions."""

    def test_resilience_error_base(self):
        """Test ResilienceError is base exception."""
        err = ResilienceError("test error")
        assert str(err) == "test error"
        assert isinstance(err, Exception)

    def test_circuit_open_error(self):
        """Test CircuitOpenError."""
        err = CircuitOpenError("backend1", 30.5)
        assert err.backend_name == "backend1"
        assert err.time_until_retry == 30.5
        assert "backend1" in str(err)
        assert "30.5" in str(err)
        assert isinstance(err, ResilienceError)

    def test_rate_limit_exceeded_error(self):
        """Test RateLimitExceededError."""
        err = RateLimitExceededError("backend2", 1.5)
        assert err.backend_name == "backend2"
        assert err.wait_time == 1.5
        assert "backend2" in str(err)
        assert "1.50" in str(err)
        assert isinstance(err, ResilienceError)

    def test_retry_exhausted_error(self):
        """Test RetryExhaustedError."""
        original = ValueError("original error")
        err = RetryExhaustedError("backend3", 5, original)
        assert err.backend_name == "backend3"
        assert err.attempts == 5
        assert err.last_error is original
        assert "backend3" in str(err)
        assert "5 attempts" in str(err)
        assert isinstance(err, ResilienceError)

    def test_degraded_service_error(self):
        """Test DegradedServiceError."""
        err = DegradedServiceError("all fallbacks failed")
        assert "all fallbacks failed" in str(err)
        assert isinstance(err, ResilienceError)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_defaults(self):
        """Test default values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_seconds == 30.0
        assert config.half_open_max_requests == 3
        assert config.failure_rate_threshold == 0.5
        assert config.window_size == 10
        assert config.excluded_exceptions == set()

    def test_custom_values(self):
        """Test custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            timeout_seconds=60.0,
            excluded_exceptions={ValueError},
        )
        assert config.failure_threshold == 10
        assert config.timeout_seconds == 60.0
        assert ValueError in config.excluded_exceptions

    def test_validation_failure_threshold(self):
        """Test failure_threshold validation."""
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreakerConfig(failure_threshold=0)

    def test_validation_timeout(self):
        """Test timeout_seconds validation."""
        with pytest.raises(ValueError, match="timeout_seconds"):
            CircuitBreakerConfig(timeout_seconds=-1)

    def test_validation_failure_rate(self):
        """Test failure_rate_threshold validation."""
        with pytest.raises(ValueError, match="failure_rate_threshold"):
            CircuitBreakerConfig(failure_rate_threshold=1.5)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RateLimitConfig()
        assert config.requests_per_second == 10.0
        assert config.burst_size == 20
        assert config.wait_for_token is False
        assert config.max_wait_seconds == 5.0

    def test_custom_values(self):
        """Test custom values."""
        config = RateLimitConfig(
            requests_per_second=100.0,
            burst_size=50,
            wait_for_token=True,
        )
        assert config.requests_per_second == 100.0
        assert config.burst_size == 50
        assert config.wait_for_token is True

    def test_validation_requests_per_second(self):
        """Test requests_per_second validation."""
        with pytest.raises(ValueError, match="requests_per_second"):
            RateLimitConfig(requests_per_second=0)

    def test_validation_burst_size(self):
        """Test burst_size validation."""
        with pytest.raises(ValueError, match="burst_size"):
            RateLimitConfig(burst_size=0)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.jitter_factor == 0.25
        assert TimeoutError in config.retryable_exceptions
        assert 500 in config.retryable_status_codes

    def test_custom_values(self):
        """Test custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay_seconds=0.5,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.base_delay_seconds == 0.5
        assert config.jitter is False

    def test_validation_max_delay(self):
        """Test max_delay_seconds validation."""
        with pytest.raises(ValueError, match="max_delay_seconds"):
            RetryConfig(base_delay_seconds=10, max_delay_seconds=5)

    def test_validation_exponential_base(self):
        """Test exponential_base validation."""
        with pytest.raises(ValueError, match="exponential_base"):
            RetryConfig(exponential_base=0.5)

    def test_validation_jitter_factor(self):
        """Test jitter_factor validation."""
        with pytest.raises(ValueError, match="jitter_factor"):
            RetryConfig(jitter_factor=1.5)


class TestDegradationConfig:
    """Tests for DegradationConfig."""

    def test_defaults(self):
        """Test default values."""
        config = DegradationConfig()
        assert config.enable_fallback is True
        assert config.fallback_timeout_seconds == 5.0
        assert config.fallback_on_circuit_open is True
        assert config.fallback_on_rate_limit is False
        assert config.fallback_on_timeout is True

    def test_validation_timeout(self):
        """Test fallback_timeout_seconds validation."""
        with pytest.raises(ValueError, match="fallback_timeout_seconds"):
            DegradationConfig(fallback_timeout_seconds=0)


class TestHealthCheckConfig:
    """Tests for HealthCheckConfig."""

    def test_defaults(self):
        """Test default values."""
        config = HealthCheckConfig()
        assert config.enabled is True
        assert config.interval_seconds == 30.0
        assert config.timeout_seconds == 5.0
        assert config.healthy_threshold == 2
        assert config.unhealthy_threshold == 3

    def test_validation_interval(self):
        """Test interval_seconds validation."""
        with pytest.raises(ValueError, match="interval_seconds"):
            HealthCheckConfig(interval_seconds=0)

    def test_validation_healthy_threshold(self):
        """Test healthy_threshold validation."""
        with pytest.raises(ValueError, match="healthy_threshold"):
            HealthCheckConfig(healthy_threshold=0)


class TestResilienceConfig:
    """Tests for ResilienceConfig."""

    def test_empty_config(self):
        """Test empty config (all None)."""
        config = ResilienceConfig()
        assert config.circuit_breaker is None
        assert config.rate_limit is None
        assert config.retry is None
        assert config.degradation is None
        assert config.health_check is None

    def test_default_factory(self):
        """Test default factory method."""
        config = ResilienceConfig.default()
        assert config.circuit_breaker is not None
        assert config.rate_limit is not None
        assert config.retry is not None
        assert config.degradation is not None
        assert config.health_check is not None

    def test_minimal_factory(self):
        """Test minimal factory method."""
        config = ResilienceConfig.minimal()
        assert config.circuit_breaker is None
        assert config.rate_limit is None
        assert config.retry is not None
        assert config.degradation is None
        assert config.health_check is None

    def test_strict_factory(self):
        """Test strict factory method."""
        config = ResilienceConfig.strict()
        assert config.circuit_breaker.failure_threshold == 3
        assert config.rate_limit.requests_per_second == 5.0
        assert config.retry.max_retries == 5
        assert config.health_check.interval_seconds == 15.0

    def test_partial_config(self):
        """Test partial configuration."""
        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(),
            retry=RetryConfig(max_retries=5),
        )
        assert config.circuit_breaker is not None
        assert config.rate_limit is None
        assert config.retry.max_retries == 5


class TestResilienceEvent:
    """Tests for ResilienceEvent."""

    def test_event_creation(self):
        """Test event creation with defaults."""
        event = ResilienceEvent(
            event_type=ResilienceEventType.CIRCUIT_OPENED,
            backend_name="test_backend",
        )
        assert event.event_type == ResilienceEventType.CIRCUIT_OPENED
        assert event.backend_name == "test_backend"
        assert isinstance(event.timestamp, datetime)
        assert event.metadata == {}

    def test_event_with_metadata(self):
        """Test event with metadata."""
        event = ResilienceEvent(
            event_type=ResilienceEventType.RETRY_ATTEMPT,
            backend_name="test_backend",
            metadata={"attempt": 3, "error": "connection refused"},
        )
        assert event.metadata["attempt"] == 3
        assert event.metadata["error"] == "connection refused"

    def test_to_dict(self):
        """Test event to_dict method."""
        event = ResilienceEvent(
            event_type=ResilienceEventType.RATE_LIMITED,
            backend_name="api_backend",
            metadata={"wait_time": 1.5},
        )
        d = event.to_dict()
        assert d["event_type"] == "rate_limited"
        assert d["backend_name"] == "api_backend"
        assert "timestamp" in d
        assert d["metadata"]["wait_time"] == 1.5

    def test_event_timestamp_utc(self):
        """Test event timestamp is UTC."""
        event = ResilienceEvent(
            event_type=ResilienceEventType.CIRCUIT_CLOSED,
            backend_name="test",
        )
        assert event.timestamp.tzinfo == timezone.utc
