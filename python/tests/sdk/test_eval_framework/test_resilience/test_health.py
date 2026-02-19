"""Tests for health check implementation."""

import time
import threading

import pytest

from fi.evals.framework.resilience.health import (
    HealthChecker,
    HealthCheckResult,
    HealthRegistry,
    HealthStats,
)
from fi.evals.framework.resilience.types import (
    HealthCheckConfig,
    HealthStatus,
    ResilienceEventType,
)


class TestHealthCheckerBasic:
    """Basic functionality tests."""

    def test_initial_state(self):
        """Health checker starts in unknown state."""
        checker = HealthChecker("test", lambda: True)
        assert checker.status == HealthStatus.UNKNOWN
        assert not checker.is_healthy
        assert not checker.is_running

    def test_check_now_success(self):
        """Immediate health check success."""
        checker = HealthChecker("test", lambda: True)
        result = checker.check_now()

        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Health check passed"
        assert result.response_time_ms is not None
        assert checker.stats.successful_checks == 1

    def test_check_now_failure(self):
        """Immediate health check failure."""
        checker = HealthChecker("test", lambda: False)
        result = checker.check_now()

        assert result.status == HealthStatus.UNHEALTHY
        assert "returned false" in result.message
        assert checker.stats.failed_checks == 1

    def test_check_now_exception(self):
        """Health check exception is recorded as failure."""

        def raise_error():
            raise ConnectionError("connection refused")

        checker = HealthChecker("test", raise_error)
        result = checker.check_now()

        assert result.status == HealthStatus.UNHEALTHY
        assert "connection refused" in result.message
        assert result.metadata["error_type"] == "ConnectionError"
        assert checker.stats.failed_checks == 1


class TestHealthCheckerStatusTransitions:
    """Tests for status transitions."""

    def test_becomes_healthy_after_threshold(self):
        """Status becomes healthy after consecutive successes."""
        config = HealthCheckConfig(healthy_threshold=2, unhealthy_threshold=3)
        checker = HealthChecker("test", lambda: True, config)

        # First success - still unknown
        checker.check_now()
        assert checker.status == HealthStatus.UNKNOWN

        # Second success - now healthy
        checker.check_now()
        assert checker.status == HealthStatus.HEALTHY
        assert checker.is_healthy

    def test_becomes_unhealthy_after_threshold(self):
        """Status becomes unhealthy after consecutive failures."""
        config = HealthCheckConfig(healthy_threshold=2, unhealthy_threshold=2)
        checker = HealthChecker("test", lambda: False, config)

        # First failure - still unknown
        checker.check_now()
        assert checker.status == HealthStatus.UNKNOWN

        # Second failure - now unhealthy
        checker.check_now()
        assert checker.status == HealthStatus.UNHEALTHY

    def test_becomes_degraded_on_failure_after_healthy(self):
        """Healthy to degraded on first failure."""
        config = HealthCheckConfig(healthy_threshold=1, unhealthy_threshold=3)
        attempts = [True, True, False]
        idx = [0]

        def flaky():
            result = attempts[idx[0]]
            idx[0] += 1
            return result

        checker = HealthChecker("test", flaky, config)

        # Get to healthy
        checker.check_now()
        assert checker.status == HealthStatus.HEALTHY

        # First failure - degraded
        checker.check_now()
        checker.check_now()
        assert checker.status == HealthStatus.DEGRADED

    def test_recovery_from_unhealthy(self):
        """Can recover from unhealthy to healthy."""
        config = HealthCheckConfig(healthy_threshold=2, unhealthy_threshold=2)
        healthy = [False, False, True, True]
        idx = [0]

        def check():
            result = healthy[idx[0]]
            idx[0] += 1
            return result

        checker = HealthChecker("test", check, config)

        # Get to unhealthy
        checker.check_now()
        checker.check_now()
        assert checker.status == HealthStatus.UNHEALTHY

        # Recover
        checker.check_now()
        checker.check_now()
        assert checker.status == HealthStatus.HEALTHY

    def test_consecutive_counters_reset(self):
        """Counters reset on status flip."""
        config = HealthCheckConfig(healthy_threshold=2, unhealthy_threshold=2)
        healthy = [True, False, True, True]
        idx = [0]

        def check():
            result = healthy[idx[0]]
            idx[0] += 1
            return result

        checker = HealthChecker("test", check, config)

        # Success then failure resets consecutive_successes
        checker.check_now()  # success: consecutive_successes=1
        checker.check_now()  # failure: consecutive_successes=0, consecutive_failures=1

        assert checker.stats.consecutive_successes == 0
        assert checker.stats.consecutive_failures == 1


class TestHealthCheckerPeriodicChecks:
    """Tests for periodic health checking."""

    def test_start_stop(self):
        """Can start and stop periodic checks."""
        config = HealthCheckConfig(interval_seconds=0.1)
        checker = HealthChecker("test", lambda: True, config)

        checker.start()
        assert checker.is_running

        time.sleep(0.25)  # Let a few checks run

        checker.stop()
        assert not checker.is_running
        assert checker.stats.total_checks >= 2

    def test_periodic_checks_run(self):
        """Periodic checks actually run at interval."""
        config = HealthCheckConfig(interval_seconds=0.05, healthy_threshold=1)
        check_count = [0]

        def counting_check():
            check_count[0] += 1
            return True

        checker = HealthChecker("test", counting_check, config)

        checker.start()
        time.sleep(0.2)  # Should get ~4 checks
        checker.stop()

        assert check_count[0] >= 3

    def test_disabled_checker_does_not_start(self):
        """Disabled checker doesn't start."""
        config = HealthCheckConfig(enabled=False)
        checker = HealthChecker("test", lambda: True, config)

        checker.start()
        assert not checker.is_running

    def test_double_start_is_noop(self):
        """Starting already running checker is no-op."""
        config = HealthCheckConfig(interval_seconds=1)
        checker = HealthChecker("test", lambda: True, config)

        checker.start()
        first_thread = checker._thread

        checker.start()  # Should not create new thread
        assert checker._thread is first_thread

        checker.stop()


class TestHealthCheckerEvents:
    """Tests for event callbacks."""

    def test_event_on_healthy(self):
        """Event emitted on transition to healthy."""
        config = HealthCheckConfig(healthy_threshold=1)
        events = []
        checker = HealthChecker(
            "test", lambda: True, config, event_callback=lambda e: events.append(e)
        )

        checker.check_now()

        assert len(events) == 1
        assert events[0].event_type == ResilienceEventType.HEALTH_CHECK_PASSED
        assert events[0].backend_name == "test"
        assert events[0].metadata["new_status"] == "healthy"

    def test_event_on_unhealthy(self):
        """Event emitted on transition to unhealthy."""
        config = HealthCheckConfig(unhealthy_threshold=1)
        events = []
        checker = HealthChecker(
            "test", lambda: False, config, event_callback=lambda e: events.append(e)
        )

        checker.check_now()

        assert len(events) == 1
        assert events[0].event_type == ResilienceEventType.HEALTH_CHECK_FAILED

    def test_no_event_on_same_status(self):
        """No event when status doesn't change."""
        config = HealthCheckConfig(healthy_threshold=1)
        events = []
        checker = HealthChecker(
            "test", lambda: True, config, event_callback=lambda e: events.append(e)
        )

        checker.check_now()  # Transition to healthy
        checker.check_now()  # Still healthy

        assert len(events) == 1  # Only one event

    def test_callback_exception_handled(self):
        """Callback exceptions don't break checker."""
        config = HealthCheckConfig(healthy_threshold=1)

        def bad_callback(e):
            raise RuntimeError("callback error")

        checker = HealthChecker("test", lambda: True, config, event_callback=bad_callback)

        # Should not raise
        checker.check_now()
        assert checker.status == HealthStatus.HEALTHY


class TestHealthCheckerStats:
    """Tests for statistics."""

    def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        config = HealthCheckConfig(healthy_threshold=1, unhealthy_threshold=1)
        results = [True, True, False]
        idx = [0]

        def check():
            result = results[idx[0]]
            idx[0] += 1
            return result

        checker = HealthChecker("test", check, config)

        checker.check_now()
        checker.check_now()
        checker.check_now()

        assert checker.stats.total_checks == 3
        assert checker.stats.successful_checks == 2
        assert checker.stats.failed_checks == 1

    def test_get_stats(self):
        """Get stats returns correct values."""
        config = HealthCheckConfig(healthy_threshold=1)
        checker = HealthChecker("test", lambda: True, config)

        checker.check_now()

        stats = checker.get_stats()
        assert stats["name"] == "test"
        assert stats["status"] == "healthy"
        assert stats["total_checks"] == 1
        assert stats["success_rate"] == 1.0
        assert "avg_response_time_ms" in stats

    def test_response_time_tracked(self):
        """Response time is tracked."""
        config = HealthCheckConfig(healthy_threshold=1)

        def slow_check():
            time.sleep(0.02)
            return True

        checker = HealthChecker("test", slow_check, config)
        checker.check_now()

        assert checker.stats.avg_response_time_ms >= 15  # At least 15ms

    def test_get_recent_results(self):
        """Can get recent check results."""
        checker = HealthChecker("test", lambda: True)

        checker.check_now()
        checker.check_now()
        checker.check_now()

        results = checker.get_recent_results(2)
        assert len(results) == 2


class TestHealthCheckerReset:
    """Tests for reset functionality."""

    def test_reset(self):
        """Reset clears state."""
        config = HealthCheckConfig(healthy_threshold=1)
        checker = HealthChecker("test", lambda: True, config)

        checker.check_now()
        assert checker.status == HealthStatus.HEALTHY
        assert checker.stats.total_checks == 1

        checker.reset()

        assert checker.status == HealthStatus.UNKNOWN
        assert checker.stats.total_checks == 0


class TestHealthRegistry:
    """Tests for health registry."""

    def test_register_checker(self):
        """Can register health checkers."""
        registry = HealthRegistry()

        checker = registry.register("api", lambda: True)

        assert checker is not None
        assert "api" in registry
        assert len(registry) == 1

    def test_duplicate_register_raises(self):
        """Registering duplicate name raises."""
        registry = HealthRegistry()
        registry.register("api", lambda: True)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("api", lambda: False)

    def test_unregister(self):
        """Can unregister health checkers."""
        registry = HealthRegistry()
        registry.register("api", lambda: True)

        registry.unregister("api")

        assert "api" not in registry
        assert len(registry) == 0

    def test_get_checker(self):
        """Can get checker by name."""
        registry = HealthRegistry()
        original = registry.register("api", lambda: True)

        retrieved = registry.get("api")

        assert retrieved is original

    def test_get_nonexistent(self):
        """Get returns None for nonexistent."""
        registry = HealthRegistry()
        assert registry.get("nonexistent") is None


class TestHealthRegistryOperations:
    """Tests for registry operations."""

    def test_start_stop_all(self):
        """Can start and stop all checkers."""
        config = HealthCheckConfig(interval_seconds=0.1)
        registry = HealthRegistry(default_config=config)
        registry.register("api", lambda: True)
        registry.register("db", lambda: True)

        registry.start_all()

        # All should be running
        assert registry.get("api").is_running
        assert registry.get("db").is_running

        registry.stop_all()

        # All should be stopped
        assert not registry.get("api").is_running
        assert not registry.get("db").is_running

    def test_check_all_now(self):
        """Can check all immediately."""
        registry = HealthRegistry()
        registry.register("api", lambda: True)
        registry.register("db", lambda: False)

        results = registry.check_all_now()

        assert results["api"].status == HealthStatus.HEALTHY
        assert results["db"].status == HealthStatus.UNHEALTHY

    def test_get_all_stats(self):
        """Can get stats for all checkers."""
        config = HealthCheckConfig(healthy_threshold=1)
        registry = HealthRegistry(default_config=config)
        registry.register("api", lambda: True)
        registry.register("db", lambda: True)

        registry.check_all_now()
        stats = registry.get_all_stats()

        assert "api" in stats
        assert "db" in stats
        assert stats["api"]["status"] == "healthy"


class TestHealthRegistryOverallStatus:
    """Tests for overall status calculation."""

    def test_overall_unknown_when_empty(self):
        """Overall is unknown when no checkers."""
        registry = HealthRegistry()
        assert registry.get_overall_status() == HealthStatus.UNKNOWN

    def test_overall_unknown_when_all_unknown(self):
        """Overall is unknown when all checkers unknown."""
        registry = HealthRegistry()
        registry.register("api", lambda: True)
        registry.register("db", lambda: True)

        # No checks performed yet
        assert registry.get_overall_status() == HealthStatus.UNKNOWN

    def test_overall_healthy_when_all_healthy(self):
        """Overall is healthy when all healthy."""
        config = HealthCheckConfig(healthy_threshold=1)
        registry = HealthRegistry(default_config=config)
        registry.register("api", lambda: True)
        registry.register("db", lambda: True)

        registry.check_all_now()

        assert registry.get_overall_status() == HealthStatus.HEALTHY

    def test_overall_unhealthy_when_any_unhealthy(self):
        """Overall is unhealthy when any unhealthy."""
        config = HealthCheckConfig(healthy_threshold=1, unhealthy_threshold=1)
        registry = HealthRegistry(default_config=config)
        registry.register("api", lambda: True)
        registry.register("db", lambda: False)

        registry.check_all_now()

        assert registry.get_overall_status() == HealthStatus.UNHEALTHY

    def test_overall_degraded_when_any_degraded(self):
        """Overall is degraded when any degraded but none unhealthy."""
        config = HealthCheckConfig(healthy_threshold=1, unhealthy_threshold=3)
        registry = HealthRegistry(default_config=config)

        # Set up one healthy, one that will become degraded
        registry.register("api", lambda: True)

        results = [True, False]  # healthy then fail
        idx = [0]

        def flaky():
            result = results[idx[0]]
            idx[0] = min(idx[0] + 1, len(results) - 1)
            return result

        registry.register("db", flaky)

        # First check - both succeed
        registry.check_all_now()
        assert registry.get_overall_status() == HealthStatus.HEALTHY

        # Second check - db fails (becomes degraded)
        registry.check_all_now()
        assert registry.get_overall_status() == HealthStatus.DEGRADED


class TestHealthRegistrySummary:
    """Tests for registry summary."""

    def test_get_summary(self):
        """Get summary returns correct values."""
        config = HealthCheckConfig(healthy_threshold=1, unhealthy_threshold=1)
        registry = HealthRegistry(default_config=config)
        registry.register("api", lambda: True)
        registry.register("db", lambda: False)

        registry.check_all_now()
        summary = registry.get_summary()

        assert summary["checker_count"] == 2
        assert summary["healthy_count"] == 1
        assert summary["unhealthy_count"] == 1
        assert summary["overall_status"] == "unhealthy"
        assert "api" in summary["checkers"]
        assert "db" in summary["checkers"]

    def test_names_property(self):
        """Names property returns all names."""
        registry = HealthRegistry()
        registry.register("api", lambda: True)
        registry.register("db", lambda: True)

        assert set(registry.names) == {"api", "db"}


class TestHealthRegistryEvents:
    """Tests for registry event handling."""

    def test_events_from_all_checkers(self):
        """Events from all checkers go to registry callback."""
        config = HealthCheckConfig(healthy_threshold=1)
        events = []
        registry = HealthRegistry(
            default_config=config, event_callback=lambda e: events.append(e)
        )

        registry.register("api", lambda: True)
        registry.register("db", lambda: True)

        registry.check_all_now()

        # Should have events from both checkers
        backends = [e.backend_name for e in events]
        assert "api" in backends
        assert "db" in backends


class TestHealthCheckerEdgeCases:
    """Edge case tests."""

    def test_very_fast_checks(self):
        """Handles very fast check functions."""
        checker = HealthChecker("test", lambda: True)
        result = checker.check_now()

        assert result.response_time_ms >= 0
        assert result.response_time_ms < 100  # Should be very fast

    def test_timeout_not_enforced(self):
        """Note: timeout is in config but not enforced by checker itself."""
        # This test documents current behavior - timeout would need
        # to be implemented with threading if needed
        config = HealthCheckConfig(timeout_seconds=0.01)

        def slow_check():
            time.sleep(0.05)
            return True

        checker = HealthChecker("test", slow_check, config)
        result = checker.check_now()

        # Check still completes (timeout not enforced)
        assert result.status == HealthStatus.HEALTHY

    def test_thread_safety(self):
        """Health checker is thread-safe."""
        config = HealthCheckConfig(healthy_threshold=1)
        checker = HealthChecker("test", lambda: True, config)
        errors = []

        def worker():
            try:
                for _ in range(10):
                    checker.check_now()
                    _ = checker.status
                    _ = checker.stats
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert checker.stats.total_checks == 50
