"""
Health check infrastructure for backend monitoring.

Provides periodic health checking with status transitions based on
consecutive pass/fail thresholds.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from .types import (
    EventCallback,
    HealthCheckConfig,
    HealthStatus,
    ResilienceEvent,
    ResilienceEventType,
)

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    status: HealthStatus
    message: Optional[str] = None
    response_time_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict = field(default_factory=dict)


@dataclass
class HealthStats:
    """Statistics for health checker."""

    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_check_time: Optional[datetime] = None
    avg_response_time_ms: float = 0.0


# Type for health check function
HealthCheckFunc = Callable[[], bool]


class HealthChecker:
    """
    Health checker for a single backend.

    Performs periodic health checks and tracks status based on
    consecutive successes/failures.

    Example:
        checker = HealthChecker(
            "api",
            lambda: api.ping(),
            config=HealthCheckConfig(interval_seconds=30)
        )
        checker.start()

        # Later
        if checker.status == HealthStatus.HEALTHY:
            # Proceed
            pass

        checker.stop()
    """

    def __init__(
        self,
        name: str,
        check_func: HealthCheckFunc,
        config: Optional[HealthCheckConfig] = None,
        event_callback: Optional[EventCallback] = None,
    ):
        """
        Initialize health checker.

        Args:
            name: Name for this health checker
            check_func: Function that returns True if healthy
            config: Health check configuration
            event_callback: Callback for health events
        """
        self.name = name
        self._check_func = check_func
        self.config = config or HealthCheckConfig()
        self.event_callback = event_callback

        self._status = HealthStatus.UNKNOWN
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self.stats = HealthStats()
        self._results: List[HealthCheckResult] = []
        self._max_results = 100  # Keep last 100 results

    @property
    def status(self) -> HealthStatus:
        """Get current health status."""
        with self._lock:
            return self._status

    @property
    def is_healthy(self) -> bool:
        """Check if currently healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_running(self) -> bool:
        """Check if health checker is running."""
        return self._running

    def start(self) -> None:
        """Start periodic health checking."""
        if not self.config.enabled:
            return

        with self._lock:
            if self._running:
                return

            self._running = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._check_loop,
                name=f"health-{self.name}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop periodic health checking."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def check_now(self) -> HealthCheckResult:
        """
        Perform an immediate health check.

        Returns:
            HealthCheckResult with check outcome
        """
        start_time = time.monotonic()
        result: HealthCheckResult

        try:
            is_healthy = self._check_func()
            elapsed_ms = (time.monotonic() - start_time) * 1000

            if is_healthy:
                result = HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Health check passed",
                    response_time_ms=elapsed_ms,
                )
                self._record_success(result)
            else:
                result = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Health check returned false",
                    response_time_ms=elapsed_ms,
                )
                self._record_failure(result)

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=elapsed_ms,
                metadata={"error": str(e), "error_type": type(e).__name__},
            )
            self._record_failure(result)

        return result

    def _check_loop(self) -> None:
        """Background loop for periodic checks."""
        while not self._stop_event.is_set():
            try:
                self.check_now()
            except Exception as e:
                logger.error(f"Health check error for {self.name}: {e}")

            # Wait for next check or stop signal
            self._stop_event.wait(self.config.interval_seconds)

    def _record_success(self, result: HealthCheckResult) -> None:
        """Record a successful health check."""
        with self._lock:
            self.stats.total_checks += 1
            self.stats.successful_checks += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0
            self.stats.last_check_time = result.timestamp

            # Update average response time
            if result.response_time_ms is not None:
                self._update_avg_response_time(result.response_time_ms)

            # Store result
            self._results.append(result)
            if len(self._results) > self._max_results:
                self._results.pop(0)

            # Check for status transition
            old_status = self._status
            if self.stats.consecutive_successes >= self.config.healthy_threshold:
                self._status = HealthStatus.HEALTHY

            if old_status != self._status:
                self._emit_status_change(old_status, self._status)

    def _record_failure(self, result: HealthCheckResult) -> None:
        """Record a failed health check."""
        with self._lock:
            self.stats.total_checks += 1
            self.stats.failed_checks += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_check_time = result.timestamp

            # Update average response time
            if result.response_time_ms is not None:
                self._update_avg_response_time(result.response_time_ms)

            # Store result
            self._results.append(result)
            if len(self._results) > self._max_results:
                self._results.pop(0)

            # Check for status transition
            old_status = self._status
            if self.stats.consecutive_failures >= self.config.unhealthy_threshold:
                self._status = HealthStatus.UNHEALTHY
            elif self._status == HealthStatus.HEALTHY:
                self._status = HealthStatus.DEGRADED

            if old_status != self._status:
                self._emit_status_change(old_status, self._status)

    def _update_avg_response_time(self, response_time_ms: float) -> None:
        """Update running average of response time."""
        if self.stats.total_checks == 1:
            self.stats.avg_response_time_ms = response_time_ms
        else:
            # Exponential moving average
            alpha = 0.2
            self.stats.avg_response_time_ms = (
                alpha * response_time_ms
                + (1 - alpha) * self.stats.avg_response_time_ms
            )

    def _emit_status_change(
        self, old_status: HealthStatus, new_status: HealthStatus
    ) -> None:
        """Emit health status change event."""
        if not self.event_callback:
            return

        event_type = (
            ResilienceEventType.HEALTH_CHECK_PASSED
            if new_status == HealthStatus.HEALTHY
            else ResilienceEventType.HEALTH_CHECK_FAILED
        )

        event = ResilienceEvent(
            event_type=event_type,
            backend_name=self.name,
            metadata={
                "old_status": old_status.value,
                "new_status": new_status.value,
                "consecutive_successes": self.stats.consecutive_successes,
                "consecutive_failures": self.stats.consecutive_failures,
            },
        )

        try:
            self.event_callback(event)
        except Exception:
            pass

    def reset(self) -> None:
        """Reset health checker state."""
        with self._lock:
            self._status = HealthStatus.UNKNOWN
            self.stats = HealthStats()
            self._results.clear()

    def get_stats(self) -> dict:
        """Get health checker statistics as dictionary."""
        with self._lock:
            return {
                "name": self.name,
                "status": self._status.value,
                "total_checks": self.stats.total_checks,
                "successful_checks": self.stats.successful_checks,
                "failed_checks": self.stats.failed_checks,
                "consecutive_successes": self.stats.consecutive_successes,
                "consecutive_failures": self.stats.consecutive_failures,
                "avg_response_time_ms": self.stats.avg_response_time_ms,
                "last_check_time": (
                    self.stats.last_check_time.isoformat()
                    if self.stats.last_check_time
                    else None
                ),
                "success_rate": (
                    self.stats.successful_checks / self.stats.total_checks
                    if self.stats.total_checks > 0
                    else 0.0
                ),
            }

    def get_recent_results(self, count: int = 10) -> List[HealthCheckResult]:
        """Get recent health check results."""
        with self._lock:
            return list(self._results[-count:])


class HealthRegistry:
    """
    Registry for managing multiple health checkers.

    Example:
        registry = HealthRegistry()
        registry.register("api", lambda: api.ping())
        registry.register("db", lambda: db.is_connected())

        registry.start_all()

        # Get overall health
        overall = registry.get_overall_status()

        registry.stop_all()
    """

    def __init__(
        self,
        default_config: Optional[HealthCheckConfig] = None,
        event_callback: Optional[EventCallback] = None,
    ):
        """
        Initialize health registry.

        Args:
            default_config: Default config for registered checkers
            event_callback: Callback for all health events
        """
        self.default_config = default_config or HealthCheckConfig()
        self.event_callback = event_callback

        self._checkers: Dict[str, HealthChecker] = {}
        self._lock = threading.RLock()  # RLock for reentrant locking

    def register(
        self,
        name: str,
        check_func: HealthCheckFunc,
        config: Optional[HealthCheckConfig] = None,
    ) -> HealthChecker:
        """
        Register a new health checker.

        Args:
            name: Name for the health checker
            check_func: Function that returns True if healthy
            config: Optional config override

        Returns:
            The registered HealthChecker
        """
        with self._lock:
            if name in self._checkers:
                raise ValueError(f"Health checker '{name}' already registered")

            checker = HealthChecker(
                name=name,
                check_func=check_func,
                config=config or self.default_config,
                event_callback=self.event_callback,
            )
            self._checkers[name] = checker
            return checker

    def unregister(self, name: str) -> None:
        """Unregister a health checker."""
        with self._lock:
            if name in self._checkers:
                self._checkers[name].stop()
                del self._checkers[name]

    def get(self, name: str) -> Optional[HealthChecker]:
        """Get a health checker by name."""
        with self._lock:
            return self._checkers.get(name)

    def start_all(self) -> None:
        """Start all registered health checkers."""
        with self._lock:
            for checker in self._checkers.values():
                checker.start()

    def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all registered health checkers."""
        with self._lock:
            for checker in self._checkers.values():
                checker.stop(timeout)

    def check_all_now(self) -> Dict[str, HealthCheckResult]:
        """Perform immediate health check on all registered checkers."""
        results = {}
        with self._lock:
            for name, checker in self._checkers.items():
                results[name] = checker.check_now()
        return results

    def get_overall_status(self) -> HealthStatus:
        """
        Get overall health status.

        Returns:
            HEALTHY if all healthy
            DEGRADED if any degraded but none unhealthy
            UNHEALTHY if any unhealthy
            UNKNOWN if no checkers or all unknown
        """
        with self._lock:
            if not self._checkers:
                return HealthStatus.UNKNOWN

            statuses = [c.status for c in self._checkers.values()]

            if all(s == HealthStatus.UNKNOWN for s in statuses):
                return HealthStatus.UNKNOWN

            if any(s == HealthStatus.UNHEALTHY for s in statuses):
                return HealthStatus.UNHEALTHY

            if any(s == HealthStatus.DEGRADED for s in statuses):
                return HealthStatus.DEGRADED

            if all(s == HealthStatus.HEALTHY for s in statuses):
                return HealthStatus.HEALTHY

            return HealthStatus.DEGRADED

    def get_all_stats(self) -> Dict[str, dict]:
        """Get statistics for all health checkers."""
        with self._lock:
            return {name: checker.get_stats() for name, checker in self._checkers.items()}

    def get_summary(self) -> dict:
        """Get summary of all health checkers."""
        with self._lock:
            return {
                "overall_status": self.get_overall_status().value,
                "checker_count": len(self._checkers),
                "healthy_count": sum(
                    1 for c in self._checkers.values() if c.status == HealthStatus.HEALTHY
                ),
                "unhealthy_count": sum(
                    1 for c in self._checkers.values() if c.status == HealthStatus.UNHEALTHY
                ),
                "degraded_count": sum(
                    1 for c in self._checkers.values() if c.status == HealthStatus.DEGRADED
                ),
                "checkers": {
                    name: checker.status.value for name, checker in self._checkers.items()
                },
            }

    @property
    def names(self) -> List[str]:
        """Get names of all registered health checkers."""
        with self._lock:
            return list(self._checkers.keys())

    def __len__(self) -> int:
        """Get number of registered health checkers."""
        with self._lock:
            return len(self._checkers)

    def __contains__(self, name: str) -> bool:
        """Check if a health checker is registered."""
        with self._lock:
            return name in self._checkers
