"""
Circuit breaker implementation for fault tolerance.

Prevents cascading failures by stopping requests to failing backends
and allowing them to recover.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, TypeVar

from .types import (
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    EventCallback,
    ResilienceEvent,
    ResilienceEventType,
)

T = TypeVar("T")


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0


class CircuitBreaker:
    """
    Thread-safe circuit breaker implementation.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests rejected
    - HALF_OPEN: Testing recovery, limited requests allowed

    Example:
        config = CircuitBreakerConfig(failure_threshold=5, timeout_seconds=30)
        cb = CircuitBreaker("my_backend", config)

        try:
            result = cb.execute(lambda: backend.call())
        except CircuitOpenError:
            # Handle circuit open
            pass
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
        event_callback: Optional[EventCallback] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name for this circuit breaker (typically backend name)
            config: Configuration options
            on_state_change: Callback for state transitions
            event_callback: Callback for resilience events
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        self.event_callback = event_callback

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.RLock()

        # Sliding window for failure rate calculation
        self._results: Deque[bool] = deque(maxlen=self.config.window_size)

        # Statistics
        self.stats = CircuitStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state (checks for automatic transitions)."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    def execute(self, func: Callable[[], T]) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.OPEN:
                time_until_retry = self._time_until_retry()
                self.stats.rejected_requests += 1
                raise CircuitOpenError(self.name, time_until_retry)

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    raise CircuitOpenError(self.name, 0)
                self._half_open_requests += 1

            self.stats.total_requests += 1

        # Execute outside lock to avoid blocking other threads
        try:
            result = func()
            self._record_success()
            return result
        except Exception as e:
            if type(e) not in self.config.excluded_exceptions:
                self._record_failure()
            raise

    def _record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._results.append(True)
            self._success_count += 1
            self.stats.successful_requests += 1
            self.stats.last_success_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def _record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._results.append(False)
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self.stats.failed_requests += 1
            self.stats.last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open trips back to open
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._should_trip():
                    self._transition_to(CircuitState.OPEN)

    def _should_trip(self) -> bool:
        """Check if circuit should trip to OPEN."""
        # Check absolute failure count
        if self._failure_count >= self.config.failure_threshold:
            return True

        # Check failure rate if window is full
        if len(self._results) >= self.config.window_size:
            failures = sum(1 for r in self._results if not r)
            failure_rate = failures / len(self._results)
            if failure_rate >= self.config.failure_rate_threshold:
                return True

        return False

    def _check_state_transition(self) -> None:
        """Check for automatic state transitions (OPEN -> HALF_OPEN)."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self.stats.state_changes += 1

        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._results.clear()
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_requests = 0
        elif new_state == CircuitState.OPEN:
            self._last_failure_time = time.monotonic()

        # Callbacks
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception:
                pass

        self._emit_event(old_state, new_state)

    def _time_until_retry(self) -> float:
        """Get time until circuit will transition to half-open."""
        if self._last_failure_time is None:
            return 0
        elapsed = time.monotonic() - self._last_failure_time
        return max(0, self.config.timeout_seconds - elapsed)

    def _emit_event(self, old_state: CircuitState, new_state: CircuitState) -> None:
        """Emit state change event."""
        if not self.event_callback:
            return

        event_type_map = {
            CircuitState.OPEN: ResilienceEventType.CIRCUIT_OPENED,
            CircuitState.CLOSED: ResilienceEventType.CIRCUIT_CLOSED,
            CircuitState.HALF_OPEN: ResilienceEventType.CIRCUIT_HALF_OPEN,
        }

        event = ResilienceEvent(
            event_type=event_type_map[new_state],
            backend_name=self.name,
            metadata={
                "old_state": old_state.value,
                "new_state": new_state.value,
                "failure_count": self._failure_count,
            },
        )

        try:
            self.event_callback(event)
        except Exception:
            pass

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_requests = 0
            self._results.clear()
            self._last_failure_time = None
            self.stats.state_changes += 1

            if old_state != CircuitState.CLOSED:
                self._emit_event(old_state, CircuitState.CLOSED)
                if self.on_state_change:
                    try:
                        self.on_state_change(old_state, CircuitState.CLOSED)
                    except Exception:
                        pass

    def force_open(self) -> None:
        """Force circuit to open state (for testing/maintenance)."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                self._transition_to(CircuitState.OPEN)

    def get_failure_rate(self) -> float:
        """Get current failure rate from sliding window."""
        with self._lock:
            if not self._results:
                return 0.0
            failures = sum(1 for r in self._results if not r)
            return failures / len(self._results)

    def get_stats(self) -> dict:
        """Get circuit breaker statistics as dictionary."""
        with self._lock:
            return {
                "state": self._state.value,
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "rejected_requests": self.stats.rejected_requests,
                "state_changes": self.stats.state_changes,
                "failure_rate": self.get_failure_rate(),
                "failure_count": self._failure_count,
            }
