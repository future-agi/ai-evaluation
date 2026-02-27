"""
Retry handler with exponential backoff and jitter.

Handles transient failures by retrying operations with configurable delays.
"""

import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, TypeVar

from .types import (
    EventCallback,
    ResilienceEvent,
    ResilienceEventType,
    RetryConfig,
    RetryExhaustedError,
)

T = TypeVar("T")


@dataclass
class RetryStats:
    """Statistics for retry handler."""

    total_calls: int = 0
    successful_first_attempt: int = 0
    successful_after_retry: int = 0
    failed_all_retries: int = 0
    total_retries: int = 0
    total_delay_ms: float = 0


class RetryHandler:
    """
    Retry handler with exponential backoff and jitter.

    Implements exponential backoff with optional jitter to prevent
    thundering herd problems in distributed systems.

    Delay formula: min(base_delay * (exponential_base ^ attempt), max_delay) + jitter

    Example:
        config = RetryConfig(max_retries=3, base_delay_seconds=1.0)
        retry = RetryHandler("my_backend", config)

        result = retry.execute(lambda: backend.call())
    """

    def __init__(
        self,
        name: str,
        config: Optional[RetryConfig] = None,
        event_callback: Optional[EventCallback] = None,
    ):
        """
        Initialize retry handler.

        Args:
            name: Name for this retry handler (typically backend name)
            config: Retry configuration
            event_callback: Callback for retry events
        """
        self.name = name
        self.config = config or RetryConfig()
        self.event_callback = event_callback

        self.stats = RetryStats()

    def execute(
        self,
        func: Callable[[], T],
        config_override: Optional[RetryConfig] = None,
    ) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            config_override: Optional config override for this call

        Returns:
            Function result

        Raises:
            RetryExhaustedError: If all retries fail
            Exception: If non-retryable exception occurs
        """
        config = config_override or self.config
        self.stats.total_calls += 1

        last_exception: Optional[Exception] = None

        for attempt in range(config.max_retries + 1):
            try:
                result = func()

                if attempt == 0:
                    self.stats.successful_first_attempt += 1
                else:
                    self.stats.successful_after_retry += 1

                return result

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not self._is_retryable(e, config):
                    raise

                # Check if we have retries left
                if attempt >= config.max_retries:
                    break

                # Calculate and apply delay
                delay = self._calculate_delay(attempt, config)
                self.stats.total_retries += 1
                self.stats.total_delay_ms += delay * 1000

                self._emit_retry_event(attempt, delay, e)

                time.sleep(delay)

        self.stats.failed_all_retries += 1
        self._emit_exhausted_event(config.max_retries + 1, last_exception)

        raise RetryExhaustedError(
            self.name,
            config.max_retries + 1,
            last_exception,
        )

    def _is_retryable(self, exception: Exception, config: RetryConfig) -> bool:
        """Check if exception should trigger retry."""
        # Check exception type
        for exc_type in config.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True

        # Check for HTTP status code in exception
        status_code = getattr(exception, "status_code", None)
        if status_code and status_code in config.retryable_status_codes:
            return True

        return False

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for next retry attempt."""
        # Exponential backoff
        delay = config.base_delay_seconds * (config.exponential_base**attempt)

        # Cap at max delay
        delay = min(delay, config.max_delay_seconds)

        # Add jitter if enabled
        if config.jitter:
            jitter_range = delay * config.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def _emit_retry_event(
        self, attempt: int, delay: float, error: Exception
    ) -> None:
        """Emit retry attempt event."""
        if not self.event_callback:
            return

        event = ResilienceEvent(
            event_type=ResilienceEventType.RETRY_ATTEMPT,
            backend_name=self.name,
            metadata={
                "attempt": attempt + 1,
                "delay_seconds": delay,
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        try:
            self.event_callback(event)
        except Exception:
            pass

    def _emit_exhausted_event(
        self, attempts: int, error: Optional[Exception]
    ) -> None:
        """Emit retry exhausted event."""
        if not self.event_callback:
            return

        event = ResilienceEvent(
            event_type=ResilienceEventType.RETRY_EXHAUSTED,
            backend_name=self.name,
            metadata={
                "total_attempts": attempts,
                "last_error": str(error) if error else None,
            },
        )

        try:
            self.event_callback(event)
        except Exception:
            pass

    def get_stats(self) -> dict:
        """Get retry handler statistics as dictionary."""
        return {
            "total_calls": self.stats.total_calls,
            "successful_first_attempt": self.stats.successful_first_attempt,
            "successful_after_retry": self.stats.successful_after_retry,
            "failed_all_retries": self.stats.failed_all_retries,
            "total_retries": self.stats.total_retries,
            "total_delay_ms": self.stats.total_delay_ms,
            "success_rate": (
                (self.stats.successful_first_attempt + self.stats.successful_after_retry)
                / self.stats.total_calls
                if self.stats.total_calls > 0
                else 0.0
            ),
        }


def with_retry(config: Optional[RetryConfig] = None, name: Optional[str] = None):
    """
    Decorator for adding retry logic to functions.

    Example:
        @with_retry(RetryConfig(max_retries=3))
        def make_api_call():
            return api.call()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        handler_name = name or func.__name__
        handler = RetryHandler(handler_name, config)

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return handler.execute(lambda: func(*args, **kwargs))

        # Attach handler for inspection
        wrapper._retry_handler = handler  # type: ignore
        return wrapper

    return decorator


def retry_on(
    *exception_types: type,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
):
    """
    Decorator for retrying on specific exception types.

    Example:
        @retry_on(ConnectionError, TimeoutError, max_retries=5)
        def fetch_data():
            return requests.get(url).json()
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay_seconds=base_delay,
        max_delay_seconds=max_delay,
        jitter=jitter,
        retryable_exceptions=set(exception_types),
    )
    return with_retry(config)
