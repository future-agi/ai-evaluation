"""
Graceful degradation with fallback chains.

Provides fallback mechanisms when primary operations fail.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Generic, List, Optional, TypeVar

from .types import (
    DegradationConfig,
    EventCallback,
    ResilienceEvent,
    ResilienceEventType,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class FallbackStats:
    """Statistics for fallback handler."""

    total_calls: int = 0
    primary_success: int = 0
    fallback_used: int = 0
    all_failed: int = 0


@dataclass
class FallbackOption(Generic[T]):
    """
    A fallback option with name and callable.

    Attributes:
        name: Identifier for this fallback
        func: Callable that returns the fallback value
        condition: Optional condition to check before using this fallback
    """

    name: str
    func: Callable[[], T]
    condition: Optional[Callable[[Exception], bool]] = None


class FallbackChain(Generic[T]):
    """
    Fallback chain for graceful degradation.

    Tries a series of fallbacks in order when the primary operation fails.

    Example:
        chain = FallbackChain("api_call")
        chain.add_fallback("cache", lambda: get_from_cache())
        chain.add_fallback("default", lambda: default_value)

        result = chain.execute(lambda: api.call())
    """

    def __init__(
        self,
        name: str,
        config: Optional[DegradationConfig] = None,
        event_callback: Optional[EventCallback] = None,
    ):
        """
        Initialize fallback chain.

        Args:
            name: Name for this fallback chain
            config: Degradation configuration
            event_callback: Callback for degradation events
        """
        self.name = name
        self.config = config or DegradationConfig()
        self.event_callback = event_callback

        self._fallbacks: List[FallbackOption[T]] = []
        self.stats = FallbackStats()

    def add_fallback(
        self,
        name: str,
        func: Callable[[], T],
        condition: Optional[Callable[[Exception], bool]] = None,
    ) -> "FallbackChain[T]":
        """
        Add a fallback option to the chain.

        Args:
            name: Name for this fallback
            func: Callable that returns the fallback value
            condition: Optional condition (receives exception, returns bool)

        Returns:
            Self for chaining
        """
        self._fallbacks.append(FallbackOption(name=name, func=func, condition=condition))
        return self

    def execute(
        self,
        primary: Callable[[], T],
        fallback_exceptions: Optional[set[type]] = None,
    ) -> T:
        """
        Execute primary with fallbacks on failure.

        Args:
            primary: Primary callable to execute
            fallback_exceptions: Exception types that trigger fallback
                                (defaults to config.fallback_exceptions)

        Returns:
            Result from primary or a fallback

        Raises:
            Exception: If primary and all fallbacks fail
        """
        exceptions = fallback_exceptions or self.config.fallback_on_exceptions
        self.stats.total_calls += 1

        # Try primary
        try:
            result = primary()
            self.stats.primary_success += 1
            return result
        except Exception as e:
            if not self._should_fallback(e, exceptions):
                raise

            primary_error = e

        # Try fallbacks in order
        for fallback in self._fallbacks:
            # Check condition if specified
            if fallback.condition and not fallback.condition(primary_error):
                continue

            try:
                result = fallback.func()
                self.stats.fallback_used += 1
                self._emit_fallback_event(fallback.name, primary_error)
                return result
            except Exception as fallback_error:
                logger.debug(
                    f"Fallback '{fallback.name}' failed: {fallback_error}"
                )
                continue

        # All failed
        self.stats.all_failed += 1
        self._emit_all_failed_event(primary_error)
        raise primary_error

    def _should_fallback(
        self, exception: Exception, exceptions: set[type]
    ) -> bool:
        """Check if exception should trigger fallback."""
        if not exceptions:
            return True  # Fallback on all exceptions

        for exc_type in exceptions:
            if isinstance(exception, exc_type):
                return True
        return False

    def _emit_fallback_event(self, fallback_name: str, error: Exception) -> None:
        """Emit fallback used event."""
        if not self.event_callback:
            return

        event = ResilienceEvent(
            event_type=ResilienceEventType.FALLBACK_USED,
            backend_name=self.name,
            metadata={
                "fallback_name": fallback_name,
                "primary_error": str(error),
                "error_type": type(error).__name__,
            },
        )

        try:
            self.event_callback(event)
        except Exception:
            pass

    def _emit_all_failed_event(self, error: Exception) -> None:
        """Emit all fallbacks failed event."""
        if not self.event_callback:
            return

        event = ResilienceEvent(
            event_type=ResilienceEventType.DEGRADATION_ACTIVE,
            backend_name=self.name,
            metadata={
                "fallback_count": len(self._fallbacks),
                "all_failed": True,
                "last_error": str(error),
            },
        )

        try:
            self.event_callback(event)
        except Exception:
            pass

    def get_stats(self) -> dict:
        """Get fallback chain statistics as dictionary."""
        return {
            "total_calls": self.stats.total_calls,
            "primary_success": self.stats.primary_success,
            "fallback_used": self.stats.fallback_used,
            "all_failed": self.stats.all_failed,
            "fallback_rate": (
                self.stats.fallback_used / self.stats.total_calls
                if self.stats.total_calls > 0
                else 0.0
            ),
        }


class DegradationHandler:
    """
    Handler for graceful degradation with primary/secondary pattern.

    Manages a primary operation with a single fallback operation.

    Example:
        handler = DegradationHandler(
            "api",
            primary=lambda: api.call(),
            fallback=lambda: cache.get(),
        )

        result = handler.execute()
    """

    def __init__(
        self,
        name: str,
        primary: Callable[[], T],
        fallback: Callable[[], T],
        config: Optional[DegradationConfig] = None,
        event_callback: Optional[EventCallback] = None,
    ):
        """
        Initialize degradation handler.

        Args:
            name: Name for this handler
            primary: Primary callable
            fallback: Fallback callable
            config: Degradation configuration
            event_callback: Callback for events
        """
        self.name = name
        self._primary = primary
        self._fallback = fallback
        self.config = config or DegradationConfig()
        self.event_callback = event_callback

        self._chain: FallbackChain = FallbackChain(
            name, config, event_callback
        )
        self._chain.add_fallback("fallback", fallback)

    def execute(self) -> T:
        """
        Execute primary with fallback on failure.

        Returns:
            Result from primary or fallback

        Raises:
            Exception: If both primary and fallback fail
        """
        return self._chain.execute(self._primary)

    @property
    def stats(self) -> FallbackStats:
        """Get statistics."""
        return self._chain.stats

    def get_stats(self) -> dict:
        """Get statistics as dictionary."""
        return self._chain.get_stats()


def with_fallback(
    fallback_value: T,
    exceptions: Optional[set[type]] = None,
    name: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding a static fallback value.

    Example:
        @with_fallback(default_config)
        def load_config():
            return api.get_config()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        handler_name = name or func.__name__
        chain: FallbackChain = FallbackChain(handler_name)
        chain.add_fallback("default", lambda: fallback_value)

        def wrapper(*args, **kwargs) -> T:
            return chain.execute(
                lambda: func(*args, **kwargs),
                fallback_exceptions=exceptions,
            )

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._fallback_chain = chain  # type: ignore
        return wrapper

    return decorator


def with_fallback_func(
    fallback_func: Callable[[], T],
    exceptions: Optional[set[type]] = None,
    name: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding a fallback function.

    Example:
        @with_fallback_func(lambda: load_from_cache())
        def load_config():
            return api.get_config()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        handler_name = name or func.__name__
        chain: FallbackChain = FallbackChain(handler_name)
        chain.add_fallback("fallback", fallback_func)

        def wrapper(*args, **kwargs) -> T:
            return chain.execute(
                lambda: func(*args, **kwargs),
                fallback_exceptions=exceptions,
            )

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._fallback_chain = chain  # type: ignore
        return wrapper

    return decorator
