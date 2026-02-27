"""Tests for graceful degradation implementation."""

import pytest

from fi.evals.framework.resilience.degradation import (
    FallbackChain,
    FallbackStats,
    DegradationHandler,
    with_fallback,
    with_fallback_func,
)
from fi.evals.framework.resilience.types import (
    DegradationConfig,
    ResilienceEventType,
)


class TestFallbackChainBasic:
    """Basic functionality tests."""

    def test_primary_success(self):
        """Primary succeeds without using fallback."""
        chain: FallbackChain[int] = FallbackChain("test")
        chain.add_fallback("fallback", lambda: 0)

        result = chain.execute(lambda: 42)

        assert result == 42
        assert chain.stats.primary_success == 1
        assert chain.stats.fallback_used == 0

    def test_fallback_on_failure(self):
        """Fallback used when primary fails."""
        chain: FallbackChain[str] = FallbackChain("test")
        chain.add_fallback("fallback", lambda: "fallback_value")

        result = chain.execute(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert result == "fallback_value"
        assert chain.stats.primary_success == 0
        assert chain.stats.fallback_used == 1

    def test_multiple_fallbacks_first_succeeds(self):
        """First working fallback is used."""
        chain: FallbackChain[str] = FallbackChain("test")
        chain.add_fallback("first", lambda: "first_fallback")
        chain.add_fallback("second", lambda: "second_fallback")

        result = chain.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        assert result == "first_fallback"
        assert chain.stats.fallback_used == 1

    def test_multiple_fallbacks_first_fails(self):
        """Second fallback used when first fails."""
        chain: FallbackChain[str] = FallbackChain("test")
        chain.add_fallback("first", lambda: (_ for _ in ()).throw(ValueError()))
        chain.add_fallback("second", lambda: "second_fallback")

        result = chain.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        assert result == "second_fallback"
        assert chain.stats.fallback_used == 1

    def test_all_fallbacks_fail(self):
        """Original exception raised when all fallbacks fail."""
        chain: FallbackChain[str] = FallbackChain("test")
        chain.add_fallback("first", lambda: (_ for _ in ()).throw(ValueError()))
        chain.add_fallback("second", lambda: (_ for _ in ()).throw(TypeError()))

        with pytest.raises(RuntimeError, match="primary failure"):
            chain.execute(lambda: (_ for _ in ()).throw(RuntimeError("primary failure")))

        assert chain.stats.all_failed == 1

    def test_chaining_api(self):
        """add_fallback returns self for chaining."""
        chain: FallbackChain[int] = FallbackChain("test")
        result = (
            chain
            .add_fallback("first", lambda: 1)
            .add_fallback("second", lambda: 2)
            .add_fallback("third", lambda: 3)
        )

        assert result is chain
        assert len(chain._fallbacks) == 3


class TestFallbackChainExceptionFiltering:
    """Tests for exception-based fallback."""

    def test_fallback_on_specific_exceptions(self):
        """Fallback only on specified exceptions."""
        config = DegradationConfig(fallback_on_exceptions={TimeoutError, ConnectionError})
        chain: FallbackChain[str] = FallbackChain("test", config)
        chain.add_fallback("fallback", lambda: "fallback")

        # TimeoutError triggers fallback
        result = chain.execute(lambda: (_ for _ in ()).throw(TimeoutError()))
        assert result == "fallback"

    def test_no_fallback_on_unspecified_exception(self):
        """Non-specified exceptions raise immediately."""
        config = DegradationConfig(fallback_on_exceptions={TimeoutError})
        chain: FallbackChain[str] = FallbackChain("test", config)
        chain.add_fallback("fallback", lambda: "fallback")

        # ValueError not in fallback_on_exceptions
        with pytest.raises(ValueError):
            chain.execute(lambda: (_ for _ in ()).throw(ValueError()))

        assert chain.stats.fallback_used == 0

    def test_override_fallback_on_exceptions(self):
        """Can override fallback exceptions per call."""
        config = DegradationConfig(fallback_on_exceptions={TimeoutError})
        chain: FallbackChain[str] = FallbackChain("test", config)
        chain.add_fallback("fallback", lambda: "fallback")

        # Override to include ValueError
        result = chain.execute(
            lambda: (_ for _ in ()).throw(ValueError()),
            fallback_exceptions={ValueError},
        )
        assert result == "fallback"


class TestFallbackChainConditions:
    """Tests for conditional fallbacks."""

    def test_conditional_fallback_used(self):
        """Fallback with matching condition is used."""
        chain: FallbackChain[str] = FallbackChain("test")
        chain.add_fallback(
            "timeout_handler",
            lambda: "timeout_fallback",
            condition=lambda e: isinstance(e, TimeoutError),
        )
        chain.add_fallback("default", lambda: "default_fallback")

        result = chain.execute(lambda: (_ for _ in ()).throw(TimeoutError()))
        assert result == "timeout_fallback"

    def test_conditional_fallback_skipped(self):
        """Fallback with non-matching condition is skipped."""
        chain: FallbackChain[str] = FallbackChain("test")
        chain.add_fallback(
            "timeout_handler",
            lambda: "timeout_fallback",
            condition=lambda e: isinstance(e, TimeoutError),
        )
        chain.add_fallback("default", lambda: "default_fallback")

        # ValueError doesn't match timeout condition
        result = chain.execute(lambda: (_ for _ in ()).throw(ValueError()))
        assert result == "default_fallback"

    def test_multiple_conditional_fallbacks(self):
        """Multiple conditional fallbacks work correctly."""
        chain: FallbackChain[str] = FallbackChain("test")
        chain.add_fallback(
            "timeout",
            lambda: "timeout",
            condition=lambda e: isinstance(e, TimeoutError),
        )
        chain.add_fallback(
            "connection",
            lambda: "connection",
            condition=lambda e: isinstance(e, ConnectionError),
        )
        chain.add_fallback("default", lambda: "default")

        # Each exception type gets appropriate fallback
        assert chain.execute(lambda: (_ for _ in ()).throw(TimeoutError())) == "timeout"
        assert chain.execute(lambda: (_ for _ in ()).throw(ConnectionError())) == "connection"
        assert chain.execute(lambda: (_ for _ in ()).throw(ValueError())) == "default"


class TestFallbackChainEvents:
    """Tests for event callbacks."""

    def test_fallback_event_emitted(self):
        """Event emitted when fallback is used."""
        events = []
        chain: FallbackChain[str] = FallbackChain(
            "test", event_callback=lambda e: events.append(e)
        )
        chain.add_fallback("cache", lambda: "cached")

        chain.execute(lambda: (_ for _ in ()).throw(RuntimeError("primary fail")))

        assert len(events) == 1
        assert events[0].event_type == ResilienceEventType.FALLBACK_USED
        assert events[0].backend_name == "test"
        assert events[0].metadata["fallback_name"] == "cache"
        assert "primary fail" in events[0].metadata["primary_error"]

    def test_all_failed_event_emitted(self):
        """Event emitted when all fallbacks fail."""
        events = []
        chain: FallbackChain[str] = FallbackChain(
            "test", event_callback=lambda e: events.append(e)
        )
        chain.add_fallback("bad", lambda: (_ for _ in ()).throw(ValueError()))

        with pytest.raises(RuntimeError):
            chain.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        # Should have degradation_active event
        assert len(events) == 1
        assert events[0].event_type == ResilienceEventType.DEGRADATION_ACTIVE
        assert events[0].metadata["all_failed"] is True

    def test_no_event_on_success(self):
        """No event when primary succeeds."""
        events = []
        chain: FallbackChain[int] = FallbackChain(
            "test", event_callback=lambda e: events.append(e)
        )
        chain.add_fallback("fallback", lambda: 0)

        chain.execute(lambda: 42)

        assert len(events) == 0

    def test_callback_exception_handled(self):
        """Callback exceptions don't break chain."""

        def bad_callback(e):
            raise RuntimeError("callback error")

        chain: FallbackChain[str] = FallbackChain("test", event_callback=bad_callback)
        chain.add_fallback("fallback", lambda: "ok")

        # Should not raise callback exception
        result = chain.execute(lambda: (_ for _ in ()).throw(ValueError()))
        assert result == "ok"


class TestFallbackChainStats:
    """Tests for statistics."""

    def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        chain: FallbackChain[int] = FallbackChain("test")
        chain.add_fallback("fallback", lambda: 0)

        # Primary success
        chain.execute(lambda: 1)
        chain.execute(lambda: 2)

        # Fallback used
        chain.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        assert chain.stats.total_calls == 3
        assert chain.stats.primary_success == 2
        assert chain.stats.fallback_used == 1

    def test_get_stats(self):
        """Get stats returns correct values."""
        chain: FallbackChain[int] = FallbackChain("test")
        chain.add_fallback("fallback", lambda: 0)

        chain.execute(lambda: 1)
        chain.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        stats = chain.get_stats()
        assert stats["total_calls"] == 2
        assert stats["primary_success"] == 1
        assert stats["fallback_used"] == 1
        assert stats["fallback_rate"] == 0.5


class TestDegradationHandler:
    """Tests for DegradationHandler."""

    def test_primary_success(self):
        """Primary succeeds."""
        handler: DegradationHandler = DegradationHandler(
            "test",
            primary=lambda: "primary",
            fallback=lambda: "fallback",
        )

        result = handler.execute()

        assert result == "primary"
        assert handler.stats.primary_success == 1

    def test_fallback_on_failure(self):
        """Fallback used when primary fails."""
        handler: DegradationHandler = DegradationHandler(
            "test",
            primary=lambda: (_ for _ in ()).throw(RuntimeError()),
            fallback=lambda: "fallback",
        )

        result = handler.execute()

        assert result == "fallback"
        assert handler.stats.fallback_used == 1

    def test_both_fail(self):
        """Exception raised when both fail."""
        handler: DegradationHandler = DegradationHandler(
            "test",
            primary=lambda: (_ for _ in ()).throw(RuntimeError("primary")),
            fallback=lambda: (_ for _ in ()).throw(ValueError("fallback")),
        )

        with pytest.raises(RuntimeError, match="primary"):
            handler.execute()

        assert handler.stats.all_failed == 1

    def test_get_stats(self):
        """Get stats works correctly."""
        handler: DegradationHandler = DegradationHandler(
            "test",
            primary=lambda: 42,
            fallback=lambda: 0,
        )

        handler.execute()

        stats = handler.get_stats()
        assert stats["total_calls"] == 1
        assert stats["primary_success"] == 1


class TestWithFallbackDecorator:
    """Tests for @with_fallback decorator."""

    def test_decorator_success(self):
        """Decorated function returns primary result."""

        @with_fallback(fallback_value=0)
        def get_value():
            return 42

        assert get_value() == 42

    def test_decorator_fallback(self):
        """Decorated function returns fallback on failure."""

        @with_fallback(fallback_value="default")
        def get_value():
            raise RuntimeError()

        assert get_value() == "default"

    def test_decorator_preserves_name(self):
        """Decorator preserves function name."""

        @with_fallback(fallback_value=None)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_with_args(self):
        """Decorated function accepts arguments."""

        @with_fallback(fallback_value=0)
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_decorator_with_exceptions(self):
        """Decorator respects exception filter."""

        @with_fallback(fallback_value="default", exceptions={TimeoutError})
        def get_value(fail_type):
            if fail_type == "timeout":
                raise TimeoutError()
            raise ValueError()

        # TimeoutError triggers fallback
        assert get_value("timeout") == "default"

        # ValueError does not
        with pytest.raises(ValueError):
            get_value("other")

    def test_decorator_chain_accessible(self):
        """Fallback chain is accessible on decorated function."""

        @with_fallback(fallback_value=0)
        def func():
            return 1

        func()
        assert hasattr(func, "_fallback_chain")
        assert func._fallback_chain.stats.total_calls == 1


class TestWithFallbackFuncDecorator:
    """Tests for @with_fallback_func decorator."""

    def test_decorator_success(self):
        """Decorated function returns primary result."""

        @with_fallback_func(lambda: 0)
        def get_value():
            return 42

        assert get_value() == 42

    def test_decorator_fallback(self):
        """Decorated function calls fallback on failure."""
        fallback_called = [False]

        def fallback():
            fallback_called[0] = True
            return "from_fallback"

        @with_fallback_func(fallback)
        def get_value():
            raise RuntimeError()

        result = get_value()

        assert result == "from_fallback"
        assert fallback_called[0] is True

    def test_decorator_fallback_with_closure(self):
        """Fallback function can access closure."""
        cache = {"value": "cached_data"}

        @with_fallback_func(lambda: cache["value"])
        def get_value():
            raise RuntimeError()

        assert get_value() == "cached_data"

    def test_decorator_preserves_name(self):
        """Decorator preserves function name."""

        @with_fallback_func(lambda: None)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


class TestFallbackChainEdgeCases:
    """Edge case tests."""

    def test_empty_fallback_chain(self):
        """Chain with no fallbacks raises original exception."""
        chain: FallbackChain[int] = FallbackChain("test")

        with pytest.raises(RuntimeError, match="no fallbacks"):
            chain.execute(lambda: (_ for _ in ()).throw(RuntimeError("no fallbacks")))

    def test_no_fallback_on_exceptions_means_all(self):
        """Empty fallback_on_exceptions means fallback on all."""
        config = DegradationConfig(fallback_on_exceptions=set())
        chain: FallbackChain[str] = FallbackChain("test", config)
        chain.add_fallback("fallback", lambda: "ok")

        # Any exception triggers fallback
        assert chain.execute(lambda: (_ for _ in ()).throw(ValueError())) == "ok"
        assert chain.execute(lambda: (_ for _ in ()).throw(TypeError())) == "ok"
        assert chain.execute(lambda: (_ for _ in ()).throw(RuntimeError())) == "ok"

    def test_fallback_returns_none(self):
        """Fallback returning None is valid."""
        chain: FallbackChain[None] = FallbackChain("test")
        chain.add_fallback("fallback", lambda: None)

        result = chain.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        assert result is None
        assert chain.stats.fallback_used == 1

    def test_fallback_returns_falsy_value(self):
        """Fallback returning falsy values works."""
        chain: FallbackChain = FallbackChain("test")
        chain.add_fallback("zero", lambda: 0)

        result = chain.execute(lambda: (_ for _ in ()).throw(RuntimeError()))

        assert result == 0
        assert chain.stats.fallback_used == 1
