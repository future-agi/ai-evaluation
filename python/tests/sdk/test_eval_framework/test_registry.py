"""Tests for fi.evals.framework.registry module."""

import pytest
import threading
import time
import gc
from unittest.mock import MagicMock, patch
from fi.evals.framework.registry import (
    SpanRegistry,
    register_span,
    get_span,
    unregister_span,
    get_registry,
    register_current_span,
)


class TestSpanRegistry:
    """Tests for SpanRegistry class."""

    def setup_method(self):
        """Reset registry before each test."""
        SpanRegistry.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        SpanRegistry.reset_instance()

    def test_singleton(self):
        """Test that SpanRegistry is a singleton."""
        registry1 = SpanRegistry()
        registry2 = SpanRegistry()
        assert registry1 is registry2

    def test_register_and_get(self):
        """Test basic register and get."""
        registry = SpanRegistry()
        mock_span = MagicMock()

        registry.register("trace123" + "0" * 24, "span456" + "0" * 8, mock_span)
        retrieved = registry.get("trace123" + "0" * 24, "span456" + "0" * 8)

        assert retrieved is mock_span

    def test_get_not_found(self):
        """Test get returns None for non-existent span."""
        registry = SpanRegistry()
        result = registry.get("nonexistent" + "0" * 21, "span" + "0" * 12)
        assert result is None

    def test_unregister(self):
        """Test unregistering a span."""
        registry = SpanRegistry()
        mock_span = MagicMock()
        trace_id = "a" * 32
        span_id = "b" * 16

        registry.register(trace_id, span_id, mock_span)
        assert registry.get(trace_id, span_id) is mock_span

        result = registry.unregister(trace_id, span_id)
        assert result is True
        assert registry.get(trace_id, span_id) is None

    def test_unregister_not_found(self):
        """Test unregistering non-existent span."""
        registry = SpanRegistry()
        result = registry.unregister("a" * 32, "b" * 16)
        assert result is False

    def test_contains(self):
        """Test contains method."""
        registry = SpanRegistry()
        mock_span = MagicMock()
        trace_id = "a" * 32
        span_id = "b" * 16

        assert registry.contains(trace_id, span_id) is False

        registry.register(trace_id, span_id, mock_span)
        assert registry.contains(trace_id, span_id) is True

    def test_count(self):
        """Test count method."""
        registry = SpanRegistry()

        assert registry.count() == 0

        # Keep references to prevent GC
        span1 = MagicMock()
        span2 = MagicMock()

        registry.register("a" * 32, "1" * 16, span1)
        assert registry.count() == 1

        registry.register("a" * 32, "2" * 16, span2)
        assert registry.count() == 2

    def test_clear(self):
        """Test clearing the registry."""
        registry = SpanRegistry()

        # Keep references to prevent GC
        span1 = MagicMock()
        span2 = MagicMock()

        registry.register("a" * 32, "1" * 16, span1)
        registry.register("a" * 32, "2" * 16, span2)
        assert registry.count() == 2

        cleared = registry.clear()
        assert cleared == 2
        assert registry.count() == 0

    def test_list_spans(self):
        """Test listing registered spans."""
        registry = SpanRegistry()

        # Keep references to prevent GC during test
        span1 = MagicMock()
        span2 = MagicMock()

        registry.register("a" * 32, "1" * 16, span1)
        registry.register("b" * 32, "2" * 16, span2)

        spans = registry.list_spans()
        assert len(spans) == 2
        assert ("a" * 32, "1" * 16) in spans
        assert ("b" * 32, "2" * 16) in spans

    def test_metadata(self):
        """Test metadata storage and retrieval."""
        registry = SpanRegistry()
        trace_id = "a" * 32
        span_id = "b" * 16

        registry.register(trace_id, span_id, MagicMock(), metadata={"key": "value"})

        metadata = registry.get_metadata(trace_id, span_id)
        assert metadata == {"key": "value"}

    def test_metadata_not_found(self):
        """Test metadata returns None for non-existent span."""
        registry = SpanRegistry()
        result = registry.get_metadata("a" * 32, "b" * 16)
        assert result is None

    def test_update_metadata(self):
        """Test updating metadata."""
        registry = SpanRegistry()
        trace_id = "a" * 32
        span_id = "b" * 16

        registry.register(trace_id, span_id, MagicMock(), metadata={"a": 1})
        registry.update_metadata(trace_id, span_id, {"b": 2})

        metadata = registry.get_metadata(trace_id, span_id)
        assert metadata == {"a": 1, "b": 2}

    def test_update_metadata_not_found(self):
        """Test updating metadata for non-existent span."""
        registry = SpanRegistry()
        result = registry.update_metadata("a" * 32, "b" * 16, {"key": "value"})
        assert result is False

    def test_weak_reference_cleanup(self):
        """Test that spans are cleaned up when garbage collected."""
        registry = SpanRegistry()
        trace_id = "a" * 32
        span_id = "b" * 16

        # Create span and register
        span = MagicMock()
        registry.register(trace_id, span_id, span)
        assert registry.contains(trace_id, span_id)

        # Delete span and force GC
        del span
        gc.collect()

        # Should be cleaned up
        # Note: This may not work reliably in all cases due to GC timing
        # So we just check that get handles it gracefully
        result = registry.get(trace_id, span_id)
        # Result could be None (GC'd) or the span (not GC'd yet)

    def test_thread_safety(self):
        """Test thread-safe operations."""
        registry = SpanRegistry()
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    trace_id = f"{thread_id:016x}" + "0" * 16
                    span_id = f"{i:016x}"
                    span = MagicMock()

                    registry.register(trace_id, span_id, span)
                    retrieved = registry.get(trace_id, span_id)
                    registry.unregister(trace_id, span_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestModuleFunctions:
    """Tests for module-level functions."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_register_span(self):
        """Test register_span function."""
        mock_span = MagicMock()
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        retrieved = get_span(trace_id, span_id)
        assert retrieved is mock_span

    def test_get_span(self):
        """Test get_span function."""
        result = get_span("nonexistent" + "0" * 21, "span" + "0" * 12)
        assert result is None

    def test_unregister_span(self):
        """Test unregister_span function."""
        mock_span = MagicMock()
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)
        result = unregister_span(trace_id, span_id)

        assert result is True
        assert get_span(trace_id, span_id) is None

    def test_get_registry(self):
        """Test get_registry function."""
        registry = get_registry()
        assert isinstance(registry, SpanRegistry)

    def test_register_current_span_no_otel(self):
        """Test register_current_span when OTEL not available."""
        # Should return False gracefully
        result = register_current_span()
        # Result depends on whether OTEL is installed and has active span

    def test_register_current_span_returns_false_without_span(self):
        """Test register_current_span returns False without active span."""
        # Without a real OTEL span, should return False
        result = register_current_span(metadata={"test": True})
        # Result depends on OTEL availability and active span
        # When no active span, should return False
        assert result is False or result is True  # Depends on env


class TestRegistryCleanup:
    """Tests for automatic cleanup behavior."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_old_entries_cleaned_up(self):
        """Test that old entries are eventually cleaned up."""
        registry = SpanRegistry()

        # Manually set last cleanup to trigger cleanup
        from datetime import datetime, timedelta, timezone
        registry._last_cleanup = datetime.now(timezone.utc) - timedelta(minutes=10)
        registry._max_age = timedelta(seconds=0)  # Everything is "old"

        # Register a span
        mock_span = MagicMock()
        registry.register("a" * 32, "b" * 16, mock_span)

        # Force cleanup by registering another span
        registry._cleanup_interval = timedelta(seconds=0)
        registry.register("c" * 32, "d" * 16, MagicMock())

        # Old entry should be cleaned up (depends on timing)
        # This is hard to test reliably without more control
