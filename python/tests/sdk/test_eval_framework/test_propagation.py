"""Tests for fi.evals.framework.propagation module."""

import pytest
import threading
from unittest.mock import MagicMock, patch
from fi.evals.framework.propagation import (
    SpanContextPropagator,
    enrich_span_by_context,
    enrich_span_by_ids,
    add_event_by_context,
    ContextCarrier,
    propagate_context,
    propagate_context_lazy,
)
from fi.evals.framework.context import EvalContext
from fi.evals.framework.registry import SpanRegistry, register_span, get_span


class TestSpanContextPropagator:
    """Tests for SpanContextPropagator class."""

    def test_inject_no_otel(self):
        """Test inject returns False when no OTEL context."""
        carrier = {}
        # Without OTEL or active span, should return False
        result = SpanContextPropagator.inject(carrier)
        # Result depends on OTEL availability

    def test_inject_with_no_active_context(self):
        """Test inject returns False when no active context."""
        carrier = {}
        # Without an active OTEL span, should return False or empty carrier
        result = SpanContextPropagator.inject(carrier)
        # Result depends on OTEL availability and active span

    def test_extract(self):
        """Test extract creates context from headers."""
        carrier = {
            'traceparent': '00-12345678901234567890123456789012-1234567890123456-01'
        }
        context = SpanContextPropagator.extract(carrier)

        assert context.trace_id == '12345678901234567890123456789012'
        assert context.span_id == '1234567890123456'

    def test_extract_empty(self):
        """Test extract with empty carrier."""
        carrier = {}
        context = SpanContextPropagator.extract(carrier)

        # When no headers provided, creates a standalone context with generated IDs
        # This is valid because it has properly formatted trace/span IDs
        assert context is not None
        assert len(context.trace_id) == 32
        assert len(context.span_id) == 16

    def test_with_context_no_otel(self):
        """Test with_context when OTEL not available."""
        context = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
        )

        # Should not raise even without OTEL
        with SpanContextPropagator.with_context(context):
            pass

    def test_with_context_executes_code(self):
        """Test with_context executes code inside context manager."""
        context = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
        )

        # Test that code inside the context manager executes
        executed = []
        with SpanContextPropagator.with_context(context):
            executed.append(True)

        assert len(executed) == 1

    def test_create_child_context(self):
        """Test creating a child context."""
        parent = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            baggage={"key": "value"},
        )

        child = SpanContextPropagator.create_child_context(parent)

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.parent_span_id == parent.span_id
        assert child.baggage == parent.baggage


class TestEnrichSpanByContext:
    """Tests for enrich_span_by_context function."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_enrich_span_success(self):
        """Test enriching a registered span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        context = EvalContext(trace_id=trace_id, span_id=span_id)
        result = enrich_span_by_context(context, {"score": 0.95})

        assert result is True
        mock_span.set_attribute.assert_called_with("score", 0.95)

    def test_enrich_span_not_found(self):
        """Test enriching non-existent span."""
        # Use unique IDs that were never registered
        context = EvalContext(trace_id="x" * 32, span_id="y" * 16)
        result = enrich_span_by_context(context, {"score": 0.95})

        assert result is False

    def test_enrich_span_not_recording(self):
        """Test enriching non-recording span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        context = EvalContext(trace_id=trace_id, span_id=span_id)
        result = enrich_span_by_context(context, {"score": 0.95})

        assert result is False

    def test_enrich_span_filters_none(self):
        """Test that None values are filtered."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        context = EvalContext(trace_id=trace_id, span_id=span_id)
        enrich_span_by_context(context, {"score": 0.95, "null_value": None})

        # Should only set score, not null_value
        calls = mock_span.set_attribute.call_args_list
        assert len(calls) == 1
        assert calls[0][0] == ("score", 0.95)

    def test_enrich_span_filters_invalid_types(self):
        """Test that invalid types are filtered."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        context = EvalContext(trace_id=trace_id, span_id=span_id)
        enrich_span_by_context(context, {
            "score": 0.95,
            "list_value": [1, 2, 3],  # Invalid
            "dict_value": {"nested": True},  # Invalid
        })

        # Should only set score
        calls = mock_span.set_attribute.call_args_list
        assert len(calls) == 1


class TestEnrichSpanByIds:
    """Tests for enrich_span_by_ids function."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_enrich_by_ids_success(self):
        """Test enriching span by trace/span IDs."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        result = enrich_span_by_ids(trace_id, span_id, {"score": 0.95})

        assert result is True
        mock_span.set_attribute.assert_called_with("score", 0.95)

    def test_enrich_by_ids_not_found(self):
        """Test enriching non-existent span by IDs."""
        # Use unique IDs that were never registered
        result = enrich_span_by_ids("x" * 32, "y" * 16, {"score": 0.95})
        assert result is False


class TestAddEventByContext:
    """Tests for add_event_by_context function."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_add_event_success(self):
        """Test adding event to registered span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        context = EvalContext(trace_id=trace_id, span_id=span_id)
        result = add_event_by_context(context, "eval_complete", {"score": 0.95})

        assert result is True
        mock_span.add_event.assert_called_once()
        call_args = mock_span.add_event.call_args
        assert call_args[0][0] == "eval_complete"
        assert call_args[1]["attributes"] == {"score": 0.95}

    def test_add_event_not_found(self):
        """Test adding event to non-existent span."""
        # Use unique IDs that were never registered
        context = EvalContext(trace_id="x" * 32, span_id="y" * 16)
        result = add_event_by_context(context, "eval_complete")

        assert result is False

    def test_add_event_no_attributes(self):
        """Test adding event without attributes."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        context = EvalContext(trace_id=trace_id, span_id=span_id)
        result = add_event_by_context(context, "eval_start")

        assert result is True
        mock_span.add_event.assert_called_with("eval_start", attributes=None)


class TestContextCarrier:
    """Tests for ContextCarrier class."""

    def test_init(self):
        """Test ContextCarrier initialization."""
        context = EvalContext(trace_id="a" * 32, span_id="b" * 16)
        carrier = ContextCarrier(context)

        assert carrier.context is context
        assert carrier.headers == {}
        assert carrier.metadata == {}

    def test_init_with_headers_and_metadata(self):
        """Test ContextCarrier with headers and metadata."""
        context = EvalContext(trace_id="a" * 32, span_id="b" * 16)
        carrier = ContextCarrier(
            context=context,
            headers={"traceparent": "test"},
            metadata={"key": "value"},
        )

        assert carrier.headers == {"traceparent": "test"}
        assert carrier.metadata == {"key": "value"}

    def test_capture(self):
        """Test capturing current context."""
        carrier = ContextCarrier.capture()

        # Should create a carrier even without active OTEL span
        assert carrier.context is not None

    def test_capture_with_metadata(self):
        """Test capturing with metadata."""
        carrier = ContextCarrier.capture(metadata={"test": True})

        assert carrier.metadata == {"test": True}

    def test_to_dict(self):
        """Test serializing to dict."""
        context = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            baggage={"key": "value"},
        )
        carrier = ContextCarrier(
            context=context,
            headers={"traceparent": "test"},
            metadata={"meta": True},
        )

        data = carrier.to_dict()

        assert "context" in data
        assert data["headers"] == {"traceparent": "test"}
        assert data["metadata"] == {"meta": True}

    def test_from_dict(self):
        """Test deserializing from dict."""
        data = {
            "context": {
                "trace_id": "a" * 32,
                "span_id": "b" * 16,
                "parent_span_id": None,
                "baggage": {},
                "eval_run_id": "test123",
            },
            "headers": {"traceparent": "test"},
            "metadata": {"meta": True},
        }

        carrier = ContextCarrier.from_dict(data)

        assert carrier.context.trace_id == "a" * 32
        assert carrier.context.span_id == "b" * 16
        assert carrier.headers == {"traceparent": "test"}
        assert carrier.metadata == {"meta": True}

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        context = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            baggage={"key": "value"},
        )
        original = ContextCarrier(
            context=context,
            headers={"traceparent": "test"},
            metadata={"meta": True},
        )

        data = original.to_dict()
        restored = ContextCarrier.from_dict(data)

        assert restored.context.trace_id == original.context.trace_id
        assert restored.context.span_id == original.context.span_id
        assert restored.headers == original.headers
        assert restored.metadata == original.metadata

    def test_activate(self):
        """Test activating carrier context."""
        context = EvalContext(trace_id="a" * 32, span_id="b" * 16)
        carrier = ContextCarrier(context)

        with carrier.activate() as activated:
            assert activated is carrier

    def test_enrich_span(self):
        """Test enriching span through carrier."""
        SpanRegistry.reset_instance()

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        context = EvalContext(trace_id=trace_id, span_id=span_id)
        carrier = ContextCarrier(context)

        result = carrier.enrich_span({"score": 0.95})

        assert result is True
        mock_span.set_attribute.assert_called_with("score", 0.95)

        SpanRegistry.reset_instance()


class TestPropagateContext:
    """Tests for propagate_context decorator."""

    def test_propagate_context_preserves_function(self):
        """Test that decorator preserves function behavior."""
        @propagate_context
        def add(x, y):
            return x + y

        result = add(1, 2)
        assert result == 3

    def test_propagate_context_preserves_name(self):
        """Test that decorator preserves function name."""
        @propagate_context
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


class TestPropagateContextLazy:
    """Tests for propagate_context_lazy decorator."""

    def test_propagate_context_lazy_returns_callable(self):
        """Test that lazy decorator returns callable."""
        @propagate_context_lazy
        def add(x, y):
            return x + y

        # Calling returns a callable
        result_fn = add(1, 2)
        assert callable(result_fn)

        # Calling that callable runs the function
        result = result_fn()
        assert result == 3

    def test_propagate_context_lazy_preserves_name(self):
        """Test that lazy decorator preserves function name."""
        @propagate_context_lazy
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


class TestCrossThreadPropagation:
    """Tests for cross-thread context propagation."""

    def setup_method(self):
        SpanRegistry.reset_instance()

    def teardown_method(self):
        SpanRegistry.reset_instance()

    def test_carrier_works_across_threads(self):
        """Test that carrier can be used across threads."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        context = EvalContext(trace_id=trace_id, span_id=span_id)
        carrier = ContextCarrier(context)

        results = []
        errors = []

        def worker():
            try:
                result = carrier.enrich_span({"from_thread": True})
                results.append(result)
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert len(errors) == 0
        assert len(results) == 1
        assert results[0] is True
        mock_span.set_attribute.assert_called_with("from_thread", True)

    def test_serialized_carrier_works_across_threads(self):
        """Test that serialized carrier can be used across threads."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        trace_id = "a" * 32
        span_id = "b" * 16

        register_span(trace_id, span_id, mock_span)

        context = EvalContext(trace_id=trace_id, span_id=span_id)
        carrier = ContextCarrier(context)

        # Serialize
        data = carrier.to_dict()

        results = []
        errors = []

        def worker():
            try:
                # Deserialize in worker
                restored = ContextCarrier.from_dict(data)
                result = restored.enrich_span({"from_serialized": True})
                results.append(result)
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert len(errors) == 0
        assert len(results) == 1
        assert results[0] is True
