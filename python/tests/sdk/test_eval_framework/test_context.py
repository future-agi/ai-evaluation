"""Tests for fi.evals.framework.context module."""

import pytest
from fi.evals.framework.context import (
    EvalContext,
    get_current_context,
    create_standalone_context,
)


class TestEvalContext:
    """Tests for EvalContext dataclass."""

    def test_create_basic_context(self):
        """Test creating a basic context."""
        ctx = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
        )

        assert ctx.trace_id == "a" * 32
        assert ctx.span_id == "b" * 16
        assert ctx.parent_span_id is None
        assert ctx.baggage == {}

    def test_auto_generate_ids(self):
        """Test that empty IDs are auto-generated."""
        ctx = EvalContext(trace_id="", span_id="")

        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16

    def test_eval_run_id_generated(self):
        """Test that eval_run_id is auto-generated."""
        ctx1 = EvalContext(trace_id="a" * 32, span_id="b" * 16)
        ctx2 = EvalContext(trace_id="a" * 32, span_id="b" * 16)

        assert len(ctx1.eval_run_id) == 16
        assert ctx1.eval_run_id != ctx2.eval_run_id

    def test_from_current_span_no_otel(self):
        """Test from_current_span when OTEL not available."""
        ctx = EvalContext.from_current_span()

        # Should create standalone context
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16

    def test_from_headers_valid(self):
        """Test extracting context from valid headers."""
        headers = {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "baggage": "key1=value1,key2=value2",
        }
        ctx = EvalContext.from_headers(headers)

        assert ctx.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert ctx.span_id == "00f067aa0ba902b7"
        assert ctx.baggage == {"key1": "value1", "key2": "value2"}

    def test_from_headers_empty(self):
        """Test extracting context from empty headers."""
        ctx = EvalContext.from_headers({})

        # Should create new IDs
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16

    def test_from_headers_invalid_traceparent(self):
        """Test handling invalid traceparent."""
        headers = {"traceparent": "invalid"}
        ctx = EvalContext.from_headers(headers)

        # Should create new IDs
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16

    def test_from_headers_baggage_parsing(self):
        """Test baggage parsing edge cases."""
        headers = {
            "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01",
            "baggage": "key1=value1, key2 = value2 ,key3=",
        }
        ctx = EvalContext.from_headers(headers)

        assert ctx.baggage.get("key1") == "value1"
        assert ctx.baggage.get("key2") == "value2"
        assert ctx.baggage.get("key3") == ""

    def test_to_headers(self):
        """Test converting to headers."""
        ctx = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            baggage={"foo": "bar"},
            eval_run_id="run123",
        )
        headers = ctx.to_headers()

        assert headers["traceparent"] == f"00-{'a' * 32}-{'b' * 16}-01"
        assert headers["baggage"] == "foo=bar"
        assert headers["x-eval-run-id"] == "run123"

    def test_to_headers_no_baggage(self):
        """Test headers without baggage."""
        ctx = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
        )
        headers = ctx.to_headers()

        assert "baggage" not in headers

    def test_roundtrip_headers(self):
        """Test headers roundtrip."""
        original = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            baggage={"key": "value"},
        )
        headers = original.to_headers()
        restored = EvalContext.from_headers(headers)

        assert restored.trace_id == original.trace_id
        assert restored.span_id == original.span_id
        assert restored.baggage == original.baggage

    def test_to_dict(self):
        """Test serialization to dict."""
        ctx = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            parent_span_id="c" * 16,
            baggage={"k": "v"},
        )
        data = ctx.to_dict()

        assert data["trace_id"] == "a" * 32
        assert data["span_id"] == "b" * 16
        assert data["parent_span_id"] == "c" * 16
        assert data["baggage"] == {"k": "v"}

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "trace_id": "a" * 32,
            "span_id": "b" * 16,
            "parent_span_id": "c" * 16,
            "baggage": {"k": "v"},
            "eval_run_id": "run456",
        }
        ctx = EvalContext.from_dict(data)

        assert ctx.trace_id == "a" * 32
        assert ctx.span_id == "b" * 16
        assert ctx.parent_span_id == "c" * 16
        assert ctx.baggage == {"k": "v"}
        assert ctx.eval_run_id == "run456"

    def test_roundtrip_dict(self):
        """Test dict roundtrip."""
        original = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            parent_span_id="c" * 16,
            baggage={"key": "value"},
        )
        data = original.to_dict()
        restored = EvalContext.from_dict(data)

        assert restored.trace_id == original.trace_id
        assert restored.span_id == original.span_id
        assert restored.parent_span_id == original.parent_span_id
        assert restored.baggage == original.baggage
        assert restored.eval_run_id == original.eval_run_id

    def test_with_baggage(self):
        """Test adding baggage."""
        ctx = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            baggage={"existing": "value"},
        )
        new_ctx = ctx.with_baggage("new_key", "new_value")

        # Original unchanged
        assert "new_key" not in ctx.baggage

        # New context has both
        assert new_ctx.baggage["existing"] == "value"
        assert new_ctx.baggage["new_key"] == "new_value"
        assert new_ctx.trace_id == ctx.trace_id
        assert new_ctx.span_id == ctx.span_id

    def test_child_context(self):
        """Test creating child context."""
        parent = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            baggage={"inherited": "value"},
        )
        child = parent.child_context()

        # Same trace
        assert child.trace_id == parent.trace_id

        # New span
        assert child.span_id != parent.span_id
        assert len(child.span_id) == 16

        # Parent tracked
        assert child.parent_span_id == parent.span_id

        # Baggage inherited
        assert child.baggage == parent.baggage

    def test_is_valid_true(self):
        """Test is_valid with valid context."""
        ctx = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
        )
        assert ctx.is_valid is True

    def test_is_valid_false_zero_trace(self):
        """Test is_valid with zero trace ID."""
        ctx = EvalContext(
            trace_id="0" * 32,
            span_id="b" * 16,
        )
        assert ctx.is_valid is False

    def test_is_valid_false_zero_span(self):
        """Test is_valid with zero span ID."""
        ctx = EvalContext(
            trace_id="a" * 32,
            span_id="0" * 16,
        )
        assert ctx.is_valid is False

    def test_is_valid_false_wrong_length(self):
        """Test is_valid with wrong ID lengths."""
        ctx = EvalContext(
            trace_id="short",
            span_id="alsoshort",
        )
        # Short IDs are invalid
        assert ctx.is_valid is False

    def test_str_representation(self):
        """Test string representation."""
        ctx = EvalContext(
            trace_id="abcd1234" + "0" * 24,
            span_id="efgh5678" + "0" * 8,
        )
        s = str(ctx)
        assert "abcd1234" in s
        assert "efgh5678" in s

    def test_repr_representation(self):
        """Test repr representation."""
        ctx = EvalContext(
            trace_id="a" * 32,
            span_id="b" * 16,
        )
        r = repr(ctx)
        assert "EvalContext" in r
        assert "trace_id" in r
        assert "span_id" in r


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_current_context(self):
        """Test get_current_context function."""
        ctx = get_current_context()

        assert isinstance(ctx, EvalContext)
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16

    def test_create_standalone_context(self):
        """Test create_standalone_context function."""
        ctx = create_standalone_context()

        assert isinstance(ctx, EvalContext)
        assert ctx.is_valid

    def test_create_standalone_context_with_baggage(self):
        """Test create_standalone_context with baggage."""
        ctx = create_standalone_context(
            user_id="123",
            environment="test",
        )

        assert ctx.baggage["user_id"] == "123"
        assert ctx.baggage["environment"] == "test"

    def test_create_standalone_contexts_unique(self):
        """Test that standalone contexts have unique IDs."""
        ctx1 = create_standalone_context()
        ctx2 = create_standalone_context()

        assert ctx1.trace_id != ctx2.trace_id
        assert ctx1.span_id != ctx2.span_id
        assert ctx1.eval_run_id != ctx2.eval_run_id
