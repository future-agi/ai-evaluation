"""
Evaluation context for trace propagation.

This module provides EvalContext, which captures and propagates trace context
across thread and process boundaries. This is essential for:
- Background threads adding attributes to original spans
- Distributed workers maintaining trace continuity
- Async evaluations enriching parent spans
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import uuid


@dataclass
class EvalContext:
    """
    Context passed to all evaluations.

    Captures trace/span IDs for propagation across threads/processes.
    Follows W3C Trace Context specification for interoperability.

    Attributes:
        trace_id: 32-character hex trace ID
        span_id: 16-character hex span ID
        parent_span_id: Optional parent span ID
        baggage: Key-value pairs propagated with the trace
        eval_run_id: Unique ID for this evaluation run

    Example:
        # Capture from current span
        ctx = EvalContext.from_current_span()

        # Serialize for propagation
        headers = ctx.to_headers()

        # Reconstruct in worker
        ctx = EvalContext.from_headers(headers)
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    eval_run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def __post_init__(self):
        """Validate context fields."""
        if not self.trace_id:
            self.trace_id = uuid.uuid4().hex
        if not self.span_id:
            self.span_id = uuid.uuid4().hex[:16]

    @classmethod
    def from_current_span(cls) -> "EvalContext":
        """
        Capture context from current OTEL span.

        If OTEL is not available or no span is active, creates a standalone context.

        Returns:
            EvalContext captured from current span or newly created
        """
        try:
            from opentelemetry import trace
            from opentelemetry import baggage as otel_baggage

            span = trace.get_current_span()
            ctx = span.get_span_context()

            if ctx.is_valid:
                return cls(
                    trace_id=format(ctx.trace_id, '032x'),
                    span_id=format(ctx.span_id, '016x'),
                    parent_span_id=None,
                    baggage=dict(otel_baggage.get_all()),
                )
        except ImportError:
            pass
        except Exception:
            pass

        # No OTEL or invalid context - create standalone
        return cls(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
        )

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "EvalContext":
        """
        Extract context from W3C Trace Context headers.

        Parses traceparent and baggage headers according to the W3C spec:
        https://www.w3.org/TR/trace-context/

        Args:
            headers: Dict containing traceparent and optionally baggage headers

        Returns:
            EvalContext reconstructed from headers
        """
        traceparent = headers.get("traceparent", "")
        baggage_str = headers.get("baggage", "")
        eval_run_id = headers.get("x-eval-run-id", uuid.uuid4().hex[:16])

        # Parse traceparent: version-trace_id-span_id-flags
        # Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
        trace_id = uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]

        parts = traceparent.split("-")
        if len(parts) >= 3:
            # Validate version (should be "00")
            if len(parts[1]) == 32:
                trace_id = parts[1]
            if len(parts[2]) == 16:
                span_id = parts[2]

        # Parse baggage: key1=value1,key2=value2
        baggage = {}
        if baggage_str:
            for item in baggage_str.split(","):
                item = item.strip()
                if "=" in item:
                    k, v = item.split("=", 1)
                    baggage[k.strip()] = v.strip()

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            baggage=baggage,
            eval_run_id=eval_run_id,
        )

    def to_headers(self) -> Dict[str, str]:
        """
        Convert to W3C Trace Context headers for propagation.

        Returns:
            Dict with traceparent, baggage, and x-eval-run-id headers
        """
        headers = {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-01",
            "x-eval-run-id": self.eval_run_id,
        }
        if self.baggage:
            headers["baggage"] = ",".join(f"{k}={v}" for k, v in self.baggage.items())
        return headers

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage/transmission."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
            "eval_run_id": self.eval_run_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalContext":
        """Deserialize from dict."""
        return cls(
            trace_id=data.get("trace_id", uuid.uuid4().hex),
            span_id=data.get("span_id", uuid.uuid4().hex[:16]),
            parent_span_id=data.get("parent_span_id"),
            baggage=data.get("baggage", {}),
            eval_run_id=data.get("eval_run_id", uuid.uuid4().hex[:16]),
        )

    def with_baggage(self, key: str, value: str) -> "EvalContext":
        """
        Create a new context with additional baggage.

        Args:
            key: Baggage key
            value: Baggage value

        Returns:
            New EvalContext with the added baggage
        """
        new_baggage = dict(self.baggage)
        new_baggage[key] = value
        return EvalContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            baggage=new_baggage,
            eval_run_id=self.eval_run_id,
        )

    def child_context(self) -> "EvalContext":
        """
        Create a child context with new span_id.

        The current span becomes the parent span.

        Returns:
            New EvalContext representing a child span
        """
        return EvalContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self.span_id,
            baggage=dict(self.baggage),
            eval_run_id=self.eval_run_id,
        )

    @property
    def is_valid(self) -> bool:
        """Check if context has valid trace and span IDs."""
        return (
            len(self.trace_id) == 32 and
            len(self.span_id) == 16 and
            self.trace_id != "0" * 32 and
            self.span_id != "0" * 16
        )

    def __str__(self) -> str:
        return f"EvalContext(trace={self.trace_id[:8]}..., span={self.span_id})"

    def __repr__(self) -> str:
        return (
            f"EvalContext(trace_id='{self.trace_id}', span_id='{self.span_id}', "
            f"eval_run_id='{self.eval_run_id}')"
        )


def get_current_context() -> EvalContext:
    """
    Get EvalContext from current span.

    Convenience function that wraps EvalContext.from_current_span().

    Returns:
        EvalContext captured from current span or newly created
    """
    return EvalContext.from_current_span()


def create_standalone_context(**baggage) -> EvalContext:
    """
    Create a standalone context not linked to any span.

    Useful for testing or when running outside of a traced context.

    Args:
        **baggage: Key-value pairs to include as baggage

    Returns:
        New standalone EvalContext
    """
    return EvalContext(
        trace_id=uuid.uuid4().hex,
        span_id=uuid.uuid4().hex[:16],
        baggage=baggage,
    )
