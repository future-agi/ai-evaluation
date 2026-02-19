"""
Span enrichment utilities.

This module provides utilities for adding evaluation results to OTEL spans.
Works with the current span or spans identified by trace context.
"""

from typing import Dict, Any, Optional, Union
from .context import EvalContext

# Type alias for span attribute values
SpanValue = Union[str, int, float, bool, None]


def enrich_current_span(
    eval_name: str,
    attributes: Dict[str, Any],
    prefix: str = "eval",
) -> bool:
    """
    Add attributes to current span.

    Args:
        eval_name: Name of evaluation (used as prefix)
        attributes: Attributes to add (will be flattened and prefixed)
        prefix: Prefix for all attributes (default: "eval")

    Returns:
        True if enrichment succeeded, False if no span or OTEL not available

    Example:
        enrich_current_span("faithfulness", {"score": 0.95, "passed": True})
        # Adds: eval.faithfulness.score = 0.95
        #       eval.faithfulness.passed = True
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span or not span.is_recording():
            return False

        full_prefix = f"{prefix}.{eval_name}"
        for key, value in attributes.items():
            if value is not None and _is_valid_span_value(value):
                span.set_attribute(f"{full_prefix}.{key}", value)

        return True
    except ImportError:
        return False
    except Exception:
        return False


def enrich_span(
    span: Any,
    eval_name: str,
    attributes: Dict[str, Any],
    prefix: str = "eval",
) -> bool:
    """
    Add attributes to a specific span.

    Args:
        span: The OTEL span to enrich
        eval_name: Name of evaluation
        attributes: Attributes to add
        prefix: Prefix for all attributes

    Returns:
        True if enrichment succeeded
    """
    if span is None:
        return False

    try:
        if not span.is_recording():
            return False

        full_prefix = f"{prefix}.{eval_name}"
        for key, value in attributes.items():
            if value is not None and _is_valid_span_value(value):
                span.set_attribute(f"{full_prefix}.{key}", value)

        return True
    except Exception:
        return False


def add_eval_event(
    eval_name: str,
    event_name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Add an event to the current span for evaluation tracking.

    Events are useful for tracking evaluation lifecycle:
    - eval.started
    - eval.completed
    - eval.failed

    Args:
        eval_name: Name of evaluation
        event_name: Event name (e.g., "started", "completed")
        attributes: Optional event attributes

    Returns:
        True if event was added
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span or not span.is_recording():
            return False

        event_attrs = {"eval.name": eval_name}
        if attributes:
            for key, value in attributes.items():
                if value is not None and _is_valid_span_value(value):
                    event_attrs[key] = value

        span.add_event(f"eval.{event_name}", attributes=event_attrs)
        return True
    except ImportError:
        return False
    except Exception:
        return False


def get_current_span() -> Optional[Any]:
    """
    Get the current OTEL span if available.

    Returns:
        Current span or None if OTEL not available or no active span
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.is_recording():
            return span
        return None
    except ImportError:
        return None


def is_span_recording() -> bool:
    """
    Check if there's an active recording span.

    Returns:
        True if there's an active span that's recording
    """
    span = get_current_span()
    return span is not None


def get_current_trace_context() -> Optional[EvalContext]:
    """
    Get EvalContext from current span.

    Returns:
        EvalContext if span available, None otherwise
    """
    try:
        return EvalContext.from_current_span()
    except Exception:
        return None


def _is_valid_span_value(value: Any) -> bool:
    """
    Check if a value is valid for span attributes.

    OTEL span attributes only support:
    - str, int, float, bool
    - Sequences of the above (but we'll skip those for simplicity)
    """
    return isinstance(value, (str, int, float, bool))


def flatten_attributes(
    data: Dict[str, Any],
    prefix: str = "",
    separator: str = ".",
) -> Dict[str, SpanValue]:
    """
    Flatten nested dict into dot-separated keys with valid span values.

    Args:
        data: Nested dictionary
        prefix: Prefix for all keys
        separator: Separator between nested keys

    Returns:
        Flat dict with only valid span values

    Example:
        flatten_attributes({"a": {"b": 1, "c": "x"}})
        # Returns: {"a.b": 1, "a.c": "x"}
    """
    result = {}

    for key, value in data.items():
        full_key = f"{prefix}{separator}{key}" if prefix else key

        if isinstance(value, dict):
            # Recurse for nested dicts
            nested = flatten_attributes(value, full_key, separator)
            result.update(nested)
        elif _is_valid_span_value(value):
            result[full_key] = value
        # Skip invalid types (lists, objects, etc.)

    return result


class SpanEnricher:
    """
    Context manager for enriching spans with evaluation results.

    Automatically adds timing and status attributes.

    Example:
        with SpanEnricher("faithfulness") as enricher:
            result = run_evaluation()
            enricher.set_result({"score": result.score})

        # Span now has:
        # - eval.faithfulness.score
        # - eval.faithfulness.latency_ms
        # - eval.faithfulness.status
    """

    def __init__(
        self,
        eval_name: str,
        eval_version: str = "1.0.0",
        prefix: str = "eval",
    ):
        self.eval_name = eval_name
        self.eval_version = eval_version
        self.prefix = prefix
        self._start_time: Optional[float] = None
        self._result_set = False

    def __enter__(self) -> "SpanEnricher":
        import time
        self._start_time = time.perf_counter()

        # Add start event
        add_eval_event(self.eval_name, "started", {
            "eval.version": self.eval_version,
        })

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time

        latency_ms = 0.0
        if self._start_time is not None:
            latency_ms = (time.perf_counter() - self._start_time) * 1000

        # Add completion attributes
        status = "failed" if exc_type else "completed"

        enrich_current_span(self.eval_name, {
            "latency_ms": latency_ms,
            "status": status,
            "version": self.eval_version,
        }, prefix=self.prefix)

        if exc_type:
            add_eval_event(self.eval_name, "failed", {
                "error": str(exc_val) if exc_val else "Unknown error",
            })
        else:
            add_eval_event(self.eval_name, "completed")

    def set_result(self, attributes: Dict[str, Any]) -> bool:
        """
        Set evaluation result attributes.

        Args:
            attributes: Result attributes to add to span

        Returns:
            True if attributes were added
        """
        self._result_set = True
        return enrich_current_span(self.eval_name, attributes, prefix=self.prefix)

    def set_error(self, error: str) -> bool:
        """
        Set error attribute.

        Args:
            error: Error message

        Returns:
            True if attribute was added
        """
        return enrich_current_span(self.eval_name, {"error": error}, prefix=self.prefix)
