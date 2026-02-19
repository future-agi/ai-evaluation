"""
Span registry for tracking active spans.

This module provides SpanRegistry, which tracks active OTEL spans by their
trace/span IDs. This enables background threads and distributed workers to
add attributes to spans that were created in a different execution context.

The registry uses weak references so spans can still be garbage collected
when they go out of scope in the original code.
"""

import threading
import weakref
from typing import Dict, Optional, Any, Tuple, Set
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field


@dataclass
class SpanEntry:
    """Entry in the span registry."""
    span_ref: weakref.ref
    registered_at: datetime
    trace_id: str
    span_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpanRegistry:
    """
    Registry for tracking active spans by trace/span ID.

    This allows background workers to add attributes to spans created elsewhere.
    Uses weak references so spans can be garbage collected normally.

    Thread-safe implementation using a lock.

    Example:
        # In main thread - register span
        span = tracer.start_span("llm_call")
        register_span(trace_id, span_id, span)

        # In background thread - enrich span
        span = get_span(trace_id, span_id)
        if span:
            span.set_attribute("eval.score", 0.95)

    Note:
        This is a singleton - all registrations go to the same global registry.
    """

    _instance: Optional["SpanRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SpanRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._spans: Dict[Tuple[str, str], SpanEntry] = {}
                    instance._registry_lock = threading.Lock()
                    instance._cleanup_interval = timedelta(minutes=5)
                    instance._max_age = timedelta(minutes=30)
                    instance._last_cleanup = datetime.now(timezone.utc)
                    cls._instance = instance
        return cls._instance

    def register(
        self,
        trace_id: str,
        span_id: str,
        span: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a span for later enrichment.

        Args:
            trace_id: The trace ID (32-char hex string)
            span_id: The span ID (16-char hex string)
            span: The span object (stored as weak reference)
            metadata: Optional metadata to store with the entry
        """
        key = (trace_id, span_id)

        # Create weak reference with callback for auto-cleanup
        def on_span_collected(ref):
            self._remove_entry(key)

        entry = SpanEntry(
            span_ref=weakref.ref(span, on_span_collected),
            registered_at=datetime.now(timezone.utc),
            trace_id=trace_id,
            span_id=span_id,
            metadata=metadata or {},
        )

        with self._registry_lock:
            self._spans[key] = entry
            self._maybe_cleanup()

    def get(self, trace_id: str, span_id: str) -> Optional[Any]:
        """
        Get a registered span.

        Args:
            trace_id: The trace ID
            span_id: The span ID

        Returns:
            The span object, or None if not found or garbage collected
        """
        key = (trace_id, span_id)

        with self._registry_lock:
            entry = self._spans.get(key)
            if entry is None:
                return None

            span = entry.span_ref()
            if span is None:
                # Span was garbage collected, clean up
                del self._spans[key]
                return None

            return span

    def unregister(self, trace_id: str, span_id: str) -> bool:
        """
        Remove a span from the registry.

        Args:
            trace_id: The trace ID
            span_id: The span ID

        Returns:
            True if span was removed, False if not found
        """
        key = (trace_id, span_id)

        with self._registry_lock:
            if key in self._spans:
                del self._spans[key]
                return True
            return False

    def _remove_entry(self, key: Tuple[str, str]) -> None:
        """Remove an entry (called from weak reference callback)."""
        with self._registry_lock:
            self._spans.pop(key, None)

    def _maybe_cleanup(self) -> None:
        """
        Clean up old entries.

        Called with lock held. Removes entries older than max_age.
        """
        now = datetime.now(timezone.utc)
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        cutoff = now - self._max_age

        # Find old keys
        old_keys = [
            key for key, entry in self._spans.items()
            if entry.registered_at < cutoff
        ]

        # Remove them
        for key in old_keys:
            del self._spans[key]

    def contains(self, trace_id: str, span_id: str) -> bool:
        """
        Check if a span is registered.

        Args:
            trace_id: The trace ID
            span_id: The span ID

        Returns:
            True if registered and not garbage collected
        """
        return self.get(trace_id, span_id) is not None

    def get_metadata(self, trace_id: str, span_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a registered span.

        Args:
            trace_id: The trace ID
            span_id: The span ID

        Returns:
            Metadata dict or None if not found
        """
        key = (trace_id, span_id)

        with self._registry_lock:
            entry = self._spans.get(key)
            if entry is None:
                return None
            return dict(entry.metadata)

    def update_metadata(
        self,
        trace_id: str,
        span_id: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Update metadata for a registered span.

        Args:
            trace_id: The trace ID
            span_id: The span ID
            metadata: Metadata to merge

        Returns:
            True if updated, False if span not found
        """
        key = (trace_id, span_id)

        with self._registry_lock:
            entry = self._spans.get(key)
            if entry is None:
                return False
            entry.metadata.update(metadata)
            return True

    def list_spans(self) -> Set[Tuple[str, str]]:
        """
        List all registered span keys.

        Returns:
            Set of (trace_id, span_id) tuples
        """
        with self._registry_lock:
            return set(self._spans.keys())

    def count(self) -> int:
        """
        Get count of registered spans.

        Returns:
            Number of registered spans
        """
        with self._registry_lock:
            return len(self._spans)

    def clear(self) -> int:
        """
        Clear all registered spans.

        Returns:
            Number of spans cleared
        """
        with self._registry_lock:
            count = len(self._spans)
            self._spans.clear()
            return count

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.

        Mainly for testing purposes.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._spans.clear()
            cls._instance = None


# Global singleton instance
_registry = SpanRegistry()


def register_span(
    trace_id: str,
    span_id: str,
    span: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Register a span in the global registry.

    Args:
        trace_id: The trace ID
        span_id: The span ID
        span: The span object
        metadata: Optional metadata
    """
    _registry.register(trace_id, span_id, span, metadata)


def get_span(trace_id: str, span_id: str) -> Optional[Any]:
    """
    Get a span from the global registry.

    Args:
        trace_id: The trace ID
        span_id: The span ID

    Returns:
        The span or None
    """
    return _registry.get(trace_id, span_id)


def unregister_span(trace_id: str, span_id: str) -> bool:
    """
    Remove a span from the global registry.

    Args:
        trace_id: The trace ID
        span_id: The span ID

    Returns:
        True if removed
    """
    return _registry.unregister(trace_id, span_id)


def get_registry() -> SpanRegistry:
    """
    Get the global SpanRegistry instance.

    Returns:
        The singleton SpanRegistry
    """
    return _registry


def register_current_span(metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Register the current OTEL span in the registry.

    Convenience function that extracts trace/span IDs from the current span
    and registers it.

    Args:
        metadata: Optional metadata to store

    Returns:
        True if registered, False if no current span
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span or not span.is_recording():
            return False

        ctx = span.get_span_context()
        if not ctx.is_valid:
            return False

        trace_id = format(ctx.trace_id, '032x')
        span_id = format(ctx.span_id, '016x')

        register_span(trace_id, span_id, span, metadata)
        return True
    except ImportError:
        return False
    except Exception:
        return False
