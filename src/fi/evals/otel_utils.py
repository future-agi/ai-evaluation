import logging
from typing import Optional

try:
    import opentelemetry.trace as otel_trace_api
except ImportError as e:
    raise ImportError(
        "Could not import opentelemetry.trace. Please install 'fi-instrumentation-otel' to use tracing features."
    ) from e


def _get_current_otel_span() -> Optional[otel_trace_api.Span]:
    current_span = otel_trace_api.get_current_span()

    if current_span is otel_trace_api.INVALID_SPAN:
        logging.warning(
            "Context error: No active span in current context. Operations that depend on an active span will be skipped. "
            "Ensure spans are created with start_as_current_span() or that you're operating within an active span context."
        )
        return None

    return current_span
        