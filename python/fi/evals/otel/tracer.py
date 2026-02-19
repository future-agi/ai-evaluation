"""
OpenTelemetry Tracer Setup.

Factory functions and utilities for setting up OpenTelemetry
tracing with LLM-specific processors and exporters.
"""

from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
import logging
import os

from .config import (
    TraceConfig,
    ExporterConfig,
    ProcessorConfig,
    SamplingStrategy,
)
from .types import ExporterType, ProcessorType
from .processors import (
    OTEL_AVAILABLE,
    CompositeSpanProcessor,
    LLMSpanProcessor,
    EvaluationSpanProcessor,
    CostSpanProcessor,
)

logger = logging.getLogger(__name__)

# Global tracer provider instance
_tracer_provider: Optional[Any] = None
_is_initialized = False


def setup_tracing(
    config: Optional[TraceConfig] = None,
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Set up OpenTelemetry tracing for LLM applications.

    This is the main entry point for configuring OTEL tracing.
    It creates exporters, processors, and a tracer provider.

    Args:
        config: Full TraceConfig, or None to use defaults/env vars
        service_name: Service name (overrides config)
        otlp_endpoint: OTLP endpoint (overrides config)
        **kwargs: Additional arguments passed to TraceConfig

    Returns:
        The configured TracerProvider

    Example:
        # Simple setup with environment variables
        setup_tracing(service_name="my-llm-service")

        # Full configuration
        from fi.evals.otel import TraceConfig, ExporterConfig, ExporterType

        config = TraceConfig(
            service_name="my-service",
            exporters=[
                ExporterConfig(type=ExporterType.OTLP_GRPC, endpoint="localhost:4317"),
                ExporterConfig(type=ExporterType.CONSOLE),
            ],
        )
        setup_tracing(config)
    """
    global _tracer_provider, _is_initialized

    if not OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry SDK not available. Install with: "
            "pip install opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        return None

    # Build configuration
    if config is None:
        config = _build_config_from_env(service_name, otlp_endpoint, **kwargs)
    else:
        # Override with explicit args
        if service_name:
            config.service_name = service_name
        if otlp_endpoint:
            config.exporters = [ExporterConfig(
                type=ExporterType.OTLP_GRPC,
                endpoint=otlp_endpoint,
            )]

    if not config.enabled:
        logger.info("Tracing is disabled")
        return None

    # Import OTEL components
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME

    # Create resource
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        "service.version": config.service_version or "unknown",
        "deployment.environment": config.deployment_environment,
        **config.resource.attributes,
    })

    # Create tracer provider
    provider = TracerProvider(
        resource=resource,
        sampler=_create_sampler(config),
    )

    # Add exporters
    for exporter_config in config.exporters:
        exporter = _create_exporter(exporter_config)
        if exporter:
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            provider.add_span_processor(BatchSpanProcessor(exporter))

    # Add custom processors
    custom_processors = _create_custom_processors(config)
    if custom_processors:
        provider.add_span_processor(custom_processors)

    # Set as global provider
    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    _is_initialized = True

    logger.info(
        f"OpenTelemetry tracing initialized for '{config.service_name}' "
        f"with {len(config.exporters)} exporter(s)"
    )

    return provider


def _build_config_from_env(
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    **kwargs,
) -> TraceConfig:
    """Build TraceConfig from environment variables."""
    # Service name
    service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "llm-service")

    # OTLP endpoint
    endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    # Build exporters
    exporters = []
    if endpoint:
        exporters.append(ExporterConfig(
            type=ExporterType.OTLP_GRPC,
            endpoint=endpoint,
            headers=_parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")),
        ))
    else:
        # Default to console
        exporters.append(ExporterConfig(type=ExporterType.CONSOLE))

    return TraceConfig(
        service_name=service_name,
        exporters=exporters,
        deployment_environment=os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "development"),
        **kwargs,
    )


def _parse_headers(headers_str: str) -> Dict[str, str]:
    """Parse headers from comma-separated key=value string."""
    if not headers_str:
        return {}

    headers = {}
    for pair in headers_str.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()
    return headers


def _create_sampler(config: TraceConfig) -> Any:
    """Create a sampler based on configuration."""
    from opentelemetry.sdk.trace.sampling import (
        ALWAYS_ON,
        ALWAYS_OFF,
        TraceIdRatioBased,
        ParentBased,
    )

    if config.sampling_strategy == SamplingStrategy.ALWAYS_ON:
        return ALWAYS_ON
    elif config.sampling_strategy == SamplingStrategy.ALWAYS_OFF:
        return ALWAYS_OFF
    elif config.sampling_strategy == SamplingStrategy.RATIO:
        return TraceIdRatioBased(config.sampling_ratio)
    elif config.sampling_strategy == SamplingStrategy.PARENT_BASED:
        return ParentBased(TraceIdRatioBased(config.sampling_ratio))
    else:
        return ALWAYS_ON


def _create_exporter(config: ExporterConfig) -> Optional[Any]:
    """Create an exporter based on configuration."""
    try:
        if config.type == ExporterType.CONSOLE:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            return ConsoleSpanExporter()

        elif config.type == ExporterType.OTLP_GRPC:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            return OTLPSpanExporter(
                endpoint=config.endpoint or "localhost:4317",
                headers=config.headers or None,
                insecure=config.insecure,
                timeout=config.timeout_ms // 1000,
            )

        elif config.type == ExporterType.OTLP_HTTP:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            return OTLPSpanExporter(
                endpoint=config.endpoint or "http://localhost:4318/v1/traces",
                headers=config.headers or None,
                timeout=config.timeout_ms // 1000,
            )

        elif config.type == ExporterType.JAEGER:
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                return JaegerExporter(
                    agent_host_name=config.endpoint.split(":")[0] if config.endpoint else "localhost",
                    agent_port=int(config.endpoint.split(":")[1]) if config.endpoint and ":" in config.endpoint else 6831,
                )
            except ImportError:
                logger.warning("Jaeger exporter not available. Install opentelemetry-exporter-jaeger")
                return None

        elif config.type == ExporterType.ZIPKIN:
            try:
                from opentelemetry.exporter.zipkin.json import ZipkinExporter
                return ZipkinExporter(
                    endpoint=config.endpoint or "http://localhost:9411/api/v2/spans",
                )
            except ImportError:
                logger.warning("Zipkin exporter not available. Install opentelemetry-exporter-zipkin")
                return None

        elif config.type in (ExporterType.ARIZE, ExporterType.PHOENIX, ExporterType.LANGFUSE):
            # These platforms support OTLP
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            return OTLPSpanExporter(
                endpoint=config.endpoint,
                headers=config.headers or None,
            )

        elif config.type == ExporterType.FUTUREAGI:
            # Use OTLP HTTP for FutureAGI backend
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            return OTLPSpanExporter(
                endpoint=config.endpoint or "https://api.futureagi.com/v1/traces",
                headers=config.headers or None,
            )

        else:
            logger.warning(f"Unsupported exporter type: {config.type}")
            return None

    except ImportError as e:
        logger.warning(f"Failed to create exporter {config.type}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error creating exporter {config.type}: {e}")
        return None


def _create_custom_processors(config: TraceConfig) -> Optional[CompositeSpanProcessor]:
    """Create custom span processors based on configuration."""
    processors = []

    # Check for processor configs
    processor_types = {p.type for p in config.processors if p.enabled}

    # LLM processor
    if ProcessorType.LLM in processor_types:
        processors.append(LLMSpanProcessor(
            capture_prompts=config.content.capture_prompts,
            capture_completions=config.content.capture_completions,
            max_content_length=config.content.max_content_length,
            redact_patterns=config.content.redact_patterns,
        ))

    # Evaluation processor
    if config.evaluation.enabled:
        processors.append(EvaluationSpanProcessor(
            metrics=config.evaluation.metrics,
            sample_rate=config.evaluation.sample_rate,
            async_evaluation=config.evaluation.async_evaluation,
            timeout_ms=config.evaluation.timeout_ms,
            cache_enabled=config.evaluation.cache_enabled,
            cache_ttl_seconds=config.evaluation.cache_ttl_seconds,
            evaluator_model=config.evaluation.evaluator_model,
        ))

    # Cost processor
    if config.cost.enabled:
        processors.append(CostSpanProcessor(
            custom_pricing=None,  # Could extend config for this
            alert_threshold_usd=config.cost.alert_threshold_usd,
        ))

    if not processors:
        return None

    return CompositeSpanProcessor(processors)


def get_tracer(name: str = "fi.evals") -> Any:
    """
    Get a tracer instance.

    Args:
        name: Tracer name (typically module name)

    Returns:
        Tracer instance
    """
    if not OTEL_AVAILABLE:
        return _NoOpTracer()

    from opentelemetry import trace
    return trace.get_tracer(name)


def get_current_span() -> Any:
    """Get the current active span."""
    if not OTEL_AVAILABLE:
        return None

    from opentelemetry import trace
    return trace.get_current_span()


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled and initialized."""
    return _is_initialized and _tracer_provider is not None


@contextmanager
def trace_llm_call(
    name: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    **attributes,
):
    """
    Context manager for tracing an LLM call.

    Creates a span with LLM-specific attributes.

    Args:
        name: Span name
        model: Model name
        system: Provider system (openai, anthropic, etc.)
        **attributes: Additional span attributes

    Yields:
        The active span

    Example:
        with trace_llm_call("chat", model="gpt-4", system="openai") as span:
            response = client.chat.completions.create(...)
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
    """
    from .conventions import GenAIAttributes, OPERATION_CHAT

    tracer = get_tracer()

    # Build initial attributes
    span_attrs = {
        GenAIAttributes.OPERATION_NAME: OPERATION_CHAT,
    }
    if model:
        span_attrs[GenAIAttributes.REQUEST_MODEL] = model
    if system:
        span_attrs[GenAIAttributes.SYSTEM] = system
    span_attrs.update(attributes)

    with tracer.start_as_current_span(name, attributes=span_attrs) as span:
        yield span


def shutdown_tracing() -> None:
    """Shutdown the tracer provider and flush pending spans."""
    global _tracer_provider, _is_initialized

    if _tracer_provider:
        try:
            _tracer_provider.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down tracer: {e}")

    _tracer_provider = None
    _is_initialized = False


class _NoOpTracer:
    """No-op tracer for when OTEL is not available."""

    def start_span(self, name: str, **kwargs):
        return _NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs):
        return _NoOpContextManager()


class _NoOpSpan:
    """No-op span."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def end(self, end_time: Optional[int] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass


class _NoOpContextManager:
    """No-op context manager."""

    def __enter__(self):
        return _NoOpSpan()

    def __exit__(self, *args):
        pass


__all__ = [
    "setup_tracing",
    "get_tracer",
    "get_current_span",
    "is_tracing_enabled",
    "trace_llm_call",
    "shutdown_tracing",
]
