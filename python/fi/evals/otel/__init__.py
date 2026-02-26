"""
OpenTelemetry Integration for LLM Observability.

Export anywhere. View anywhere.

This module provides comprehensive OpenTelemetry integration for LLM
applications, following the GenAI semantic conventions. Traces can be
exported to any OTEL-compatible backend including:

- Jaeger, Zipkin, Grafana Tempo
- Datadog, Honeycomb, New Relic
- Arize, Langfuse, Phoenix
- Any OTLP-compatible endpoint

Features:
- Automatic LLM span enrichment
- Evaluation score attachment
- Cost tracking
- Multi-backend export
- Sampling strategies

Quick Start:
    # Simple setup with console output
    from fi.evals.otel import setup_tracing
    setup_tracing(service_name="my-ai-service")

    # Production setup with OTLP export
    from fi.evals.otel import setup_tracing, TraceConfig, ExporterType

    setup_tracing(
        service_name="my-ai-service",
        otlp_endpoint="localhost:4317",
    )

    # Multi-backend setup
    config = TraceConfig.multi_backend(
        service_name="my-service",
        backends=[
            {"type": "otlp_grpc", "endpoint": "localhost:4317"},
            {"type": "console"},
        ],
    )
    setup_tracing(config)

Manual Span Creation:
    from fi.evals.otel import trace_llm_call

    with trace_llm_call("chat", model="gpt-4", system="openai") as span:
        response = client.chat.completions.create(...)
        span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)

Environment Variables:
    OTEL_SERVICE_NAME - Service name (default: llm-service)
    OTEL_EXPORTER_OTLP_ENDPOINT - OTLP endpoint
    OTEL_EXPORTER_OTLP_HEADERS - Headers as key=value,key=value
    OTEL_DEPLOYMENT_ENVIRONMENT - Environment (default: development)
"""

# Core setup functions
from .tracer import (
    setup_tracing,
    get_tracer,
    get_current_span,
    is_tracing_enabled,
    trace_llm_call,
    shutdown_tracing,
)

# Configuration
from .config import (
    TraceConfig,
    ExporterConfig,
    ProcessorConfig,
    EvaluationConfig,
    CostConfig,
    ContentConfig,
    ResourceConfig,
    SamplingStrategy,
    EXPORTER_PRESETS,
    get_exporter_preset,
)

# Types
from .types import (
    ExporterType,
    SpanKind,
    ProcessorType,
    TokenPricing,
    SpanAttributes,
    EvaluationResult,
    ExportResult,
    TraceContext,
)

# Semantic conventions
from .conventions import (
    GenAIAttributes,
    CostAttributes,
    LLMCostAttributes,
    EvaluationAttributes,
    RAGAttributes,
    GuardrailAttributes,
    SpanNames,
    normalize_system_name,
    create_llm_span_attributes,
    create_evaluation_attributes,
    # System constants
    SYSTEM_OPENAI,
    SYSTEM_ANTHROPIC,
    SYSTEM_COHERE,
    SYSTEM_GOOGLE,
    SYSTEM_MISTRAL,
    SYSTEM_META,
    SYSTEM_AWS_BEDROCK,
    SYSTEM_AZURE_OPENAI,
    # Operation constants
    OPERATION_CHAT,
    OPERATION_COMPLETION,
    OPERATION_EMBEDDING,
    # Finish reasons
    FINISH_STOP,
    FINISH_LENGTH,
    FINISH_TOOL_CALLS,
)

# Processors
from .processors import (
    OTEL_AVAILABLE,
    BaseSpanProcessor,
    FilteringSpanProcessor,
    CompositeSpanProcessor,
    LLMSpanProcessor,
    EvaluationSpanProcessor,
    BatchEvaluationProcessor,
    CostSpanProcessor,
    calculate_cost,
    DEFAULT_PRICING,
)

# Instrumentors
from .instrumentors import (
    BaseInstrumentor,
    InstrumentorManager,
    OpenAIInstrumentor,
    AnthropicInstrumentor,
    instrument_all,
    uninstrument_all,
    instrument,
    uninstrument,
    is_instrumented,
    get_instrumented_libraries,
)

# Auto-enrichment (evals automatically add to spans)
from .enrichment import (
    enable_auto_enrichment,
    disable_auto_enrichment,
    is_auto_enrichment_enabled,
    enrich_span_with_evaluation,
    enrich_span_with_eval_result,
    enrich_span_with_batch_result,
    EvaluationSpanContext,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",

    # Core setup
    "setup_tracing",
    "get_tracer",
    "get_current_span",
    "is_tracing_enabled",
    "trace_llm_call",
    "shutdown_tracing",

    # Configuration
    "TraceConfig",
    "ExporterConfig",
    "ProcessorConfig",
    "EvaluationConfig",
    "CostConfig",
    "ContentConfig",
    "ResourceConfig",
    "SamplingStrategy",
    "EXPORTER_PRESETS",
    "get_exporter_preset",

    # Types
    "ExporterType",
    "SpanKind",
    "ProcessorType",
    "TokenPricing",
    "SpanAttributes",
    "EvaluationResult",
    "ExportResult",
    "TraceContext",
    "OTEL_AVAILABLE",

    # Semantic conventions
    "GenAIAttributes",
    "CostAttributes",
    "LLMCostAttributes",
    "EvaluationAttributes",
    "RAGAttributes",
    "GuardrailAttributes",
    "SpanNames",
    "normalize_system_name",
    "create_llm_span_attributes",
    "create_evaluation_attributes",

    # System constants
    "SYSTEM_OPENAI",
    "SYSTEM_ANTHROPIC",
    "SYSTEM_COHERE",
    "SYSTEM_GOOGLE",
    "SYSTEM_MISTRAL",
    "SYSTEM_META",
    "SYSTEM_AWS_BEDROCK",
    "SYSTEM_AZURE_OPENAI",

    # Operation constants
    "OPERATION_CHAT",
    "OPERATION_COMPLETION",
    "OPERATION_EMBEDDING",

    # Finish reasons
    "FINISH_STOP",
    "FINISH_LENGTH",
    "FINISH_TOOL_CALLS",

    # Processors
    "BaseSpanProcessor",
    "FilteringSpanProcessor",
    "CompositeSpanProcessor",
    "LLMSpanProcessor",
    "EvaluationSpanProcessor",
    "BatchEvaluationProcessor",
    "CostSpanProcessor",
    "calculate_cost",
    "DEFAULT_PRICING",

    # Instrumentors
    "BaseInstrumentor",
    "InstrumentorManager",
    "OpenAIInstrumentor",
    "AnthropicInstrumentor",
    "instrument_all",
    "uninstrument_all",
    "instrument",
    "uninstrument",
    "is_instrumented",
    "get_instrumented_libraries",

    # Auto-enrichment
    "enable_auto_enrichment",
    "disable_auto_enrichment",
    "is_auto_enrichment_enabled",
    "enrich_span_with_evaluation",
    "enrich_span_with_eval_result",
    "enrich_span_with_batch_result",
    "EvaluationSpanContext",
]
