# OpenTelemetry Integration

> **Language Support:** Python ✅ | TypeScript ✅ | Go 📋 | Java 📋

**Export anywhere. View anywhere.**

The `fi.evals.otel` module provides comprehensive OpenTelemetry integration for LLM applications, following the GenAI semantic conventions. Traces can be exported to any OTEL-compatible backend.

## Automatic Span Enrichment

**Evaluation data automatically flows into spans.** When you run any evaluation through `fi.evals`, the results are automatically added to the current active span (if one exists). This happens by default with zero configuration.

```python
from fi.evals.otel import setup_tracing, trace_llm_call
from fi.evals import Evaluator

# Setup tracing
setup_tracing(service_name="my-app")
evaluator = Evaluator()

# When you run an eval inside an active span, results are auto-attached
with trace_llm_call("my-operation", model="gpt-4") as span:
    # Run evaluation
    result = evaluator.evaluate(
        eval_templates="Relevance",
        inputs={"response": "The capital of France is Paris."}
    )
    # The span now automatically has:
    # - eval.Relevance = 0.95
    # - eval.Relevance.reason = "Response directly answers the query"
    # - eval.Relevance.latency_ms = 245.3
```

### Controlling Auto-Enrichment

```python
from fi.evals.otel import (
    enable_auto_enrichment,
    disable_auto_enrichment,
    is_auto_enrichment_enabled,
)

# Check if enabled (default: True)
print(is_auto_enrichment_enabled())  # True

# Disable if needed
disable_auto_enrichment()

# Re-enable
enable_auto_enrichment()
```

### Manual Enrichment

For more control, you can manually enrich spans:

```python
from fi.evals.otel import enrich_span_with_evaluation

# Add evaluation to current span
enrich_span_with_evaluation(
    metric_name="custom_metric",
    score=0.85,
    reason="High quality response",
    latency_ms=100.0,
)

# Or enrich a specific span
enrich_span_with_evaluation(
    metric_name="relevance",
    score=0.9,
    span=my_span,  # Specific span to enrich
)
```

## Quick Start

### Basic Setup

```python
from fi.evals.otel import setup_tracing

# Simple setup with console output (development)
setup_tracing(service_name="my-ai-service")

# Production setup with OTLP endpoint
setup_tracing(
    service_name="my-ai-service",
    otlp_endpoint="localhost:4317",
)
```

### Environment Variables

The module respects standard OTEL environment variables:

```bash
export OTEL_SERVICE_NAME=my-ai-service
export OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317
export OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer token
export OTEL_DEPLOYMENT_ENVIRONMENT=production
```

Then just call:

```python
from fi.evals.otel import setup_tracing
setup_tracing()  # Picks up config from environment
```

## Configuration

### TraceConfig

Full configuration is available through the `TraceConfig` class:

```python
from fi.evals.otel import (
    TraceConfig,
    ExporterConfig,
    ExporterType,
    SamplingStrategy,
)

config = TraceConfig(
    service_name="my-ai-service",
    service_version="1.0.0",
    deployment_environment="production",

    # Multiple exporters - export to multiple backends
    exporters=[
        ExporterConfig(type=ExporterType.OTLP_GRPC, endpoint="localhost:4317"),
        ExporterConfig(type=ExporterType.CONSOLE),
    ],

    # Sampling - reduce volume in production
    sampling_strategy=SamplingStrategy.RATIO,
    sampling_ratio=0.5,  # Sample 50% of traces

    # Content capture
    content=ContentConfig(
        capture_prompts=True,
        capture_completions=True,
        redact_pii=True,
    ),
)

setup_tracing(config)
```

### Factory Methods

Convenient factory methods for common configurations:

```python
from fi.evals.otel import TraceConfig

# Development: console output, all spans sampled
config = TraceConfig.development(service_name="my-service")

# Production: OTLP export, PII redaction, sampling
config = TraceConfig.production(
    service_name="my-service",
    otlp_endpoint="otlp.example.com:4317",
    eval_sample_rate=0.1,
)

# Multi-backend: export to multiple destinations
config = TraceConfig.multi_backend(
    service_name="my-service",
    backends=[
        {"type": "otlp_grpc", "endpoint": "localhost:4317"},
        {"type": "jaeger", "endpoint": "localhost:6831"},
        {"type": "console"},
    ],
)
```

### Method Chaining

Build configuration fluently:

```python
config = (
    TraceConfig(service_name="my-service")
    .add_exporter(ExporterType.OTLP_GRPC, endpoint="localhost:4317")
    .with_evaluation(metrics=["relevance", "coherence"], sample_rate=0.1)
    .with_cost_tracking(alert_threshold=1.0)
)
```

## Supported Backends

### OTLP (gRPC & HTTP)

Standard OpenTelemetry Protocol - works with most observability platforms:

```python
# gRPC (default, more efficient)
ExporterConfig(type=ExporterType.OTLP_GRPC, endpoint="localhost:4317")

# HTTP (works through proxies)
ExporterConfig(type=ExporterType.OTLP_HTTP, endpoint="http://localhost:4318/v1/traces")
```

### Jaeger

```python
from fi.evals.otel import get_exporter_preset

jaeger = get_exporter_preset("jaeger")
# Or manually:
ExporterConfig(type=ExporterType.JAEGER, endpoint="localhost:6831")
```

### Zipkin

```python
ExporterConfig(
    type=ExporterType.ZIPKIN,
    endpoint="http://localhost:9411/api/v2/spans",
)
```

### LLM Observability Platforms

#### Arize

```python
ExporterConfig(
    type=ExporterType.ARIZE,
    endpoint="https://otlp.arize.com",
    headers={
        "space_key": "YOUR_SPACE_KEY",
        "api_key": "YOUR_API_KEY",
    },
)
```

#### Langfuse

```python
ExporterConfig(
    type=ExporterType.LANGFUSE,
    endpoint="https://cloud.langfuse.com",
    headers={
        "x-langfuse-public-key": "pk-...",
        "x-langfuse-secret-key": "sk-...",
    },
)
```

#### Phoenix (Arize)

```python
ExporterConfig(
    type=ExporterType.PHOENIX,
    endpoint="http://localhost:6006/v1/traces",
)
```

## Automatic Instrumentation

Automatically trace LLM calls with zero code changes:

### OpenAI

```python
from fi.evals.otel import setup_tracing, OpenAIInstrumentor

# Setup tracing
setup_tracing(service_name="my-service")

# Instrument OpenAI
OpenAIInstrumentor().instrument()

# All OpenAI calls are now traced!
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Span automatically created with:
# - gen_ai.system: openai
# - gen_ai.request.model: gpt-4
# - gen_ai.usage.input_tokens: ...
# - gen_ai.usage.output_tokens: ...
# - gen_ai.prompt.0.content: Hello!
# - gen_ai.completion.0.content: ...
```

### Anthropic

```python
from fi.evals.otel import setup_tracing, AnthropicInstrumentor

setup_tracing(service_name="my-service")
AnthropicInstrumentor().instrument()

from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Automatic tracing!
```

### Instrument All Available

```python
from fi.evals.otel import setup_tracing, instrument_all

setup_tracing(service_name="my-service")

# Instrument all available libraries (OpenAI, Anthropic, etc.)
instrumented = instrument_all()
print(f"Instrumented: {instrumented}")  # ['openai', 'anthropic']
```

## Manual Span Creation

For custom tracing or when automatic instrumentation isn't enough:

### Context Manager

```python
from fi.evals.otel import trace_llm_call

with trace_llm_call("chat", model="gpt-4", system="openai") as span:
    response = client.chat.completions.create(...)

    # Add attributes
    span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
```

### Using Tracer Directly

```python
from fi.evals.otel import get_tracer, GenAIAttributes

tracer = get_tracer("my-module")

with tracer.start_as_current_span("my-llm-operation") as span:
    span.set_attribute(GenAIAttributes.SYSTEM, "openai")
    span.set_attribute(GenAIAttributes.REQUEST_MODEL, "gpt-4")

    # Your LLM call here
    result = call_llm()

    span.set_attribute(GenAIAttributes.USAGE_INPUT_TOKENS, result.input_tokens)
    span.set_attribute(GenAIAttributes.USAGE_OUTPUT_TOKENS, result.output_tokens)
```

## Span Processors

Enrich spans with additional data using processors.

### LLM Processor

Normalizes and enriches LLM span attributes:

```python
from fi.evals.otel import LLMSpanProcessor

processor = LLMSpanProcessor(
    capture_prompts=True,
    capture_completions=True,
    max_content_length=10000,
    redact_patterns=[
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"sk-[a-zA-Z0-9]{48}",  # API keys
    ],
)
```

### Evaluation Processor

Automatically evaluate LLM outputs:

```python
from fi.evals.otel import EvaluationSpanProcessor

processor = EvaluationSpanProcessor(
    metrics=["relevance", "coherence", "faithfulness"],
    sample_rate=0.1,  # Evaluate 10% of spans
    async_evaluation=True,  # Don't block
    cache_enabled=True,  # Cache results
    timeout_ms=5000,
)
```

### Cost Processor

Track LLM costs:

```python
from fi.evals.otel import CostSpanProcessor, TokenPricing

processor = CostSpanProcessor(
    custom_pricing={
        "my-fine-tuned-model": TokenPricing("my-model", 0.01, 0.02),
    },
    alert_threshold_usd=1.0,
    on_cost_alert=lambda cost, span_id: print(f"Cost alert: ${cost}"),
)

# Get cost summary
print(processor.get_summary())
# {
#     "total_cost_usd": 0.45,
#     "total_calls": 100,
#     "average_cost_per_call": 0.0045,
# }
```

### Composite Processor

Combine multiple processors:

```python
from fi.evals.otel import CompositeSpanProcessor

processor = CompositeSpanProcessor([
    LLMSpanProcessor(),
    EvaluationSpanProcessor(metrics=["relevance"]),
    CostSpanProcessor(),
], parallel=True)  # Run in parallel for efficiency
```

## Semantic Conventions

The module follows OpenTelemetry GenAI semantic conventions:

### Core Attributes

```python
from fi.evals.otel import GenAIAttributes

# System identification
GenAIAttributes.SYSTEM  # "gen_ai.system" - e.g., "openai", "anthropic"
GenAIAttributes.OPERATION_NAME  # "gen_ai.operation.name" - e.g., "chat"

# Model
GenAIAttributes.REQUEST_MODEL  # "gen_ai.request.model"
GenAIAttributes.RESPONSE_MODEL  # "gen_ai.response.model"

# Token usage
GenAIAttributes.USAGE_INPUT_TOKENS  # "gen_ai.usage.input_tokens"
GenAIAttributes.USAGE_OUTPUT_TOKENS  # "gen_ai.usage.output_tokens"
GenAIAttributes.USAGE_TOTAL_TOKENS  # "gen_ai.usage.total_tokens"

# Request parameters
GenAIAttributes.REQUEST_TEMPERATURE  # "gen_ai.request.temperature"
GenAIAttributes.REQUEST_MAX_TOKENS  # "gen_ai.request.max_tokens"

# Content (indexed for multi-turn)
GenAIAttributes.prompt_content(0)  # "gen_ai.prompt.0.content"
GenAIAttributes.completion_content(0)  # "gen_ai.completion.0.content"
```

### Cost Attributes

```python
from fi.evals.otel import LLMCostAttributes

LLMCostAttributes.INPUT_COST_USD  # "llm.cost.input_usd"
LLMCostAttributes.OUTPUT_COST_USD  # "llm.cost.output_usd"
LLMCostAttributes.TOTAL_COST_USD  # "llm.cost.total_usd"
```

### Evaluation Attributes

```python
from fi.evals.otel import EvaluationAttributes

EvaluationAttributes.score("relevance")  # "eval.relevance"
EvaluationAttributes.reason("relevance")  # "eval.relevance.reason"
EvaluationAttributes.latency("relevance")  # "eval.relevance.latency_ms"
```

## Cost Calculation

Calculate LLM costs programmatically:

```python
from fi.evals.otel import calculate_cost, TokenPricing

# Standard model pricing
cost = calculate_cost(
    model="gpt-4",
    input_tokens=1000,
    output_tokens=500,
)
print(f"Total: ${cost['total_cost']:.4f}")  # Total: $0.0600

# Custom pricing
custom_pricing = {
    "my-model": TokenPricing("my-model", 0.01, 0.02),
}
cost = calculate_cost("my-model", 1000, 1000, custom_pricing)
```

### Supported Models

Built-in pricing for:
- OpenAI: GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo
- Anthropic: Claude 3 Opus, Sonnet, Haiku
- Google: Gemini 1.5 Pro, Flash
- Mistral: Large, Medium, Small
- Cohere: Command R+, Command R
- Embeddings: text-embedding-3-large, ada-002

## Real-World Examples

### RAG Application with Full Observability

```python
from fi.evals.otel import (
    setup_tracing,
    TraceConfig,
    ExporterType,
    instrument_all,
    get_tracer,
    GenAIAttributes,
    RAGAttributes,
)

# Configure tracing
config = TraceConfig.production(
    service_name="rag-app",
    otlp_endpoint="localhost:4317",
    eval_sample_rate=0.2,
).with_cost_tracking(alert_threshold=10.0)

setup_tracing(config)
instrument_all()

tracer = get_tracer("rag-app")

def rag_query(question: str, context: list[str]) -> str:
    with tracer.start_as_current_span("rag.query") as span:
        # Add RAG attributes
        span.set_attribute(RAGAttributes.NUM_DOCUMENTS, len(context))
        span.set_attribute(RAGAttributes.CONTEXT_LENGTH, sum(len(c) for c in context))

        # Retrieval is automatically traced
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "\n".join(context)},
                {"role": "user", "content": question}
            ]
        )

        return response.choices[0].message.content
```

### Multi-Model Comparison

```python
from fi.evals.otel import setup_tracing, get_tracer, GenAIAttributes

setup_tracing(service_name="model-comparison")
tracer = get_tracer("comparison")

def compare_models(prompt: str, models: list[str]):
    with tracer.start_as_current_span("model.comparison") as parent:
        parent.set_attribute("comparison.prompt", prompt)
        parent.set_attribute("comparison.models", ",".join(models))

        results = {}
        for model in models:
            with tracer.start_as_current_span(f"model.{model}") as span:
                span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)

                # Call appropriate client based on model
                response = call_model(model, prompt)
                results[model] = response

        return results
```

### Streaming with Cost Tracking

```python
from fi.evals.otel import (
    setup_tracing,
    OpenAIInstrumentor,
    CostSpanProcessor,
)

# Setup with cost tracking
setup_tracing(service_name="streaming-app")

cost_processor = CostSpanProcessor(
    alert_threshold_usd=0.1,
    on_cost_calculated=lambda span_id, i, o, t: print(f"Cost: ${t:.4f}")
)

# Instrument OpenAI (handles streaming automatically)
OpenAIInstrumentor(capture_streaming=True).instrument()

# Streaming responses are automatically traced
from openai import OpenAI
client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Viewing Traces

### Jaeger

1. Run Jaeger: `docker run -d -p 16686:16686 -p 6831:6831/udp jaegertracing/all-in-one`
2. Configure exporter: `ExporterConfig(type=ExporterType.JAEGER, endpoint="localhost:6831")`
3. View at: http://localhost:16686

### Grafana Tempo

1. Configure exporter: `ExporterConfig(type=ExporterType.OTLP_GRPC, endpoint="localhost:4317")`
2. Use Grafana to query Tempo datasource

### Console

For debugging, export to console:

```python
config = TraceConfig.development(service_name="debug")
setup_tracing(config)
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `setup_tracing(config)` | Initialize OTEL tracing |
| `get_tracer(name)` | Get a tracer instance |
| `get_current_span()` | Get the current active span |
| `trace_llm_call(name, ...)` | Context manager for LLM spans |
| `shutdown_tracing()` | Shutdown and flush traces |
| `is_tracing_enabled()` | Check if tracing is active |

### Instrumentors

| Function | Description |
|----------|-------------|
| `instrument_all()` | Instrument all available libraries |
| `uninstrument_all()` | Remove all instrumentation |
| `instrument(library)` | Instrument specific library |
| `is_instrumented(library)` | Check instrumentation status |

### Cost Functions

| Function | Description |
|----------|-------------|
| `calculate_cost(model, input, output)` | Calculate LLM call cost |
| `DEFAULT_PRICING` | Built-in model pricing |

## Tests

Run the test suite:

```bash
pytest tests/sdk/test_otel.py -v
```

## Dependencies

Required:
- `opentelemetry-sdk>=1.20.0`
- `opentelemetry-api>=1.20.0`

Optional (for specific exporters):
- `opentelemetry-exporter-otlp>=1.20.0` - OTLP export
- `opentelemetry-exporter-jaeger>=1.20.0` - Jaeger export
- `opentelemetry-exporter-zipkin>=1.20.0` - Zipkin export
