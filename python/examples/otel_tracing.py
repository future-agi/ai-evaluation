#!/usr/bin/env python3
"""
OpenTelemetry Tracing Examples

This file demonstrates various ways to use the fi.evals.otel module
for LLM observability. Export anywhere, view anywhere.

Examples:
1. Basic setup with console output
2. Production setup with OTLP export
3. Multi-backend export
4. Automatic instrumentation
5. Manual span creation
6. Cost tracking
7. Evaluation integration
8. RAG application tracing

Run these examples with:
    python examples/otel_tracing.py
"""

import os
from typing import List, Dict, Any


def example_1_basic_setup():
    """
    Example 1: Basic Setup

    The simplest way to get started - console output for development.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Setup (Console Output)")
    print("="*60)

    from fi.evals.otel import setup_tracing, TraceConfig

    # Development config with console output
    config = TraceConfig.development(service_name="example-service")
    setup_tracing(config)

    print("Tracing initialized with console exporter")
    print("All LLM calls will now be traced to console")


def example_2_production_setup():
    """
    Example 2: Production Setup

    Configure for production with OTLP export, sampling, and PII redaction.
    """
    print("\n" + "="*60)
    print("Example 2: Production Setup (OTLP Export)")
    print("="*60)

    from fi.evals.otel import (
        setup_tracing,
        TraceConfig,
        ExporterType,
        SamplingStrategy,
    )

    config = TraceConfig.production(
        service_name="production-llm-service",
        otlp_endpoint="localhost:4317",  # Your OTLP collector
        service_version="1.2.3",
        eval_sample_rate=0.1,  # Evaluate 10% of spans
    )

    # Show configuration
    print(f"Service: {config.service_name}")
    print(f"Version: {config.service_version}")
    print(f"Environment: {config.deployment_environment}")
    print(f"Sampling: {config.sampling_strategy.value} ({config.sampling_ratio})")
    print(f"Evaluation sample rate: {config.evaluation.sample_rate}")
    print(f"PII redaction: {config.content.redact_pii}")

    # Note: Uncomment to actually initialize (requires OTLP endpoint)
    # setup_tracing(config)


def example_3_multi_backend():
    """
    Example 3: Multi-Backend Export

    Export traces to multiple backends simultaneously.
    Useful for migration or redundancy.
    """
    print("\n" + "="*60)
    print("Example 3: Multi-Backend Export")
    print("="*60)

    from fi.evals.otel import TraceConfig, ExporterType

    config = TraceConfig.multi_backend(
        service_name="multi-export-service",
        backends=[
            # OTLP to Grafana Tempo
            {"type": "otlp_grpc", "endpoint": "tempo:4317"},
            # Jaeger for local development
            {"type": "jaeger", "endpoint": "jaeger:6831"},
            # Console for debugging
            {"type": "console"},
        ],
    )

    print(f"Configured {len(config.exporters)} exporters:")
    for exp in config.exporters:
        print(f"  - {exp.type.value}: {exp.endpoint or 'stdout'}")


def example_4_automatic_instrumentation():
    """
    Example 4: Automatic Instrumentation

    Automatically trace OpenAI and Anthropic calls with zero code changes.
    """
    print("\n" + "="*60)
    print("Example 4: Automatic Instrumentation")
    print("="*60)

    from fi.evals.otel import (
        OpenAIInstrumentor,
        AnthropicInstrumentor,
        instrument_all,
        get_instrumented_libraries,
    )

    # Method 1: Instrument specific libraries
    openai_inst = OpenAIInstrumentor(
        capture_prompts=True,
        capture_completions=True,
        capture_streaming=True,
    )
    print(f"OpenAI instrumentor created: {openai_inst.library_name}")

    # Method 2: Instrument all available
    # instrumented = instrument_all()
    # print(f"Instrumented: {instrumented}")

    print("\nWith instrumentation, this code:")
    print("""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    """)
    print("...will automatically create spans with:")
    print("  - gen_ai.system: openai")
    print("  - gen_ai.request.model: gpt-4")
    print("  - gen_ai.usage.input_tokens: ...")
    print("  - gen_ai.completion.0.content: ...")


def example_5_manual_spans():
    """
    Example 5: Manual Span Creation

    Create custom spans for fine-grained control.
    """
    print("\n" + "="*60)
    print("Example 5: Manual Span Creation")
    print("="*60)

    from fi.evals.otel import (
        get_tracer,
        trace_llm_call,
        GenAIAttributes,
        create_llm_span_attributes,
    )

    # Method 1: trace_llm_call context manager
    print("\nMethod 1: Using trace_llm_call context manager")
    print("""
    with trace_llm_call("chat", model="gpt-4", system="openai") as span:
        response = call_llm()
        span.set_attribute("gen_ai.usage.input_tokens", 100)
    """)

    # Method 2: Using tracer directly
    print("\nMethod 2: Using tracer directly")
    tracer = get_tracer("my-app")
    print(f"Got tracer: {tracer}")

    # Method 3: Helper function for attributes
    print("\nMethod 3: Using create_llm_span_attributes helper")
    attrs = create_llm_span_attributes(
        system="openai",
        model="gpt-4",
        operation="chat",
        input_tokens=100,
        output_tokens=50,
        temperature=0.7,
    )
    print(f"Created attributes: {len(attrs)} keys")
    for key, value in attrs.items():
        print(f"  {key}: {value}")


def example_6_cost_tracking():
    """
    Example 6: Cost Tracking

    Track and calculate LLM costs across providers.
    """
    print("\n" + "="*60)
    print("Example 6: Cost Tracking")
    print("="*60)

    from fi.evals.otel import (
        calculate_cost,
        CostSpanProcessor,
        TokenPricing,
        DEFAULT_PRICING,
    )

    # Calculate costs for different models
    models_to_test = [
        ("gpt-4", 1000, 500),
        ("gpt-4o-mini", 1000, 500),
        ("claude-3-opus-20240229", 1000, 500),
        ("claude-3-haiku-20240307", 1000, 500),
    ]

    print("\nCost comparison (1000 input, 500 output tokens):")
    print("-" * 50)

    for model, input_tokens, output_tokens in models_to_test:
        cost = calculate_cost(model, input_tokens, output_tokens)
        if cost["total_cost"] > 0:
            print(f"{model}:")
            print(f"  Input:  ${cost['input_cost']:.6f}")
            print(f"  Output: ${cost['output_cost']:.6f}")
            print(f"  Total:  ${cost['total_cost']:.6f}")

    # Custom pricing
    print("\nCustom pricing for fine-tuned model:")
    custom = {"my-model": TokenPricing("my-model", 0.05, 0.10)}
    cost = calculate_cost("my-model", 1000, 1000, custom)
    print(f"  Total: ${cost['total_cost']:.4f}")

    # Cost processor with alerts
    print("\nCost processor with alerting:")
    alert_count = [0]

    def on_alert(cost, span_id):
        alert_count[0] += 1
        print(f"  ALERT: Cost ${cost:.4f} exceeded threshold!")

    processor = CostSpanProcessor(
        alert_threshold_usd=0.01,
        on_cost_alert=on_alert,
    )
    print(f"  Alert threshold: ${processor._alert_threshold_usd}")


def example_7_evaluation_integration():
    """
    Example 7: Evaluation Integration

    Automatically evaluate LLM outputs and attach scores to spans.
    """
    print("\n" + "="*60)
    print("Example 7: Evaluation Integration")
    print("="*60)

    from fi.evals.otel import (
        EvaluationSpanProcessor,
        BatchEvaluationProcessor,
        EvaluationAttributes,
        EvaluationResult,
    )

    # Standard evaluation processor
    processor = EvaluationSpanProcessor(
        metrics=["relevance", "coherence", "faithfulness"],
        sample_rate=0.1,  # Evaluate 10% of spans
        async_evaluation=True,  # Non-blocking
        cache_enabled=True,
        cache_ttl_seconds=3600,
    )

    print("Evaluation processor configured:")
    print(f"  Metrics: {processor._metrics}")
    print(f"  Sample rate: {processor._sample_rate}")
    print(f"  Async: {processor._async_evaluation}")
    print(f"  Cache TTL: {processor._cache_ttl_seconds}s")

    # Batch processor for efficiency
    batch_processor = BatchEvaluationProcessor(
        batch_size=10,
        batch_timeout_ms=1000,
        metrics=["relevance"],
    )
    print(f"\nBatch processor:")
    print(f"  Batch size: {batch_processor._batch_size}")

    # Evaluation attributes
    print("\nEvaluation attribute format:")
    print(f"  Score: {EvaluationAttributes.score('relevance')}")
    print(f"  Reason: {EvaluationAttributes.reason('relevance')}")
    print(f"  Latency: {EvaluationAttributes.latency('relevance')}")


def example_8_rag_application():
    """
    Example 8: RAG Application Tracing

    Comprehensive example for a RAG (Retrieval-Augmented Generation) app.
    """
    print("\n" + "="*60)
    print("Example 8: RAG Application Tracing")
    print("="*60)

    from fi.evals.otel import (
        TraceConfig,
        GenAIAttributes,
        RAGAttributes,
        SpanNames,
        create_llm_span_attributes,
    )

    print("RAG tracing would include spans for:")
    print(f"  - Retrieval: {SpanNames.RAG_RETRIEVE}")
    print(f"  - Reranking: {SpanNames.RAG_RERANK}")
    print(f"  - Generation: {SpanNames.RAG_GENERATE}")
    print(f"  - Full pipeline: {SpanNames.RAG_PIPELINE}")

    print("\nRAG-specific attributes:")
    print(f"  - {RAGAttributes.NUM_DOCUMENTS}: Number of retrieved docs")
    print(f"  - {RAGAttributes.CONTEXT_LENGTH}: Total context length")
    print(f"  - {RAGAttributes.TOP_K}: Retrieval top-k")
    print(f"  - {RAGAttributes.document_content(0)}: First doc content")
    print(f"  - {RAGAttributes.document_score(0)}: First doc score")

    print("\nExample RAG tracing code:")
    print("""
    with tracer.start_as_current_span(SpanNames.RAG_PIPELINE) as pipeline:
        # Retrieval
        with tracer.start_as_current_span(SpanNames.RAG_RETRIEVE) as retrieve:
            retrieve.set_attribute(RAGAttributes.TOP_K, 10)
            docs = retriever.search(query)
            retrieve.set_attribute(RAGAttributes.NUM_DOCUMENTS, len(docs))

        # Generation (auto-traced by instrumentor)
        response = client.chat.completions.create(...)

        pipeline.set_attribute(RAGAttributes.CONTEXT_LENGTH, total_context_len)
    """)


def example_9_semantic_conventions():
    """
    Example 9: Semantic Conventions Reference

    Overview of all available semantic conventions.
    """
    print("\n" + "="*60)
    print("Example 9: Semantic Conventions Reference")
    print("="*60)

    from fi.evals.otel import (
        GenAIAttributes,
        LLMCostAttributes,
        EvaluationAttributes,
        RAGAttributes,
        GuardrailAttributes,
        SpanNames,
        normalize_system_name,
        SYSTEM_OPENAI,
        SYSTEM_ANTHROPIC,
        OPERATION_CHAT,
        FINISH_STOP,
    )

    print("\nGenAI Attributes (OTEL Standard):")
    attrs = [
        GenAIAttributes.SYSTEM,
        GenAIAttributes.OPERATION_NAME,
        GenAIAttributes.REQUEST_MODEL,
        GenAIAttributes.USAGE_INPUT_TOKENS,
        GenAIAttributes.USAGE_OUTPUT_TOKENS,
        GenAIAttributes.REQUEST_TEMPERATURE,
        GenAIAttributes.RESPONSE_FINISH_REASON,
    ]
    for attr in attrs:
        print(f"  {attr}")

    print("\nSystem name normalization:")
    systems = ["openai", "gpt-4", "claude-3", "gemini", "llama-2"]
    for s in systems:
        print(f"  '{s}' -> '{normalize_system_name(s)}'")

    print("\nPredefined constants:")
    print(f"  SYSTEM_OPENAI: {SYSTEM_OPENAI}")
    print(f"  SYSTEM_ANTHROPIC: {SYSTEM_ANTHROPIC}")
    print(f"  OPERATION_CHAT: {OPERATION_CHAT}")
    print(f"  FINISH_STOP: {FINISH_STOP}")


def main():
    """Run all examples."""
    examples = [
        example_1_basic_setup,
        example_2_production_setup,
        example_3_multi_backend,
        example_4_automatic_instrumentation,
        example_5_manual_spans,
        example_6_cost_tracking,
        example_7_evaluation_integration,
        example_8_rag_application,
        example_9_semantic_conventions,
    ]

    print("="*60)
    print("OpenTelemetry Integration Examples")
    print("Export anywhere. View anywhere.")
    print("="*60)

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
