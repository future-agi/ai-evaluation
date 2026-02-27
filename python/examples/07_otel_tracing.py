#!/usr/bin/env python3
"""
Cookbook 07 — See Every LLM Call in Your Observability Stack

SCENARIO:
    Your team runs 10,000 LLM calls per day across multiple services.
    When something goes wrong — a hallucination, a slow response, a
    safety violation — you need to trace it back to the exact call.

    This cookbook shows how to wire fi-evals into your OpenTelemetry
    stack so that every LLM call gets:
    - A trace with input/output/tokens/latency
    - Quality scores (faithfulness, toxicity) attached as span attributes
    - Auto-instrumentation for OpenAI/Anthropic SDKs
    - Export to Jaeger, Datadog, Grafana, or your custom backend

Usage:
    cd python && uv run python -m examples.07_otel_tracing
"""

import time

from fi.evals import evaluate
from fi.evals.otel import (
    setup_tracing,
    trace_llm_call,
    get_tracer,
    is_tracing_enabled,
    enable_auto_enrichment,
    enrich_span_with_evaluation,
    EvaluationSpanContext,
    TraceConfig,
    ExporterConfig,
    ExporterType,
    shutdown_tracing,
)


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ── Scenario 1: Basic setup — see traces in your terminal ───────
divider("SCENARIO 1: Console tracing (see spans in your terminal)")

setup_tracing(service_name="medical-chatbot-v2")

print(f"Tracing enabled: {is_tracing_enabled()}")
print("All spans will be printed to console.\n")


# ── Scenario 2: Trace an LLM call ───────────────────────────────
divider("SCENARIO 2: Trace a simulated LLM call")
print("Each LLM call becomes a span with input/output/token attributes.\n")

# Simulate an LLM call with tracing
with trace_llm_call("chat", model="gemini-2.5-flash", system="google") as span:
    # In production, this would be: client.chat.completions.create(...)
    prompt = "What is the recommended dosage for ibuprofen?"
    response = "Take 200-400mg every 4-6 hours as needed for pain."
    input_tokens = len(prompt.split()) * 2  # rough estimate
    output_tokens = len(response.split()) * 2

    # Record the call details
    span.set_attribute("gen_ai.prompt.0.content", prompt)
    span.set_attribute("gen_ai.completion.0.content", response)
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    print(f"Prompt:  {prompt}")
    print(f"Response: {response}")
    print(f"Tokens:  {input_tokens} in / {output_tokens} out")


# ── Scenario 3: Attach quality scores to spans ──────────────────
divider("SCENARIO 3: Attach quality scores to the trace")
print("Run metrics and attach results as span attributes.\n")

with trace_llm_call("chat", model="gemini-2.5-flash", system="google") as span:
    response = "Take 200-400mg ibuprofen every 4-6 hours for pain."
    context = "Ibuprofen: 200-400mg q4-6h PRN. Maximum 1200mg/day."

    # Attach the generation details
    span.set_attribute("gen_ai.completion.0.content", response)

    # Run faithfulness check and attach to the span
    result = evaluate("faithfulness", output=response, context=context)

    enrich_span_with_evaluation(
        metric_name="faithfulness",
        score=result.score,
        reason=result.reason[:200],
        latency_ms=result.latency_ms,
        span=span,
    )
    print(f"Faithfulness: {result.score:.2f} (attached to span)")

    # Run another check
    result = evaluate("answer_relevancy", output=response, input="What's the ibuprofen dose?")

    enrich_span_with_evaluation(
        metric_name="answer_relevancy",
        score=result.score,
        reason=result.reason[:200],
        latency_ms=result.latency_ms,
        span=span,
    )
    print(f"Relevancy:    {result.score:.2f} (attached to span)")

print("\nNow in Jaeger/Datadog, you can filter traces by:")
print("  gen_ai.evaluation.faithfulness.score >= 0.8")
print("  gen_ai.evaluation.answer_relevancy.score >= 0.7")


# ── Scenario 4: Auto-enrichment (hands-free) ────────────────────
divider("SCENARIO 4: Auto-enrichment (zero code changes)")
print("When auto-enrichment is on, every evaluate() call automatically")
print("attaches results to the current active span.\n")

enable_auto_enrichment()

tracer = get_tracer()
with tracer.start_as_current_span("rag-pipeline") as parent_span:
    with tracer.start_as_current_span("generate-answer"):
        # These scores auto-attach to the current span
        r1 = evaluate(
            "faithfulness",
            output="Ibuprofen is an NSAID for pain relief.",
            context="Ibuprofen: nonsteroidal anti-inflammatory drug for pain.",
        )
        print(f"faithfulness: {r1.score:.2f} (auto-attached)")

        r2 = evaluate(
            "contains",
            output="Ibuprofen is an NSAID for pain relief.",
            keyword="NSAID",
        )
        print(f"contains NSAID: {r2.score:.0f} (auto-attached)")

print("\nThe parent span 'rag-pipeline' now has child spans with scores.")


# ── Scenario 5: Structured scoring context ───────────────────────
divider("SCENARIO 5: Structured scoring context manager")
print("Use EvaluationSpanContext for cleaner structured traces.\n")

with EvaluationSpanContext("quality-gate") as ctx:
    result = evaluate(
        "faithfulness",
        output="Take aspirin daily for heart health.",
        context="Low-dose aspirin may be recommended for heart disease prevention.",
    )
    ctx.record_result(
        score=result.score,
        reason=result.reason[:100],
    )
    print(f"Score: {result.score:.2f}")
    print(f"Span created: 'quality-gate' with score attribute")


# ── Scenario 6: Production setup examples ────────────────────────
divider("SCENARIO 6: Production configurations")

print("1. OTLP to Jaeger/Grafana/Datadog:")
print("   setup_tracing(")
print("       service_name='my-service',")
print("       otlp_endpoint='localhost:4317'")
print("   )")

print("\n2. FutureAGI backend:")
print("   setup_tracing(config=TraceConfig(")
print("       service_name='my-service',")
print("       exporters=[ExporterConfig(type=ExporterType.FUTUREAGI)]")
print("   ))")

print("\n3. Multi-backend (console + OTLP):")
print("   config = TraceConfig.multi_backend(")
print("       service_name='my-service',")
print("       backends=[")
print("           {'type': 'console'},")
print("           {'type': 'otlp_grpc', 'endpoint': 'localhost:4317'},")
print("       ]")
print("   )")

print(f"\n4. Supported exporters: {[e.value for e in ExporterType]}")


# ── Scenario 7: Full RAG pipeline trace ──────────────────────────
divider("SCENARIO 7: Trace a full RAG pipeline")

tracer = get_tracer()

with tracer.start_as_current_span("rag-request") as request_span:
    request_span.set_attribute("user.query", "What's the ibuprofen dosage?")

    # Step 1: Retrieval (simulated)
    with tracer.start_as_current_span("retrieval") as ret_span:
        chunks = ["Ibuprofen: 200-400mg q4-6h PRN. Max 1200mg/day."]
        ret_span.set_attribute("retrieval.num_chunks", len(chunks))
        ret_span.set_attribute("retrieval.strategy", "vector_search")
        time.sleep(0.01)  # simulate retrieval
        print("  retrieval: 1 chunk retrieved")

    # Step 2: Generation (simulated)
    with tracer.start_as_current_span("generation") as gen_span:
        answer = "Take 200-400mg of ibuprofen every 4-6 hours."
        gen_span.set_attribute("gen_ai.model", "gemini-2.5-flash")
        gen_span.set_attribute("gen_ai.completion.0.content", answer)
        time.sleep(0.01)  # simulate LLM call
        print(f"  generation: {answer}")

    # Step 3: Quality gate (real metrics)
    with tracer.start_as_current_span("quality-gate") as gate_span:
        faith = evaluate("faithfulness", output=answer, context=chunks)
        gate_span.set_attribute("quality.faithfulness", faith.score)

        relevancy = evaluate("answer_relevancy", output=answer, input="ibuprofen dosage")
        gate_span.set_attribute("quality.relevancy", relevancy.score)

        passed = faith.passed and relevancy.passed
        gate_span.set_attribute("quality.passed", passed)

        print(f"  quality-gate: faith={faith.score:.2f} rel={relevancy.score:.2f} "
              f"passed={passed}")

    request_span.set_attribute("response.passed_quality", passed)

print("\nThis creates a trace tree:")
print("  rag-request")
print("    ├── retrieval (chunks, strategy)")
print("    ├── generation (model, response)")
print("    └── quality-gate (faithfulness, relevancy, passed)")


# Cleanup
shutdown_tracing()


divider("DONE")
print("OTEL tracing gives you full visibility into your LLM pipeline.")
print("  - Every call traced with input/output/tokens/latency")
print("  - Quality scores attached as searchable span attributes")
print("  - Auto-enrichment: zero-code metric attachment")
print("  - Export to Jaeger, Datadog, Grafana, Arize, Langfuse, etc.")
