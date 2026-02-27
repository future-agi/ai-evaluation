"""
Evaluation Framework - Scalable evaluation infrastructure.

This module provides a unified framework for running evaluations in different modes:
- Blocking (synchronous): For development and when results are needed immediately
- Non-blocking (async): For production with zero latency impact
- Distributed: For batch processing at scale via pluggable backends

Key Features:
- Trace context propagation across threads/processes
- Automatic span enrichment with evaluation results
- Pluggable backends (thread pool built-in, Temporal/Celery/Ray extensible)
- Write evaluation logic once, run anywhere

Quick Start:
    from fi.evals.framework import Evaluator, ExecutionMode, register_evaluation

    # Define an evaluation
    @register_evaluation
    class MyEval:
        name = "my_eval"
        version = "1.0.0"

        def evaluate(self, inputs):
            score = compute_score(inputs["response"])
            return {"score": score, "passed": score > 0.7}

        def get_span_attributes(self, result):
            return {"score": result["score"], "passed": result["passed"]}

    # Run blocking (development)
    evaluator = Evaluator([MyEval()], mode=ExecutionMode.BLOCKING)
    result = evaluator.run({"response": "..."})
    print(result.results[0].value)

    # Run non-blocking (production - zero latency)
    evaluator = Evaluator([MyEval()], mode=ExecutionMode.NON_BLOCKING)
    result = evaluator.run({"response": "..."})  # Returns immediately
    batch = result.wait()  # Get results when needed

Factory Functions:
    # Convenience functions for common configurations
    from fi.evals.framework import blocking_evaluator, async_evaluator

    # Blocking
    evaluator = blocking_evaluator(MyEval())
    result = evaluator.run(inputs)

    # Async (non-blocking)
    evaluator = async_evaluator(MyEval(), max_workers=4)
    result = evaluator.run(inputs)
    batch = result.wait()

Example with OpenTelemetry:
    from fi.evals.framework import async_evaluator, register_current_span

    evaluator = async_evaluator(ToxicityEval(), BiasEval())

    with tracer.start_as_current_span("llm_call") as span:
        register_current_span()  # Enable cross-thread enrichment
        response = llm.complete(prompt)
        evaluator.run({"response": response})  # Zero latency
        return response

    # Span automatically enriched with:
    #   eval.toxicity.score, eval.toxicity.status
    #   eval.bias.score, eval.bias.status
"""

__version__ = "0.1.0"

# Core types
from .types import (
    ExecutionMode,
    EvalStatus,
    FrameworkEvalResult,
    BatchEvalResult,
    EvalInputs,
    SpanAttributes,
)

# Context
from .context import (
    EvalContext,
    get_current_context,
    create_standalone_context,
)

# Protocols
from .protocols import (
    BaseEvaluation,
    EvalRegistry,
    register_evaluation,
    create_evaluation,
)

# Enrichment
from .enrichment import (
    enrich_current_span,
    enrich_span,
    add_eval_event,
    get_current_span,
    is_span_recording,
    flatten_attributes,
    SpanEnricher,
)

# Registry
from .registry import (
    SpanRegistry,
    register_span,
    get_span,
    unregister_span,
    get_registry,
    register_current_span,
)

# Propagation
from .propagation import (
    SpanContextPropagator,
    enrich_span_by_context,
    enrich_span_by_ids,
    add_event_by_context,
    ContextCarrier,
    propagate_context,
    propagate_context_lazy,
)

# Evaluators
from .evaluators import (
    BlockingEvaluator,
    blocking_evaluate,
    NonBlockingEvaluator,
    non_blocking_evaluate,
    EvalFuture,
    BatchEvalFuture,
    EvalResultAggregator,
)

# Backends
from .backends import (
    Backend,
    BackendConfig,
    TaskHandle,
    TaskStatus,
    ThreadPoolBackend,
    ThreadPoolConfig,
)

# Resilience
from .resilience import (
    ResilientBackend,
    ResilienceConfig,
    CircuitBreakerConfig,
    RateLimitConfig,
    RetryConfig,
    DegradationConfig,
    HealthCheckConfig,
    wrap_backend,
)

# Built-in evals + builder
from .evals import (
    custom_eval,
    simple_eval,
    EvalBuilder,
)

# Unified API
from .evaluator import (
    FrameworkEvaluator,
    EvaluatorResult,
    blocking_evaluator,
    async_evaluator,
    distributed_evaluator,
    resilient_evaluator,
)

# Backwards-compatible alias used by tests and examples
Evaluator = FrameworkEvaluator

__all__ = [
    # Version
    "__version__",

    # Types
    "ExecutionMode",
    "EvalStatus",
    "FrameworkEvalResult",
    "BatchEvalResult",
    "EvalInputs",
    "SpanAttributes",

    # Context
    "EvalContext",
    "get_current_context",
    "create_standalone_context",

    # Protocols
    "BaseEvaluation",
    "EvalRegistry",
    "register_evaluation",
    "create_evaluation",

    # Enrichment
    "enrich_current_span",
    "enrich_span",
    "add_eval_event",
    "get_current_span",
    "is_span_recording",
    "flatten_attributes",
    "SpanEnricher",

    # Registry
    "SpanRegistry",
    "register_span",
    "get_span",
    "unregister_span",
    "get_registry",
    "register_current_span",

    # Propagation
    "SpanContextPropagator",
    "enrich_span_by_context",
    "enrich_span_by_ids",
    "add_event_by_context",
    "ContextCarrier",
    "propagate_context",
    "propagate_context_lazy",

    # Evaluators - Blocking
    "BlockingEvaluator",
    "blocking_evaluate",
    # Evaluators - Non-Blocking
    "NonBlockingEvaluator",
    "non_blocking_evaluate",
    "EvalFuture",
    "BatchEvalFuture",
    "EvalResultAggregator",
    # Backends
    "Backend",
    "BackendConfig",
    "TaskHandle",
    "TaskStatus",
    "ThreadPoolBackend",
    "ThreadPoolConfig",
    # Resilience
    "ResilientBackend",
    "ResilienceConfig",
    "CircuitBreakerConfig",
    "RateLimitConfig",
    "RetryConfig",
    "DegradationConfig",
    "HealthCheckConfig",
    "wrap_backend",
    # Built-in evals + builder
    "custom_eval",
    "simple_eval",
    "EvalBuilder",
    # Unified API
    "FrameworkEvaluator",
    "EvaluatorResult",
    "blocking_evaluator",
    "async_evaluator",
    "distributed_evaluator",
    "resilient_evaluator",
]
