"""Local execution module for running evaluations without API calls.

This module provides the infrastructure for running heuristic metrics locally,
enabling offline evaluation and faster feedback loops during development.

It also supports local LLM inference via Ollama for running LLM-as-judge
evaluations without cloud API calls.

Example:
    >>> from fi.evals.local import LocalEvaluator, ExecutionMode
    >>>
    >>> # Run a metric locally
    >>> evaluator = LocalEvaluator()
    >>> result = evaluator.evaluate(
    ...     metric_name="contains",
    ...     inputs=[{"response": "Hello world"}],
    ...     config={"keyword": "world"}
    ... )
    >>> print(result.results.eval_results[0].output)
    1.0

    >>> # Check if a metric can run locally
    >>> evaluator.can_run_locally("contains")  # True
    >>> evaluator.can_run_locally("groundedness")  # False (requires LLM)

    >>> # Use hybrid mode to automatically route
    >>> from fi.evals.local import HybridEvaluator
    >>> hybrid = HybridEvaluator()
    >>> partitions = hybrid.partition_evaluations([
    ...     {"metric_name": "contains", "inputs": [...]},
    ...     {"metric_name": "groundedness", "inputs": [...]},
    ... ])
    >>> # partitions[ExecutionMode.LOCAL] = [contains eval]
    >>> # partitions[ExecutionMode.CLOUD] = [groundedness eval]

    >>> # Use local LLM for LLM-based evaluations
    >>> from fi.evals.local import OllamaLLM, HybridEvaluator
    >>> llm = OllamaLLM()
    >>> hybrid = HybridEvaluator(local_llm=llm)
    >>> result = llm.judge(
    ...     query="What is AI?",
    ...     response="AI is artificial intelligence.",
    ...     criteria="Evaluate if the response correctly answers the question."
    ... )
    >>> print(result["score"])
    0.9
"""

from .execution_mode import (
    ExecutionMode,
    LOCAL_CAPABLE_METRICS,
    can_run_locally,
    select_execution_mode,
)
from .registry import (
    LocalMetricRegistry,
    get_registry,
)
from .evaluator import (
    LocalEvaluator,
    LocalEvaluatorConfig,
    LocalEvaluationResult,
    HybridEvaluator,
)
from .llm import (
    LocalLLMConfig,
    OllamaLLM,
    LocalLLMFactory,
)


__all__ = [
    # Execution mode
    "ExecutionMode",
    "LOCAL_CAPABLE_METRICS",
    "can_run_locally",
    "select_execution_mode",
    # Registry
    "LocalMetricRegistry",
    "get_registry",
    # Evaluator
    "LocalEvaluator",
    "LocalEvaluatorConfig",
    "LocalEvaluationResult",
    "HybridEvaluator",
    # Local LLM
    "LocalLLMConfig",
    "OllamaLLM",
    "LocalLLMFactory",
]
