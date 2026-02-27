"""Local metrics module.

This module provides the base class and utilities for local metric implementations.
The actual metric implementations live in fi.evals.metrics.heuristics/ and are
registered via the LocalMetricRegistry.
"""

# Re-export commonly used types for convenience
from ...types import TextMetricInput, JsonMetricInput, EvalResult, BatchRunResult
from ...metrics.base_metric import BaseMetric


__all__ = [
    "BaseMetric",
    "TextMetricInput",
    "JsonMetricInput",
    "EvalResult",
    "BatchRunResult",
]
