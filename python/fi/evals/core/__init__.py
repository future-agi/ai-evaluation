"""
Unified evaluation core — result types, registry, engines, and the evaluate() entrypoint.
"""

from .result import EvalResult, BatchResult
from .registry import UnifiedRegistry, get_unified_registry, Turing
from .evaluate import evaluate

__all__ = [
    "EvalResult",
    "BatchResult",
    "Turing",
    "UnifiedRegistry",
    "get_unified_registry",
    "evaluate",
]
