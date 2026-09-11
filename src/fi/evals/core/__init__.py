"""
Unified evaluation core — result types, registry, engines, and the evaluate() entrypoint.
"""

from .result import EvalResult, BatchResult
from .registry import Turing, resolve_engine, is_turing_model
from .evaluate import evaluate

__all__ = [
    "EvalResult",
    "BatchResult",
    "Turing",
    "resolve_engine",
    "is_turing_model",
    "evaluate",
]
