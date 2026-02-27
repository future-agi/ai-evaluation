"""
Unified registry — resolves eval names to engines.

Routing is purely based on user-provided kwargs:
    1. Explicit engine= → use that
    2. Turing model → "turing"
    3. Any other model → "llm"
    4. No model → "local" (default)
"""

from enum import Enum
from typing import Optional


class Turing(str, Enum):
    """Model options for the Turing (FutureAGI) cloud engine."""

    FLASH = "turing_flash"
    SMALL = "turing_small"
    LARGE = "turing_large"


TURING_MODEL_PREFIXES = ("turing",)


def is_turing_model(model: Optional[str]) -> bool:
    """Check if the model string indicates a Turing platform model."""
    if not model:
        return False
    return model.lower().startswith(TURING_MODEL_PREFIXES)


def resolve_engine(
    name: Optional[str] = None,
    *,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    engine: Optional[str] = None,
) -> str:
    """Auto-detect which engine based on user-provided kwargs.

    Priority:
        1. Explicit engine kwarg
        2. Turing model → "turing"
        3. Any other model → "llm"
        4. No model → "local" (default, fails gracefully if metric not found)
    """
    if engine:
        return engine

    if is_turing_model(model):
        return "turing"

    if model:
        return "llm"

    return "local"
