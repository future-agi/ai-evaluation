"""
LLM Client Instrumentors.

Automatic instrumentation for popular LLM client libraries.
Each instrumentor patches the library to automatically create
spans with standardized attributes.

Example:
    # Instrument individual libraries
    from fi.evals.otel.instrumentors import OpenAIInstrumentor, AnthropicInstrumentor

    OpenAIInstrumentor().instrument()
    AnthropicInstrumentor().instrument()

    # Or use the convenience function to instrument all available
    from fi.evals.otel.instrumentors import instrument_all

    instrumented = instrument_all()
    print(f"Instrumented: {instrumented}")

    # Clean up
    from fi.evals.otel.instrumentors import uninstrument_all
    uninstrument_all()
"""

from typing import List, Dict, Optional

from .base import BaseInstrumentor, InstrumentorManager
from .openai import OpenAIInstrumentor
from .anthropic import AnthropicInstrumentor

# Global manager instance
_manager: Optional[InstrumentorManager] = None


def get_manager() -> InstrumentorManager:
    """Get the global instrumentor manager."""
    global _manager
    if _manager is None:
        _manager = InstrumentorManager()
        # Register available instrumentors
        _manager.add(OpenAIInstrumentor())
        _manager.add(AnthropicInstrumentor())
    return _manager


def instrument_all(**kwargs) -> List[str]:
    """
    Instrument all available LLM libraries.

    Returns:
        List of library names that were instrumented

    Example:
        instrumented = instrument_all()
        # ['openai', 'anthropic']
    """
    return get_manager().instrument_all(**kwargs)


def uninstrument_all(**kwargs) -> List[str]:
    """
    Remove instrumentation from all libraries.

    Returns:
        List of library names that were uninstrumented
    """
    return get_manager().uninstrument_all(**kwargs)


def instrument(library: str, **kwargs) -> bool:
    """
    Instrument a specific library.

    Args:
        library: Library name ('openai', 'anthropic', etc.)
        **kwargs: Options passed to the instrumentor

    Returns:
        True if instrumented successfully
    """
    instrumentor = get_manager().get(library)
    if instrumentor is None:
        return False
    try:
        instrumentor.instrument(**kwargs)
        return instrumentor.is_instrumented
    except Exception:
        return False


def uninstrument(library: str, **kwargs) -> bool:
    """
    Remove instrumentation from a specific library.

    Args:
        library: Library name

    Returns:
        True if uninstrumented successfully
    """
    instrumentor = get_manager().get(library)
    if instrumentor is None:
        return False
    try:
        instrumentor.uninstrument(**kwargs)
        return not instrumentor.is_instrumented
    except Exception:
        return False


def is_instrumented(library: str) -> bool:
    """Check if a library is currently instrumented."""
    instrumentor = get_manager().get(library)
    return instrumentor.is_instrumented if instrumentor else False


def get_instrumented_libraries() -> List[str]:
    """Get list of currently instrumented libraries."""
    return list(get_manager().instrumented_libraries)


__all__ = [
    # Base classes
    "BaseInstrumentor",
    "InstrumentorManager",

    # Specific instrumentors
    "OpenAIInstrumentor",
    "AnthropicInstrumentor",

    # Convenience functions
    "instrument_all",
    "uninstrument_all",
    "instrument",
    "uninstrument",
    "is_instrumented",
    "get_instrumented_libraries",
    "get_manager",
]
