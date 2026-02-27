"""
OpenTelemetry Span Processors.

Custom processors that enrich LLM spans with evaluation scores,
cost data, and other metadata.
"""

from .base import (
    BaseSpanProcessor,
    FilteringSpanProcessor,
    CompositeSpanProcessor,
    AttributeEnrichmentProcessor,
    ConditionalProcessor,
    OTEL_AVAILABLE,
)
from .llm import LLMSpanProcessor
from .evaluation import EvaluationSpanProcessor, BatchEvaluationProcessor
from .cost import CostSpanProcessor, calculate_cost, DEFAULT_PRICING

__all__ = [
    # Base classes
    "BaseSpanProcessor",
    "FilteringSpanProcessor",
    "CompositeSpanProcessor",
    "AttributeEnrichmentProcessor",
    "ConditionalProcessor",
    "OTEL_AVAILABLE",
    # Specialized processors
    "LLMSpanProcessor",
    "EvaluationSpanProcessor",
    "BatchEvaluationProcessor",
    "CostSpanProcessor",
    # Utilities
    "calculate_cost",
    "DEFAULT_PRICING",
]
