"""
Hallucination Detection Metrics.

NLI-based and semantic analysis for detecting hallucinations in LLM outputs.
Provides fast, deterministic evaluation without LLM-as-judge dependency.

Based on:
- HaluGate methodology (vLLM)
- Natural Language Inference (NLI) approaches
- Semantic Entropy (Nature 2024)
"""

from .types import (
    HallucinationInput,
    ClaimExtractionInput,
    FactualConsistencyInput,
    Claim,
)
from .metrics import (
    Faithfulness,
    ClaimSupport,
    FactualConsistency,
    ContradictionDetection,
    HallucinationScore,
)

__all__ = [
    # Types
    "HallucinationInput",
    "ClaimExtractionInput",
    "FactualConsistencyInput",
    "Claim",
    # Metrics
    "Faithfulness",
    "ClaimSupport",
    "FactualConsistency",
    "ContradictionDetection",
    "HallucinationScore",
]
