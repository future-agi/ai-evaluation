"""
Hallucination Detection Metrics.

NLI-based evaluation for detecting hallucinations in LLM outputs.
Provides fast, deterministic evaluation without LLM-as-judge dependency.

Architecture:
- nli.py: NLI entailment checking (transformer or heuristic fallback)
- sentinel.py: Fast rule-based pre-screening
- detector.py: Main orchestrator combining sentinel + NLI
- metrics.py: 5 BaseMetric classes for the evaluate() API
"""

from .types import (
    HallucinationInput,
    ClaimExtractionInput,
    FactualConsistencyInput,
    Claim,
    NLIResult,
    HallucinationResult,
)
from .metrics import (
    Faithfulness,
    ClaimSupport,
    FactualConsistency,
    ContradictionDetection,
    HallucinationScore,
)
from .nli import NLILabel, check_entailment, check_contradiction
from .sentinel import HallucinationSentinel
from .detector import HallucinationDetector

__all__ = [
    # Types
    "HallucinationInput",
    "ClaimExtractionInput",
    "FactualConsistencyInput",
    "Claim",
    "NLIResult",
    "HallucinationResult",
    # Metrics
    "Faithfulness",
    "ClaimSupport",
    "FactualConsistency",
    "ContradictionDetection",
    "HallucinationScore",
    # NLI utilities
    "NLILabel",
    "check_entailment",
    "check_contradiction",
    # Components
    "HallucinationSentinel",
    "HallucinationDetector",
]
