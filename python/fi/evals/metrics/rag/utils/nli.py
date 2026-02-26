"""
Natural Language Inference utilities for RAG evaluation.

Provides entailment checking between claims and context.
Reuses the shared NLI pipeline from the hallucination module to avoid
loading the model twice. Falls back to word-overlap heuristic when
transformers is not installed.
"""

import re
import warnings
from typing import Tuple, List, Optional
from enum import Enum

from ...hallucination.nli import (
    NLILabel,
    check_entailment,
    check_entailment_heuristic,
    check_contradiction,
)

# Re-export for backwards compatibility
__all__ = [
    "NLILabel",
    "check_entailment",
    "check_entailment_heuristic",
    "check_claim_supported",
    "check_attribution",
    "check_contradiction",
]


def check_claim_supported(
    claim: str, contexts: List[str], threshold: float = 0.5
) -> Tuple[bool, float, Optional[str]]:
    """
    Check if a claim is supported by any of the contexts.

    Args:
        claim: The claim to verify
        contexts: List of context passages
        threshold: Minimum score to consider claim supported

    Returns:
        Tuple of (is_supported, best_score, best_supporting_context)
    """
    best_score = 0.0
    best_context = None

    for ctx in contexts:
        label, score = check_entailment(ctx, claim)

        if label == NLILabel.ENTAILMENT and score > best_score:
            best_score = score
            best_context = ctx[:200] + "..." if len(ctx) > 200 else ctx

    is_supported = best_score >= threshold

    return is_supported, best_score, best_context


def check_attribution(
    sentence: str, contexts: List[str], threshold: float = 0.5
) -> Tuple[bool, Optional[str], float]:
    """
    Check if a sentence can be attributed to any context.

    Args:
        sentence: The sentence to check attribution for
        contexts: List of context passages
        threshold: Minimum score to consider attributed

    Returns:
        Tuple of (is_attributed, best_context, best_score)
    """
    best_score = 0.0
    best_context = None

    for ctx in contexts:
        label, score = check_entailment(ctx, sentence)

        if label == NLILabel.ENTAILMENT:
            if score > best_score:
                best_score = score
                best_context = ctx[:200] + "..." if len(ctx) > 200 else ctx
        elif label == NLILabel.NEUTRAL and score > 0.5:
            # High-confidence neutral as partial attribution
            adjusted_score = score * 0.7
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_context = ctx[:200] + "..." if len(ctx) > 200 else ctx

    is_attributed = best_score >= threshold

    return is_attributed, best_context, best_score
