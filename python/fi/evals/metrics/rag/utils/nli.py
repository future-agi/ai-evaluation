"""
Natural Language Inference utilities for RAG evaluation.

Provides entailment checking between claims and context.
Supports both transformer-based NLI and heuristic fallbacks.
"""

from typing import Tuple, List, Optional
from enum import Enum
import re

# Try to load NLI model
_NLI_AVAILABLE = False
_nli_pipeline = None

try:
    from transformers import pipeline

    _NLI_AVAILABLE = True
except ImportError:
    pass


class NLILabel(Enum):
    """NLI classification labels."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


def _get_nli_pipeline():
    """Lazy-load NLI pipeline."""
    global _nli_pipeline
    if _nli_pipeline is None and _NLI_AVAILABLE:
        try:
            _nli_pipeline = pipeline(
                "text-classification",
                model="microsoft/deberta-v3-xsmall-mnli-fever-anli",
                device=-1,  # CPU
            )
        except Exception:
            _nli_pipeline = False
    return _nli_pipeline


def check_entailment(premise: str, hypothesis: str) -> Tuple[NLILabel, float]:
    """
    Check if premise entails hypothesis using NLI model.

    Args:
        premise: The source text (context)
        hypothesis: The claim to verify

    Returns:
        Tuple of (NLI label, confidence score)
    """
    nli = _get_nli_pipeline()

    if not nli:
        return check_entailment_heuristic(premise, hypothesis)

    try:
        # Format for NLI model
        result = nli(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)

        label_map = {
            "ENTAILMENT": NLILabel.ENTAILMENT,
            "CONTRADICTION": NLILabel.CONTRADICTION,
            "NEUTRAL": NLILabel.NEUTRAL,
            "entailment": NLILabel.ENTAILMENT,
            "contradiction": NLILabel.CONTRADICTION,
            "neutral": NLILabel.NEUTRAL,
        }

        label = label_map.get(result[0]["label"], NLILabel.NEUTRAL)
        score = result[0]["score"]

        return label, score
    except Exception:
        return check_entailment_heuristic(premise, hypothesis)


def check_entailment_heuristic(
    premise: str, hypothesis: str
) -> Tuple[NLILabel, float]:
    """
    Heuristic entailment check using word overlap and similarity.

    Fallback when NLI model not available.

    Args:
        premise: The source text (context)
        hypothesis: The claim to verify

    Returns:
        Tuple of (NLI label, confidence score)
    """
    premise_words = set(premise.lower().split())
    hypothesis_words = set(hypothesis.lower().split())

    # Remove stopwords
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "if",
        "that",
        "this",
        "it",
        "its",
        "they",
        "their",
        "he",
        "she",
        "him",
        "her",
        "his",
        "we",
        "our",
        "you",
        "your",
    }

    premise_content = premise_words - stopwords
    hypothesis_content = hypothesis_words - stopwords

    if not hypothesis_content:
        return NLILabel.NEUTRAL, 0.5

    overlap = len(premise_content & hypothesis_content)
    coverage = overlap / len(hypothesis_content)

    # Check for negation
    negations = {"not", "n't", "never", "no", "none", "neither", "nor", "cannot"}
    premise_has_neg = bool(premise_words & negations)
    hypothesis_has_neg = bool(hypothesis_words & negations)

    if premise_has_neg != hypothesis_has_neg and coverage > 0.5:
        return NLILabel.CONTRADICTION, 0.6

    if coverage >= 0.7:
        return NLILabel.ENTAILMENT, coverage
    elif coverage >= 0.3:
        return NLILabel.NEUTRAL, coverage
    else:
        return NLILabel.NEUTRAL, coverage


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
        # Check entailment
        label, score = check_entailment(ctx, sentence)

        if label == NLILabel.ENTAILMENT:
            if score > best_score:
                best_score = score
                best_context = ctx[:200] + "..." if len(ctx) > 200 else ctx
        elif label == NLILabel.NEUTRAL and score > 0.5:
            # Also consider high-confidence neutral as partial attribution
            adjusted_score = score * 0.7
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_context = ctx[:200] + "..." if len(ctx) > 200 else ctx

    is_attributed = best_score >= threshold

    return is_attributed, best_context, best_score


def check_contradiction(claim: str, context: str) -> Tuple[bool, float]:
    """
    Check if claim contradicts the context.

    Args:
        claim: The claim to check
        context: The context to check against

    Returns:
        Tuple of (is_contradiction, confidence)
    """
    label, score = check_entailment(context, claim)

    if label == NLILabel.CONTRADICTION:
        return True, score

    # Additional heuristic checks
    claim_norm = claim.lower()
    context_norm = context.lower()

    # Check for opposing numbers/values
    claim_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", claim)
    context_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", context)

    if claim_numbers and context_numbers:
        # Check for word overlap indicating same topic
        claim_words = set(claim_norm.split())
        context_words = set(context_norm.split())
        overlap = len(claim_words & context_words) / max(len(claim_words), 1)

        if overlap > 0.5:
            for c_num in claim_numbers:
                for ctx_num in context_numbers:
                    if c_num != ctx_num:
                        return True, 0.5

    return False, 0.0
