"""
Natural Language Inference utilities for hallucination detection.

Provides entailment checking between claims and context.
Uses transformer-based NLI when available (pip install ai-evaluation[nli]),
falls back to word-overlap heuristic with a warning.
"""

import re
import warnings
from typing import Tuple, List, Optional
from enum import Enum


_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# Try to import transformers
_NLI_AVAILABLE = False
try:
    from transformers import pipeline as _hf_pipeline
    _NLI_AVAILABLE = True
except ImportError:
    pass

_nli_pipeline = None
_heuristic_warning_issued = False


class NLILabel(Enum):
    """NLI classification labels."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


def _get_nli_pipeline():
    """Lazy-load NLI pipeline. Returns the pipeline or False on failure."""
    global _nli_pipeline
    if _nli_pipeline is not None:
        return _nli_pipeline

    if not _NLI_AVAILABLE:
        _nli_pipeline = False
        return False

    try:
        _nli_pipeline = _hf_pipeline(
            "text-classification",
            model=_NLI_MODEL,
            device=-1,  # CPU
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to load NLI model '{_NLI_MODEL}': {exc}. "
            "Falling back to word-overlap heuristic. "
            "Install with: pip install ai-evaluation[nli]",
            RuntimeWarning,
            stacklevel=2,
        )
        _nli_pipeline = False

    return _nli_pipeline


def _warn_heuristic_fallback():
    """Issue a one-time warning that we're using the heuristic fallback."""
    global _heuristic_warning_issued
    if not _heuristic_warning_issued:
        _heuristic_warning_issued = True
        warnings.warn(
            "NLI model not available — using word-overlap heuristic for "
            "hallucination detection. Results will be approximate. "
            "For accurate NLI, install: pip install ai-evaluation[nli]",
            RuntimeWarning,
            stacklevel=3,
        )


_LABEL_MAP = {
    "ENTAILMENT": NLILabel.ENTAILMENT,
    "CONTRADICTION": NLILabel.CONTRADICTION,
    "NEUTRAL": NLILabel.NEUTRAL,
    "entailment": NLILabel.ENTAILMENT,
    "contradiction": NLILabel.CONTRADICTION,
    "neutral": NLILabel.NEUTRAL,
}


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
        _warn_heuristic_fallback()
        return check_entailment_heuristic(premise, hypothesis)

    try:
        result = nli(
            {"text": premise, "text_pair": hypothesis},
            truncation=True,
            max_length=512,
        )
        # Dict input returns a single dict, not a list
        entry = result[0] if isinstance(result, list) else result
        label = _LABEL_MAP.get(entry["label"], NLILabel.NEUTRAL)
        score = entry["score"]
        return label, score
    except Exception:
        return check_entailment_heuristic(premise, hypothesis)


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "and", "or", "but",
    "if", "that", "this", "it", "its", "they", "their", "he", "she",
    "him", "her", "his", "we", "our", "you", "your",
})

_NEGATIONS = frozenset({
    "not", "n't", "never", "no", "none", "neither", "nor", "cannot",
})


def _tokenize(text: str) -> set:
    """Tokenize text into content words, stripping punctuation."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def check_entailment_heuristic(
    premise: str, hypothesis: str
) -> Tuple[NLILabel, float]:
    """
    Heuristic entailment check using word overlap and similarity.

    Fallback when NLI model is not available. Uses:
    - Content word overlap for entailment signal
    - Negation asymmetry for contradiction detection
    - Numeric mismatch detection

    Args:
        premise: The source text (context)
        hypothesis: The claim to verify

    Returns:
        Tuple of (NLI label, confidence score)
    """
    premise_words = _tokenize(premise)
    hypothesis_words = _tokenize(hypothesis)

    premise_content = premise_words - _STOPWORDS
    hypothesis_content = hypothesis_words - _STOPWORDS

    if not hypothesis_content:
        return NLILabel.NEUTRAL, 0.5

    overlap = len(premise_content & hypothesis_content)
    coverage = overlap / len(hypothesis_content)

    # Check for negation asymmetry
    premise_has_neg = bool(premise_words & _NEGATIONS)
    hypothesis_has_neg = bool(hypothesis_words & _NEGATIONS)

    if premise_has_neg != hypothesis_has_neg and coverage > 0.5:
        return NLILabel.CONTRADICTION, 0.6

    # Check for numeric mismatch — only when high overlap + claim has numbers not in premise
    premise_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", premise))
    hypothesis_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", hypothesis))

    if hypothesis_numbers and premise_numbers and coverage > 0.5:
        novel_numbers = hypothesis_numbers - premise_numbers
        if novel_numbers and len(novel_numbers) <= 2:
            return NLILabel.CONTRADICTION, 0.55

    if coverage >= 0.65:
        return NLILabel.ENTAILMENT, coverage
    elif coverage >= 0.3:
        return NLILabel.NEUTRAL, coverage
    else:
        return NLILabel.NEUTRAL, coverage


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

    return False, 0.0


def nli_score_for_claim(
    claim: str, contexts: List[str]
) -> Tuple[NLILabel, float, Optional[str]]:
    """
    Get the best NLI score for a claim against multiple contexts.

    Args:
        claim: The claim to verify
        contexts: List of context passages

    Returns:
        Tuple of (best_label, best_score, best_context_snippet)
    """
    best_label = NLILabel.NEUTRAL
    best_score = 0.0
    best_context = None

    for ctx in contexts:
        label, score = check_entailment(ctx, claim)

        if label == NLILabel.CONTRADICTION and score > 0.5:
            # Contradiction takes priority if confident
            snippet = ctx[:200] + "..." if len(ctx) > 200 else ctx
            return label, score, snippet

        if label == NLILabel.ENTAILMENT and score > best_score:
            best_label = label
            best_score = score
            best_context = ctx[:200] + "..." if len(ctx) > 200 else ctx
        elif label == NLILabel.NEUTRAL and best_label != NLILabel.ENTAILMENT:
            if score > best_score:
                best_label = label
                best_score = score
                best_context = ctx[:200] + "..." if len(ctx) > 200 else ctx

    return best_label, best_score, best_context
