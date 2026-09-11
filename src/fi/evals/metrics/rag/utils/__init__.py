"""
RAG Evaluation Utilities.

Provides common functions for NLI, claim extraction,
entity extraction, and semantic similarity.
"""

from .nli import (
    NLILabel,
    check_entailment,
    check_entailment_heuristic,
    check_claim_supported,
    check_attribution,
    check_contradiction,
)
from .claims import (
    split_into_sentences,
    extract_claims,
    extract_key_phrases,
    extract_atomic_claims,
)
from .entities import (
    extract_entities,
    extract_entities_heuristic,
    entities_match,
    normalize_entity,
)
from .similarity import (
    compute_text_similarity,
    compute_word_overlap,
    compute_semantic_similarity,
    compute_ngram_overlap,
    extract_keywords,
    normalize_text,
)

__all__ = [
    # NLI
    "NLILabel",
    "check_entailment",
    "check_entailment_heuristic",
    "check_claim_supported",
    "check_attribution",
    "check_contradiction",
    # Claims
    "split_into_sentences",
    "extract_claims",
    "extract_key_phrases",
    "extract_atomic_claims",
    # Entities
    "extract_entities",
    "extract_entities_heuristic",
    "entities_match",
    "normalize_entity",
    # Similarity
    "compute_text_similarity",
    "compute_word_overlap",
    "compute_semantic_similarity",
    "compute_ngram_overlap",
    "extract_keywords",
    "normalize_text",
]
