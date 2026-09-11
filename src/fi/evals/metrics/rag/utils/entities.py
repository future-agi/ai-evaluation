"""
Entity extraction utilities for RAG evaluation.

Provides functions to extract named entities from text.
Supports spaCy NER with heuristic fallback.
"""

import re
from typing import List, Optional, Set

# Optional spaCy import
_SPACY_AVAILABLE = False
_nlp = None

try:
    import spacy

    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None


def _get_spacy_model():
    """Lazy-load spaCy model."""
    global _nlp
    if _nlp is None and _SPACY_AVAILABLE:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded, use fallback
            _nlp = False
    return _nlp


def extract_entities_spacy(
    text: str, entity_types: Optional[List[str]] = None
) -> Set[str]:
    """
    Extract entities using spaCy NER.

    Args:
        text: The text to extract entities from
        entity_types: Optional list of entity types to filter by

    Returns:
        Set of entity strings
    """
    nlp = _get_spacy_model()
    if not nlp:
        return extract_entities_heuristic(text)

    try:
        doc = nlp(text)
        entities = set()

        # Default entity types of interest
        default_types = {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "DATE",
            "TIME",
            "MONEY",
            "QUANTITY",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
            "PERCENT",
            "CARDINAL",
            "ORDINAL",
        }
        target_types = set(entity_types) if entity_types else default_types

        for ent in doc.ents:
            if ent.label_ in target_types:
                entities.add(ent.text)

        return entities
    except Exception:
        return extract_entities_heuristic(text)


def extract_entities_heuristic(text: str) -> Set[str]:
    """
    Fallback entity extraction using heuristics.

    Extracts:
    - Capitalized phrases (likely proper nouns)
    - Numbers with units
    - Dates in common formats
    - Monetary values

    Args:
        text: The text to extract entities from

    Returns:
        Set of entity strings
    """
    entities = set()

    # Capitalized words/phrases (proper nouns)
    # Match sequences of capitalized words
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    # Filter out sentence starters by checking position
    for noun in proper_nouns:
        # Check if it appears mid-sentence (not after . or at start)
        pattern = rf"[a-z,;:]\s+{re.escape(noun)}\b"
        if re.search(pattern, text) or text.strip().startswith(noun):
            entities.add(noun)

    # Dates - various formats
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # 01/15/2024
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{4}\b",  # Years
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.update(matches)

    # Monetary values
    money_patterns = [
        r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?",
        r"[\d,]+(?:\.\d{2})?\s*(?:dollars?|USD|euros?|EUR|pounds?|GBP)",
        r"\d+(?:\.\d+)?\s*(?:million|billion|trillion)\s*(?:dollars?|euros?|pounds?)?",
    ]
    for pattern in money_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.update(matches)

    # Percentages
    percentages = re.findall(r"\b\d+(?:\.\d+)?%", text)
    entities.update(percentages)

    # Quantities with units
    quantities = re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:kg|km|miles?|meters?|feet|inches|pounds?|ounces?|liters?|gallons?|hours?|minutes?|seconds?|days?|weeks?|months?|years?)\b",
        text,
        re.IGNORECASE,
    )
    entities.update(quantities)

    # Email addresses
    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    entities.update(emails)

    # URLs
    urls = re.findall(r"https?://[^\s]+", text)
    entities.update(urls)

    return entities


def extract_entities(
    text: str,
    entity_types: Optional[List[str]] = None,
    use_llm: bool = False,
) -> Set[str]:
    """
    Main entity extraction function.

    Tries spaCy first, falls back to heuristics.

    Args:
        text: The text to extract entities from
        entity_types: Optional list of entity types to filter by
        use_llm: Whether to use LLM for extraction (not implemented)

    Returns:
        Set of entity strings
    """
    if use_llm:
        # LLM-based extraction could be added here
        pass

    if _SPACY_AVAILABLE:
        return extract_entities_spacy(text, entity_types)

    return extract_entities_heuristic(text)


def normalize_entity(entity: str) -> str:
    """
    Normalize an entity for comparison.

    Handles case, whitespace, and common variations.

    Args:
        entity: The entity string to normalize

    Returns:
        Normalized entity string
    """
    # Lowercase
    normalized = entity.lower().strip()

    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized)

    # Remove common prefixes/suffixes
    normalized = re.sub(r"^(the|a|an)\s+", "", normalized)

    return normalized


def entities_match(entity1: str, entity2: str, threshold: float = 0.8) -> bool:
    """
    Check if two entities match.

    Handles exact matches and fuzzy matching for variations.

    Args:
        entity1: First entity
        entity2: Second entity
        threshold: Similarity threshold for fuzzy matching

    Returns:
        True if entities match
    """
    norm1 = normalize_entity(entity1)
    norm2 = normalize_entity(entity2)

    # Exact match
    if norm1 == norm2:
        return True

    # Substring match
    if norm1 in norm2 or norm2 in norm1:
        return True

    # Simple character overlap for fuzzy matching
    if len(norm1) > 3 and len(norm2) > 3:
        chars1 = set(norm1)
        chars2 = set(norm2)
        overlap = len(chars1 & chars2) / max(len(chars1), len(chars2))
        if overlap >= threshold:
            return True

    return False
