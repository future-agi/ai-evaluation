"""
Claim extraction utilities for RAG evaluation.

Provides functions to split text into sentences and extract
verifiable claims from responses.
"""

import re
from typing import List, Set


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Uses regex-based sentence boundary detection.
    Handles common abbreviations and edge cases.

    Args:
        text: The text to split

    Returns:
        List of sentence strings
    """
    if not text or not text.strip():
        return []

    # Handle common abbreviations
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|vs|i\.e|e\.g)\.", r"\1<DOT>", text)

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Restore abbreviations
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]

    return sentences


def extract_claims(text: str, min_words: int = 3) -> List[str]:
    """
    Extract verifiable claims from text.

    Filters out questions, meta-statements, and very short sentences.

    Args:
        text: The text to extract claims from
        min_words: Minimum number of words for a valid claim

    Returns:
        List of claim strings
    """
    sentences = split_into_sentences(text)
    claims = []

    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < min_words:
            continue

        # Skip questions
        if sentence.strip().endswith("?"):
            continue

        # Skip meta-statements and hedging
        meta_patterns = [
            r"^(I think|I believe|In my opinion|It seems|Perhaps|Maybe)",
            r"^(Here is|Here are|The following|Below is|Above is)",
            r"^(Note:|Important:|Warning:|Disclaimer:|Please note)",
            r"^(Let me|I will|I can|I would|I should)",
            r"^(This is|That is|These are|Those are)\s+(a|an|the)\s+(question|query|request)",
        ]
        is_meta = any(re.match(p, sentence, re.IGNORECASE) for p in meta_patterns)
        if is_meta:
            continue

        # Skip acknowledgments and greetings
        ack_patterns = [
            r"^(Thank you|Thanks|Sure|Of course|Certainly|Absolutely)",
            r"^(Hello|Hi|Hey|Good morning|Good afternoon|Good evening)",
        ]
        is_ack = any(re.match(p, sentence, re.IGNORECASE) for p in ack_patterns)
        if is_ack:
            continue

        claims.append(sentence)

    return claims


def extract_key_phrases(text: str, max_phrases: int = 20) -> List[str]:
    """
    Extract key phrases from text.

    Uses heuristics to identify important noun phrases and
    subject-verb-object patterns.

    Args:
        text: The text to extract phrases from
        max_phrases: Maximum number of phrases to return

    Returns:
        List of key phrase strings
    """
    phrases = set()

    # Extract capitalized phrases (proper nouns)
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    phrases.update(proper_nouns)

    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', text)
    phrases.update(quoted)

    # Extract phrases with numbers
    numbered = re.findall(
        r"\b\d+(?:\.\d+)?(?:\s+(?:percent|%|million|billion|thousand|hundred|dollars?|euros?|pounds?|years?|months?|days?|hours?|minutes?|seconds?))\b",
        text,
        re.IGNORECASE,
    )
    phrases.update(numbered)

    # Extract noun phrases using simple patterns
    # Pattern: adjective(s) + noun(s)
    noun_phrases = re.findall(
        r"\b(?:the\s+)?(?:[A-Za-z]+\s+){0,2}[A-Za-z]+(?:tion|ment|ness|ity|ence|ance|ing|ed)\b",
        text,
    )
    phrases.update(noun_phrases[:10])

    # Filter and limit
    filtered = [p for p in phrases if len(p) > 2 and len(p.split()) <= 5]

    return filtered[:max_phrases]


def extract_atomic_claims(text: str) -> List[str]:
    """
    Extract atomic (indivisible) claims from text.

    Breaks down compound sentences into simpler claims.

    Args:
        text: The text to extract atomic claims from

    Returns:
        List of atomic claim strings
    """
    claims = extract_claims(text)
    atomic_claims = []

    for claim in claims:
        # Split compound sentences on conjunctions
        parts = re.split(r"\b(?:and|but|however|although|while|whereas)\b", claim)

        for part in parts:
            part = part.strip()
            # Clean up leading/trailing punctuation
            part = re.sub(r"^[,;:\s]+|[,;:\s]+$", "", part)

            if len(part.split()) >= 3:
                atomic_claims.append(part)

    return atomic_claims


def get_claim_entities(claim: str) -> Set[str]:
    """
    Extract entity-like tokens from a claim.

    Simple extraction without requiring NER models.

    Args:
        claim: The claim text

    Returns:
        Set of entity-like strings
    """
    entities = set()

    # Capitalized words (likely proper nouns)
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim)
    entities.update(caps)

    # Numbers with context
    numbers = re.findall(r"\b\d+(?:\.\d+)?(?:\s+\w+)?\b", claim)
    entities.update(numbers)

    return entities
