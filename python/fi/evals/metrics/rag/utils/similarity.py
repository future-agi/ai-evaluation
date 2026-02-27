"""
Similarity computation utilities for RAG evaluation.

Provides functions for text similarity, word overlap,
and semantic similarity computation.
"""

import re
from difflib import SequenceMatcher
from typing import Optional, Set, List

# Optional sentence-transformers import
_EMBEDDINGS_AVAILABLE = False
_embedding_model = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    pass


# Stopwords for filtering
STOPWORDS = {
    "a",
    "an",
    "the",
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
    "need",
    "dare",
    "ought",
    "used",
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
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "just",
    "don",
    "now",
    "and",
    "but",
    "or",
    "if",
    "because",
    "while",
    "although",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "what",
    "which",
    "who",
    "whom",
}


def _get_embedding_model():
    """Lazy-load embedding model."""
    global _embedding_model
    if _embedding_model is None and _EMBEDDINGS_AVAILABLE:
        try:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedding_model = False
    return _embedding_model


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text: The text to normalize

    Returns:
        Normalized text
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between two texts using sequence matching.

    Uses difflib's SequenceMatcher for character-level similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    return SequenceMatcher(None, text1, text2).ratio()


def compute_word_overlap(
    text1: str, text2: str, remove_stopwords: bool = True
) -> float:
    """
    Compute word overlap between two texts.

    Args:
        text1: First text
        text2: Second text
        remove_stopwords: Whether to remove stopwords

    Returns:
        Overlap score between 0 and 1
    """
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())

    if remove_stopwords:
        words1 = words1 - STOPWORDS
        words2 = words2 - STOPWORDS

    if not words1 or not words2:
        return 0.0

    overlap = words1 & words2

    # Jaccard similarity
    union = words1 | words2
    return len(overlap) / len(union) if union else 0.0


def compute_semantic_similarity(
    text1: str, text2: str, use_embeddings: bool = True
) -> float:
    """
    Compute semantic similarity between two texts.

    Uses sentence embeddings if available, falls back to word overlap.

    Args:
        text1: First text
        text2: Second text
        use_embeddings: Whether to try using embeddings

    Returns:
        Similarity score between 0 and 1
    """
    if use_embeddings:
        model = _get_embedding_model()
        if model:
            try:
                import numpy as np

                embeddings = model.encode([text1, text2])
                # Cosine similarity
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return float(max(0, similarity))
            except Exception:
                pass

    # Fallback to combined heuristic
    word_overlap = compute_word_overlap(text1, text2)
    char_similarity = compute_text_similarity(text1, text2)

    return 0.6 * word_overlap + 0.4 * char_similarity


def extract_keywords(text: str, max_keywords: int = 20) -> Set[str]:
    """
    Extract keywords from text.

    Removes stopwords and returns content words.

    Args:
        text: The text to extract keywords from
        max_keywords: Maximum number of keywords to return

    Returns:
        Set of keyword strings
    """
    words = set(normalize_text(text).split())
    content_words = words - STOPWORDS

    # Filter very short words
    content_words = {w for w in content_words if len(w) > 2}

    # Limit to max_keywords
    return set(list(content_words)[:max_keywords])


def compute_ngram_overlap(
    text1: str, text2: str, n: int = 3
) -> float:
    """
    Compute n-gram overlap between two texts.

    Args:
        text1: First text
        text2: Second text
        n: Size of n-grams

    Returns:
        Overlap score between 0 and 1
    """
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    words1 = text1.split()
    words2 = text2.split()

    if len(words1) < n or len(words2) < n:
        return compute_word_overlap(text1, text2)

    ngrams1 = set()
    ngrams2 = set()

    for i in range(len(words1) - n + 1):
        ngrams1.add(" ".join(words1[i : i + n]))

    for i in range(len(words2) - n + 1):
        ngrams2.add(" ".join(words2[i : i + n]))

    if not ngrams1 or not ngrams2:
        return 0.0

    overlap = ngrams1 & ngrams2
    union = ngrams1 | ngrams2

    return len(overlap) / len(union) if union else 0.0


def find_best_matching_sentence(
    query: str, sentences: List[str]
) -> tuple:
    """
    Find the sentence that best matches the query.

    Args:
        query: The query text
        sentences: List of candidate sentences

    Returns:
        Tuple of (best_sentence, similarity_score, index)
    """
    best_sentence = ""
    best_score = 0.0
    best_idx = -1

    for idx, sentence in enumerate(sentences):
        score = compute_semantic_similarity(query, sentence)
        if score > best_score:
            best_score = score
            best_sentence = sentence
            best_idx = idx

    return best_sentence, best_score, best_idx
