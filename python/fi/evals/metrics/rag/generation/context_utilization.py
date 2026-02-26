"""
Context Utilization Metric.

Measures whether the model actually uses the provided context
vs. relying on parametric knowledge.

Novel metric addressing "context neglect" problem identified
in 2025 research from Google DeepMind.
"""

import re
from typing import Any, Dict, List, Optional, Set

from ...base_metric import BaseMetric
from ..types import ContextUtilizationInput
from ..utils import (
    extract_entities,
    extract_key_phrases,
    compute_ngram_overlap,
    normalize_text,
)


class ContextUtilization(BaseMetric[ContextUtilizationInput]):
    """
    Measures whether the model actually USES the provided context.

    Detects "context neglect" - when models ignore retrieved context
    and generate from parametric knowledge instead.

    Approach:
    1. Extract key information units from context (entities, phrases)
    2. Check which units appear in the response
    3. Check for context-specific phrasing (n-gram overlap)
    4. Calculate utilization ratio

    Novel metric addressing 2025 research findings on context neglect.

    Score: 0.0 (context ignored) to 1.0 (context fully utilized)

    Example:
        >>> utilization = ContextUtilization()
        >>> result = utilization.evaluate([{
        ...     "query": "What did Einstein discover?",
        ...     "response": "Einstein developed the theory of relativity.",
        ...     "contexts": ["Albert Einstein developed the theory of relativity in 1905."]
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "context_utilization"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_utilization = self.config.get("min_utilization", 0.3)
        self.entity_weight = self.config.get("entity_weight", 0.4)
        self.phrase_weight = self.config.get("phrase_weight", 0.3)
        self.ngram_weight = self.config.get("ngram_weight", 0.3)

    def compute_one(self, inputs: ContextUtilizationInput) -> Dict[str, Any]:
        response = inputs.response
        contexts = inputs.contexts

        if not contexts:
            return {
                "output": 0.0,
                "reason": "No context provided — cannot measure utilization",
            }

        if not response or not response.strip():
            return {
                "output": 0.0,
                "reason": "Empty response",
            }

        # 1. Extract information units from contexts
        context_entities: Set[str] = set()
        context_phrases: Set[str] = set()

        for ctx in contexts:
            entities = extract_entities(ctx)
            context_entities.update(entities)

            phrases = extract_key_phrases(ctx)
            context_phrases.update(phrases)

        # 2. Check entity utilization
        entity_utilization = self._check_entity_utilization(response, context_entities)

        # 3. Check phrase utilization
        phrase_utilization = self._check_phrase_utilization(response, context_phrases)

        # 4. Check n-gram overlap (context-specific phrasing)
        ngram_utilization = self._check_ngram_utilization(response, contexts)

        # Combined score
        utilization = (
            self.entity_weight * entity_utilization +
            self.phrase_weight * phrase_utilization +
            self.ngram_weight * ngram_utilization
        )

        # Determine if context was neglected
        neglected = utilization < self.min_utilization

        return {
            "output": round(utilization, 4),
            "reason": f"Utilization={utilization:.2f} (entities={entity_utilization:.2f}, phrases={phrase_utilization:.2f}, ngrams={ngram_utilization:.2f})",
            "entity_utilization": round(entity_utilization, 4),
            "phrase_utilization": round(phrase_utilization, 4),
            "ngram_utilization": round(ngram_utilization, 4),
            "context_neglected": neglected,
            "context_entities_found": len(context_entities),
            "context_phrases_found": len(context_phrases),
        }

    def _check_entity_utilization(
        self, response: str, context_entities: Set[str]
    ) -> float:
        """Check how many context entities appear in the response."""
        if not context_entities:
            return 1.0  # No entities to utilize

        response_lower = response.lower()
        utilized = 0

        for entity in context_entities:
            entity_lower = entity.lower()
            if entity_lower in response_lower:
                utilized += 1
            else:
                # Check for partial matches (e.g., "Einstein" in "Albert Einstein")
                entity_words = entity_lower.split()
                if any(word in response_lower for word in entity_words if len(word) > 3):
                    utilized += 0.5

        return min(1.0, utilized / len(context_entities))

    def _check_phrase_utilization(
        self, response: str, context_phrases: Set[str]
    ) -> float:
        """Check how many context phrases appear in the response."""
        if not context_phrases:
            return 1.0  # No phrases to utilize

        response_lower = response.lower()
        utilized = 0

        for phrase in context_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in response_lower:
                utilized += 1
            else:
                # Check for word overlap
                phrase_words = set(phrase_lower.split())
                response_words = set(response_lower.split())
                overlap = len(phrase_words & response_words)
                if overlap >= len(phrase_words) * 0.7:
                    utilized += 0.7

        return min(1.0, utilized / len(context_phrases))

    def _check_ngram_utilization(
        self, response: str, contexts: List[str]
    ) -> float:
        """
        Check for context-specific phrasing using n-gram overlap.

        High n-gram overlap suggests the model is actually reading
        and using the context, not just generating from memory.
        """
        combined_context = " ".join(contexts)

        # Calculate 3-gram overlap
        overlap_3 = compute_ngram_overlap(response, combined_context, n=3)

        # Calculate 4-gram overlap (more specific)
        overlap_4 = compute_ngram_overlap(response, combined_context, n=4)

        # Combine with more weight on longer n-grams
        return 0.4 * overlap_3 + 0.6 * overlap_4


class ContextRelevanceToResponse(BaseMetric[ContextUtilizationInput]):
    """
    Measures how relevant the retrieved context is to the actual response.

    Complementary to ContextUtilization - this checks from context's
    perspective rather than response's perspective.

    Useful for identifying when context retrieval was poor (low relevance)
    vs when the model ignored good context (high relevance, low utilization).

    Score: 0.0 (context irrelevant) to 1.0 (context highly relevant)
    """

    @property
    def metric_name(self) -> str:
        return "context_relevance_to_response"

    def compute_one(self, inputs: ContextUtilizationInput) -> Dict[str, Any]:
        response = inputs.response
        contexts = inputs.contexts

        if not contexts or not response:
            return {
                "output": 0.0,
                "reason": "Missing context or response",
            }

        # Check relevance of each context to the response
        relevance_scores = []

        for ctx in contexts:
            # N-gram overlap
            ngram_score = compute_ngram_overlap(response, ctx, n=3)

            # Entity overlap
            ctx_entities = extract_entities(ctx)
            response_entities = extract_entities(response)

            if ctx_entities:
                entity_overlap = len(ctx_entities & response_entities) / len(ctx_entities)
            else:
                entity_overlap = 0.5

            # Combined relevance for this context
            ctx_relevance = 0.5 * ngram_score + 0.5 * entity_overlap
            relevance_scores.append(ctx_relevance)

        # Average relevance across contexts
        avg_relevance = sum(relevance_scores) / len(relevance_scores)

        # Max relevance (best context)
        max_relevance = max(relevance_scores)

        # Use weighted combination
        final_score = 0.6 * avg_relevance + 0.4 * max_relevance

        return {
            "output": round(final_score, 4),
            "reason": f"Context relevance: avg={avg_relevance:.2f}, max={max_relevance:.2f}",
            "average_relevance": round(avg_relevance, 4),
            "max_relevance": round(max_relevance, 4),
            "per_context_relevance": [round(s, 4) for s in relevance_scores],
        }
