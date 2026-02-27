"""
Answer Relevancy Metric.

Measures how well the generated response addresses the original query.
"""

from typing import Any, Dict, List, Optional

from ...base_metric import BaseMetric
from ..types import AnswerRelevancyInput
from ..utils import (
    extract_keywords,
    compute_semantic_similarity,
    compute_word_overlap,
)


class AnswerRelevancy(BaseMetric[AnswerRelevancyInput]):
    """
    Measures how well the response addresses the query.

    Combines multiple signals:
    - Keyword coverage (query keywords in response)
    - Semantic similarity (embedding-based)
    - Direct answer indicators

    Penalizes:
    - Off-topic responses
    - Incomplete answers
    - Over-general responses

    Score: 0.0 (irrelevant) to 1.0 (highly relevant)

    Example:
        >>> relevancy = AnswerRelevancy()
        >>> result = relevancy.evaluate([{
        ...     "query": "What is the capital of France?",
        ...     "response": "The capital of France is Paris."
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "answer_relevancy"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.keyword_weight = self.config.get("keyword_weight", 0.3)
        self.semantic_weight = self.config.get("semantic_weight", 0.5)
        self.structure_weight = self.config.get("structure_weight", 0.2)

    def compute_one(self, inputs: AnswerRelevancyInput) -> Dict[str, Any]:
        query = inputs.query
        response = inputs.response

        if not response or not response.strip():
            return {
                "output": 0.0,
                "reason": "Empty response",
            }

        if not query or not query.strip():
            return {
                "output": 1.0,
                "reason": "No query to evaluate against",
            }

        # 1. Keyword coverage
        query_keywords = extract_keywords(query)
        response_keywords = extract_keywords(response)

        if query_keywords:
            overlap = len(query_keywords & response_keywords)
            keyword_coverage = overlap / len(query_keywords)
        else:
            keyword_coverage = 0.5  # Neutral if no keywords

        # 2. Semantic similarity
        semantic_sim = compute_semantic_similarity(query, response)

        # 3. Structural relevancy indicators
        structure_score = self._check_structure(query, response)

        # 4. Check for refusal or non-answer patterns
        refusal_penalty = self._check_refusal(response)

        # Combine scores
        base_score = (
            self.keyword_weight * keyword_coverage +
            self.semantic_weight * semantic_sim +
            self.structure_weight * structure_score
        )

        # Apply refusal penalty
        final_score = base_score * (1.0 - refusal_penalty)

        return {
            "output": round(final_score, 4),
            "reason": f"Relevancy={final_score:.2f} (keywords={keyword_coverage:.2f}, semantic={semantic_sim:.2f})",
            "keyword_coverage": round(keyword_coverage, 4),
            "semantic_similarity": round(semantic_sim, 4),
            "structure_score": round(structure_score, 4),
            "refusal_penalty": round(refusal_penalty, 4),
        }

    def _check_structure(self, query: str, response: str) -> float:
        """Check structural indicators of a relevant answer."""
        score = 0.5  # Neutral starting point
        query_lower = query.lower()
        response_lower = response.lower()

        # Question type detection and answer format checking
        question_patterns = {
            "what is": ["is", "are", "the", "a", "an"],
            "what are": ["are", "include", "consist"],
            "who is": ["is", "was", "name"],
            "who are": ["are", "were", "include"],
            "when": ["in", "on", "at", "during", "year", "date"],
            "where": ["in", "at", "located", "place", "city", "country"],
            "how": ["by", "through", "using", "step", "method"],
            "why": ["because", "since", "due to", "reason", "cause"],
            "how many": ["number", "total", "count"] + [str(i) for i in range(10)],
            "how much": ["amount", "cost", "price", "$", "dollar"],
        }

        for question_type, answer_indicators in question_patterns.items():
            if question_type in query_lower:
                # Check if response has appropriate indicators
                has_indicator = any(ind in response_lower for ind in answer_indicators)
                if has_indicator:
                    score += 0.3
                break

        # Check for direct answer patterns
        direct_starters = [
            "the answer is",
            "it is",
            "they are",
            "yes,",
            "no,",
            "the",
        ]
        if any(response_lower.startswith(starter) for starter in direct_starters):
            score += 0.2

        return min(1.0, score)

    def _check_refusal(self, response: str) -> float:
        """Check for refusal or non-answer patterns."""
        response_lower = response.lower()

        refusal_patterns = [
            "i cannot",
            "i can't",
            "i'm unable",
            "i don't know",
            "i do not know",
            "i'm not sure",
            "i am not sure",
            "i don't have",
            "i do not have",
            "i cannot provide",
            "unable to answer",
            "don't have information",
            "no information",
            "not available",
        ]

        for pattern in refusal_patterns:
            if pattern in response_lower:
                return 0.5  # Partial penalty for refusal

        # Check for extremely short responses
        if len(response.split()) < 3:
            return 0.2  # Small penalty for very short responses

        return 0.0
