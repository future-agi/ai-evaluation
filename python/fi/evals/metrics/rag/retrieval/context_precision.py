"""
Context Precision Metric.

Measures retrieval ranking quality - whether relevant contexts
appear before irrelevant ones.
"""

from typing import Any, Dict, List, Optional

from ...base_metric import BaseMetric
from ..types import RAGRetrievalInput
from ..utils import (
    compute_semantic_similarity,
    compute_word_overlap,
    split_into_sentences,
    check_entailment,
    NLILabel,
)


class ContextPrecision(BaseMetric[RAGRetrievalInput]):
    """
    Measures retrieval ranking quality.

    Evaluates whether relevant contexts are ranked higher than
    irrelevant ones. A single irrelevant chunk at position 1
    significantly reduces the score vs. the same chunk at position 5.

    Formula (Average Precision style):
        Context Precision@K = Σ(Precision@k × v_k) / |Relevant items in top K|

        Where:
        - Precision@k = TP@k / (TP@k + FP@k)
        - v_k ∈ {0,1} indicates relevance at rank k

    Score: 0.0 (poor ranking) to 1.0 (perfect ranking)

    Example:
        >>> precision = ContextPrecision()
        >>> result = precision.evaluate([{
        ...     "query": "What is machine learning?",
        ...     "contexts": [
        ...         "Machine learning is a branch of AI.",  # Relevant
        ...         "The weather is nice today.",  # Irrelevant
        ...     ],
        ...     "reference": "Machine learning is an AI technique."
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "context_precision"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.3)
        self.use_reference = self.config.get("use_reference", True)

    def compute_one(self, inputs: RAGRetrievalInput) -> Dict[str, Any]:
        if not inputs.contexts:
            return {
                "output": 0.0,
                "reason": "No contexts to evaluate",
            }

        # Determine relevance of each context
        relevance = []
        relevance_details = []

        for i, ctx in enumerate(inputs.contexts):
            if inputs.relevance_labels is not None and i < len(inputs.relevance_labels):
                # Use provided labels
                is_relevant = inputs.relevance_labels[i] == 1
                relevance_score = 1.0 if is_relevant else 0.0
            else:
                # Compute relevance
                relevance_score = self._compute_relevance(
                    ctx, inputs.reference, inputs.query
                )
                is_relevant = relevance_score >= self.relevance_threshold

            relevance.append(1 if is_relevant else 0)
            relevance_details.append({
                "position": i + 1,
                "relevant": is_relevant,
                "score": round(relevance_score, 3),
                "context_preview": ctx[:80] + "..." if len(ctx) > 80 else ctx,
            })

        # Calculate Average Precision
        precision_sum = 0.0
        relevant_count = 0

        for k, is_rel in enumerate(relevance, 1):
            if is_rel:
                relevant_count += 1
                precision_at_k = relevant_count / k
                precision_sum += precision_at_k

        total_relevant = sum(relevance)
        if total_relevant == 0:
            return {
                "output": 0.0,
                "reason": "No relevant contexts found",
                "relevance_by_position": relevance,
                "details": relevance_details,
            }

        avg_precision = precision_sum / total_relevant

        return {
            "output": round(avg_precision, 4),
            "reason": f"AP={avg_precision:.3f}, {total_relevant}/{len(relevance)} contexts relevant",
            "total_contexts": len(relevance),
            "relevant_contexts": total_relevant,
            "relevance_by_position": relevance,
            "details": relevance_details,
        }

    def _compute_relevance(
        self, context: str, reference: str, query: str
    ) -> float:
        """
        Compute relevance score for a context.

        Combines multiple signals:
        - Semantic similarity to reference
        - Word overlap with reference
        - Query relevance (if reference unavailable)
        """
        scores = []

        if reference:
            # Semantic similarity to reference
            sem_sim = compute_semantic_similarity(context, reference)
            scores.append(sem_sim)

            # Word overlap with reference
            word_overlap = compute_word_overlap(context, reference)
            scores.append(word_overlap)

            # NLI-based check
            label, nli_score = check_entailment(context, reference)
            if label == NLILabel.ENTAILMENT:
                scores.append(nli_score)
            elif label == NLILabel.CONTRADICTION:
                scores.append(0.0)
            else:
                scores.append(nli_score * 0.5)

        # Query relevance as fallback or additional signal
        if query:
            query_sim = compute_semantic_similarity(context, query)
            scores.append(query_sim * 0.8)  # Slightly lower weight

        if not scores:
            return 0.0

        return sum(scores) / len(scores)
