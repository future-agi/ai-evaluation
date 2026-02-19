"""
Context Recall Metric.

Measures retrieval completeness - how much of the ground truth
information is covered by retrieved contexts.
"""

from typing import Any, Dict, List, Optional

from ...base_metric import BaseMetric
from ..types import RAGRetrievalInput
from ..utils import split_into_sentences, check_attribution


class ContextRecall(BaseMetric[RAGRetrievalInput]):
    """
    Measures retrieval completeness.

    Evaluates whether retrieved contexts contain all information
    required to produce the ground truth answer.

    Uses NLI-based attribution to check if reference sentences
    can be inferred from the retrieved contexts.

    Formula (Ragas-style):
        Context Recall = |GT sentences attributable to context| / |GT sentences|

    Score: 0.0 (no coverage) to 1.0 (complete coverage)

    Example:
        >>> recall = ContextRecall()
        >>> result = recall.evaluate([{
        ...     "query": "What is the capital of France?",
        ...     "contexts": ["Paris is the capital of France."],
        ...     "reference": "The capital of France is Paris."
        ... }])
        >>> print(result.eval_results[0].output)  # Close to 1.0
    """

    @property
    def metric_name(self) -> str:
        return "context_recall"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.attribution_threshold = self.config.get("attribution_threshold", 0.5)
        self.min_sentence_words = self.config.get("min_sentence_words", 3)

    def compute_one(self, inputs: RAGRetrievalInput) -> Dict[str, Any]:
        # Handle empty inputs
        if not inputs.reference or not inputs.reference.strip():
            return {
                "output": 1.0,
                "reason": "No reference provided - recall is trivially 1.0",
            }

        if not inputs.contexts:
            return {
                "output": 0.0,
                "reason": "No contexts provided - cannot recall any information",
            }

        # Split reference into sentences
        reference_sentences = split_into_sentences(inputs.reference)

        # Filter very short sentences
        reference_sentences = [
            s for s in reference_sentences
            if len(s.split()) >= self.min_sentence_words
        ]

        if not reference_sentences:
            return {
                "output": 1.0,
                "reason": "No verifiable sentences in reference",
            }

        # For each sentence, check attribution to any context
        attributed = 0
        attribution_details = []

        for sentence in reference_sentences:
            is_attributed, best_context, score = check_attribution(
                sentence, inputs.contexts, self.attribution_threshold
            )

            if is_attributed:
                attributed += 1

            attribution_details.append({
                "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                "attributed": is_attributed,
                "score": round(score, 3),
                "matched_context": best_context[:80] + "..." if best_context and len(best_context) > 80 else best_context,
            })

        # Calculate recall
        recall = attributed / len(reference_sentences)

        return {
            "output": round(recall, 4),
            "reason": f"{attributed}/{len(reference_sentences)} reference sentences found in context",
            "total_sentences": len(reference_sentences),
            "attributed_sentences": attributed,
            "details": attribution_details,
        }
