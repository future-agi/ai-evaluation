"""
Ranking Metrics for RAG Retrieval.

Provides NDCG (Normalized Discounted Cumulative Gain) and
MRR (Mean Reciprocal Rank) metrics.
"""

import math
from typing import Any, Dict, List, Optional

from ...base_metric import BaseMetric
from ..types import RAGRankingInput


class NDCG(BaseMetric[RAGRankingInput]):
    """
    Normalized Discounted Cumulative Gain for ranked retrieval.

    Accounts for graded relevance (not just binary) and position.
    Higher scores for relevant items appearing early in ranking.

    Formula:
        DCG@k = Σ (2^rel_i - 1) / log2(i + 2)
        NDCG@k = DCG@k / IDCG@k

    Where IDCG is the ideal DCG (perfect ranking).

    Score: 0.0 (worst ranking) to 1.0 (perfect ranking)

    Example:
        >>> ndcg = NDCG(config={"k": 5})
        >>> result = ndcg.evaluate([{
        ...     "query": "machine learning",
        ...     "contexts": ["ML intro", "Unrelated", "ML advanced"],
        ...     "relevance_scores": [1.0, 0.0, 0.8]
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "ndcg"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k = self.config.get("k", None)  # None = use all

    def compute_one(self, inputs: RAGRankingInput) -> Dict[str, Any]:
        scores = inputs.relevance_scores

        if not scores:
            return {
                "output": 0.0,
                "reason": "No relevance scores provided",
            }

        k = self.k if self.k is not None else len(scores)
        k = min(k, len(scores))

        # DCG@k
        dcg = self._compute_dcg(scores[:k])

        # Ideal DCG@k (perfect ranking - sorted descending)
        ideal_scores = sorted(scores, reverse=True)
        idcg = self._compute_dcg(ideal_scores[:k])

        if idcg == 0:
            # All scores are 0
            return {
                "output": 0.0,
                "reason": "All relevance scores are 0",
                "dcg": 0.0,
                "idcg": 0.0,
            }

        ndcg = dcg / idcg

        return {
            "output": round(ndcg, 4),
            "reason": f"NDCG@{k}={ndcg:.3f}, DCG={dcg:.3f}, IDCG={idcg:.3f}",
            "k": k,
            "dcg": round(dcg, 4),
            "idcg": round(idcg, 4),
            "relevance_scores": scores[:k],
        }

    def _compute_dcg(self, scores: List[float]) -> float:
        """Compute Discounted Cumulative Gain."""
        dcg = 0.0
        for i, score in enumerate(scores):
            # Position is 1-indexed in the formula
            dcg += (2**score - 1) / math.log2(i + 2)
        return dcg


class MRR(BaseMetric[RAGRankingInput]):
    """
    Mean Reciprocal Rank - how quickly the first relevant result appears.

    Best for single-answer retrieval tasks where you only need
    one correct result.

    Formula:
        MRR = 1 / rank_of_first_relevant

    Score: 0.0 (no relevant results) to 1.0 (first result is relevant)

    Example:
        >>> mrr = MRR(config={"relevance_threshold": 0.5})
        >>> result = mrr.evaluate([{
        ...     "query": "capital of France",
        ...     "contexts": ["Unrelated", "Paris is capital", "More info"],
        ...     "relevance_scores": [0.1, 0.9, 0.3]
        ... }])
        >>> print(result.eval_results[0].output)  # 0.5 (found at position 2)
    """

    @property
    def metric_name(self) -> str:
        return "mrr"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.5)

    def compute_one(self, inputs: RAGRankingInput) -> Dict[str, Any]:
        scores = inputs.relevance_scores

        if not scores:
            return {
                "output": 0.0,
                "reason": "No relevance scores provided",
            }

        # Find first relevant result
        for i, score in enumerate(scores, 1):
            if score >= self.relevance_threshold:
                reciprocal_rank = 1.0 / i
                return {
                    "output": round(reciprocal_rank, 4),
                    "reason": f"First relevant result at position {i}",
                    "first_relevant_position": i,
                    "first_relevant_score": round(score, 4),
                    "threshold": self.relevance_threshold,
                }

        return {
            "output": 0.0,
            "reason": f"No results above relevance threshold ({self.relevance_threshold})",
            "first_relevant_position": None,
            "threshold": self.relevance_threshold,
        }


class PrecisionAtK(BaseMetric[RAGRankingInput]):
    """
    Precision@K - fraction of top-K results that are relevant.

    Simple metric for evaluating retrieval quality at a fixed cutoff.

    Formula:
        Precision@K = |relevant in top K| / K

    Score: 0.0 (no relevant in top K) to 1.0 (all top K are relevant)
    """

    @property
    def metric_name(self) -> str:
        return "precision_at_k"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k = self.config.get("k", 5)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.5)

    def compute_one(self, inputs: RAGRankingInput) -> Dict[str, Any]:
        scores = inputs.relevance_scores

        if not scores:
            return {
                "output": 0.0,
                "reason": "No relevance scores provided",
            }

        k = min(self.k, len(scores))
        top_k_scores = scores[:k]

        relevant_count = sum(
            1 for score in top_k_scores if score >= self.relevance_threshold
        )

        precision = relevant_count / k

        return {
            "output": round(precision, 4),
            "reason": f"Precision@{k}={precision:.3f}, {relevant_count}/{k} relevant",
            "k": k,
            "relevant_count": relevant_count,
            "threshold": self.relevance_threshold,
        }


class RecallAtK(BaseMetric[RAGRankingInput]):
    """
    Recall@K - fraction of all relevant results that appear in top K.

    Measures how many of the relevant items were retrieved.

    Formula:
        Recall@K = |relevant in top K| / |total relevant|

    Score: 0.0 (no relevant recalled) to 1.0 (all relevant in top K)
    """

    @property
    def metric_name(self) -> str:
        return "recall_at_k"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k = self.config.get("k", 5)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.5)

    def compute_one(self, inputs: RAGRankingInput) -> Dict[str, Any]:
        scores = inputs.relevance_scores

        if not scores:
            return {
                "output": 0.0,
                "reason": "No relevance scores provided",
            }

        # Count total relevant
        total_relevant = sum(
            1 for score in scores if score >= self.relevance_threshold
        )

        if total_relevant == 0:
            return {
                "output": 1.0,  # Trivially recalled all (none) relevant
                "reason": "No relevant items to recall",
                "k": self.k,
                "total_relevant": 0,
            }

        k = min(self.k, len(scores))
        top_k_scores = scores[:k]

        recalled = sum(
            1 for score in top_k_scores if score >= self.relevance_threshold
        )

        recall = recalled / total_relevant

        return {
            "output": round(recall, 4),
            "reason": f"Recall@{k}={recall:.3f}, {recalled}/{total_relevant} recalled",
            "k": k,
            "recalled": recalled,
            "total_relevant": total_relevant,
            "threshold": self.relevance_threshold,
        }
