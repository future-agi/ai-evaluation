"""
Comprehensive RAG Score Metric.

Combines multiple RAG metrics into a single comprehensive score.
"""

from typing import Any, Dict, List, Optional

from ..base_metric import BaseMetric
from .types import RAGInput, RAGRetrievalInput
from .retrieval import ContextRecall, ContextPrecision
from .generation import AnswerRelevancy, ContextUtilization, RAGFaithfulness


class RAGScore(BaseMetric[RAGInput]):
    """
    Comprehensive RAG evaluation combining all metrics.

    Evaluates both retrieval and generation components in a single call.

    Default weights (configurable):
    - Retrieval (40%): Context Recall, Context Precision
    - Generation (40%): Faithfulness, Answer Relevancy
    - Advanced (20%): Context Utilization

    Score: 0.0 (poor RAG performance) to 1.0 (excellent RAG performance)

    Example:
        >>> rag_score = RAGScore()
        >>> result = rag_score.evaluate([{
        ...     "query": "What is the capital of France?",
        ...     "response": "The capital of France is Paris.",
        ...     "contexts": ["Paris is the capital and largest city of France."],
        ...     "reference": "Paris is the capital of France."
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "rag_score"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.weights = self.config.get("weights", {
            "retrieval": 0.4,
            "generation": 0.4,
            "advanced": 0.2,
        })
        self.include_details = self.config.get("include_details", True)

    def compute_one(self, inputs: RAGInput) -> Dict[str, Any]:
        scores = {}
        details = {}

        # Prepare retrieval input
        reference = inputs.reference or inputs.response
        retrieval_input = RAGRetrievalInput(
            query=inputs.query,
            contexts=inputs.contexts,
            reference=reference,
        )

        # Retrieval metrics
        recall_metric = ContextRecall()
        recall_result = recall_metric.compute_one(retrieval_input)
        scores["context_recall"] = recall_result["output"]
        if self.include_details:
            details["context_recall"] = recall_result.get("reason", "")

        precision_metric = ContextPrecision()
        precision_result = precision_metric.compute_one(retrieval_input)
        scores["context_precision"] = precision_result["output"]
        if self.include_details:
            details["context_precision"] = precision_result.get("reason", "")

        retrieval_score = (scores["context_recall"] + scores["context_precision"]) / 2

        # Generation metrics
        from .types import AnswerRelevancyInput, ContextUtilizationInput

        relevancy_input = AnswerRelevancyInput(
            query=inputs.query,
            response=inputs.response,
            contexts=inputs.contexts,
        )
        relevancy_metric = AnswerRelevancy()
        relevancy_result = relevancy_metric.compute_one(relevancy_input)
        scores["answer_relevancy"] = relevancy_result["output"]
        if self.include_details:
            details["answer_relevancy"] = relevancy_result.get("reason", "")

        faithfulness_metric = RAGFaithfulness()
        faithfulness_result = faithfulness_metric.compute_one(inputs)
        scores["faithfulness"] = faithfulness_result["output"]
        if self.include_details:
            details["faithfulness"] = faithfulness_result.get("reason", "")

        generation_score = (scores["answer_relevancy"] + scores["faithfulness"]) / 2

        # Advanced metrics
        utilization_input = ContextUtilizationInput(
            query=inputs.query,
            response=inputs.response,
            contexts=inputs.contexts,
        )
        utilization_metric = ContextUtilization()
        utilization_result = utilization_metric.compute_one(utilization_input)
        scores["context_utilization"] = utilization_result["output"]
        if self.include_details:
            details["context_utilization"] = utilization_result.get("reason", "")

        advanced_score = scores["context_utilization"]

        # Weighted combination
        final_score = (
            self.weights["retrieval"] * retrieval_score +
            self.weights["generation"] * generation_score +
            self.weights["advanced"] * advanced_score
        )

        result = {
            "output": round(final_score, 4),
            "reason": f"Retrieval={retrieval_score:.2f}, Generation={generation_score:.2f}, Advanced={advanced_score:.2f}",
            "retrieval_score": round(retrieval_score, 4),
            "generation_score": round(generation_score, 4),
            "advanced_score": round(advanced_score, 4),
            "component_scores": {k: round(v, 4) for k, v in scores.items()},
        }

        if self.include_details:
            result["component_details"] = details

        return result


class RAGScoreDetailed(BaseMetric[RAGInput]):
    """
    Detailed RAG evaluation with all available metrics.

    Runs all RAG metrics and provides comprehensive analysis.
    More expensive but gives complete picture.

    Components:
    - Retrieval: Recall, Precision, Entity Recall
    - Generation: Faithfulness, Relevancy, Groundedness
    - Advanced: Context Utilization, Multi-hop (if applicable)

    Score: 0.0 to 1.0
    """

    @property
    def metric_name(self) -> str:
        return "rag_score_detailed"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.retrieval_weight = self.config.get("retrieval_weight", 0.35)
        self.generation_weight = self.config.get("generation_weight", 0.35)
        self.advanced_weight = self.config.get("advanced_weight", 0.30)

    def compute_one(self, inputs: RAGInput) -> Dict[str, Any]:
        from .retrieval import ContextRecall, ContextPrecision, ContextEntityRecall
        from .generation import AnswerRelevancy, ContextUtilization, Groundedness, RAGFaithfulness
        from .types import AnswerRelevancyInput, ContextUtilizationInput

        all_scores = {}
        all_details = {}

        # Prepare inputs
        reference = inputs.reference or inputs.response
        retrieval_input = RAGRetrievalInput(
            query=inputs.query,
            contexts=inputs.contexts,
            reference=reference,
        )

        # === RETRIEVAL METRICS ===
        # Context Recall
        recall = ContextRecall().compute_one(retrieval_input)
        all_scores["context_recall"] = recall["output"]
        all_details["context_recall"] = recall

        # Context Precision
        precision = ContextPrecision().compute_one(retrieval_input)
        all_scores["context_precision"] = precision["output"]
        all_details["context_precision"] = precision

        # Context Entity Recall
        entity_recall = ContextEntityRecall().compute_one(retrieval_input)
        all_scores["context_entity_recall"] = entity_recall["output"]
        all_details["context_entity_recall"] = entity_recall

        retrieval_avg = (
            all_scores["context_recall"] +
            all_scores["context_precision"] +
            all_scores["context_entity_recall"]
        ) / 3

        # === GENERATION METRICS ===
        # Faithfulness
        faithfulness = RAGFaithfulness().compute_one(inputs)
        all_scores["faithfulness"] = faithfulness["output"]
        all_details["faithfulness"] = faithfulness

        # Answer Relevancy
        relevancy_input = AnswerRelevancyInput(
            query=inputs.query,
            response=inputs.response,
            contexts=inputs.contexts,
        )
        relevancy = AnswerRelevancy().compute_one(relevancy_input)
        all_scores["answer_relevancy"] = relevancy["output"]
        all_details["answer_relevancy"] = relevancy

        # Groundedness
        groundedness = Groundedness().compute_one(inputs)
        all_scores["groundedness"] = groundedness["output"]
        all_details["groundedness"] = groundedness

        generation_avg = (
            all_scores["faithfulness"] +
            all_scores["answer_relevancy"] +
            all_scores["groundedness"]
        ) / 3

        # === ADVANCED METRICS ===
        # Context Utilization
        util_input = ContextUtilizationInput(
            query=inputs.query,
            response=inputs.response,
            contexts=inputs.contexts,
        )
        utilization = ContextUtilization().compute_one(util_input)
        all_scores["context_utilization"] = utilization["output"]
        all_details["context_utilization"] = utilization

        advanced_avg = all_scores["context_utilization"]

        # === FINAL SCORE ===
        final_score = (
            self.retrieval_weight * retrieval_avg +
            self.generation_weight * generation_avg +
            self.advanced_weight * advanced_avg
        )

        # Determine quality level
        if final_score >= 0.8:
            quality = "excellent"
        elif final_score >= 0.6:
            quality = "good"
        elif final_score >= 0.4:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "output": round(final_score, 4),
            "reason": f"Quality: {quality} - Retrieval={retrieval_avg:.2f}, Generation={generation_avg:.2f}, Advanced={advanced_avg:.2f}",
            "quality_level": quality,
            "retrieval_average": round(retrieval_avg, 4),
            "generation_average": round(generation_avg, 4),
            "advanced_average": round(advanced_avg, 4),
            "all_scores": {k: round(v, 4) for k, v in all_scores.items()},
            "details": all_details,
        }
