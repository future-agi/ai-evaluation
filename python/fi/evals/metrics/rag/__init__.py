"""
RAG (Retrieval-Augmented Generation) Evaluation Metrics.

Comprehensive metrics for evaluating RAG systems, including:
- Retrieval metrics (recall, precision, ranking)
- Generation metrics (faithfulness, relevancy, groundedness)
- Advanced metrics (multi-hop reasoning, source attribution)

Quick Start:
    from fi.evals.metrics.rag import RAGScore, ContextRecall, ContextPrecision

    # Comprehensive evaluation
    rag_score = RAGScore()
    result = rag_score.evaluate([{
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "contexts": ["Paris is the capital of France."],
        "reference": "Paris"
    }])
    print(f"RAG Score: {result.eval_results[0].output}")

    # Individual metrics
    recall = ContextRecall()
    precision = ContextPrecision()
"""

# Types
from .types import (
    RAGInput,
    RAGRetrievalInput,
    RAGRankingInput,
    RAGMultiHopInput,
    NoiseSensitivityInput,
    SourceAttributionInput,
    ContextUtilizationInput,
    AnswerRelevancyInput,
)

# Retrieval metrics
from .retrieval import (
    ContextRecall,
    ContextPrecision,
    ContextEntityRecall,
    NoiseSensitivity,
    NDCG,
    MRR,
)

# Generation metrics
from .generation import (
    AnswerRelevancy,
    ContextUtilization,
    Groundedness,
    RAGFaithfulness,
)

# Advanced metrics
from .advanced import (
    MultiHopReasoning,
    SourceAttribution,
)

# Comprehensive scorers
from .rag_score import RAGScore, RAGScoreDetailed

# Utilities (for advanced users)
from .utils import (
    NLILabel,
    check_entailment,
    check_claim_supported,
    extract_entities,
    extract_claims,
    compute_semantic_similarity,
)

__all__ = [
    # Types
    "RAGInput",
    "RAGRetrievalInput",
    "RAGRankingInput",
    "RAGMultiHopInput",
    "NoiseSensitivityInput",
    "SourceAttributionInput",
    "ContextUtilizationInput",
    "AnswerRelevancyInput",
    # Retrieval metrics
    "ContextRecall",
    "ContextPrecision",
    "ContextEntityRecall",
    "NoiseSensitivity",
    "NDCG",
    "MRR",
    # Generation metrics
    "AnswerRelevancy",
    "ContextUtilization",
    "Groundedness",
    "RAGFaithfulness",
    # Advanced metrics
    "MultiHopReasoning",
    "SourceAttribution",
    # Comprehensive scorers
    "RAGScore",
    "RAGScoreDetailed",
    # Utilities
    "NLILabel",
    "check_entailment",
    "check_claim_supported",
    "extract_entities",
    "extract_claims",
    "compute_semantic_similarity",
]
