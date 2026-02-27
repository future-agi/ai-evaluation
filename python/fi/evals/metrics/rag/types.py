"""
RAG Evaluation Input/Output Types.

Defines strongly-typed inputs for all RAG metrics.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

from ...types import BaseMetricInput


class RAGInput(BaseMetricInput):
    """Standard RAG evaluation input for generation metrics."""

    query: str = Field(..., description="The user query/question")
    response: str = Field(..., description="The generated response")
    contexts: List[str] = Field(..., description="Retrieved context passages")
    reference: Optional[str] = Field(None, description="Ground truth answer")


class RAGRetrievalInput(BaseMetricInput):
    """Input for retrieval-only metrics (recall, precision, entity recall)."""

    query: str = Field(..., description="The user query")
    contexts: List[str] = Field(..., description="Retrieved context passages")
    reference: str = Field(..., description="Ground truth/reference text")
    relevance_labels: Optional[List[int]] = Field(
        None, description="Binary relevance labels (1=relevant, 0=irrelevant)"
    )


class RAGRankingInput(BaseMetricInput):
    """Input for ranking metrics (NDCG, MRR)."""

    query: str = Field(..., description="The user query")
    contexts: List[str] = Field(..., description="Retrieved contexts in ranked order")
    relevance_scores: List[float] = Field(
        ..., description="Relevance scores/grades for each context (0-1 or graded)"
    )


class NoiseSensitivityInput(BaseMetricInput):
    """Input for noise sensitivity evaluation."""

    query: str = Field(..., description="The user query")
    response_clean: str = Field(
        ..., description="Response generated with only relevant contexts"
    )
    response_noisy: str = Field(
        ..., description="Response generated with relevant + irrelevant contexts"
    )
    relevant_contexts: List[str] = Field(..., description="Relevant context passages")
    irrelevant_contexts: List[str] = Field(
        ..., description="Irrelevant/noisy context passages"
    )
    reference: Optional[str] = Field(None, description="Ground truth answer")


class RAGMultiHopInput(BaseMetricInput):
    """Input for multi-hop reasoning evaluation."""

    query: str = Field(..., description="Complex query requiring multiple hops")
    response: str = Field(..., description="Generated response")
    contexts: List[str] = Field(..., description="Retrieved context passages")
    hop_chain: Optional[List[str]] = Field(
        None, description="Expected reasoning chain across contexts"
    )
    reference: Optional[str] = Field(None, description="Ground truth answer")


class SourceAttributionInput(BaseMetricInput):
    """Input for source attribution evaluation."""

    response: str = Field(..., description="Generated response with citations")
    contexts: List[str] = Field(..., description="Source documents")
    citation_format: str = Field(
        "bracketed", description="Citation format: bracketed [1], inline, footnote"
    )
    require_citations: bool = Field(
        True, description="Whether citations are required for all claims"
    )


class ContextUtilizationInput(BaseMetricInput):
    """Input for context utilization evaluation."""

    query: str = Field(..., description="The user query")
    response: str = Field(..., description="The generated response")
    contexts: List[str] = Field(..., description="Retrieved context passages")


class AnswerRelevancyInput(BaseMetricInput):
    """Input for answer relevancy evaluation."""

    query: str = Field(..., description="The user query")
    response: str = Field(..., description="The generated response")
    contexts: Optional[List[str]] = Field(
        None, description="Retrieved context passages (optional)"
    )
