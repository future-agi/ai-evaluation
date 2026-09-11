"""
Types for Hallucination Detection.

These types support NLI-based and semantic analysis for
detecting hallucinations in LLM outputs.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from ...types import BaseMetricInput


class Claim(BaseModel):
    """Represents a single claim extracted from text."""

    text: str = Field(..., description="The claim text")
    source_span: Optional[str] = Field(
        default=None,
        description="Original text span the claim was extracted from"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for claim extraction (0-1)"
    )


class HallucinationInput(BaseMetricInput):
    """
    Input for hallucination detection metrics.

    Evaluates whether the response contains claims not supported
    by the provided context/source.
    """

    # The LLM response to check for hallucinations
    response: str = Field(
        ...,
        description="The LLM response to evaluate for hallucinations."
    )

    # The source/context that the response should be faithful to
    context: Union[str, List[str]] = Field(
        ...,
        description="Source context(s) the response should be grounded in."
    )

    # Optional: pre-extracted claims from response
    claims: Optional[List[Claim]] = Field(
        default=None,
        description="Pre-extracted claims from response. If not provided, claims are extracted automatically."
    )

    # Optional: the query that generated the response
    query: Optional[str] = Field(
        default=None,
        description="The original query/question that generated the response."
    )


class ClaimExtractionInput(BaseMetricInput):
    """
    Input for claim extraction from text.

    Extracts atomic, verifiable claims from a text passage.
    """

    response: str = Field(
        ...,
        description="The text to extract claims from."
    )

    # Extraction granularity
    granularity: Literal["sentence", "clause", "atomic"] = Field(
        default="sentence",
        description="Granularity of claim extraction: sentence, clause, or atomic (finest)."
    )


class FactualConsistencyInput(BaseMetricInput):
    """
    Input for factual consistency checking.

    Evaluates whether claims in the response are consistent with
    known facts or a reference text.
    """

    response: str = Field(
        ...,
        description="The response to check for factual consistency."
    )

    # Reference for fact-checking
    reference: Optional[str] = Field(
        default=None,
        description="Reference text containing ground truth facts."
    )

    # Pre-extracted claims
    claims: Optional[List[Claim]] = Field(
        default=None,
        description="Pre-extracted claims from the response."
    )


class NLIResult(BaseModel):
    """Result of Natural Language Inference classification."""

    premise: str = Field(..., description="The premise (context)")
    hypothesis: str = Field(..., description="The hypothesis (claim)")
    label: Literal["entailment", "neutral", "contradiction"] = Field(
        ...,
        description="NLI classification label"
    )
    scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability scores for each label"
    )


class HallucinationResult(BaseModel):
    """Detailed result of hallucination detection."""

    score: float = Field(..., description="Overall hallucination score (0=hallucinated, 1=faithful)")
    claims_analyzed: int = Field(..., description="Number of claims analyzed")
    supported_claims: int = Field(..., description="Number of claims supported by context")
    unsupported_claims: int = Field(..., description="Number of unsupported claims")
    contradicted_claims: int = Field(..., description="Number of contradicted claims")
    claim_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed analysis for each claim"
    )
