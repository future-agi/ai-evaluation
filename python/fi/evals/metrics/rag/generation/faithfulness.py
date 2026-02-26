"""
RAG Faithfulness Metric.

Enhanced faithfulness evaluation specifically designed for RAG systems.
Uses the shared NLI pipeline (hallucination/nli.py) for claim verification.
"""

from typing import Any, Dict, List, Optional

from ...base_metric import BaseMetric
from ...hallucination.nli import NLILabel, nli_score_for_claim
from ..types import RAGInput
from ..utils import extract_claims, extract_atomic_claims

# Neutral claims get partial credit — consistent with hallucination/Faithfulness
_NEUTRAL_SCORE = 0.4


class RAGFaithfulness(BaseMetric[RAGInput]):
    """
    Evaluates response faithfulness to provided context.

    Measures the proportion of claims in the response that are
    supported by the context. Enhanced for RAG-specific patterns.

    Key differences from basic faithfulness:
    - RAG-aware claim extraction (filters query echoing)
    - Multi-context support checking
    - Confidence-weighted scoring

    Formula (Ragas-style):
        Faithfulness = |Claims supported by context| / |Total claims|

    Score: 0.0 (all hallucinated) to 1.0 (fully faithful)

    Example:
        >>> faithfulness = RAGFaithfulness()
        >>> result = faithfulness.evaluate([{
        ...     "query": "What is the capital of France?",
        ...     "response": "The capital of France is Paris.",
        ...     "contexts": ["Paris is the capital and largest city of France."]
        ... }])
    """

    supports_llm_judge = True
    judge_description = (
        "Whether every claim in the RAG response is supported by the retrieved context. "
        "Filters query echoing and handles multi-context retrieval."
    )

    @property
    def metric_name(self) -> str:
        return "rag_faithfulness"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.support_threshold = self.config.get("support_threshold", 0.5)
        self.use_atomic_claims = self.config.get("use_atomic_claims", False)
        self.filter_query_echo = self.config.get("filter_query_echo", True)

    def compute_one(self, inputs: RAGInput) -> Dict[str, Any]:
        response = inputs.response
        contexts = inputs.contexts
        query = inputs.query

        if not response or not response.strip():
            return {
                "output": 0.0,
                "reason": "Empty response",
            }

        if not contexts:
            return {
                "output": 0.0,
                "reason": "No context provided - cannot verify faithfulness",
            }

        # Extract claims
        if self.use_atomic_claims:
            claims = extract_atomic_claims(response)
        else:
            claims = extract_claims(response)

        # Filter out query echoing if enabled
        if self.filter_query_echo and query:
            claims = self._filter_query_echo(claims, query)

        if not claims:
            return {
                "output": 0.0,
                "reason": "No verifiable claims in response",
                "claims_analyzed": 0,
            }

        # Verify each claim against contexts using shared NLI pipeline
        supported = 0
        contradicted = 0
        neutral_count = 0
        claim_results = []

        for claim in claims:
            label, score, best_ctx = nli_score_for_claim(claim, contexts)

            if label == NLILabel.ENTAILMENT and score >= self.support_threshold:
                supported += 1
                status = "supported"
            elif label == NLILabel.CONTRADICTION:
                contradicted += 1
                status = "contradicted"
            elif label == NLILabel.NEUTRAL:
                neutral_count += 1
                status = "neutral"
            else:
                status = "unsupported"

            claim_results.append({
                "claim": claim[:100] + "..." if len(claim) > 100 else claim,
                "status": status,
                "nli_score": round(score, 3),
            })

        total = len(claims)
        unsupported = total - supported - contradicted - neutral_count
        # Supported = 1.0, neutral = partial credit, contradicted/unsupported = 0.0
        faithfulness = (supported + neutral_count * _NEUTRAL_SCORE) / total

        reason_parts = [f"{supported}/{total} supported"]
        if neutral_count > 0:
            reason_parts.append(f"{neutral_count} neutral")
        if contradicted > 0:
            reason_parts.append(f"{contradicted} contradicted")

        return {
            "output": round(faithfulness, 4),
            "reason": ", ".join(reason_parts),
            "claims_analyzed": total,
            "supported_claims": supported,
            "contradicted_claims": contradicted,
            "neutral_claims": neutral_count,
            "claim_results": claim_results,
        }

    def _filter_query_echo(self, claims: List[str], query: str) -> List[str]:
        """Filter out claims that are just echoing the query."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        filtered = []
        for claim in claims:
            claim_lower = claim.lower()
            claim_words = set(claim_lower.split())

            # Skip if claim is mostly query words
            overlap = len(query_words & claim_words)
            if claim_words and overlap / len(claim_words) < 0.8:
                filtered.append(claim)

        return filtered if filtered else claims  # Return original if all filtered


class RAGFaithfulnessWithReference(BaseMetric[RAGInput]):
    """
    Faithfulness evaluation that also considers reference answer.

    Checks faithfulness to both context AND reference, useful when
    ground truth is available.

    Score: 0.0 to 1.0
    """

    @property
    def metric_name(self) -> str:
        return "rag_faithfulness_with_reference"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.context_weight = self.config.get("context_weight", 0.7)
        self.reference_weight = self.config.get("reference_weight", 0.3)
        self.support_threshold = self.config.get("support_threshold", 0.5)

    def compute_one(self, inputs: RAGInput) -> Dict[str, Any]:
        response = inputs.response
        contexts = inputs.contexts
        reference = inputs.reference

        if not response or not response.strip():
            return {"output": 0.0, "reason": "Empty response"}

        if not contexts and not reference:
            return {"output": 0.0, "reason": "No context or reference provided"}

        claims = extract_claims(response)

        if not claims:
            return {
                "output": 0.0,
                "reason": "No verifiable claims",
                "claims_analyzed": 0,
            }

        # Check faithfulness to context via NLI
        context_supported = 0
        if contexts:
            for claim in claims:
                label, score, _ = nli_score_for_claim(claim, contexts)
                if label == NLILabel.ENTAILMENT and score >= self.support_threshold:
                    context_supported += 1
                elif label == NLILabel.NEUTRAL:
                    context_supported += _NEUTRAL_SCORE

        context_faithfulness = context_supported / len(claims) if contexts else 0.0

        # Check faithfulness to reference via NLI
        reference_supported = 0
        if reference:
            for claim in claims:
                label, score, _ = nli_score_for_claim(claim, [reference])
                if label == NLILabel.ENTAILMENT and score >= self.support_threshold:
                    reference_supported += 1
                elif label == NLILabel.NEUTRAL:
                    reference_supported += _NEUTRAL_SCORE

        reference_faithfulness = reference_supported / len(claims) if reference else 0.0

        # Combined score
        if contexts and reference:
            combined = (
                self.context_weight * context_faithfulness +
                self.reference_weight * reference_faithfulness
            )
        elif contexts:
            combined = context_faithfulness
        else:
            combined = reference_faithfulness

        return {
            "output": round(combined, 4),
            "reason": f"Context: {context_faithfulness:.2f}, Reference: {reference_faithfulness:.2f}",
            "context_faithfulness": round(context_faithfulness, 4),
            "reference_faithfulness": round(reference_faithfulness, 4),
            "claims_analyzed": len(claims),
        }
