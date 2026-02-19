"""
Groundedness Metric.

Ensures responses are grounded in the provided context,
with detailed claim-level analysis.
"""

from typing import Any, Dict, List, Optional

from ...base_metric import BaseMetric
from ..types import RAGInput
from ..utils import (
    extract_claims,
    check_claim_supported,
    check_contradiction,
    NLILabel,
)


class Groundedness(BaseMetric[RAGInput]):
    """
    Evaluates whether the response is grounded in the provided context.

    Similar to faithfulness but with more detailed analysis:
    - Claim-level breakdown
    - Contradiction detection
    - Confidence scores per claim

    Formula:
        Groundedness = (Supported claims - Contradicted claims) / Total claims

    Score: 0.0 (not grounded) to 1.0 (fully grounded)

    Example:
        >>> groundedness = Groundedness()
        >>> result = groundedness.evaluate([{
        ...     "query": "When was Einstein born?",
        ...     "response": "Einstein was born in 1879 in Germany.",
        ...     "contexts": ["Albert Einstein was born on March 14, 1879, in Ulm, Germany."]
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "groundedness"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.support_threshold = self.config.get("support_threshold", 0.5)
        self.contradiction_penalty = self.config.get("contradiction_penalty", 2.0)

    def compute_one(self, inputs: RAGInput) -> Dict[str, Any]:
        response = inputs.response
        contexts = inputs.contexts

        if not response or not response.strip():
            return {
                "output": 0.0,
                "reason": "Empty response",
            }

        if not contexts:
            return {
                "output": 0.0,
                "reason": "No context provided - cannot assess groundedness",
            }

        # Extract claims from response
        claims = extract_claims(response)

        if not claims:
            return {
                "output": 1.0,
                "reason": "No verifiable claims in response",
                "claims_analyzed": 0,
            }

        # Analyze each claim
        supported = 0
        unsupported = 0
        contradicted = 0
        claim_details = []

        combined_context = " ".join(contexts)

        for claim in claims:
            # Check support
            is_supported, support_score, best_context = check_claim_supported(
                claim, contexts, self.support_threshold
            )

            # Check contradiction
            is_contradicted, contradiction_conf = check_contradiction(
                claim, combined_context
            )

            # Determine status
            if is_contradicted and contradiction_conf > 0.5:
                status = "contradicted"
                contradicted += 1
            elif is_supported:
                status = "supported"
                supported += 1
            else:
                status = "unsupported"
                unsupported += 1

            claim_details.append({
                "claim": claim[:100] + "..." if len(claim) > 100 else claim,
                "status": status,
                "support_score": round(support_score, 3),
                "matched_context": best_context[:80] + "..." if best_context and len(best_context) > 80 else best_context,
            })

        # Calculate groundedness score
        # Contradictions are penalized more heavily
        total = len(claims)
        if total == 0:
            groundedness = 1.0
        else:
            # Score = (supported - penalty * contradicted) / total
            effective_score = supported - (self.contradiction_penalty * contradicted)
            groundedness = max(0.0, effective_score / total)

        # Determine severity
        if groundedness >= 0.8:
            severity = "well_grounded"
        elif groundedness >= 0.5:
            severity = "partially_grounded"
        elif groundedness >= 0.2:
            severity = "poorly_grounded"
        else:
            severity = "ungrounded"

        return {
            "output": round(groundedness, 4),
            "reason": f"{severity}: {supported}/{total} supported, {contradicted} contradicted",
            "severity": severity,
            "claims_analyzed": total,
            "supported": supported,
            "unsupported": unsupported,
            "contradicted": contradicted,
            "details": claim_details,
        }
