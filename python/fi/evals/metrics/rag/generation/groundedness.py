"""
Groundedness Metric.

Ensures responses are grounded in the provided context,
with detailed claim-level analysis. Uses the shared NLI pipeline
for single-call claim verification (support + contradiction).
"""

from typing import Any, Dict, List, Optional

from ...base_metric import BaseMetric
from ...hallucination.nli import NLILabel, nli_score_for_claim
from ..types import RAGInput
from ..utils import extract_claims


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

    supports_llm_judge = True
    judge_description = (
        "Whether the response is grounded in the provided context. "
        "Contradictions are penalized more heavily than unsupported claims."
    )

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
                "output": 0.0,
                "reason": "No verifiable claims in response",
                "claims_analyzed": 0,
            }

        # Analyze each claim — single NLI call detects both support and contradiction
        supported = 0
        unsupported = 0
        contradicted = 0
        claim_details = []

        for claim in claims:
            label, score, best_ctx = nli_score_for_claim(claim, contexts)

            if label == NLILabel.CONTRADICTION and score > 0.5:
                status = "contradicted"
                contradicted += 1
            elif label == NLILabel.ENTAILMENT and score >= self.support_threshold:
                status = "supported"
                supported += 1
            else:
                status = "unsupported"
                unsupported += 1

            claim_details.append({
                "claim": claim[:100] + "..." if len(claim) > 100 else claim,
                "status": status,
                "nli_score": round(score, 3),
                "matched_context": best_ctx[:80] + "..." if best_ctx and len(best_ctx) > 80 else best_ctx,
            })

        # Calculate groundedness score — contradictions penalized more heavily
        total = len(claims)
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
