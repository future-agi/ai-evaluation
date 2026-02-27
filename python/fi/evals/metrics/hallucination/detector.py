"""
HallucinationDetector — main orchestrator for hallucination detection.

Combines sentinel screening, claim extraction, and NLI classification
to provide comprehensive hallucination analysis.
"""

import re
from typing import Any, Dict, List, Optional

from .types import Claim, HallucinationResult
from .nli import NLILabel, check_entailment, nli_score_for_claim
from .sentinel import HallucinationSentinel


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


_META_PATTERNS = [
    r"^(I think|I believe|In my opinion|It seems|Perhaps)",
    r"^(Here is|Here are|The following|Below is)",
    r"^(Note:|Important:|Warning:|Disclaimer:)",
]


def extract_claims(text: str) -> List[Claim]:
    """
    Extract verifiable claims from text.

    Uses sentence-level extraction with heuristic filtering
    to remove questions, meta-statements, and very short fragments.
    """
    sentences = _split_into_sentences(text)

    claims = []
    for sentence in sentences:
        if len(sentence.split()) < 3:
            continue
        if sentence.strip().endswith("?"):
            continue
        if any(re.match(p, sentence, re.IGNORECASE) for p in _META_PATTERNS):
            continue

        claims.append(
            Claim(text=sentence, source_span=sentence, confidence=1.0)
        )

    return claims


class HallucinationDetector:
    """
    Main orchestrator for hallucination detection.

    Pipeline:
    1. Sentinel screens for obvious risk level
    2. Claims are extracted from the response
    3. Each claim is classified via NLI against the context
    4. Results are aggregated into a HallucinationResult
    """

    def __init__(self, sentinel: Optional[HallucinationSentinel] = None):
        self.sentinel = sentinel or HallucinationSentinel()

    def detect(
        self,
        response: str,
        context: str,
        claims: Optional[List[Claim]] = None,
    ) -> HallucinationResult:
        """
        Run full hallucination detection pipeline.

        Args:
            response: The LLM response to check
            context: The source context
            claims: Optional pre-extracted claims

        Returns:
            HallucinationResult with detailed analysis
        """
        contexts = [context] if isinstance(context, str) else context

        # Step 1: sentinel screening
        risk_level, sentinel_details = self.sentinel.screen(response, contexts[0])

        # Step 2: extract claims
        if claims is None:
            claims = extract_claims(response)

        if not claims:
            return HallucinationResult(
                score=1.0,
                claims_analyzed=0,
                supported_claims=0,
                unsupported_claims=0,
                contradicted_claims=0,
                claim_details=[],
            )

        # Step 3: classify each claim via NLI
        supported = 0
        unsupported = 0
        contradicted = 0
        claim_details = []

        for claim in claims:
            label, score, best_ctx = nli_score_for_claim(claim.text, contexts)

            if label == NLILabel.ENTAILMENT:
                status = "supported"
                supported += 1
            elif label == NLILabel.CONTRADICTION:
                status = "contradicted"
                contradicted += 1
            else:
                status = "unsupported"
                unsupported += 1

            claim_details.append({
                "claim": claim.text[:100],
                "status": status,
                "nli_score": round(score, 3),
                "best_context": best_ctx,
            })

        # Step 4: aggregate
        total = len(claims)
        support_ratio = supported / total
        contradiction_penalty = contradicted / total

        # Weighted score: support is good, contradictions are heavily penalized
        overall_score = 0.6 * support_ratio + 0.4 * (1.0 - contradiction_penalty)

        # Sentinel risk can further adjust: high risk with no support is worse
        if risk_level == "high" and support_ratio < 0.3:
            overall_score *= 0.8

        return HallucinationResult(
            score=round(overall_score, 4),
            claims_analyzed=total,
            supported_claims=supported,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            claim_details=claim_details,
        )
