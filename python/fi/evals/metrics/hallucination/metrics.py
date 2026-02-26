"""
Hallucination Detection Metrics.

NLI-based evaluation for detecting hallucinations in LLM outputs.
Uses the NLI layer (nli.py) for semantic entailment checking,
with heuristic fallback when transformers is not installed.

Metrics:
- Faithfulness — proportion of claims supported by context
- ClaimSupport — per-claim entailment detail
- FactualConsistency — NLI score against a reference
- ContradictionDetection — NLI contradiction detection
- HallucinationScore — composite (sentinel + NLI)
"""

from typing import Any, Dict, List, Optional

from ..base_metric import BaseMetric
from .types import HallucinationInput, FactualConsistencyInput, Claim
from .nli import NLILabel, check_entailment, check_contradiction, nli_score_for_claim
from .detector import extract_claims
from .sentinel import HallucinationSentinel


# Neutral claims get partial credit in scoring
_NEUTRAL_SCORE = 0.4


class Faithfulness(BaseMetric[HallucinationInput]):
    """
    Evaluates response faithfulness to provided context.

    Measures the proportion of claims in the response that are
    supported by the context, using NLI entailment checking.

    Returns score from 0.0 (all hallucinated) to 1.0 (fully faithful).
    """

    supports_llm_judge = True
    judge_description = "Whether every claim in the output is supported by the provided context."

    @property
    def metric_name(self) -> str:
        return "faithfulness"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.support_threshold = self.config.get("support_threshold", 0.5)

    def compute_one(self, inputs: HallucinationInput) -> Dict[str, Any]:
        claims = inputs.claims or extract_claims(inputs.response)

        if not claims:
            return {
                "output": 1.0,
                "reason": "No verifiable claims found in response.",
            }

        contexts = inputs.context if isinstance(inputs.context, list) else [inputs.context]

        supported = 0
        unsupported = 0
        neutral_count = 0
        claim_details = []

        for claim in claims:
            label, score, best_ctx = nli_score_for_claim(claim.text, contexts)

            if label == NLILabel.ENTAILMENT and score >= self.support_threshold:
                supported += 1
                status = "supported"
            elif label == NLILabel.CONTRADICTION:
                unsupported += 1
                status = "contradicted"
            elif label == NLILabel.NEUTRAL:
                neutral_count += 1
                status = "neutral"
            else:
                unsupported += 1
                status = "unsupported"

            claim_details.append({
                "claim": claim.text[:100],
                "label": status,
                "score": round(score, 3),
            })

        total = len(claims)
        # Supported = 1.0, neutral = partial, unsupported/contradicted = 0.0
        faithfulness_score = (supported + neutral_count * _NEUTRAL_SCORE) / total

        reason_parts = [f"{supported}/{total} claims supported"]
        if neutral_count > 0:
            reason_parts.append(f"{neutral_count} neutral")
        if unsupported > 0:
            reason_parts.append(f"{unsupported} unsupported")

        return {
            "output": round(faithfulness_score, 4),
            "reason": ". ".join(reason_parts),
            "details": claim_details,
        }


class ClaimSupport(BaseMetric[HallucinationInput]):
    """
    Evaluates support level for individual claims via NLI.

    Returns average NLI-based support score from 0.0 to 1.0.
    Neutral claims receive partial credit.
    """

    supports_llm_judge = True
    judge_description = "Per-claim entailment analysis — how well each claim is supported by context."

    @property
    def metric_name(self) -> str:
        return "claim_support"

    def compute_one(self, inputs: HallucinationInput) -> Dict[str, Any]:
        claims = inputs.claims or extract_claims(inputs.response)

        if not claims:
            return {
                "output": 1.0,
                "reason": "No claims to evaluate.",
            }

        contexts = inputs.context if isinstance(inputs.context, list) else [inputs.context]

        total_score = 0.0
        claim_results = []

        for claim in claims:
            label, score, _ = nli_score_for_claim(claim.text, contexts)

            if label == NLILabel.ENTAILMENT:
                claim_score = score
            elif label == NLILabel.NEUTRAL:
                claim_score = score * _NEUTRAL_SCORE
            else:
                claim_score = 0.0

            total_score += claim_score
            claim_results.append({
                "claim": claim.text[:80],
                "support_score": round(claim_score, 3),
                "nli_label": label.value,
                "raw_nli_score": round(score, 3),
            })

        avg_score = total_score / len(claims)

        return {
            "output": round(avg_score, 4),
            "reason": f"Average claim support: {avg_score:.1%} across {len(claims)} claims",
            "claims": claim_results,
        }


class FactualConsistency(BaseMetric[FactualConsistencyInput]):
    """
    Evaluates factual consistency between response and reference using NLI.

    Checks each claim against the reference text for entailment or contradiction.

    Returns score from 0.0 to 1.0.
    """

    supports_llm_judge = True
    judge_description = "Whether the output is factually consistent with a reference text."

    @property
    def metric_name(self) -> str:
        return "factual_consistency"

    def compute_one(self, inputs: FactualConsistencyInput) -> Dict[str, Any]:
        if not inputs.reference:
            return {
                "output": 0.0,
                "reason": "No reference provided for factual consistency check.",
            }

        claims = inputs.claims or extract_claims(inputs.response)

        if not claims:
            return {
                "output": 1.0,
                "reason": "No factual claims found in response.",
            }

        consistent = 0
        inconsistent = 0
        neutral_count = 0
        details = []

        for claim in claims:
            label, score = check_entailment(inputs.reference, claim.text)
            is_contradicted, contradiction_conf = check_contradiction(
                claim.text, inputs.reference
            )

            if is_contradicted and contradiction_conf > 0.5:
                inconsistent += 1
                details.append({"claim": claim.text[:80], "status": "contradicted"})
            elif label == NLILabel.ENTAILMENT:
                consistent += 1
                details.append({"claim": claim.text[:80], "status": "consistent"})
            elif label == NLILabel.NEUTRAL:
                neutral_count += 1
                details.append({"claim": claim.text[:80], "status": "neutral"})
            else:
                details.append({"claim": claim.text[:80], "status": "unverified"})

        total = len(claims)
        # Contradictions heavily penalized
        if inconsistent > 0:
            score = max(0.0, (consistent + neutral_count * _NEUTRAL_SCORE - inconsistent * 2) / total)
        else:
            score = (consistent + neutral_count * _NEUTRAL_SCORE) / total

        return {
            "output": round(score, 4),
            "reason": f"Consistent: {consistent}, Contradicted: {inconsistent}, Neutral: {neutral_count}, Total: {total}",
            "details": details,
        }


class ContradictionDetection(BaseMetric[HallucinationInput]):
    """
    Detects contradictions between response and context using NLI.

    Uses the NLI CONTRADICTION label for detection instead of
    crude negation removal heuristics.

    Returns 1.0 if no contradictions, 0.0 if contradictions found.
    """

    supports_llm_judge = True
    judge_description = "Whether the output contradicts the provided context."

    @property
    def metric_name(self) -> str:
        return "contradiction_detection"

    def compute_one(self, inputs: HallucinationInput) -> Dict[str, Any]:
        claims = inputs.claims or extract_claims(inputs.response)

        if not claims:
            return {
                "output": 1.0,
                "reason": "No claims to check for contradictions.",
            }

        contexts = inputs.context if isinstance(inputs.context, list) else [inputs.context]
        full_context = " ".join(contexts)

        contradictions = []

        for claim in claims:
            is_contradicted, confidence = check_contradiction(claim.text, full_context)
            if is_contradicted:
                contradictions.append({
                    "claim": claim.text[:100],
                    "confidence": round(confidence, 3),
                })

        if contradictions:
            return {
                "output": 0.0,
                "reason": f"Found {len(contradictions)} contradiction(s) in response.",
                "contradictions": contradictions,
            }

        return {
            "output": 1.0,
            "reason": "No contradictions detected.",
        }


class HallucinationScore(BaseMetric[HallucinationInput]):
    """
    Comprehensive hallucination detection score.

    Combines:
    - Sentinel pre-screening (fast risk assessment)
    - NLI-based claim support analysis
    - NLI contradiction detection
    - Neutral handling with partial credit

    Returns score from 0.0 (severe hallucination) to 1.0 (no hallucination).
    """

    supports_llm_judge = True
    judge_description = (
        "Composite hallucination detection — how much of the output is fabricated "
        "vs grounded in context."
    )

    @property
    def metric_name(self) -> str:
        return "hallucination_score"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.support_weight = self.config.get("support_weight", 0.6)
        self.contradiction_weight = self.config.get("contradiction_weight", 0.4)
        self._sentinel = HallucinationSentinel()

    def compute_one(self, inputs: HallucinationInput) -> Dict[str, Any]:
        claims = inputs.claims or extract_claims(inputs.response)

        if not claims:
            return {
                "output": 1.0,
                "reason": "No verifiable claims in response.",
                "claims_analyzed": 0,
                "supported": 0,
                "unsupported": 0,
                "contradicted": 0,
            }

        contexts = inputs.context if isinstance(inputs.context, list) else [inputs.context]

        # Sentinel screening
        risk_level, _ = self._sentinel.screen(inputs.response, contexts[0])

        supported = 0
        unsupported = 0
        contradicted = 0
        neutral_count = 0
        total_support_score = 0.0
        details = []

        for claim in claims:
            label, score, _ = nli_score_for_claim(claim.text, contexts)

            if label == NLILabel.ENTAILMENT:
                total_support_score += score
                status = "supported"
                supported += 1
            elif label == NLILabel.CONTRADICTION:
                total_support_score += 0.0
                status = "contradicted"
                contradicted += 1
            else:
                # Neutral: partial credit
                total_support_score += score * _NEUTRAL_SCORE
                status = "neutral"
                neutral_count += 1

            details.append({
                "claim": claim.text[:80],
                "status": status,
                "support_score": round(score, 3),
            })

        total = len(claims)
        avg_support = total_support_score / total
        contradiction_penalty = contradicted / total

        final_score = (
            self.support_weight * avg_support
            + self.contradiction_weight * (1.0 - contradiction_penalty)
        )

        # Sentinel adjustment: high risk with low support is worse
        if risk_level == "high" and avg_support < 0.3:
            final_score *= 0.8

        severity = "none"
        if final_score < 0.3:
            severity = "severe"
        elif final_score < 0.6:
            severity = "moderate"
        elif final_score < 0.8:
            severity = "minor"

        return {
            "output": round(final_score, 4),
            "reason": f"Hallucination severity: {severity}. "
                     f"Supported: {supported}, Neutral: {neutral_count}, "
                     f"Unsupported: {unsupported}, Contradicted: {contradicted}",
            "claims_analyzed": total,
            "supported": supported,
            "unsupported": unsupported,
            "contradicted": contradicted,
            "sentinel_risk": risk_level,
            "details": details,
        }
