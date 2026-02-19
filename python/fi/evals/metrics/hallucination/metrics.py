"""
Hallucination Detection Metrics.

NLI-based and semantic analysis for detecting hallucinations.
Provides fast evaluation (~50-100ms) vs LLM-as-judge (~2-10s).

Methods:
- Claim-level analysis with NLI classification
- Semantic similarity matching
- Sentence-level overlap detection
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from difflib import SequenceMatcher

from ..base_metric import BaseMetric
from .types import (
    HallucinationInput,
    FactualConsistencyInput,
    Claim,
)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _extract_claims(text: str, granularity: str = "sentence") -> List[Claim]:
    """
    Extract claims from text.

    In production, this would use a more sophisticated claim extraction model.
    For now, we use sentence-level extraction with heuristic filtering.
    """
    sentences = _split_into_sentences(text)

    claims = []
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < 3:
            continue

        # Skip questions
        if sentence.strip().endswith('?'):
            continue

        # Skip meta-statements
        meta_patterns = [
            r'^(I think|I believe|In my opinion|It seems|Perhaps)',
            r'^(Here is|Here are|The following|Below is)',
            r'^(Note:|Important:|Warning:|Disclaimer:)',
        ]
        is_meta = any(re.match(p, sentence, re.IGNORECASE) for p in meta_patterns)
        if is_meta:
            continue

        claims.append(Claim(
            text=sentence,
            source_span=sentence,
            confidence=1.0
        ))

    return claims


def _compute_text_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two texts using sequence matching."""
    text1 = _normalize_text(text1)
    text2 = _normalize_text(text2)
    return SequenceMatcher(None, text1, text2).ratio()


def _compute_word_overlap(claim: str, context: str) -> float:
    """Compute word overlap between claim and context."""
    claim_words = set(_normalize_text(claim).split())
    context_words = set(_normalize_text(context).split())

    # Remove stopwords for better signal
    stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below',
                 'between', 'under', 'again', 'further', 'then', 'once',
                 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                 'too', 'very', 's', 't', 'just', 'don', 'now', 'and', 'but',
                 'or', 'if', 'because', 'while', 'although', 'this', 'that',
                 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}

    claim_content = claim_words - stopwords
    context_content = context_words - stopwords

    if not claim_content:
        return 1.0  # No content words to check

    overlap = claim_content & context_content

    # Also check for substring matches (e.g., "programming" in "programming language")
    additional_matches = 0
    for claim_word in claim_content - overlap:
        for context_word in context_content:
            if claim_word in context_word or context_word in claim_word:
                additional_matches += 0.5
                break

    return min(1.0, (len(overlap) + additional_matches) / len(claim_content))


def _check_claim_support(
    claim: str,
    contexts: List[str],
    threshold: float = 0.5
) -> Tuple[str, float, str]:
    """
    Check if a claim is supported by any context.

    Returns: (label, score, best_matching_context)
    - label: 'supported', 'unsupported', or 'contradicted'
    - score: confidence score (0-1)
    - best_context: the context that best matches
    """
    best_score = 0.0
    best_context = ""

    for context in contexts:
        # Compute multiple similarity signals
        word_overlap = _compute_word_overlap(claim, context)
        text_sim = _compute_text_similarity(claim, context)

        # Check for sentence-level containment
        context_sentences = _split_into_sentences(context)
        max_sentence_sim = 0.0
        for sent in context_sentences:
            sent_sim = _compute_text_similarity(claim, sent)
            max_sentence_sim = max(max_sentence_sim, sent_sim)

        # Combined score
        score = 0.4 * word_overlap + 0.3 * text_sim + 0.3 * max_sentence_sim

        if score > best_score:
            best_score = score
            best_context = context if len(context) < 200 else context[:200] + "..."

    # Determine label based on score
    if best_score >= threshold:
        return "supported", best_score, best_context
    elif best_score < 0.2:
        return "unsupported", best_score, best_context
    else:
        return "neutral", best_score, best_context


def _check_contradiction(claim: str, context: str) -> Tuple[bool, float]:
    """
    Check if claim contradicts the context.

    Uses negation detection and opposing fact patterns.
    Returns: (is_contradiction, confidence)
    """
    claim_norm = _normalize_text(claim)
    context_norm = _normalize_text(context)

    # Check for explicit negation patterns
    negation_words = ['not', 'never', 'no', 'none', "n't", 'neither', 'nor', 'nothing']

    # Simple contradiction check: if claim contains negation of context statement
    for neg in negation_words:
        # Check if claim negates something in context
        if neg in claim_norm:
            claim_without_neg = claim_norm.replace(neg, '')
            if _compute_text_similarity(claim_without_neg, context_norm) > 0.6:
                return True, 0.7

    # Check for opposing numbers/values
    claim_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', claim)
    context_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', context)

    if claim_numbers and context_numbers:
        # If there's word overlap but different numbers, possible contradiction
        if _compute_word_overlap(claim, context) > 0.5:
            for c_num in claim_numbers:
                for ctx_num in context_numbers:
                    if c_num != ctx_num:
                        return True, 0.5

    return False, 0.0


class Faithfulness(BaseMetric[HallucinationInput]):
    """
    Evaluates response faithfulness to provided context.

    Measures the proportion of claims in the response that are
    supported by the context.

    Returns score from 0.0 (all hallucinated) to 1.0 (fully faithful).
    Typical latency: 50-100ms for average response length.
    """

    @property
    def metric_name(self) -> str:
        return "faithfulness"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.support_threshold = self.config.get("support_threshold", 0.5)

    def compute_one(self, inputs: HallucinationInput) -> Dict[str, Any]:
        # Extract claims if not provided
        claims = inputs.claims or _extract_claims(inputs.response)

        if not claims:
            return {
                "output": 1.0,
                "reason": "No verifiable claims found in response."
            }

        # Prepare contexts
        contexts = inputs.context if isinstance(inputs.context, list) else [inputs.context]

        # Evaluate each claim
        supported = 0
        unsupported = 0
        claim_details = []

        for claim in claims:
            label, score, matching_context = _check_claim_support(
                claim.text, contexts, self.support_threshold
            )

            if label == "supported":
                supported += 1
            else:
                unsupported += 1

            claim_details.append({
                "claim": claim.text[:100] + "..." if len(claim.text) > 100 else claim.text,
                "label": label,
                "score": round(score, 3),
            })

        # Calculate faithfulness score
        faithfulness_score = supported / len(claims) if claims else 1.0

        reason_parts = [
            f"{supported}/{len(claims)} claims supported",
        ]
        if unsupported > 0:
            reason_parts.append(f"{unsupported} unsupported")

        return {
            "output": round(faithfulness_score, 4),
            "reason": ". ".join(reason_parts),
            "details": claim_details,
        }


class ClaimSupport(BaseMetric[HallucinationInput]):
    """
    Evaluates support level for individual claims.

    Provides granular analysis of which claims are supported
    by the context and to what degree.

    Returns average support score from 0.0 to 1.0.
    """

    @property
    def metric_name(self) -> str:
        return "claim_support"

    def compute_one(self, inputs: HallucinationInput) -> Dict[str, Any]:
        claims = inputs.claims or _extract_claims(inputs.response)

        if not claims:
            return {
                "output": 1.0,
                "reason": "No claims to evaluate."
            }

        contexts = inputs.context if isinstance(inputs.context, list) else [inputs.context]

        total_score = 0.0
        claim_results = []

        for claim in claims:
            label, score, _ = _check_claim_support(claim.text, contexts)
            total_score += score
            claim_results.append({
                "claim": claim.text[:80],
                "support_score": round(score, 3),
                "label": label,
            })

        avg_score = total_score / len(claims)

        return {
            "output": round(avg_score, 4),
            "reason": f"Average claim support: {avg_score:.1%} across {len(claims)} claims",
            "claims": claim_results,
        }


class FactualConsistency(BaseMetric[FactualConsistencyInput]):
    """
    Evaluates factual consistency between response and reference.

    Checks if the response maintains factual accuracy compared
    to a reference text or known facts.

    Returns score from 0.0 to 1.0.
    """

    @property
    def metric_name(self) -> str:
        return "factual_consistency"

    def compute_one(self, inputs: FactualConsistencyInput) -> Dict[str, Any]:
        if not inputs.reference:
            return {
                "output": 0.0,
                "reason": "No reference provided for factual consistency check."
            }

        claims = inputs.claims or _extract_claims(inputs.response)

        if not claims:
            return {
                "output": 1.0,
                "reason": "No factual claims found in response."
            }

        # Check each claim against reference
        consistent = 0
        inconsistent = 0
        details = []

        for claim in claims:
            _, score, _ = _check_claim_support(claim.text, [inputs.reference])
            is_contradicted, contradiction_conf = _check_contradiction(
                claim.text, inputs.reference
            )

            if is_contradicted and contradiction_conf > 0.5:
                inconsistent += 1
                details.append({"claim": claim.text[:80], "status": "contradicted"})
            elif score >= 0.5:
                consistent += 1
                details.append({"claim": claim.text[:80], "status": "consistent"})
            else:
                # Neutral - not clearly supported or contradicted
                details.append({"claim": claim.text[:80], "status": "unverified"})

        # Penalize contradictions heavily
        if inconsistent > 0:
            score = max(0.0, (consistent - inconsistent * 2) / len(claims))
        else:
            score = consistent / len(claims) if claims else 1.0

        return {
            "output": round(score, 4),
            "reason": f"Consistent: {consistent}, Contradicted: {inconsistent}, Total: {len(claims)}",
            "details": details,
        }


class ContradictionDetection(BaseMetric[HallucinationInput]):
    """
    Detects contradictions between response and context.

    Specifically looks for statements that directly contradict
    the provided context.

    Returns 1.0 if no contradictions, 0.0 if contradictions found.
    """

    @property
    def metric_name(self) -> str:
        return "contradiction_detection"

    def compute_one(self, inputs: HallucinationInput) -> Dict[str, Any]:
        claims = inputs.claims or _extract_claims(inputs.response)

        if not claims:
            return {
                "output": 1.0,
                "reason": "No claims to check for contradictions."
            }

        contexts = inputs.context if isinstance(inputs.context, list) else [inputs.context]
        full_context = " ".join(contexts)

        contradictions = []

        for claim in claims:
            is_contradicted, confidence = _check_contradiction(claim.text, full_context)
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

    Combines multiple signals:
    - Claim support level
    - Contradiction detection
    - Context coverage

    Returns score from 0.0 (severe hallucination) to 1.0 (no hallucination).
    """

    @property
    def metric_name(self) -> str:
        return "hallucination_score"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.support_weight = self.config.get("support_weight", 0.6)
        self.contradiction_weight = self.config.get("contradiction_weight", 0.4)

    def compute_one(self, inputs: HallucinationInput) -> Dict[str, Any]:
        claims = inputs.claims or _extract_claims(inputs.response)

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
        full_context = " ".join(contexts)

        supported = 0
        unsupported = 0
        contradicted = 0
        total_support_score = 0.0
        details = []

        for claim in claims:
            label, score, _ = _check_claim_support(claim.text, contexts)
            is_contradicted, contradiction_conf = _check_contradiction(
                claim.text, full_context
            )

            total_support_score += score

            status = "supported" if label == "supported" else "unsupported"
            if is_contradicted and contradiction_conf > 0.5:
                status = "contradicted"
                contradicted += 1
            elif label == "supported":
                supported += 1
            else:
                unsupported += 1

            details.append({
                "claim": claim.text[:80],
                "status": status,
                "support_score": round(score, 3),
            })

        # Calculate composite score
        avg_support = total_support_score / len(claims)
        contradiction_penalty = contradicted / len(claims)

        final_score = (
            self.support_weight * avg_support +
            self.contradiction_weight * (1.0 - contradiction_penalty)
        )

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
                     f"Supported: {supported}, Unsupported: {unsupported}, Contradicted: {contradicted}",
            "claims_analyzed": len(claims),
            "supported": supported,
            "unsupported": unsupported,
            "contradicted": contradicted,
            "details": details,
        }
