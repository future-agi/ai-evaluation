"""
Source Attribution Metric.

Evaluates citation quality in RAG responses - whether the response
properly cites its sources and whether citations are accurate.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from ...base_metric import BaseMetric
from ..types import SourceAttributionInput
from ..utils import (
    extract_claims,
    check_claim_supported,
    compute_text_similarity,
    split_into_sentences,
)


class SourceAttribution(BaseMetric[SourceAttributionInput]):
    """
    Evaluates citation quality in RAG responses.

    Checks:
    1. Are claims properly cited?
    2. Are citations accurate (do they support the claim)?
    3. Is citation coverage complete?

    Supports multiple citation formats:
    - Bracketed: [1], [2], etc.
    - Inline: (Source A), (Document 1)
    - Footnote: ¹, ², etc.

    Score: 0.0 (poor attribution) to 1.0 (excellent attribution)

    Example:
        >>> attribution = SourceAttribution()
        >>> result = attribution.evaluate([{
        ...     "response": "Paris is the capital of France [1].",
        ...     "contexts": ["Paris is the capital and largest city of France."],
        ...     "citation_format": "bracketed"
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "source_attribution"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.coverage_weight = self.config.get("coverage_weight", 0.5)
        self.accuracy_weight = self.config.get("accuracy_weight", 0.5)
        self.verification_threshold = self.config.get("verification_threshold", 0.5)

    def compute_one(self, inputs: SourceAttributionInput) -> Dict[str, Any]:
        response = inputs.response
        contexts = inputs.contexts
        citation_format = inputs.citation_format
        require_citations = inputs.require_citations

        if not response or not response.strip():
            return {"output": 0.0, "reason": "Empty response"}

        if not contexts:
            return {"output": 0.0, "reason": "No contexts provided"}

        # 1. Extract citations from response
        citations = self._extract_citations(response, citation_format)

        # 2. Extract claims from response
        claims = extract_claims(response)

        # Handle case with no citations
        if not citations:
            if require_citations and claims:
                return {
                    "output": 0.0,
                    "reason": "No citations found but citations required",
                    "total_claims": len(claims),
                    "citations_found": 0,
                }
            else:
                # No citations required or no claims to cite
                return {
                    "output": 1.0,
                    "reason": "No citations needed",
                    "total_claims": len(claims),
                }

        # 3. Check citation coverage
        coverage_result = self._check_coverage(claims, citations, response)

        # 4. Verify citation accuracy
        accuracy_result = self._verify_accuracy(citations, contexts)

        # Combined score
        final_score = (
            self.coverage_weight * coverage_result["coverage"] +
            self.accuracy_weight * accuracy_result["accuracy"]
        )

        return {
            "output": round(final_score, 4),
            "reason": f"Coverage: {coverage_result['coverage']:.0%}, Accuracy: {accuracy_result['accuracy']:.0%}",
            "citation_coverage": round(coverage_result["coverage"], 4),
            "citation_accuracy": round(accuracy_result["accuracy"], 4),
            "total_claims": len(claims),
            "cited_claims": coverage_result["cited_claims"],
            "total_citations": len(citations),
            "accurate_citations": accuracy_result["accurate_count"],
            "uncited_claims": coverage_result["uncited_claims"][:5],
            "inaccurate_citations": accuracy_result["inaccurate"][:5],
        }

    def _extract_citations(
        self, text: str, format: str
    ) -> List[Dict[str, Any]]:
        """Extract citations based on format."""
        citations = []

        if format == "bracketed":
            # Match [1], [2], [1,2], [1-3], etc.
            pattern = r"\[(\d+(?:[,\-]\d+)*)\]"
            for match in re.finditer(pattern, text):
                citation_refs = match.group(1)
                # Parse reference numbers
                source_indices = self._parse_citation_refs(citation_refs)

                # Get surrounding context as the claim
                start = max(0, match.start() - 300)
                claim_text = text[start:match.start()]
                # Get the last sentence before citation
                sentences = split_into_sentences(claim_text)
                claim_text = sentences[-1] if sentences else claim_text[-200:]

                citations.append({
                    "source_indices": source_indices,
                    "claim_text": claim_text.strip(),
                    "position": match.start(),
                    "raw": match.group(0),
                })

        elif format == "inline":
            # Match (Source 1), (Document A), etc.
            pattern = r"\((?:Source|Document|Doc|Ref)\.?\s*(\d+|[A-Z])\)"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                ref = match.group(1)
                source_idx = int(ref) - 1 if ref.isdigit() else ord(ref.upper()) - ord('A')

                claim_text = text[max(0, match.start()-300):match.start()]
                sentences = split_into_sentences(claim_text)
                claim_text = sentences[-1] if sentences else claim_text[-200:]

                citations.append({
                    "source_indices": [source_idx],
                    "claim_text": claim_text.strip(),
                    "position": match.start(),
                    "raw": match.group(0),
                })

        elif format == "footnote":
            # Match superscript numbers: ¹, ², ³ or ^1, ^2
            pattern = r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+|\^(\d+)"
            superscript_map = {"¹": 1, "²": 2, "³": 3, "⁴": 4, "⁵": 5,
                            "⁶": 6, "⁷": 7, "⁸": 8, "⁹": 9, "⁰": 0}

            for match in re.finditer(pattern, text):
                if match.group(1):  # ^1 format
                    source_idx = int(match.group(1)) - 1
                else:  # Superscript format
                    # Convert superscript to number
                    num_str = "".join(
                        str(superscript_map.get(c, ""))
                        for c in match.group(0)
                    )
                    source_idx = int(num_str) - 1 if num_str else 0

                claim_text = text[max(0, match.start()-300):match.start()]
                sentences = split_into_sentences(claim_text)
                claim_text = sentences[-1] if sentences else claim_text[-200:]

                citations.append({
                    "source_indices": [source_idx],
                    "claim_text": claim_text.strip(),
                    "position": match.start(),
                    "raw": match.group(0),
                })

        return citations

    def _parse_citation_refs(self, refs: str) -> List[int]:
        """Parse citation reference string like '1,2' or '1-3'."""
        indices = []

        parts = refs.split(",")
        for part in parts:
            if "-" in part:
                # Range: 1-3 -> [0, 1, 2]
                start, end = part.split("-")
                indices.extend(range(int(start) - 1, int(end)))
            else:
                indices.append(int(part) - 1)

        return indices

    def _check_coverage(
        self, claims: List[str], citations: List[Dict], response: str
    ) -> Dict[str, Any]:
        """Check what proportion of claims have citations."""
        if not claims:
            return {"coverage": 1.0, "cited_claims": 0, "uncited_claims": []}

        cited_claims = 0
        uncited_claims = []

        for claim in claims:
            # Check if this claim has an associated citation
            has_citation = self._claim_has_citation(claim, citations, response)

            if has_citation:
                cited_claims += 1
            else:
                uncited_claims.append(claim[:80] + "..." if len(claim) > 80 else claim)

        coverage = cited_claims / len(claims)

        return {
            "coverage": coverage,
            "cited_claims": cited_claims,
            "uncited_claims": uncited_claims,
        }

    def _claim_has_citation(
        self, claim: str, citations: List[Dict], response: str
    ) -> bool:
        """Check if a claim has an associated citation."""
        claim_lower = claim.lower()

        # Find claim position in response
        claim_pos = response.lower().find(claim_lower[:50])
        if claim_pos == -1:
            # Try fuzzy match
            for i, citation in enumerate(citations):
                cited_claim = citation.get("claim_text", "").lower()
                if compute_text_similarity(claim_lower, cited_claim) > 0.6:
                    return True
            return False

        # Check if there's a citation near this claim
        claim_end = claim_pos + len(claim)

        for citation in citations:
            cite_pos = citation["position"]
            # Citation should be within 50 chars after the claim
            if claim_pos <= cite_pos <= claim_end + 50:
                return True

        return False

    def _verify_accuracy(
        self, citations: List[Dict], contexts: List[str]
    ) -> Dict[str, Any]:
        """Verify that citations accurately point to supporting sources."""
        if not citations:
            return {"accuracy": 1.0, "accurate_count": 0, "inaccurate": []}

        accurate = 0
        inaccurate = []

        for citation in citations:
            source_indices = citation.get("source_indices", [])
            claim_text = citation.get("claim_text", "")

            if not claim_text:
                continue

            # Check if any cited source supports the claim
            is_accurate = False
            for idx in source_indices:
                if 0 <= idx < len(contexts):
                    source = contexts[idx]
                    is_supported, score, _ = check_claim_supported(
                        claim_text, [source], self.verification_threshold
                    )
                    if is_supported:
                        is_accurate = True
                        break

            if is_accurate:
                accurate += 1
            else:
                inaccurate.append({
                    "claim": claim_text[:80],
                    "cited_sources": source_indices,
                })

        accuracy = accurate / len(citations) if citations else 1.0

        return {
            "accuracy": accuracy,
            "accurate_count": accurate,
            "inaccurate": inaccurate,
        }


class CitationPresence(BaseMetric[SourceAttributionInput]):
    """
    Simple metric to check if citations are present.

    Useful as a quick check before detailed attribution analysis.

    Score: 0.0 (no citations) to 1.0 (citations present)
    """

    @property
    def metric_name(self) -> str:
        return "citation_presence"

    def compute_one(self, inputs: SourceAttributionInput) -> Dict[str, Any]:
        response = inputs.response

        # Check for any citation patterns
        patterns = [
            r"\[\d+\]",  # [1]
            r"\(\d+\)",  # (1)
            r"\((?:Source|Document|Ref)\s*\d+\)",  # (Source 1)
            r"[¹²³⁴⁵⁶⁷⁸⁹]",  # Superscripts
            r"\^\d+",  # ^1
        ]

        citations_found = []
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            citations_found.extend(matches)

        has_citations = len(citations_found) > 0

        return {
            "output": 1.0 if has_citations else 0.0,
            "reason": f"Found {len(citations_found)} citations" if has_citations else "No citations found",
            "citations_found": citations_found[:10],
            "citation_count": len(citations_found),
        }
