"""
Hallucination Sentinel — fast pre-screening before full NLI.

Provides rule-based screening to quickly flag responses
that are likely or unlikely to contain hallucinations,
avoiding expensive NLI inference for obvious cases.
"""

import re
from typing import Dict, List, Literal, Tuple


RiskLevel = Literal["low", "medium", "high"]


# Patterns that indicate high hallucination risk
_HIGH_RISK_PATTERNS = [
    r"\baccording to (?:recent|latest|new)\b",
    r"\bstudies (?:show|prove|confirm|suggest)\b",
    r"\bresearch (?:shows|proves|confirms|suggests)\b",
    r"\bstatistics (?:show|indicate|reveal)\b",
    r"\b\d+(?:\.\d+)?%\b",  # Specific percentages
    r"\bin \d{4}\b",  # Specific years
    r"\bexactly \d+\b",  # Exact numbers
    r"\bproven (?:fact|to be)\b",
    r"\bit is (?:well[- ]known|widely accepted|universally agreed)\b",
]

# Patterns that indicate the response is hedging (lower risk)
_HEDGE_PATTERNS = [
    r"\bI (?:think|believe|am not sure)\b",
    r"\b(?:might|may|could|possibly|perhaps|probably)\b",
    r"\b(?:it seems|it appears|it looks like)\b",
    r"\bI don't (?:know|have)\b",
    r"\bnot (?:certain|sure|clear)\b",
]


class HallucinationSentinel:
    """Fast rule-based screening for hallucination risk."""

    def __init__(
        self,
        extra_risk_patterns: List[str] = None,
    ):
        self.risk_patterns = _HIGH_RISK_PATTERNS + (extra_risk_patterns or [])

    def screen(
        self, response: str, context: str
    ) -> Tuple[RiskLevel, Dict]:
        """
        Screen a response for hallucination risk.

        Args:
            response: The LLM response to screen
            context: The source context

        Returns:
            Tuple of (risk_level, details dict)
        """
        response_lower = response.lower()
        context_lower = context.lower()

        details: Dict = {
            "risk_signals": [],
            "hedge_signals": [],
        }

        # Check risk patterns
        risk_count = 0
        for pattern in self.risk_patterns:
            matches = re.findall(pattern, response_lower, re.IGNORECASE)
            if matches:
                risk_count += len(matches)
                details["risk_signals"].append(pattern)

        # Check hedging patterns
        hedge_count = 0
        for pattern in _HEDGE_PATTERNS:
            if re.search(pattern, response_lower, re.IGNORECASE):
                hedge_count += 1
                details["hedge_signals"].append(pattern)

        # Check if claims reference things not in context
        # Simple: response words not found in context
        response_words = set(re.findall(r"\b\w{4,}\b", response_lower))
        context_words = set(re.findall(r"\b\w{4,}\b", context_lower))
        novel_ratio = len(response_words - context_words) / max(len(response_words), 1)
        details["novel_word_ratio"] = round(novel_ratio, 3)

        # Determine risk level
        if risk_count >= 3 or (risk_count >= 1 and novel_ratio > 0.6):
            risk_level: RiskLevel = "high"
        elif risk_count >= 1 or novel_ratio > 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Hedging reduces risk
        if hedge_count >= 2 and risk_level == "high":
            risk_level = "medium"

        details["risk_count"] = risk_count
        details["hedge_count"] = hedge_count

        return risk_level, details
