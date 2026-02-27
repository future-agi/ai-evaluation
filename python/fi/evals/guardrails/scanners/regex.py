"""
Regex Scanner for Guardrails.

Provides configurable regex-based pattern matching for custom rules.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fi.evals.guardrails.scanners.base import (
    BaseScanner,
    ScanResult,
    ScanMatch,
    ScannerAction,
    register_scanner,
)


@dataclass
class RegexPattern:
    """Definition of a regex pattern for scanning."""
    name: str
    pattern: str
    confidence: float = 0.8
    action: ScannerAction = ScannerAction.BLOCK
    description: str = ""
    flags: int = re.IGNORECASE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compile the pattern."""
        self._compiled = re.compile(self.pattern, self.flags)

    @property
    def compiled(self) -> re.Pattern:
        """Get compiled pattern."""
        if not hasattr(self, '_compiled'):
            self._compiled = re.compile(self.pattern, self.flags)
        return self._compiled


# Common pre-defined patterns
COMMON_PATTERNS: Dict[str, RegexPattern] = {
    # Credit card numbers
    "credit_card": RegexPattern(
        name="credit_card",
        pattern=r'\b(?:\d{4}[- ]?){3}\d{4}\b',
        confidence=0.85,
        description="Credit card number pattern",
    ),

    # Social Security Numbers
    "ssn": RegexPattern(
        name="ssn",
        pattern=r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b',
        confidence=0.8,
        description="Social Security Number pattern",
    ),

    # Email addresses
    "email": RegexPattern(
        name="email",
        pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        confidence=0.9,
        action=ScannerAction.FLAG,
        description="Email address pattern",
    ),

    # Phone numbers (US format)
    "phone_us": RegexPattern(
        name="phone_us",
        pattern=r'\b(?:\+?1[-. ]?)?\(?[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b',
        confidence=0.75,
        action=ScannerAction.FLAG,
        description="US phone number pattern",
    ),

    # IP addresses
    "ip_address": RegexPattern(
        name="ip_address",
        pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        confidence=0.7,
        action=ScannerAction.FLAG,
        description="IPv4 address pattern",
    ),

    # Profanity (basic, expand as needed)
    "profanity_basic": RegexPattern(
        name="profanity",
        pattern=r'\b(?:fuck|shit|ass|damn|bitch|crap)\b',
        confidence=0.9,
        action=ScannerAction.FLAG,
        description="Basic profanity filter",
    ),

    # URLs
    "url": RegexPattern(
        name="url",
        pattern=r'https?://[^\s<>"]+',
        confidence=0.9,
        action=ScannerAction.FLAG,
        description="URL pattern",
    ),

    # Date patterns (various formats)
    "date": RegexPattern(
        name="date",
        pattern=r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
        confidence=0.7,
        action=ScannerAction.FLAG,
        description="Date pattern",
    ),

    # Bank account numbers (generic)
    "bank_account": RegexPattern(
        name="bank_account",
        pattern=r'\b\d{8,17}\b',
        confidence=0.5,  # Low confidence due to false positives
        action=ScannerAction.FLAG,
        description="Potential bank account number",
    ),

    # Medical record numbers (generic)
    "mrn": RegexPattern(
        name="mrn",
        pattern=r'(?i)\b(?:mrn|medical\s*record)\s*[#:]?\s*\d{6,10}\b',
        confidence=0.85,
        description="Medical record number pattern",
    ),

    # Passport numbers (generic)
    "passport": RegexPattern(
        name="passport",
        pattern=r'(?i)\b(?:passport)\s*[#:]?\s*[A-Z0-9]{6,9}\b',
        confidence=0.8,
        description="Passport number pattern",
    ),

    # Driver's license (generic)
    "drivers_license": RegexPattern(
        name="drivers_license",
        pattern=r'(?i)\b(?:dl|driver\'?s?\s*license)\s*[#:]?\s*[A-Z0-9]{5,15}\b',
        confidence=0.75,
        description="Driver's license pattern",
    ),
}


@register_scanner("regex")
class RegexScanner(BaseScanner):
    """
    Scanner for custom regex-based pattern matching.

    Allows defining custom patterns for domain-specific content detection.

    Usage:
        # Use pre-defined patterns
        scanner = RegexScanner(patterns=["credit_card", "ssn", "email"])

        # Use custom patterns
        custom_patterns = [
            RegexPattern(
                name="internal_id",
                pattern=r"INT-\d{6}",
                confidence=0.9,
                description="Internal ID format",
            ),
        ]
        scanner = RegexScanner(custom_patterns=custom_patterns)

        result = scanner.scan("My card is 4111-1111-1111-1111")
    """

    name = "regex"
    category = "custom_pattern"
    description = "Custom regex pattern matching"
    default_action = ScannerAction.FLAG

    def __init__(
        self,
        action: Optional[ScannerAction] = None,
        enabled: bool = True,
        threshold: float = 0.5,
        patterns: Optional[List[str]] = None,
        custom_patterns: Optional[List[RegexPattern]] = None,
        match_mode: str = "any",  # "any" or "all"
    ):
        """
        Initialize regex scanner.

        Args:
            action: Default action (can be overridden per pattern)
            enabled: Whether scanner is enabled
            threshold: Confidence threshold
            patterns: List of pre-defined pattern names to use
            custom_patterns: List of custom RegexPattern objects
            match_mode: "any" (flag if any pattern matches) or "all" (flag only if all match)
        """
        super().__init__(action, enabled)
        self.threshold = threshold
        self.match_mode = match_mode

        # Build pattern list
        self.patterns: List[RegexPattern] = []

        # Add pre-defined patterns
        if patterns:
            for name in patterns:
                if name in COMMON_PATTERNS:
                    self.patterns.append(COMMON_PATTERNS[name])

        # Add custom patterns
        if custom_patterns:
            self.patterns.extend(custom_patterns)

    def add_pattern(self, pattern: RegexPattern) -> "RegexScanner":
        """Add a pattern to the scanner."""
        self.patterns.append(pattern)
        return self

    def remove_pattern(self, name: str) -> "RegexScanner":
        """Remove a pattern by name."""
        self.patterns = [p for p in self.patterns if p.name != name]
        return self

    def scan(self, content: str, context: Optional[str] = None) -> ScanResult:
        """
        Scan content using configured regex patterns.

        Args:
            content: Content to scan
            context: Optional context (also scanned if provided)

        Returns:
            ScanResult with pattern match details
        """
        start = time.perf_counter()
        matches = []
        max_confidence = 0.0
        matched_patterns = set()

        # Combine content and context for scanning
        text_to_scan = content
        if context:
            text_to_scan = f"{context}\n{content}"

        for pattern in self.patterns:
            for match in pattern.compiled.finditer(text_to_scan):
                matches.append(ScanMatch(
                    pattern_name=pattern.name,
                    matched_text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=pattern.confidence,
                    metadata={
                        "description": pattern.description,
                        "action": pattern.action.value,
                    },
                ))
                max_confidence = max(max_confidence, pattern.confidence)
                matched_patterns.add(pattern.name)

        latency = (time.perf_counter() - start) * 1000

        # Filter by threshold
        significant_matches = [m for m in matches if m.confidence >= self.threshold]

        if not significant_matches:
            return self._create_result(
                passed=True,
                matches=[],
                score=0.0,
                reason="No patterns matched",
                latency_ms=latency,
            )

        # Determine action based on matched patterns
        # Use the most restrictive action among matches
        actions = [
            COMMON_PATTERNS.get(m.pattern_name, RegexPattern(name="", pattern="")).action
            if m.pattern_name in COMMON_PATTERNS
            else self.action
            for m in significant_matches
        ]

        final_action = self.action
        if ScannerAction.BLOCK in actions:
            final_action = ScannerAction.BLOCK
        elif ScannerAction.REDACT in actions:
            final_action = ScannerAction.REDACT
        elif ScannerAction.FLAG in actions:
            final_action = ScannerAction.FLAG

        return self._create_result(
            passed=False,
            matches=significant_matches,
            score=max_confidence,
            reason=f"Patterns matched: {', '.join(matched_patterns)}",
            latency_ms=latency,
            metadata={
                "matched_patterns": list(matched_patterns),
                "total_matches": len(matches),
            },
        )

    @classmethod
    def from_patterns(cls, pattern_names: List[str], **kwargs) -> "RegexScanner":
        """
        Create a scanner from a list of pre-defined pattern names.

        Args:
            pattern_names: List of pattern names from COMMON_PATTERNS
            **kwargs: Additional arguments for the scanner

        Returns:
            Configured RegexScanner
        """
        return cls(patterns=pattern_names, **kwargs)

    @classmethod
    def pii_scanner(cls, **kwargs) -> "RegexScanner":
        """Create a scanner configured for common PII patterns."""
        return cls(
            patterns=["credit_card", "ssn", "email", "phone_us", "passport", "drivers_license"],
            **kwargs
        )
