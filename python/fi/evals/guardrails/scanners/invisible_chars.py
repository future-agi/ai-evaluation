"""
Invisible Character Scanner for Guardrails.

Detects Unicode manipulation, zero-width characters, homoglyphs,
and other invisible character attacks.
"""

import re
import time
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

from fi.evals.guardrails.scanners.base import (
    BaseScanner,
    ScanResult,
    ScanMatch,
    ScannerAction,
    register_scanner,
)


# Zero-width and invisible characters
INVISIBLE_CHARS: Dict[int, str] = {
    0x200B: "zero_width_space",
    0x200C: "zero_width_non_joiner",
    0x200D: "zero_width_joiner",
    0x200E: "left_to_right_mark",
    0x200F: "right_to_left_mark",
    0x2060: "word_joiner",
    0x2061: "function_application",
    0x2062: "invisible_times",
    0x2063: "invisible_separator",
    0x2064: "invisible_plus",
    0xFEFF: "byte_order_mark",
    0x00AD: "soft_hyphen",
    0x034F: "combining_grapheme_joiner",
    0x061C: "arabic_letter_mark",
    0x115F: "hangul_choseong_filler",
    0x1160: "hangul_jungseong_filler",
    0x17B4: "khmer_vowel_inherent_aq",
    0x17B5: "khmer_vowel_inherent_aa",
    0x180E: "mongolian_vowel_separator",
    0x3164: "hangul_filler",
    0xFFA0: "halfwidth_hangul_filler",
}

# Bidirectional override characters (can be used for attacks)
BIDI_CHARS: Dict[int, str] = {
    0x202A: "left_to_right_embedding",
    0x202B: "right_to_left_embedding",
    0x202C: "pop_directional_formatting",
    0x202D: "left_to_right_override",
    0x202E: "right_to_left_override",  # This is particularly dangerous
    0x2066: "left_to_right_isolate",
    0x2067: "right_to_left_isolate",
    0x2068: "first_strong_isolate",
    0x2069: "pop_directional_isolate",
}

# Tag characters (can be used to hide information)
TAG_CHARS_RANGE = (0xE0000, 0xE007F)

# Common homoglyph mappings (confusable characters)
HOMOGLYPHS: Dict[str, List[str]] = {
    'a': ['а', 'ɑ', 'α', 'ạ', 'ą'],  # Cyrillic а, Latin alpha, etc.
    'c': ['с', 'ϲ', 'ċ'],  # Cyrillic с
    'd': ['ԁ', 'ɗ'],
    'e': ['е', 'ė', 'ẹ'],  # Cyrillic е
    'g': ['ɡ', 'ġ'],
    'h': ['һ', 'ḥ'],  # Cyrillic һ
    'i': ['і', 'ı', 'ị'],  # Cyrillic і
    'j': ['ј'],  # Cyrillic ј
    'k': ['κ', 'ķ'],
    'l': ['ӏ', 'ḷ'],  # Cyrillic palochka
    'm': ['м', 'ṃ'],  # Cyrillic м
    'n': ['ո', 'ņ'],  # Armenian
    'o': ['о', 'ο', 'ọ', '০'],  # Cyrillic о, Greek omicron, Bengali zero
    'p': ['р', 'ρ'],  # Cyrillic р, Greek rho
    's': ['ѕ', 'ṣ'],  # Cyrillic ѕ
    't': ['τ', 'ṭ'],
    'u': ['υ', 'ụ'],
    'v': ['ν', 'ѵ'],  # Greek nu, Cyrillic izhitsa
    'w': ['ԝ', 'ẉ'],
    'x': ['х', 'χ'],  # Cyrillic х, Greek chi
    'y': ['у', 'γ'],  # Cyrillic у, Greek gamma
    'z': ['ᴢ'],
    'A': ['А', 'Α', 'Ạ'],  # Cyrillic А, Greek Alpha
    'B': ['В', 'Β', 'Ḅ'],  # Cyrillic В, Greek Beta
    'C': ['С', 'Ϲ'],  # Cyrillic С
    'E': ['Е', 'Ε'],  # Cyrillic Е, Greek Epsilon
    'H': ['Н', 'Η'],  # Cyrillic Н, Greek Eta
    'I': ['І', 'Ι'],  # Cyrillic І, Greek Iota
    'K': ['К', 'Κ'],  # Cyrillic К, Greek Kappa
    'M': ['М', 'Μ'],  # Cyrillic М, Greek Mu
    'N': ['Ν'],  # Greek Nu
    'O': ['О', 'Ο', '০'],  # Cyrillic О, Greek Omicron
    'P': ['Р', 'Ρ'],  # Cyrillic Р, Greek Rho
    'S': ['Ѕ'],  # Cyrillic Ѕ
    'T': ['Т', 'Τ'],  # Cyrillic Т, Greek Tau
    'X': ['Х', 'Χ'],  # Cyrillic Х, Greek Chi
    'Y': ['Υ', 'У'],  # Greek Upsilon, Cyrillic У
    'Z': ['Ζ'],  # Greek Zeta
    '0': ['О', 'о', '০'],  # Letters that look like zero
    '1': ['І', 'ӏ'],  # Letters that look like one (only non-ASCII)
}


def _create_homoglyph_set() -> Set[str]:
    """Create a set of all homoglyph characters."""
    chars = set()
    for variants in HOMOGLYPHS.values():
        chars.update(variants)
    return chars


HOMOGLYPH_CHARS = _create_homoglyph_set()


@register_scanner("invisible_chars")
class InvisibleCharScanner(BaseScanner):
    """
    Scanner for detecting invisible character attacks.

    Detects:
    - Zero-width characters (U+200B, U+FEFF, etc.)
    - Bidirectional override characters (U+202E attack)
    - Homoglyphs (Cyrillic а vs Latin a)
    - Tag characters
    - Mixed script text

    Usage:
        scanner = InvisibleCharScanner()
        result = scanner.scan("Hello\\u200Bworld")  # Zero-width space
        if not result.passed:
            print(f"Hidden chars detected: {result.matched_patterns}")
    """

    name = "invisible_chars"
    category = "unicode_attack"
    description = "Detects invisible characters and Unicode manipulation"
    default_action = ScannerAction.FLAG

    def __init__(
        self,
        action: Optional[ScannerAction] = None,
        enabled: bool = True,
        threshold: float = 0.5,
        check_invisible: bool = True,
        check_bidi: bool = True,
        check_homoglyphs: bool = True,
        check_tags: bool = True,
        check_mixed_scripts: bool = True,
        allowed_scripts: Optional[Set[str]] = None,
    ):
        """
        Initialize invisible character scanner.

        Args:
            action: Action on detection
            enabled: Whether scanner is enabled
            threshold: Minimum confidence to trigger
            check_invisible: Check for invisible characters
            check_bidi: Check for bidirectional overrides
            check_homoglyphs: Check for homoglyph attacks
            check_tags: Check for tag characters
            check_mixed_scripts: Check for mixed Unicode scripts
            allowed_scripts: Set of allowed script names (e.g., {"Latin", "Common"})
        """
        super().__init__(action, enabled)
        self.threshold = threshold
        self.check_invisible = check_invisible
        self.check_bidi = check_bidi
        self.check_homoglyphs = check_homoglyphs
        self.check_tags = check_tags
        self.check_mixed_scripts = check_mixed_scripts
        self.allowed_scripts = allowed_scripts or {"Latin", "Common"}

    def _get_script(self, char: str) -> str:
        """Get the Unicode script for a character."""
        try:
            name = unicodedata.name(char, "")
            # Extract script from character name (simplified)
            if "LATIN" in name:
                return "Latin"
            elif "CYRILLIC" in name:
                return "Cyrillic"
            elif "GREEK" in name:
                return "Greek"
            elif "ARABIC" in name:
                return "Arabic"
            elif "HEBREW" in name:
                return "Hebrew"
            elif "CJK" in name or "CHINESE" in name:
                return "CJK"
            elif "HANGUL" in name or "KOREAN" in name:
                return "Hangul"
            elif "HIRAGANA" in name or "KATAKANA" in name:
                return "Japanese"
            elif name:
                return "Other"
            return "Common"
        except Exception:
            return "Unknown"

    def scan(self, content: str, context: Optional[str] = None) -> ScanResult:
        """
        Scan content for invisible character attacks.

        Args:
            content: Content to scan
            context: Optional context

        Returns:
            ScanResult with detection details
        """
        start = time.perf_counter()
        matches = []
        max_confidence = 0.0
        issues = set()

        # Check invisible characters
        if self.check_invisible:
            for i, char in enumerate(content):
                code = ord(char)
                if code in INVISIBLE_CHARS:
                    matches.append(ScanMatch(
                        pattern_name=INVISIBLE_CHARS[code],
                        matched_text=f"U+{code:04X}",
                        start=i,
                        end=i + 1,
                        confidence=0.9,
                    ))
                    max_confidence = max(max_confidence, 0.9)
                    issues.add("Invisible character")

        # Check bidirectional overrides
        if self.check_bidi:
            for i, char in enumerate(content):
                code = ord(char)
                if code in BIDI_CHARS:
                    # RTL override is particularly dangerous
                    confidence = 0.95 if code == 0x202E else 0.7
                    matches.append(ScanMatch(
                        pattern_name=BIDI_CHARS[code],
                        matched_text=f"U+{code:04X}",
                        start=i,
                        end=i + 1,
                        confidence=confidence,
                    ))
                    max_confidence = max(max_confidence, confidence)
                    issues.add("Bidirectional override")

        # Check tag characters
        if self.check_tags:
            for i, char in enumerate(content):
                code = ord(char)
                if TAG_CHARS_RANGE[0] <= code <= TAG_CHARS_RANGE[1]:
                    matches.append(ScanMatch(
                        pattern_name="tag_character",
                        matched_text=f"U+{code:04X}",
                        start=i,
                        end=i + 1,
                        confidence=0.85,
                    ))
                    max_confidence = max(max_confidence, 0.85)
                    issues.add("Tag character")

        # Check homoglyphs
        if self.check_homoglyphs:
            for i, char in enumerate(content):
                if char in HOMOGLYPH_CHARS:
                    matches.append(ScanMatch(
                        pattern_name="homoglyph",
                        matched_text=f"{char} (U+{ord(char):04X})",
                        start=i,
                        end=i + 1,
                        confidence=0.75,
                        metadata={"character": char, "code": f"U+{ord(char):04X}"},
                    ))
                    max_confidence = max(max_confidence, 0.75)
                    issues.add("Homoglyph character")

        # Check mixed scripts
        if self.check_mixed_scripts:
            scripts_found = set()
            for char in content:
                if char.isalpha():  # Only check letters
                    script = self._get_script(char)
                    if script not in ("Common", "Unknown"):
                        scripts_found.add(script)

            # Check for suspicious script mixing
            non_allowed = scripts_found - self.allowed_scripts
            if non_allowed and len(scripts_found) > 1:
                matches.append(ScanMatch(
                    pattern_name="mixed_scripts",
                    matched_text=f"Scripts: {', '.join(scripts_found)}",
                    start=0,
                    end=len(content),
                    confidence=0.7,
                    metadata={"scripts": list(scripts_found)},
                ))
                max_confidence = max(max_confidence, 0.7)
                issues.add(f"Mixed scripts ({', '.join(non_allowed)})")

        latency = (time.perf_counter() - start) * 1000

        # Filter by threshold
        significant_matches = [m for m in matches if m.confidence >= self.threshold]

        if significant_matches:
            return self._create_result(
                passed=False,
                matches=significant_matches,
                score=max_confidence,
                reason=f"Unicode manipulation detected: {', '.join(issues)}",
                latency_ms=latency,
                metadata={"issues": list(issues)},
            )

        return self._create_result(
            passed=True,
            matches=[],
            score=0.0,
            reason="No invisible character attacks detected",
            latency_ms=latency,
        )

    def sanitize(self, content: str) -> str:
        """
        Remove invisible and dangerous characters from content.

        Args:
            content: Content to sanitize

        Returns:
            Sanitized content
        """
        result = []
        for char in content:
            code = ord(char)
            # Skip invisible characters
            if code in INVISIBLE_CHARS:
                continue
            # Skip bidi overrides
            if code in BIDI_CHARS:
                continue
            # Skip tag characters
            if TAG_CHARS_RANGE[0] <= code <= TAG_CHARS_RANGE[1]:
                continue
            result.append(char)
        return ''.join(result)
