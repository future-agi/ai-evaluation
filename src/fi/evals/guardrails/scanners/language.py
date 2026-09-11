"""
Language Detection Scanner for Guardrails.

Detects and filters content by language.
"""

import re
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from fi.evals.guardrails.scanners.base import (
    BaseScanner,
    ScanResult,
    ScanMatch,
    ScannerAction,
    register_scanner,
)


# Character ranges for script detection (simplified)
SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "Latin": [(0x0041, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
    "Cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "Greek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF)],
    "Hebrew": [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)],
    "CJK": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF)],
    "Hangul": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "Hiragana": [(0x3040, 0x309F)],
    "Katakana": [(0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    "Thai": [(0x0E00, 0x0E7F)],
    "Devanagari": [(0x0900, 0x097F)],
}

# Common words for language detection (trigram-based approximation)
LANGUAGE_MARKERS: Dict[str, Set[str]] = {
    "en": {"the", "and", "for", "that", "with", "this", "you", "are", "not", "have"},
    "es": {"que", "del", "los", "las", "con", "una", "por", "para", "como", "más"},
    "fr": {"les", "des", "que", "pour", "est", "dans", "une", "pas", "sur", "avec"},
    "de": {"der", "die", "und", "den", "das", "ist", "nicht", "mit", "auf", "für"},
    "it": {"che", "della", "per", "sono", "con", "una", "della", "questo", "anche"},
    "pt": {"que", "para", "com", "uma", "não", "por", "mais", "como", "dos", "das"},
    "nl": {"het", "van", "een", "dat", "met", "zijn", "voor", "niet", "maar", "ook"},
    "pl": {"nie", "się", "jest", "jak", "tak", "ale", "czy", "już", "tylko", "być"},
    "ru": {"что", "как", "это", "был", "для", "все", "его", "она", "они", "при"},
    "zh": {"的", "是", "在", "了", "有", "和", "人", "这", "中", "大"},
    "ja": {"の", "に", "は", "を", "た", "が", "で", "て", "と", "し"},
    "ko": {"이", "는", "을", "의", "가", "에", "하", "고", "다", "로"},
    "ar": {"في", "من", "على", "إلى", "أن", "هذا", "التي", "مع", "كان", "عن"},
}


def _detect_script(text: str) -> Dict[str, int]:
    """Detect scripts present in text based on character ranges."""
    script_counts: Dict[str, int] = Counter()

    for char in text:
        code = ord(char)
        for script, ranges in SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= code <= end:
                    script_counts[script] += 1
                    break

    return dict(script_counts)


def _detect_language_simple(text: str) -> Optional[str]:
    """Simple language detection based on common words."""
    # Normalize and tokenize
    words = set(re.findall(r'\b\w+\b', text.lower()))

    if not words:
        return None

    # Score each language by marker word overlap
    scores: Dict[str, int] = {}
    for lang, markers in LANGUAGE_MARKERS.items():
        overlap = len(words & markers)
        if overlap > 0:
            scores[lang] = overlap

    if scores:
        return max(scores.keys(), key=lambda k: scores[k])

    return None


@register_scanner("language")
class LanguageScanner(BaseScanner):
    """
    Scanner for detecting and filtering by language.

    Detects:
    - Primary language of content
    - Script types used (Latin, Cyrillic, CJK, etc.)
    - Mixed language content

    Usage:
        scanner = LanguageScanner(allowed_languages={"en", "es"})
        result = scanner.scan("Hola, ¿cómo estás?")
        print(f"Detected language: {result.metadata.get('detected_language')}")
    """

    name = "language"
    category = "language"
    description = "Detects and filters by language"
    default_action = ScannerAction.FLAG

    def __init__(
        self,
        action: Optional[ScannerAction] = None,
        enabled: bool = True,
        threshold: float = 0.5,
        allowed_languages: Optional[Set[str]] = None,
        blocked_languages: Optional[Set[str]] = None,
        allowed_scripts: Optional[Set[str]] = None,
        use_langdetect: bool = True,
        min_text_length: int = 20,
    ):
        """
        Initialize language scanner.

        Args:
            action: Action on detection
            enabled: Whether scanner is enabled
            threshold: Confidence threshold
            allowed_languages: Set of allowed language codes (e.g., {"en", "es"})
            blocked_languages: Set of blocked language codes
            allowed_scripts: Set of allowed scripts (e.g., {"Latin"})
            use_langdetect: Try to use langdetect library if available
            min_text_length: Minimum text length for detection
        """
        super().__init__(action, enabled)
        self.threshold = threshold
        self.allowed_languages = allowed_languages
        self.blocked_languages = blocked_languages or set()
        self.allowed_scripts = allowed_scripts
        self.use_langdetect = use_langdetect
        self.min_text_length = min_text_length

        # Try to import langdetect
        self._langdetect = None
        if use_langdetect:
            try:
                import langdetect
                self._langdetect = langdetect
            except ImportError:
                pass

    def _detect_language(self, text: str) -> Tuple[Optional[str], float]:
        """
        Detect language of text.

        Returns:
            Tuple of (language_code, confidence)
        """
        # Use langdetect if available
        if self._langdetect and len(text) >= self.min_text_length:
            try:
                results = self._langdetect.detect_langs(text)
                if results:
                    return (results[0].lang, results[0].prob)
            except Exception:
                pass

        # Fallback to simple detection
        lang = _detect_language_simple(text)
        if lang:
            return (lang, 0.7)

        return (None, 0.0)

    def scan(self, content: str, context: Optional[str] = None) -> ScanResult:
        """
        Scan content for language compliance.

        Args:
            content: Content to scan
            context: Optional context

        Returns:
            ScanResult with language detection details
        """
        start = time.perf_counter()
        matches = []
        issues = []

        # Skip very short content
        if len(content.strip()) < self.min_text_length:
            latency = (time.perf_counter() - start) * 1000
            return self._create_result(
                passed=True,
                matches=[],
                score=0.0,
                reason="Content too short for language detection",
                latency_ms=latency,
                metadata={"skipped": True},
            )

        # Detect language
        detected_lang, confidence = self._detect_language(content)

        # Detect scripts
        scripts = _detect_script(content)
        primary_script = max(scripts.keys(), key=lambda k: scripts[k]) if scripts else None

        # Check language restrictions
        lang_violation = False
        if detected_lang:
            # Check blocked languages
            if detected_lang in self.blocked_languages:
                matches.append(ScanMatch(
                    pattern_name="blocked_language",
                    matched_text=f"Language: {detected_lang}",
                    start=0,
                    end=len(content),
                    confidence=confidence,
                    metadata={"language": detected_lang},
                ))
                issues.append(f"Blocked language: {detected_lang}")
                lang_violation = True

            # Check allowed languages (if specified)
            elif self.allowed_languages and detected_lang not in self.allowed_languages:
                matches.append(ScanMatch(
                    pattern_name="disallowed_language",
                    matched_text=f"Language: {detected_lang}",
                    start=0,
                    end=len(content),
                    confidence=confidence,
                    metadata={"language": detected_lang, "allowed": list(self.allowed_languages)},
                ))
                issues.append(f"Language not allowed: {detected_lang}")
                lang_violation = True

        # Check script restrictions
        script_violation = False
        if self.allowed_scripts and primary_script:
            disallowed_scripts = set(scripts.keys()) - set(self.allowed_scripts)
            if disallowed_scripts:
                for script in disallowed_scripts:
                    matches.append(ScanMatch(
                        pattern_name="disallowed_script",
                        matched_text=f"Script: {script} ({scripts[script]} chars)",
                        start=0,
                        end=len(content),
                        confidence=0.8,
                        metadata={"script": script, "count": scripts[script]},
                    ))
                issues.append(f"Disallowed scripts: {', '.join(disallowed_scripts)}")
                script_violation = True

        latency = (time.perf_counter() - start) * 1000

        # Determine result
        passed = not (lang_violation or script_violation)
        max_confidence = max([m.confidence for m in matches], default=0.0)

        if not passed:
            return self._create_result(
                passed=False,
                matches=matches,
                score=max_confidence,
                reason="; ".join(issues),
                latency_ms=latency,
                metadata={
                    "detected_language": detected_lang,
                    "language_confidence": confidence,
                    "scripts": scripts,
                    "primary_script": primary_script,
                },
            )

        return self._create_result(
            passed=True,
            matches=[],
            score=0.0,
            reason=f"Language OK: {detected_lang or 'unknown'}",
            latency_ms=latency,
            metadata={
                "detected_language": detected_lang,
                "language_confidence": confidence,
                "scripts": scripts,
                "primary_script": primary_script,
            },
        )
