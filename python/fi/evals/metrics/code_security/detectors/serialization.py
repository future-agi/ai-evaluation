"""
Serialization Vulnerability Detectors.

Detects unsafe deserialization vulnerabilities:
- CWE-502: Deserialization of Untrusted Data
"""

import re
from typing import List, Optional

from .base import BaseDetector, register_detector
from ..types import (
    SecurityFinding,
    Severity,
    VulnerabilityCategory,
)
from ..analyzer import AnalysisResult


@register_detector("unsafe_deserialization")
class UnsafeDeserializationDetector(BaseDetector):
    """
    Detects unsafe deserialization (CWE-502).

    Identifies:
    - pickle.loads with untrusted data
    - yaml.load without safe_load
    - Unsafe JSON deserialization with code execution
    - Java ObjectInputStream

    Examples of vulnerable code:
        pickle.loads(user_data)
        yaml.load(data)  # without Loader=SafeLoader
        marshal.loads(data)
    """

    name = "unsafe_deserialization"
    cwe_ids = ["CWE-502"]
    category = VulnerabilityCategory.SERIALIZATION
    description = "Deserialization of untrusted data"
    default_severity = Severity.CRITICAL

    UNSAFE_PATTERNS = {
        "python": [
            # Pickle
            (r"pickle\.loads?\s*\(", "pickle.load(s) can execute arbitrary code", Severity.CRITICAL),
            (r"cPickle\.loads?\s*\(", "cPickle.load(s) can execute arbitrary code", Severity.CRITICAL),
            (r"_pickle\.loads?\s*\(", "_pickle.load(s) can execute arbitrary code", Severity.CRITICAL),
            # YAML
            (r"yaml\.load\s*\([^)]*\)", "yaml.load without SafeLoader can execute code", Severity.HIGH),
            (r"yaml\.unsafe_load\s*\(", "yaml.unsafe_load can execute code", Severity.CRITICAL),
            # Marshal
            (r"marshal\.loads?\s*\(", "marshal.load(s) can execute arbitrary code", Severity.CRITICAL),
            # Shelve
            (r"shelve\.open\s*\(", "shelve.open can execute code on load", Severity.HIGH),
            # Dill
            (r"dill\.loads?\s*\(", "dill.load(s) can execute arbitrary code", Severity.CRITICAL),
        ],
        "javascript": [
            # Node.js serialize
            (r"serialize\.unserialize\s*\(", "node-serialize unserialize can execute code", Severity.CRITICAL),
            # eval-based JSON
            (r"eval\s*\(\s*['\"]?\s*\(", "eval used for parsing", Severity.CRITICAL),
        ],
        "java": [
            # ObjectInputStream
            (r"ObjectInputStream\s*\(", "ObjectInputStream can execute arbitrary code", Severity.CRITICAL),
            (r"\.readObject\s*\(", "readObject can execute arbitrary code", Severity.CRITICAL),
            # XMLDecoder
            (r"XMLDecoder\s*\(", "XMLDecoder can execute arbitrary code", Severity.CRITICAL),
            # XStream
            (r"XStream\.fromXML\s*\(", "XStream can execute arbitrary code", Severity.CRITICAL),
            # SnakeYAML
            (r"new\s+Yaml\s*\(\s*\)", "SnakeYAML without SafeConstructor", Severity.HIGH),
        ],
        "php": [
            (r"unserialize\s*\(", "unserialize can execute code via __wakeup", Severity.CRITICAL),
        ],
    }

    # Safe patterns that indicate proper usage
    SAFE_PATTERNS = {
        "python": [
            r"yaml\.safe_load",
            r"Loader\s*=\s*SafeLoader",
            r"Loader\s*=\s*yaml\.SafeLoader",
        ],
        "java": [
            r"SafeConstructor",
            r"setAllowedTypes",
        ],
    }

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect unsafe deserialization."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        patterns = self.UNSAFE_PATTERNS.get(lang_lower, [])
        safe_patterns = self.SAFE_PATTERNS.get(lang_lower, [])

        for i, line in enumerate(lines, 1):
            # Skip if safe pattern is present
            if any(re.search(p, line, re.IGNORECASE) for p in safe_patterns):
                continue

            for pattern, description, severity in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Special case for yaml.load - check if SafeLoader is used
                    if "yaml.load" in pattern and "Loader" in line:
                        continue  # Safe usage with Loader parameter

                    findings.append(self.create_finding(
                        vulnerability_type="Unsafe Deserialization",
                        description=description,
                        line=i,
                        snippet=line.strip()[:100],
                        severity=severity,
                        confidence=0.9,
                        suggested_fix="Use safe alternatives: json.loads, yaml.safe_load, or validate input before deserializing.",
                    ))

        return findings


@register_detector("json_injection")
class JSONInjectionDetector(BaseDetector):
    """
    Detects potential JSON injection vulnerabilities.

    Identifies:
    - String concatenation to build JSON
    - Unescaped user input in JSON strings

    Examples of vulnerable code:
        json_str = '{"user": "' + user_input + '"}'
    """

    name = "json_injection"
    cwe_ids = ["CWE-94"]
    category = VulnerabilityCategory.INJECTION
    description = "JSON injection via string concatenation"
    default_severity = Severity.MEDIUM

    JSON_CONCAT_PATTERNS = [
        r'["\'][{[].*["\'].*\+',  # JSON string concatenation
        r'\+.*["\'][}\]].*["\']',  # Concatenation with JSON
    ]

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect JSON injection."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Look for JSON-like patterns with concatenation
            if any(re.search(p, line) for p in self.JSON_CONCAT_PATTERNS):
                # Check for JSON structure indicators
                if any(c in line for c in ['{', '[', '}']):
                    findings.append(self.create_finding(
                        vulnerability_type="JSON Injection",
                        description="Building JSON via string concatenation",
                        line=i,
                        snippet=line.strip()[:100],
                        confidence=0.7,
                        suggested_fix="Use json.dumps() to properly escape values.",
                    ))

        return findings


__all__ = [
    "UnsafeDeserializationDetector",
    "JSONInjectionDetector",
]
