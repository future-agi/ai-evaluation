"""
Code Security Detectors.

Comprehensive vulnerability detection covering:
- Injection: SQL, Command, XSS, XXE, SSRF, Path Traversal
- Secrets: Hardcoded credentials, API keys, private keys
- Cryptography: Weak algorithms, insecure random, weak keys
- Serialization: Unsafe deserialization, JSON injection

Usage:
    from fi.evals.metrics.code_security.detectors import (
        SQLInjectionDetector,
        CommandInjectionDetector,
        HardcodedSecretsDetector,
        WeakCryptoDetector,
        get_all_detectors,
    )

    # Run single detector
    detector = SQLInjectionDetector()
    findings = detector.detect(code, "python")

    # Get all detectors
    detectors = get_all_detectors()
    for detector in detectors:
        findings = detector.detect(code, "python")
"""

from .base import (
    BaseDetector,
    PatternBasedDetector,
    CompositeDetector,
    register_detector,
    get_detector,
    list_detectors,
)

# Import injection detectors
from .injection import (
    SQLInjectionDetector,
    CommandInjectionDetector,
    XSSDetector,
    CodeInjectionDetector,
    XXEDetector,
    SSRFDetector,
    PathTraversalDetector,
)

# Import secrets detectors
from .secrets import (
    HardcodedSecretsDetector,
    SensitiveLoggingDetector,
)

# Import cryptography detectors
from .cryptography import (
    WeakCryptoDetector,
    InsecureRandomDetector,
    WeakKeySizeDetector,
    HardcodedIVDetector,
)

# Import serialization detectors
from .serialization import (
    UnsafeDeserializationDetector,
    JSONInjectionDetector,
)


def get_all_detectors(
    enabled_only: bool = True,
    languages: list = None,
) -> list:
    """
    Get instances of all registered detectors.

    Args:
        enabled_only: Only return enabled detectors
        languages: Filter to detectors supporting these languages

    Returns:
        List of detector instances
    """
    detectors = []

    for name in list_detectors():
        detector_cls = get_detector(name)
        if detector_cls:
            detector = detector_cls()
            if enabled_only and not detector.enabled:
                continue
            if languages:
                if not any(detector.supports_language(lang) for lang in languages):
                    continue
            detectors.append(detector)

    return detectors


def get_detectors_by_category(category: str) -> list:
    """Get all detectors for a specific category."""
    detectors = []

    for name in list_detectors():
        detector_cls = get_detector(name)
        if detector_cls:
            detector = detector_cls()
            if detector.category.value == category:
                detectors.append(detector)

    return detectors


def get_detectors_by_cwe(cwe_id: str) -> list:
    """Get all detectors that check for a specific CWE."""
    detectors = []

    for name in list_detectors():
        detector_cls = get_detector(name)
        if detector_cls:
            detector = detector_cls()
            if cwe_id in detector.cwe_ids:
                detectors.append(detector)

    return detectors


# Convenience function to run all detectors
def scan_code(code: str, language: str = "python") -> list:
    """
    Scan code with all available detectors.

    Args:
        code: Source code to scan
        language: Programming language

    Returns:
        List of SecurityFinding objects
    """
    from ..analyzer import CodeAnalyzer

    analyzer = CodeAnalyzer()
    analysis = analyzer.analyze(code, language)

    all_findings = []
    for detector in get_all_detectors(languages=[language]):
        findings = detector.detect(code, language, analysis)
        all_findings.extend(findings)

    return all_findings


__all__ = [
    # Base classes
    "BaseDetector",
    "PatternBasedDetector",
    "CompositeDetector",
    # Registry functions
    "register_detector",
    "get_detector",
    "list_detectors",
    # Helper functions
    "get_all_detectors",
    "get_detectors_by_category",
    "get_detectors_by_cwe",
    "scan_code",
    # Injection detectors
    "SQLInjectionDetector",
    "CommandInjectionDetector",
    "XSSDetector",
    "CodeInjectionDetector",
    "XXEDetector",
    "SSRFDetector",
    "PathTraversalDetector",
    # Secrets detectors
    "HardcodedSecretsDetector",
    "SensitiveLoggingDetector",
    # Cryptography detectors
    "WeakCryptoDetector",
    "InsecureRandomDetector",
    "WeakKeySizeDetector",
    "HardcodedIVDetector",
    # Serialization detectors
    "UnsafeDeserializationDetector",
    "JSONInjectionDetector",
]
