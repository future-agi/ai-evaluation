"""
Guardrails Scanners — fast, local content screening (<10ms per scanner).

No API keys, no GPU, no external services. Just regex + optional small ML models.

Usage:
    from fi.evals.guardrails.scanners import (
        ScannerPipeline,
        JailbreakScanner,
        CodeInjectionScanner,
        SecretsScanner,
    )

    pipeline = ScannerPipeline([
        JailbreakScanner(),
        CodeInjectionScanner(),
        SecretsScanner(),
    ])

    result = pipeline.scan("user input here")
    if not result.passed:
        print(f"Blocked by: {result.blocked_by}")

Available Scanners:
    JailbreakScanner      — prompt manipulation, DAN attacks, role-play exploits
    CodeInjectionScanner  — SQL, shell, path traversal, SSTI, LDAP, XXE
    SecretsScanner        — API keys, passwords, private keys, JWTs, DB URLs
    MaliciousURLScanner   — phishing, IP URLs, suspicious TLDs, shorteners
    InvisibleCharScanner  — zero-width chars, BIDI overrides, homoglyphs
    LanguageScanner       — language detection and filtering
    TopicRestrictionScanner — keyword/embedding-based topic restriction
    RegexScanner          — custom regex patterns + common PII patterns
"""

from fi.evals.guardrails.scanners.base import (
    ScanResult,
    ScannerAction,
)
from fi.evals.guardrails.scanners.pipeline import (
    ScannerPipeline,
    PipelineResult,
)

# Scanners
from fi.evals.guardrails.scanners.jailbreak import JailbreakScanner
from fi.evals.guardrails.scanners.code_injection import CodeInjectionScanner
from fi.evals.guardrails.scanners.secrets import SecretsScanner
from fi.evals.guardrails.scanners.urls import MaliciousURLScanner
from fi.evals.guardrails.scanners.invisible_chars import InvisibleCharScanner
from fi.evals.guardrails.scanners.language import LanguageScanner
from fi.evals.guardrails.scanners.topics import TopicRestrictionScanner
from fi.evals.guardrails.scanners.regex import RegexScanner, RegexPattern, COMMON_PATTERNS


def create_default_pipeline(
    jailbreak: bool = True,
    code_injection: bool = True,
    secrets: bool = True,
    urls: bool = False,
    invisible_chars: bool = False,
    **kwargs,
) -> ScannerPipeline:
    """
    Create a scanner pipeline with sensible defaults.

    Args:
        jailbreak: Enable jailbreak detection (default: True)
        code_injection: Enable code injection detection (default: True)
        secrets: Enable secrets detection (default: True)
        urls: Enable malicious URL detection (default: False)
        invisible_chars: Enable invisible character detection (default: False)

    Returns:
        Configured ScannerPipeline
    """
    scanners = []
    if jailbreak:
        scanners.append(JailbreakScanner(**kwargs.get("jailbreak_config", {})))
    if code_injection:
        scanners.append(CodeInjectionScanner(**kwargs.get("code_injection_config", {})))
    if secrets:
        scanners.append(SecretsScanner(**kwargs.get("secrets_config", {})))
    if urls:
        scanners.append(MaliciousURLScanner(**kwargs.get("urls_config", {})))
    if invisible_chars:
        scanners.append(InvisibleCharScanner(**kwargs.get("invisible_chars_config", {})))
    return ScannerPipeline(scanners)


__all__ = [
    # Result types
    "ScanResult",
    "ScannerAction",
    "PipelineResult",
    # Pipeline
    "ScannerPipeline",
    "create_default_pipeline",
    # Scanners
    "JailbreakScanner",
    "CodeInjectionScanner",
    "SecretsScanner",
    "MaliciousURLScanner",
    "InvisibleCharScanner",
    "LanguageScanner",
    "TopicRestrictionScanner",
    "RegexScanner",
    # Regex helpers
    "RegexPattern",
    "COMMON_PATTERNS",
]
