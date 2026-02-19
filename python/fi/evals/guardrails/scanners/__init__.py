"""
Guardrails Scanners Module.

Provides lightweight, fast content scanners that run before model-based backends.
Scanners are optimized for speed (<10ms) and detect specific threat patterns.

Available Scanners:
- JailbreakScanner: Detects jailbreak and prompt manipulation attempts
  * Pattern-based detection (fast, no dependencies)
  * ML-based detection using Prompt-Guard-86M (requires transformers)
  * Hybrid mode combining both approaches
- CodeInjectionScanner: Detects SQL, shell, and other code injection attacks
- SecretsScanner: Detects leaked API keys, passwords, and credentials
- MaliciousURLScanner: Detects phishing URLs and suspicious links
- InvisibleCharScanner: Detects Unicode manipulation and invisible characters
- LanguageScanner: Detects and filters by language
- TopicRestrictionScanner: Restricts conversations to allowed topics
  * Keyword-based detection (fast, no dependencies)
  * Semantic embedding-based detection (requires sentence-transformers)
  * Hybrid mode combining both approaches
- RegexScanner: Custom regex-based pattern matching
- EvalDelegateScanner: Delegates to existing evaluation templates
  * PII detection (Template 14)
  * Toxicity detection (Template 15)
  * Prompt injection detection (Template 18)
  * Bias detection (Templates 69, 77-79)
  * Content safety (Template 93)
  * NSFW/Sexist content (Templates 17, 20)

Usage:
    from fi.evals.guardrails.scanners import (
        ScannerPipeline,
        JailbreakScanner,
        CodeInjectionScanner,
        SecretsScanner,
        TopicRestrictionScanner,
    )

    # Create a pipeline with multiple scanners (pattern-based, fast)
    pipeline = ScannerPipeline([
        JailbreakScanner(),
        CodeInjectionScanner(),
        SecretsScanner(),
    ])

    # ML-enhanced jailbreak detection
    ml_jailbreak = JailbreakScanner.with_ml()

    # Semantic topic restriction
    semantic_topics = TopicRestrictionScanner.with_embeddings(
        denied_topics=["politics", "religion"],
    )

    # Scan content
    result = pipeline.scan("user input here")
    if not result.passed:
        print(f"Blocked by: {result.blocked_by}")
"""

from fi.evals.guardrails.scanners.base import (
    BaseScanner,
    ScanResult,
    ScanMatch,
    ScannerAction,
    register_scanner,
    get_scanner,
    list_scanners,
)

from fi.evals.guardrails.scanners.pipeline import (
    ScannerPipeline,
    PipelineResult,
)

from fi.evals.guardrails.scanners.jailbreak import JailbreakScanner
from fi.evals.guardrails.scanners.code_injection import CodeInjectionScanner
from fi.evals.guardrails.scanners.secrets import SecretsScanner
from fi.evals.guardrails.scanners.urls import MaliciousURLScanner
from fi.evals.guardrails.scanners.invisible_chars import InvisibleCharScanner
from fi.evals.guardrails.scanners.language import LanguageScanner
from fi.evals.guardrails.scanners.topics import (
    TopicRestrictionScanner,
    TOPIC_KEYWORDS,
    TOPIC_DESCRIPTIONS,
)
from fi.evals.guardrails.scanners.regex import RegexScanner, RegexPattern, COMMON_PATTERNS
from fi.evals.guardrails.scanners.eval_delegate import (
    EvalDelegateScanner,
    EvalCategory,
    EVAL_TEMPLATE_MAP,
    PIIScanner,
    ToxicityScanner,
    PromptInjectionScanner,
    BiasScanner,
    SafetyScanner,
    ContentModerationScanner,
)


def create_default_pipeline(
    jailbreak: bool = True,
    code_injection: bool = True,
    secrets: bool = True,
    urls: bool = False,
    invisible_chars: bool = False,
    language: bool = False,
    topics: bool = False,
    **kwargs,
) -> ScannerPipeline:
    """
    Create a scanner pipeline with sensible defaults.

    Args:
        jailbreak: Enable jailbreak detection
        code_injection: Enable code injection detection
        secrets: Enable secrets detection
        urls: Enable malicious URL detection
        invisible_chars: Enable invisible character detection
        language: Enable language detection
        topics: Enable topic restriction
        **kwargs: Additional scanner configuration

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

    if language:
        scanners.append(LanguageScanner(**kwargs.get("language_config", {})))

    if topics:
        scanners.append(TopicRestrictionScanner(**kwargs.get("topics_config", {})))

    return ScannerPipeline(scanners)


__all__ = [
    # Base classes
    "BaseScanner",
    "ScanResult",
    "ScanMatch",
    "ScannerAction",
    "register_scanner",
    "get_scanner",
    "list_scanners",
    # Pipeline
    "ScannerPipeline",
    "PipelineResult",
    # Scanners
    "JailbreakScanner",
    "CodeInjectionScanner",
    "SecretsScanner",
    "MaliciousURLScanner",
    "InvisibleCharScanner",
    "LanguageScanner",
    "TopicRestrictionScanner",
    "RegexScanner",
    "RegexPattern",
    "COMMON_PATTERNS",
    # Eval Delegate Scanner
    "EvalDelegateScanner",
    "EvalCategory",
    "EVAL_TEMPLATE_MAP",
    "PIIScanner",
    "ToxicityScanner",
    "PromptInjectionScanner",
    "BiasScanner",
    "SafetyScanner",
    "ContentModerationScanner",
    # Topic constants
    "TOPIC_KEYWORDS",
    "TOPIC_DESCRIPTIONS",
    # Factory
    "create_default_pipeline",
]
