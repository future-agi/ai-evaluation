"""
Guardrails Configuration Module.

Defines configuration classes for the guardrails system including:
- GuardrailModel: Enum of supported models
- RailType: Types of rails (input, output, retrieval)
- AggregationStrategy: How to combine results from multiple models
- SafetyCategory: Per-category configuration
- GuardrailsConfig: Main configuration class
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Set
from enum import Enum


class GuardrailModel(Enum):
    """Supported guardrail models."""

    # Turing Models (FutureAGI API)
    TURING_FLASH = "turing_flash"
    TURING_SAFETY = "turing_safety"

    # Local Models
    QWEN3GUARD_8B = "qwen3guard-8b"
    QWEN3GUARD_4B = "qwen3guard-4b"
    GRANITE_GUARDIAN_8B = "granite-guardian-3.3-8b"
    GRANITE_GUARDIAN_5B = "granite-guardian-3.2-5b"
    WILDGUARD_7B = "wildguard-7b"
    LLAMAGUARD_3_8B = "llamaguard-3-8b"
    LLAMAGUARD_3_1B = "llamaguard-3-1b"
    SHIELDGEMMA_2B = "shieldgemma-2b"

    # Third-party API Models
    OPENAI_MODERATION = "openai-moderation"
    ANTHROPIC_SAFETY = "anthropic-safety"
    AZURE_CONTENT_SAFETY = "azure-content-safety"


class RailType(Enum):
    """Types of rails for screening content."""
    INPUT = "input"       # Screen user input before LLM
    OUTPUT = "output"     # Screen LLM response before user
    RETRIEVAL = "retrieval"  # Screen RAG chunks


class AggregationStrategy(Enum):
    """Strategy for combining results from multiple models."""
    ANY = "any"           # Fail if ANY model flags
    ALL = "all"           # Fail if ALL models flag
    MAJORITY = "majority" # Fail if majority flags
    WEIGHTED = "weighted" # Weighted voting


@dataclass
class SafetyCategory:
    """Configuration for a specific safety category."""
    name: str
    enabled: bool = True
    threshold: float = 0.7
    action: Literal["block", "flag", "redact", "warn"] = "block"
    models: List[GuardrailModel] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {self.threshold}")
        if self.action not in ("block", "flag", "redact", "warn"):
            raise ValueError(f"action must be one of block, flag, redact, warn, got {self.action}")


@dataclass
class TopicConfig:
    """Configuration for topic restriction scanner."""
    allowed: List[str] = field(default_factory=list)
    denied: List[str] = field(default_factory=list)
    custom_topics: Dict[str, Set[str]] = field(default_factory=dict)
    min_keyword_matches: int = 2


@dataclass
class LanguageConfig:
    """Configuration for language detection scanner."""
    allowed: List[str] = field(default_factory=list)  # e.g., ["en", "es", "fr"]
    blocked: List[str] = field(default_factory=list)
    allowed_scripts: List[str] = field(default_factory=lambda: ["Latin", "Common"])


@dataclass
class RegexPatternConfig:
    """Configuration for a custom regex pattern."""
    name: str
    pattern: str
    confidence: float = 0.8
    action: Literal["block", "flag", "redact", "warn"] = "block"
    description: str = ""


@dataclass
class ScannerConfig:
    """
    Configuration for content scanners.

    Scanners are lightweight, fast detectors that run before model-based backends.
    They provide quick detection of specific threats like jailbreaks, code injection, etc.

    Attributes:
        enabled: Master switch for all scanners
        jailbreak: Enable jailbreak detection
        code_injection: Enable SQL/shell injection detection
        secrets: Enable secrets/credential detection
        urls: Enable malicious URL detection
        invisible_chars: Enable invisible character detection
        language: Language restriction config
        topics: Topic restriction config
        regex_patterns: Custom regex patterns
        parallel: Run scanners in parallel
        fail_fast: Stop on first failure
    """
    enabled: bool = True

    # Individual scanner toggles
    jailbreak: bool = True
    code_injection: bool = True
    secrets: bool = True
    urls: bool = False  # Disabled by default (can be noisy)
    invisible_chars: bool = False  # Disabled by default
    language: Optional[LanguageConfig] = None
    topics: Optional[TopicConfig] = None

    # Custom regex patterns
    regex_patterns: List[RegexPatternConfig] = field(default_factory=list)
    predefined_patterns: List[str] = field(default_factory=list)  # e.g., ["credit_card", "ssn"]

    # Performance settings
    parallel: bool = True
    fail_fast: bool = True

    # Thresholds
    jailbreak_threshold: float = 0.7
    code_injection_threshold: float = 0.7
    secrets_threshold: float = 0.7
    urls_threshold: float = 0.7


@dataclass
class GuardrailsConfig:
    """
    Main configuration for the guardrails system.

    Attributes:
        models: List of models to use for screening
        rails: Types of rails to enable
        aggregation: How to combine results from multiple models
        categories: Per-category configuration
        timeout_ms: Timeout for each model in milliseconds
        parallel: Whether to run models in parallel
        max_workers: Maximum parallel workers
        fail_open: If True, allow content when guardrails fail
        fallback_model: Model to use if primary fails
    """

    # Model selection
    models: List[GuardrailModel] = field(default_factory=lambda: [
        GuardrailModel.TURING_FLASH,
    ])

    # Rail types to enable
    rails: List[RailType] = field(default_factory=lambda: [
        RailType.INPUT,
        RailType.OUTPUT,
    ])

    # Aggregation strategy for ensemble
    aggregation: AggregationStrategy = AggregationStrategy.ANY

    # Category-specific configurations
    categories: Dict[str, SafetyCategory] = field(default_factory=lambda: {
        "toxicity": SafetyCategory(name="toxicity", threshold=0.7),
        "hate_speech": SafetyCategory(name="hate_speech", threshold=0.7),
        "violence": SafetyCategory(name="violence", threshold=0.8),
        "sexual_content": SafetyCategory(name="sexual_content", threshold=0.8),
        "self_harm": SafetyCategory(name="self_harm", threshold=0.6, action="block"),
        "prompt_injection": SafetyCategory(name="prompt_injection", threshold=0.8),
        "jailbreak": SafetyCategory(name="jailbreak", threshold=0.7),
        "pii": SafetyCategory(name="pii", action="redact"),
        "harmful_content": SafetyCategory(name="harmful_content", threshold=0.7),
        "harassment": SafetyCategory(name="harassment", threshold=0.7),
        "fraud": SafetyCategory(name="fraud", threshold=0.8),
        "illegal_activity": SafetyCategory(name="illegal_activity", threshold=0.8),
    })

    # Performance settings
    timeout_ms: int = 100
    parallel: bool = True
    max_workers: int = 5

    # Fallback behavior
    fail_open: bool = False
    fallback_model: Optional[GuardrailModel] = None

    # Scanner configuration
    scanners: Optional[ScannerConfig] = None

    def __post_init__(self):
        """Validate configuration."""
        if not self.models:
            raise ValueError("At least one model must be specified")
        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {self.timeout_ms}")
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")
