"""
Guardrails — content screening for AI applications.

Screen user input, LLM output, and RAG chunks for safety threats
using scanners (fast, local, no API keys) and model backends
(Turing, OpenAI, Azure, or local models via VLLM/HuggingFace).

Quick Start — Scanners Only (no API keys needed):
    from fi.evals.guardrails.scanners import (
        ScannerPipeline, JailbreakScanner, CodeInjectionScanner, SecretsScanner,
    )

    pipeline = ScannerPipeline([
        JailbreakScanner(),
        CodeInjectionScanner(),
        SecretsScanner(),
    ])
    result = pipeline.scan("user input here")
    if not result.passed:
        print(f"Blocked: {result.blocked_by}")

Full Guardrails (with model backend):
    from fi.evals.guardrails import Guardrails, GuardrailsConfig, GuardrailModel

    guardrails = Guardrails(
        config=GuardrailsConfig(models=[GuardrailModel.OPENAI_MODERATION])
    )
    result = guardrails.screen_input("user message")

Ensemble with Weighted Voting:
    from fi.evals.guardrails import (
        Guardrails, GuardrailsConfig, GuardrailModel, AggregationStrategy,
    )

    config = GuardrailsConfig(
        models=[GuardrailModel.TURING_FLASH, GuardrailModel.OPENAI_MODERATION],
        aggregation=AggregationStrategy.WEIGHTED,
        model_weights={"turing_flash": 2.0, "openai-moderation": 1.0},
    )
    guardrails = Guardrails(config=config)
"""

# Config classes (always available — no external dependencies)
from fi.evals.guardrails.config import (
    GuardrailModel,
    RailType,
    AggregationStrategy,
    SafetyCategory,
    GuardrailsConfig,
    ScannerConfig,
    TopicConfig,
    LanguageConfig,
    RegexPatternConfig,
)
from fi.evals.guardrails.types import (
    GuardrailResult,
    GuardrailsResponse,
)

# Optional imports that depend on fi.api (backends, etc.)
_full_api_available = False
try:
    from fi.evals.guardrails.base import Guardrails
    from fi.evals.guardrails.gateway import (
        GuardrailsGateway,
        ScreeningSession,
        AsyncScreeningSession,
    )
    _full_api_available = True
except (ImportError, ModuleNotFoundError):
    Guardrails = None
    GuardrailsGateway = None
    ScreeningSession = None
    AsyncScreeningSession = None

__all__ = [
    # Main class
    "Guardrails",
    # Configuration
    "GuardrailsConfig",
    "GuardrailModel",
    "RailType",
    "AggregationStrategy",
    "SafetyCategory",
    "ScannerConfig",
    "TopicConfig",
    "LanguageConfig",
    "RegexPatternConfig",
    # Response types
    "GuardrailResult",
    "GuardrailsResponse",
    # Gateway
    "GuardrailsGateway",
    "ScreeningSession",
    "AsyncScreeningSession",
]
