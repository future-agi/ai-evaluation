"""
Guardrails Module - Modal Gateway for AI Safety.

Comprehensive content screening system with support for:
- Multiple models (Turing, OpenAI, Azure, local models)
- Input, output, and retrieval rails
- Ensemble mode with configurable aggregation
- Async and sync APIs
- Auto-discovery of available backends

Quick Start:
    from fi.evals.guardrails import Guardrails

    # Default: Uses Turing Flash for speed
    guardrails = Guardrails()

    # Screen user input
    result = guardrails.screen_input("How can I help you today?")
    if result.passed:
        print("Content is safe")
    else:
        print(f"Blocked: {result.blocked_categories}")

Discover Available Backends:
    from fi.evals.guardrails import Guardrails

    # See what's available
    available = Guardrails.discover_backends()
    print(f"Available: {[m.value for m in available]}")

    # Get detailed info
    details = Guardrails.get_backend_details()
    for model, info in details.items():
        print(f"{model}: {info['status']}")

Using OpenAI Moderation (FREE):
    import os
    os.environ["OPENAI_API_KEY"] = "sk-..."

    from fi.evals.guardrails import Guardrails, GuardrailsConfig, GuardrailModel

    guardrails = Guardrails(
        config=GuardrailsConfig(models=[GuardrailModel.OPENAI_MODERATION])
    )
    result = guardrails.screen_input("some content")

Using Local Models (WildGuard, LlamaGuard, etc.):
    # Option 1: Via VLLM server
    os.environ["VLLM_SERVER_URL"] = "http://localhost:28000"

    # Option 2: Direct model loading (requires GPU)
    guardrails = Guardrails(
        config=GuardrailsConfig(models=[GuardrailModel.WILDGUARD_7B])
    )

Ensemble Mode:
    from fi.evals.guardrails import (
        Guardrails,
        GuardrailsConfig,
        GuardrailModel,
        AggregationStrategy,
    )

    config = GuardrailsConfig(
        models=[
            GuardrailModel.TURING_FLASH,
            GuardrailModel.OPENAI_MODERATION,
        ],
        aggregation=AggregationStrategy.MAJORITY,
    )
    guardrails = Guardrails(config=config)
"""

# Config classes (always available - no external dependencies)
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
# These may fail if the futureagi package is not installed
_full_api_available = False
try:
    from fi.evals.guardrails.base import Guardrails
    from fi.evals.guardrails.discovery import (
        discover_backends,
        get_backend_details,
        BackendDiscovery,
    )
    from fi.evals.guardrails.registry import (
        MODEL_REGISTRY,
        ModelInfo,
        get_model_info,
        list_models,
        list_api_models,
        list_local_models,
    )
    from fi.evals.guardrails.gateway import (
        GuardrailsGateway,
        Gateway,
        ScreeningSession,
        AsyncScreeningSession,
    )
    _full_api_available = True
except (ImportError, ModuleNotFoundError):
    # Provide stubs for missing components
    Guardrails = None
    discover_backends = None
    get_backend_details = None
    BackendDiscovery = None
    MODEL_REGISTRY = {}
    ModelInfo = None
    get_model_info = None
    list_models = None
    list_api_models = None
    list_local_models = None
    GuardrailsGateway = None
    Gateway = None
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
    # Discovery
    "discover_backends",
    "get_backend_details",
    "BackendDiscovery",
    # Registry
    "MODEL_REGISTRY",
    "ModelInfo",
    "get_model_info",
    "list_models",
    "list_api_models",
    "list_local_models",
    # Gateway
    "GuardrailsGateway",
    "Gateway",
    "ScreeningSession",
    "AsyncScreeningSession",
]
