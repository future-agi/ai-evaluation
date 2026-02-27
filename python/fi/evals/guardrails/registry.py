"""
Model Registry for Guardrails.

Central registry for all supported guardrail models with metadata
about backends, model types, and requirements.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from fi.evals.guardrails.config import GuardrailModel


@dataclass
class ModelInfo:
    """Information about a guardrail model."""
    model: GuardrailModel
    backend_class: str  # String reference to avoid circular imports
    backend_module: str  # Module containing the backend class
    model_type: str  # "api", "local", or "vllm"
    hf_model_name: Optional[str] = None  # HuggingFace model name
    vram_required_gb: Optional[float] = None  # Minimum VRAM in GB
    description: str = ""
    is_gated: bool = False  # Requires HF token


# Central registry of all supported models
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # Turing (FutureAGI API)
    "turing_flash": ModelInfo(
        model=GuardrailModel.TURING_FLASH,
        backend_class="TuringBackend",
        backend_module="fi.evals.guardrails.backends.turing",
        model_type="api",
        description="Fast binary classification via FutureAGI API",
    ),
    "turing_safety": ModelInfo(
        model=GuardrailModel.TURING_SAFETY,
        backend_class="TuringBackend",
        backend_module="fi.evals.guardrails.backends.turing",
        model_type="api",
        description="Detailed safety analysis via FutureAGI API",
    ),

    # OpenAI (Free API)
    "openai-moderation": ModelInfo(
        model=GuardrailModel.OPENAI_MODERATION,
        backend_class="OpenAIBackend",
        backend_module="fi.evals.guardrails.backends.openai",
        model_type="api",
        description="OpenAI Moderation API (FREE, 13 categories)",
    ),

    # Azure (Paid API)
    "azure-content-safety": ModelInfo(
        model=GuardrailModel.AZURE_CONTENT_SAFETY,
        backend_class="AzureBackend",
        backend_module="fi.evals.guardrails.backends.azure",
        model_type="api",
        description="Azure Content Safety API (4 categories)",
    ),

    # WildGuard (Local)
    "wildguard-7b": ModelInfo(
        model=GuardrailModel.WILDGUARD_7B,
        backend_class="WildGuardBackend",
        backend_module="fi.evals.guardrails.backends.wildguard",
        model_type="local",
        hf_model_name="allenai/wildguard",
        vram_required_gb=8.0,
        description="AllenAI WildGuard safety classifier",
        is_gated=True,
    ),

    # LlamaGuard (Local)
    "llamaguard-3-8b": ModelInfo(
        model=GuardrailModel.LLAMAGUARD_3_8B,
        backend_class="LlamaGuardBackend",
        backend_module="fi.evals.guardrails.backends.llamaguard",
        model_type="local",
        hf_model_name="meta-llama/Llama-Guard-3-8B",
        vram_required_gb=16.0,
        description="Meta LlamaGuard 3 8B safety classifier",
        is_gated=True,
    ),
    "llamaguard-3-1b": ModelInfo(
        model=GuardrailModel.LLAMAGUARD_3_1B,
        backend_class="LlamaGuardBackend",
        backend_module="fi.evals.guardrails.backends.llamaguard",
        model_type="local",
        hf_model_name="meta-llama/Llama-Guard-3-1B",
        vram_required_gb=4.0,
        description="Meta LlamaGuard 3 1B safety classifier (lightweight)",
        is_gated=True,
    ),

    # Granite Guardian (Local)
    "granite-guardian-3.3-8b": ModelInfo(
        model=GuardrailModel.GRANITE_GUARDIAN_8B,
        backend_class="GraniteGuardianBackend",
        backend_module="fi.evals.guardrails.backends.granite",
        model_type="local",
        hf_model_name="ibm-granite/granite-guardian-3.3-8b",
        vram_required_gb=16.0,
        description="IBM Granite Guardian 3.3 8B",
    ),
    "granite-guardian-3.2-5b": ModelInfo(
        model=GuardrailModel.GRANITE_GUARDIAN_5B,
        backend_class="GraniteGuardianBackend",
        backend_module="fi.evals.guardrails.backends.granite",
        model_type="local",
        hf_model_name="ibm-granite/granite-guardian-3.2-5b",
        vram_required_gb=10.0,
        description="IBM Granite Guardian 3.2 5B (lightweight)",
    ),

    # Qwen3Guard (Local)
    "qwen3guard-8b": ModelInfo(
        model=GuardrailModel.QWEN3GUARD_8B,
        backend_class="Qwen3GuardBackend",
        backend_module="fi.evals.guardrails.backends.qwen",
        model_type="local",
        hf_model_name="Qwen/Qwen3Guard-8B",
        vram_required_gb=16.0,
        description="Alibaba Qwen3Guard 8B (119 languages)",
    ),
    "qwen3guard-4b": ModelInfo(
        model=GuardrailModel.QWEN3GUARD_4B,
        backend_class="Qwen3GuardBackend",
        backend_module="fi.evals.guardrails.backends.qwen",
        model_type="local",
        hf_model_name="Qwen/Qwen3Guard-4B",
        vram_required_gb=8.0,
        description="Alibaba Qwen3Guard 4B (lightweight, 119 languages)",
    ),

    "qwen3guard-0.6b": ModelInfo(
        model=GuardrailModel.QWEN3GUARD_0_6B,
        backend_class="Qwen3GuardBackend",
        backend_module="fi.evals.guardrails.backends.qwen",
        model_type="local",
        hf_model_name="Qwen/Qwen3Guard-0.6B",
        vram_required_gb=1.0,
        description="Alibaba Qwen3Guard 0.6B (ultra-lightweight, 119 languages)",
    ),

    # Generic LLM as guard (prompted for safety classification)
    "llama3.2-3b": ModelInfo(
        model=GuardrailModel.LLAMA_3_2_3B,
        backend_class="GenericLLMGuardBackend",
        backend_module="fi.evals.guardrails.backends.generic_llm",
        model_type="local",
        hf_model_name="meta-llama/Llama-3.2-3B-Instruct",
        vram_required_gb=4.0,
        description="Llama 3.2 3B as prompted safety classifier",
    ),

    # ShieldGemma (Local)
    "shieldgemma-2b": ModelInfo(
        model=GuardrailModel.SHIELDGEMMA_2B,
        backend_class="ShieldGemmaBackend",
        backend_module="fi.evals.guardrails.backends.shieldgemma",
        model_type="local",
        hf_model_name="google/shieldgemma-2b",
        vram_required_gb=4.0,
        description="Google ShieldGemma 2B (lightweight, fast)",
    ),
}


def get_model_info(model: GuardrailModel) -> Optional[ModelInfo]:
    """
    Get model information from registry.

    Args:
        model: GuardrailModel enum value

    Returns:
        ModelInfo or None if not found
    """
    return MODEL_REGISTRY.get(model.value)


def get_backend_class(model: GuardrailModel) -> Type:
    """
    Get the backend class for a model.

    Args:
        model: GuardrailModel enum value

    Returns:
        Backend class

    Raises:
        ValueError: If model not found in registry
        ImportError: If backend module cannot be imported
    """
    info = get_model_info(model)
    if not info:
        raise ValueError(f"Model {model.value} not found in registry")

    # Dynamic import to avoid circular dependencies
    import importlib
    module = importlib.import_module(info.backend_module)
    return getattr(module, info.backend_class)


def list_models(model_type: Optional[str] = None) -> List[ModelInfo]:
    """
    List all models in registry.

    Args:
        model_type: Filter by type ("api", "local", "vllm")

    Returns:
        List of ModelInfo objects
    """
    models = list(MODEL_REGISTRY.values())
    if model_type:
        models = [m for m in models if m.model_type == model_type]
    return models


def list_api_models() -> List[ModelInfo]:
    """List all API-based models."""
    return list_models(model_type="api")


def list_local_models() -> List[ModelInfo]:
    """List all local models."""
    return list_models(model_type="local")
