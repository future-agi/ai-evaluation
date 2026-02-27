"""
Guardrails Backends — model-based content screening.

Users normally don't import backends directly. Instead, configure via
GuardrailModel enum in GuardrailsConfig and let the registry handle
instantiation.

For custom backends, subclass BaseBackend or LocalModelBackend.
"""

from fi.evals.guardrails.backends.base import BaseBackend
from fi.evals.guardrails.backends.turing import TuringBackend

# API backends
from fi.evals.guardrails.backends.openai import OpenAIBackend
from fi.evals.guardrails.backends.azure import AzureBackend

# Local model base (for custom backends)
from fi.evals.guardrails.backends.local_base import LocalModelBackend

# Local model backends
from fi.evals.guardrails.backends.wildguard import WildGuardBackend
from fi.evals.guardrails.backends.llamaguard import LlamaGuardBackend
from fi.evals.guardrails.backends.granite import GraniteGuardianBackend
from fi.evals.guardrails.backends.qwen import Qwen3GuardBackend
from fi.evals.guardrails.backends.shieldgemma import ShieldGemmaBackend
from fi.evals.guardrails.backends.generic_llm import GenericLLMGuardBackend

__all__ = [
    # Base classes (for extension)
    "BaseBackend",
    "LocalModelBackend",
    # Backends
    "TuringBackend",
    "OpenAIBackend",
    "AzureBackend",
    "WildGuardBackend",
    "LlamaGuardBackend",
    "GraniteGuardianBackend",
    "Qwen3GuardBackend",
    "ShieldGemmaBackend",
    "GenericLLMGuardBackend",
]
