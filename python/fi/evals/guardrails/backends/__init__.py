"""
Guardrails Backends Module.

Provides different backend implementations for content screening:

API Backends:
- TuringBackend: FutureAGI Turing API (fast, accurate)
- OpenAIBackend: OpenAI Moderation API (FREE, 13 categories)
- AzureBackend: Azure Content Safety API (4 categories)

Local Model Backends:
- WildGuardBackend: AllenAI WildGuard (7B)
- LlamaGuardBackend: Meta LlamaGuard 3 (8B, 1B)
- GraniteGuardianBackend: IBM Granite Guardian (8B, 5B)
- Qwen3GuardBackend: Alibaba Qwen3Guard (8B, 4B, 119 languages)
- ShieldGemmaBackend: Google ShieldGemma (2B, fast)

Infrastructure:
- BaseBackend: Abstract base class
- LocalModelBackend: Base for local models
- VLLMClient: Client for VLLM servers
"""

from fi.evals.guardrails.backends.base import BaseBackend
from fi.evals.guardrails.backends.turing import TuringBackend

# API backends (always available)
from fi.evals.guardrails.backends.openai import OpenAIBackend
from fi.evals.guardrails.backends.azure import AzureBackend

# Local model infrastructure
from fi.evals.guardrails.backends.local_base import LocalModelBackend
from fi.evals.guardrails.backends.vllm_client import VLLMClient, VLLMResponse, get_vllm_url

# Local model backends
from fi.evals.guardrails.backends.wildguard import WildGuardBackend
from fi.evals.guardrails.backends.llamaguard import LlamaGuardBackend
from fi.evals.guardrails.backends.granite import GraniteGuardianBackend
from fi.evals.guardrails.backends.qwen import Qwen3GuardBackend
from fi.evals.guardrails.backends.shieldgemma import ShieldGemmaBackend

__all__ = [
    # Base classes
    "BaseBackend",
    "LocalModelBackend",
    # API backends
    "TuringBackend",
    "OpenAIBackend",
    "AzureBackend",
    # Local model backends
    "WildGuardBackend",
    "LlamaGuardBackend",
    "GraniteGuardianBackend",
    "Qwen3GuardBackend",
    "ShieldGemmaBackend",
    # VLLM client
    "VLLMClient",
    "VLLMResponse",
    "get_vllm_url",
]
