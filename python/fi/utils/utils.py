import os
from enum import Enum

from .constants import get_base_url
from .errors import InvalidAuthError


def get_base_url_from_env() -> str:
    """Backward-compat alias — use ``get_base_url()`` from constants."""
    return get_base_url()


def get_keys_from_env() -> tuple[str, str]:
    api_key = os.getenv("FI_API_KEY")
    secret_key = os.getenv("FI_SECRET_KEY")
    if not api_key or not secret_key:
        raise InvalidAuthError()
    return api_key, secret_key


class ApiKeyName(str, Enum):
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    AZURE_API_KEY = "AZURE_API_KEY"
    AZURE_AI_API_KEY = "AZURE_AI_API_KEY"
    BEDROCK_API_KEY = "BEDROCK_API_KEY"
    CLOUDFLARE_API_KEY = "CLOUDFLARE_API_KEY"
    COHERE_API_KEY = "COHERE_API_KEY"
    DATABRICKS_API_KEY = "DATABRICKS_API_KEY"
    DEEPINFRA_API_KEY = "DEEPINFRA_API_KEY"
    FIREWORKS_AI_API_KEY = "FIREWORKS_AI_API_KEY"
    GEMINI_API_KEY = "GEMINI_API_KEY"
    HUGGINGFACE_API_KEY = "HUGGINGFACE_API_KEY"
    OLLAMA_API_KEY = "OLLAMA_API_KEY"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    PERPLEXITY_API_KEY = "PERPLEXITY_API_KEY"
    VOYAGE_API_KEY = "VOYAGE_API_KEY"
