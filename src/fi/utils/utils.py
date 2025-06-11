import os
from fi.utils.errors import InvalidAuthError
import tempfile
from enum import Enum

from fi.utils.constants import (
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
    get_base_url,
)


def is_timestamp_in_range(now: int, ts: int):
    max_time = now + (MAX_FUTURE_YEARS_FROM_CURRENT_TIME * 365 * 24 * 60 * 60)
    min_time = now - (MAX_PAST_YEARS_FROM_CURRENT_TIME * 365 * 24 * 60 * 60)
    return min_time <= ts <= max_time


def get_tempfile_path(prefix: str, suffix: str) -> str:
    """Create a temporary file with a random name and given suffix.

    Args:
        prefix (str): Prefix to prepend to the random filename.
        suffix (str): Suffix to append to the random filename.

    Returns:
        str: Path to the created temporary file.
    """
    # Create temporary file with given prefix and suffix
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)  # Close the file descriptor
    return path

def get_base_url_from_env():
    """Get the base URL from environment variable at runtime.
    
    This function is kept for backward compatibility but delegates to
    the centralized get_base_url function from constants.
    
    Returns:
        str: The base URL for the FutureAGI API
    """
    return get_base_url()

def get_keys_from_env():
    api_key = os.getenv("FI_API_KEY")
    secret_key = os.getenv("FI_SECRET_KEY")
    if not api_key or not secret_key:
        raise InvalidAuthError()
    return api_key, secret_key

class ApiKeyName(str, Enum):
    ANYSCALE_API_KEY = "ANYSCALE_API_KEY"
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    AZURE_API_KEY = "AZURE_API_KEY"
    AZURE_AI_API_KEY = "AZURE_AI_API_KEY"
    BEDROCK_API_KEY = "BEDROCK_API_KEY"
    CLOUDFLARE_API_KEY = "CLOUDFLARE_API_KEY"
    COHERE_API_KEY = "COHERE_API_KEY"
    COHERE_CHAT_API_KEY = "COHERE_CHAT_API_KEY"
    DATABRICKS_API_KEY = "DATABRICKS_API_KEY"
    DEEPINFRA_API_KEY = "DEEPINFRA_API_KEY"
    FIREWORKS_AI_API_KEY = "FIREWORKS_AI_API_KEY"
    FIREWORKS_AI_EMBEDDING_MODELS_API_KEY = "FIREWORKS_AI-EMBEDDING-MODELS_API_KEY"
    GEMINI_API_KEY = "GEMINI_API_KEY"
    HUGGINGFACE_API_KEY = "HUGGINGFACE_API_KEY"
    OLLAMA_API_KEY = "OLLAMA_API_KEY"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    PALM_API_KEY = "PALM_API_KEY"
    PERPLEXITY_API_KEY = "PERPLEXITY_API_KEY"
    TEXT_COMPLETION_OPENAI_API_KEY = "TEXT-COMPLETION-OPENAI_API_KEY"
    VERTEX_AI_CHAT_MODELS_API_KEY = "VERTEX_AI-CHAT-MODELS_API_KEY"
    VERTEX_AI_CODE_CHAT_MODELS_API_KEY = "VERTEX_AI-CODE-CHAT-MODELS_API_KEY"
    VERTEX_AI_CODE_TEXT_MODELS_API_KEY = "VERTEX_AI-CODE-TEXT-MODELS_API_KEY"
    VERTEX_AI_EMBEDDING_MODELS_API_KEY = "VERTEX_AI-EMBEDDING-MODELS_API_KEY"
    VERTEX_AI_TEXT_MODELS_API_KEY = "VERTEX_AI-TEXT-MODELS_API_KEY"
    VOYAGE_API_KEY = "VOYAGE_API_KEY"
