import os

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
