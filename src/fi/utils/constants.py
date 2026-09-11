import os

SECRET_KEY_ENVVAR_NAME = "FI_SECRET_KEY"
API_KEY_ENVVAR_NAME = "FI_API_KEY"


def get_base_url() -> str:
    """Return the api base URL from the environment, falling back to prod."""
    return os.getenv("FI_BASE_URL", "https://api.futureagi.com")


DEFAULT_TIMEOUT = 200
DEFAULT_MAX_WORKERS = 8
DEFAULT_MAX_QUEUE = 5000
