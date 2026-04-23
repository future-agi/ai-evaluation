"""Shared fixtures for contract tests.

Contract tests catch SDK ⇄ api drift: template renames, response shape
changes, key-mapping regressions, and the silent-empty bug. They must
run in under 30s and must not depend on LLM determinism.
"""
import os

import pytest


@pytest.fixture(scope="session")
def backend_url() -> str:
    return os.environ["FI_BASE_URL"]


@pytest.fixture(scope="session")
def api_key() -> str:
    return os.environ["FI_API_KEY"]


@pytest.fixture(scope="session")
def secret_key() -> str:
    return os.environ["FI_SECRET_KEY"]


@pytest.fixture(scope="session")
def live_registry(backend_url: str, api_key: str, secret_key: str) -> dict:
    """Fetch the api's eval template registry once per test session."""
    from fi.evals.core.cloud_registry import load_registry

    reg = load_registry(backend_url, api_key, secret_key, force_refresh=True)
    assert reg, f"Registry fetch returned empty from {backend_url}"
    return reg
