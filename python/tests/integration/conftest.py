"""
Integration test fixtures for testing against the real backend.

Requires the backend to be running with test services.
See: core-backend/docs/TESTING.md

Setup:
    1. Start backend test services:
       cd /path/to/core-backend
       docker compose -f docker-compose.test.yml -p futureagi-test up -d

    2. Run backend dev server (optional - for HTTP tests):
       set -a && source .env.test.local && set +a
       python manage.py runserver 0.0.0.0:8001

    3. Set environment variables:
       export FI_API_KEY="test_api_key_12345"
       export FI_SECRET_KEY="test_secret_key_67890"
       export FI_BASE_URL="http://localhost:8001"

    4. Run integration tests:
       pytest tests/integration/ -v
"""

import os
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring backend"
    )
    config.addinivalue_line(
        "markers",
        "requires_model_serving: mark test as requiring model serving service "
        "(not available in local test environments)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with requires_model_serving unless explicitly enabled."""
    # Check if --run-model-serving flag is passed
    run_model_serving = config.getoption("--run-model-serving", default=False)

    if not run_model_serving:
        skip_marker = pytest.mark.skip(
            reason="Requires model serving service. Use --run-model-serving to run."
        )
        for item in items:
            if "requires_model_serving" in item.keywords:
                item.add_marker(skip_marker)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-model-serving",
        action="store_true",
        default=False,
        help="Run tests that require the model serving service",
    )


@pytest.fixture(scope="session")
def backend_url():
    """Get the backend URL from environment or use default test URL."""
    return os.environ.get("FI_BASE_URL", "http://localhost:8001")


@pytest.fixture(scope="session")
def api_credentials():
    """Get API credentials for testing.

    These should match the credentials created by the backend's api_key fixture.
    """
    api_key = os.environ.get("FI_API_KEY", "test_api_key_12345")
    secret_key = os.environ.get("FI_SECRET_KEY", "test_secret_key_67890")
    return {"api_key": api_key, "secret_key": secret_key}


@pytest.fixture(scope="session")
def skip_if_no_backend(backend_url):
    """Skip test if backend is not available."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(backend_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8001

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex((host, port))
        if result != 0:
            pytest.skip(f"Backend not available at {backend_url}")
    except socket.error:
        pytest.skip(f"Backend not available at {backend_url}")
    finally:
        sock.close()


@pytest.fixture
def evaluator(api_credentials, backend_url, skip_if_no_backend):
    """Create an Evaluator instance configured for the test backend."""
    from fi.evals import Evaluator

    return Evaluator(
        fi_api_key=api_credentials["api_key"],
        fi_secret_key=api_credentials["secret_key"],
        fi_base_url=backend_url,
    )
