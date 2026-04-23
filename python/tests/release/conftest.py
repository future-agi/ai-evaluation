"""Release-gate test fixtures. Runs against a live api.

These tests gate the dev → main merge. They hit a real backend with
real LLM calls, so expect 3-5s per call. Full suite budget: ~3 min.

Required env vars:
    FI_API_KEY
    FI_SECRET_KEY
    FI_BASE_URL  (e.g. https://dev.api.futureagi.com)
"""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_api_creds():
    required = ["FI_API_KEY", "FI_SECRET_KEY", "FI_BASE_URL"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        pytest.skip(f"Missing env vars for release tests: {', '.join(missing)}")


@pytest.fixture(scope="session")
def evaluator():
    from fi.evals import Evaluator
    return Evaluator()


@pytest.fixture(scope="session")
def manager():
    from fi.evals import EvalTemplateManager
    return EvalTemplateManager()
