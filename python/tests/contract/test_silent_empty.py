"""Regression test for the silent-empty bug.

Before the fix: backend 4xx → ``BatchRunResult(eval_results=[])`` silently,
caller's ``batch.eval_results[0]`` crashes with IndexError and no clue why.

After the fix: failed ``EvalResult`` with readable error text always
populated, so downstream code can detect failure deterministically.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from fi.evals import Evaluator


@pytest.fixture(autouse=True)
def fake_creds(monkeypatch):
    monkeypatch.setenv("FI_API_KEY", "fake")
    monkeypatch.setenv("FI_SECRET_KEY", "fake")
    monkeypatch.setenv("FI_BASE_URL", "http://fake")


def _failed_response(body: dict, status: int = 400) -> MagicMock:
    resp = MagicMock()
    resp.ok = False
    resp.status_code = status
    resp.json.return_value = body
    resp.text = str(body)
    return resp


def test_400_returns_failed_eval_result_not_empty_batch():
    """The whole point: caller must see a concrete failure, not an empty list."""
    ev = Evaluator()

    # Bypass the dynamic registry lookup so the test is pure-unit.
    with patch("fi.evals.core.cloud_registry.map_inputs_to_backend", side_effect=lambda n, i, **_: i):
        # Stub the HTTP layer to return a 400.
        with patch.object(
            ev,
            "request",
            side_effect=Exception("Evaluation failed with a 400 Bad Request"),
        ):
            result = ev.evaluate(
                eval_templates="toxicity",
                inputs={"output": "x"},
                model_name="turing_flash",
            )

    assert len(result.eval_results) == 1, (
        "Regression: silent empty BatchRunResult returned on 4xx. "
        "Caller's batch.eval_results[0] would crash."
    )
    r = result.eval_results[0]
    assert r.output is None, "Failed result must have no output value"
    assert r.reason, "Failed result must carry the backend error text"
    assert "400" in r.reason
    assert r.name == "toxicity"


def test_500_returns_failed_eval_result():
    """Same for 5xx — never silent."""
    ev = Evaluator()
    with patch("fi.evals.core.cloud_registry.map_inputs_to_backend", side_effect=lambda n, i, **_: i):
        with patch.object(
            ev,
            "request",
            side_effect=Exception("Error in evaluation: 500"),
        ):
            result = ev.evaluate(
                eval_templates="toxicity",
                inputs={"output": "x"},
                model_name="turing_flash",
            )
    assert len(result.eval_results) == 1
    assert "500" in result.eval_results[0].reason
