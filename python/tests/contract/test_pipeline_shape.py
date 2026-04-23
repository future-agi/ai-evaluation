"""Pin the payload shape ``evaluate_pipeline`` / ``get_pipeline_results``
send to the backend. If someone tweaks the client and forgets to match
the api contract, this catches it before integration tests run.
"""
from unittest.mock import MagicMock, patch

import pytest

from fi.evals import Evaluator
from fi.utils.routes import Routes


@pytest.fixture(autouse=True)
def _fake_creds(monkeypatch):
    monkeypatch.setenv("FI_API_KEY", "fake")
    monkeypatch.setenv("FI_SECRET_KEY", "fake")
    monkeypatch.setenv("FI_BASE_URL", "http://fake")


def _mock_response(body: dict, status: int = 200):
    resp = MagicMock()
    resp.ok = status == 200
    resp.status_code = status
    resp.json.return_value = body
    return resp


def test_evaluate_pipeline_payload_shape():
    """POSTs {project_name, version, eval_data} to evaluate_pipeline route."""
    ev = Evaluator()
    captured = {}

    def capture(config, **_):
        captured["method"] = config.method
        captured["url"] = config.url
        captured["json"] = config.json
        return _mock_response({"status": True, "result": {"evaluation_run_id": "x"}})

    with patch.object(ev, "request", side_effect=capture):
        ev.evaluate_pipeline(
            project_name="my-project",
            version="v1",
            eval_data=[{"eval_template": "toxicity", "inputs": {"output": ["hi"]}}],
        )

    assert Routes.evaluate_pipeline.value in captured["url"]
    payload = captured["json"]
    assert payload["project_name"] == "my-project"
    assert payload["version"] == "v1"
    assert payload["eval_data"][0]["eval_template"] == "toxicity"


def test_get_pipeline_results_payload_shape():
    """GET sends versions as comma-joined query param."""
    ev = Evaluator()
    captured = {}

    def capture(config, **_):
        captured["method"] = config.method
        captured["url"] = config.url
        captured["params"] = config.params
        return _mock_response({"status": True, "result": {}})

    with patch.object(ev, "request", side_effect=capture):
        ev.get_pipeline_results(project_name="my-project", versions=["v1", "v2"])

    assert Routes.evaluate_pipeline.value in captured["url"]
    params = captured["params"]
    assert params["project_name"] == "my-project"
    assert params["versions"] == "v1,v2"


def test_get_pipeline_results_rejects_non_list_versions():
    ev = Evaluator()
    with pytest.raises(TypeError, match="list of strings"):
        ev.get_pipeline_results(project_name="p", versions="v1")  # type: ignore[arg-type]
