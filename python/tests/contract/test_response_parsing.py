"""Pin response parsing — both legacy and revamped schemas decode.

If the api changes response shape, this fails before any integration
test. The two shapes have historically been:

  legacy: outputType/evalId (camelCase)
  revamp: output_type/eval_id (snake_case)

The SDK must accept both so partially rolled-out backends don't break
customers mid-deploy.
"""
from unittest.mock import Mock

from fi.evals.evaluator import EvalResponseHandler


def _mock_response(payload: dict, status: int = 200) -> Mock:
    m = Mock()
    m.ok = status == 200
    m.status_code = status
    m.json.return_value = payload
    m.text = str(payload)
    return m


def test_revamp_snake_case_response():
    payload = {
        "status": True,
        "result": [
            {
                "evaluations": [
                    {
                        "name": "toxicity",
                        "output": "Passed",
                        "reason": "fine",
                        "runtime": 1234,
                        "output_type": "Pass/Fail",
                        "eval_id": "uuid-1",
                    }
                ]
            }
        ],
    }
    result = EvalResponseHandler._parse_success(_mock_response(payload))
    assert len(result.eval_results) == 1
    r = result.eval_results[0]
    assert r.name == "toxicity"
    assert r.output == "Passed"
    assert r.output_type == "Pass/Fail"
    assert r.eval_id == "uuid-1"


def test_legacy_camel_case_response_still_decodes():
    """Belt-and-suspenders: should the api ever emit the old shape, we
    shouldn't crash — a blank output_type / eval_id is acceptable.
    """
    payload = {
        "status": True,
        "result": [
            {
                "evaluations": [
                    {
                        "name": "toxicity",
                        "output": "Passed",
                        "reason": "fine",
                        "runtime": 1000,
                        "outputType": "Pass/Fail",  # legacy key
                        "evalId": "uuid-legacy",  # legacy key
                    }
                ]
            }
        ],
    }
    result = EvalResponseHandler._parse_success(_mock_response(payload))
    assert len(result.eval_results) == 1
    r = result.eval_results[0]
    assert r.name == "toxicity"
    assert r.output == "Passed"


def test_empty_result_list_returns_empty_batch():
    """api returns no evaluations array at all — shouldn't crash."""
    payload = {"status": True, "result": []}
    result = EvalResponseHandler._parse_success(_mock_response(payload))
    assert result.eval_results == []


def test_unwrapped_async_eval_result():
    """Async submission returns the eval directly, not wrapped in evaluations[]."""
    payload = {
        "status": True,
        "result": [
            {
                "name": "toxicity",
                "output": "Pending",
                "reason": "still processing",
                "runtime": 0,
                "output_type": "",
                "eval_id": "async-uuid",
            }
        ],
    }
    result = EvalResponseHandler._parse_success(_mock_response(payload))
    assert len(result.eval_results) == 1
    assert result.eval_results[0].eval_id == "async-uuid"
