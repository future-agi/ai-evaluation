from __future__ import annotations

import logging
from typing import Any

import pytest

from fi.alk.harness import hosted_entrypoint as he
from fi.alk.harness import outbound as ob

SCENARIOS_URL = (
    "https://platform.example/simulate/api/harness/attempts/attempt-1/scenarios/"
)
ACCEPTED = ob.TransportResponse(
    200, {"result": {"run_test_id": "run-test-1", "scenarios": []}}, {}
)


def _refused(message: str, field: str) -> ob.TransportResponse:
    return ob.TransportResponse(
        400,
        {
            "status": False,
            "code": "invalid",
            "message": message,
            "attr": field,
            "details": {field: [message.split(": ", 1)[1]]},
        },
        {},
    )


AGENT_UNKNOWN = _refused("agent: Unknown field.", "agent")
AGENT_NAME_TOO_LONG = _refused(
    "agent_name: Ensure this field has no more than 200 characters.", "agent_name"
)


class _ScriptedTransport:
    def __init__(self, *responses: ob.TransportResponse) -> None:
        self._responses = list(responses)
        self.bodies: list[dict[str, Any]] = []

    def request(
        self, method: str, url: str, *, headers: dict[str, str], json_body: Any
    ) -> ob.TransportResponse:
        assert (method, url) == ("POST", SCENARIOS_URL)
        self.bodies.append(json_body)
        return self._responses.pop(0)


def _client(transport: _ScriptedTransport) -> he.ScenariosClient:
    capabilities = ob.HostedCapabilities.model_validate(
        {
            "schema_version": ob.CAPABILITIES_SCHEMA_VERSION,
            "job_id": "job-1",
            "attempt_id": "attempt-1",
            "attempt_number": 1,
            "fence": "fence-1",
            "expires_at": "2999-01-01T00:00:00.000Z",
            "token": "bearer-token",
            "endpoints": {
                "events": "https://platform.example/events/",
                "results": "https://platform.example/results/",
                "artifacts": "https://platform.example/artifacts/",
                "scenarios": SCENARIOS_URL,
            },
        }
    )
    return he.ScenariosClient(capabilities, transport, sleep=lambda _: None)


PAYLOAD = {
    "operation": "provision",
    "name": "run-1",
    "personas": [],
    "agent_name": "Clinic",
    "agent": {"sdk_version": "0.3.0", "agent": {"kind": "source"}},
}


def test_a_400_naming_agent_is_retried_once_without_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = _ScriptedTransport(AGENT_UNKNOWN, ACCEPTED)

    with caplog.at_level(logging.WARNING, logger=he.__name__):
        result = _client(transport).provision(PAYLOAD)

    assert result == {"run_test_id": "run-test-1", "scenarios": []}
    first, second = transport.bodies
    assert first == PAYLOAD
    assert second == {key: value for key, value in PAYLOAD.items() if key != "agent"}
    assert [record.getMessage() for record in caplog.records] == [
        "provision refused the agent field: class=ScenarioPreallocationError"
    ]


@pytest.mark.parametrize(
    ("responses", "payload", "requests"),
    [
        ((AGENT_NAME_TOO_LONG,), PAYLOAD, 1),
        (
            (AGENT_UNKNOWN,),
            {key: value for key, value in PAYLOAD.items() if key != "agent"},
            1,
        ),
        ((AGENT_UNKNOWN, AGENT_NAME_TOO_LONG), PAYLOAD, 2),
    ],
    ids=["other_field", "no_agent_sent", "retry_refused_too"],
)
def test_any_other_400_is_raised_as_today(
    responses: tuple[ob.TransportResponse, ...], payload: dict[str, Any], requests: int
) -> None:
    transport = _ScriptedTransport(*responses)

    with pytest.raises(he.ScenarioPreallocationError) as raised:
        _client(transport).provision(payload)

    assert raised.value.error.status_code == 400
    assert len(transport.bodies) == requests


def test_a_400_whose_message_is_not_text_is_raised_and_not_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    odd = ob.TransportResponse(
        400, {"status": False, "message": {"agent": ["Unknown field."]}}, {}
    )
    transport = _ScriptedTransport(odd)

    with caplog.at_level(logging.WARNING, logger=he.__name__):
        with pytest.raises(he.ScenarioPreallocationError):
            _client(transport).provision(PAYLOAD)

    assert len(transport.bodies) == 1
    assert not any("Unknown field" in record.getMessage() for record in caplog.records)
