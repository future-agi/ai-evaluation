from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

import httpx

from fi.simulate.agent.definition import ProviderEvidenceConfig
from fi.simulate.evidence.providers.base import EvidenceContext
from fi.simulate.evidence.providers.retell import RetellEvidenceSource


def test_retell_originator_response_uses_exact_call_id(tmp_path) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        # terminal status on the first GET so the poll exits without sleeping
        return httpx.Response(
            200, json={"call_id": "call_retell_123", "call_status": "ended"}
        )

    async def run() -> None:
        client = httpx.AsyncClient(
            base_url="https://api.retellai.com",
            transport=httpx.MockTransport(handler),
        )
        source = RetellEvidenceSource(
            ProviderEvidenceConfig(
                provider="retell",
                call_id_source="originator_response",
            ),
            api_key="test-key",
            client=client,
        )
        await source.connect(
            EvidenceContext(
                run_id="run_retell",
                test_case_id="case_retell",
                case_directory=tmp_path,
                started_at=datetime.now(timezone.utc),
                call_id_hint="call_retell_123",
            )
        )
        payload = await source._locate_and_fetch_call()
        await client.aclose()
        assert payload == {"call_id": "call_retell_123", "call_status": "ended"}

    asyncio.run(run())

    assert requested_paths == ["/v2/get-call/call_retell_123"]


def test_retell_polling_uses_v3_typed_filters(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v3/list-calls":
            return httpx.Response(
                200,
                json={
                    "calls": [
                        {
                            "call_id": "call_retell_456",
                            "call_status": "ended",
                            "end_timestamp": 1_753_881_600_000,
                        }
                    ]
                },
            )
        if request.url.path == "/v2/get-call/call_retell_456":
            return httpx.Response(200, json={"call_id": "call_retell_456"})
        raise AssertionError(request.url.path)

    async def run() -> dict:
        client = httpx.AsyncClient(
            base_url="https://api.retellai.com",
            transport=httpx.MockTransport(handler),
        )
        source = RetellEvidenceSource(
            ProviderEvidenceConfig(
                provider="retell",
                call_id_source="polling_window",
                polling_window_seconds=60,
            ),
            api_key="test-key",
            client=client,
        )
        await source.connect(
            EvidenceContext(
                run_id="run_retell",
                test_case_id="case_retell",
                case_directory=tmp_path,
                started_at=datetime(2026, 7, 30, 10, 0, tzinfo=timezone.utc),
                caller_phone="+14155550100",
            )
        )
        payload = await source._locate_and_fetch_call()
        await client.aclose()
        assert payload is not None
        return payload

    assert asyncio.run(run()) == {"call_id": "call_retell_456"}
    request_payload = json.loads(requests[0].content)
    filters = request_payload["filter_criteria"]
    assert filters["start_timestamp"]["type"] == "range"
    assert filters["start_timestamp"]["op"] == "bt"
    assert filters["from_number"] == {
        "type": "string",
        "op": "eq",
        "value": "+14155550100",
    }


def test_retell_hint_path_polls_until_ended(tmp_path) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        status = "ongoing" if len(requested_paths) == 1 else "ended"
        return httpx.Response(
            200, json={"call_id": "call_retell_789", "call_status": status}
        )

    async def run():
        client = httpx.AsyncClient(
            base_url="https://api.retellai.com",
            transport=httpx.MockTransport(handler),
        )
        source = RetellEvidenceSource(
            ProviderEvidenceConfig(
                provider="retell",
                call_id_source="originator_response",
                poll_interval_seconds=0.01,
                # deadline isn't the subject here; keep it well clear of the two GETs
                poll_deadline_seconds=5.0,
            ),
            api_key="test-key",
            client=client,
        )
        await source.connect(
            EvidenceContext(
                run_id="run_retell",
                test_case_id="case_retell",
                case_directory=tmp_path,
                started_at=datetime.now(timezone.utc),
                call_id_hint="call_retell_789",
            )
        )
        result = await source.fetch_final()
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert requested_paths == [
        "/v2/get-call/call_retell_789",
        "/v2/get-call/call_retell_789",
    ]
    assert result.summary.available is True


def test_retell_hint_path_not_connected_is_terminal_on_first_get(tmp_path) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(
            200, json={"call_id": "call_retell_999", "call_status": "not_connected"}
        )

    async def run():
        client = httpx.AsyncClient(
            base_url="https://api.retellai.com",
            transport=httpx.MockTransport(handler),
        )
        source = RetellEvidenceSource(
            ProviderEvidenceConfig(
                provider="retell",
                call_id_source="originator_response",
                poll_interval_seconds=0.01,
                # deadline isn't the subject here; keep it well clear of the one GET
                poll_deadline_seconds=5.0,
            ),
            api_key="test-key",
            client=client,
        )
        await source.connect(
            EvidenceContext(
                run_id="run_retell",
                test_case_id="case_retell",
                case_directory=tmp_path,
                started_at=datetime.now(timezone.utc),
                call_id_hint="call_retell_999",
            )
        )
        result = await source.fetch_final()
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert requested_paths == ["/v2/get-call/call_retell_999"]
    assert result.summary.available is True


def test_retell_hint_path_deadline_returns_unavailable_without_raising(
    tmp_path,
) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        # each GET sees one more transcript entry, so "the last payload" is observable
        return httpx.Response(
            200,
            json={
                "call_id": "call_retell_stuck",
                "call_status": "ongoing",
                "transcript_with_tool_calls": [{"role": "agent"}]
                * len(requested_paths),
            },
        )

    async def run():
        client = httpx.AsyncClient(
            base_url="https://api.retellai.com",
            transport=httpx.MockTransport(handler),
        )
        source = RetellEvidenceSource(
            ProviderEvidenceConfig(
                provider="retell",
                call_id_source="originator_response",
                poll_interval_seconds=0.01,
                poll_deadline_seconds=0.05,
            ),
            api_key="test-key",
            client=client,
        )
        await source.connect(
            EvidenceContext(
                run_id="run_retell",
                test_case_id="case_retell",
                case_directory=tmp_path,
                started_at=datetime.now(timezone.utc),
                call_id_hint="call_retell_stuck",
            )
        )
        result = await source.fetch_final()
        await client.aclose()
        return result

    t0 = time.monotonic()
    result = asyncio.run(run())
    elapsed = time.monotonic() - t0
    assert all(path == "/v2/get-call/call_retell_stuck" for path in requested_paths)
    assert result.summary.available is False
    # deadline (0.05s / 0.01s interval) is this poll's only bound -- an ignored
    # deadline still finishes green but blows past both of these
    assert 2 <= len(requested_paths) <= 12
    assert elapsed < 1.0
    # the last payload survives the deadline: real status, not a "not matched" report
    assert result.summary.metadata["status"] == "ongoing"
    assert "reason" not in result.summary.metadata
    # pins LAST not first: each GET grew the transcript, so this only holds for the last one
    assert result.summary.metadata["message_count"] == len(requested_paths)


def test_retell_hint_path_404_on_get_call_is_unavailable_no_raise(tmp_path) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(404, json={"error": "not found"})

    async def run():
        client = httpx.AsyncClient(
            base_url="https://api.retellai.com",
            transport=httpx.MockTransport(handler),
        )
        source = RetellEvidenceSource(
            ProviderEvidenceConfig(
                provider="retell",
                call_id_source="originator_response",
                poll_interval_seconds=0.01,
                poll_deadline_seconds=0.05,
            ),
            api_key="test-key",
            client=client,
        )
        await source.connect(
            EvidenceContext(
                run_id="run_retell",
                test_case_id="case_retell",
                case_directory=tmp_path,
                started_at=datetime.now(timezone.utc),
                call_id_hint="call_retell_404",
            )
        )
        result = await source.fetch_final()
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert requested_paths == ["/v2/get-call/call_retell_404"]
    assert result.summary.available is False
    # hint path has no matching step; a GET failure is "failed", not "no match"
    assert result.summary.metadata.get("reason") == "retell_get_call_failed"


def test_retell_completed_is_not_terminal_and_polls_to_deadline(tmp_path) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        # "completed" is not a Retell call_status; it must not stop the poll early
        return httpx.Response(
            200, json={"call_id": "call_retell_looping", "call_status": "completed"}
        )

    async def run():
        client = httpx.AsyncClient(
            base_url="https://api.retellai.com",
            transport=httpx.MockTransport(handler),
        )
        source = RetellEvidenceSource(
            ProviderEvidenceConfig(
                provider="retell",
                call_id_source="originator_response",
                poll_interval_seconds=0.01,
                poll_deadline_seconds=0.05,
            ),
            api_key="test-key",
            client=client,
        )
        await source.connect(
            EvidenceContext(
                run_id="run_retell",
                test_case_id="case_retell",
                case_directory=tmp_path,
                started_at=datetime.now(timezone.utc),
                call_id_hint="call_retell_looping",
            )
        )
        result = await source.fetch_final()
        await client.aclose()
        return result

    t0 = time.monotonic()
    result = asyncio.run(run())
    elapsed = time.monotonic() - t0
    assert all(path == "/v2/get-call/call_retell_looping" for path in requested_paths)
    assert result.summary.available is False
    # same bound as the deadline test: "completed" never satisfies the terminal check
    assert 2 <= len(requested_paths) <= 12
    assert elapsed < 1.0
