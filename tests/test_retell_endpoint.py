from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
import retell

from fi.simulate.endpoints.retell import RetellCallOriginator
from fi.simulate.endpoints.vapi import VapiCallOriginator

_LOGGER_NAME = "fi.simulate.endpoints.retell"


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _originator(
    handler: Any,
    *,
    from_number: str = "+15550000001",
    destination: str = "+15550000099",
    agent_id: str = "agent_123",
) -> tuple[RetellCallOriginator, httpx.AsyncClient]:
    client = httpx.AsyncClient(
        base_url="https://api.retellai.com",
        transport=httpx.MockTransport(handler),
    )
    originator = RetellCallOriginator(
        api_key="test-key",
        agent_id=agent_id,
        from_number=from_number,
        destination=destination,
        client=client,
    )
    return originator, client


# --- start() -----------------------------------------------------------


def test_retell_originator_posts_create_phone_call() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = request.content
        return httpx.Response(
            201, json={"call_id": "call_123", "call_status": "registered"}
        )

    async def run() -> None:
        originator, client = _originator(handler)
        call = await originator.start()
        await client.aclose()
        assert call.call_id == "call_123"
        assert call.status == "registered"

    _run(run())

    assert captured["path"] == "/v2/create-phone-call"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"] == (
        b'{"from_number":"+15550000001","to_number":"+15550000099",'
        b'"override_agent_id":"agent_123"}'
    )


def test_retell_originator_rejects_missing_response_id() -> None:
    async def run() -> None:
        originator, client = _originator(lambda _: httpx.Response(201, json={}))
        with pytest.raises(ValueError, match="retell_call_response_missing_id"):
            await originator.start()
        await client.aclose()

    _run(run())


@pytest.mark.parametrize(
    ("status", "content", "headers"),
    [
        (200, b"", {"content-type": "text/plain"}),
        (204, b"", {}),
    ],
)
def test_retell_originator_rejects_non_json_2xx_response(
    status: int, content: bytes, headers: dict[str, str]
) -> None:
    # A 2xx without a JSON content-type is passed through by the
    # SDK as raw text (or NoneType for a 204), not the typed
    # PhoneCallResponse model, so `response.call_id` would raise
    # AttributeError — getattr(response, "call_id", None) must keep the
    # contracted ValueError the only failure mode here.
    async def run() -> None:
        originator, client = _originator(
            lambda _: httpx.Response(status, content=content, headers=headers)
        )
        with pytest.raises(ValueError, match="retell_call_response_missing_id"):
            await originator.start()
        await client.aclose()

    _run(run())


def test_retell_originator_start_raises_on_non_2xx() -> None:
    # The retell-sdk client raises its own typed error hierarchy, not
    # httpx.HTTPStatusError, for a non-2xx response.
    async def run() -> None:
        originator, client = _originator(lambda _: httpx.Response(500))
        with pytest.raises(retell.APIStatusError):
            await originator.start()
        await client.aclose()

    _run(run())


# --- stop() --------------------------------------------------------------


@pytest.mark.parametrize("status", [200, 202, 204, 404, 422])
def test_retell_originator_stop_tolerates_expected_statuses(status: int) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["accept"] = request.headers.get("Accept")
        return httpx.Response(status)

    async def run() -> None:
        originator, client = _originator(handler)
        await originator.stop("call_123")
        await client.aclose()

    _run(run())
    assert captured["path"] == "/v2/stop-call/call_123"
    # Mirror the SDK's own NoneType-cast pattern (call.py
    # delete(): Accept: */*) rather than advertising a JSON body this
    # bodyless POST never sends.
    assert captured["accept"] == "*/*"


def test_retell_originator_stop_raises_on_unexpected_status() -> None:
    # A non-tolerated status is re-raised as the SDK's own typed error, not
    # httpx.HTTPStatusError.
    async def run() -> None:
        originator, client = _originator(lambda _: httpx.Response(500))
        with pytest.raises(retell.APIStatusError):
            await originator.stop("call_123")
        await client.aclose()

    _run(run())


def test_retell_originator_never_calls_delete_call() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"calls": []})

    async def run() -> None:
        originator, client = _originator(handler)
        await originator.stop("call_123")
        await originator.reconcile_and_stop(started_after_ms=0, ended_before_ms=1)
        await client.aclose()

    _run(run())
    assert all(r.method != "DELETE" for r in requests)
    assert all("delete-call" not in r.url.path for r in requests)


def test_stop_rejects_path_bearing_call_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    async def run() -> None:
        originator, client = _originator(handler)
        for bad_id in (
            "a/b",
            "a?b",
            "a#b",
            "a%2f",
            "a b",
            "../../v2/delete-call/x",
            "",
            ".",
            "..",
            "a..b",
            ".x",
        ):
            with pytest.raises(ValueError, match="retell_call_id_invalid"):
                await originator.stop(bad_id)
        assert requests == []
        # '.' and ':' are legitimate call-id characters (not path-bearing);
        # pin the boundary as accepted, not merely untested.
        await originator.stop("call_a1b2.c3")
        await client.aclose()

    _run(run())
    assert requests[0].url.path == "/v2/stop-call/call_a1b2.c3"


def test_reconcile_rejects_path_bearing_call_id_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [
        {
            "call_id": "../../v2/delete-call/x",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_700,
            "call_status": "registered",
        }
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_list_calls_handler(rows, capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    assert "stopped_ids" not in capture
    record = next(
        r for r in caplog.records if "retell_reconcile_target_without_id" in r.message
    )
    assert record.has_id is True
    assert record.count == 1


def test_reconcile_logs_target_without_id_when_call_id_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [
        {
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_700,
            "call_status": "registered",
        }
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_list_calls_handler(rows, capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    assert "stopped_ids" not in capture
    record = next(
        r for r in caplog.records if "retell_reconcile_target_without_id" in r.message
    )
    assert record.has_id is False
    assert record.count == 1


def test_reconcile_int_call_id_logs_target_without_id_and_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # `has_id = isinstance(call_id, str)` must short-circuit the
    # `or` in `if not has_id or not _CALL_ID_PATTERN.fullmatch(call_id)`
    # before `fullmatch` ever sees a non-str call_id — `fullmatch(7)` raises
    # TypeError. No prior test drove a non-str, non-missing call_id (only a
    # path-bearing string and a fully-absent field), so this row is
    # otherwise a valid target: correct destination, from_number, window and
    # status.
    rows = [
        {
            "call_id": 7,
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_700,
            "call_status": "registered",
        }
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_bare_list_calls_handler(rows, capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    assert "stopped_ids" not in capture
    record = next(
        r for r in caplog.records if "retell_reconcile_target_without_id" in r.message
    )
    assert record.has_id is False
    assert record.count == 1


# --- from_env() ------------------------------------------------------------


def test_retell_originator_from_env_reports_all_missing_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "RETELL_API_KEY",
        "RETELL_AGENT_ID",
        "RETELL_FROM_NUMBER",
        "LIVEKIT_INBOUND_DID",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError) as excinfo:
        RetellCallOriginator.from_env()

    message = str(excinfo.value)
    assert message.startswith("retell_originator_config_missing: ")
    missing = message.split(": ", 1)[1].split(", ")
    assert missing == sorted(missing)
    assert missing == [
        "LIVEKIT_INBOUND_DID",
        "RETELL_AGENT_ID",
        "RETELL_API_KEY",
        "RETELL_FROM_NUMBER",
    ]


def test_retell_originator_from_env_transport_beats_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETELL_API_KEY", "env-key")
    monkeypatch.setenv("RETELL_AGENT_ID", "env-agent")
    monkeypatch.setenv("RETELL_FROM_NUMBER", "+15550000002")
    monkeypatch.setenv("LIVEKIT_INBOUND_DID", "+15550000099")

    class _Transport:
        originator_agent_id = "transport-agent"
        originator_from_number = "+15550000001"

    originator = RetellCallOriginator.from_env(_Transport())
    assert originator._agent_id == "transport-agent"
    assert originator._from_number == "+15550000001"
    assert originator._destination == "+15550000099"


def test_retell_originator_from_env_falls_back_when_transport_lacks_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETELL_API_KEY", "env-key")
    monkeypatch.setenv("RETELL_AGENT_ID", "env-agent")
    monkeypatch.setenv("RETELL_FROM_NUMBER", "+15550000002")
    monkeypatch.setenv("LIVEKIT_INBOUND_DID", "+15550000099")

    class _EmptyTransport:
        pass

    originator = RetellCallOriginator.from_env(_EmptyTransport())
    assert originator._agent_id == "env-agent"
    assert originator._from_number == "+15550000002"

    originator_no_transport = RetellCallOriginator.from_env(None)
    assert originator_no_transport._agent_id == "env-agent"


# --- reconcile_and_stop(): blast-radius fencing ----------------------------


def _list_calls_handler(rows: list[dict[str, Any]], capture: dict[str, Any]) -> Any:
    # {"items": rows} — the real v3 CallListResponse envelope shape
    # (retell/types/call_list_response.py, PLAN D12a): a paginated object
    # with items/has_more/pagination_key/total, not a bare array or the
    # older {"calls": [...]} dict. See _bare_list_calls_handler and
    # _calls_key_calls_handler below for the two legacy shapes this
    # kit still tolerates.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/list-calls":
            capture["list_body"] = request.content
            capture["list_timeout"] = request.extensions.get("timeout")
            return httpx.Response(200, json={"items": rows})
        if request.url.path.startswith("/v2/stop-call/"):
            capture.setdefault("stopped_ids", []).append(
                request.url.path.rsplit("/", 1)[-1]
            )
            capture["stop_timeout"] = request.extensions.get("timeout")
            return httpx.Response(200)
        raise AssertionError(f"unexpected path {request.url.path}")

    return handler


def _bare_list_calls_handler(
    rows: list[dict[str, Any]], capture: dict[str, Any]
) -> Any:
    # A bare JSON array — not what the real v3 API sends (see
    # _list_calls_handler above), but the installed 5.64 SDK still
    # constructs one CallListResponse per top-level array element
    # (extra="allow" absorption, probed), and reconcile_and_stop keeps a
    # legacy isinstance(response, list) branch for it. Exercised by the
    # fence-behaviour tests below purely as a convenient row-carrying
    # transport, and pinned directly by
    # test_reconcile_stops_target_row_legacy_bare_list.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/list-calls":
            capture["list_body"] = request.content
            capture["list_timeout"] = request.extensions.get("timeout")
            return httpx.Response(200, json=rows)
        if request.url.path.startswith("/v2/stop-call/"):
            capture.setdefault("stopped_ids", []).append(
                request.url.path.rsplit("/", 1)[-1]
            )
            capture["stop_timeout"] = request.extensions.get("timeout")
            return httpx.Response(200)
        raise AssertionError(f"unexpected path {request.url.path}")

    return handler


def _calls_key_calls_handler(
    rows: list[dict[str, Any]], capture: dict[str, Any]
) -> Any:
    # {"calls": rows} — a pre-D12a legacy dict shape. The installed 5.64
    # SDK folds this into a CallListResponse with items=None and an extra
    # "calls" field (probed); reconcile_and_stop's model_dump fallback
    # branch hands it back to _select_call_rows, which still recognises
    # the "calls" key. Kept only for
    # test_reconcile_stops_target_row_legacy_calls_key.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/list-calls":
            capture["list_body"] = request.content
            capture["list_timeout"] = request.extensions.get("timeout")
            return httpx.Response(200, json={"calls": rows})
        if request.url.path.startswith("/v2/stop-call/"):
            capture.setdefault("stopped_ids", []).append(
                request.url.path.rsplit("/", 1)[-1]
            )
            capture["stop_timeout"] = request.extensions.get("timeout")
            return httpx.Response(200)
        raise AssertionError(f"unexpected path {request.url.path}")

    return handler


def test_reconcile_sends_v3_typed_filter_criteria_and_limit_50() -> None:
    # PLAN D12 (supersedes D10a): retell-sdk 5.64+'s AsyncCallResource.list
    # posts to /v3/list-calls with the v3 {type, op, value} FilterCriteria
    # vocabulary (retell/types/call_list_params.py) — from_number/to_number
    # are {"type": "string", "op": "eq", "value": <str>} filters,
    # start_timestamp is a {"type": "range", "op": "bt", "value": [lower_ms,
    # upper_ms]} filter. Confirmed on the wire via MockTransport probe
    # against the installed SDK.
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_list_calls_handler([], capture))
        result = await originator.reconcile_and_stop(
            started_after_ms=1_000, ended_before_ms=2_000
        )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    import json

    body = json.loads(capture["list_body"])
    assert body["limit"] == 50
    assert set(body["filter_criteria"].keys()) == {
        "from_number",
        "to_number",
        "start_timestamp",
    }
    assert body["filter_criteria"]["from_number"] == {
        "type": "string",
        "op": "eq",
        "value": "+15550000001",
    }
    assert body["filter_criteria"]["to_number"] == {
        "type": "string",
        "op": "eq",
        "value": "+15550000099",
    }
    assert body["filter_criteria"]["start_timestamp"] == {
        "type": "range",
        "op": "bt",
        "value": [1_000, 2_000],
    }


def test_installed_sdk_list_calls_posts_to_v3() -> None:
    """Canary for PLAN D12: retell-sdk 5.64+'s AsyncCallResource.list posts
    to /v3/list-calls with the v3 {type, op, value} FilterCriteria
    vocabulary, not the /v2 list-valued shape earlier SDK versions used. The
    reconcile body above (and this test) is hand-typed to match that
    vocabulary rather than importing the SDK's own request-building code, so
    nothing here would fail if the SDK moved the endpoint or vocabulary
    again — only this canary would. If it starts failing, treat it as a
    signal to re-verify test_reconcile_sends_v3_typed_filter_criteria_and_
    limit_50 and the reconcile_and_stop request body against the newly
    installed SDK's source (and, if needed, against the wire via
    MockTransport) before bumping the pyproject.toml pin.
    """
    import inspect
    import typing

    source = inspect.getsource(retell.resources.call.AsyncCallResource.list)
    assert '"/v3/list-calls"' in source

    hints = typing.get_type_hints(
        retell.types.call_list_params.FilterCriteriaFromNumber, include_extras=True
    )
    # Required[str], not a sequence (the v2 shape's from_number was
    # list-valued) — get_type_hints resolves the module's `from __future__
    # import annotations` string annotations back to real typing objects.
    assert typing.get_args(hints["value"]) == (str,)


def test_reconcile_stops_only_exact_destination_in_window_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Uses the bare-list handler purely as a convenient row-carrying
    # transport for this fence-behaviour check; the real v3 API returns
    # the {"items": [...]} envelope (see _list_calls_handler /
    # test_reconcile_unwraps_v3_items_envelope for the shape itself).
    rows = [
        # Newest row of all: if the to_number equality check were ever
        # removed, latest-wins would pick this row instead of "ours".
        {
            "call_id": "wrong_destination",
            "to_number": "+19990000000",
            "from_number": "+15550000001",
            "start_timestamp": 1_900,
            "call_status": "registered",
        },
        {
            "call_id": "no_destination_field",
            "from_number": "+15550000001",
            "start_timestamp": 1_600,
            "call_status": "registered",
        },
        {
            "call_id": "null_start_timestamp",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": None,
            "call_status": "registered",
        },
        {
            "call_id": "ours",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_700,
            "call_status": "registered",
        },
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_bare_list_calls_handler(rows, capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == ["ours"]
    assert capture.get("stopped_ids") == ["ours"]
    assert any(
        "retell_reconcile_no_start_timestamp" in r.message for r in caplog.records
    )


def test_reconcile_never_stops_out_of_window_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Uses the bare-list handler purely as a convenient row-carrying
    # transport for this fence-behaviour check; the real v3 API returns
    # the {"items": [...]} envelope (see _list_calls_handler /
    # test_reconcile_unwraps_v3_items_envelope for the shape itself).
    rows = [
        {
            "call_id": "in_window_ours",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_700,
            "call_status": "registered",
        },
        # Newest row of all, our DID, and stoppable status — if the
        # client-side window recheck were ever removed, latest-wins would
        # pick this row over the in-window one.
        {
            "call_id": "out_of_window_ours",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 999_999_999,
            "call_status": "ongoing",
        },
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_bare_list_calls_handler(rows, capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == ["in_window_ours"]
    assert capture.get("stopped_ids") == ["in_window_ours"]
    assert any("retell_reconcile_out_of_window" in r.message for r in caplog.records)


def test_reconcile_drops_from_number_mismatch_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The server-side from_number filter is unproven; an
    # in-window, correctly-destined, stoppable-status row must still not be
    # stopped if its from_number isn't ours. Sweep several mismatched
    # formats the same way the destination-mismatch and
    # ambiguous-row siblings do (tests below), so a future edit that adds
    # the raw from_number values to this log's `extra=` can't slip past
    # silently.
    rows = [
        {
            "call_id": "wrong_from_number",
            "to_number": "+15550000099",
            "from_number": "+19990000000",
            "start_timestamp": 1_700,
            "call_status": "ongoing",
        },
        {
            "call_id": "spaced_from_number",
            "to_number": "+15550000099",
            "from_number": "+1 555 000 0001",
            "start_timestamp": 1_710,
            "call_status": "ongoing",
        },
        {
            "call_id": "no_plus_from_number",
            "to_number": "+15550000099",
            "from_number": "15550000001",
            "start_timestamp": 1_720,
            "call_status": "ongoing",
        },
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_bare_list_calls_handler(rows, capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    assert "stopped_ids" not in capture
    record = next(
        r
        for r in caplog.records
        if "retell_reconcile_from_number_mismatch" in r.message
    )
    assert record.dropped == 3
    # The raw from_number values — including formatting variants that are
    # *not* exact matches (spaces, missing "+") — must never leave the
    # process in this record, under any attribute name.
    assert not hasattr(record, "observed_from_numbers")
    row_numbers = {"+19990000000", "+1 555 000 0001", "15550000001"}
    for value in vars(record).values():
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            assert item not in row_numbers


def test_reconcile_status_fence_only_registered_or_ongoing() -> None:
    # Uses the bare-list handler purely as a convenient row-carrying
    # transport for this fence-behaviour check; the real v3 API returns
    # the {"items": [...]} envelope (see _list_calls_handler /
    # test_reconcile_unwraps_v3_items_envelope for the shape itself).
    rows = [
        # Newest row of all, but "ended" is not stoppable.
        {
            "call_id": "ended_ours",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_900,
            "call_status": "ended",
        },
        {
            "call_id": "ongoing_upper",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_700,
            "call_status": "ONGOING",
        },
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_bare_list_calls_handler(rows, capture))
        result = await originator.reconcile_and_stop(
            started_after_ms=1_000, ended_before_ms=2_000
        )
        await client.aclose()
        return result

    result = _run(run())
    assert result == ["ongoing_upper"]
    assert capture.get("stopped_ids") == ["ongoing_upper"]


def test_reconcile_two_matching_rows_stops_newer_and_logs_ambiguous(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Uses the bare-list handler purely as a convenient row-carrying
    # transport for this fence-behaviour check; the real v3 API returns
    # the {"items": [...]} envelope (see _list_calls_handler /
    # test_reconcile_unwraps_v3_items_envelope for the shape itself).
    rows = [
        {
            "call_id": "older",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_500,
            "call_status": "ongoing",
        },
        {
            "call_id": "newer",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_800,
            "call_status": "registered",
        },
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_bare_list_calls_handler(rows, capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == ["newer"]
    assert capture.get("stopped_ids") == ["newer"]
    record = next(
        r for r in caplog.records if "retell_reconcile_ambiguous" in r.message
    )
    assert record.left_alone == 1
    assert record.stopped_call_id == "newer"
    # The row we left alone is a third-party id and must never leave the
    # process in this record, under any attribute name.
    for value in vars(record).values():
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            assert item != "older"


def test_reconcile_no_destination_field_at_all_stops_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [
        {
            "call_id": "a",
            "from_number": "+15550000001",
            "start_timestamp": 1_500,
            "call_status": "registered",
        },
        {
            "call_id": "b",
            "from_number": "+15550000001",
            "start_timestamp": 1_600,
            "call_status": "ongoing",
        },
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_list_calls_handler(rows, capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    assert "stopped_ids" not in capture
    assert any(
        "retell_reconcile_no_destination_field" in r.message for r in caplog.records
    )


def test_reconcile_destination_mismatch_logs_count_not_raw_numbers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [
        {
            "call_id": "a",
            "to_number": "+19990000001",
            "from_number": "+15550000001",
            "start_timestamp": 1_500,
            "call_status": "registered",
        },
        {
            "call_id": "b",
            "to_number": "+19990000002",
            "from_number": "+15550000001",
            "start_timestamp": 1_600,
            "call_status": "ongoing",
        },
        # Non-scalar to_number values must never reach the set comprehension
        # used to count distinct values, and must never raise; they are
        # treated as "no usable destination" like a missing field.
        {
            "call_id": "c",
            "to_number": ["+19990000003"],
            "from_number": "+15550000001",
            "start_timestamp": 1_620,
            "call_status": "ongoing",
        },
        {
            "call_id": "d",
            "to_number": {},
            "from_number": "+15550000001",
            "start_timestamp": 1_640,
            "call_status": "ongoing",
        },
        # A duplicate of row "a"'s destination: makes count (3) and
        # distinct_observed (2) diverge, so distinct_observed can't be
        # confused with a plain row count.
        {
            "call_id": "e",
            "to_number": "+19990000001",
            "from_number": "+15550000001",
            "start_timestamp": 1_650,
            "call_status": "registered",
        },
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_list_calls_handler(rows, capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    assert "stopped_ids" not in capture
    record = next(
        r
        for r in caplog.records
        if "retell_reconcile_destination_mismatch" in r.message
    )
    # Our own leased DID may appear (it's ours, not the customer's data).
    assert record.destination == "+15550000099"
    # The raw callee numbers must never leave the process in this record.
    assert not hasattr(record, "observed_to_numbers")
    assert record.distinct_observed == 2
    assert record.count == 3
    row_numbers = {"+19990000001", "+19990000002", "+19990000003"}
    for value in vars(record).values():
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            assert item not in row_numbers


def test_reconcile_zero_rows_logs_no_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_list_calls_handler([], capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    assert any("retell_reconcile_no_candidates" in r.message for r in caplog.records)


def test_reconcile_stops_target_row_legacy_bare_list() -> None:
    # Pins the legacy bare-JSON-array shape (see _bare_list_calls_handler):
    # not what the real v3 API sends, but a pre-5.64/compat path the
    # installed SDK still constructs a usable response from, so
    # reconcile_and_stop must keep stopping a valid target through it.
    rows = [
        {
            "call_id": "ours",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_500,
            "call_status": "registered",
        }
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_bare_list_calls_handler(rows, capture))
        result = await originator.reconcile_and_stop(
            started_after_ms=1_000, ended_before_ms=2_000
        )
        await client.aclose()
        return result

    result = _run(run())
    assert result == ["ours"]
    assert capture.get("stopped_ids") == ["ours"]


def test_reconcile_stops_target_row_legacy_calls_key() -> None:
    # Pins the legacy {"calls": [...]} dict shape (see
    # _calls_key_calls_handler): the installed 5.64 SDK folds it into a
    # CallListResponse with items=None and an extra "calls" field;
    # reconcile_and_stop's model_dump fallback branch must still hand it to
    # _select_call_rows and find the target.
    rows = [
        {
            "call_id": "ours",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_500,
            "call_status": "registered",
        }
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_calls_key_calls_handler(rows, capture))
        result = await originator.reconcile_and_stop(
            started_after_ms=1_000, ended_before_ms=2_000
        )
        await client.aclose()
        return result

    result = _run(run())
    assert result == ["ours"]
    assert capture.get("stopped_ids") == ["ours"]


def test_reconcile_unwraps_v3_items_envelope() -> None:
    # PLAN D12a: the real v3 CallListResponse envelope — items plus the
    # pagination metadata fields — must unwrap to the target row and issue
    # a stop, not just a bare {"items": rows} shortcut.
    capture: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/list-calls":
            return httpx.Response(
                200,
                json={
                    "items": [_VALID_TARGET_ROW],
                    "has_more": False,
                    "pagination_key": None,
                    "total": 1,
                },
            )
        if request.url.path.startswith("/v2/stop-call/"):
            capture.setdefault("stopped_ids", []).append(
                request.url.path.rsplit("/", 1)[-1]
            )
            return httpx.Response(200)
        raise AssertionError(f"unexpected path {request.url.path}")

    async def run() -> list[str]:
        originator, client = _originator(handler)
        result = await originator.reconcile_and_stop(
            started_after_ms=1_000, ended_before_ms=2_000
        )
        await client.aclose()
        return result

    result = _run(run())
    assert result == ["ours"]
    assert capture.get("stopped_ids") == ["ours"]


def test_reconcile_envelope_without_items_is_unexpected_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An envelope-shaped object with no `items` key at all (not a list, not
    # missing-but-dict — a genuine CallListResponse with items=None) must
    # be treated as an unrecognised payload, not silently read as zero
    # candidates.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/list-calls":
            return httpx.Response(200, json={"has_more": False})
        raise AssertionError(f"unexpected path {request.url.path}")

    async def run() -> list[str]:
        originator, client = _originator(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    record = next(
        r for r in caplog.records if "retell_reconcile_unexpected_payload" in r.message
    )
    assert record.payload_type == "CallListResponse"


_VALID_TARGET_ROW: dict[str, Any] = {
    "call_id": "ours",
    "to_number": "+15550000099",
    "from_number": "+15550000001",
    "start_timestamp": 1_700,
    "call_status": "registered",
}


@pytest.mark.parametrize(
    ("rows", "expected_skipped", "expected_result"),
    [
        ([_VALID_TARGET_ROW, "junk"], 1, ["ours"]),
        ([None], 1, []),
        ([1, 2], 2, []),
    ],
)
def test_reconcile_skips_non_object_bare_list_rows(
    rows: list[Any],
    expected_skipped: int,
    expected_result: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A malformed 200 JSON array can contain a non-object element
    # (str/int/None/nested list) — the SDK's construct_type leaves it as the
    # raw value rather than a typed row, so it has no model_dump. That must
    # be skipped and counted, not let AttributeError escape past the
    # swallow with no log at all — and a valid target elsewhere on the same
    # page must still be found and stopped.
    capture: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/list-calls":
            return httpx.Response(200, json=rows)
        if request.url.path.startswith("/v2/stop-call/"):
            capture.setdefault("stopped_ids", []).append(
                request.url.path.rsplit("/", 1)[-1]
            )
            return httpx.Response(200)
        raise AssertionError(f"unexpected path {request.url.path}")

    async def run() -> list[str]:
        originator, client = _originator(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == expected_result
    assert capture.get("stopped_ids", []) == expected_result
    record = next(
        r for r in caplog.records if "retell_reconcile_unexpected_row" in r.message
    )
    assert record.skipped == expected_skipped


def test_bare_list_row_keeps_to_number_after_model_dump() -> None:
    # CallResponse is an undiscriminated Union
    # (Union[WebCallResponse, PhoneCallResponse]) and construct_type always
    # takes the first variant — probed, a phone-call row always constructs
    # as WebCallResponse, which declares neither to_number nor from_number.
    # The fences survive only because pydantic's extra="allow"
    # (retell/_models.py) carries unknown fields through model_dump. Pin
    # that directly against the installed SDK so a bump that tightens
    # `extra` or adds a real discriminator can't silently blind the fences.
    row = {
        "call_id": "call_abc",
        "call_type": "phone_call",
        "to_number": "+15550000099",
        "from_number": "+15550000001",
        "start_timestamp": 1_500,
        "call_status": "registered",
        "future_unknown_field": "KEEPME",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[row])

    async def run() -> dict[str, Any]:
        client = httpx.AsyncClient(
            base_url="https://api.retellai.com",
            transport=httpx.MockTransport(handler),
        )
        sdk_client = retell.AsyncRetell(
            api_key="test-key", http_client=client, max_retries=0
        )
        response = await sdk_client.call.list(filter_criteria={}, limit=1)
        dumped = response[0].model_dump(mode="json")
        await client.aclose()
        return dumped

    dumped = _run(run())
    assert dumped["to_number"] == "+15550000099"
    assert dumped["from_number"] == "+15550000001"
    assert dumped["start_timestamp"] == 1_500
    assert dumped["call_status"] == "registered"
    assert dumped["call_id"] == "call_abc"
    assert dumped["future_unknown_field"] == "KEEPME"


@pytest.mark.parametrize(
    ("body", "expected_type"),
    [(b'"not-a-call-list"', "str"), (b"null", "NoneType")],
)
def test_reconcile_logs_unexpected_payload_for_unrecognised_shape(
    body: bytes, expected_type: str, caplog: pytest.LogCaptureFixture
) -> None:
    # An explicit JSON content-type mirrors what the real API always sends;
    # the SDK only attempts to decode the body as JSON when it sees one.
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "application/json"}
        )

    async def run() -> list[str]:
        originator, client = _originator(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    record = next(
        r for r in caplog.records if "retell_reconcile_unexpected_payload" in r.message
    )
    assert record.payload_type == expected_type
    assert not any(
        "retell_reconcile_no_candidates" in r.message for r in caplog.records
    )


def test_reconcile_swallows_non_json_response_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An explicit JSON content-type is what makes the SDK attempt
    # to parse this malformed body as JSON at all, raising json.JSONDecodeError
    # (a ValueError) — this is the ValueError arm of the swallow, distinct
    # from the transport-error net (test_reconcile_swallows_transport_error)
    # and from the no-content-type case (test_reconcile_text_html_body_is_
    # unexpected_payload below), which doesn't raise at all.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>gateway</html>",
            headers={"content-type": "application/json"},
        )

    async def run() -> list[str]:
        originator, client = _originator(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    record = next(r for r in caplog.records if "retell_call_reconcile" in r.message)
    assert record.phase == "list_calls"
    assert record.error == "JSONDecodeError"


def test_reconcile_text_html_body_is_unexpected_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Without a JSON content-type the SDK does not attempt to
    # parse the body at all and hands back the raw text — the exact
    # proxy/WAF-page case the ValueError arm above was written for no
    # longer raises; it now surfaces as retell_reconcile_unexpected_payload
    # (payload_type "str") rather than the swallowed-fetch-failure code.
    # PLAN §8's operator decision table treats the two codes as opposite
    # signals, so pin which one actually fires here.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>gateway</html>")

    async def run() -> list[str]:
        originator, client = _originator(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    record = next(
        r for r in caplog.records if "retell_reconcile_unexpected_payload" in r.message
    )
    assert record.payload_type == "str"


def test_reconcile_full_page_logs_page_full(caplog: pytest.LogCaptureFixture) -> None:
    rows = [
        {
            "call_id": f"call_{i}",
            "to_number": "+19990000000",
            "start_timestamp": 1_000 + i,
            "call_status": "ended",
        }
        for i in range(50)
    ]
    capture: dict[str, Any] = {}

    async def run() -> list[str]:
        originator, client = _originator(_list_calls_handler(rows, capture))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    assert any("retell_reconcile_page_full" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "bad_value",
    [
        datetime.now(timezone.utc),
        1000.0,  # a float json.dumps CAN serialize; must still be rejected
        "1000",  # a numeric string json.dumps CAN serialize
        True,  # bool is an int subclass; the guard must exclude it
        None,
    ],
)
def test_reconcile_rejects_non_int_time_args_before_any_request(bad_value: Any) -> None:
    requested = {"called": False}

    def handler(request: httpx.Request) -> httpx.Response:
        requested["called"] = True
        return httpx.Response(200, json={"calls": []})

    async def run() -> None:
        originator, client = _originator(handler)
        with pytest.raises(TypeError, match="requires int epoch ms"):
            await originator.reconcile_and_stop(
                started_after_ms=bad_value,  # type: ignore[arg-type]
                ended_before_ms=2_000,
            )
        await client.aclose()

    _run(run())
    assert requested["called"] is False


def test_reconcile_swallows_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async def run() -> list[str]:
        originator, client = _originator(handler)
        result = await originator.reconcile_and_stop(
            started_after_ms=1_000, ended_before_ms=2_000
        )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []


def test_reconcile_swallows_transport_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # retell.APIConnectionError is NOT an httpx.HTTPError subclass
    # (the SDK wraps every transport failure in its own hierarchy) — this is
    # the one test that drives an actual transport error through
    # reconcile_and_stop, rather than a non-2xx status
    # (test_reconcile_swallows_http_errors above tests the latter).
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    async def run() -> list[str]:
        originator, client = _originator(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    record = next(r for r in caplog.records if "retell_call_reconcile" in r.message)
    assert record.phase == "list_calls"
    assert record.error == "APIConnectionError"


def test_reconcile_uses_10s_per_request_timeout() -> None:
    rows = [
        {
            "call_id": "ours",
            "to_number": "+15550000099",
            "from_number": "+15550000001",
            "start_timestamp": 1_500,
            "call_status": "registered",
        }
    ]
    capture: dict[str, Any] = {}

    async def run() -> None:
        originator, client = _originator(_list_calls_handler(rows, capture))
        await originator.reconcile_and_stop(
            started_after_ms=1_000, ended_before_ms=2_000
        )
        await client.aclose()

    _run(run())
    for key in ("list_timeout", "stop_timeout"):
        timeout = capture[key]
        assert timeout["connect"] == 10.0
        assert timeout["read"] == 10.0


def test_retell_api_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETELL_API_BASE_URL", "https://custom.retell.example")
    originator = RetellCallOriginator(
        api_key="k",
        agent_id="a",
        from_number="+15550000001",
        destination="+15550000099",
    )
    assert str(originator._client.base_url) == "https://custom.retell.example"
    _run(originator.close())


def test_close_only_closes_owned_client() -> None:
    async def run() -> None:
        client = httpx.AsyncClient(
            base_url="https://api.retellai.com",
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )
        originator = RetellCallOriginator(
            api_key="k",
            agent_id="a",
            from_number="+15550000001",
            destination="+15550000099",
            client=client,
        )
        await originator.close()
        assert client.is_closed is False
        await client.aclose()

    _run(run())


def test_close_closes_client_it_created(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        originator = RetellCallOriginator(
            api_key="k",
            agent_id="a",
            from_number="+15550000001",
            destination="+15550000099",
        )
        await originator.close()
        assert originator._http.is_closed is True

    _run(run())


def test_client_timeout_is_30s_total_10s_connect() -> None:
    # The default client (used by start()/stop()
    # outside the reconcile path, which passes its own explicit 10 s
    # timeout) must be constructed with 30 s total, 10 s connect — assert
    # on the constructed httpx.Timeout directly rather than inferring it
    # from wire behaviour.
    originator = RetellCallOriginator(
        api_key="k",
        agent_id="a",
        from_number="+15550000001",
        destination="+15550000099",
    )
    assert originator._client.timeout == httpx.Timeout(30.0, connect=10.0)
    assert originator._client.max_retries == 0
    _run(originator.close())


def test_retell_client_constructed_with_max_retries_zero() -> None:
    # The SDK's own retry loop must never layer hidden retries on top of our
    # own tolerance/retry contract (the tolerated-stop-status set, the
    # reconcile guard's swallow-and-log).
    request_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        request_count["n"] += 1
        return httpx.Response(500)

    async def run() -> list[str]:
        originator, client = _originator(handler)
        assert originator._client.max_retries == 0
        result = await originator.reconcile_and_stop(
            started_after_ms=1_000, ended_before_ms=2_000
        )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    assert request_count["n"] == 1


# --- lazy `retell` import (P10-2 hardening) --------------------------------


def test_retell_call_originator_requires_retell_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # retell-sdk is optional at the package level (see the module's
    # try/except ImportError); only RetellCallOriginator — the class that
    # actually talks to the SDK — must fail loudly, and only at
    # construction time, when it's unavailable.
    import fi.simulate.endpoints.retell as retell_module

    monkeypatch.setattr(retell_module, "retell", None)
    with pytest.raises(RuntimeError, match="retell_sdk_missing"):
        RetellCallOriginator(
            api_key="k",
            agent_id="a",
            from_number="+15550000001",
            destination="+15550000099",
        )


def test_import_fi_simulate_endpoints_succeeds_without_retell_sdk() -> None:
    # A chat/Vapi-only environment may not have retell-sdk installed at
    # all; importing fi.simulate.endpoints (and this module within it)
    # must not raise. Run in a subprocess — not this test process — since
    # `retell` is already imported and cached in sys.modules here.
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['retell'] = None; "
            "import fi.simulate.endpoints; "
            "import fi.simulate.endpoints.retell as m; "
            "assert m.retell is None; "
            "assert m.AsyncRetell is None; "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# --- Vapi's no-op reconcile + from_env -------------------------------------


def test_vapi_reconcile_and_stop_is_a_noop() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    async def run() -> list[str]:
        client = httpx.AsyncClient(
            base_url="https://api.vapi.ai", transport=httpx.MockTransport(handler)
        )
        originator = VapiCallOriginator(
            api_key="k",
            assistant_id="a",
            phone_number_id="p",
            destination="+15550000099",
            client=client,
        )
        result = await originator.reconcile_and_stop(
            started_after_ms=1_000, ended_before_ms=2_000
        )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    assert calls == []


def test_vapi_originator_from_env_reports_all_missing_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "VAPI_API_KEY",
        "VAPI_ASSISTANT_ID",
        "VAPI_PHONE_NUMBER_ID",
        "LIVEKIT_INBOUND_DID",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError) as excinfo:
        VapiCallOriginator.from_env()

    message = str(excinfo.value)
    assert message.startswith("vapi_originator_config_missing: ")
    missing = message.split(": ", 1)[1].split(", ")
    assert missing == sorted(missing)
    assert missing == [
        "LIVEKIT_INBOUND_DID",
        "VAPI_API_KEY",
        "VAPI_ASSISTANT_ID",
        "VAPI_PHONE_NUMBER_ID",
    ]


def test_vapi_originator_from_env_builds_originator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VAPI_API_KEY", "k")
    monkeypatch.setenv("VAPI_ASSISTANT_ID", "a")
    monkeypatch.setenv("VAPI_PHONE_NUMBER_ID", "p")
    monkeypatch.setenv("LIVEKIT_INBOUND_DID", "+15550000099")

    originator = VapiCallOriginator.from_env()
    assert originator._assistant_id == "a"
    assert originator._phone_number_id == "p"
    assert originator._destination == "+15550000099"


def test_reconcile_items_envelope_skips_non_object_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The production shape: a junk element INSIDE the v3 ``items`` envelope.
    # Pins the explicit ``.items`` unwrap branch — without it the envelope
    # still resolves through the legacy dict-key fallback, but the per-row
    # skip count (an operator-facing log code) is silently lost.
    capture: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/list-calls":
            return httpx.Response(
                200,
                json={"items": [_VALID_TARGET_ROW, "junk"], "has_more": False},
            )
        if request.url.path.startswith("/v2/stop-call/"):
            capture.setdefault("stopped_ids", []).append(
                request.url.path.rsplit("/", 1)[-1]
            )
            return httpx.Response(200)
        raise AssertionError(f"unexpected path {request.url.path}")

    async def run() -> list[str]:
        originator, client = _originator(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == ["ours"]
    assert capture.get("stopped_ids", []) == ["ours"]
    record = next(
        r for r in caplog.records if "retell_reconcile_unexpected_row" in r.message
    )
    assert record.skipped == 1


def test_reconcile_truncated_empty_page_logs_page_full_not_no_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # ``has_more`` is the envelope's explicit truncation flag. An empty page
    # that the server says is truncated must report page_full (guard may
    # have missed its own call), never no_candidates (the "guard worked,
    # page genuinely empty" signal the operator decision table keys on).
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/list-calls":
            return httpx.Response(200, json={"items": [], "has_more": True})
        raise AssertionError(f"unexpected path {request.url.path}")

    async def run() -> list[str]:
        originator, client = _originator(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await originator.reconcile_and_stop(
                started_after_ms=1_000, ended_before_ms=2_000
            )
        await client.aclose()
        return result

    result = _run(run())
    assert result == []
    messages = [r.message for r in caplog.records]
    assert any("retell_reconcile_page_full" in m for m in messages)
    assert not any("retell_reconcile_no_candidates" in m for m in messages)
    record = next(
        r for r in caplog.records if "retell_reconcile_page_full" in r.message
    )
    assert record.has_more is True
    assert record.row_count == 0
