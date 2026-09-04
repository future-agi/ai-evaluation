"""Retell target-agent adapter — capability declaration + Stage 6/8 seam,
plus ``RetellCallOriginator``, the outbound call originator the LiveKit
engine uses to place and clean up Retell phone calls.

Retell has no PSTN outbound API; ``VapiAgentEndpoint`` supports both
directions but this adapter refuses ``sip_outbound`` explicitly to
match the guard in ``AgentDefinition``.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import httpx

try:
    import retell
    from retell import AsyncRetell, NoneType
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    # retell-sdk is an optional dependency: an environment running only
    # chat/Vapi jobs must still be able to import this module (and the rest
    # of fi.simulate.endpoints) without it installed. Only
    # RetellCallOriginator (the class that actually talks to the SDK) fails
    # loudly, in its own __init__ below — RetellAgentEndpoint/RetellCall
    # stay httpx-only and never touch these names.
    retell = None  # type: ignore[assignment]
    AsyncRetell = None  # type: ignore[assignment,misc]
    NoneType = type(None)

from fi.simulate.realtime.events import RealtimeEvent
from fi.simulate.realtime.media import AudioFrame
from fi.simulate.runtime.capabilities import EndpointCapabilities

from .base import (
    AgentEndpointManifest,
    DiscoveryRequest,
    DiscoverySnapshot,
    EndpointHandle,
    ReadinessResult,
    ReconciliationResult,
)

logger = logging.getLogger(__name__)


def _select_call_rows(payload: Any) -> tuple[list[dict[str, Any]], bool]:
    # Retell's list-calls response shape isn't pinned in docs; accept a bare
    # list or a dict wrapping one under any of the observed key names. The
    # bool distinguishes "recognised shape, zero rows" from "a shape this
    # parser doesn't understand", so callers can tell an empty page apart
    # from a payload they failed to read.
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], True
    if isinstance(payload, dict):
        for key in ("calls", "results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)], True
    # Falls through here for a str (or any other) payload too: the SDK only
    # raises when a JSON content-type response fails to parse as JSON at
    # all, so a plain string body — including a proxy/WAF page served
    # without a JSON content-type — reaches this function as `str` rather
    # than the ValueError arm in reconcile_and_stop. recognised=False is
    # still the right signal; it's the caller's job to remember
    # that "unexpected payload" can mean either an unknown API shape or a
    # gateway page, not only the former.
    return [], False


def _dump_rows(raw_rows: Any) -> tuple[list[dict[str, Any]], int]:
    # Convert an iterable of SDK row models to plain dicts so the existing
    # dict-based row logic in reconcile_and_stop is untouched regardless of
    # which response shape it came from (a bare-list body or a 5.64
    # envelope's .items list). Returns (dumped_rows, skipped_count).
    dumped: list[dict[str, Any]] = []
    skipped = 0
    # model_dump emits a pydantic UserWarning
    # (PydanticSerializationUnexpectedValue) per row with a malformed field
    # (e.g. a non-int start_timestamp); the value still comes through
    # unchanged and is then type-checked by the fences in
    # reconcile_and_stop, so this is noise, not a correctness signal. Scope
    # the suppression tightly to this call only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row in raw_rows:
            # A non-object array element (e.g. a malformed 200 JSON array
            # containing a bare string/int/None) has no model_dump — the
            # SDK's construct_type leaves it as the raw value instead of a
            # typed row. Skip and count it rather than let AttributeError
            # escape the swallow below, past every fence, with no log at
            # all.
            if not hasattr(row, "model_dump"):
                skipped += 1
                continue
            dumped.append(row.model_dump(mode="json"))
    return dumped, skipped


# A call id reaches an f-string URL untouched; restrict it to a plain token
# so it can never redirect a request at delete-call or any other path.
# '.' and ':' are included because they show up in real id formats and
# cannot create a traversal on their own; requiring an alphanumeric start
# and forbidding consecutive dots keeps a bare "." or ".." from matching.
_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:-]*(?:\.[A-Za-z0-9_:-]+)*$")


def _validate_call_id(call_id: str) -> None:
    if not isinstance(call_id, str) or not _CALL_ID_PATTERN.fullmatch(call_id):
        raise ValueError("retell_call_id_invalid")


def _strip_or_none(value: Any) -> Any:
    return value.strip() or None if isinstance(value, str) else value


@dataclass(frozen=True)
class RetellCall:
    call_id: str
    status: str | None


class RetellCallOriginator:
    """Create an opt-in Retell call to the LiveKit inbound DID.

    ``reconcile_and_stop`` exists because ``asyncio.wait_for`` cancels our
    coroutine, not Retell's server-side dial — a slow response can leave a
    live, billed call with no id in hand. Because ``from_number`` is the
    customer's production Retell number, the guard is fenced hard: proven
    filter vocabulary only, a client-side window and exact-destination
    match, and at most one stop.
    """

    _base_url = "https://api.retellai.com"
    _RECONCILE_TIMEOUT = httpx.Timeout(10.0, connect=10.0)
    _STOPPABLE_STATUSES = {"registered", "ongoing"}
    _TOLERATED_STOP_STATUSES = {200, 202, 204, 404, 422}
    _LIST_CALLS_LIMIT = 50

    def __init__(
        self,
        *,
        api_key: str,
        agent_id: str,
        from_number: str,
        destination: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if retell is None:
            raise RuntimeError(
                "retell_sdk_missing: RetellCallOriginator needs "
                "retell-sdk>=5.64,<6 (pip install 'retell-sdk>=5.64,<6')"
            )
        self._agent_id = agent_id
        self._from_number = from_number
        self._destination = destination
        # Track ownership of the raw httpx client ourselves rather than via
        # AsyncRetell.close() — that call closes whatever http_client it was
        # given, owned or not, which would close a caller-injected client.
        self._owns_client = client is None
        self._http = client or httpx.AsyncClient()
        self._client = AsyncRetell(
            api_key=api_key,
            base_url=os.environ.get("RETELL_API_BASE_URL", self._base_url),
            timeout=httpx.Timeout(30.0, connect=10.0),
            # Our contract already defines retry/tolerance behaviour (the
            # tolerated-stop-status set, the reconcile guard); the SDK must
            # not layer hidden retries on top of it.
            max_retries=0,
            http_client=self._http,
        )

    @classmethod
    def from_env(cls, transport: Any = None) -> "RetellCallOriginator":
        # Transport (the job's non-secret config) wins; env vars are the
        # local-CLI fallback. Secrets and the leased DID are env-only.
        agent_id = _strip_or_none(getattr(transport, "originator_agent_id", None)) or (
            os.environ.get("RETELL_AGENT_ID", "").strip() or None
        )
        from_number = _strip_or_none(
            getattr(transport, "originator_from_number", None)
        ) or (os.environ.get("RETELL_FROM_NUMBER", "").strip() or None)
        api_key = os.environ.get("RETELL_API_KEY", "").strip() or None
        destination = os.environ.get("LIVEKIT_INBOUND_DID", "").strip() or None
        values = {
            "RETELL_API_KEY": api_key,
            "RETELL_AGENT_ID": agent_id,
            "RETELL_FROM_NUMBER": from_number,
            "LIVEKIT_INBOUND_DID": destination,
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(
                "retell_originator_config_missing: " + ", ".join(sorted(missing))
            )
        return cls(
            api_key=api_key,
            agent_id=agent_id,
            from_number=from_number,
            destination=destination,
        )

    async def start(self) -> RetellCall:
        # Non-2xx raises retell.APIStatusError (a subclass covers each HTTP
        # status); we let it propagate, same failure surface as before.
        response = await self._client.call.create_phone_call(
            from_number=self._from_number,
            to_number=self._destination,
            override_agent_id=self._agent_id,
        )
        # A 2xx without a JSON content-type is passed through by the SDK as
        # raw text (or NoneType for a 204), not the typed PhoneCallResponse
        # model, so `.call_id` would raise AttributeError; getattr keeps the
        # contracted ValueError the only failure mode here.
        call_id = getattr(response, "call_id", None)
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("retell_call_response_missing_id")
        status = response.call_status
        return RetellCall(
            call_id=call_id,
            status=str(status) if status is not None else None,
        )

    async def stop(self, call_id: str, *, timeout: httpx.Timeout | None = None) -> None:
        # Only stop-call may end a call; delete-call destroys the record and
        # transcript. 5.8.0 has no typed stop-call method, so this uses the
        # client's escape hatch to POST the path directly.
        _validate_call_id(call_id)
        # Mirror the SDK's own NoneType-cast pattern (call.py's delete():
        # extra_headers={"Accept": "*/*"}) so this bodyless POST doesn't
        # advertise a JSON body it never sends, nor demand JSON back from an
        # endpoint that may answer 204. The escape-hatch post()
        # takes a RequestOptions dict, not the typed extra_headers kwarg
        # delete() has, so the header goes directly under "headers" —
        # probed against the installed SDK: "extra_headers" here is silently
        # ignored by FinalRequestOptions.construct, "headers" is not.
        options: dict[str, Any] = {"headers": {"Accept": "*/*"}}
        if timeout is not None:
            options["timeout"] = timeout
        try:
            await self._client.post(
                f"/v2/stop-call/{call_id}", cast_to=NoneType, options=options
            )
        except retell.APIStatusError as exc:
            if exc.status_code not in self._TOLERATED_STOP_STATUSES:
                raise

    async def reconcile_and_stop(
        self, *, started_after_ms: int, ended_before_ms: int
    ) -> list[str]:
        # Argument validation runs before any request so a wrong call site
        # fails loudly instead of silently returning [].
        for label, value in (
            ("started_after_ms", started_after_ms),
            ("ended_before_ms", ended_before_ms),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"reconcile_and_stop requires int epoch ms for {label}, "
                    f"got {type(value).__name__}"
                )

        try:
            # SDK-typed v3 FilterCriteria (retell/types/call_list_params.py,
            # confirmed against the installed 5.64.0 wheel ~:605-745):
            # AsyncCallResource.list posts to /v3/list-calls, and
            # from_number/to_number are {"type": "string", "op": "eq",
            # "value": <str>} filters, start_timestamp is a {"type": "range",
            # "op": "bt", "value": [lower_ms, upper_ms]} filter — not the
            # list-valued/{lower_threshold, upper_threshold} v2 shape a prior
            # version of this comment described (PLAN D12, supersedes D10a).
            # Confirmed on the wire via MockTransport against the installed
            # SDK. The to_number filter is defense in depth only; every
            # client-side fence below still applies.
            response = await self._client.call.list(
                filter_criteria={
                    "from_number": {
                        "type": "string",
                        "op": "eq",
                        "value": self._from_number,
                    },
                    "to_number": {
                        "type": "string",
                        "op": "eq",
                        "value": self._destination,
                    },
                    "start_timestamp": {
                        "type": "range",
                        "op": "bt",
                        "value": [started_after_ms, ended_before_ms],
                    },
                },
                limit=self._LIST_CALLS_LIMIT,
                timeout=self._RECONCILE_TIMEOUT,
            )
            # 5.64+: call.list() returns retell.types.CallListResponse, a
            # paginated envelope (items/has_more/pagination_key/total), not
            # a bare list of rows (PLAN D12a). Response handling order:
            # 1. dict — a test double, or any future/legacy body the SDK
            #    hands back unparsed. Checked first because dict.items is a
            #    bound method, not a key list — isinstance(response, dict)
            #    is the only reliable way to keep that from being confused
            #    with the envelope's `.items` attribute probed in branch 3.
            # 2. bare list — a pre-5.64 SDK compatibility path (probed: the
            #    installed 5.64 SDK constructs one CallListResponse per
            #    top-level array element via extra="allow" absorption when
            #    the body itself is a JSON array). Kept for the legacy test
            #    and any older-SDK deploy.
            # 3. else (the real 5.64 envelope) — `.items` is a list of
            #    typed Item pydantic models when the server sent one.
            #    When it's absent or not a list (an unrecognised envelope,
            #    or a legacy `{"calls": [...]}` body the SDK folded into an
            #    extra field instead of `.items` — probed), fall back to
            #    _select_call_rows on a full model_dump so its legacy
            #    dict-key fallbacks get one more pass at the envelope, and
            #    an unrecognised shape still logs a real payload_type
            #    (e.g. "CallListResponse") instead of "dict" below.
            skipped = 0
            if isinstance(response, dict):
                payload: Any = response
                rows, recognised = _select_call_rows(payload)
            elif isinstance(response, list):
                payload, skipped = _dump_rows(response)
                rows, recognised = _select_call_rows(payload)
            else:
                items = getattr(response, "items", None)
                if isinstance(items, list):
                    payload, skipped = _dump_rows(items)
                    rows, recognised = _select_call_rows(payload)
                elif hasattr(response, "model_dump"):
                    rows, recognised = _select_call_rows(
                        response.model_dump(mode="json")
                    )
                    payload = response
                else:
                    # Not actually a pydantic model — e.g. a scalar JSON
                    # body (a bare string or null) the SDK couldn't
                    # construct CallListResponse from at all, so it comes
                    # through as the raw str/None value. _select_call_rows
                    # already falls through such values to ([], False);
                    # payload stays this raw value so the unexpected_payload
                    # log below still reports its real type ("str",
                    # "NoneType"), same as before this round.
                    payload = response
                    rows, recognised = _select_call_rows(payload)
        except (retell.APIError, httpx.HTTPError, ValueError) as exc:
            # retell.APIError covers both APIStatusError (non-2xx) and
            # APIConnectionError (transport failure) — the SDK's hierarchy
            # is not httpx's, so APIConnectionError is NOT an
            # httpx.HTTPError subclass (the test is the actual net
            # for a transport failure). httpx.HTTPError is kept only as
            # insurance for a raw-httpx code path that doesn't exist today —
            # probed dead against the SDK, harmless to keep. ValueError
            # covers a non-JSON 2xx body (e.g. a malformed-but-JSON-typed
            # proxy/WAF page raising json.JSONDecodeError); the
            # swallow-and-log promise applies to that too, not just
            # transport-level errors.
            logger.warning(
                "retell_call_reconcile",
                extra={"phase": "list_calls", "error": type(exc).__name__},
            )
            return []

        if skipped:
            # Ignore, informational: a non-object row is not a payload shape
            # failure (recognised can still be True) and carries no
            # customer data worth logging — count only, same as the other
            # dropped-row counters below.
            logger.warning(
                "retell_reconcile_unexpected_row", extra={"skipped": skipped}
            )

        if not recognised:
            # A JSON body that parsed fine but isn't a shape this parser
            # understands (e.g. a bare string or null) must not be reported
            # the same as a genuinely empty page — the two drive opposite
            # calls on whether the guard is working.
            logger.warning(
                "retell_reconcile_unexpected_payload",
                extra={"payload_type": type(payload).__name__},
            )
            return []

        # 5.64's envelope carries an explicit ``has_more``; a full page is the
        # pre-envelope heuristic. Either means the candidate set may be
        # truncated; ordering isn't guaranteed so our own call could be off
        # the page. Read from the envelope object, never from ``payload``.
        # Checked BEFORE the empty-page return: an empty page the server says
        # is truncated is page_full (guard may have missed its own call),
        # never no_candidates (the "page genuinely empty" signal).
        has_more = bool(getattr(response, "has_more", False))
        if len(rows) == self._LIST_CALLS_LIMIT or has_more:
            logger.warning(
                "retell_reconcile_page_full",
                extra={"row_count": len(rows), "has_more": has_more},
            )
            if not rows:
                return []

        if not rows:
            # The likely inert mode: a range filter on start_timestamp can't
            # match a row that never got one, so a registered call with no
            # timestamp yet is dropped server-side and never comes back.
            logger.warning("retell_reconcile_no_candidates")
            return []

        candidates: list[dict[str, Any]] = []
        dropped_no_start_timestamp = 0
        dropped_out_of_window = 0
        for row in rows:
            start_ts = row.get("start_timestamp")
            if not isinstance(start_ts, (int, float)) or isinstance(start_ts, bool):
                dropped_no_start_timestamp += 1
                continue
            # Server-side `bt` filtering is unverified against a live key;
            # re-check the window here so a widened query can never reach a
            # call outside it.
            if not (started_after_ms <= start_ts <= ended_before_ms):
                dropped_out_of_window += 1
                continue
            candidates.append(row)
        if dropped_no_start_timestamp:
            logger.warning(
                "retell_reconcile_no_start_timestamp",
                extra={"dropped": dropped_no_start_timestamp},
            )
        if dropped_out_of_window:
            logger.warning(
                "retell_reconcile_out_of_window",
                extra={"dropped": dropped_out_of_window},
            )
        if not candidates:
            return []

        # The server-side from_number filter (like to_number) is defense in
        # depth only, not proven — the leased destination DID comes from a
        # shared pool, so recheck from_number client-side too, the same way
        # the window is rechecked above, rather than trusting the server
        # filter to be the only thing scoping this query to our own line.
        dropped_from_number_mismatch = 0
        from_number_matches: list[dict[str, Any]] = []
        for row in candidates:
            if row.get("from_number") == self._from_number:
                from_number_matches.append(row)
            else:
                dropped_from_number_mismatch += 1
        if dropped_from_number_mismatch:
            logger.warning(
                "retell_reconcile_from_number_mismatch",
                extra={"dropped": dropped_from_number_mismatch},
            )
        candidates = from_number_matches
        if not candidates:
            return []

        # A non-string to_number (e.g. a malformed API row shaped as a list
        # or dict) can never equal our destination string either, so it is
        # folded into "no usable destination" rather than reaching a set
        # comprehension, where it would raise instead of just not matching.
        with_destination = [
            row for row in candidates if isinstance(row.get("to_number"), str)
        ]
        if not with_destination:
            # Structurally inert against this response shape — not proof no
            # orphan existed.
            logger.warning("retell_reconcile_no_destination_field")
            return []

        # Exact match only: a formatting difference (e.g. E.164 vs local)
        # must fail closed rather than risk matching a stranger's row.
        destination_matches = [
            row for row in with_destination if row.get("to_number") == self._destination
        ]
        if not destination_matches:
            # Never log the observed numbers themselves — they are the
            # customer's own callees on their production line, not ours.
            distinct_observed = len({row.get("to_number") for row in with_destination})
            logger.warning(
                "retell_reconcile_destination_mismatch",
                extra={
                    "destination": self._destination,
                    "distinct_observed": distinct_observed,
                    "count": len(with_destination),
                },
            )
            return []

        stoppable = [
            row
            for row in destination_matches
            if str(row.get("call_status") or "").lower() in self._STOPPABLE_STATUSES
        ]
        if not stoppable:
            return []

        # Our own dial is the last event inside the window; stop only the
        # latest match and leave every other in-window match alone.
        stoppable.sort(key=lambda row: row["start_timestamp"], reverse=True)
        target, ambiguous = stoppable[0], stoppable[1:]

        call_id = target.get("call_id")
        has_id = isinstance(call_id, str)
        if not has_id or not _CALL_ID_PATTERN.fullmatch(call_id):
            # Silent [] here would look identical to "the guard worked and
            # found nothing" — log which of the two shapes it was.
            logger.warning(
                "retell_reconcile_target_without_id",
                extra={"has_id": has_id, "count": len(stoppable)},
            )
            return []

        if ambiguous:
            # The id we stopped is ours by construction (validated above);
            # the others are the customer's rows and only their count
            # belongs in our logs.
            logger.warning(
                "retell_reconcile_ambiguous",
                extra={"stopped_call_id": call_id, "left_alone": len(ambiguous)},
            )

        try:
            await self.stop(call_id, timeout=self._RECONCILE_TIMEOUT)
        except (retell.APIError, httpx.HTTPError) as exc:
            # httpx.HTTPError is kept as insurance only — the SDK's own
            # errors (APIStatusError, APIConnectionError) are never
            # httpx.HTTPError subclasses (probed); the transport-error net
            # is the test on the list_calls phase above.
            logger.warning(
                "retell_call_reconcile",
                extra={"phase": "stop_call", "error": type(exc).__name__},
            )
            return []

        return [call_id]

    async def close(self) -> None:
        # Never AsyncRetell.close() here — it closes self._http regardless
        # of who created it, which would close a caller-injected client.
        if self._owns_client:
            await self._http.aclose()


class RetellAgentEndpoint:
    def __init__(
        self,
        *,
        name: str,
        channel: Literal["sip_inbound", "web_call"] = "sip_inbound",
        agent_id: str | None = None,
    ) -> None:
        if channel == "sip_outbound":
            raise ValueError(
                "retell_pstn_outbound_unsupported: Retell has no outbound API"
            )
        self.manifest = AgentEndpointManifest(
            name=name,
            provider="retell",
            world_kinds=["voice"],
            capabilities=EndpointCapabilities(
                audio=True,
                text=True,
                streaming=True,
                interruption=True,
                dtmf=False,
                transfer=False,
                transcript_events=True,
                tool_events=True,
                usage_events=True,
                internal_metrics=False,
                recording=True,
                web_rtc=channel == "web_call",
                sip=channel == "sip_inbound",
            ),
            metadata={"channel": channel, "agent_id": agent_id}
            if agent_id
            else {"channel": channel},
        )
        self._channel = channel
        self.capabilities = self.manifest.capabilities

    async def discover(self, request: DiscoveryRequest) -> DiscoverySnapshot:
        del request
        return DiscoverySnapshot(capabilities=self.capabilities)

    async def prepare(self, plan) -> EndpointHandle:  # noqa: ANN001
        return EndpointHandle(
            handle_id=f"retell-{uuid.uuid4().hex[:12]}",
            endpoint_name=self.manifest.name,
            created_at=datetime.now(timezone.utc),
            metadata={
                "plan_id": getattr(plan, "plan_id", None),
                "channel": self._channel,
            },
        )

    async def wait_ready(self, handle: EndpointHandle) -> ReadinessResult:
        del handle
        raise NotImplementedError(
            "Retell direct execution seam; live path uses LiveKit SIP inbound today"
        )

    async def send(
        self, handle: EndpointHandle, event: RealtimeEvent | AudioFrame
    ) -> None:
        raise NotImplementedError("RetellAgentEndpoint.send is a Stage-8 seam")

    async def receive(
        self, handle: EndpointHandle
    ) -> AsyncIterator[RealtimeEvent | AudioFrame]:
        raise NotImplementedError("RetellAgentEndpoint.receive is a Stage-8 seam")
        yield  # type: ignore[unreachable]

    async def stop(self, handle: EndpointHandle) -> None:
        del handle

    async def cleanup(self, handle: EndpointHandle) -> None:
        del handle

    async def reconcile(self, handle: EndpointHandle) -> ReconciliationResult:
        del handle
        return ReconciliationResult(reconciled=True)


__all__ = ["RetellAgentEndpoint", "RetellCall", "RetellCallOriginator"]
