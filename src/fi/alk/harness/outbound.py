"""Outbound reporting — `outbound-channels.md` v1.3, paired with `hosted-execution-seams.md` v1.11
(the spine). All guest -> platform traffic (events, result receipts, artifacts) is outbound HTTPS;
the platform never calls in. Two halves:

Foundations (part 1): the platform capability declaration the gateway uploads to
`/run/futureagi/capabilities.json`, the byte-exact canonical serialization every digest in the
contract is built from, and the durable local spool (with its monotonic sequence allocator) that
emission sits behind so a killed process within a live sandbox never loses or duplicates a
record. (A killed sandbox is deleted by the gateway, spool and all -- there is no in-sandbox
restart producer in the spine today, so the cross-restart recovery machinery guards a scenario
that isn't triggerable yet, but the in-process failed-write and watermark-durability guarantees
are load-bearing from the first event.)

Transport (part 2): a channel-neutral `Transport` protocol plus a `requests`-backed production
implementation; the closed status-code error map (`classify_response`) shared by all three channel
clients; and the clients themselves — `EventsClient` (batches spooled events, advances the spool
watermark only on confirmed delivery), `ResultsClient` (typed `ResultReceiptDraft` + delivery), and
`ArtifactsClient` (content-addressed upload + `ArtifactManifestDraft` + delivery), each sharing one
retry/backoff engine (`_perform_with_retry`) that raises on the two channel-ending outcomes
(`HostedFencedError` for 401/403, `HostedChannelFailedError` for 404 exhausted) and returns a typed
`ChannelError` for everything else a caller must log-and-continue on.

`HostedEvent`/`HostedEventDraft` model Channel 1's wire shape (the "hosted event model" the spine's
implementation-delta list calls for — `sequence`/`attempt_id`/`attempt_number`/`stage`/`digest` —
distinct from `fi.simulate.runtime.events.CanonicalEvent`, which remains the local-SDK wire and is
untouched by this module).

Redaction (v1.3 Channel 1; seams v1.11 §3): `redact_outbound_text` scrubs URL userinfo
(`scheme://user:pw@` -> `scheme://user:***@`) plus an adapter-supplied secret-value list, applied
inside `build_event_record`/`build_result_receipt` to every free-text field the contract names
(`log.message`, `world_unhealthy.cause`, `terminal`/receipt `failure.message`, sub_goal/evaluation
`reason`). It is NOT the full "same secret-content scan as the artifact sealer" the contract also
requires — that broader scan is a separate, sealer-side obligation this module does not implement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Annotated, Any, ClassVar, Literal, Protocol
from urllib.parse import urlparse

try:  # N30: fcntl is POSIX-only; the guest is Linux-only, but the module must still IMPORT
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

import requests
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_serializer,
    model_validator,
)

from .job import FailureDomain, HarnessStage

logger = logging.getLogger(__name__)

CAPABILITIES_SCHEMA_VERSION = "futureagi.harness-capabilities.v1"
CAPABILITIES_PATH = "/run/futureagi/capabilities.json"
EVENT_SCHEMA_VERSION = "futureagi.harness-event.v1"
RESULT_SCHEMA_VERSION = "futureagi.harness-result.v1"
MANIFEST_SCHEMA_VERSION = "futureagi.harness-manifest.v1"

# "Channel 1" limits (outbound-channels.md v1.3) that every producer of an event record needs
# to honor before it ever reaches a transport client. MIN-12: consumed by EventsClient.flush()
# (stamps `schema_version`, clamps the batch to EVENTS_MAX_BATCH) and by HostedEventDraft's own
# size check -- no longer just declared and unused.
EVENTS_MAX_BATCH = 100
EVENT_PAYLOAD_MAX_BYTES = 32 * 1024
# N7: a cumulative-bytes cap on top of EVENTS_MAX_BATCH's event-count cap. 100 events * 32KB could
# reach ~3.2MB; this keeps a proactively-built batch comfortably under a common ~1MB ingress cap
# (nginx's default) so 413 is the exception, not the steady state -- EventsClient.flush() still
# halves and retries reactively on an observed 413 regardless of this cap.
EVENTS_MAX_BATCH_BYTES = 900_000
# §3a: uploads over this size use chunked transfer; consumed by ArtifactsClient's default
# chunk_threshold_bytes.
ARTIFACT_CHUNKED_UPLOAD_THRESHOLD_BYTES = 64 * 1024 * 1024
# "Sequencing"/"Flush window": the drain deadline from the cancel signal, TTL, or terminal event.
# N5/N24: this module does not compute a deadline from it -- every public client method
# (EventsClient.flush / ResultsClient.push / ArtifactsClient.upload / .push_manifest) instead
# accepts an explicit `deadline: float | None` (a `time.monotonic()` value). The adapter (P10) is
# the one process-wide owner of "when did the window start," so it is the one that turns this
# constant into the deadline value it passes in -- not this module.
FLUSH_WINDOW_SECONDS = 120


# =================================================================================================
# Canonicalization -- the byte-exact serialization every digest in the contract is built from.
# =================================================================================================


class OutboundError(RuntimeError):
    """Generic typed failure for this module's canonicalization/digest layer -- same `code`/
    `message` shape as `CapabilitiesError`/`OutboundSpoolError`, used where neither of those is the
    right domain (canonicalization itself, and shape checks that run before any spool or
    capabilities object exists)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def canonical_bytes(value: Any) -> bytes:
    """The contract's canonical form ("Canonicalization (every digest in this file)"):
    ``json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)``,
    encoded UTF-8 (v1.3 pins `allow_nan=False` explicitly). This is the ONLY place that call is
    made -- every digest function in this module goes through it, so a change to the algorithm
    cannot happen in only one of them.

    `allow_nan=False` changes nothing about the bytes for any value that was already valid JSON --
    NaN/Infinity are not RFC 8259, so a float that would have silently produced unparseable bytes
    now fails loudly here instead of downstream at the platform's parser.

    Never re-derive an already-spooled record's bytes by calling this again on retry: a dict's key
    order is stable within one process but nothing guarantees float formatting or dict construction
    order is bit-identical across a restart. `OutboundSpool` hands back the literal bytes it wrote;
    those are what a retry re-sends, per the contract's "serialize once, spool the bytes, re-send
    verbatim; never re-serialize on retry."
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except ValueError as exc:
        raise OutboundError("canonical_value_not_finite", str(exc)) from exc


def sha256_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def event_payload_digest(payload: dict[str, Any]) -> str:
    """Event digest scope: the `payload` object alone (not the envelope around it)."""
    return sha256_digest(canonical_bytes(payload))


def _json_native_offense(value: Any, path: str) -> str | None:
    """Walks `value` looking for the first thing `canonical_bytes` cannot represent for a reason
    OTHER than NaN/Infinity (which `canonical_bytes` itself catches): a non-JSON-native Python
    value (`datetime`, `Decimal`, `UUID`, ...) or a non-string dict key. Returns the offending key
    path (e.g. `"call.started_at"`), or `None` if the tree is clean."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                return (
                    f"{path}[<non-string-key>:{key!r}]"
                    if path
                    else f"<non-string-key>:{key!r}"
                )
            offense = _json_native_offense(item, f"{path}.{key}" if path else key)
            if offense is not None:
                return offense
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            offense = _json_native_offense(item, f"{path}[{index}]")
            if offense is not None:
                return offense
        return None
    return path or "<root>"


def whole_object_digest(obj: dict[str, Any]) -> str:
    """Receipt/manifest digest scope: the whole object with the `digest` key ABSENT.

    The key is popped, never set to `None` -- the contract is explicit that "absent and null are
    different bytes," so silently keeping `digest: null` in the canonicalized form would compute a
    different (wrong) hash than what the platform verifies against.

    Unlike an event payload (already pydantic-validated as `dict[str, JsonValue]` before it ever
    reaches `event_payload_digest`), receipts and manifests reach this function as hand-built
    dicts from whatever calls it -- a bare `TypeError` from `json.dumps` on a non-JSON-native value
    or a non-string key is a debugging dead end with no indication of WHERE in the object the bad
    value lives. This walks the tree first and raises a typed `OutboundError` naming the offending
    key path instead.
    """
    core = {key: value for key, value in obj.items() if key != "digest"}
    offense = _json_native_offense(core, "")
    if offense is not None:
        raise OutboundError(
            "digest_value_not_json_native", f"non-JSON-native value at: {offense}"
        )
    return sha256_digest(canonical_bytes(core))


_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def is_valid_digest(value: str) -> bool:
    return bool(_DIGEST_PATTERN.fullmatch(value))


_RFC3339_MILLIS_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


def format_rfc3339_millis(value: datetime) -> str:
    """The contract's exact timestamp wire form ("Timestamps: RFC 3339, UTC, `Z`, millisecond
    precision"). Unlike `HostedEvent.emitted_at` (an envelope field the event digest scope never
    covers, so its exact string form doesn't matter), receipt/manifest timestamps such as
    `call.started_at` sit INSIDE the `whole_object_digest` scope -- what we hash must be byte-
    identical to what we send, so those fields are plain `str` on the wire models, produced only
    through this function, never through a datetime's default serialization (which pydantic would
    render as e.g. `+00:00` offset and six-digit microseconds, not `Z` and milliseconds).
    """
    if value.tzinfo is None:
        raise ValueError("naive_datetime_not_allowed")
    utc = value.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"


def is_valid_rfc3339_millis(value: str) -> bool:
    return bool(_RFC3339_MILLIS_PATTERN.fullmatch(value))


def _require_utc_millis(value: datetime) -> datetime:
    """Shared `AfterValidator` for every `datetime` field this module serializes through
    `format_rfc3339_millis` (`HostedEventDraft.emitted_at`, `HostedCapabilities.expires_at`):
    rejects a naive datetime outright (the contract's wire form has no naive representation),
    converts any other offset to UTC, and truncates to millisecond precision so the VALUE itself --
    not just its string rendering -- matches what gets sent on the wire."""
    if value.tzinfo is None:
        raise ValueError("naive_datetime_not_allowed")
    utc = value.astimezone(timezone.utc)
    return utc.replace(microsecond=(utc.microsecond // 1000) * 1000)


UtcMillisDatetime = Annotated[datetime, AfterValidator(_require_utc_millis)]


# N9/P3: URL userinfo -- `scheme://user:pw@host` -> `scheme://user:***@host`, matching the seams
# contract's own example (`postgresql://harness:***@...`). The username group is now OPTIONAL
# (`redis://:pw@host`, the canonical empty-username shape for Redis/RabbitMQ/Mongo, is a real
# managed-store DSN form) and so is the whole password group (`https://<token>@host`, the standard
# way a bearer token appears in git/registry output) -- a bare userinfo token is masked outright
# rather than left verbatim on the theory that "a username alone is not a secret," which is false
# for a token.
_USERINFO_PATTERN = re.compile(
    r"([a-zA-Z][a-zA-Z0-9+.\-]*://)([^\s:/?#@]*)(:[^\s/?#]*)?@"
)


def _mask_userinfo(match: re.Match[str]) -> str:
    scheme, user, password = match.group(1), match.group(2), match.group(3)
    return f"{scheme}{user}:***@" if password is not None else f"{scheme}***@"


def redact_outbound_text(value: str, extra_secret_values: tuple[str, ...] = ()) -> str:
    """Scrubs a single free-text field before it can leave the sandbox on any of the three
    channels (outbound-channels.md v1.3 "Redaction (enforced before emit)"; hosted-execution-
    seams.md v1.11 §3 "any outbound projection ... redacts userinfo"). Two things, applied in
    order:

    1. URL userinfo (`_USERINFO_PATTERN`/`_mask_userinfo`, above) -- a password is masked and the
       username kept (matching the contract's own `postgresql://harness:***@...` example); a bare
       token/username-only userinfo (no `:`) is masked outright, since that shape is how a bearer
       token appears, not a username.
    2. `extra_secret_values` -- exact-substring replacement for a caller-supplied list of secret
       values. Always `()` today: the adapter (P10) is what will know the job's declared secrets
       and pass them in -- this parameter exists now so `build_event_record`/`build_result_receipt`
       never need to change shape when that wiring lands.

    NOT a general secret-content scanner -- the contract's "same secret-content scan as the
    artifact sealer" is a separate, sealer-side obligation. This is the narrow subset this module
    can enforce on every string field it controls without false-positiving on ordinary diagnostic
    text.
    """
    redacted = _USERINFO_PATTERN.sub(_mask_userinfo, value)
    for secret in extra_secret_values:
        if secret:
            redacted = redacted.replace(secret, "***")
    return redacted


# =================================================================================================
# Capabilities -- `/run/futureagi/capabilities.json`, "Authentication" section.
# =================================================================================================


class CapabilitiesError(RuntimeError):
    """A capability declaration is missing, malformed, or fails a shape rule.

    Mirrors the `code`/`message` shape `BundleV2Error`/`PreflightError` use elsewhere in this
    package. `code` is one of the nine closed values in outbound-channels.md v1.3's "Capabilities-
    file rejection table" -- see `load_capabilities`, which is the sole place that maps a raw
    failure onto one of them.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class HostedEndpoints(BaseModel):
    """The four outbound routes, per attempt. Shape-only checks (trailing slash, https) live here
    as defense-in-depth for direct construction; `load_capabilities` runs the SAME checks earlier,
    outside pydantic, so each has its own `CapabilitiesError.code` instead of collapsing into the
    generic `capabilities_field_invalid` (see v1.3's rejection table)."""

    model_config = ConfigDict(extra="forbid")

    events: str
    results: str
    artifacts: str
    scenarios: str
    ingress: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> "HostedEndpoints":
        for name in ("events", "results", "artifacts", "scenarios", "ingress"):
            value = getattr(self, name)
            if value is None:
                continue
            if not value or not value.endswith("/"):
                raise ValueError(f"{name} endpoint must end with '/'")
            if not value.startswith("https://"):
                raise ValueError(f"{name} endpoint must use https")
        return self


class HostedCapabilities(BaseModel):
    """The per-attempt bearer plus the four endpoint URLs. Loaded once at emitter startup from the
    file the gateway uploads (§0 step 4) -- see `load_capabilities`, which also implements the
    contract's "loaded into memory ... and unlinked" lifetime rule."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    job_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    attempt_number: int = Field(ge=1)
    fence: str = Field(min_length=1)
    expires_at: UtcMillisDatetime
    token: str = Field(min_length=1)
    endpoints: HostedEndpoints

    @model_validator(mode="after")
    def _schema_shape(self) -> "HostedCapabilities":
        # Defense-in-depth only -- `load_capabilities` checks this first, outside pydantic, so it
        # can raise `capabilities_schema_unsupported` specifically rather than the generic
        # `capabilities_field_invalid` this validator's `ValueError` would collapse into.
        if self.schema_version != CAPABILITIES_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        return self

    @field_serializer("expires_at")
    def _serialize_expires_at(self, value: datetime) -> str:
        return format_rfc3339_millis(value)

    def auth_headers(self) -> dict[str, str]:
        """ "Every request: `Authorization: Bearer <token>` + `X-Harness-Fence: <fence>`." Pure
        formatting -- issuing the request itself is a P8 transport-client concern."""
        return {"Authorization": f"Bearer {self.token}", "X-Harness-Fence": self.fence}

    def event_builder(
        self, *, extra_secret_values: tuple[str, ...] = ()
    ) -> Callable[..., dict[str, Any]]:
        """A `build_event_record`-shaped callable with `job_id`/`attempt_id`/`attempt_number`
        closed over from THIS capabilities object. The contract's `403 attempt_mismatch` fires when
        an event's identity disagrees with the token authenticating it -- binding these three
        fields here makes that class of caller bug unrepresentable at the call site instead of a
        runtime 403 discovered mid-attempt.

        P4: `extra_secret_values` is bound here too, alongside identity, rather than left as a
        per-call parameter -- `build_event_record` could not previously receive the job's declared
        secret list at all through this binder, so a caller wired for `event_builder()` had no way
        to satisfy N9's redaction requirement on `log.message`/`terminal.failure.message` without
        routing around this method entirely. Binding once here matches how identity is already
        bound and gives the adapter one place to get it wrong instead of two.
        """

        def build(
            *,
            event_id: str,
            emitted_at: datetime,
            stage: HarnessStage,
            type: OutboundEventType,
            payload: dict[str, JsonValue],
        ) -> dict[str, Any]:
            return build_event_record(
                event_id=event_id,
                job_id=self.job_id,
                attempt_id=self.attempt_id,
                attempt_number=self.attempt_number,
                emitted_at=emitted_at,
                stage=stage,
                type=type,
                payload=payload,
                extra_secret_values=extra_secret_values,
            )

        return build


def _endpoint_matches_attempt(url: str, attempt_id: str) -> bool:
    """ "an endpoint's `<attempt_id>` path segment disagrees with the declared `attempt_id`"
    (`capabilities_attempt_mismatch`, v1.3) -- checked as a whole path segment, not a substring, so
    an attempt_id that happens to be a substring of another segment can't produce a false match."""
    segments = [segment for segment in urlparse(url).path.split("/") if segment]
    return attempt_id in segments


def _redact_validation_error(exc: ValidationError) -> str:
    """Builds a `capabilities_field_invalid` message from `loc`/`msg` only -- pydantic's default
    `str(exc)` embeds each failing field's `input_value`, and the capabilities file carries the
    bearer token; a caller that logs this message must never be able to leak it."""
    parts = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        msg = error.get("msg", "")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "capabilities file failed validation"


def _warn_if_capabilities_file_insecure(target: Path) -> None:
    """ "owner svc-control, mode 0600" is the contract's posture for this file, but a wrong mode or
    owner is NOT a load-time rejection (MIN-5, fail-safe): the bearer is only a per-attempt token
    that expires on its own, and refusing to load it entirely over a permissions mistake would turn
    a minor hardening gap into a hard attempt failure. Loud warning only."""
    try:
        info = target.stat()
    except OSError:
        return
    mode = info.st_mode & 0o777
    if mode != 0o600:
        logger.warning(
            "%s: capabilities file mode is %o, expected 0600 -- a world/group-readable bearer in "
            "a multi-user sandbox is a leak risk (not blocking the load)",
            target,
            mode,
        )
    try:
        running_uid = os.geteuid()
    except AttributeError:
        return  # os.geteuid() is POSIX-only
    if info.st_uid != running_uid:
        logger.warning(
            "%s: capabilities file is owned by uid %s, not the running uid %s -- expected owner "
            "svc-control per the contract (not blocking the load)",
            target,
            info.st_uid,
            running_uid,
        )


def load_capabilities(
    path: str | Path = CAPABILITIES_PATH,
    *,
    unlink: bool = True,
    now: Callable[[], datetime] | None = None,
    on_unlink_failure: Callable[[OSError], None] | None = None,
) -> HostedCapabilities:
    """Parse and validate one capabilities file against outbound-channels.md v1.3's closed
    "Capabilities-file rejection table" (nine codes, all reachable as `CapabilitiesError.code`).

    A guest that cannot load this file has no channel at all -- no token, no endpoints -- so it
    cannot report its own failure; every branch below raises before any network-capable object
    exists. Most checks run BEFORE `HostedCapabilities.model_validate`, outside any pydantic
    validator: wrapping them in a pydantic `ValidationError` (the old shape) meant they only ever
    surfaced as the generic `capabilities_field_invalid`, never as their own named code -- checking
    here first makes each one an independently raised, independently testable `CapabilitiesError`.

    ``unlink=True`` (the default) implements "loaded into memory at emitter startup and unlinked" --
    the file is only ever removed AFTER a successful parse and validation, never before, so a
    crash mid-load leaves the file in place for the next attempt to read rather than destroying the
    only copy of a not-yet-consumed bearer. A failure to unlink is never fatal to an otherwise-
    successful load (the sandbox is destroyed by the gateway at attempt end regardless) but is no
    longer silently swallowed either: it is reported via `on_unlink_failure` if given, else logged
    (MIN-7) -- the caller can still tell a 0600 bearer may be lingering on disk.

    ``now`` is injectable (defaults to the real clock) so `capabilities_expired` is testable without
    manipulating the wall clock.
    """
    target = Path(path).expanduser()
    try:
        exists = target.is_file()
    except OSError as exc:
        raise CapabilitiesError("capabilities_file_unreadable", str(exc)) from exc
    if not exists:
        raise CapabilitiesError("capabilities_file_missing", str(target))
    _warn_if_capabilities_file_insecure(target)

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise CapabilitiesError("capabilities_file_unreadable", str(exc)) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CapabilitiesError("capabilities_file_malformed", str(exc)) from exc
    if not isinstance(raw, dict):
        raise CapabilitiesError("capabilities_file_malformed", "not a JSON object")

    schema_version = raw.get("schema_version")
    if schema_version != CAPABILITIES_SCHEMA_VERSION:
        raise CapabilitiesError("capabilities_schema_unsupported", str(schema_version))

    attempt_id = raw.get("attempt_id")
    endpoints_raw = raw.get("endpoints")
    if isinstance(endpoints_raw, dict):
        for name in ("events", "results", "artifacts", "scenarios"):
            value = endpoints_raw.get(name)
            if not isinstance(value, str):
                continue  # missing/wrong-typed -- a shape error pydantic below will catch
            if not value.endswith("/"):
                raise CapabilitiesError(
                    "capabilities_endpoint_invalid",
                    f"endpoints.{name} must end with '/'",
                )
            if not value.startswith("https://"):
                raise CapabilitiesError(
                    "capabilities_endpoint_insecure", f"endpoints.{name} must use https"
                )
            if (
                isinstance(attempt_id, str)
                and attempt_id
                and not _endpoint_matches_attempt(value, attempt_id)
            ):
                raise CapabilitiesError(
                    "capabilities_attempt_mismatch",
                    f"endpoints.{name} does not carry the declared attempt_id {attempt_id!r}",
                )

    try:
        capabilities = HostedCapabilities.model_validate(raw)
    except ValidationError as exc:
        raise CapabilitiesError(
            "capabilities_field_invalid", _redact_validation_error(exc)
        ) from exc

    current_time = (now or (lambda: datetime.now(timezone.utc)))()
    if capabilities.expires_at <= current_time:
        raise CapabilitiesError(
            "capabilities_expired", f"expires_at={capabilities.expires_at.isoformat()}"
        )

    if unlink:
        try:
            target.unlink()
        except OSError as exc:
            if on_unlink_failure is not None:
                on_unlink_failure(exc)
            else:
                logger.warning(
                    "%s: failed to unlink the capabilities file after a successful load (%s) -- a "
                    "0600 bearer may still be on disk; the sandbox is destroyed at attempt end "
                    "regardless, so this does not fail the load",
                    target,
                    exc,
                )
    return capabilities


# =================================================================================================
# Channel 1 -- Events. The closed `type` vocabulary and each type's payload shape.
# =================================================================================================


class DegradeReason(str, Enum):
    # C4 v1.3 §2 (FROZEN) -- the `parallelism_degraded` reason vocabulary, closed at EXACTLY
    # these FIVE members, unconditionally. `port_not_consumable` was the former sixth member;
    # C1 v1.3 §4 decision 2 / D28 reclassifies it OUT of the degrade enum to a TERMINAL job
    # failure (raised out-of-band, surfaced via the job-failure path), so it is deliberately
    # absent here and can never be constructed into a `parallelism_degraded` payload.
    RESOURCE_LIMITED = "resource_limited"
    LITERAL_LOCAL_ENDPOINT = "literal_local_endpoint"
    WORLD_START_FAILED = "world_start_failed"
    FIXED_PORT = "fixed_port"
    CONFORMANCE_GATE_FAILED = "conformance_gate_failed"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TerminalReason(str, Enum):
    TTL_EXCEEDED = "ttl_exceeded"
    USER_CANCELED = "user_canceled"


class OutboundEventType(str, Enum):
    STAGE_CHANGED = "stage_changed"
    PARALLELISM_DEGRADED = "parallelism_degraded"
    BASELINE_FROZEN = "baseline_frozen"
    BASELINE_INPUTS_CHANGED = "baseline_inputs_changed"
    WORLD_UNHEALTHY = "world_unhealthy"
    SCENARIO_STARTED = "scenario_started"
    SCENARIO_RETRIED = "scenario_retried"
    LOG = "log"
    TERMINAL = "terminal"


class StageChangedPayload(BaseModel):
    """No `populate_by_name` -- the wire key is `from` (a Python keyword, hence the `from_stage`
    attribute name + alias), and this model is validation-only (`HostedEventDraft` never
    normalizes `self.payload`; it emits the caller's dict verbatim). Allowing population by the
    attribute name too would let a caller who writes `from_stage` in their payload dict pass
    validation while spooling an undefined wire key -- `populate_by_name=True` previously made
    exactly that mistake succeed silently."""

    model_config = ConfigDict(extra="forbid")

    from_stage: HarnessStage | None = Field(alias="from")
    to: HarnessStage


class ParallelismDegradedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: int = Field(ge=1)
    effective: int = Field(ge=1)
    reason: DegradeReason

    @model_validator(mode="after")
    def _range(self) -> "ParallelismDegradedPayload":
        if not (1 <= self.effective < self.requested):
            raise ValueError(
                f"parallelism_degraded_effective_out_of_range: effective={self.effective} "
                f"requested={self.requested}"
            )
        return self


class BaselineFrozenPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs_digest: str
    baseline_ref: str


class BaselineInputsChangedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    previous_digest: str | None
    current_digest: str


class WorldUnhealthyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_index: int = Field(ge=0)
    cause: str = Field(max_length=200)


class ScenarioStartedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_key: str = Field(min_length=1)
    world_index: int = Field(ge=0)
    scenario_attempt: Literal[1, 2]


class ScenarioRetriedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_key: str = Field(min_length=1)
    from_world: int = Field(ge=0)
    to_world: int = Field(ge=0)


class LogPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: LogLevel
    message: str


_LOG_TRUNCATION_MARKER = "…[truncated]"


def truncate_log_message(
    level: str, message: str, *, max_payload_bytes: int = EVENT_PAYLOAD_MAX_BYTES
) -> str:
    """The `log` event's own contract rule -- "truncated to fit with a trailing `…[truncated]`
    marker" -- unlike the other eight event types, which are hard-rejected when oversized (M8:
    `log` is the contract's designated escape hatch for reporting every other permanent failure, so
    the one channel meant to report an oversized diagnostic must not itself throw on size).

    Sizing is against `canonical_bytes({"level": level, "message": <candidate>})`, the exact bytes
    `HostedEventDraft`'s own size check measures, so a truncated message is guaranteed to fit
    before it ever reaches that check. A no-op when already within budget.
    """
    if len(canonical_bytes({"level": level, "message": message})) <= max_payload_bytes:
        return message
    if (
        len(canonical_bytes({"level": level, "message": _LOG_TRUNCATION_MARKER}))
        > max_payload_bytes
    ):
        raise OutboundError(
            "log_payload_budget_too_small",
            f"max_payload_bytes={max_payload_bytes} cannot fit even the truncation marker",
        )
    lo, hi, best = 0, len(message), ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = message[:mid] + _LOG_TRUNCATION_MARKER
        if (
            len(canonical_bytes({"level": level, "message": candidate}))
            <= max_payload_bytes
        ):
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


class TerminalFailure(BaseModel):
    """The terminal event's `failure` shape: `{domain, stage, code, message}` -- a leaner subset of
    `job.HarnessFailure` (no `retryable`/`details`), matching §"Event `type` vocabulary" exactly so
    a canonicalized terminal payload never carries fields the contract doesn't name."""

    model_config = ConfigDict(extra="forbid")

    domain: FailureDomain
    stage: HarnessStage
    code: str
    message: str


class ScenarioCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errored: int = Field(ge=0)
    skipped: int = Field(ge=0)


class TerminalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: HarnessStage
    reason: TerminalReason | None
    failure: TerminalFailure | None
    scenario_counts: ScenarioCounts

    @model_validator(mode="after")
    def _terminal_stage(self) -> "TerminalPayload":
        if not self.stage.terminal:
            raise ValueError(f"terminal_event_stage_not_terminal: {self.stage.value}")
        return self


_PAYLOAD_MODELS: dict[OutboundEventType, type[BaseModel]] = {
    OutboundEventType.STAGE_CHANGED: StageChangedPayload,
    OutboundEventType.PARALLELISM_DEGRADED: ParallelismDegradedPayload,
    OutboundEventType.BASELINE_FROZEN: BaselineFrozenPayload,
    OutboundEventType.BASELINE_INPUTS_CHANGED: BaselineInputsChangedPayload,
    OutboundEventType.WORLD_UNHEALTHY: WorldUnhealthyPayload,
    OutboundEventType.SCENARIO_STARTED: ScenarioStartedPayload,
    OutboundEventType.SCENARIO_RETRIED: ScenarioRetriedPayload,
    OutboundEventType.LOG: LogPayload,
    OutboundEventType.TERMINAL: TerminalPayload,
}


class HostedEventDraft(BaseModel):
    """A Channel 1 event before spool-assigned `sequence`. The caller supplies `digest` itself
    (computed via `event_payload_digest`) -- the model then re-derives it and rejects a mismatch,
    so a caller can never accidentally spool a record whose embedded digest disagrees with its own
    payload bytes.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=64)
    job_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    attempt_number: int = Field(ge=1)
    emitted_at: UtcMillisDatetime
    stage: HarnessStage
    type: OutboundEventType
    payload: dict[str, JsonValue]
    digest: str

    @field_serializer("emitted_at")
    def _serialize_emitted_at(self, value: datetime) -> str:
        return format_rfc3339_millis(value)

    @model_validator(mode="after")
    def _validate(self) -> "HostedEventDraft":
        # `max_length=64` above counts characters; the contract says "opaque <=64 chars," but the
        # platform's column is presumably bytes -- a multi-byte-UTF-8 id could pass the character
        # count and still overflow it (N-2).
        if len(self.event_id.encode("utf-8")) > 64:
            raise ValueError(f"event_id_too_long_in_bytes: {self.event_id!r}")
        if not is_valid_digest(self.digest):
            raise ValueError(f"event_digest_invalid: {self.digest!r}")
        expected = event_payload_digest(self.payload)
        if self.digest != expected:
            raise ValueError("event_digest_mismatch")
        if len(canonical_bytes(self.payload)) > EVENT_PAYLOAD_MAX_BYTES:
            raise ValueError(f"event_payload_too_large: {self.event_id}")

        model_cls = _PAYLOAD_MODELS[self.type]
        try:
            model_cls.model_validate(self.payload)
        except ValidationError as exc:
            raise ValueError(
                f"event_payload_invalid: {self.type.value}: {exc}"
            ) from exc

        if (
            self.type is OutboundEventType.STAGE_CHANGED
            and self.payload.get("to") != self.stage.value
        ):
            raise ValueError(
                "event_stage_mismatch: stage_changed.to must equal the event's stage"
            )
        if (
            self.type is OutboundEventType.TERMINAL
            and self.payload.get("stage") != self.stage.value
        ):
            raise ValueError(
                "event_stage_mismatch: terminal.stage must equal the event's stage"
            )
        return self


class HostedEvent(HostedEventDraft):
    """The full Channel 1 wire object, `sequence` included -- what actually gets spooled and sent.
    Distinct from `fi.simulate.runtime.events.CanonicalEvent` (the untouched local-SDK wire)."""

    sequence: int = Field(ge=1)


def build_event_record(
    *,
    event_id: str,
    job_id: str,
    attempt_id: str,
    attempt_number: int,
    emitted_at: datetime,
    stage: HarnessStage,
    type: OutboundEventType,
    payload: dict[str, JsonValue],
    extra_secret_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Validate one event's shape and compute its digest, returning a plain dict with no
    `sequence` key -- ready for `OutboundSpool.append`, which assigns `sequence` and performs the
    one-time serialization. Raises `ValueError` (via pydantic) on any shape violation; callers that
    want a typed/coded failure should catch `pydantic.ValidationError` themselves, matching how the
    rest of this package surfaces model-layer rejections (`bundle_v2.py`, `job.py`).

    N9: `redact_outbound_text` runs on every free-text field the contract names BEFORE the digest
    is computed -- `log.message`, `world_unhealthy.cause`, `terminal.failure.{code,message}`,
    `baseline_frozen.baseline_ref` (P8) -- so the embedded digest always matches the redacted bytes
    actually spooled and sent, never the unredacted original. `log` events are then truncated to
    fit (M8), also before the digest -- redact first, since truncation must size against the final
    (redacted) text, not text that would still shrink again once secrets are scrubbed. Every other
    event type still hard-rejects when oversized, via `HostedEventDraft`'s own size check.

    P8: `failure.code` is redacted alongside `failure.message` -- both are free `str` fields (the
    contract's `code` vocabularies are closed in prose, but nothing enforces that here), and
    `baseline_ref` is likewise a free `str` that plausibly carries an OCI/registry reference in the
    same `https://<token>@registry/...` shape `redact_outbound_text` already scrubs.
    """
    payload = dict(payload)
    if type is OutboundEventType.LOG:
        level, message = payload.get("level"), payload.get("message")
        if isinstance(message, str):
            message = redact_outbound_text(message, extra_secret_values)
            if isinstance(level, str):
                message = truncate_log_message(level, message)
            payload["message"] = message
    elif type is OutboundEventType.WORLD_UNHEALTHY:
        cause = payload.get("cause")
        if isinstance(cause, str):
            payload["cause"] = redact_outbound_text(cause, extra_secret_values)
    elif type is OutboundEventType.BASELINE_FROZEN:
        baseline_ref = payload.get("baseline_ref")
        if isinstance(baseline_ref, str):
            payload["baseline_ref"] = redact_outbound_text(
                baseline_ref, extra_secret_values
            )
    elif type is OutboundEventType.TERMINAL:
        failure = payload.get("failure")
        if isinstance(failure, dict):
            redacted_failure = dict(failure)
            if isinstance(failure.get("code"), str):
                redacted_failure["code"] = redact_outbound_text(
                    failure["code"], extra_secret_values
                )
            if isinstance(failure.get("message"), str):
                redacted_failure["message"] = redact_outbound_text(
                    failure["message"], extra_secret_values
                )
            payload["failure"] = redacted_failure
    digest = event_payload_digest(payload)
    draft = HostedEventDraft(
        event_id=event_id,
        job_id=job_id,
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        emitted_at=emitted_at,
        stage=stage,
        type=type,
        payload=payload,
        digest=digest,
    )
    return draft.model_dump(mode="json")


# =================================================================================================
# Spool -- durable on-disk queue + monotonic sequence allocator.
# =================================================================================================


@dataclass(frozen=True)
class SpooledRecord:
    """One durably-appended record. `body` is the EXACT canonical bytes written to disk -- a P8
    transport client re-sends `body` verbatim on retry rather than re-serializing the decoded
    dict, per the contract's "serialize once ... never re-serialize on retry.\""""

    sequence: int | None
    body: bytes

    def decode(self) -> dict[str, Any]:
        return json.loads(self.body.decode("utf-8"))


class OutboundSpoolError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _iter_complete_records(
    data: bytes,
) -> tuple[
    list[
        tuple[int, bytes, dict[str, Any] | list[Any] | str | int | float | bool | None]
    ],
    int,
    int | None,
]:
    """Shared by `_recover` and `records()`/`pending_since_watermark()` -- the ONE place spool
    bytes are split into records, so both ever agree on what a "complete record" is.

    Splits on the literal `b"\\n"` byte (N-1: not `bytes.splitlines()`, which also treats `\\r`/
    `\\r\\n` as separators `append` never writes -- canonical JSON never contains a raw newline of
    any kind, so `\\n` is the only byte that can legitimately end a line).

    Returns `(records_before_corruption, valid_length, corruption_offset)` (N8):
    - `records_before_corruption`: every complete, successfully decoded, non-blank line UP TO the
      first corrupt one (or all of them, if none is corrupt), as `(start_offset, raw_line,
      decoded_value)`, in file order.
    - `corruption_offset`: the byte offset of the first `\\n`-terminated line that failed to parse
      as JSON -- genuine corruption (a torn write, by definition, never got its trailing `\\n`, so
      this is never that) -- or `None` if no such line was found. Once found, scanning STOPS: never
      renumber or trust anything past a corrupt byte (B1).
    - `valid_length`: when `corruption_offset is None`, the byte offset immediately after the last
      complete line -- where `_recover` truncates away a torn tail. When corruption WAS found, this
      equals `corruption_offset` and callers must NOT use it to truncate -- corrupt bytes are left
      on disk, never deleted (N8: "never truncates mid-file damage").
    """
    records: list[tuple[int, bytes, Any]] = []
    start = 0
    while True:
        newline_index = data.find(b"\n", start)
        if newline_index == -1:
            return records, start, None
        line = data[start:newline_index]
        line_end = newline_index + 1
        if line:
            try:
                decoded = json.loads(line.decode("utf-8"))
            except ValueError:
                return records, start, start
            records.append((start, line, decoded))
        start = line_end


class OutboundSpool:
    """Durable, crash-safe local queue for one outbound record stream (events, results, or
    artifact-manifest state) -- the "fsync-first local spool" the contract requires emission to sit
    behind ("Emission is an async flusher over the fsync-first local spool -- it never blocks the
    call loop"). `sequenced=True` is for Channel 1 only ("Sequencing: one allocator, one lock,
    assigned at spool append, contiguous from 1" -- receipts and the manifest carry no `sequence`
    field and use their own idempotency keys instead).

    ONE ALLOCATOR PER STREAM (M6): `OutboundSpool(root, name, ...)` is keyed on
    `(resolved_root, name)` -- a second construction for the same key, anywhere in this process,
    returns the SAME instance rather than a second independent allocator (see `__new__`); a second
    OS PROCESS pointing at the same directory fails loudly instead, via an `fcntl.flock` on
    `<name>.spool.lock` held for the life of the owning instance.

    Recovery rule (the contract specifies the sequencing invariant -- contiguous from 1, no gaps or
    dupes across a restart -- but not the recovery mechanism; this is the FAIL-SAFE/REVERSIBLE
    choice under the stuck-decision rule, surfaced in the P7 report):

    The next sequence number is derived by SCANNING the spool's own JSONL log at startup, never
    from an independent counter file. A separate counter file could be durably advanced in a write
    that lands, while the record it was allocated for does not (crash between the two writes),
    producing a sequence number with no corresponding record -- a permanent, undetectable gap.
    Scanning the log makes the durably-written records themselves the only source of truth. The
    scan is then reconciled against the durable watermark (M5): `next_sequence =
    max(max_sequence_in_log, watermark) + 1` -- the watermark can be AHEAD of the log (the log lost
    already-processed records, e.g. via the directory-fsync gap M2 closes) but never behind it, so
    taking the max is always safe and never skips a record that was actually spooled.

    A torn last line -- a crash mid-write, since a single `write()` of `body + b"\\n"` is not
    guaranteed atomic by POSIX for a regular file -- is detected (the trailing bytes don't end in
    `b"\\n"`) and the file is truncated back to the end of the last complete record before any
    further append. The next append then reuses that same sequence number rather than skipping it:
    a torn write is treated as though it never happened, closing the gap instead of creating one.
    This depends on canonical JSON never containing a raw newline byte (control characters are
    always escaped by `json.dumps`), which `append` asserts on every write. A COMPLETE line that
    still fails to parse is a different, worse fault: genuine corruption degrades the stream to its
    readable prefix rather than raising (N8, `is_corrupt`) -- see `_recover`/`records()`.

    Registration (N4): a construction is only added to `_registry` at the END of a successful
    `__init__`, under `_registry_lock` -- never a half-built instance. A failed construction (e.g.
    `mkdir` EACCES, or `_recover` finding corruption) therefore never poisons the key: it raises
    without registering anything, and the NEXT `OutboundSpool(root, name, ...)` call starts a
    completely fresh attempt rather than returning (or conflicting with) wreckage. The real mutual-
    exclusion primitive across a same-key construction race is `_acquire_process_lock`'s `flock`
    (an OS-level device, safe across threads and processes alike) -- the registry dict on top is
    only a same-process memoization cache.
    """

    _registry: ClassVar[dict[tuple[Path, str], "OutboundSpool"]] = {}
    _registry_lock: ClassVar[RLock] = RLock()

    def __new__(
        cls, root: str | Path, name: str, *, sequenced: bool
    ) -> "OutboundSpool":
        resolved_root = Path(root).expanduser().resolve()
        key = (resolved_root, name)
        with cls._registry_lock:
            existing = cls._registry.get(key)
            if existing is not None:
                # getattr belt-and-braces (N4): `existing` is only ever registered after a fully
                # successful __init__, so `_sequenced` should always be set -- but never trust that
                # invariant harder than a defensive read costs.
                if getattr(existing, "_sequenced", None) != sequenced:
                    raise OutboundSpoolError(
                        "outbound_spool_sequenced_mismatch",
                        f"{name}: existing instance has sequenced={getattr(existing, '_sequenced', None)}, "
                        f"requested sequenced={sequenced}",
                    )
                return existing
            return super().__new__(cls)

    def __init__(self, root: str | Path, name: str, *, sequenced: bool) -> None:
        if getattr(self, "_initialized", False):
            return
        resolved_root = Path(root).expanduser().resolve()
        key = (resolved_root, name)
        with type(self)._registry_lock:
            if getattr(self, "_initialized", False):
                return
            lock_fd: int | None = None
            try:
                self.root = resolved_root
                self.root.mkdir(parents=True, exist_ok=True)
                try:
                    os.chmod(
                        self.root, 0o700
                    )  # MIN-10: mkdir's mode is subject to umask
                except OSError:
                    pass
                self._name = name
                self._sequenced = sequenced
                self._path = self.root / f"{name}.spool.jsonl"
                self._watermark_path = self.root / f"{name}.spool.watermark.json"
                self._lock = RLock()
                self._dir_synced = False
                self._offset_by_sequence: dict[int, int] = {}
                self._next_sequence = 1 if sequenced else None
                self._poisoned = False  # N12
                self._corrupt_since_offset: int | None = None  # N8
                self._forked = False  # N25
                self._closed = False  # P2
                self._lock_fd = self._acquire_process_lock()
                lock_fd = self._lock_fd
                self._recover()
            except BaseException:
                # N4: never leave a half-built instance registered -- it was never added (below),
                # so there is nothing to evict; just release whatever this attempt itself opened.
                if lock_fd is not None:
                    try:
                        os.close(lock_fd)
                    except OSError:
                        pass
                raise
            self._initialized = True
            type(self)._registry[key] = self

    @classmethod
    def _forget_for_tests(cls, root: str | Path, name: str) -> None:
        """Test-only escape hatch: a real process restart naturally gets a fresh, empty registry
        (a new interpreter); simulating that WITHIN one process/test needs an explicit evict so the
        next `OutboundSpool(root, name, ...)` call re-scans the on-disk log instead of returning the
        still-live cached instance. Never called from production code."""
        resolved_root = Path(root).expanduser().resolve()
        with cls._registry_lock:
            instance = cls._registry.get((resolved_root, name))
        if instance is not None:
            instance.close()

    @classmethod
    def _clear_registry_for_tests(cls) -> None:
        """Broader sibling of `_forget_for_tests`: releases every cached instance's lock fd and
        empties the registry. Intended for an autouse test fixture so flock fds don't accumulate
        across a whole test session."""
        with cls._registry_lock:
            instances = list(cls._registry.values())
        for instance in instances:
            instance.close()

    def close(self) -> None:
        """N26/P2: releases this instance's process lock and evicts it from the registry, so a
        later `OutboundSpool(root, name, ...)` call re-scans the on-disk log instead of reusing this
        instance. Idempotent -- safe to call more than once, or on an instance never fully
        constructed.

        P2: also sets `_closed`, so THIS instance -- not just the registry slot -- refuses further
        mutation. Evicting the registry entry alone left the closed instance itself fully live: a
        caller still holding a reference could keep appending with no flock held (the lock fd was
        released), and a fresh `OutboundSpool(...)` call for the same key would allocate a second,
        independent `_next_sequence` -- two live allocators for one stream, each unaware of the
        other, which is exactly the M6 invariant `close()` must not itself reopen."""
        with type(self)._registry_lock:
            key = (getattr(self, "root", None), getattr(self, "_name", None))
            if type(self)._registry.get(key) is self:
                del type(self)._registry[key]
            fd = getattr(self, "_lock_fd", None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                self._lock_fd = None
            self._closed = True

    def _require_writable(self) -> None:
        """P2/P7: the single gate `append`, `advance_watermark`, and `_rewrite_retaining` all call
        before touching disk -- refuses a closed, forked, or poisoned instance instead of letting it
        silently duplicate allocators, advance a parent's watermark from a forked child, or compound
        a rollback failure `_truncate_to` already flagged as unrecoverable."""
        if self._closed:
            raise OutboundSpoolError(
                "outbound_spool_closed",
                f"{self._name}: this OutboundSpool was closed; construct a new one for this stream",
            )
        if self._forked:
            raise OutboundSpoolError(
                "outbound_spool_forked",
                f"{self._name}: this OutboundSpool was constructed before a fork; construct a "
                f"new one in the child process instead of reusing this one",
            )
        if self._poisoned:
            raise OutboundSpoolError(
                "outbound_spool_poisoned",
                f"{self._name}: a prior rollback failed and left this spool in an unknown "
                f"state; it must not be mutated again",
            )

    def _acquire_process_lock(self) -> int:
        """M6: cross-PROCESS protection (the in-process registry above only protects against a
        second Python-level instance in this same interpreter). Held for the life of this instance
        -- released implicitly when its fd closes (process exit, `close()`, or `_forget_for_tests`
        in tests)."""
        if fcntl is None:  # N30
            raise OutboundSpoolError(
                "outbound_spool_platform_unsupported",
                f"{self._name}: fcntl (POSIX file locking) is unavailable on this platform",
            )
        lock_path = self.root / f"{self._name}.spool.lock"
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise OutboundSpoolError(
                "outbound_spool_locked",
                f"{self._name}: already locked by another process",
            ) from exc
        return fd

    def _fsync_dir(self) -> None:
        fd = os.open(str(self.root), os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _report_corruption(self, offset: int) -> None:
        """N8: the client-visible half of "degrade, don't wedge" -- `is_corrupt`/`corruption_offset`
        stay true/set for the life of this instance once discovered, but the loud `logger.error`
        fires only the FIRST time (repeated reads of an already-known-corrupt spool would otherwise
        spam the log every flush cycle)."""
        with self._lock:
            already_reported = self._corrupt_since_offset is not None
            self._corrupt_since_offset = offset
        if not already_reported:
            logger.error(
                "%s: spool corrupt at byte offset %d -- the file is left untouched; only records "
                "before that offset are trusted. Reading degrades to the readable prefix rather "
                "than raising -- this is reported once per process, not per read.",
                self._name,
                offset,
            )

    @property
    def is_corrupt(self) -> bool:
        """N8: set once `_recover` or a later read finds a genuinely corrupt (not merely torn)
        record. A caller can turn this into a `log`-kind event; nothing in this class does that
        itself (no scenario/attempt context here to build one)."""
        return self._corrupt_since_offset is not None

    @property
    def corruption_offset(self) -> int | None:
        return self._corrupt_since_offset

    def _recover(self) -> None:
        # NOTE: the sequenced branch below must run even when the log is missing or empty -- an
        # M2-style lost file (durable watermark, vanished log) is exactly the case M5 needs to
        # reconcile against; an early return here would skip that reconciliation entirely and
        # silently reset the allocator to 1.
        records: list[tuple[int, bytes, Any]] = []
        if self._path.exists():
            data = self._path.read_bytes()
            if data:
                records, valid_length, corruption_offset = _iter_complete_records(data)
                if corruption_offset is not None:
                    # N8: never truncate mid-file damage -- leave the bytes exactly as they are,
                    # trust only what came before, and let construction succeed anyway (B1 already
                    # forbids renumbering past it; the OTHER extreme -- raising here -- would
                    # discard every future emit, including the terminal event, forever).
                    self._report_corruption(corruption_offset)
                elif valid_length < len(data):
                    with self._path.open("r+b") as stream:
                        stream.truncate(valid_length)
                        stream.flush()
                        os.fsync(
                            stream.fileno()
                        )  # MIN-11: durable, not left as a crash window
        if self._sequenced:
            max_sequence = 0
            offsets: dict[int, int] = {}
            for offset, _line, record in records:
                if isinstance(record, dict):
                    sequence = record.get("sequence")
                    if isinstance(sequence, int):
                        offsets[sequence] = offset
                        if sequence > max_sequence:
                            max_sequence = sequence
            self._offset_by_sequence = offsets
            watermark = self.watermark()
            if watermark > max_sequence:
                logger.warning(
                    "%s: watermark (%s) is ahead of the highest sequence found in the spool (%s) "
                    "-- the log lost records the platform already processed; seeding "
                    "next_sequence from the watermark so newly allocated sequences don't collide "
                    "with ones the platform already closed",
                    self._name,
                    watermark,
                    max_sequence,
                )
            self._next_sequence = max(max_sequence, watermark) + 1

    def _truncate_to(self, size: int) -> None:
        """B1: a true no-op on a failed append. `_next_sequence` is only advanced AFTER a
        successful write, so the retry reuses the same sequence number -- this makes sure it reuses
        clean ground too, instead of appending immediately after torn bytes with no `\\n` between
        them (which would merge into one unparseable line the rest of this class can't recover
        from)."""
        try:
            with self._path.open("r+b") as stream:
                stream.truncate(size)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            # N12: the write may have failed before the file even existed (nothing to truncate,
            # harmless) OR the rollback itself failed on an existing torn write -- in the latter
            # case B1's "a failed append is a true no-op" no longer holds, so poison this spool
            # rather than let a future append silently merge into the torn bytes.
            self._poisoned = True
            logger.error(
                "%s: rollback of a failed append could not truncate the spool back to %d bytes "
                "(%s) -- the file may now carry torn bytes; poisoning this spool so a caller sees "
                "a typed error instead of a future append compounding the corruption",
                self._name,
                size,
                exc,
            )

    def append(self, record: dict[str, Any]) -> SpooledRecord:
        with self._lock:
            self._require_writable()
            if self._sequenced:
                assigned = self._next_sequence
                record = {**record, "sequence": assigned}
            elif "sequence" in record:
                raise OutboundSpoolError(
                    "outbound_spool_caller_supplied_sequence",
                    f"{self._name} spool does not assign sequence numbers; caller must not pass one",
                )
            body = canonical_bytes(record)
            if b"\n" in body:
                # Framing invariant `_iter_complete_records`/`_recover` depend on: canonical JSON
                # never contains a raw newline (json.dumps escapes control characters inside
                # strings), so this would only fire on a value this module's own canonicalization
                # contract disallows.
                raise OutboundSpoolError(
                    "outbound_spool_record_unframable",
                    f"{self._name}: record contains a raw newline",
                )
            existed_before = self._path.exists()
            size_before = self._path.stat().st_size if existed_before else 0
            try:
                with self._path.open("ab") as stream:
                    stream.write(body)
                    stream.write(b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                self._truncate_to(size_before)
                raise
            if not existed_before and not self._dir_synced:
                # M2: the directory entry for a brand-new file isn't durable just because the
                # file's own data is -- fsync it once (not per append; existing files' entries were
                # already synced by whichever append first created them).
                self._fsync_dir()
                self._dir_synced = True
            if self._sequenced:
                self._offset_by_sequence[assigned] = size_before
                self._next_sequence = assigned + 1
                return SpooledRecord(sequence=assigned, body=body)
            return SpooledRecord(sequence=None, body=body)

    def records(self) -> list[SpooledRecord]:
        """Reads the WHOLE log. The read itself happens OUTSIDE `self._lock` (M4): only the size
        snapshot that bounds it is taken under the lock, so the flusher's (potentially large) read
        never blocks `append`, the call loop's write path, for its duration. Safe because `append`
        only ever grows the file -- a read bounded to a size captured a moment earlier can only be
        stale, never torn. (`compact_through`/`drop` DO shrink the file and take the lock for their
        entire duration; this module assumes the flusher serializes its own reads against its own
        compactions rather than running them from two different threads.)
        """
        with self._lock:
            if not self._path.exists():
                return []
            size = self._path.stat().st_size
        with self._path.open("rb") as stream:
            data = stream.read(size)
        parsed, _valid_length, corruption_offset = _iter_complete_records(data)
        if (
            corruption_offset is not None
        ):  # N8: degrade to the readable prefix, never raise here
            self._report_corruption(corruption_offset)
        out: list[SpooledRecord] = []
        for _offset, line, decoded in parsed:
            sequence = (
                decoded.get("sequence")
                if self._sequenced and isinstance(decoded, dict)
                else None
            )
            out.append(SpooledRecord(sequence=sequence, body=line))
        return out

    def records_after(self, sequence: int) -> list[SpooledRecord]:
        if not self._sequenced:
            raise OutboundSpoolError("outbound_spool_unsequenced", self._name)
        return [item for item in self.records() if (item.sequence or 0) > sequence]

    @property
    def next_sequence(self) -> int:
        if not self._sequenced:
            raise OutboundSpoolError("outbound_spool_unsequenced", self._name)
        assert self._next_sequence is not None
        return self._next_sequence

    def watermark(self) -> int:
        """The highest-processed sequence acknowledged so far ("the watermark is highest-processed
        -- accepted AND rejected sequences both advance it"). Durable across a restart via a
        fsync'd-temp-then-rename-then-fsync'd-directory write (M1) -- a corrupt or unreadable
        watermark file DEGRADES TO 0 with a loud diagnostic rather than raising and wedging the
        spool: re-sending already-acked events is safe (at-least-once delivery + platform-side
        dedupe on `event_id`), while a permanently unreadable outbound channel is not.
        """
        if not self._sequenced:
            raise OutboundSpoolError("outbound_spool_unsequenced", self._name)
        if not self._watermark_path.exists():
            return 0
        try:
            raw = json.loads(self._watermark_path.read_text(encoding="utf-8"))
            return int(raw["acked_through_sequence"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "%s: watermark file is corrupt or unreadable (%s) -- degrading to 0. Re-sending "
                "already-acked events is safe (at-least-once + dedupe on event_id); wedging the "
                "spool permanently is not.",
                self._name,
                exc,
            )
            return 0

    def advance_watermark(self, sequence: int) -> None:
        """v1.3: `acked_through_sequence` is untrusted platform input. A value outside
        `[current_watermark, next_sequence)` is rejected locally with a typed error and the
        watermark is left exactly as it was -- "a malformed ack must not be able to discard
        pending records" (M7). `sequence == current_watermark` is a legitimate no-op, not an error
        (repeating the same ack, or a same-valued out-of-order response).

        P7: this is the one operation that DESTROYS delivery state (it durably advances what a
        future `pending_since_watermark()` will ever return again) -- a forked child or a poisoned
        instance advancing it would silently orphan every pending record below the new value, the
        N1 outcome by a different route. `_require_writable()` guards it for that reason even
        though nothing here writes to the JSONL log itself.
        """
        if not self._sequenced:
            raise OutboundSpoolError("outbound_spool_unsequenced", self._name)
        with self._lock:
            self._require_writable()
            current = self.watermark()
            if sequence < current or sequence >= self._next_sequence:
                raise OutboundSpoolError(
                    "outbound_spool_watermark_out_of_range",
                    f"{self._name}: acked_through_sequence={sequence} outside the trusted range "
                    f"[{current}, {self._next_sequence}) -- untrusted platform input, watermark "
                    f"left unchanged",
                )
            if sequence == current:
                return
            temporary = (
                self.root
                / f"{self._name}.spool.watermark.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            with temporary.open("wb") as stream:
                stream.write(
                    json.dumps(
                        {"acked_through_sequence": sequence}, separators=(",", ":")
                    ).encode("utf-8")
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._watermark_path)
            self._fsync_dir()

    def pending_since_watermark(self) -> list[SpooledRecord]:
        """Convenience for "the guest advances its spool cursor through the watermark": every
        spooled record not yet acknowledged, in sequence order. Uses the offset `append` recorded
        for the first pending sequence to SEEK directly there (M4) instead of re-reading and
        re-parsing the whole log on every flush cycle; falls back to the generic full scan when no
        cached offset exists yet (e.g. a fresh recovery whose watermark sits past every record this
        process itself has written an offset for).

        N13: the drained steady state (nothing pending -- what a polling flusher sees most cycles)
        is checked first and returns `[]` with NO file IO at all: `watermark + 1 == next_sequence`
        means every allocated sequence has already been acknowledged, so there is nothing on disk
        to seek to regardless of what `_offset_by_sequence` does or doesn't have cached.
        """
        watermark = self.watermark()
        with self._lock:
            if watermark + 1 == self._next_sequence:
                return []
            offset = self._offset_by_sequence.get(watermark + 1)
        if offset is None:
            return self.records_after(watermark)
        with self._lock:
            if not self._path.exists():
                return []
            size = self._path.stat().st_size
        if offset >= size:
            return []
        with self._path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(size - offset)
        parsed, _valid_length, corruption_offset = _iter_complete_records(data)
        if (
            corruption_offset is not None
        ):  # N8: absolute offset -- `data` starts at `offset`
            self._report_corruption(offset + corruption_offset)
        return [
            SpooledRecord(
                sequence=decoded.get("sequence") if isinstance(decoded, dict) else None,
                body=line,
            )
            for _offset, line, decoded in parsed
        ]

    def _rewrite_retaining(self, keep: Callable[[Any], bool]) -> None:
        """Shared by `compact_through` and `drop_many`: rewrites the log keeping only records
        `keep` accepts, via the same write-fsync/atomic-replace/fsync-directory durability shape
        `append`/`advance_watermark` use -- a crash mid-rewrite leaves either the old file intact
        or the new one complete, never a torn hybrid.

        P1: if the log is already corrupt (N8), this REFUSES instead of rewriting. A rewrite always
        replaces the file from what it read, and reading stops at the corruption offset -- so
        rewriting on a corrupt spool would not merely skip the corrupt bytes, it would silently
        destroy every intact record PAST them too, including ones this process itself appended
        after recovery that have never been sent (the terminal event, in the worst case). That
        directly defeats N8's "the readable prefix stays usable, delivery keeps working" posture
        the moment the first drop/compact happens. Refusing costs only unbounded disk growth until
        the attempt ends (`compact_through`'s whole job) or a rejected record staying spooled but
        never re-emitted anyway, since it's at or below the watermark (`drop_many`'s whole job) --
        both strictly better than deleting undelivered records.
        """
        with self._lock:
            self._require_writable()
            if not self._path.exists():
                return
            size = self._path.stat().st_size
            with self._path.open("rb") as stream:
                data = stream.read(size)
            parsed, _valid_length, corruption_offset = _iter_complete_records(data)
            if corruption_offset is not None:
                self._report_corruption(corruption_offset)
                logger.error(
                    "%s: refusing to compact/drop on a corrupt spool -- a rewrite replaces the file "
                    "from what it read, and reading stops at byte %d, so every record past that "
                    "offset (including not-yet-delivered ones) would be destroyed. The log is left "
                    "intact and grows unbounded until the attempt ends; that is the fail-safe half "
                    "of degrade-not-wedge.",
                    self._name,
                    corruption_offset,
                )
                return
            temporary = (
                self.root
                / f"{self._name}.spool.jsonl.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            offsets: dict[int, int] = {}
            offset = 0
            with temporary.open("wb") as stream:
                for _old_offset, line, decoded in parsed:
                    if not keep(decoded):
                        continue
                    if (
                        self._sequenced
                        and isinstance(decoded, dict)
                        and isinstance(decoded.get("sequence"), int)
                    ):
                        offsets[decoded["sequence"]] = offset
                    stream.write(line)
                    stream.write(b"\n")
                    offset += len(line) + 1
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            self._fsync_dir()
            if self._sequenced:
                self._offset_by_sequence = offsets

    def compact_through(self, sequence: int) -> None:
        """M4: physically drops every durably-acked record (`sequence <= min(sequence,
        watermark())`) from the on-disk log, bounding its growth for a long `running` stage. The
        allocator's `next_sequence` is unaffected -- it is only ever derived from the log at
        `_recover` time, and recovery's own `max(max_sequence, watermark)` rule (M5) already
        tolerates a log whose historical records were compacted away, since none of them can be
        the true maximum (compaction only ever removes sequences at or below the watermark, and
        the watermark is always <= every pending, uncompacted sequence).

        Clamped to the current watermark regardless of what the caller passes -- compacting past
        an event the platform hasn't actually processed yet would be irreversible data loss, and
        this module's posture throughout is fail-safe over trusting the caller.
        """
        if not self._sequenced:
            raise OutboundSpoolError("outbound_spool_unsequenced", self._name)
        effective = min(sequence, self.watermark())
        self._rewrite_retaining(
            lambda decoded: (
                not (
                    isinstance(decoded, dict)
                    and isinstance(decoded.get("sequence"), int)
                    and decoded["sequence"] <= effective
                )
            )
        )

    def drop_many(self, sequences: Collection[int]) -> None:
        """The contract's rejected-event mechanism: "a rejected event is dropped from the spool ...
        it is never re-emitted" (M5). PURE physical removal (N1) -- every sequence in `sequences`
        is deleted from the on-disk log in ONE rewrite pass (N14: a batch of 100 rejections is one
        `_rewrite_retaining` call, not 100), and the watermark is left untouched.

        N1: an earlier version had `drop` also advance the watermark to `sequence`, on the theory
        that "a rejected event closes its sequence." That let an UNTRUSTED `rejected[].sequence`
        from the platform silently orphan every pending record below it, bypassing
        `advance_watermark`'s own M7 clamp entirely -- the clamp only guards `acked_through_sequence`
        callers, and `drop` skipped straight past it. The batch-level
        `advance_watermark(acked_through_sequence)` a caller performs separately is the ONE place
        the watermark ever moves; it already covers every rejected sequence under a conformant
        platform (rejections advance the watermark by contract), and under a non-conformant one
        M7's clamp is then the single, correct chokepoint -- this method has no clamp of its own to
        bypass.

        Writing the record's payload to the artifact spool as a `log` kind (the other half of the
        contract's drop rule) is NOT this method's job -- that hand-off needs scenario/attempt
        context and an `ArtifactsClient` this layer doesn't own; it is P9/P10 wiring, documented
        here as the seam rather than guessed at.
        """
        if not self._sequenced:
            raise OutboundSpoolError("outbound_spool_unsequenced", self._name)
        sequence_set = set(sequences)
        if not sequence_set:
            return
        self._rewrite_retaining(
            lambda decoded: (
                not (
                    isinstance(decoded, dict)
                    and decoded.get("sequence") in sequence_set
                )
            )
        )

    def drop(self, sequence: int) -> None:
        """Single-sequence convenience wrapper over `drop_many` -- see its docstring for why this
        no longer touches the watermark."""
        self.drop_many((sequence,))


def _poison_after_fork() -> None:
    """N25: `os.fork()` inherits both `OutboundSpool._registry` (with a live `_next_sequence`) and
    every instance's flock fd (the SAME open file description, so the lock is merely shared, not
    contended, across parent and child) -- without this, parent and child would allocate identical
    sequence numbers with no complaint. Marks every currently-registered instance so its next
    mutating call raises instead. Not reachable via `subprocess` (fork+exec resets memory); this
    guards a bare `os.fork()` specifically."""
    with OutboundSpool._registry_lock:
        for instance in OutboundSpool._registry.values():
            instance._forked = True


if hasattr(os, "register_at_fork"):  # POSIX-only, like fcntl (N30)
    # P7: `before=`/`after_in_parent=` pair the registry lock around the fork itself -- without
    # this, a fork occurring while some OTHER thread holds `_registry_lock` hands the child a
    # locked RLock owned by a thread that no longer exists there, and `_poison_after_fork`'s own
    # `with OutboundSpool._registry_lock:` deadlocks at the fork point instead of poisoning
    # anything. Acquiring on `before` guarantees the FORKING thread itself owns the lock at fork
    # time, so the child's single surviving thread already owns it too -- `_poison_after_fork`'s
    # acquire becomes a safe reentrant no-op there, and `after_in_parent` restores normal locking
    # in the parent.
    os.register_at_fork(
        before=OutboundSpool._registry_lock.acquire,
        after_in_parent=OutboundSpool._registry_lock.release,
        after_in_child=_poison_after_fork,
    )


# =================================================================================================
# Transport -- the HTTP boundary every channel client speaks through, and its production impl.
# =================================================================================================


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    body: dict[str, Any] | None
    headers: dict[str, str]


class TransportError(RuntimeError):
    """Raised by a `Transport.request` implementation when no HTTP response was ever received
    (connection refused, DNS failure, timeout, ...). `classify_response` treats this identically
    to an unreachable 5xx -- the guest cannot distinguish "server errored" from "server unreachable"
    and the contract's retry policy doesn't ask it to."""


class Transport(Protocol):
    """The seam every channel client is built against, so the fake-platform tests exercise the
    exact same code path production traffic does -- only what sits behind this protocol differs.
    """

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        data: bytes | Iterator[bytes] | None = None,
        timeout: float = 30.0,
    ) -> TransportResponse: ...


class RequestsTransport:
    """Production `Transport`: a thin `requests.Session` wrapper. Every network-layer failure
    (`requests.RequestException`, which covers connection errors, timeouts, and retries `requests`
    itself doesn't handle) is normalized to `TransportError` so `classify_response` never needs to
    know which HTTP library is underneath."""

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        data: bytes | Iterator[bytes] | None = None,
        timeout: float = 30.0,
    ) -> TransportResponse:
        try:
            response = self._session.request(
                method, url, headers=headers, json=json_body, data=data, timeout=timeout
            )
        except requests.RequestException as exc:
            raise TransportError(str(exc)) from exc
        try:
            body = response.json() if response.content else None
        except ValueError:
            body = None
        return TransportResponse(
            status_code=response.status_code, body=body, headers=dict(response.headers)
        )


def _iter_chunks(data: bytes, chunk_size: int) -> Iterator[bytes]:
    """§3a: "Uploads >64 MB use chunked transfer." A fresh generator is built per send attempt
    (never reused across a retry) -- a generator is single-use, and reusing an exhausted one would
    silently upload an empty body on the second attempt."""
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size]


def _parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """ "429 -> honor `Retry-After`." Only the delta-seconds form is parsed (the integer count of
    seconds to wait) -- the contract never mentions the alternative HTTP-date form and every
    platform emitter in this ecosystem is expected to send the simple form; an unparseable value is
    treated as absent so the caller falls back to the computed backoff rather than crashing.

    N5: a negative value is ALSO treated as absent -- `time.sleep(-5)` raises `ValueError`, and a
    server sending a negative `Retry-After` is malformed input this module owes no obedience to.
    The upper clamp (`[0, retry_policy.max_backoff_seconds]`) needs the policy, which isn't
    available here -- `_perform_with_retry` applies that half.

    P6: header-name lookup is case-INsensitive (RFC 9110 §5.1 -- field names are case-insensitive).
    `RequestsTransport` builds `dict(response.headers)` from `requests`' own `CaseInsensitiveDict`,
    which drops the case-insensitivity and preserves whatever casing the server actually sent -- a
    plain `.get("Retry-After")` would miss `RETRY-AFTER`/`Retry-after` and silently fall back to
    computed backoff instead of honoring the server's wait.
    """
    if not headers:
        return None
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


# =================================================================================================
# Error map -- the contract's closed status-code vocabulary ("Error responses" / "Failure semantics
# summary"), and the shared retry engine every channel client drives it through.
# =================================================================================================


class ChannelOutcome(str, Enum):
    """Every way one outbound HTTP attempt can resolve, per the contract's failure table. Not a
    contract vocabulary itself (the wire only ever carries a status code + `{error, message,
    retryable}`) -- this is this module's own closed classification of that table, the thing
    `classify_response` computes and every client branches on."""

    DELIVERED = "delivered"
    RETRYABLE = "retryable"
    FENCED = "fenced"
    PERMANENT_ITEM = "permanent_item"
    CHANNEL_FAILED = "channel_failed"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass(frozen=True)
class ChannelError:
    outcome: ChannelOutcome
    domain: FailureDomain | None
    code: str
    message: str
    retry_after_seconds: float | None = None
    # N7: the raw status this was classified from (`None` for a `TransportError`/no-response
    # outcome) -- carried so a caller can recognize a specific status (413, for the events-batch
    # halving retry) without `ChannelOutcome`/`code` alone being expressive enough for that.
    status_code: int | None = None


class HostedFencedError(RuntimeError):
    """401 (expired) / 403 (fence, scope, mismatch): "stop emitting, exit code 3 ... never an infra
    retry." Raised by the shared retry engine and never retried -- the entrypoint (outside this
    module) is the one that translates this into the process exit code."""

    def __init__(self, error: ChannelError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


class HostedChannelFailedError(RuntimeError):
    """404, retried 3x per the contract, still 404: "finalize `platform_sync`." Raised by the
    shared retry engine once `classify_response` reaches the third 404 attempt."""

    def __init__(self, error: ChannelError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


class HostedAttemptSupersededError(RuntimeError):
    """N22: `409 attempt_superseded` folded into the `ChannelState` latch below -- "a fenced
    attempt's in-flight requests cannot land after registration of its successor" is a fence in
    substance, even though the ONE request that received it is still correctly classified
    `PERMANENT_ITEM` (contract-correct: 409 is item-level, not fence-level). Only raised by
    `ChannelState.check()` on a LATER call, once a prior call has already seen this code -- the
    call that actually observed the 409 still returns its normal item-level result."""

    def __init__(self, error: ChannelError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


class ChannelState:
    """N10: shared "stop emitting" latch across the three channel clients for one attempt -- a
    fence (401/403) or an exhausted channel (404x3) on ANY channel must stop ALL of them, since
    the token/fence is per-ATTEMPT, not per-channel ("stop emitting ... never an infra retry").
    Also carries the N22 attempt-supersession latch (409 `attempt_superseded`).

    Construct ONE `ChannelState` per attempt and pass it to `EventsClient`/`ResultsClient`/
    `ArtifactsClient` alike (each defaults to a private one if not given, which only latches
    itself -- correct for a single-channel caller, but callers driving more than one channel for
    the same attempt MUST share one instance to get the cross-channel guarantee this class exists
    for). Once latched, `check()` raises the SAME error on every subsequent call, from any client
    sharing this state, without ever touching the transport.
    """

    def __init__(self) -> None:
        self._error: (
            HostedFencedError
            | HostedChannelFailedError
            | HostedAttemptSupersededError
            | None
        ) = None
        self._lock = RLock()

    def check(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error

    def latch(
        self,
        exc: "HostedFencedError | HostedChannelFailedError | HostedAttemptSupersededError",
    ) -> None:
        with self._lock:
            if self._error is None:
                self._error = exc


def classify_response(
    status_code: int | None,
    body: dict[str, Any] | None,
    *,
    attempt: int,
    retry_after_seconds: float | None = None,
) -> ChannelError | None:
    """The single call site every channel client classifies a transport outcome through. Returns
    `None` for a delivered (2xx) response, a `ChannelError` for everything else. `status_code=None`
    means `TransportError` (no response was ever received) -- classified exactly like an
    unreachable 5xx. For the 404 branch specifically, `attempt` must be the count of 404 RESPONSES
    seen so far in this call (not the overall attempt number) -- `_perform_with_retry` tracks that
    counter separately (N6) so an interleaved 5xx never shortens the 404 budget; every other branch
    ignores `attempt` entirely.

    N28: the error body's `retryable` field is deliberately never read here -- classification is
    status-keyed throughout this module (every branch below is keyed on `status_code` alone), which
    is defensible per the contract's own closed, status-code-driven failure table; if the platform
    ever marks an unexpected status `retryable` the guest disagrees silently, a known, accepted gap.

    Every branch below is transcribed directly from the contract's "Error responses" paragraph and
    "Failure semantics summary" table -- see `outbound-channels.md` v1.3 for the prose this mirrors.
    """
    if status_code is not None and 200 <= status_code < 300:
        return None

    error_code = (body or {}).get("error") if body else None
    message = (body or {}).get("message", "") if body else ""

    if status_code is None:
        return ChannelError(
            ChannelOutcome.RETRYABLE,
            FailureDomain.CONNECTIVITY,
            error_code or "network_error",
            message or "transport failure: no response received",
            status_code=None,
        )
    if status_code in (401, 403):
        return ChannelError(
            ChannelOutcome.FENCED,
            None,
            error_code or "fenced",
            message,
            status_code=status_code,
        )
    if status_code == 404:
        # N29: PLATFORM_SYNC on every 404 attempt, not just the third -- a 404 is never a
        # connectivity fault under §4.6; only the third attempt's outcome is ever surfaced to a
        # caller, but the domain should not silently disagree across attempts 1-2 vs 3.
        domain = FailureDomain.PLATFORM_SYNC
        if attempt < 3:
            return ChannelError(
                ChannelOutcome.RETRYABLE,
                domain,
                error_code or "not_found",
                message,
                status_code=404,
            )
        return ChannelError(
            ChannelOutcome.CHANNEL_FAILED,
            domain,
            error_code or "not_found",
            message,
            status_code=404,
        )
    if status_code == 413:
        # N7: 413 is Channel 3's artifact-budget code specifically -- only classify it
        # BUDGET_EXCEEDED when the body actually says so; a 413 on any other channel (e.g. an
        # events batch that simply exceeded the platform's ingress size limit) falls through to
        # the unlisted-4xx catch-all below instead of being mislabeled as a budget condition that
        # channel doesn't have.
        if error_code == "artifact_budget_exceeded":
            return ChannelError(
                ChannelOutcome.BUDGET_EXCEEDED,
                None,
                error_code,
                message,
                status_code=413,
            )
        return ChannelError(
            ChannelOutcome.PERMANENT_ITEM,
            None,
            error_code or "http_413",
            message,
            status_code=413,
        )
    if status_code == 429:
        return ChannelError(
            ChannelOutcome.RETRYABLE,
            FailureDomain.CONNECTIVITY,
            error_code or "rate_limited",
            message,
            retry_after_seconds=retry_after_seconds,
            status_code=429,
        )
    if status_code in (400, 409, 422):
        return ChannelError(
            ChannelOutcome.PERMANENT_ITEM,
            None,
            error_code or f"http_{status_code}",
            message,
            status_code=status_code,
        )
    if 500 <= status_code < 600:
        return ChannelError(
            ChannelOutcome.RETRYABLE,
            FailureDomain.CONNECTIVITY,
            error_code or "server_error",
            message,
            status_code=status_code,
        )
    if 400 <= status_code < 500:
        # "Catch-all: any unlisted 4xx is permanent for that item."
        return ChannelError(
            ChannelOutcome.PERMANENT_ITEM,
            None,
            error_code or f"http_{status_code}",
            message,
            status_code=status_code,
        )
    # N27: no other status family is contractual -- notably a 3xx, which should never occur (every
    # endpoint ends in "/" precisely so Django's POST-redirect problem never arises). Treat as
    # permanent rather than retrying an endpoint misconfiguration `max_attempts` times before
    # giving up anyway.
    return ChannelError(
        ChannelOutcome.PERMANENT_ITEM,
        None,
        error_code or f"http_{status_code}",
        message,
        status_code=status_code,
    )


def compute_backoff_seconds(
    attempt: int,
    *,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
    rng: Callable[[], float] = random.random,
) -> float:
    """ "retry with backoff (base `retry.initial_backoff_seconds`, cap `retry.max_backoff_seconds`,
    full jitter)" -- `uniform(0, min(cap, base * 2**(attempt-1)))`. `attempt` is the 1-based attempt
    that just failed. `rng` is injectable so callers (and tests) can get a deterministic value.
    """
    ceiling = min(
        max_backoff_seconds, initial_backoff_seconds * (2 ** max(0, attempt - 1))
    )
    return rng() * ceiling


@dataclass(frozen=True)
class RetryPolicy:
    """Backoff parameters, shared by all three channel clients. Field names mirror
    `job.HarnessRetryPolicy` deliberately (the contract states these come from the job's own
    `retry.initial_backoff_seconds`/`retry.max_backoff_seconds`), but this module does not import
    that class -- a client only needs two floats, and importing the full job-retry model (with its
    `retryable_domains` field, which governs WHOLE-JOB attempt retries, a distinct concept from a
    single outbound delivery's backoff) would be a coupling this module doesn't need.

    `max_attempts` bounds one `_perform_with_retry` call's own retry loop for the classes the
    contract leaves unbounded (network/5xx/429 -- "spool + backoff", no stated attempt ceiling).
    STUCK DECISION (fail-safe/reversible, contract silent): capped at a generous default (8) rather
    than looped forever, because durability already lives in the spool/idempotent-wire-design, not
    in one blocking call -- a caller that wants to keep trying simply invokes the client method
    again later (`EventsClient.flush()` is designed to be called repeatedly for exactly this
    reason). Must stay >= 3 for the 404 rule to ever reach its own `CHANNEL_FAILED` transition
    within a single call; the default comfortably clears that.
    """

    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 15.0
    max_attempts: int = 8


def _perform_with_retry(
    perform: Callable[[int], TransportResponse],
    *,
    retry_policy: RetryPolicy,
    sleep: Callable[[float], None],
    rng: Callable[[], float] = random.random,
    deadline: float | None = None,
    now: Callable[[], float] = time.monotonic,
) -> tuple[TransportResponse | None, ChannelError | None]:
    """The one retry/backoff engine all three channel clients drive their single HTTP call through.
    `perform(attempt)` makes ONE attempt (1-based); this loops it, classifies each outcome via
    `classify_response`, and:

    - returns `(response, None)` once delivered;
    - raises `HostedFencedError` immediately on 401/403 (never retried, by contract);
    - raises `HostedChannelFailedError` once a 404 reaches its third attempt;
    - returns `(response_or_None, error)` for every other terminal outcome (`PERMANENT_ITEM`,
      `BUDGET_EXCEEDED`) without retrying -- "deterministic rejections are never retried";
    - otherwise (`RETRYABLE`) sleeps -- honoring a server `Retry-After` over the computed backoff
      when present -- and tries again, up to `retry_policy.max_attempts`.

    N5/P5: `deadline` (a `time.monotonic()` value, typically the adapter's flush-window end) bounds
    ATTEMPT SCHEDULING AND SLEEPS ONLY -- no new attempt starts once `now() >= deadline`, and every
    sleep (computed backoff OR a server `Retry-After`) is clamped to whatever budget remains. It
    does NOT bound an attempt's own in-flight request: `perform`'s transport call carries its own,
    separate `timeout` (the caller's second knob -- `Transport.request`'s `timeout` parameter,
    unrelated to `deadline`), and nothing here clamps that value against the remaining deadline
    budget. A caller sizing `deadline` against a hard wall-clock guarantee for the whole call is
    sizing against a guarantee this function does not provide; only "no new attempt starts, and no
    sleep runs, once the window is gone" is guaranteed. (P5: widening `perform` to accept a clamped
    per-attempt timeout was considered and is the more complete fix, but at least one caller outside
    this module -- `hosted_entrypoint.py`'s `ScenariosClient._post`, which calls this function
    directly with its own single-argument `perform` closure -- is out of this fix's scope, so
    changing the call signature here would silently break that caller instead of fixing it. This
    docstring correction is the floor the round-3 review named for exactly that situation.)

    N6: 404 retries are counted SEPARATELY from the overall attempt number (`not_found_attempts`),
    so an interleaved 5xx (e.g. `503, 503, 404, 404, 404`) does not shorten the 404 budget -- only
    three OBSERVED 404s reach `classify_response`'s `CHANNEL_FAILED` transition, regardless of what
    else happened in between.
    """
    attempt = 0
    not_found_attempts = 0
    while True:
        attempt += 1
        if deadline is not None and now() >= deadline:
            return None, ChannelError(
                ChannelOutcome.RETRYABLE,
                FailureDomain.CONNECTIVITY,
                "deadline_exceeded",
                "the flush-window deadline elapsed before this attempt could be made",
            )
        try:
            response = perform(attempt)
        except TransportError:
            error = classify_response(None, None, attempt=attempt)
            response = None
        else:
            status = response.status_code
            if status == 404:
                not_found_attempts += 1
            error = classify_response(
                status,
                response.body,
                attempt=not_found_attempts if status == 404 else attempt,
                retry_after_seconds=_parse_retry_after(response.headers),
            )
        if error is None:
            return response, None
        if error.outcome is ChannelOutcome.FENCED:
            raise HostedFencedError(error)
        if error.outcome is ChannelOutcome.CHANNEL_FAILED:
            raise HostedChannelFailedError(error)
        if error.outcome is not ChannelOutcome.RETRYABLE:
            return response, error
        if attempt >= retry_policy.max_attempts:
            return response, error
        delay = error.retry_after_seconds
        if delay is not None:
            delay = min(
                delay, retry_policy.max_backoff_seconds
            )  # N5: clamp Retry-After
        else:
            delay = compute_backoff_seconds(
                attempt,
                initial_backoff_seconds=retry_policy.initial_backoff_seconds,
                max_backoff_seconds=retry_policy.max_backoff_seconds,
                rng=rng,
            )
        if deadline is not None:
            remaining = deadline - now()
            if remaining <= 0:
                return response, error
            delay = min(delay, remaining)
        sleep(delay)


# =================================================================================================
# Channel 1 -- Events client. Batches spooled events, advances the watermark only on confirmed
# delivery.
# =================================================================================================


@dataclass(frozen=True)
class EventsFlushResult:
    delivered_count: int
    acked_through_sequence: int | None
    rejected: list[dict[str, Any]]
    error: ChannelError | None
    # v1.3 (M7): set when the platform's `acked_through_sequence` fell outside the spool's trusted
    # range and was ignored rather than trusted -- `error` stays `None` because the HTTP delivery
    # itself succeeded; only the ack body was untrustworthy.
    ack_out_of_range: bool = False
    # N2/N3: set when a 2xx response's `acked_through_sequence` was missing, `null`, or not an
    # int -- a protocol violation distinct from "present but out of range" above. `error` stays
    # `None` for the same reason: the HTTP delivery itself succeeded.
    ack_missing: bool = False
    # P9: the SpooledRecord bodies dropped this call (a subset of `batch`, keyed by `rejected`),
    # captured BEFORE `drop_many` removes them from disk. The contract requires a rejected event's
    # payload be "written to the artifact spool (`log` kind)" -- without this, a caller has no way
    # to recover that payload at all once `flush()` returns, since the cap applied inside `flush()`
    # makes it impossible to reliably re-derive which spooled records were even in this batch.
    dropped_records: list[SpooledRecord] = field(default_factory=list)


_EVENTS_BATCH_PREFIX = (
    b'{"schema_version":"' + EVENT_SCHEMA_VERSION.encode("utf-8") + b'","events":['
)
_EVENTS_BATCH_SUFFIX = b"]}"


def _encode_events_batch(records: list[SpooledRecord]) -> bytes:
    """N19: the contract says "serialize once, spool the bytes, re-send verbatim; never
    re-serialize on retry" -- for the events BATCH ENVELOPE itself, not just each event inside it.
    Handing a decoded dict to `json_body=` (the previous shape) let `requests` re-serialize the
    envelope with its own settings on every send; this instead concatenates the exact bytes
    `OutboundSpool.append` already wrote for each event, closing the deviation rather than merely
    documenting it. Safe because `EVENT_SCHEMA_VERSION` is a fixed ASCII constant with no bytes
    needing escape."""
    return (
        _EVENTS_BATCH_PREFIX
        + b",".join(record.body for record in records)
        + _EVENTS_BATCH_SUFFIX
    )


class EventsClient:
    """Delivers `OutboundSpool`-backed Channel 1 events to `endpoints.events`. One `flush()` call
    sends one batch (<= `EVENTS_MAX_BATCH` events, <= `EVENTS_MAX_BATCH_BYTES`) of everything
    spooled since the last confirmed watermark, in spool order
    (`OutboundSpool.pending_since_watermark`, which reads the log in append/sequence order -- the
    platform's own ordering rule, "`(attempt_number, sequence)`", so a single-attempt process
    satisfies it for free). Re-sends spooled bytes verbatim (N19, `_encode_events_batch`) -- never
    recomputing an event's own `digest`, so "serialize once ... never re-serialize on retry" holds
    for the one thing that must never drift (the per-event digest, embedded as data).

    `flush()` is meant to be called repeatedly (by whatever background loop owns the call cadence,
    a scheduler concern outside this module) -- each call is a complete, self-contained delivery
    attempt (with its own internal retry/backoff via `_perform_with_retry`) that advances the
    watermark exactly as far as the platform confirmed and leaves everything else spooled for the
    next call.

    The ack body is UNTRUSTED PLATFORM INPUT end to end (v1.3): `acked_through_sequence` goes
    through the spool's own M7 clamp (`advance_watermark`); `rejected[]` is filtered to sequences
    actually present in the batch just sent BEFORE anything is done with it (N1) -- a value the
    guest never sent cannot cause a drop, and dropping never itself advances the watermark (see
    `OutboundSpool.drop_many`) -- so the batch-level `advance_watermark(acked_through_sequence)` is
    the single chokepoint either way.
    """

    def __init__(
        self,
        capabilities: HostedCapabilities,
        spool: OutboundSpool,
        transport: Transport | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
        batch_size: int = EVENTS_MAX_BATCH,
        channel_state: ChannelState | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._spool = spool
        self._transport = transport or RequestsTransport()
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._rng = rng
        self._batch_size = max(1, min(batch_size, EVENTS_MAX_BATCH))
        self._channel_state = channel_state or ChannelState()

    def _cap_batch(self, records: list[SpooledRecord]) -> list[SpooledRecord]:
        """Proactive half of N7: cap by event count (`_batch_size`) AND cumulative canonical bytes
        (`EVENTS_MAX_BATCH_BYTES`) before ever building a request -- reduces how often the reactive
        413-halving in `flush()` below is ever needed. Always includes at least one record (its own
        oversized payload is HostedEventDraft's problem, at spool-append time, not this cap's)."""
        capped = records[: self._batch_size]
        limited: list[SpooledRecord] = []
        total_bytes = 0
        for record in capped:
            if limited and total_bytes + len(record.body) > EVENTS_MAX_BATCH_BYTES:
                break
            limited.append(record)
            total_bytes += len(record.body)
        return limited

    def flush(self, *, deadline: float | None = None) -> EventsFlushResult:
        self._channel_state.check()
        batch = self._cap_batch(self._spool.pending_since_watermark())
        if not batch:
            return EventsFlushResult(
                delivered_count=0,
                acked_through_sequence=self._spool.watermark(),
                rejected=[],
                error=None,
            )

        response: TransportResponse | None = None
        error: ChannelError | None = None
        while True:
            body_bytes = _encode_events_batch(batch)
            headers = {
                **self._capabilities.auth_headers(),
                "Content-Type": "application/json",
            }

            def perform(_attempt: int) -> TransportResponse:
                return self._transport.request(
                    "POST",
                    self._capabilities.endpoints.events,
                    headers=headers,
                    data=body_bytes,
                )

            try:
                response, error = _perform_with_retry(
                    perform,
                    retry_policy=self._retry_policy,
                    sleep=self._sleep,
                    rng=self._rng,
                    deadline=deadline,
                )
            except (HostedFencedError, HostedChannelFailedError) as exc:
                self._channel_state.latch(exc)
                raise
            # N7: a 413 on the events channel -- reactively halve the batch and try again rather
            # than returning a permanent, non-progressing error for the whole thing. Stops once a
            # single event alone still 413s (defensive; should not happen under EVENT_PAYLOAD_MAX_BYTES).
            if error is not None and error.status_code == 413 and len(batch) > 1:
                logger.warning(
                    "events flush: batch of %d events (%d bytes) was rejected with 413 -- halving "
                    "and retrying",
                    len(batch),
                    len(body_bytes),
                )
                batch = batch[: len(batch) // 2]
                continue
            break

        if error is not None or response is None:
            return EventsFlushResult(
                delivered_count=0, acked_through_sequence=None, rejected=[], error=error
            )

        # N2/N3: the ack body is untrusted platform input, defensively parsed -- a wrong type or a
        # missing key must never raise out of flush() (that would silence the flusher loop, B1's
        # outcome by a different route) and must never be treated as "0" (that would silently
        # re-send the same batch forever, N3).
        body = response.body if isinstance(response.body, dict) else {}
        raw_acked = body.get("acked_through_sequence")
        acked_through: int | None
        if isinstance(raw_acked, int) and not isinstance(raw_acked, bool):
            acked_through = raw_acked
        else:
            acked_through = None
            logger.warning(
                "events flush: 2xx response has a missing/invalid acked_through_sequence (got %r) "
                "-- treating as a protocol violation, not advancing the watermark",
                raw_acked,
            )

        raw_rejected = body.get("rejected")
        if isinstance(raw_rejected, list):
            rejected_entries = [
                entry for entry in raw_rejected if isinstance(entry, dict)
            ]
            if len(rejected_entries) != len(raw_rejected):
                logger.warning(
                    "events flush: rejected[] contained non-object entries -- ignoring them"
                )
        else:
            rejected_entries = []
            if raw_rejected is not None:
                logger.warning(
                    "events flush: rejected is %r, not a list -- treating as empty",
                    type(raw_rejected).__name__,
                )

        if acked_through is None:
            return EventsFlushResult(
                delivered_count=0,
                acked_through_sequence=self._spool.watermark(),
                rejected=[],
                error=None,
                ack_missing=True,
            )

        # N1: filter rejected[] to sequences the guest ACTUALLY sent in this batch, before doing
        # anything with them -- a sequence the platform names that was never in `batch` is
        # untrusted input this module owes no obedience to (it cannot be dropped, since it was
        # never spooled under that number in the first place, and trusting it would let a
        # malformed ack orphan pending records by a route the M7 clamp doesn't guard).
        batch_sequences = {
            record.sequence for record in batch if record.sequence is not None
        }
        valid_rejected: list[dict[str, Any]] = []
        for entry in rejected_entries:
            sequence = entry.get("sequence")
            if (
                isinstance(sequence, int)
                and not isinstance(sequence, bool)
                and sequence in batch_sequences
            ):
                valid_rejected.append(entry)
            else:
                logger.warning(
                    "events flush: rejected entry names sequence=%r, which was not sent in this "
                    "batch (sent=%s) -- ignoring as untrusted platform input",
                    sequence,
                    sorted(batch_sequences),
                )

        # P9: capture the dropped records' own bodies BEFORE drop_many physically removes them --
        # once removed, this is the only place a caller can still recover the payload the contract
        # requires be "written to the artifact spool (`log` kind)" for a rejected event.
        rejected_sequences = {entry["sequence"] for entry in valid_rejected}
        dropped_records = [
            record for record in batch if record.sequence in rejected_sequences
        ]

        # N1/N14: pure physical removal, batched into one rewrite -- drop_many never touches the
        # watermark; the batch-level advance_watermark(acked_through) below is the ONLY chokepoint.
        self._spool.drop_many(rejected_sequences)

        try:
            # "the watermark is highest-processed -- accepted AND rejected sequences both advance
            # it." v1.3: `acked_through_sequence` is untrusted input -- the spool itself enforces
            # the clamp (M7).
            self._spool.advance_watermark(acked_through)
        except OutboundSpoolError:
            logger.warning(
                "events flush: platform returned an untrusted acked_through_sequence=%s outside "
                "the guest's trusted range -- ignoring the ack, watermark unchanged at %s",
                acked_through,
                self._spool.watermark(),
            )
            return EventsFlushResult(
                delivered_count=0,
                acked_through_sequence=self._spool.watermark(),
                rejected=[],
                error=None,
                ack_out_of_range=True,
                dropped_records=dropped_records,
            )

        delivered_count = sum(
            1
            for record in batch
            if record.sequence is not None
            and record.sequence <= acked_through
            and record.sequence not in rejected_sequences
        )
        return EventsFlushResult(
            delivered_count=delivered_count,
            acked_through_sequence=acked_through,
            rejected=valid_rejected,
            error=None,
            dropped_records=dropped_records,
        )


# =================================================================================================
# Channel 2 -- Result receipts. Typed `ResultReceiptDraft` + delivery.
# =================================================================================================


class ScenarioStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"
    SKIPPED = "skipped"


class SubGoalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    held: bool | None
    reason: str | None
    judged: bool


class MetricEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["metric"]
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class CheckpointEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["checkpoint"]
    passed: bool
    reason: str


EvaluationResult = MetricEvaluation | CheckpointEvaluation


class CallSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    started_at: str
    ended_at: str
    duration_ms: int = Field(ge=0)
    turns: int = Field(ge=0)
    transcript_artifact: str | None
    recording_artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "CallSummary":
        for label, value in (
            ("started_at", self.started_at),
            ("ended_at", self.ended_at),
        ):
            if not is_valid_rfc3339_millis(value):
                raise ValueError(f"call_timestamp_invalid: {label}={value!r}")
        if self.transcript_artifact is not None and not is_valid_digest(
            self.transcript_artifact
        ):
            raise ValueError(
                f"call_transcript_artifact_invalid: {self.transcript_artifact!r}"
            )
        for artifact in self.recording_artifacts:
            if not is_valid_digest(artifact):
                raise ValueError(f"call_recording_artifact_invalid: {artifact!r}")
        return self


def _unset_default_fields(model: BaseModel, prefix: str = "") -> list[str]:
    """N23: names the fields a caller did NOT explicitly set (filled from a pydantic default) --
    the actual shape of a digest-mismatch bug like the review's example: a raw `call` dict that
    omits `recording_artifacts` computes an external digest over an object without that key, while
    the model's own re-derivation fills in `recording_artifacts: []`. This is not a general diff
    against the caller's original raw dict (a model validator has no access to that, only to what
    pydantic recorded via `model_fields_set`) -- it is the honest, dotted-path subset available from
    inside the model: which fields with defaults were left unset, one level into nested models
    (covers `call.recording_artifacts`, not just top-level `world_index`/`schema_version`)."""
    names: list[str] = []
    for name in type(model).model_fields:
        if name == "digest":
            continue
        path = f"{prefix}{name}"
        if name not in model.model_fields_set:
            names.append(path)
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            names.extend(_unset_default_fields(value, f"{path}."))
    return names


class ResultReceiptDraft(BaseModel):
    """Channel 2's wire shape. Mirrors `HostedEventDraft`'s pattern: the caller supplies `digest`
    (via `build_result_receipt`, computed with `whole_object_digest`), the model re-derives and
    rejects a mismatch, plus the two exact-shape rules the contract states as literal requirements
    rather than general validation ("`skipped` receipt body (exact)" and "`errored` receipt body").
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = RESULT_SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    attempt_number: int = Field(ge=1)
    scenario_key: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_attempt: Literal[1, 2]
    world_index: int | None = Field(default=None, ge=0)
    status: ScenarioStatus
    sub_goals: list[SubGoalResult]
    evaluations: list[EvaluationResult]
    call: CallSummary | None
    failure: TerminalFailure | None
    digest: str

    @model_validator(mode="after")
    def _validate(self) -> "ResultReceiptDraft":
        if self.schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(f"result_schema_unsupported: {self.schema_version}")
        if not is_valid_digest(self.digest):
            raise ValueError(f"receipt_digest_invalid: {self.digest!r}")
        expected = whole_object_digest(self.model_dump(mode="json", exclude={"digest"}))
        if self.digest != expected:
            unset = _unset_default_fields(self)
            hint = (
                f" -- fields not explicitly set, filled from defaults: {', '.join(unset)}"
                if unset
                else ""
            )
            raise ValueError(f"receipt_digest_mismatch{hint}")

        if self.status is ScenarioStatus.SKIPPED:
            if (
                self.scenario_attempt != 1
                or self.world_index is not None
                or self.sub_goals
                or self.evaluations
                or self.call is not None
                or self.failure is not None
            ):
                raise ValueError("skipped_receipt_shape_invalid")
        elif self.status is ScenarioStatus.ERRORED and self.failure is None:
            raise ValueError("errored_receipt_requires_failure")
        return self


def build_result_receipt(
    *,
    job_id: str,
    attempt_id: str,
    attempt_number: int,
    scenario_key: str,
    scenario_id: str,
    scenario_attempt: Literal[1, 2],
    world_index: int | None,
    status: ScenarioStatus,
    sub_goals: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    call: dict[str, Any] | None,
    failure: dict[str, Any] | None,
    extra_secret_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Validate one receipt's shape and compute its digest, returning a plain wire-ready dict.
    Mirrors `build_event_record`'s contract exactly: callers must pass already wire-typed values
    (e.g. a metric `score` as `float`, never `int`) -- the digest is computed on the RAW input
    before model validation/coercion, so a type looseness here fails loudly as a digest mismatch
    rather than silently spooling a digest that doesn't match what gets sent.

    N9: `redact_outbound_text` runs on `sub_goals[].reason`, `evaluations[].reason`, and
    `failure.{code,message}` (P8) BEFORE the digest is computed -- same ordering rationale as
    `build_event_record`: the embedded digest must match the redacted bytes actually sent.
    """

    def _redact(text: Any) -> Any:
        return (
            redact_outbound_text(text, extra_secret_values)
            if isinstance(text, str)
            else text
        )

    sub_goals = [
        {**goal, "reason": _redact(goal.get("reason"))}
        if isinstance(goal, dict)
        else goal
        for goal in sub_goals
    ]
    evaluations = [
        {**item, "reason": _redact(item.get("reason"))}
        if isinstance(item, dict)
        else item
        for item in evaluations
    ]
    if isinstance(failure, dict):
        updates = {
            key: _redact(failure[key])
            for key in ("code", "message")
            if isinstance(failure.get(key), str)
        }
        if updates:
            failure = {**failure, **updates}

    core: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "scenario_key": scenario_key,
        "scenario_id": scenario_id,
        "scenario_attempt": scenario_attempt,
        "world_index": world_index,
        "status": status.value if isinstance(status, ScenarioStatus) else status,
        "sub_goals": sub_goals,
        "evaluations": evaluations,
        "call": call,
        "failure": failure,
    }
    digest = whole_object_digest(core)
    draft = ResultReceiptDraft.model_validate({**core, "digest": digest})
    return draft.model_dump(mode="json")


def build_skipped_receipt(
    *,
    job_id: str,
    attempt_id: str,
    attempt_number: int,
    scenario_key: str,
    scenario_id: str,
) -> dict[str, Any]:
    """The "exact" synthesized body for a scenario that never ran ("The guest synthesizes these
    during the flush window; the finalizer backfills any still missing")."""
    return build_result_receipt(
        job_id=job_id,
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        scenario_key=scenario_key,
        scenario_id=scenario_id,
        scenario_attempt=1,
        world_index=None,
        status=ScenarioStatus.SKIPPED,
        sub_goals=[],
        evaluations=[],
        call=None,
        failure=None,
    )


@dataclass(frozen=True)
class ReceiptPushResult:
    delivered: bool
    already_existed: bool
    error: ChannelError | None


class ResultsClient:
    """Delivers one Channel 2 receipt per call to `endpoints.results`. No spool/watermark of its
    own -- unlike events, receipts carry no `sequence`; their idempotency key is `(job_id,
    scenario_key)` (contract), so redelivery safety comes from the wire protocol itself (`200`
    duplicate on a matching digest) rather than from a local ack cursor. A caller that wants
    durable at-least-once delivery across a process crash owns that queuing (e.g. an
    `OutboundSpool(sequenced=False)`, exposed by this module for exactly this) and simply calls
    `push()` again for anything not yet confirmed -- safe because the platform's own idempotency
    check is what makes a redelivery a no-op, not any state this client keeps.
    """

    def __init__(
        self,
        capabilities: HostedCapabilities,
        transport: Transport | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
        channel_state: ChannelState | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._transport = transport or RequestsTransport()
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._rng = rng
        self._channel_state = channel_state or ChannelState()

    def push(
        self, receipt: dict[str, Any], *, deadline: float | None = None
    ) -> ReceiptPushResult:
        self._channel_state.check()

        def perform(_attempt: int) -> TransportResponse:
            return self._transport.request(
                "POST",
                self._capabilities.endpoints.results,
                headers=self._capabilities.auth_headers(),
                json_body=receipt,
            )

        try:
            response, error = _perform_with_retry(
                perform,
                retry_policy=self._retry_policy,
                sleep=self._sleep,
                rng=self._rng,
                deadline=deadline,
            )
        except (HostedFencedError, HostedChannelFailedError) as exc:
            self._channel_state.latch(exc)
            raise
        if error is not None and error.code == "attempt_superseded":  # N22
            self._channel_state.latch(HostedAttemptSupersededError(error))
        if error is not None or response is None:
            return ReceiptPushResult(
                delivered=False, already_existed=False, error=error
            )
        # N20: `200` is read as "already exists / duplicate" per the contract's idempotency rule;
        # the contract never states the success code for a genuinely NEW receipt (this module's own
        # `FakePlatform` uses `201`, unconfirmed against the real platform -- see the review report).
        return ReceiptPushResult(
            delivered=True, already_existed=response.status_code == 200, error=None
        )


# =================================================================================================
# Channel 3 -- Artifacts. Content-addressed upload + `ArtifactManifestDraft` + delivery.
# =================================================================================================


class ArtifactKind(str, Enum):
    RECORDING_COMBINED = "recording_combined"
    RECORDING_STEREO = "recording_stereo"
    RECORDING_CUSTOMER = "recording_customer"
    RECORDING_ASSISTANT = "recording_assistant"
    TRANSCRIPT = "transcript"
    TOOL_TRACE = "tool_trace"
    RESULT = "result"
    BUILD = "build"
    TRACE = "trace"
    LOG = "log"
    OTHER = "other"


_RESERVED_ARTIFACT_KINDS = frozenset(
    {
        ArtifactKind.BUILD,
        ArtifactKind.TRANSCRIPT,
        ArtifactKind.TOOL_TRACE,
        ArtifactKind.RESULT,
    }
)


def is_reserved_artifact_kind(kind: ArtifactKind) -> bool:
    """ "the budget is partitioned by reservation: `build` + `transcript` + `tool_trace` + `result`
    are reserved (always admitted); recordings next; `trace`/`log`/`other` last.\""""
    return kind in _RESERVED_ARTIFACT_KINDS


_RECORDING_ARTIFACT_KINDS = frozenset(
    {
        ArtifactKind.RECORDING_COMBINED,
        ArtifactKind.RECORDING_STEREO,
        ArtifactKind.RECORDING_CUSTOMER,
        ArtifactKind.RECORDING_ASSISTANT,
    }
)


def priority_class(kind: ArtifactKind) -> int:
    """N16: the contract's three-tier budget partition as a total order, lower = admitted first --
    `0` reserved (`is_reserved_artifact_kind`, always admitted), `1` recordings, `2` `trace`/`log`/
    `other` (admitted last)."""
    if is_reserved_artifact_kind(kind):
        return 0
    if kind in _RECORDING_ARTIFACT_KINDS:
        return 1
    return 2


class ArtifactBudgetTracker:
    """Client-side mirror of "budget = upload admission ... the guest enforces it first": a
    per-job cumulative cap across attempts, deduplicated by digest. `would_admit` is a pure check a
    caller makes before calling `ArtifactsClient.upload` for a non-reserved kind; when it returns
    `False` the upload is skipped (and named in a `log` event -- the caller's job, not this
    tracker's). This class does not sequence "refused newest-first" itself -- it has no visibility
    into candidate ordering across scenarios, which only the scheduler (P9) has; it supplies the
    admission arithmetic that policy is built on.

    `recording_headroom_bytes` (N16, default 0 -- no behavior change unless a caller opts in):
    bytes of the remaining budget reserved for recordings not yet seen, subtracted from what a
    `trace`/`log`/`other` (priority class 2) candidate is allowed to consume. This tracker has no
    visibility into how many recording bytes are still coming (only the scheduler does), so it
    cannot give a perfect answer -- reserving a caller-supplied headroom is the honest, testable
    subset of "recordings next; trace/log/other last" this class alone can enforce.
    """

    def __init__(
        self, max_artifact_bytes: int, *, recording_headroom_bytes: int = 0
    ) -> None:
        self._max_bytes = max_artifact_bytes
        self._admitted_bytes = 0
        self._seen_digests: set[str] = set()
        self._recording_headroom_bytes = recording_headroom_bytes

    def would_admit(self, kind: ArtifactKind, size: int, *, digest: str) -> bool:
        if digest in self._seen_digests:
            return True  # already counted; a duplicate upload never grows the budget further
        tier = priority_class(kind)
        if tier == 0:
            return True
        remaining = self._max_bytes - self._admitted_bytes
        if tier == 2:
            remaining -= self._recording_headroom_bytes
        return size <= remaining

    def record(self, kind: ArtifactKind, size: int, *, digest: str) -> None:
        del kind  # reservation already resolved by would_admit; recorded uniformly here
        if digest in self._seen_digests:
            return
        self._seen_digests.add(digest)
        self._admitted_bytes += size

    @property
    def admitted_bytes(self) -> int:
        return self._admitted_bytes


class ArtifactManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: ArtifactKind
    size: int = Field(ge=0)
    scenario_key: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ArtifactManifestEntry":
        if not is_valid_digest(self.artifact_id):
            raise ValueError(f"artifact_id_invalid: {self.artifact_id!r}")
        return self


class ArtifactManifestDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = MANIFEST_SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    attempt_number: int = Field(ge=1)
    entries: list[ArtifactManifestEntry]
    complete: bool
    digest: str

    @model_validator(mode="after")
    def _validate(self) -> "ArtifactManifestDraft":
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"manifest_schema_unsupported: {self.schema_version}")
        if not is_valid_digest(self.digest):
            raise ValueError(f"manifest_digest_invalid: {self.digest!r}")
        expected = whole_object_digest(self.model_dump(mode="json", exclude={"digest"}))
        if self.digest != expected:
            unset = _unset_default_fields(self)
            hint = (
                f" -- fields not explicitly set, filled from defaults: {', '.join(unset)}"
                if unset
                else ""
            )
            raise ValueError(f"manifest_digest_mismatch{hint}")
        return self


def build_artifact_manifest(
    *,
    job_id: str,
    attempt_id: str,
    attempt_number: int,
    entries: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    """Same pattern as `build_result_receipt`/`build_event_record`: digest computed on the raw
    input, then re-verified by the model that consumes it."""
    core: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "entries": entries,
        "complete": complete,
    }
    digest = whole_object_digest(core)
    draft = ArtifactManifestDraft.model_validate({**core, "digest": digest})
    return draft.model_dump(mode="json")


@dataclass(frozen=True)
class ArtifactUploadResult:
    delivered: bool
    already_existed: bool
    error: ChannelError | None


@dataclass(frozen=True)
class ManifestPushResult:
    delivered: bool
    already_existed: bool
    error: ChannelError | None


_DEFAULT_ARTIFACT_CONTENT_TYPES: dict[ArtifactKind, str] = {
    ArtifactKind.RECORDING_COMBINED: "video/mp4",
    ArtifactKind.RECORDING_STEREO: "video/mp4",
    ArtifactKind.RECORDING_CUSTOMER: "video/mp4",
    ArtifactKind.RECORDING_ASSISTANT: "video/mp4",
    ArtifactKind.TRANSCRIPT: "application/json",
    ArtifactKind.RESULT: "application/json",
}


def _default_content_type(kind: ArtifactKind) -> str:
    """N17: §3a pins recordings to mp4 and `transcript` to a JSON array; a platform serving these
    back to a UI needs an accurate `Content-Type`, not a blanket octet-stream."""
    return _DEFAULT_ARTIFACT_CONTENT_TYPES.get(kind, "application/octet-stream")


def _artifact_content_type(kind: ArtifactKind, data: bytes) -> str:
    """Return the wire MIME type, preferring the bytes over the nominal format.

    Hosted voice engines currently materialize RIFF/WAVE recordings even though
    the v1.4 artifact contract's preferred recording format is MP4.  Advertising
    those bytes as ``video/mp4`` makes browsers reject an otherwise valid audio
    artifact.  Keep the contractual default for opaque/test payloads, but sniff
    the two recording formats we actually support before uploading.
    """
    if kind in {
        ArtifactKind.RECORDING_COMBINED,
        ArtifactKind.RECORDING_STEREO,
        ArtifactKind.RECORDING_CUSTOMER,
        ArtifactKind.RECORDING_ASSISTANT,
    }:
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return "audio/wav"
        if len(data) >= 12 and data[4:8] == b"ftyp":
            return "video/mp4"
    return _default_content_type(kind)


class ArtifactsClient:
    """Content-addressed upload (§3a) + manifest delivery (§3b) to `endpoints.artifacts`.

    `upload` verifies the given bytes actually hash to the claimed `artifact_id` BEFORE ever
    calling the transport -- a local, zero-cost check that catches a caller bug (wrong id, wrong
    bytes) without spending a round trip on it; the platform's own `422 digest_mismatch` remains
    the authority for anything this local check cannot see (partial reads, transport corruption).
    On a `422 digest_mismatch` FROM THE PLATFORM specifically, the contract grants exactly one
    extra whole-upload retry ("re-upload once, then the referencing scenario is `errored`") --
    distinct from `_perform_with_retry`'s own loop, which treats every 422 as `PERMANENT_ITEM` and
    never retries it; this method wraps that loop in one more, narrower retry layer that fires only
    for that one code.

    Size accounting: `X-Artifact-Size` is never a caller-supplied value -- it is always derived
    from `len(data)`, the same bytes actually transmitted, so a `422 size_mismatch` against what
    this client sends is structurally unreachable from here (the platform's own count remains the
    authority for what actually arrived over the wire).

    N18: once a 413 `artifact_budget_exceeded` is observed, this instance latches locally -- every
    later `upload()` for a NON-reserved kind is refused without contacting the platform at all
    ("stop uploading non-reserved kinds, log, continue the run"); reserved kinds keep uploading
    (they are always admitted, budget or not).
    """

    def __init__(
        self,
        capabilities: HostedCapabilities,
        transport: Transport | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
        chunk_threshold_bytes: int = ARTIFACT_CHUNKED_UPLOAD_THRESHOLD_BYTES,
        chunk_size_bytes: int = 8 * 1024 * 1024,
        channel_state: ChannelState | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._transport = transport or RequestsTransport()
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._rng = rng
        self._chunk_threshold_bytes = chunk_threshold_bytes
        self._chunk_size_bytes = chunk_size_bytes
        self._channel_state = channel_state or ChannelState()
        self._budget_exhausted = False  # N18

    def upload(
        self,
        artifact_id_hex: str,
        data: bytes,
        *,
        kind: ArtifactKind,
        scenario_key: str | None = None,
        content_type: str | None = None,
        deadline: float | None = None,
    ) -> ArtifactUploadResult:
        self._channel_state.check()
        if self._budget_exhausted and not is_reserved_artifact_kind(kind):
            logger.warning(
                "artifacts upload: budget already exhausted (413 artifact_budget_exceeded observed "
                "earlier this attempt) -- skipping non-reserved kind=%s without contacting the "
                "platform",
                kind.value,
            )
            return ArtifactUploadResult(
                delivered=False,
                already_existed=False,
                error=ChannelError(
                    ChannelOutcome.BUDGET_EXCEEDED,
                    None,
                    "artifact_budget_exceeded",
                    "budget already exhausted for this attempt (latched locally)",
                    status_code=None,
                ),
            )
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_id_hex):
            raise ValueError(f"artifact_id_invalid: {artifact_id_hex!r}")
        if artifact_id_hex == "manifest":
            # Unreachable through the hex check above (`manifest` isn't 64 hex chars) but the
            # contract calls this collision out by name ("`manifest` is a reserved segment").
            raise ValueError("artifact_id_reserved: manifest")
        computed = hashlib.sha256(data).hexdigest()
        if computed != artifact_id_hex:
            raise ValueError(
                f"artifact_digest_mismatch_local: expected {artifact_id_hex}, computed {computed}"
            )

        url = f"{self._capabilities.endpoints.artifacts}{artifact_id_hex}/"
        size = len(data)
        headers = {
            **self._capabilities.auth_headers(),
            "X-Artifact-Kind": kind.value,
            "X-Artifact-Size": str(size),
            "Content-Type": content_type or _artifact_content_type(kind, data),
        }
        if scenario_key is not None:
            headers["X-Scenario-Key"] = scenario_key

        def perform(_attempt: int) -> TransportResponse:
            # A fresh generator per attempt when chunked -- see `_iter_chunks`.
            body: bytes | Iterator[bytes] = (
                _iter_chunks(data, self._chunk_size_bytes)
                if size > self._chunk_threshold_bytes
                else data
            )
            return self._transport.request("PUT", url, headers=headers, data=body)

        response: TransportResponse | None = None
        error: ChannelError | None = None
        for outer_attempt in range(
            2
        ):  # "re-upload once" on a platform-confirmed digest mismatch
            try:
                response, error = _perform_with_retry(
                    perform,
                    retry_policy=self._retry_policy,
                    sleep=self._sleep,
                    rng=self._rng,
                    deadline=deadline,
                )
            except (HostedFencedError, HostedChannelFailedError) as exc:
                self._channel_state.latch(exc)
                raise
            if not (
                error is not None
                and error.code == "digest_mismatch"
                and outer_attempt == 0
            ):
                break

        if error is not None and error.outcome is ChannelOutcome.BUDGET_EXCEEDED:
            self._budget_exhausted = True  # N18

        if error is not None or response is None:
            return ArtifactUploadResult(
                delivered=False, already_existed=False, error=error
            )
        # N20: `200` is read as "already exists" per the contract's content-addressed upload
        # semantics; the success code for a genuinely NEW upload is `201` (this module's own
        # `FakePlatform` matches that but it is unconfirmed against the real platform).
        return ArtifactUploadResult(
            delivered=True, already_existed=response.status_code == 200, error=None
        )

    def push_manifest(
        self, manifest: dict[str, Any], *, deadline: float | None = None
    ) -> ManifestPushResult:
        self._channel_state.check()
        url = f"{self._capabilities.endpoints.artifacts}manifest/"

        def perform(_attempt: int) -> TransportResponse:
            return self._transport.request(
                "POST",
                url,
                headers=self._capabilities.auth_headers(),
                json_body=manifest,
            )

        try:
            response, error = _perform_with_retry(
                perform,
                retry_policy=self._retry_policy,
                sleep=self._sleep,
                rng=self._rng,
                deadline=deadline,
            )
        except (HostedFencedError, HostedChannelFailedError) as exc:
            self._channel_state.latch(exc)
            raise
        if error is not None and error.code == "attempt_superseded":  # N22
            self._channel_state.latch(HostedAttemptSupersededError(error))
        if error is not None or response is None:
            return ManifestPushResult(
                delivered=False, already_existed=False, error=error
            )
        # N20: same caveat as receipts/uploads above -- `200` == duplicate is contract-stated,
        # the new-manifest success code is `201` per `FakePlatform`, unconfirmed against the real
        # platform.
        return ManifestPushResult(
            delivered=True, already_existed=response.status_code == 200, error=None
        )


__all__ = [
    "ARTIFACT_CHUNKED_UPLOAD_THRESHOLD_BYTES",
    "CAPABILITIES_PATH",
    "CAPABILITIES_SCHEMA_VERSION",
    "EVENTS_MAX_BATCH",
    "EVENTS_MAX_BATCH_BYTES",
    "EVENT_PAYLOAD_MAX_BYTES",
    "EVENT_SCHEMA_VERSION",
    "FLUSH_WINDOW_SECONDS",
    "MANIFEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "ArtifactBudgetTracker",
    "ArtifactKind",
    "ArtifactManifestDraft",
    "ArtifactManifestEntry",
    "ArtifactUploadResult",
    "ArtifactsClient",
    "BaselineFrozenPayload",
    "BaselineInputsChangedPayload",
    "CallSummary",
    "CapabilitiesError",
    "ChannelError",
    "ChannelOutcome",
    "ChannelState",
    "CheckpointEvaluation",
    "DegradeReason",
    "EvaluationResult",
    "EventsClient",
    "EventsFlushResult",
    "HostedAttemptSupersededError",
    "HostedCapabilities",
    "HostedChannelFailedError",
    "HostedEndpoints",
    "HostedEvent",
    "HostedEventDraft",
    "HostedFencedError",
    "LogLevel",
    "LogPayload",
    "ManifestPushResult",
    "MetricEvaluation",
    "OutboundError",
    "OutboundEventType",
    "OutboundSpool",
    "OutboundSpoolError",
    "ParallelismDegradedPayload",
    "ReceiptPushResult",
    "RequestsTransport",
    "ResultReceiptDraft",
    "ResultsClient",
    "RetryPolicy",
    "ScenarioCounts",
    "ScenarioRetriedPayload",
    "ScenarioStartedPayload",
    "ScenarioStatus",
    "SpooledRecord",
    "StageChangedPayload",
    "SubGoalResult",
    "TerminalFailure",
    "TerminalPayload",
    "TerminalReason",
    "Transport",
    "TransportError",
    "TransportResponse",
    "WorldUnhealthyPayload",
    "build_artifact_manifest",
    "build_event_record",
    "build_result_receipt",
    "build_skipped_receipt",
    "canonical_bytes",
    "classify_response",
    "compute_backoff_seconds",
    "event_payload_digest",
    "format_rfc3339_millis",
    "is_reserved_artifact_kind",
    "is_valid_digest",
    "is_valid_rfc3339_millis",
    "load_capabilities",
    "priority_class",
    "redact_outbound_text",
    "sha256_digest",
    "truncate_log_message",
    "whole_object_digest",
]
