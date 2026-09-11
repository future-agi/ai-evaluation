"""Outbound reporting, per `outbound-channels.md` v1.2. Two parts, matching `outbound.py`:

Foundations (part 1):

- Canonicalization: `canonical_bytes`/`event_payload_digest`/`whole_object_digest` against fixed
  expected digests, independently derived by running the contract's own literal
  ``json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`` + sha256 call in
  a bare Python shell -- not by calling back into this module -- so the vectors don't just prove
  the implementation agrees with itself.
- Capabilities: `HostedCapabilities`/`load_capabilities` against the contract's own example,
  its endpoint-trailing-slash rule, schema-version gate, and the load-then-unlink lifetime rule.
- Events + spool: `build_event_record`/`HostedEvent` shape validation for all nine closed event
  types, and `OutboundSpool`'s contiguous-from-1 sequencing, its crash/torn-tail recovery rule
  (a synthetic partial write, simulating a crash mid-append), and watermark semantics.

Transport (part 2):

- Pure functions: the full `classify_response` error map (every status the contract's "Error
  responses"/"Failure semantics summary" names, plus the unlisted-4xx catch-all), the full-jitter
  `compute_backoff_seconds` formula, `format_rfc3339_millis`, and `ArtifactBudgetTracker`.
- Typed models: `ResultReceiptDraft` (contract-shaped, the `skipped`/`errored` exact-shape rules,
  digest tamper rejection) and `ArtifactManifestDraft` (digest scope, `complete` distinguishing
  documents).
- `FakePlatform`: an in-process, no-sockets `Transport` implementation with real server-side state
  (accepted events + watermark, receipts keyed by `(job_id, scenario_key)`, artifacts keyed by
  digest, manifests keyed by `(attempt_id, digest)`) that `EventsClient`/`ResultsClient`/
  `ArtifactsClient` are driven against for: success + watermark advance, transient-then-success
  (a queued `TransportError` then a queued 5xx then real success), 429 honoring `Retry-After` over
  the computed backoff, deterministic rejection (never retried, at both the per-event and
  whole-request level), `FENCED`/`CHANNEL_FAILED` raising the two channel-ending exceptions, the
  artifact digest-mismatch "re-upload once" rule, budget refusal, chunked-transfer reassembly, and
  crash-between-send-and-ack redelivery (the platform durably records a write and then the
  response is lost -- `FakePlatform.crash_after_next_write` -- proving a redelivery of the same
  record is a safe no-op, both within one client call via its own retry loop and across two
  independent calls simulating a real process restart).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fi.alk.harness.job import FailureDomain, HarnessStage
from fi.alk.harness.outbound import (
    CAPABILITIES_SCHEMA_VERSION,
    EVENT_PAYLOAD_MAX_BYTES,
    ArtifactBudgetTracker,
    ArtifactKind,
    ArtifactManifestDraft,
    ArtifactsClient,
    CapabilitiesError,
    ChannelOutcome,
    ChannelState,
    EventsClient,
    HostedAttemptSupersededError,
    HostedCapabilities,
    HostedChannelFailedError,
    HostedEndpoints,
    HostedEvent,
    HostedEventDraft,
    HostedFencedError,
    OutboundError,
    OutboundEventType,
    OutboundSpool,
    OutboundSpoolError,
    ResultReceiptDraft,
    ResultsClient,
    RetryPolicy,
    ScenarioStatus,
    StageChangedPayload,
    TransportError,
    TransportResponse,
    build_artifact_manifest,
    build_event_record,
    build_result_receipt,
    build_skipped_receipt,
    canonical_bytes,
    classify_response,
    compute_backoff_seconds,
    event_payload_digest,
    format_rfc3339_millis,
    is_valid_digest,
    is_valid_rfc3339_millis,
    load_capabilities,
    priority_class,
    redact_outbound_text,
    sha256_digest,
    truncate_log_message,
    whole_object_digest,
)

NOW = datetime(2026, 8, 25, 10, 14, 3, 412000, tzinfo=timezone.utc)
LATER = datetime(2026, 8, 25, 10, 17, 7, 623000, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_spool_registry() -> Any:
    """M6's in-process registry caches `OutboundSpool` instances (and their open flock fds) for the
    life of the interpreter, keyed on `(resolved_root, name)`. Different tests use different
    `tmp_path` roots so there is no cross-test key collision, but without this the fds would still
    accumulate across the whole session. Runs after every test."""
    yield
    OutboundSpool._clear_registry_for_tests()


# --- Canonicalization -------------------------------------------------------------------------


def test_canonical_bytes_sorts_keys_and_uses_compact_separators() -> None:
    assert canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_canonical_bytes_does_not_escape_non_ascii() -> None:
    # ensure_ascii=False: a non-ASCII character is emitted as its raw UTF-8 bytes, not a \uXXXX
    # escape -- the contract is explicit about the exact `json.dumps` call, and this is the one
    # kwarg that differs from Python's default.
    encoded = canonical_bytes({"x": "…"})
    assert encoded == b'{"x":"\xe2\x80\xa6"}'
    assert b"\\u" not in encoded


def test_canonical_bytes_absent_and_null_are_different_bytes() -> None:
    assert canonical_bytes({}) != canonical_bytes({"x": None})
    assert canonical_bytes({"x": None}) == b'{"x":null}'


def test_event_payload_digest_matches_independently_computed_vector() -> None:
    # The Channel 1 example payload from outbound-channels.md, transcribed verbatim (including its
    # literal "…" ellipsis character, which also exercises the ensure_ascii=False rule end-to-end).
    payload = {"scenario_key": "…", "world_index": 2, "scenario_attempt": 1}
    digest = event_payload_digest(payload)
    # Computed independently via `python -c "import json,hashlib; ..."` against the contract's own
    # literal algorithm text, not by calling back into this module.
    assert digest == (
        "sha256:7a8037965edd1f5391e6d59d0556313938abbd272ca162cbf75aa1510ca7e0f1"
    )
    assert is_valid_digest(digest)


def test_event_payload_digest_scope_is_payload_only_not_the_envelope() -> None:
    payload = {"world_index": 2, "cause": "boom"}
    envelope = {
        "event_id": "event_x",
        "stage": "running",
        "type": "world_unhealthy",
        "payload": payload,
    }
    # The envelope's other fields must not affect the digest -- only `payload` is in scope.
    assert event_payload_digest(payload) == event_payload_digest(envelope["payload"])
    assert sha256_digest(canonical_bytes(payload)) == event_payload_digest(payload)


def test_whole_object_digest_scope_excludes_the_digest_key_by_popping_it() -> None:
    core = {"a": 1, "b": 2}
    with_digest = {"a": 1, "digest": "sha256:" + "f" * 64, "b": 2}
    # "the whole object with the `digest` key absent" -- popped, so the digest's own value (right
    # or wrong, present or not) can never affect the hash the platform re-derives and checks.
    assert whole_object_digest(core) == whole_object_digest(with_digest)


def test_whole_object_digest_matches_receipt_and_manifest_shaped_examples() -> None:
    # Channel 2's example receipt object, minus its own `digest` field (as if the guest is about
    # to compute it) -- proves the generic function against a receipt-shaped whole object, not
    # just the toy `{a, b}` case above.
    receipt = {
        "schema_version": "futureagi.harness-result.v1",
        "job_id": "j1",
        "attempt_id": "a1",
        "attempt_number": 1,
        "scenario_key": "suspended-account-blocked",
        "scenario_id": "sid-1",
        "scenario_attempt": 1,
        "world_index": 2,
        "status": "passed",
        "sub_goals": [{"name": "n", "held": True, "reason": None, "judged": False}],
        "evaluations": [],
        "call": None,
        "failure": None,
    }
    digest = whole_object_digest(receipt)
    assert is_valid_digest(digest)
    # FIXED expected hex (ranked missing test 2): computed independently via
    # `json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    # allow_nan=False)` + sha256 in a bare Python shell against the contract's own canonicalization
    # clause -- not by calling back into this module. This is the digest the platform ACTUALLY
    # verifies (event digests are only MAY-verified; receipt/manifest digests are verified), so a
    # self-consistency-only test (is_valid_digest, invariance to `digest`, inequality between two
    # objects) cannot catch a canonicalization regression that both this test and the module agree
    # on by construction -- a fixed vector can.
    assert (
        digest
        == "sha256:b1492bcb2f62c3d79dc4d0d1b3f5097a3784112fc00276a2dc483297375652af"
    )
    # Adding a `digest` key of any value must not change the result (scope excludes it).
    assert whole_object_digest({**receipt, "digest": "sha256:" + "0" * 64}) == digest

    # Channel 3b's manifest example, minus `digest`.
    manifest = {
        "schema_version": "futureagi.harness-manifest.v1",
        "job_id": "j1",
        "attempt_id": "a1",
        "attempt_number": 1,
        "entries": [
            {
                "artifact_id": "sha256:" + "1" * 64,
                "kind": "result",
                "size": 10,
                "scenario_key": "k",
            }
        ],
        "complete": True,
    }
    manifest_digest = whole_object_digest(manifest)
    assert is_valid_digest(manifest_digest)
    # FIXED expected hex, same independent-derivation method as above.
    assert (
        manifest_digest
        == "sha256:f18786749321d103cf30ebe11e935bf9177d5297783ee2dec00cb20ad8dae13d"
    )
    assert manifest_digest != digest


def test_canonical_bytes_rejects_nan() -> None:
    # v1.3: allow_nan=False -- NaN/Infinity are not RFC 8259 and must fail typed at the emitter
    # rather than produce bytes no strict parser downstream can read.
    with pytest.raises(OutboundError) as excinfo:
        canonical_bytes({"score": float("nan")})
    assert excinfo.value.code == "canonical_value_not_finite"


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_canonical_bytes_rejects_infinity(value: float) -> None:
    with pytest.raises(OutboundError):
        canonical_bytes({"score": value})


def test_canonical_bytes_is_still_byte_identical_for_valid_input_after_allow_nan_false() -> (
    None
):
    # "for valid input the bytes are identical" (v1.3) -- allow_nan=False changes nothing about
    # ordinary values.
    assert canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_whole_object_digest_rejects_a_non_json_native_value_naming_the_key_path() -> (
    None
):
    with pytest.raises(OutboundError) as excinfo:
        whole_object_digest({"a": {"b": datetime(2026, 1, 1, tzinfo=timezone.utc)}})
    assert excinfo.value.code == "digest_value_not_json_native"
    assert "a.b" in excinfo.value.message


def test_whole_object_digest_rejects_a_non_string_dict_key() -> None:
    with pytest.raises(OutboundError):
        whole_object_digest({"a": {1: "x"}})


# --- Capabilities ------------------------------------------------------------------------------

_ATTEMPT_ID = "22222222-2222-2222-2222-222222222222"

VALID_CAPABILITIES = {
    "schema_version": CAPABILITIES_SCHEMA_VERSION,
    "job_id": "11111111-1111-1111-1111-111111111111",
    "attempt_id": _ATTEMPT_ID,
    "attempt_number": 1,
    "fence": "opaque-fence",
    # Far future: load_capabilities rejects an expired token against wall-clock
    # time, so a near-term timestamp turns the whole fixture into a time bomb.
    "expires_at": "2099-01-01T00:00:00.000Z",
    "token": "bearer-token",
    "endpoints": {
        "events": f"https://platform.example/simulate/api/harness/attempts/{_ATTEMPT_ID}/events/",
        "results": f"https://platform.example/simulate/api/harness/attempts/{_ATTEMPT_ID}/results/",
        "artifacts": f"https://platform.example/simulate/api/harness/attempts/{_ATTEMPT_ID}/artifacts/",
        "scenarios": f"https://platform.example/simulate/api/harness/attempts/{_ATTEMPT_ID}/scenarios/",
    },
}


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_load_capabilities_parses_the_contract_example_and_builds_auth_headers(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "capabilities.json", VALID_CAPABILITIES)
    capabilities = load_capabilities(path, unlink=False)
    assert capabilities.job_id == VALID_CAPABILITIES["job_id"]
    assert capabilities.endpoints.events.endswith("/")
    assert capabilities.auth_headers() == {
        "Authorization": "Bearer bearer-token",
        "X-Harness-Fence": "opaque-fence",
    }


def test_load_capabilities_unlinks_the_file_after_a_successful_load(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "capabilities.json", VALID_CAPABILITIES)
    load_capabilities(path)  # default unlink=True
    assert not path.exists()


def test_load_capabilities_does_not_unlink_on_a_validation_failure(
    tmp_path: Path,
) -> None:
    bad = {**VALID_CAPABILITIES, "schema_version": "futureagi.harness-capabilities.v0"}
    path = _write(tmp_path / "capabilities.json", bad)
    with pytest.raises(CapabilitiesError):
        load_capabilities(path)
    # A crash mid-load must never destroy the only copy of a not-yet-consumed bearer.
    assert path.exists()


def test_load_capabilities_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CapabilitiesError) as excinfo:
        load_capabilities(tmp_path / "missing.json")
    assert excinfo.value.code == "capabilities_file_missing"


def test_load_capabilities_rejects_unreadable_file(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("permission bits don't block a root reader")
    path = _write(tmp_path / "capabilities.json", VALID_CAPABILITIES)
    path.chmod(0o000)
    try:
        with pytest.raises(CapabilitiesError) as excinfo:
            load_capabilities(path)
        assert excinfo.value.code == "capabilities_file_unreadable"
    finally:
        path.chmod(0o600)


def test_load_capabilities_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "capabilities.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CapabilitiesError) as excinfo:
        load_capabilities(path)
    assert excinfo.value.code == "capabilities_file_malformed"


def test_load_capabilities_rejects_valid_json_that_is_not_an_object(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capabilities.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(CapabilitiesError) as excinfo:
        load_capabilities(path)
    assert excinfo.value.code == "capabilities_file_malformed"


def test_load_capabilities_rejects_wrong_schema_version(tmp_path: Path) -> None:
    bad = {**VALID_CAPABILITIES, "schema_version": "futureagi.harness-capabilities.v0"}
    path = _write(tmp_path / "capabilities.json", bad)
    with pytest.raises(CapabilitiesError) as excinfo:
        load_capabilities(path)
    # Lens 5: this code used to be unreachable -- pydantic wrapped it into a ValidationError that
    # only ever surfaced as the generic capabilities_field_invalid.
    assert excinfo.value.code == "capabilities_schema_unsupported"


@pytest.mark.parametrize("channel", ["events", "results", "artifacts", "scenarios"])
def test_load_capabilities_rejects_an_endpoint_missing_its_trailing_slash(
    tmp_path: Path, channel: str
) -> None:
    bad = json.loads(json.dumps(VALID_CAPABILITIES))
    bad["endpoints"][channel] = bad["endpoints"][channel].rstrip("/")
    path = _write(tmp_path / "capabilities.json", bad)
    with pytest.raises(CapabilitiesError) as excinfo:
        load_capabilities(path)
    # Also previously unreachable as a code -- same dead-code shape as the schema check above.
    assert excinfo.value.code == "capabilities_endpoint_invalid"


@pytest.mark.parametrize("channel", ["events", "results", "artifacts", "scenarios"])
def test_load_capabilities_rejects_a_non_https_endpoint(
    tmp_path: Path, channel: str
) -> None:
    bad = json.loads(json.dumps(VALID_CAPABILITIES))
    bad["endpoints"][channel] = bad["endpoints"][channel].replace("https://", "http://")
    path = _write(tmp_path / "capabilities.json", bad)
    with pytest.raises(CapabilitiesError) as excinfo:
        load_capabilities(path)
    assert excinfo.value.code == "capabilities_endpoint_insecure"


def test_load_capabilities_rejects_an_expired_token(tmp_path: Path) -> None:
    path = _write(tmp_path / "capabilities.json", VALID_CAPABILITIES)
    with pytest.raises(CapabilitiesError) as excinfo:
        load_capabilities(path, now=lambda: datetime(2100, 1, 1, tzinfo=timezone.utc))
    assert excinfo.value.code == "capabilities_expired"
    assert (
        path.exists()
    )  # a failed load must never unlink -- same rule as any other rejection


def test_load_capabilities_accepts_a_token_not_yet_expired(tmp_path: Path) -> None:
    path = _write(tmp_path / "capabilities.json", VALID_CAPABILITIES)
    capabilities = load_capabilities(
        path, now=lambda: datetime(2020, 1, 1, tzinfo=timezone.utc)
    )
    assert capabilities.job_id == VALID_CAPABILITIES["job_id"]


@pytest.mark.parametrize("channel", ["events", "results", "artifacts", "scenarios"])
def test_load_capabilities_rejects_an_endpoint_whose_attempt_id_segment_disagrees(
    tmp_path: Path, channel: str
) -> None:
    bad = json.loads(json.dumps(VALID_CAPABILITIES))
    bad["endpoints"][channel] = bad["endpoints"][channel].replace(
        _ATTEMPT_ID, "some-other-attempt"
    )
    path = _write(tmp_path / "capabilities.json", bad)
    with pytest.raises(CapabilitiesError) as excinfo:
        load_capabilities(path)
    assert excinfo.value.code == "capabilities_attempt_mismatch"


def test_load_capabilities_field_invalid_message_never_echoes_the_input_value(
    tmp_path: Path,
) -> None:
    # MIN-3: the file carries the bearer -- a validation message built from pydantic's default
    # str(exc) would embed the failing field's raw value.
    bad = json.loads(json.dumps(VALID_CAPABILITIES))
    bad["token"] = 987654321
    path = _write(tmp_path / "capabilities.json", bad)
    with pytest.raises(CapabilitiesError) as excinfo:
        load_capabilities(path)
    assert excinfo.value.code == "capabilities_field_invalid"
    assert "987654321" not in excinfo.value.message


def test_load_capabilities_unlink_failure_is_reported_via_the_callback(
    tmp_path: Path,
) -> None:
    # MIN-7: previously swallowed unconditionally; now the caller can tell.
    path = _write(tmp_path / "capabilities.json", VALID_CAPABILITIES)
    original_unlink = Path.unlink

    def failing_unlink(self: Path, *a: Any, **kw: Any) -> None:
        if self == path:
            raise OSError("simulated unlink failure")
        return original_unlink(self, *a, **kw)

    reported: list[OSError] = []
    Path.unlink = failing_unlink  # type: ignore[method-assign]
    try:
        load_capabilities(path, on_unlink_failure=reported.append)
    finally:
        Path.unlink = original_unlink  # type: ignore[method-assign]
    assert len(reported) == 1
    assert path.exists()  # the failed unlink really did leave the file behind


def test_load_capabilities_insecure_file_mode_warns_but_does_not_block_the_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # MIN-5: fail-safe -- a wrong mode/owner is a loud warning, never a rejection.
    path = _write(tmp_path / "capabilities.json", VALID_CAPABILITIES)
    path.chmod(0o644)
    with caplog.at_level("WARNING"):
        capabilities = load_capabilities(path, unlink=False)
    assert capabilities.job_id == VALID_CAPABILITIES["job_id"]
    assert any("capabilities file mode" in record.message for record in caplog.records)


def test_hosted_endpoints_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        HostedCapabilities.model_validate({**VALID_CAPABILITIES, "unexpected": "field"})


def test_hosted_endpoints_shape_validator_still_fires_on_direct_construction() -> None:
    # Defense-in-depth: constructing the model directly (bypassing load_capabilities'
    # pre-pydantic checks entirely) must still reject a malformed endpoint.
    with pytest.raises(ValidationError):
        HostedEndpoints.model_validate(
            {
                "events": "http://x/",
                "results": "https://x/",
                "artifacts": "https://x/",
                "scenarios": "https://x/",
            }
        )


def test_capabilities_expires_at_is_serialized_as_millis_z() -> None:
    capabilities = HostedCapabilities.model_validate(
        {**VALID_CAPABILITIES, "expires_at": "2026-08-25T12:00:00.123456+00:00"}
    )
    assert (
        capabilities.model_dump(mode="json")["expires_at"] == "2026-08-25T12:00:00.123Z"
    )


def test_capabilities_event_builder_binds_identity_fields_from_the_token(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "capabilities.json", VALID_CAPABILITIES)
    capabilities = load_capabilities(path, unlink=False)
    build = capabilities.event_builder()
    record = build(
        event_id="event_" + "a" * 32,
        emitted_at=NOW,
        stage=HarnessStage.RUNNING,
        type=OutboundEventType.LOG,
        payload={"level": "info", "message": "m"},
    )
    assert record["job_id"] == capabilities.job_id
    assert record["attempt_id"] == capabilities.attempt_id
    assert record["attempt_number"] == capabilities.attempt_number


def test_capabilities_event_builder_forwards_extra_secret_values_to_log_message(
    tmp_path: Path,
) -> None:
    # P4: `event_builder()` had no way to receive the job's declared secret VALUES, so a caller
    # wired for it (the shipped adapter binds identity via `event_builder()` and routes every event
    # through it) could never satisfy N9's redaction requirement for `log.message`/
    # `terminal.failure.message` through this API -- only the userinfo scrub reached those fields.
    path = _write(tmp_path / "capabilities.json", VALID_CAPABILITIES)
    capabilities = load_capabilities(path, unlink=False)
    build = capabilities.event_builder(extra_secret_values=("s3cr3t-value",))
    record = build(
        event_id="event_" + "a" * 32,
        emitted_at=NOW,
        stage=HarnessStage.RUNNING,
        type=OutboundEventType.LOG,
        payload={
            "level": "error",
            "message": "build failed: s3cr3t-value leaked in the output",
        },
    )
    assert "s3cr3t-value" not in record["payload"]["message"]
    assert "***" in record["payload"]["message"]


# --- Events --------------------------------------------------------------------------------


def _event(type_: OutboundEventType, stage: HarnessStage, payload: dict) -> dict:
    return build_event_record(
        event_id="event_" + "a" * 32,
        job_id="job-1",
        attempt_id="attempt-1",
        attempt_number=1,
        emitted_at=NOW,
        stage=stage,
        type=type_,
        payload=payload,
    )


def test_build_event_record_has_no_sequence_key_and_embeds_a_valid_digest() -> None:
    record = _event(
        OutboundEventType.LOG,
        HarnessStage.RUNNING,
        {"level": "info", "message": "hello"},
    )
    assert "sequence" not in record
    assert is_valid_digest(record["digest"])
    assert record["digest"] == event_payload_digest(record["payload"])


def test_hosted_event_requires_sequence_but_draft_does_not() -> None:
    record = _event(
        OutboundEventType.LOG, HarnessStage.RUNNING, {"level": "debug", "message": "m"}
    )
    HostedEventDraft.model_validate(record)  # no sequence needed
    with pytest.raises(ValidationError):
        HostedEvent.model_validate(
            record
        )  # sequence is required on the full wire object
    HostedEvent.model_validate({**record, "sequence": 1})


@pytest.mark.parametrize(
    ("type_", "stage", "payload"),
    [
        (
            OutboundEventType.STAGE_CHANGED,
            HarnessStage.RUNNING,
            {"from": "connecting_agent", "to": "running"},
        ),
        (
            OutboundEventType.STAGE_CHANGED,
            HarnessStage.QUEUED,
            {"from": None, "to": "queued"},
        ),
        (
            OutboundEventType.PARALLELISM_DEGRADED,
            HarnessStage.VALIDATING_ENVIRONMENT,
            {"requested": 4, "effective": 1, "reason": "conformance_gate_failed"},
        ),
        (
            OutboundEventType.BASELINE_FROZEN,
            HarnessStage.VALIDATING_ENVIRONMENT,
            {"inputs_digest": "sha256:" + "1" * 64, "baseline_ref": "build-1"},
        ),
        (
            OutboundEventType.BASELINE_INPUTS_CHANGED,
            HarnessStage.VALIDATING_ENVIRONMENT,
            {"previous_digest": None, "current_digest": "sha256:" + "2" * 64},
        ),
        (
            OutboundEventType.WORLD_UNHEALTHY,
            HarnessStage.RUNNING,
            {"world_index": 1, "cause": "sentinel failed"},
        ),
        (
            OutboundEventType.SCENARIO_STARTED,
            HarnessStage.RUNNING,
            {"scenario_key": "k", "world_index": 0, "scenario_attempt": 1},
        ),
        (
            OutboundEventType.SCENARIO_RETRIED,
            HarnessStage.RUNNING,
            {"scenario_key": "k", "from_world": 0, "to_world": 1},
        ),
        (
            OutboundEventType.LOG,
            HarnessStage.RUNNING,
            {"level": "warning", "message": "m"},
        ),
        (
            OutboundEventType.TERMINAL,
            HarnessStage.COMPLETED,
            {
                "stage": "completed",
                "reason": None,
                "failure": None,
                "scenario_counts": {
                    "passed": 2,
                    "failed": 0,
                    "errored": 0,
                    "skipped": 0,
                },
            },
        ),
        (
            OutboundEventType.TERMINAL,
            HarnessStage.CANCELED,
            {
                "stage": "canceled",
                "reason": "user_canceled",
                "failure": None,
                "scenario_counts": {
                    "passed": 0,
                    "failed": 0,
                    "errored": 0,
                    "skipped": 1,
                },
            },
        ),
    ],
)
def test_every_closed_event_type_accepts_its_contract_shaped_payload(
    type_: OutboundEventType, stage: HarnessStage, payload: dict
) -> None:
    record = _event(type_, stage, payload)
    HostedEvent.model_validate({**record, "sequence": 1})


def test_stage_changed_to_must_equal_the_events_own_stage() -> None:
    with pytest.raises(ValidationError):
        _event(
            OutboundEventType.STAGE_CHANGED,
            HarnessStage.RUNNING,
            {"from": None, "to": "grading"},
        )


def test_stage_changed_rejects_the_python_attribute_name_instead_of_the_wire_alias() -> (
    None
):
    # M9: populate_by_name=True previously let a caller who writes `from_stage` (the natural
    # mistake -- that's the attribute name) pass validation while spooling an undefined wire key.
    with pytest.raises(ValidationError):
        StageChangedPayload.model_validate({"from_stage": "queued", "to": "running"})


def test_terminal_payload_stage_must_equal_the_events_own_stage() -> None:
    with pytest.raises(ValidationError):
        _event(
            OutboundEventType.TERMINAL,
            HarnessStage.COMPLETED,
            {
                "stage": "failed",
                "reason": None,
                "failure": None,
                "scenario_counts": {
                    "passed": 0,
                    "failed": 0,
                    "errored": 0,
                    "skipped": 0,
                },
            },
        )


def test_terminal_payload_rejects_a_non_terminal_stage() -> None:
    with pytest.raises(ValidationError):
        _event(
            OutboundEventType.TERMINAL,
            HarnessStage.RUNNING,
            {
                "stage": "running",
                "reason": None,
                "failure": None,
                "scenario_counts": {
                    "passed": 0,
                    "failed": 0,
                    "errored": 0,
                    "skipped": 0,
                },
            },
        )


def test_scenario_started_scenario_attempt_is_closed_to_one_or_two() -> None:
    with pytest.raises(ValidationError):
        _event(
            OutboundEventType.SCENARIO_STARTED,
            HarnessStage.RUNNING,
            {"scenario_key": "k", "world_index": 0, "scenario_attempt": 3},
        )


def test_parallelism_degraded_effective_must_be_strictly_below_requested() -> None:
    with pytest.raises(ValidationError):
        _event(
            OutboundEventType.PARALLELISM_DEGRADED,
            HarnessStage.VALIDATING_ENVIRONMENT,
            {"requested": 3, "effective": 3, "reason": "fixed_port"},
        )


def test_unknown_event_type_is_rejected_by_the_closed_vocabulary() -> None:
    with pytest.raises(ValidationError):
        build_event_record(
            event_id="event_x",
            job_id="job-1",
            attempt_id="attempt-1",
            attempt_number=1,
            emitted_at=NOW,
            stage=HarnessStage.RUNNING,
            type="not_a_real_type",  # type: ignore[arg-type]
            payload={},
        )


def test_payload_with_an_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _event(
            OutboundEventType.LOG,
            HarnessStage.RUNNING,
            {"level": "info", "message": "m", "extra": "nope"},
        )


def test_oversized_payload_for_a_non_log_type_is_still_hard_rejected() -> None:
    # M8: only `log` is truncated instead of rejected -- every other type keeps the hard rejection.
    # `baseline_frozen` has no field-level max_length, so an artificially huge `baseline_ref` is a
    # clean way to exercise the envelope-level size check for a non-log type.
    with pytest.raises(ValidationError):
        _event(
            OutboundEventType.BASELINE_FROZEN,
            HarnessStage.VALIDATING_ENVIRONMENT,
            {
                "inputs_digest": "sha256:" + "1" * 64,
                "baseline_ref": "x" * (EVENT_PAYLOAD_MAX_BYTES + 1),
            },
        )


def test_log_payload_is_truncated_rather_than_rejected_when_oversized() -> None:
    huge_message = "x" * (EVENT_PAYLOAD_MAX_BYTES * 2)
    record = _event(
        OutboundEventType.LOG,
        HarnessStage.RUNNING,
        {"level": "error", "message": huge_message},
    )
    assert len(canonical_bytes(record["payload"])) <= EVENT_PAYLOAD_MAX_BYTES
    assert record["payload"]["message"].endswith("…[truncated]")
    # The digest must match the TRUNCATED bytes actually spooled, not the original message.
    assert record["digest"] == event_payload_digest(record["payload"])
    HostedEvent.model_validate({**record, "sequence": 1})  # round-trips cleanly


def test_truncate_log_message_is_a_no_op_when_already_within_budget() -> None:
    assert truncate_log_message("info", "short message") == "short message"


def test_truncate_log_message_fits_the_exact_byte_budget() -> None:
    message = truncate_log_message("error", "y" * 1000, max_payload_bytes=64)
    assert message.endswith("…[truncated]")
    assert len(canonical_bytes({"level": "error", "message": message})) <= 64


def test_a_tampered_digest_is_rejected() -> None:
    record = _event(
        OutboundEventType.LOG, HarnessStage.RUNNING, {"level": "info", "message": "m"}
    )
    record["digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError):
        HostedEventDraft.model_validate(record)


def test_event_id_is_checked_against_utf8_byte_length_not_just_character_count() -> (
    None
):
    # N-2: max_length=64 on the model counts characters; the platform's column is presumably
    # bytes. 64 non-ASCII characters can exceed 64 bytes while passing the character-count check.
    event_id = "€" * 64  # 64 chars, 3 bytes each in UTF-8 => 192 bytes
    with pytest.raises(ValidationError):
        build_event_record(
            event_id=event_id,
            job_id="job-1",
            attempt_id="attempt-1",
            attempt_number=1,
            emitted_at=NOW,
            stage=HarnessStage.RUNNING,
            type=OutboundEventType.LOG,
            payload={"level": "info", "message": "m"},
        )


def test_hosted_event_emitted_at_is_serialized_as_millis_z_not_the_pydantic_default() -> (
    None
):
    record = _event(
        OutboundEventType.LOG, HarnessStage.RUNNING, {"level": "info", "message": "m"}
    )
    assert record["emitted_at"] == "2026-08-25T10:14:03.412Z"


def test_hosted_event_emitted_at_rejects_a_naive_datetime() -> None:
    with pytest.raises(ValidationError):
        build_event_record(
            event_id="event_x",
            job_id="job-1",
            attempt_id="attempt-1",
            attempt_number=1,
            emitted_at=datetime(2026, 8, 25, 10, 0, 0),  # naive
            stage=HarnessStage.RUNNING,
            type=OutboundEventType.LOG,
            payload={"level": "info", "message": "m"},
        )


def test_hosted_event_emitted_at_normalizes_a_non_utc_offset_to_z() -> None:
    ist = datetime(
        2026, 8, 25, 15, 44, 3, 412000, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    record = build_event_record(
        event_id="event_x",
        job_id="job-1",
        attempt_id="attempt-1",
        attempt_number=1,
        emitted_at=ist,
        stage=HarnessStage.RUNNING,
        type=OutboundEventType.LOG,
        payload={"level": "info", "message": "m"},
    )
    assert record["emitted_at"] == "2026-08-25T10:14:03.412Z"


# --- Spool ---------------------------------------------------------------------------------


def test_spool_assigns_contiguous_sequence_numbers_starting_at_one(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    sequences = [spool.append({"event_id": f"e{i}"}).sequence for i in range(5)]
    assert sequences == [1, 2, 3, 4, 5]
    assert spool.next_sequence == 6


def test_spool_appended_body_round_trips_the_assigned_sequence(tmp_path: Path) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    record = spool.append({"event_id": "e1", "payload": {"x": 1}})
    decoded = record.decode()
    assert decoded["sequence"] == 1
    assert decoded["event_id"] == "e1"


def test_spool_recovers_next_sequence_across_a_clean_restart(tmp_path: Path) -> None:
    first = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(3):
        first.append({"event_id": f"e{i}"})

    # M6: the same (root, name) returns the CACHED instance within one process -- simulating a
    # real restart (a fresh interpreter, an empty registry) needs an explicit evict.
    OutboundSpool._forget_for_tests(tmp_path, "events")
    second = OutboundSpool(tmp_path, "events", sequenced=True)
    assert second is not first
    assert second.next_sequence == 4
    assert second.append({"event_id": "e3"}).sequence == 4


def test_spool_recovers_from_a_torn_tail_by_truncating_and_reusing_the_sequence(
    tmp_path: Path,
) -> None:
    """Simulates a crash mid-write: a partial line with no trailing newline is appended directly
    to the spool file (bypassing `append`, the way a killed process would leave the file). The
    recovery rule this module documents says the next spool instance must detect this, truncate
    the file back to the last complete record, and reuse the torn record's sequence number on its
    next append -- never skip past it and leave a gap."""
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e1"})
    spool.append({"event_id": "e2"})

    path = tmp_path / "events.spool.jsonl"
    size_before = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(
            b'{"event_id":"e3","sequ'
        )  # torn: no closing brace, no trailing newline

    OutboundSpool._forget_for_tests(tmp_path, "events")
    recovered = OutboundSpool(tmp_path, "events", sequenced=True)
    assert recovered.next_sequence == 3
    assert path.stat().st_size == size_before  # torn bytes truncated away

    replay = recovered.append({"event_id": "e3-retry"})
    assert replay.sequence == 3

    ids_in_order = [record.decode()["event_id"] for record in recovered.records()]
    assert ids_in_order == ["e1", "e2", "e3-retry"]


def test_spool_recovers_from_a_torn_tail_with_no_trailing_newline_at_all(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e1"})
    path = tmp_path / "events.spool.jsonl"
    # Drop the final newline entirely, simulating a crash after the JSON bytes landed but before
    # the trailing separator (and its fsync) did.
    data = path.read_bytes()
    assert data.endswith(b"\n")
    path.write_bytes(data[:-1])

    OutboundSpool._forget_for_tests(tmp_path, "events")
    recovered = OutboundSpool(tmp_path, "events", sequenced=True)
    assert recovered.next_sequence == 1
    assert recovered.records() == []
    assert recovered.append({"event_id": "e1-retry"}).sequence == 1


def test_spool_recovery_degrades_on_mid_file_corruption_instead_of_raising(
    tmp_path: Path,
) -> None:
    """N8 (supersedes the old B1-era "raises" posture): a bad line with COMPLETE records after it
    (a trailing `\\n` of its own) is genuine corruption, never a torn tail -- B1 already forbids
    silently renumbering past it, but raising unconditionally was the OTHER extreme: it discarded
    even the records BEFORE the corruption, permanently, including a future terminal event. Recovery
    must instead succeed, read the readable prefix, and flag the corruption once."""
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e1"})
    spool.append({"event_id": "e2"})
    spool.append({"event_id": "e3"})

    path = tmp_path / "events.spool.jsonl"
    lines = path.read_bytes().split(b"\n")
    lines[1] = (
        b"{not valid json but has a trailing newline"  # complete line, unparseable
    )
    path.write_bytes(b"\n".join(lines))

    OutboundSpool._forget_for_tests(tmp_path, "events")
    recovered = OutboundSpool(tmp_path, "events", sequenced=True)  # must NOT raise
    assert recovered.is_corrupt
    assert recovered.corruption_offset is not None
    assert [r.decode()["event_id"] for r in recovered.records()] == ["e1"]
    # Never renumbers past the corruption: e1 was sequence 1, so the next allocation is 2 -- even
    # though the file still physically contains e2/e3 at sequences 2/3 beyond the corrupt line.
    assert recovered.next_sequence == 2


def test_spool_recovery_ships_a_record_appended_after_the_corruption_via_the_seek_path(
    tmp_path: Path,
) -> None:
    """N8, ranked missing test 5: the readable prefix is still usable AND the stream keeps working
    going forward -- a new record appended after recovery (standing in for "the terminal event
    still ships") is reachable via `pending_since_watermark()`'s seek path (what `EventsClient.flush`
    actually uses), even though a full `records()` scan still stops at the untouched corrupt byte."""
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e1"})
    spool.append({"event_id": "e2"})
    spool.append({"event_id": "e3"})
    spool.advance_watermark(
        1
    )  # e1 already acked before the corruption is ever discovered

    path = tmp_path / "events.spool.jsonl"
    lines = path.read_bytes().split(b"\n")
    lines[1] = b"{not valid json but has a trailing newline"
    path.write_bytes(b"\n".join(lines))

    OutboundSpool._forget_for_tests(tmp_path, "events")
    recovered = OutboundSpool(tmp_path, "events", sequenced=True)
    assert recovered.is_corrupt
    assert (
        recovered.next_sequence == 2
    )  # only e1 (pre-corruption) counted; watermark=1 agrees

    terminal = recovered.append({"event_id": "terminal"})
    assert terminal.sequence == 2
    pending = recovered.pending_since_watermark()
    assert [r.decode()["event_id"] for r in pending] == ["terminal"]


def test_spool_compact_through_refuses_on_a_corrupt_spool_instead_of_destroying_records(
    tmp_path: Path,
) -> None:
    # P1, second victim: `_rewrite_retaining` (what BOTH `compact_through` and `drop_many` funnel
    # through) used to rewrite the file from ONLY the records-before-corruption prefix, so a rewrite
    # on a corrupt spool silently destroyed every intact record past the corrupt byte too --
    # regardless of `compact_through`'s own watermark clamp, which only bounds WHICH sequences it
    # asks to keep, not what the underlying rewrite can actually see. Must now refuse entirely
    # rather than lose an already-durable, never-sent record this way.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e1"})

    path = tmp_path / "events.spool.jsonl"
    lines = path.read_bytes().split(b"\n")
    lines[0] = b"{not valid json but has a trailing newline"
    path.write_bytes(b"\n".join(lines))

    OutboundSpool._forget_for_tests(tmp_path, "events")
    recovered = OutboundSpool(tmp_path, "events", sequenced=True)
    assert recovered.is_corrupt

    terminal = recovered.append({"event_id": "event_terminal"})
    recovered.advance_watermark(
        terminal.sequence
    )  # fully acked -- would normally be compactable
    recovered.compact_through(terminal.sequence)

    assert b"event_terminal" in path.read_bytes(), (
        "P1 regression: compact_through destroyed a record on a corrupt spool"
    )


def test_spool_records_degrades_on_corruption_instead_of_raising(
    tmp_path: Path,
) -> None:
    # N8: records() shares _iter_complete_records with _recover -- corruption discovered via a
    # fresh read (not at construction time) degrades the same way: the readable prefix is still
    # returned, never a bare exception (nor the old typed raise) a caller has no reason to catch.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e1"})
    path = tmp_path / "events.spool.jsonl"
    with path.open("ab") as stream:
        stream.write(b"{not valid json}\n")  # complete (trailing \n), unparseable
    assert not spool.is_corrupt
    result = spool.records()
    assert [r.decode()["event_id"] for r in result] == ["e1"]
    assert spool.is_corrupt
    assert spool.corruption_offset is not None


def test_spool_corruption_is_logged_once_not_once_per_read(
    tmp_path: Path, caplog: Any
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e1"})
    path = tmp_path / "events.spool.jsonl"
    with path.open("ab") as stream:
        stream.write(b"{not valid json}\n")
    with caplog.at_level(logging.ERROR, logger="fi.alk.harness.outbound"):
        spool.records()
        spool.records()
        spool.records()
    corruption_logs = [r for r in caplog.records if "spool corrupt" in r.message]
    assert len(corruption_logs) == 1


def test_spool_failed_append_is_a_true_no_op(tmp_path: Path) -> None:
    """B1 part 1: a write that raises mid-flush must leave the file exactly as it was before the
    attempt (truncated back to size_before, fsync'd) so the retry reuses the same sequence number
    on clean ground -- never merges torn bytes with the retry's complete record into one
    unparseable line."""
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "ok1"})
    path = tmp_path / "events.spool.jsonl"
    size_before = path.stat().st_size

    class Boom(Exception):
        pass

    original_open = Path.open

    def failing_open(self: Path, mode: str = "r", *a: Any, **kw: Any) -> Any:
        if self == path and mode == "ab":
            raise Boom("simulated write failure")
        return original_open(self, mode, *a, **kw)

    Path.open = failing_open  # type: ignore[method-assign]
    try:
        with pytest.raises(Boom):
            spool.append({"event_id": "boom"})
    finally:
        Path.open = original_open  # type: ignore[method-assign]

    assert path.stat().st_size == size_before

    replay = spool.append({"event_id": "ok2"})
    assert replay.sequence == 2  # reused, not skipped
    assert [r.decode()["event_id"] for r in spool.records()] == ["ok1", "ok2"]


def test_spool_fsyncs_the_directory_once_after_first_file_creation_only(
    tmp_path: Path,
) -> None:
    # M2: a new file's directory entry isn't durable just because the file's own data is fsync'd.
    # The guard flag means exactly ONE extra fsync (the directory's) on the append that creates the
    # file, and none on later appends to the same, now-existing file.
    calls: list[int] = []
    original_fsync = os.fsync

    def counting_fsync(fd: int) -> None:
        calls.append(fd)
        original_fsync(fd)

    os.fsync = counting_fsync  # type: ignore[assignment]
    try:
        spool = OutboundSpool(tmp_path, "events", sequenced=True)
        before = len(calls)
        spool.append({"event_id": "e1"})
        after_first = len(calls)
        spool.append({"event_id": "e2"})
        after_second = len(calls)
    finally:
        os.fsync = original_fsync  # type: ignore[assignment]

    assert (
        after_first - before == 2
    )  # the record's own fsync + the one-time directory fsync
    assert (
        after_second - after_first == 1
    )  # just the record's own fsync -- no repeat directory sync


def test_canonical_bytes_never_contains_a_raw_newline_even_with_embedded_newlines_in_a_string() -> (
    None
):
    # The recovery rule's line-per-record framing depends on this holding for every value this
    # module ever canonicalizes: json.dumps escapes control characters inside strings, so an
    # embedded "\n" in a field value becomes the two-byte escape `\\n`, never a raw newline byte.
    # `OutboundSpool.append` asserts this invariant defensively (`outbound_spool_record_unframable`)
    # but the assertion is unreachable through this function -- there is no dict this module can be
    # asked to canonicalize that trips it.
    assert b"\n" not in canonical_bytes({"message": "line one\nline two"})


def test_spool_watermark_starts_at_zero_and_advances(tmp_path: Path) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(5):
        spool.append({"event_id": f"e{i}"})
    assert spool.watermark() == 0
    assert [r.sequence for r in spool.pending_since_watermark()] == [1, 2, 3, 4, 5]

    spool.advance_watermark(3)
    assert spool.watermark() == 3
    assert [r.sequence for r in spool.pending_since_watermark()] == [4, 5]


def test_spool_watermark_rejects_a_regression_as_untrusted_input(
    tmp_path: Path,
) -> None:
    # v1.3/M7: acked_through_sequence is untrusted platform input -- a value below the current
    # watermark is rejected (typed error), not silently ignored; the watermark stays unchanged.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e0"})
    spool.advance_watermark(1)
    with pytest.raises(OutboundSpoolError) as excinfo:
        spool.advance_watermark(0)
    assert excinfo.value.code == "outbound_spool_watermark_out_of_range"
    assert spool.watermark() == 1


def test_spool_watermark_rejects_a_value_at_or_above_next_sequence(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e0"})  # next_sequence becomes 2
    with pytest.raises(OutboundSpoolError) as excinfo:
        spool.advance_watermark(2)  # not yet allocated
    assert excinfo.value.code == "outbound_spool_watermark_out_of_range"
    assert spool.watermark() == 0


def test_spool_watermark_advancing_to_the_same_value_is_a_no_op(tmp_path: Path) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e0"})
    spool.advance_watermark(1)
    spool.advance_watermark(1)  # repeating the same ack must not raise
    assert spool.watermark() == 1


def test_spool_watermark_is_durable_across_a_restart(tmp_path: Path) -> None:
    first = OutboundSpool(tmp_path, "events", sequenced=True)
    first.append({"event_id": "e0"})
    first.advance_watermark(1)

    OutboundSpool._forget_for_tests(tmp_path, "events")
    second = OutboundSpool(tmp_path, "events", sequenced=True)
    assert second.watermark() == 1


def test_spool_watermark_degrades_to_zero_on_a_corrupt_file_instead_of_wedging(
    tmp_path: Path,
) -> None:
    # M1: a corrupt/torn watermark file must not permanently brick the spool -- re-sending
    # already-acked events is safe (at-least-once + dedupe on event_id); raising forever is not.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e0"})
    spool.advance_watermark(1)
    watermark_path = tmp_path / "events.spool.watermark.json"
    assert watermark_path.exists()
    watermark_path.write_text("not json")
    assert spool.watermark() == 0
    # And it self-heals: a fresh valid advance still works normally afterward.
    spool.advance_watermark(1)
    assert spool.watermark() == 1


def test_spool_advance_watermark_leaves_no_leftover_temp_files(tmp_path: Path) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e0"})
    spool.advance_watermark(1)
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_advance_watermark_raises_on_a_forked_instance(tmp_path: Path) -> None:
    # P7: `advance_watermark` is the one operation that DURABLY destroys delivery state -- once a
    # value is written, `pending_since_watermark()` can never return anything at or below it again
    # -- yet it was the one mutating method the N25 fork guard skipped. A forked child advancing the
    # parent's watermark would silently orphan every record below the new value, the N1 outcome by
    # a different route. `_poison_after_fork` is what `os.register_at_fork`'s `after_in_child` hook
    # actually calls; invoking it directly simulates the fork without an actual `os.fork()`, which a
    # shared test process can't safely drive.
    from fi.alk.harness.outbound import _poison_after_fork

    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e1"})
    _poison_after_fork()

    with pytest.raises(OutboundSpoolError) as excinfo:
        spool.advance_watermark(1)
    assert excinfo.value.code == "outbound_spool_forked"


def test_spool_recovery_seeds_next_sequence_from_the_watermark_when_the_log_is_lost(
    tmp_path: Path,
) -> None:
    """M5: a watermark ahead of the log (an M2-style lost file, or a compacted/dropped log) must
    seed next_sequence from the watermark, not just the empty/short log -- otherwise the allocator
    reissues sequence numbers the platform already closed."""
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(7):
        spool.append({"event_id": f"e{i}"})
    spool.advance_watermark(7)
    (tmp_path / "events.spool.jsonl").write_bytes(b"")  # simulate the log vanishing

    OutboundSpool._forget_for_tests(tmp_path, "events")
    recovered = OutboundSpool(tmp_path, "events", sequenced=True)
    assert recovered.next_sequence == 8
    assert recovered.append({"event_id": "e7"}).sequence == 8


def test_spool_recovery_seeds_next_sequence_from_the_watermark_when_the_file_is_missing_entirely(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(4):
        spool.append({"event_id": f"e{i}"})
    spool.advance_watermark(4)
    (tmp_path / "events.spool.jsonl").unlink()

    OutboundSpool._forget_for_tests(tmp_path, "events")
    recovered = OutboundSpool(tmp_path, "events", sequenced=True)
    assert recovered.next_sequence == 5


def test_spool_drop_is_pure_physical_removal_and_never_touches_the_watermark(
    tmp_path: Path,
) -> None:
    # N1: an earlier version had `drop` also advance the watermark to `sequence` -- that let an
    # UNTRUSTED `rejected[].sequence` silently orphan every pending record below it, bypassing
    # `advance_watermark`'s own M7 clamp entirely. `drop`/`drop_many` are now pure physical removal;
    # the batch-level `advance_watermark(acked_through_sequence)` is the only chokepoint.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(5):
        spool.append({"event_id": f"e{i}"})  # sequences 1..5
    spool.drop(3)
    remaining = [r.decode()["event_id"] for r in spool.records()]
    assert remaining == ["e0", "e1", "e3", "e4"]  # e2 was assigned sequence 3
    assert spool.watermark() == 0  # drop() must NEVER move the watermark


def test_spool_drop_many_batches_every_removal_into_one_rewrite(tmp_path: Path) -> None:
    # N14: a batch of rejections is one _rewrite_retaining call, not one per sequence -- exercised
    # indirectly here by checking the end result of a single drop_many call over a set.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(6):
        spool.append({"event_id": f"e{i}"})  # sequences 1..6
    spool.drop_many({2, 4, 6})
    remaining = [r.decode()["event_id"] for r in spool.records()]
    assert remaining == ["e0", "e2", "e4"]
    assert (
        spool.watermark() == 0
    )  # still untouched -- drop_many never advances it either


def test_spool_drop_many_with_an_empty_set_is_a_no_op(tmp_path: Path) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e0"})
    spool.drop_many(set())
    assert [r.decode()["event_id"] for r in spool.records()] == ["e0"]


def test_spool_compact_through_drops_acked_records_and_keeps_pending_ones(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(6):
        spool.append({"event_id": f"e{i}"})
    spool.advance_watermark(4)
    spool.compact_through(4)
    assert [r.decode()["event_id"] for r in spool.records()] == ["e4", "e5"]

    # The allocator survives a restart correctly even though history was compacted away.
    OutboundSpool._forget_for_tests(tmp_path, "events")
    recovered = OutboundSpool(tmp_path, "events", sequenced=True)
    assert recovered.next_sequence == 7
    assert recovered.append({"event_id": "e6"}).sequence == 7


def test_spool_compact_through_clamps_to_the_watermark_never_drops_pending_records(
    tmp_path: Path,
) -> None:
    # Fail-safe: a caller passing a too-high sequence must not be able to compact away undelivered
    # records -- compaction is clamped to whatever the platform has actually acknowledged.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(5):
        spool.append({"event_id": f"e{i}"})
    spool.advance_watermark(2)
    spool.compact_through(10**9)
    assert [r.decode()["event_id"] for r in spool.records()] == ["e2", "e3", "e4"]


def test_spool_pending_since_watermark_uses_the_cached_offset_to_seek(
    tmp_path: Path,
) -> None:
    # P11: previously this only asserted the RESULT ([5..10]), which the generic records_after()
    # fallback produces identically -- it could not fail if the seek path regressed to the
    # fallback. Monkeypatching records_after to raise pins the claim: pending_since_watermark()
    # must succeed WITHOUT ever calling it, proving the cached-offset seek path was actually taken.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(10):
        spool.append({"event_id": f"e{i}"})
    spool.advance_watermark(4)

    def _must_not_be_called(sequence: int) -> Any:
        raise AssertionError(
            "pending_since_watermark fell back to records_after instead of seeking"
        )

    spool.records_after = _must_not_be_called  # type: ignore[method-assign]
    pending = spool.pending_since_watermark()
    assert [r.sequence for r in pending] == list(range(5, 11))


def test_unsequenced_spool_assigns_no_sequence(tmp_path: Path) -> None:
    spool = OutboundSpool(tmp_path, "results", sequenced=False)
    record = spool.append({"scenario_key": "k1", "status": "passed"})
    assert record.sequence is None
    assert record.decode() == {"scenario_key": "k1", "status": "passed"}


def test_unsequenced_spool_rejects_a_caller_supplied_sequence_key(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "results", sequenced=False)
    with pytest.raises(OutboundSpoolError) as excinfo:
        spool.append({"sequence": 1, "scenario_key": "k1"})
    # MIN-9: distinct from the accessor-misuse code below -- "caller passed a sequence key into an
    # unsequenced append" and "caller called a sequence-only method" are different faults.
    assert excinfo.value.code == "outbound_spool_caller_supplied_sequence"


@pytest.mark.parametrize(
    "member",
    [
        "next_sequence",
        "watermark",
        "pending_since_watermark",
        "records_after",
        "drop",
        "compact_through",
    ],
)
def test_unsequenced_spool_rejects_sequence_only_operations(
    tmp_path: Path, member: str
) -> None:
    spool = OutboundSpool(tmp_path, "results", sequenced=False)
    with pytest.raises(OutboundSpoolError) as excinfo:
        if member == "next_sequence":
            spool.next_sequence
        elif member in ("records_after", "drop", "compact_through"):
            getattr(spool, member)(1)
        else:
            getattr(spool, member)()
    assert excinfo.value.code == "outbound_spool_unsequenced"


def test_unsequenced_spool_rejects_advance_watermark(tmp_path: Path) -> None:
    spool = OutboundSpool(tmp_path, "results", sequenced=False)
    with pytest.raises(OutboundSpoolError):
        spool.advance_watermark(1)


def test_separate_streams_do_not_share_sequence_numbers(tmp_path: Path) -> None:
    events = OutboundSpool(tmp_path, "events", sequenced=True)
    manifests = OutboundSpool(tmp_path, "artifact_manifest", sequenced=False)
    events.append({"event_id": "e1"})
    events.append({"event_id": "e2"})
    manifests.append({"complete": False})
    assert events.next_sequence == 3
    assert [r.sequence for r in events.records()] == [1, 2]
    assert [r.sequence for r in manifests.records()] == [None]


def test_spool_append_is_safe_under_concurrent_callers(tmp_path: Path) -> None:
    # "one allocator, one lock" -- concurrent appends from multiple threads must still produce a
    # contiguous, gap-free, duplicate-free sequence.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    sequences: list[int] = []
    lock = threading.Lock()

    def worker(start: int) -> None:
        for i in range(25):
            record = spool.append({"event_id": f"t{start}-{i}"})
            with lock:
                sequences.append(record.sequence)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(sequences) == list(range(1, 201))
    assert spool.next_sequence == 201


def test_spool_creates_its_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "outbound"
    assert not root.exists()
    OutboundSpool(root, "events", sequenced=True)
    assert root.is_dir()


def test_spool_root_directory_is_created_with_restrictive_permissions(
    tmp_path: Path,
) -> None:
    # MIN-10: the spool holds event payloads and receipt bodies in a multi-user sandbox.
    root = tmp_path / "outbound"
    OutboundSpool(root, "events", sequenced=True)
    assert (root.stat().st_mode & 0o777) == 0o700


def test_spool_records_survive_being_read_back_after_many_appends(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(50):
        spool.append({"event_id": f"e{i}", "n": i})
    records = spool.records()
    assert len(records) == 50
    assert [r.decode()["n"] for r in records] == list(range(50))
    assert (tmp_path / "events.spool.jsonl").is_file()


# --- M6: one allocator per (root, name) -----------------------------------------------------


def test_same_stream_name_returns_the_cached_instance_not_a_second_allocator(
    tmp_path: Path,
) -> None:
    first = OutboundSpool(tmp_path, "events", sequenced=True)
    second = OutboundSpool(tmp_path, "events", sequenced=True)
    assert second is first
    sequences = [
        first.append({"event_id": "a"}).sequence,
        second.append({"event_id": "b"}).sequence,
    ]
    assert sequences == [
        1,
        2,
    ]  # one allocator regardless of how many handles point at it


def test_same_stream_name_with_a_conflicting_sequenced_flag_raises(
    tmp_path: Path,
) -> None:
    OutboundSpool(tmp_path, "events", sequenced=True)
    with pytest.raises(OutboundSpoolError) as excinfo:
        OutboundSpool(tmp_path, "events", sequenced=False)
    assert excinfo.value.code == "outbound_spool_sequenced_mismatch"


def test_flock_rejects_a_second_lock_holder_on_the_same_lock_file(
    tmp_path: Path,
) -> None:
    # M6: cross-PROCESS protection -- the in-process registry above only protects against a second
    # Python-level instance in this same interpreter; a genuinely separate process trying to open
    # the same lock file must be rejected by the OS-level flock regardless of the registry.
    OutboundSpool(tmp_path, "events", sequenced=True)  # holds the flock via its own fd
    lock_path = tmp_path / "events.spool.lock"
    fd = os.open(str(lock_path), os.O_RDWR)
    try:
        with pytest.raises(OSError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_a_forgotten_spool_can_be_reacquired_after_the_lock_is_released(
    tmp_path: Path,
) -> None:
    OutboundSpool(tmp_path, "events", sequenced=True)
    OutboundSpool._forget_for_tests(tmp_path, "events")  # releases the flock fd too
    # A fresh construction must succeed cleanly -- proves _forget_for_tests actually released the
    # OS-level lock, not just the in-process cache entry.
    fresh = OutboundSpool(tmp_path, "events", sequenced=True)
    assert fresh.next_sequence == 1


def test_close_releases_the_lock_and_evicts_the_registry(tmp_path: Path) -> None:
    # N26: OutboundSpool had no supported teardown -- close() is the real one (_forget_for_tests is
    # test-only and reaches for it internally now).
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.close()
    fresh = OutboundSpool(
        tmp_path, "events", sequenced=True
    )  # must not raise outbound_spool_locked
    assert fresh is not spool
    assert fresh.next_sequence == 1
    spool.close()  # idempotent -- a second close on the old instance must not raise


def test_close_poisons_the_instance_so_a_closed_spool_cannot_append_with_no_lock_held(
    tmp_path: Path,
) -> None:
    # P2: close() evicted the registry entry and released the flock fd, but left the closed
    # instance itself fully mutable -- `_initialized` stayed True and nothing else guarded it, so a
    # caller still holding the closed reference could keep appending with NO lock held, and a fresh
    # construction for the same key would allocate an independent, unaware second `_next_sequence`.
    # `close()` must poison the instance too, not just the registry slot.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    spool.append({"event_id": "e1"})
    spool.close()

    with pytest.raises(OutboundSpoolError) as excinfo:
        spool.append({"event_id": "after-close"})
    assert excinfo.value.code == "outbound_spool_closed"

    with pytest.raises(OutboundSpoolError):
        spool.advance_watermark(1)

    # A fresh instance for the same key is the ONLY live allocator now -- it continues the SAME
    # sequence the closed instance left off at, rather than starting a second, independent one.
    fresh = OutboundSpool(tmp_path, "events", sequenced=True)
    fresh_record = fresh.append({"event_id": "fresh"})
    assert fresh_record.sequence == 2
    assert [r.decode()["event_id"] for r in fresh.records()] == ["e1", "fresh"]


def test_failed_construction_does_not_poison_the_registry_for_the_next_attempt(
    tmp_path: Path,
) -> None:
    # N4, ranked missing test 4: a construction that fails PARTWAY (after acquiring the process
    # lock, before finishing) must not leave a half-built instance registered -- the next
    # construction attempt for the SAME key gets a typed error reflecting its OWN failure, never
    # AttributeError (a half-built instance missing _sequenced) or a spurious outbound_spool_locked
    # (a leaked fd from the first attempt).
    root = tmp_path / "n4"
    lock_path = root / "events.spool.lock"
    root.mkdir()
    # Simulate "another process" already holding the flock -- this is exactly what
    # _acquire_process_lock hits AFTER self._sequenced is already set, so it reproduces the review's
    # second failure mode (a failure after the half-built-instance point, not before it).
    held_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(held_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(OutboundSpoolError) as excinfo:
            OutboundSpool(root, "events", sequenced=True)
        assert excinfo.value.code == "outbound_spool_locked"
    finally:
        os.close(held_fd)  # release the "other process"'s hold

    # The registry must have NOTHING registered for this key -- construction now succeeds cleanly,
    # not with AttributeError ('_sequenced') and not with a second outbound_spool_locked from a
    # leaked fd.
    spool = OutboundSpool(root, "events", sequenced=True)
    assert spool.next_sequence == 1


# =================================================================================================
# PART 2 -- Transport: error map, retry/backoff, typed receipt/manifest models, channel clients.
# =================================================================================================


# --- classify_response (the closed error map) ---------------------------------------------------


def test_classify_response_returns_none_for_every_2xx() -> None:
    assert classify_response(200, {}, attempt=1) is None
    assert classify_response(201, {"status": "stored"}, attempt=1) is None
    assert classify_response(299, {}, attempt=1) is None


def test_classify_response_network_failure_is_retryable_connectivity() -> None:
    error = classify_response(None, None, attempt=1)
    assert error is not None
    assert error.outcome is ChannelOutcome.RETRYABLE
    assert error.domain is FailureDomain.CONNECTIVITY
    assert error.code == "network_error"


@pytest.mark.parametrize("status", [401, 403])
def test_classify_response_401_403_are_fenced_never_retried(status: int) -> None:
    error = classify_response(
        status, {"error": "e", "message": "m", "retryable": False}, attempt=1
    )
    assert error is not None
    assert error.outcome is ChannelOutcome.FENCED
    assert (
        error.domain is None
    )  # "never an infra retry" -- not a FailureDomain-tagged outcome


def test_classify_response_404_retries_twice_then_channel_failed_on_the_third() -> None:
    first = classify_response(404, {"error": "not_found", "message": "m"}, attempt=1)
    second = classify_response(404, {"error": "not_found", "message": "m"}, attempt=2)
    third = classify_response(404, {"error": "not_found", "message": "m"}, attempt=3)
    assert first is not None and first.outcome is ChannelOutcome.RETRYABLE
    assert second is not None and second.outcome is ChannelOutcome.RETRYABLE
    assert third is not None
    assert third.outcome is ChannelOutcome.CHANNEL_FAILED
    assert third.domain is FailureDomain.PLATFORM_SYNC


def test_classify_response_413_is_budget_exceeded() -> None:
    error = classify_response(
        413, {"error": "artifact_budget_exceeded", "message": "m"}, attempt=1
    )
    assert error is not None
    assert error.outcome is ChannelOutcome.BUDGET_EXCEEDED
    assert error.status_code == 413


def test_classify_response_413_without_artifact_budget_exceeded_is_permanent_item() -> (
    None
):
    # N7: 413 is Channel 3's code specifically -- a 413 that does NOT report
    # artifact_budget_exceeded (e.g. an events batch too big for the platform's own ingress cap)
    # must not be mislabeled BUDGET_EXCEEDED on a channel with no such concept.
    error = classify_response(
        413, {"error": "request_entity_too_large", "message": "m"}, attempt=1
    )
    assert error is not None
    assert error.outcome is ChannelOutcome.PERMANENT_ITEM
    assert error.status_code == 413

    error_no_body = classify_response(413, None, attempt=1)
    assert error_no_body is not None
    assert error_no_body.outcome is ChannelOutcome.PERMANENT_ITEM


def test_classify_response_404_domain_is_platform_sync_on_every_attempt() -> None:
    # N29: a 404 is never a connectivity fault under §4.6 -- PLATFORM_SYNC on attempts 1-2 too, not
    # just the third (cosmetic before; the inconsistency invited a wrong read).
    first = classify_response(404, {"error": "not_found", "message": "m"}, attempt=1)
    second = classify_response(404, {"error": "not_found", "message": "m"}, attempt=2)
    assert first is not None and first.domain is FailureDomain.PLATFORM_SYNC
    assert second is not None and second.domain is FailureDomain.PLATFORM_SYNC


def test_classify_response_unlisted_3xx_is_permanent_not_retryable() -> None:
    # N27: the contract never contemplates a 3xx (every endpoint ends in "/" precisely so Django's
    # POST-redirect problem never arises) -- a 301 from a mis-slashed endpoint should surface as a
    # permanent misconfiguration, not loop max_attempts times before giving up as connectivity.
    error = classify_response(301, {"error": "e", "message": "m"}, attempt=1)
    assert error is not None
    assert error.outcome is ChannelOutcome.PERMANENT_ITEM


def test_classify_response_429_is_retryable_and_carries_retry_after() -> None:
    error = classify_response(
        429,
        {"error": "rate_limited", "message": "m"},
        attempt=1,
        retry_after_seconds=7.0,
    )
    assert error is not None
    assert error.outcome is ChannelOutcome.RETRYABLE
    assert error.retry_after_seconds == 7.0


@pytest.mark.parametrize("status", [400, 409, 422])
def test_classify_response_400_409_422_are_permanent_item(status: int) -> None:
    error = classify_response(
        status, {"error": f"e{status}", "message": "m"}, attempt=1
    )
    assert error is not None
    assert error.outcome is ChannelOutcome.PERMANENT_ITEM
    assert error.domain is None


def test_classify_response_unlisted_4xx_is_the_permanent_catch_all() -> None:
    error = classify_response(418, {"error": "teapot", "message": "m"}, attempt=1)
    assert error is not None
    assert error.outcome is ChannelOutcome.PERMANENT_ITEM


@pytest.mark.parametrize("status", [500, 502, 503])
def test_classify_response_5xx_is_retryable_connectivity(status: int) -> None:
    error = classify_response(status, {"error": "e", "message": "m"}, attempt=1)
    assert error is not None
    assert error.outcome is ChannelOutcome.RETRYABLE
    assert error.domain is FailureDomain.CONNECTIVITY


# --- compute_backoff_seconds (full jitter) --------------------------------------------------------


def test_compute_backoff_seconds_full_jitter_formula() -> None:
    # rng pinned to 1.0 -> exercises the pure ceiling formula, unjittered.
    assert (
        compute_backoff_seconds(
            1, initial_backoff_seconds=1.0, max_backoff_seconds=15.0, rng=lambda: 1.0
        )
        == 1.0
    )
    assert (
        compute_backoff_seconds(
            2, initial_backoff_seconds=1.0, max_backoff_seconds=15.0, rng=lambda: 1.0
        )
        == 2.0
    )
    assert (
        compute_backoff_seconds(
            3, initial_backoff_seconds=1.0, max_backoff_seconds=15.0, rng=lambda: 1.0
        )
        == 4.0
    )


def test_compute_backoff_seconds_caps_at_max_backoff() -> None:
    assert (
        compute_backoff_seconds(
            10, initial_backoff_seconds=1.0, max_backoff_seconds=15.0, rng=lambda: 1.0
        )
        == 15.0
    )


def test_compute_backoff_seconds_scales_by_rng() -> None:
    assert (
        compute_backoff_seconds(
            3, initial_backoff_seconds=1.0, max_backoff_seconds=15.0, rng=lambda: 0.5
        )
        == 2.0
    )


# --- format_rfc3339_millis -------------------------------------------------------------------------


def test_format_rfc3339_millis_matches_the_contracts_wire_form() -> None:
    assert format_rfc3339_millis(NOW) == "2026-08-25T10:14:03.412Z"
    assert is_valid_rfc3339_millis(format_rfc3339_millis(NOW))


def test_format_rfc3339_millis_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError):
        format_rfc3339_millis(datetime(2026, 8, 25))


def test_is_valid_rfc3339_millis_rejects_the_default_pydantic_offset_form() -> None:
    # The exact failure mode `format_rfc3339_millis` exists to prevent: pydantic's default
    # datetime JSON serialization uses a `+00:00` offset and six-digit microseconds, not `Z` and
    # milliseconds -- this string is what NOT using the helper would have produced.
    assert not is_valid_rfc3339_millis("2026-08-25T10:14:03.412000+00:00")


# --- ArtifactBudgetTracker -------------------------------------------------------------------------


def test_artifact_budget_tracker_reserved_kinds_always_admitted() -> None:
    tracker = ArtifactBudgetTracker(max_artifact_bytes=10)
    assert tracker.would_admit(ArtifactKind.RESULT, 1_000_000, digest="d1")
    tracker.record(ArtifactKind.RESULT, 1_000_000, digest="d1")
    # Reserved kinds are counted (for accounting) but never refused, even once over budget.
    assert tracker.admitted_bytes == 1_000_000
    assert tracker.would_admit(ArtifactKind.BUILD, 999, digest="d2")


def test_artifact_budget_tracker_refuses_non_reserved_once_budget_is_exhausted() -> (
    None
):
    tracker = ArtifactBudgetTracker(max_artifact_bytes=100)
    assert tracker.would_admit(ArtifactKind.TRACE, 60, digest="d1")
    tracker.record(ArtifactKind.TRACE, 60, digest="d1")
    assert tracker.would_admit(ArtifactKind.TRACE, 40, digest="d2")
    tracker.record(ArtifactKind.TRACE, 40, digest="d2")
    assert not tracker.would_admit(ArtifactKind.LOG, 1, digest="d3")


def test_artifact_budget_tracker_duplicate_digest_is_free() -> None:
    tracker = ArtifactBudgetTracker(max_artifact_bytes=10)
    tracker.record(ArtifactKind.LOG, 10, digest="dup")
    assert tracker.admitted_bytes == 10
    assert tracker.would_admit(ArtifactKind.LOG, 10, digest="dup")
    tracker.record(ArtifactKind.LOG, 10, digest="dup")
    assert tracker.admitted_bytes == 10  # unchanged -- already counted


def test_priority_class_orders_reserved_recordings_other() -> None:
    # N16: reserved (0) > recordings (1) > trace/log/other (2), per the contract's three-tier
    # budget partition.
    assert priority_class(ArtifactKind.RESULT) == 0
    assert priority_class(ArtifactKind.BUILD) == 0
    assert priority_class(ArtifactKind.TRANSCRIPT) == 0
    assert priority_class(ArtifactKind.TOOL_TRACE) == 0
    assert priority_class(ArtifactKind.RECORDING_COMBINED) == 1
    assert priority_class(ArtifactKind.RECORDING_STEREO) == 1
    assert priority_class(ArtifactKind.TRACE) == 2
    assert priority_class(ArtifactKind.LOG) == 2
    assert priority_class(ArtifactKind.OTHER) == 2


def test_artifact_budget_tracker_reserves_recording_headroom_from_trace_log_other() -> (
    None
):
    # N16: a non-zero recording_headroom_bytes shrinks what a trace/log/other candidate may
    # consume, leaving room for recordings not yet seen -- default (0) behavior is unaffected
    # (covered by the pre-existing tracker tests above).
    tracker = ArtifactBudgetTracker(max_artifact_bytes=100, recording_headroom_bytes=30)
    # 70 bytes remain after the 30-byte reservation -- a 71-byte trace is refused, a 70-byte one is not.
    assert not tracker.would_admit(ArtifactKind.TRACE, 71, digest="d1")
    assert tracker.would_admit(ArtifactKind.TRACE, 70, digest="d2")
    # Recordings are NOT subject to the headroom reservation themselves -- only class-2 candidates.
    assert tracker.would_admit(ArtifactKind.RECORDING_COMBINED, 100, digest="d3")


# --- ResultReceiptDraft / build_result_receipt / build_skipped_receipt -----------------------------


def _passed_receipt(**overrides: Any) -> dict[str, Any]:
    base = dict(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        scenario_key="suspended-account-blocked",
        scenario_id="sid-1",
        scenario_attempt=1,
        world_index=2,
        status=ScenarioStatus.PASSED,
        sub_goals=[{"name": "n", "held": True, "reason": None, "judged": False}],
        evaluations=[
            {
                "name": "customer_agent_task_completion",
                "kind": "metric",
                "score": 0.86,
                "reason": "r",
            },
            {
                "name": "checked_refund",
                "kind": "checkpoint",
                "passed": True,
                "reason": "r",
            },
        ],
        call={
            "started_at": format_rfc3339_millis(NOW),
            "ended_at": format_rfc3339_millis(LATER),
            "duration_ms": 184211,
            "turns": 5,
            "transcript_artifact": "sha256:" + "a" * 64,
            "recording_artifacts": ["sha256:" + "b" * 64],
        },
        failure=None,
    )
    base.update(overrides)
    return build_result_receipt(**base)


def test_build_result_receipt_matches_the_contract_shaped_example() -> None:
    receipt = _passed_receipt()
    assert is_valid_digest(receipt["digest"])
    assert receipt["status"] == "passed"
    assert receipt["digest"] == whole_object_digest(
        {k: v for k, v in receipt.items() if k != "digest"}
    )
    ResultReceiptDraft.model_validate(
        receipt
    )  # round-trips through the model unchanged


def test_build_skipped_receipt_has_the_exact_contract_shape() -> None:
    receipt = build_skipped_receipt(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        scenario_key="k2",
        scenario_id="sid-2",
    )
    assert receipt["scenario_attempt"] == 1
    assert receipt["world_index"] is None
    assert receipt["sub_goals"] == []
    assert receipt["evaluations"] == []
    assert receipt["call"] is None
    assert receipt["failure"] is None
    assert receipt["status"] == "skipped"


def test_skipped_shape_is_enforced_even_outside_the_synthesizing_helper() -> None:
    with pytest.raises(ValidationError):
        # world_index must be null for a skipped receipt -- even via the general-purpose builder.
        _passed_receipt(
            status=ScenarioStatus.SKIPPED,
            world_index=0,
            sub_goals=[],
            evaluations=[],
            call=None,
        )


def test_errored_receipt_requires_a_failure() -> None:
    with pytest.raises(ValidationError):
        _passed_receipt(status=ScenarioStatus.ERRORED, call=None, failure=None)

    errored = _passed_receipt(
        status=ScenarioStatus.ERRORED,
        call=None,
        failure={
            "domain": "agent",
            "stage": "running",
            "code": "agent_crashed",
            "message": "m",
        },
    )
    assert is_valid_digest(errored["digest"])


def test_receipt_evaluations_discriminated_union_rejects_a_mixed_up_kind() -> None:
    with pytest.raises(ValidationError):
        _passed_receipt(
            evaluations=[{"name": "n", "kind": "metric", "passed": True, "reason": "r"}]
        )


def test_receipt_call_timestamp_must_be_the_canonical_rfc3339_millis_form() -> None:
    with pytest.raises(ValidationError):
        _passed_receipt(
            call={
                **_passed_receipt()["call"],
                "started_at": "2026-08-25T10:14:03.412000+00:00",
            }
        )


def test_a_tampered_receipt_digest_is_rejected() -> None:
    receipt = _passed_receipt()
    receipt["digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError):
        ResultReceiptDraft.model_validate(receipt)


def test_receipt_digest_mismatch_names_the_unset_default_field() -> None:
    # N23: a caller whose raw `call` dict omits `recording_artifacts` (an optional field with a
    # default) computes a digest over an object without it; the model's own re-derivation fills in
    # `recording_artifacts: []`, so the two digests disagree. The error should name the differing
    # field -- a bare `receipt_digest_mismatch` was the review's named "opaque, undiagnosable" gap.
    call_without_recording_artifacts = {
        "started_at": format_rfc3339_millis(NOW),
        "ended_at": format_rfc3339_millis(LATER),
        "duration_ms": 100,
        "turns": 1,
        "transcript_artifact": None,
        # "recording_artifacts" intentionally omitted -- CallSummary fills it via default_factory.
    }
    with pytest.raises(ValidationError) as excinfo:
        build_result_receipt(
            job_id="j1",
            attempt_id="a1",
            attempt_number=1,
            scenario_key="k",
            scenario_id="sid",
            scenario_attempt=1,
            world_index=0,
            status=ScenarioStatus.PASSED,
            sub_goals=[{"name": "n", "held": True, "reason": None, "judged": False}],
            evaluations=[],
            call=call_without_recording_artifacts,
            failure=None,
        )
    assert "call.recording_artifacts" in str(excinfo.value)


# --- ArtifactManifestDraft / build_artifact_manifest ------------------------------------------------


def test_build_artifact_manifest_matches_the_contract_shaped_example() -> None:
    manifest = build_artifact_manifest(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        entries=[
            {
                "artifact_id": "sha256:" + "1" * 64,
                "kind": "result",
                "size": 1048576,
                "scenario_key": "k",
            }
        ],
        complete=True,
    )
    assert is_valid_digest(manifest["digest"])
    ArtifactManifestDraft.model_validate(manifest)


def test_manifest_complete_flag_changes_the_digest() -> None:
    entries = [
        {
            "artifact_id": "sha256:" + "2" * 64,
            "kind": "log",
            "size": 10,
            "scenario_key": None,
        }
    ]
    complete_manifest = build_artifact_manifest(
        job_id="j1", attempt_id="a1", attempt_number=1, entries=entries, complete=True
    )
    partial_manifest = build_artifact_manifest(
        job_id="j1", attempt_id="a1", attempt_number=1, entries=entries, complete=False
    )
    # "a later `complete: true` manifest is a distinct document that supersedes an earlier
    # `complete: false` one from the same attempt."
    assert complete_manifest["digest"] != partial_manifest["digest"]


def test_manifest_entry_rejects_a_malformed_artifact_id() -> None:
    with pytest.raises(ValidationError):
        build_artifact_manifest(
            job_id="j1",
            attempt_id="a1",
            attempt_number=1,
            entries=[
                {
                    "artifact_id": "not-a-digest",
                    "kind": "log",
                    "size": 1,
                    "scenario_key": None,
                }
            ],
            complete=True,
        )


# --- FakePlatform: in-process, no-sockets Transport ------------------------------------------------


class FakePlatform:
    """A `Transport` implementation backed by real, persistent server-side state -- no sockets, no
    `requests` -- so the channel clients above are exercised through the exact same interface
    `RequestsTransport` implements. `programmed` lets a test inject canned failures (an `Exception`
    instance to raise, or a `TransportResponse` to return) that are consumed FIFO before requests
    fall through to the real per-channel handlers below, which maintain idempotent state matching
    the contract's own dedupe rules (`event_id`, `(job_id, scenario_key)`, artifact digest,
    `(attempt_id, digest)`). `crash_after_next_write` lets a test simulate "the platform durably
    processed the write, but the response never reached the guest" -- the handler still records
    state and THEN raises `TransportError`, so the next request (whether inside the same client
    call's retry loop, or from a brand new client after a simulated restart) hits the idempotent
    path rather than reprocessing.
    """

    def __init__(
        self,
        *,
        events_url: str,
        results_url: str,
        artifacts_url: str,
        expected_token: str = "tok",
        expected_fence: str = "fence1",
    ) -> None:
        self.events_url = events_url
        self.results_url = results_url
        self.artifacts_url = artifacts_url
        self.manifest_url = artifacts_url + "manifest/"
        self.expected_token = expected_token
        self.expected_fence = expected_fence

        self._events_by_id: dict[str, dict[str, Any]] = {}
        self._watermark = 0
        self._receipts: dict[tuple[str, str], dict[str, Any]] = {}
        self._artifacts: dict[str, bytes] = {}
        self._manifests: dict[tuple[str, str], dict[str, Any]] = {}

        self.events_to_reject: set[str] = set()
        self.budget_remaining: int | None = None
        self.reserved_kinds = {"build", "transcript", "tool_trace", "result"}

        self.programmed: list[Exception | TransportResponse] = []
        self.crash_after_next_write = False
        self.calls: list[tuple[str, str]] = []
        self.received_headers: list[dict[str, str]] = []

    def queue(self, item: Exception | TransportResponse) -> None:
        self.programmed.append(item)

    def _maybe_crash(self, response: TransportResponse) -> TransportResponse:
        if self.crash_after_next_write:
            self.crash_after_next_write = False
            raise TransportError(
                "simulated crash: response lost after platform-side processing"
            )
        return response

    def _check_auth(self, headers: dict[str, str]) -> TransportResponse | None:
        # Strengthened per the union review: every real (non-programmed) request must carry the
        # bearer + fence -- exercising the auth-header wiring the contract mandates on every one of
        # the four endpoints, not just implicitly via "the call succeeded."
        if headers.get("Authorization") != f"Bearer {self.expected_token}":
            return TransportResponse(
                401,
                {
                    "error": "token_invalid",
                    "message": "missing/wrong bearer",
                    "retryable": False,
                },
                {},
            )
        if headers.get("X-Harness-Fence") != self.expected_fence:
            return TransportResponse(
                403,
                {
                    "error": "fence_mismatch",
                    "message": "missing/wrong fence",
                    "retryable": False,
                },
                {},
            )
        return None

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        data: Any = None,
        timeout: float = 30.0,
    ) -> TransportResponse:
        self.calls.append((method, url))
        self.received_headers.append(dict(headers))
        if self.programmed:
            item = self.programmed.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        auth_error = self._check_auth(headers)
        if auth_error is not None:
            return auth_error

        if method == "POST" and url == self.events_url:
            # EventsClient.flush() sends the batch as spooled bytes verbatim via `data=` (N19), not
            # `json_body=` -- decode it the same way the real platform would.
            if json_body is not None:
                parsed_body = json_body
            elif isinstance(data, (bytes, bytearray)):
                parsed_body = json.loads(data.decode("utf-8"))
            else:
                parsed_body = {}
            return self._handle_events(parsed_body)
        if method == "POST" and url == self.results_url:
            return self._handle_receipt(json_body or {})
        if method == "POST" and url == self.manifest_url:
            return self._handle_manifest(json_body or {})
        if method == "PUT" and url.startswith(self.artifacts_url):
            return self._handle_upload(url, headers, data)
        raise AssertionError(f"unrouted fake-platform request: {method} {url}")

    def _handle_events(self, body: dict[str, Any]) -> TransportResponse:
        assert body.get("schema_version") == "futureagi.harness-event.v1", (
            f"batch envelope missing/wrong schema_version: {body.get('schema_version')!r}"
        )
        rejected: list[dict[str, Any]] = []
        max_seq = self._watermark
        for event in body["events"]:
            sequence = event["sequence"]
            max_seq = max(max_seq, sequence)
            if event["event_id"] in self.events_to_reject:
                rejected.append(
                    {
                        "event_id": event["event_id"],
                        "sequence": sequence,
                        "code": "deterministic_rejection",
                        "message": "rejected by fake",
                    }
                )
                continue
            self._events_by_id[event["event_id"]] = event
        self._watermark = max(self._watermark, max_seq)
        return self._maybe_crash(
            TransportResponse(
                200,
                {"acked_through_sequence": self._watermark, "rejected": rejected},
                {},
            )
        )

    def _handle_receipt(self, body: dict[str, Any]) -> TransportResponse:
        # Strengthened per the union review: the real platform verifies the receipt digest --
        # recompute it the same way `whole_object_digest` does and reject a mismatch, so a test that
        # tampers with a receipt body actually exercises this path instead of it being silently
        # unenforced.
        core = {k: v for k, v in body.items() if k != "digest"}
        if body.get("digest") != whole_object_digest(core):
            return self._maybe_crash(
                TransportResponse(
                    422,
                    {"error": "digest_mismatch", "message": "m", "retryable": False},
                    {},
                )
            )
        key = (body["job_id"], body["scenario_key"])
        existing = self._receipts.get(key)
        if existing is not None:
            if existing["digest"] == body["digest"]:
                return self._maybe_crash(
                    TransportResponse(200, {"status": "duplicate"}, {})
                )
            if existing["attempt_number"] >= body["attempt_number"]:
                code = (
                    "attempt_superseded"
                    if existing["attempt_number"] > body["attempt_number"]
                    else "receipt_conflict"
                )
                return self._maybe_crash(
                    TransportResponse(
                        409, {"error": code, "message": "m", "retryable": False}, {}
                    )
                )
        self._receipts[key] = body
        return self._maybe_crash(TransportResponse(201, {"status": "stored"}, {}))

    def _handle_upload(
        self, url: str, headers: dict[str, str], data: Any
    ) -> TransportResponse:
        digest_hex = url[len(self.artifacts_url) :].rstrip("/")
        body_bytes = data if isinstance(data, (bytes, bytearray)) else b"".join(data)
        if digest_hex in self._artifacts:
            return self._maybe_crash(
                TransportResponse(200, {"status": "already_exists"}, {})
            )
        if hashlib.sha256(body_bytes).hexdigest() != digest_hex:
            return self._maybe_crash(
                TransportResponse(
                    422,
                    {"error": "digest_mismatch", "message": "m", "retryable": False},
                    {},
                )
            )
        kind = headers.get("X-Artifact-Kind")
        if (
            self.budget_remaining is not None
            and kind not in self.reserved_kinds
            and len(body_bytes) > self.budget_remaining
        ):
            return self._maybe_crash(
                TransportResponse(
                    413,
                    {
                        "error": "artifact_budget_exceeded",
                        "message": "m",
                        "retryable": False,
                    },
                    {},
                )
            )
        self._artifacts[digest_hex] = body_bytes
        if self.budget_remaining is not None:
            self.budget_remaining -= len(body_bytes)
        return self._maybe_crash(TransportResponse(201, {"status": "stored"}, {}))

    def _handle_manifest(self, body: dict[str, Any]) -> TransportResponse:
        # Strengthened per the union review: recompute and verify the manifest digest too.
        core = {k: v for k, v in body.items() if k != "digest"}
        if body.get("digest") != whole_object_digest(core):
            return self._maybe_crash(
                TransportResponse(
                    422,
                    {"error": "digest_mismatch", "message": "m", "retryable": False},
                    {},
                )
            )
        key = (body["attempt_id"], body["digest"])
        if key in self._manifests:
            return self._maybe_crash(
                TransportResponse(200, {"status": "duplicate"}, {})
            )
        for entry in body["entries"]:
            digest_hex = entry["artifact_id"].split(":", 1)[1]
            if digest_hex not in self._artifacts:
                return self._maybe_crash(
                    TransportResponse(
                        422,
                        {
                            "error": "artifact_unknown",
                            "message": entry["artifact_id"],
                            "retryable": False,
                        },
                        {},
                    )
                )
        self._manifests[key] = body
        return self._maybe_crash(TransportResponse(201, {"status": "stored"}, {}))


ENDPOINTS = {
    "events": "https://platform.example/events/",
    "results": "https://platform.example/results/",
    "artifacts": "https://platform.example/artifacts/",
    "scenarios": "https://platform.example/scenarios/",
}


def _capabilities() -> HostedCapabilities:
    return HostedCapabilities.model_validate(
        {
            "schema_version": CAPABILITIES_SCHEMA_VERSION,
            "job_id": "j1",
            "attempt_id": "a1",
            "attempt_number": 1,
            "fence": "fence1",
            "expires_at": "2099-01-01T00:00:00.000Z",
            "token": "tok",
            "endpoints": ENDPOINTS,
        }
    )


def _platform() -> FakePlatform:
    return FakePlatform(
        events_url=ENDPOINTS["events"],
        results_url=ENDPOINTS["results"],
        artifacts_url=ENDPOINTS["artifacts"],
    )


def _spooled_log_event(spool: OutboundSpool, event_id: str) -> None:
    record = build_event_record(
        event_id=event_id,
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        emitted_at=NOW,
        stage=HarnessStage.RUNNING,
        type=OutboundEventType.LOG,
        payload={"level": "info", "message": event_id},
    )
    spool.append(record)


@pytest.mark.parametrize(
    "channel", ["events", "results", "artifacts_upload", "artifacts_manifest"]
)
def test_every_channel_client_sends_authorization_and_fence_headers(
    tmp_path: Path, channel: str
) -> None:
    # Ranked missing test 8: one parametrized test over all four endpoints. `FakePlatform` now
    # enforces these headers itself (`_check_auth`) -- a wrong/missing one would 401/403 here, so a
    # passing run proves the header is both sent AND correct, not merely present in a recorded list.
    platform = _platform()
    capabilities = _capabilities()
    if channel == "events":
        spool = OutboundSpool(tmp_path, "events", sequenced=True)
        _spooled_log_event(spool, "event_x")
        result = EventsClient(
            capabilities, spool, platform, sleep=lambda s: None
        ).flush()
        assert result.error is None
    elif channel == "results":
        receipt = build_skipped_receipt(
            job_id="j1",
            attempt_id="a1",
            attempt_number=1,
            scenario_key="k",
            scenario_id="sid",
        )
        result = ResultsClient(capabilities, platform, sleep=lambda s: None).push(
            receipt
        )
        assert result.delivered
    elif channel == "artifacts_upload":
        data = b"auth header check"
        digest_hex = hashlib.sha256(data).hexdigest()
        result = ArtifactsClient(capabilities, platform, sleep=lambda s: None).upload(
            digest_hex, data, kind=ArtifactKind.LOG
        )
        assert result.delivered
    else:
        manifest = build_artifact_manifest(
            job_id="j1", attempt_id="a1", attempt_number=1, entries=[], complete=True
        )
        result = ArtifactsClient(
            capabilities, platform, sleep=lambda s: None
        ).push_manifest(manifest)
        assert result.delivered

    assert platform.received_headers  # at least one request actually happened
    for headers in platform.received_headers:
        assert headers.get("Authorization") == "Bearer tok"
        assert headers.get("X-Harness-Fence") == "fence1"


# --- EventsClient ------------------------------------------------------------------------------


def test_events_client_flush_delivers_and_advances_the_watermark(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(3):
        _spooled_log_event(spool, f"event_{i}")
    platform = _platform()
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    assert spool.watermark() == 0
    result = client.flush()
    assert result.error is None
    assert result.delivered_count == 3
    assert result.acked_through_sequence == 3
    assert result.rejected == []
    assert spool.watermark() == 3


def test_events_client_flush_is_a_no_op_transport_call_when_nothing_is_pending(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    platform = _platform()
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)
    result = client.flush()
    assert result.delivered_count == 0
    assert platform.calls == []


def test_events_client_never_advances_watermark_on_a_retryable_failure(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(TransportError("boom"))
    client = EventsClient(
        _capabilities(),
        spool,
        platform,
        sleep=lambda s: None,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    result = client.flush()
    assert result.error is not None
    assert result.error.outcome is ChannelOutcome.RETRYABLE
    assert (
        spool.watermark() == 0
    )  # "advancing the spool watermark only on confirmed delivery"


def test_events_client_transient_then_success_network_error_then_5xx(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(TransportError("connection refused"))
    platform.queue(
        TransportResponse(
            503, {"error": "service_unavailable", "message": "m", "retryable": True}, {}
        )
    )
    sleeps: list[float] = []
    client = EventsClient(
        _capabilities(), spool, platform, sleep=sleeps.append, rng=lambda: 1.0
    )

    result = client.flush()
    assert result.error is None
    assert result.delivered_count == 1
    assert sleeps == [
        1.0,
        2.0,
    ]  # base backoff, then doubled -- full jitter pinned to 1.0
    assert len(platform.calls) == 3


def test_events_client_429_honors_retry_after_over_computed_backoff(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(
            429,
            {"error": "rate_limited", "message": "m", "retryable": True},
            {"Retry-After": "9"},
        )
    )
    sleeps: list[float] = []
    client = EventsClient(
        _capabilities(), spool, platform, sleep=sleeps.append, rng=lambda: 1.0
    )

    result = client.flush()
    assert result.error is None
    assert sleeps == [9.0]


def test_events_client_retry_after_86400_is_clamped_to_max_backoff(
    tmp_path: Path,
) -> None:
    # N5, ranked missing test 6: an absurd Retry-After must not be honored verbatim -- clamp to
    # retry_policy.max_backoff_seconds (15.0 by default).
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(
            429,
            {"error": "rate_limited", "message": "m", "retryable": True},
            {"Retry-After": "86400"},
        )
    )
    sleeps: list[float] = []
    client = EventsClient(
        _capabilities(), spool, platform, sleep=sleeps.append, rng=lambda: 1.0
    )

    result = client.flush()
    assert result.error is None
    assert sleeps == [15.0]


def test_events_client_retry_after_negative_is_treated_as_absent(
    tmp_path: Path,
) -> None:
    # N5: a negative Retry-After must not reach `time.sleep` (ValueError) -- treated as absent, the
    # caller falls back to the computed full-jitter backoff instead.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(
            429,
            {"error": "rate_limited", "message": "m", "retryable": True},
            {"Retry-After": "-5"},
        )
    )
    sleeps: list[float] = []
    client = EventsClient(
        _capabilities(), spool, platform, sleep=sleeps.append, rng=lambda: 1.0
    )

    result = client.flush()
    assert result.error is None
    assert sleeps == [
        1.0
    ]  # the computed backoff for attempt 1, not -5.0 and not a crash


def test_perform_with_retry_deadline_refuses_a_new_attempt_once_elapsed() -> None:
    from fi.alk.harness.outbound import RetryPolicy, _perform_with_retry

    calls: list[int] = []

    def perform(attempt: int) -> TransportResponse:
        calls.append(attempt)
        return TransportResponse(
            503, {"error": "e", "message": "m", "retryable": True}, {}
        )

    response, error = _perform_with_retry(
        perform,
        retry_policy=RetryPolicy(max_attempts=8),
        sleep=lambda s: None,
        now=lambda: 100.0,
        deadline=99.0,  # already in the past relative to `now`
    )
    assert calls == []  # never even attempted once
    assert response is None
    assert error is not None
    assert error.code == "deadline_exceeded"


def test_perform_with_retry_deadline_clamps_the_sleep_to_the_remaining_budget() -> None:
    from fi.alk.harness.outbound import RetryPolicy, _perform_with_retry

    clock = [0.0]

    def now() -> float:
        return clock[0]

    def perform(_attempt: int) -> TransportResponse:
        return TransportResponse(
            503, {"error": "e", "message": "m", "retryable": True}, {}
        )

    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    _perform_with_retry(
        perform,
        retry_policy=RetryPolicy(
            max_attempts=8, initial_backoff_seconds=100.0, max_backoff_seconds=100.0
        ),
        sleep=sleep,
        rng=lambda: 1.0,
        now=now,
        deadline=5.0,  # far less than the ~100s computed backoff
    )
    assert sleeps[0] == 5.0  # clamped to what remained, not the full computed backoff


def test_events_client_deterministic_rejection_advances_watermark_but_never_retries(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_ok")
    _spooled_log_event(spool, "event_bad")
    platform = _platform()
    platform.events_to_reject = {"event_bad"}
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    result = client.flush()
    assert result.error is None
    assert result.acked_through_sequence == 2
    assert [item["event_id"] for item in result.rejected] == ["event_bad"]
    assert result.delivered_count == 1
    assert (
        spool.watermark() == 2
    )  # "the watermark is highest-processed -- accepted AND rejected"
    assert len(platform.calls) == 1  # never retried


def test_events_client_physically_drops_a_rejected_record_from_the_spool(
    tmp_path: Path,
) -> None:
    # P8 alignment: rejected-event handling must go through the spool's own drop(sequence), not
    # leave the rejected record sitting in the log forever.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_ok")
    _spooled_log_event(spool, "event_bad")
    platform = _platform()
    platform.events_to_reject = {"event_bad"}
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    client.flush()
    remaining_ids = [record.decode()["event_id"] for record in spool.records()]
    assert "event_bad" not in remaining_ids
    assert "event_ok" in remaining_ids


def test_events_client_flush_rejection_does_not_destroy_an_unacked_record_past_a_corrupt_spool(
    tmp_path: Path,
) -> None:
    # P1 (BLOCKER->MAJOR), reproduces the round-3 review's exact end-to-end scenario through the
    # public API: `_rewrite_retaining` (what the rejection-driven `drop_many` funnels through) used
    # to rewrite the log from ONLY the records-before-corruption prefix -- so an ordinary,
    # contract-defined rejection during `flush()` silently destroyed every intact record past the
    # corrupt byte too, including one the platform had NOT even acknowledged yet (the terminal
    # event, in the worst case). This is the N1 failure class (silent, zero-diagnostic loss of the
    # outbound event stream) reintroduced by the N8/N14 rework's interaction, and it is the one
    # thing round 3 was explicitly asked to break.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_1")

    path = tmp_path / "events.spool.jsonl"
    lines = path.read_bytes().split(b"\n")
    lines[0] = b"{not valid json but has a trailing newline"
    path.write_bytes(b"\n".join(lines))

    OutboundSpool._forget_for_tests(tmp_path, "events")
    recovered = OutboundSpool(tmp_path, "events", sequenced=True)
    assert recovered.is_corrupt

    _spooled_log_event(recovered, "event_bad")  # sequence 1, appended post-recovery
    _spooled_log_event(
        recovered, "event_terminal"
    )  # sequence 2, genuinely never yet acked

    platform = _platform()
    # Ack ONLY sequence 1 (rejected) -- sequence 2 (event_terminal) is deliberately left un-acked,
    # so it must still be reported pending after this flush, not silently vanish from disk.
    platform.queue(
        TransportResponse(
            200,
            {
                "acked_through_sequence": 1,
                "rejected": [
                    {"sequence": 1, "code": "event_type_unknown", "message": "m"}
                ],
            },
            {},
        )
    )
    client = EventsClient(_capabilities(), recovered, platform, sleep=lambda s: None)

    result = client.flush()
    assert result.error is None

    pending_ids = [r.decode()["event_id"] for r in recovered.pending_since_watermark()]
    assert pending_ids == ["event_terminal"], (
        "P1 regression: an un-acked record past a corrupt byte was destroyed by the rewrite the "
        "rejection triggered"
    )
    assert b"event_terminal" in path.read_bytes()


def test_events_client_rejects_an_untrusted_acked_through_sequence_and_leaves_watermark_unchanged(
    tmp_path: Path,
) -> None:
    # M7: acked_through_sequence is untrusted platform input -- a value outside the spool's trusted
    # range must not be applied; the flush still reports success (the HTTP call itself delivered).
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(200, {"acked_through_sequence": 10**9, "rejected": []}, {})
    )
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    result = client.flush()
    assert result.ack_out_of_range is True
    assert result.error is None
    assert spool.watermark() == 0


def test_events_client_rejected_sequence_outside_the_batch_is_ignored_not_orphaning(
    tmp_path: Path,
) -> None:
    # N1 (BLOCKER), ranked missing test 1: a `rejected[].sequence` the guest never sent in this
    # batch must be ignored -- not dropped, not used to move anything -- so it cannot orphan pending
    # records. Reproduces the union review's exact scenario: 10 spooled events, watermark forced to
    # 0, a rejected entry naming a sequence far outside [1, 10].
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(10):
        _spooled_log_event(spool, f"event_{i}")
    platform = _platform()
    platform.queue(
        TransportResponse(
            200,
            {
                "acked_through_sequence": 0,
                "rejected": [
                    {"event_id": "x9", "sequence": 10**9, "code": "c", "message": "m"}
                ],
            },
            {},
        )
    )
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    result = client.flush()
    assert result.error is None
    assert (
        result.rejected == []
    )  # the bogus entry never makes it into the reported rejections
    assert spool.watermark() == 0
    assert len(spool.pending_since_watermark()) == 10  # nothing orphaned
    assert len(spool.records()) == 10  # nothing physically dropped either


def test_events_client_ack_body_missing_acked_through_sequence_sets_ack_missing(
    tmp_path: Path,
) -> None:
    # N2/N3, ranked missing test 1: a 2xx with no acked_through_sequence at all must not silently
    # re-send the same batch forever -- ack_missing is set, watermark stays put, no exception.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(TransportResponse(200, {"rejected": []}, {}))
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    result = client.flush()
    assert result.error is None
    assert result.ack_missing is True
    assert spool.watermark() == 0


@pytest.mark.parametrize(
    "body",
    [
        {"acked_through_sequence": None, "rejected": []},
        {"acked_through_sequence": "abc", "rejected": []},
        {
            "acked_through_sequence": True,
            "rejected": [],
        },  # bool must not pass an int check
        {},
    ],
)
def test_events_client_hostile_ack_bodies_never_raise(
    tmp_path: Path, body: dict[str, Any]
) -> None:
    # N2, ranked missing test 1: a non-int/missing acked_through_sequence must produce a typed,
    # non-exception result, never crash the flusher (`TypeError`/`ValueError` from a bare `int()`).
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(TransportResponse(200, body, {}))
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    result = client.flush()  # must not raise
    assert result.error is None
    assert result.ack_missing is True
    assert spool.watermark() == 0


def test_events_client_rejected_is_not_a_list_never_raises(tmp_path: Path) -> None:
    # N2: `rejected: null` (or any non-list) must be treated as empty, not crash `sorted()`/iteration.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(200, {"acked_through_sequence": 1, "rejected": None}, {})
    )
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    result = client.flush()  # must not raise
    assert result.error is None
    assert result.rejected == []
    assert spool.watermark() == 1


def test_events_client_rejected_list_of_non_dicts_never_raises(tmp_path: Path) -> None:
    # N2: `rejected: ["oops"]` (strings, not objects) must not crash `entry.get("sequence")`.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(
            200, {"acked_through_sequence": 1, "rejected": ["oops", 5, None]}, {}
        )
    )
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    result = client.flush()  # must not raise
    assert result.error is None
    assert result.rejected == []
    assert spool.watermark() == 1


def test_events_client_503_503_404_does_not_raise_channel_failed(
    tmp_path: Path,
) -> None:
    # N6: interleaved 5xx must not shorten the 404 budget -- only 3 OBSERVED 404s (not 3 attempts
    # total) reach CHANNEL_FAILED. The ranked missing test names this sequence exactly.
    # max_attempts=3 stops the retry loop right after this exact sequence (rather than letting it
    # roll into a 4th, real-success attempt against FakePlatform, which would also prove the point
    # but less precisely) -- classify_response must still read the single 404 as RETRYABLE, not
    # CHANNEL_FAILED, because only ONE 404 has been observed.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(
            503, {"error": "service_unavailable", "message": "m", "retryable": True}, {}
        )
    )
    platform.queue(
        TransportResponse(
            503, {"error": "service_unavailable", "message": "m", "retryable": True}, {}
        )
    )
    platform.queue(
        TransportResponse(
            404, {"error": "not_found", "message": "m", "retryable": False}, {}
        )
    )
    client = EventsClient(
        _capabilities(),
        spool,
        platform,
        sleep=lambda s: None,
        retry_policy=RetryPolicy(max_attempts=3),
    )

    result = (
        client.flush()
    )  # must NOT raise HostedChannelFailedError -- only one 404 seen so far
    assert result.error is not None
    assert result.error.outcome is ChannelOutcome.RETRYABLE
    assert result.error.status_code == 404
    assert len(platform.calls) == 3


def test_events_client_503_503_404_404_404_raises_channel_failed(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(
            503, {"error": "service_unavailable", "message": "m", "retryable": True}, {}
        )
    )
    platform.queue(
        TransportResponse(
            503, {"error": "service_unavailable", "message": "m", "retryable": True}, {}
        )
    )
    for _ in range(3):
        platform.queue(
            TransportResponse(
                404, {"error": "not_found", "message": "m", "retryable": False}, {}
            )
        )
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    with pytest.raises(HostedChannelFailedError) as excinfo:
        client.flush()
    assert excinfo.value.error.domain is FailureDomain.PLATFORM_SYNC
    assert len(platform.calls) == 5


def test_events_client_413_on_the_events_channel_does_not_forever_loop(
    tmp_path: Path,
) -> None:
    # N7, ranked missing test 7: a 413 with no artifact_budget_exceeded body classifies as a
    # PERMANENT_ITEM (not a nonsensical BUDGET_EXCEEDED on a non-artifact channel) so the batch does
    # not silently re-send forever with no diagnostic.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(
            413,
            {"error": "request_entity_too_large", "message": "m", "retryable": False},
            {},
        )
    )
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    result = client.flush()
    assert result.error is not None
    assert result.error.outcome is ChannelOutcome.PERMANENT_ITEM
    assert result.error.status_code == 413


def test_events_client_413_halves_the_batch_and_retries(tmp_path: Path) -> None:
    # N7: a 413 that DOES report artifact_budget_exceeded (unexpected on this channel, but the
    # reactive halving does not care why) triggers a halve-and-retry rather than giving up outright.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(4):
        _spooled_log_event(spool, f"event_{i}")
    platform = _platform()
    platform.queue(
        TransportResponse(
            413,
            {"error": "artifact_budget_exceeded", "message": "m", "retryable": False},
            {},
        )
    )
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    result = client.flush()
    assert result.error is None
    # First attempt (batch of 4) 413s; second attempt (batch of 2) succeeds.
    assert result.acked_through_sequence == 2
    assert len(spool.pending_since_watermark()) == 2


def test_events_client_channel_state_latches_a_fence_and_stops_touching_the_transport(
    tmp_path: Path,
) -> None:
    # N10: once fenced, EVERY subsequent call on a client sharing this ChannelState raises the SAME
    # error immediately, without ever hitting the transport again.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    _spooled_log_event(spool, "event_y")
    platform = _platform()
    platform.queue(
        TransportResponse(
            401, {"error": "token_expired", "message": "m", "retryable": False}, {}
        )
    )
    state = ChannelState()
    client = EventsClient(
        _capabilities(), spool, platform, sleep=lambda s: None, channel_state=state
    )

    with pytest.raises(HostedFencedError):
        client.flush()
    calls_after_first = len(platform.calls)

    with pytest.raises(HostedFencedError) as excinfo:
        client.flush()
    assert len(platform.calls) == calls_after_first  # no new transport call at all
    assert excinfo.value.error.code == "token_expired"


def test_channel_state_shared_across_clients_fences_all_of_them(tmp_path: Path) -> None:
    # N10: the fence is per-ATTEMPT, not per-channel -- a fence observed on EventsClient must also
    # stop a ResultsClient sharing the same ChannelState, with no transport call from the second.
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(
            403, {"error": "fence_invalid", "message": "m", "retryable": False}, {}
        )
    )
    state = ChannelState()
    events_client = EventsClient(
        _capabilities(), spool, platform, sleep=lambda s: None, channel_state=state
    )
    results_client = ResultsClient(
        _capabilities(), platform, sleep=lambda s: None, channel_state=state
    )

    with pytest.raises(HostedFencedError):
        events_client.flush()
    calls_before = len(platform.calls)

    receipt = build_skipped_receipt(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        scenario_key="k",
        scenario_id="sid",
    )
    with pytest.raises(HostedFencedError) as excinfo:
        results_client.push(receipt)
    assert (
        len(platform.calls) == calls_before
    )  # results_client never touched the transport
    assert excinfo.value.error.code == "fence_invalid"


def test_events_client_401_raises_hosted_fenced_error(tmp_path: Path) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    platform.queue(
        TransportResponse(
            401, {"error": "token_expired", "message": "m", "retryable": False}, {}
        )
    )
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    with pytest.raises(HostedFencedError) as excinfo:
        client.flush()
    assert excinfo.value.error.code == "token_expired"
    assert spool.watermark() == 0


def test_events_client_404_exhausted_three_times_raises_hosted_channel_failed_error(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_x")
    platform = _platform()
    for _ in range(3):
        platform.queue(
            TransportResponse(
                404, {"error": "not_found", "message": "m", "retryable": False}, {}
            )
        )
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    with pytest.raises(HostedChannelFailedError) as excinfo:
        client.flush()
    assert excinfo.value.error.domain is FailureDomain.PLATFORM_SYNC
    assert len(platform.calls) == 3


def test_events_client_crash_between_send_and_ack_redelivers_safely_within_one_flush(
    tmp_path: Path,
) -> None:
    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    _spooled_log_event(spool, "event_crash")
    platform = _platform()
    platform.crash_after_next_write = True
    client = EventsClient(_capabilities(), spool, platform, sleep=lambda s: None)

    result = client.flush()
    assert result.error is None
    assert result.acked_through_sequence == 1
    assert spool.watermark() == 1
    assert len(platform.calls) == 2
    assert "event_crash" in platform._events_by_id


def test_events_client_batch_size_is_clamped_to_the_contract_max(
    tmp_path: Path,
) -> None:
    from fi.alk.harness.outbound import EVENTS_MAX_BATCH

    spool = OutboundSpool(tmp_path, "events", sequenced=True)
    for i in range(EVENTS_MAX_BATCH + 10):
        _spooled_log_event(spool, f"event_{i}")
    platform = _platform()
    client = EventsClient(
        _capabilities(), spool, platform, sleep=lambda s: None, batch_size=10_000
    )

    result = client.flush()
    assert result.error is None
    assert result.acked_through_sequence == EVENTS_MAX_BATCH
    assert spool.watermark() == EVENTS_MAX_BATCH
    # Everything past the batch cap is still pending for the next flush.
    assert len(spool.pending_since_watermark()) == 10


# --- ResultsClient -------------------------------------------------------------------------------


def test_results_client_pushes_a_receipt_successfully() -> None:
    platform = _platform()
    client = ResultsClient(_capabilities(), platform, sleep=lambda s: None)
    receipt = build_skipped_receipt(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        scenario_key="k",
        scenario_id="sid",
    )
    result = client.push(receipt)
    assert result.delivered
    assert not result.already_existed
    assert platform._receipts[("j1", "k")]["digest"] == receipt["digest"]


def test_results_client_permanent_item_is_never_retried() -> None:
    platform = _platform()
    platform.queue(
        TransportResponse(
            422, {"error": "artifact_unknown", "message": "m", "retryable": False}, {}
        )
    )
    client = ResultsClient(_capabilities(), platform, sleep=lambda s: None)
    receipt = build_skipped_receipt(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        scenario_key="k",
        scenario_id="sid",
    )

    result = client.push(receipt)
    assert not result.delivered
    assert result.error is not None
    assert result.error.outcome is ChannelOutcome.PERMANENT_ITEM
    assert result.error.code == "artifact_unknown"
    assert len(platform.calls) == 1


def test_results_client_401_raises_hosted_fenced_error() -> None:
    platform = _platform()
    platform.queue(
        TransportResponse(
            401, {"error": "token_expired", "message": "m", "retryable": False}, {}
        )
    )
    client = ResultsClient(_capabilities(), platform, sleep=lambda s: None)
    with pytest.raises(HostedFencedError):
        client.push(
            {
                "job_id": "j1",
                "scenario_key": "k",
                "digest": "sha256:" + "0" * 64,
                "attempt_number": 1,
            }
        )


def test_results_client_crash_between_send_and_ack_redelivers_within_one_push() -> None:
    platform = _platform()
    receipt = build_skipped_receipt(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        scenario_key="k",
        scenario_id="sid",
    )
    platform.crash_after_next_write = True
    client = ResultsClient(_capabilities(), platform, sleep=lambda s: None)

    result = client.push(receipt)
    assert result.delivered
    assert (
        result.already_existed
    )  # the retry loop's own second attempt saw the already-stored write
    assert len(platform.calls) == 2


def test_results_client_crash_between_send_and_ack_redelivers_across_a_simulated_restart() -> (
    None
):
    platform = _platform()
    receipt = build_skipped_receipt(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        scenario_key="k",
        scenario_id="sid",
    )
    platform.crash_after_next_write = True
    # max_attempts=1: the retry loop can't self-heal, so the crash surfaces as a RETRYABLE return
    # from this one push() call -- as if the guest process died right here.
    first_client = ResultsClient(
        _capabilities(),
        platform,
        sleep=lambda s: None,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    first = first_client.push(receipt)
    assert not first.delivered
    assert first.error is not None and first.error.outcome is ChannelOutcome.RETRYABLE

    # A brand new client (simulating a fresh process) retries the same receipt: idempotent duplicate.
    second_client = ResultsClient(_capabilities(), platform, sleep=lambda s: None)
    second = second_client.push(receipt)
    assert second.delivered
    assert second.already_existed
    assert len(platform.calls) == 2


def test_results_client_attempt_superseded_latches_the_shared_channel_state() -> None:
    # N22: 409 attempt_superseded is item-level PERMANENT_ITEM for the ONE call that received it
    # (contract-correct -- still returned normally, not raised), but folds into the N10 latch for
    # every SUBSEQUENT call: "a fenced attempt's in-flight requests cannot land after registration
    # of its successor" is a fence in substance.
    platform = _platform()
    platform.queue(
        TransportResponse(
            409, {"error": "attempt_superseded", "message": "m", "retryable": False}, {}
        )
    )
    state = ChannelState()
    client = ResultsClient(
        _capabilities(), platform, sleep=lambda s: None, channel_state=state
    )
    receipt = build_skipped_receipt(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        scenario_key="k",
        scenario_id="sid",
    )

    first = client.push(receipt)
    assert not first.delivered
    assert first.error is not None and first.error.code == "attempt_superseded"

    with pytest.raises(HostedAttemptSupersededError):
        client.push(receipt)
    assert len(platform.calls) == 1  # the second call never touched the transport


def test_build_event_record_redacts_a_dsn_in_world_unhealthy_cause() -> None:
    # N9, ranked missing test 9: outbound-channels.md v1.3's redaction obligation + seams v1.11
    # §3's "any outbound projection ... redacts userinfo" example DSN shape, verbatim.
    record = _event(
        OutboundEventType.WORLD_UNHEALTHY,
        HarnessStage.RUNNING,
        {
            "world_index": 2,
            "cause": "connect failed: postgresql://harness:s3cr3t@localhost:14000/w0",
        },
    )
    cause = record["payload"]["cause"]
    assert "s3cr3t" not in cause
    assert cause == "connect failed: postgresql://harness:***@localhost:14000/w0"
    # The digest must match the REDACTED bytes actually spooled, not the original with the secret.
    assert record["digest"] == event_payload_digest(record["payload"])


def test_build_event_record_redacts_terminal_failure_message() -> None:
    record = _event(
        OutboundEventType.TERMINAL,
        HarnessStage.FAILED,
        {
            "stage": "failed",
            "reason": None,
            "failure": {
                "domain": "infrastructure",
                "stage": "running",
                "code": "world_pool_exhausted",
                "message": "dial postgresql://harness:hunter2@localhost:14000/w0: connection refused",
            },
            "scenario_counts": {"passed": 0, "failed": 0, "errored": 0, "skipped": 0},
        },
    )
    message = record["payload"]["failure"]["message"]
    assert "hunter2" not in message
    assert "postgresql://harness:***@localhost:14000/w0" in message


def test_build_result_receipt_redacts_sub_goal_and_evaluation_reasons_and_failure_message() -> (
    None
):
    receipt = build_result_receipt(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        scenario_key="k",
        scenario_id="sid",
        scenario_attempt=1,
        world_index=None,
        status=ScenarioStatus.ERRORED,
        sub_goals=[
            {
                "name": "n",
                "held": None,
                "reason": "leaked postgresql://harness:pw123@host/db",
                "judged": False,
            }
        ],
        evaluations=[
            {
                "name": "n2",
                "kind": "metric",
                "score": 0.5,
                "reason": "saw postgresql://harness:pw123@host/db",
            }
        ],
        call=None,
        failure={
            "domain": "agent",
            "stage": "running",
            "code": "c",
            "message": "postgresql://harness:pw123@host/db",
        },
    )
    assert "pw123" not in str(receipt)
    assert (
        receipt["sub_goals"][0]["reason"] == "leaked postgresql://harness:***@host/db"
    )
    assert receipt["evaluations"][0]["reason"] == "saw postgresql://harness:***@host/db"
    assert receipt["failure"]["message"] == "postgresql://harness:***@host/db"
    # The digest must match the redacted bytes.
    ResultReceiptDraft.model_validate(receipt)  # round-trips cleanly, digest agrees


def test_redact_outbound_text_scrubs_userinfo_and_extra_secrets() -> None:
    assert (
        redact_outbound_text("postgresql://harness:pw@localhost/db")
        == "postgresql://harness:***@localhost/db"
    )
    assert redact_outbound_text("no secrets here") == "no secrets here"
    assert (
        redact_outbound_text("token=abc123 leaked", extra_secret_values=("abc123",))
        == "token=*** leaked"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # P3: the empty-username shape -- the canonical Redis/RabbitMQ/Mongo DSN form -- used to
        # leave the password verbatim (the old pattern required a NON-empty username group).
        ("redis://:secretpw@h:6379/0", "redis://:***@h:6379/0"),
        # A password containing '@' must not truncate the mask at the first '@' seen.
        ("mysql://u:p@ss@h/db", "mysql://u:***@h/db"),
        # A bare token/username-only userinfo (no ':') is the standard shape a bearer token takes
        # in git/registry output -- must be masked outright, not treated as "just a username."
        ("https://ghp_TOKEN@github.com/o/r.git", "https://***@github.com/o/r.git"),
        # Negative: a path segment containing '@' with no userinfo before it must be left alone.
        ("https://example.com/a@b", "https://example.com/a@b"),
        # Negative: no "://" at all -- not a userinfo shape.
        ("mailto:a@b", "mailto:a@b"),
    ],
)
def test_redact_outbound_text_handles_the_empty_username_and_bare_token_userinfo_shapes(
    raw: str, expected: str
) -> None:
    assert redact_outbound_text(raw) == expected


# --- ArtifactsClient -----------------------------------------------------------------------------


def test_artifacts_client_rejects_a_locally_wrong_digest_before_any_request() -> None:
    platform = _platform()
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    with pytest.raises(ValueError, match="artifact_digest_mismatch_local"):
        client.upload("0" * 64, b"hello world", kind=ArtifactKind.LOG)
    assert platform.calls == []


def test_artifacts_client_upload_new_then_already_exists() -> None:
    platform = _platform()
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    data = b"result contents"
    digest_hex = hashlib.sha256(data).hexdigest()

    first = client.upload(digest_hex, data, kind=ArtifactKind.RESULT, scenario_key="k1")
    assert first.delivered and not first.already_existed
    second = client.upload(
        digest_hex, data, kind=ArtifactKind.RESULT, scenario_key="k1"
    )
    assert second.delivered and second.already_existed


def test_artifacts_client_digest_mismatch_gets_exactly_one_re_upload() -> None:
    platform = _platform()
    platform.queue(
        TransportResponse(
            422, {"error": "digest_mismatch", "message": "m", "retryable": False}, {}
        )
    )
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    data = b"re-upload me"
    digest_hex = hashlib.sha256(data).hexdigest()

    result = client.upload(digest_hex, data, kind=ArtifactKind.LOG)
    assert result.delivered
    assert (
        len(platform.calls) == 2
    )  # the queued failure, then the one allowed re-upload


def test_artifacts_client_digest_mismatch_twice_is_permanent_the_scenario_is_errored() -> (
    None
):
    platform = _platform()
    for _ in range(2):
        platform.queue(
            TransportResponse(
                422,
                {"error": "digest_mismatch", "message": "m", "retryable": False},
                {},
            )
        )
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    data = b"re-upload me twice"
    digest_hex = hashlib.sha256(data).hexdigest()

    result = client.upload(digest_hex, data, kind=ArtifactKind.LOG)
    assert not result.delivered
    assert result.error is not None and result.error.code == "digest_mismatch"
    assert (
        len(platform.calls) == 2
    )  # exactly one re-upload attempted, per "re-upload once"


def test_artifacts_client_sends_x_artifact_size_derived_from_the_actual_bytes() -> None:
    # "X-Artifact-Size (authoritative; mismatch -> 422 size_mismatch)": this client never accepts a
    # caller-supplied size that could drift from the real payload -- it always derives the header
    # from `len(data)` itself, so a size mismatch against what's actually sent is structurally
    # unreachable from this client (the platform's own check remains the authority regardless).
    platform = _platform()
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    data = b"exactly this many bytes"
    digest_hex = hashlib.sha256(data).hexdigest()

    client.upload(digest_hex, data, kind=ArtifactKind.TRACE, scenario_key="k9")
    assert platform.received_headers[-1]["X-Artifact-Size"] == str(len(data))
    assert platform.received_headers[-1]["X-Artifact-Kind"] == "trace"
    assert platform.received_headers[-1]["X-Scenario-Key"] == "k9"


def test_artifacts_client_content_type_defaults_by_kind() -> None:
    # N17: recordings mp4, transcript a JSON array (per §3a "Formats") -- not a blanket octet-stream.
    platform = _platform()
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    data = b"recording bytes"
    digest_hex = hashlib.sha256(data).hexdigest()
    client.upload(digest_hex, data, kind=ArtifactKind.RECORDING_COMBINED)
    assert platform.received_headers[-1]["Content-Type"] == "video/mp4"

    data2 = b"transcript bytes"
    digest2 = hashlib.sha256(data2).hexdigest()
    client.upload(digest2, data2, kind=ArtifactKind.TRANSCRIPT)
    assert platform.received_headers[-1]["Content-Type"] == "application/json"

    data3 = b"trace bytes"
    digest3 = hashlib.sha256(data3).hexdigest()
    client.upload(digest3, data3, kind=ArtifactKind.TRACE)
    assert platform.received_headers[-1]["Content-Type"] == "application/octet-stream"


def test_artifacts_client_detects_wave_recording_bytes() -> None:
    platform = _platform()
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    data = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 20
    digest_hex = hashlib.sha256(data).hexdigest()

    client.upload(digest_hex, data, kind=ArtifactKind.RECORDING_COMBINED)

    assert platform.received_headers[-1]["Content-Type"] == "audio/wav"


def test_artifacts_client_content_type_override() -> None:
    platform = _platform()
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    data = b"custom type"
    digest_hex = hashlib.sha256(data).hexdigest()
    client.upload(digest_hex, data, kind=ArtifactKind.OTHER, content_type="text/plain")
    assert platform.received_headers[-1]["Content-Type"] == "text/plain"


def test_artifacts_client_latches_after_413_and_skips_non_reserved_without_the_transport() -> (
    None
):
    # N18: once a 413 artifact_budget_exceeded is observed, later non-reserved uploads must be
    # refused LOCALLY (never touching the transport again); reserved kinds keep going.
    platform = _platform()
    platform.budget_remaining = 1
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    first_data = b"too big for the tiny budget"
    first = client.upload(
        hashlib.sha256(first_data).hexdigest(), first_data, kind=ArtifactKind.TRACE
    )
    assert (
        first.error is not None
        and first.error.outcome is ChannelOutcome.BUDGET_EXCEEDED
    )
    calls_after_413 = len(platform.calls)

    second_data = b"another non-reserved upload"
    second = client.upload(
        hashlib.sha256(second_data).hexdigest(), second_data, kind=ArtifactKind.LOG
    )
    assert not second.delivered
    assert (
        second.error is not None
        and second.error.outcome is ChannelOutcome.BUDGET_EXCEEDED
    )
    assert len(platform.calls) == calls_after_413  # no new transport call

    reserved_data = b"reserved kind always goes through"
    reserved_digest = hashlib.sha256(reserved_data).hexdigest()
    third = client.upload(reserved_digest, reserved_data, kind=ArtifactKind.RESULT)
    assert third.delivered  # reserved kinds are never latched out
    assert len(platform.calls) == calls_after_413 + 1


def test_artifacts_client_413_budget_exceeded_is_never_retried() -> None:
    platform = _platform()
    platform.budget_remaining = 5
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    data = b"too big for the budget"
    digest_hex = hashlib.sha256(data).hexdigest()

    result = client.upload(digest_hex, data, kind=ArtifactKind.LOG)
    assert not result.delivered
    assert result.error is not None
    assert result.error.outcome is ChannelOutcome.BUDGET_EXCEEDED
    assert len(platform.calls) == 1


def test_artifacts_client_chunks_uploads_over_the_threshold_and_reassembles_correctly() -> (
    None
):
    platform = _platform()
    client = ArtifactsClient(
        _capabilities(),
        platform,
        sleep=lambda s: None,
        chunk_threshold_bytes=10,
        chunk_size_bytes=4,
    )
    data = b"0123456789abcdef"
    digest_hex = hashlib.sha256(data).hexdigest()

    result = client.upload(digest_hex, data, kind=ArtifactKind.TRACE)
    assert result.delivered
    assert platform._artifacts[digest_hex] == data


def test_artifacts_client_crash_between_send_and_ack_redelivers_as_already_exists() -> (
    None
):
    platform = _platform()
    data = b"artifact bytes" * 100
    digest_hex = hashlib.sha256(data).hexdigest()
    platform.crash_after_next_write = True
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)

    result = client.upload(digest_hex, data, kind=ArtifactKind.TRACE)
    assert result.delivered
    assert result.already_existed
    assert len(platform.calls) == 2
    assert platform._artifacts[digest_hex] == data


def test_artifacts_client_push_manifest_is_idempotent_and_checks_references() -> None:
    platform = _platform()
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    data = b"manifest target"
    digest_hex = hashlib.sha256(data).hexdigest()
    client.upload(digest_hex, data, kind=ArtifactKind.RESULT)

    manifest = build_artifact_manifest(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        entries=[
            {
                "artifact_id": f"sha256:{digest_hex}",
                "kind": "result",
                "size": len(data),
                "scenario_key": "k",
            }
        ],
        complete=True,
    )
    first = client.push_manifest(manifest)
    assert first.delivered and not first.already_existed
    second = client.push_manifest(manifest)
    assert (
        second.delivered and second.already_existed
    )  # idempotent on (attempt_id, digest)


def test_artifacts_client_push_manifest_rejects_an_unknown_artifact_reference() -> None:
    platform = _platform()
    client = ArtifactsClient(_capabilities(), platform, sleep=lambda s: None)
    manifest = build_artifact_manifest(
        job_id="j1",
        attempt_id="a1",
        attempt_number=1,
        entries=[
            {
                "artifact_id": "sha256:" + "f" * 64,
                "kind": "result",
                "size": 1,
                "scenario_key": "k",
            }
        ],
        complete=True,
    )
    result = client.push_manifest(manifest)
    assert not result.delivered
    assert result.error is not None and result.error.code == "artifact_unknown"
