"""FutureAGIResultSink — local write + real platform submission.

Composes ``LocalFilesystemResultSink`` and adds ``submit(...)`` that POSTs
report data to the Future AGI platform using the ALK ingestion endpoints:

  POST  /simulate/alk-simulate/run-tests/{run_test_id}/test-executions/
  POST  /simulate/alk-simulate/test-executions/{test_execution_id}/batch/
  PATCH /simulate/alk-simulate/call-executions/{call_execution_id}/result/

Configuration is env-driven so local runs stay unaffected when the platform
target is not set. Submission accepts either the internal service bearer or
the external API-key pair:

  FI_BASE_URL / FUTURE_AGI_API_URL / AGENT_LEARNING_API_URL — base URL
  FI_API_KEY / FUTURE_AGI_API_KEY / AGENT_LEARNING_API_KEY — x-api-key
  FI_SECRET_KEY / FUTURE_AGI_SECRET_KEY / AGENT_LEARNING_SECRET_KEY — x-secret-key
  FI_RUN_TEST_ID / FUTURE_AGI_RUN_TEST_ID / AGENT_LEARNING_RUN_TEST_ID — target run test
  FI_TEST_EXECUTION_ID / … — optional pre-created TestExecution (hosted runs); when
    set the sink submits into it instead of creating one from the run test.

When any of those are absent the sink records ``status: "not_configured"``
in ``submission.json`` and returns cleanly — no HTTP is attempted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from fi.simulate.runtime import (
    CanonicalEvent,
    SimulationPlan,
    SimulationReport,
    SimulationSpec,
)

from .filesystem import LocalFilesystemResultSink

logger = logging.getLogger("fi.simulate.results.futureagi")

_STATUS_MAP = {
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "timed_out": "failed",
    "agent_unavailable": "failed",
}
_API_KEY_ENV = ("FI_API_KEY", "FUTURE_AGI_API_KEY", "AGENT_LEARNING_API_KEY")
_SECRET_KEY_ENV = (
    "FI_SECRET_KEY",
    "FUTURE_AGI_SECRET_KEY",
    "AGENT_LEARNING_SECRET_KEY",
)
_INTERNAL_SECRET_ENV = (
    "FI_INTERNAL_SUBMIT_SECRET",
    "ALK_RUNNER_INTERNAL_SECRET",
    "INTERNAL_API_SECRET",
)
_API_URL_ENV = ("FI_BASE_URL", "FUTURE_AGI_API_URL", "AGENT_LEARNING_API_URL")
_RUN_TEST_ID_ENV = (
    "FI_RUN_TEST_ID",
    "FUTURE_AGI_RUN_TEST_ID",
    "AGENT_LEARNING_RUN_TEST_ID",
)
_TEST_EXECUTION_ID_ENV = (
    "FI_TEST_EXECUTION_ID",
    "FUTURE_AGI_TEST_EXECUTION_ID",
    "AGENT_LEARNING_TEST_EXECUTION_ID",
)
_HTTP_TIMEOUT_SECONDS = 60.0
_RECORDING_UPLOAD_TIMEOUT_SECONDS = 300.0
_CONTENT_TYPE_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".m4a": "audio/mp4",
}


class FutureAGIResultSink:
    """Local sink + platform submission over HTTP."""

    def __init__(
        self,
        *,
        root: str | Path = ".fagi/runs",
        api_url: str | None = None,
        api_key_env: tuple[str, ...] = _API_KEY_ENV,
        secret_key_env: tuple[str, ...] = _SECRET_KEY_ENV,
        run_test_id: str | None = None,
        test_execution_id: str | None = None,
    ) -> None:
        self._local = LocalFilesystemResultSink(root=root)
        self._api_url = api_url or _first_env(_API_URL_ENV)
        self._api_key_env = api_key_env
        self._secret_key_env = secret_key_env
        self._run_test_id = run_test_id or _first_env(_RUN_TEST_ID_ENV)
        self._test_execution_id = test_execution_id or _first_env(
            _TEST_EXECUTION_ID_ENV
        )
        self._event_count = 0
        self._spec: SimulationSpec | None = None
        self._plan: SimulationPlan | None = None
        # Streaming state (hosted runs only). ``begin_stream`` opens the client
        # and allocates rows up front; ``submit_case`` PATCHes one row by index as
        # its case finishes; ``finalize_stream`` reconciles + closes.
        self._streaming = False
        self._stream_client: httpx.Client | None = None
        self._stream_call_ids: list[str] = []
        self._streamed_indices: set[int] = set()
        self._stream_failures: dict[int, dict[str, Any]] = {}

    @property
    def run_directory(self) -> Path | None:
        return self._local.run_directory

    def prepare(
        self,
        spec: SimulationSpec,
        plan: SimulationPlan | None = None,
    ) -> Path:
        self._spec = spec
        self._plan = plan
        self._event_count = 0
        return self._local.prepare(spec, plan)

    def write_event(self, event: CanonicalEvent) -> None:
        self._event_count += 1
        self._local.write_event(event)

    def write_report(self, report: SimulationReport) -> Path:
        report_path = self._local.write_report(report)
        # When streaming, cases were already PATCHed one-by-one as they finished;
        # finalize only reconciles the stragglers and writes submission.json.
        # Otherwise the whole report is submitted here in one batch (local/chat).
        if self._streaming:
            self.finalize_stream(report)
        else:
            self.submit(report)
        return report_path

    def begin_stream(
        self,
        spec: SimulationSpec,
        plan: SimulationPlan | None = None,
    ) -> bool:
        """Open a run-scoped submission session for per-case streaming.

        Hosted only: a pre-created ``test_execution_id`` is the signal. Local and
        chat runs (no pre-created execution) return ``False`` and keep the
        batch-at-end path untouched. On any setup error we also return ``False``
        and fall back to that path — streaming setup must never break the run.
        """
        if not self._test_execution_id:
            return False
        api_key = _first_env(self._api_key_env)
        secret_key = _first_env(self._secret_key_env)
        internal_secret = _first_env(_INTERNAL_SECRET_ENV)
        if _missing_config(
            api_url=self._api_url,
            api_key=api_key,
            secret_key=secret_key,
            internal_secret=internal_secret,
            run_test_id=self._run_test_id,
        ):
            return False
        try:
            client = _open_client(self._api_url, api_key, secret_key, internal_secret)
            call_ids = _allocate_call_ids(client, self._test_execution_id)
        except Exception:
            if self._stream_client is not None:
                self._stream_client.close()
            self._stream_client = None
            return False

        self._stream_client = client
        self._stream_call_ids = call_ids
        self._streamed_indices = set()
        self._stream_failures = {}
        self._streaming = True
        return True

    def submit_case(self, index: int, case: Any) -> None:
        """PATCH one finished case into its pre-allocated CallExecution row.

        Called off the event loop (``asyncio.to_thread``) from the runner's
        per-case callback; ``httpx.Client`` is thread-safe, so the run-scoped
        client is shared across concurrent case submissions. Failures are logged
        and left for ``finalize_stream`` to reconcile — never raised.
        """
        if not self._streaming or self._stream_client is None:
            return
        if index >= len(self._stream_call_ids):
            # More results than allocated rows: the platform under-provisioned.
            # The batch path drops these silently via ``zip``; record it here so
            # submission.json shows the drop.
            self._stream_failures[index] = {
                "index": index,
                "reason": "no_allocated_row",
            }
            return
        call_id = self._stream_call_ids[index]
        try:
            payload = _build_result_payload(case)
            recording_url = _maybe_upload_recording(self._stream_client, call_id, case)
            if recording_url:
                payload["recording_url"] = recording_url
            stereo_url = _maybe_upload_stereo_recording(
                self._stream_client, call_id, case
            )
            if stereo_url:
                payload["stereo_recording_url"] = stereo_url
            _attach_channel_recordings(self._stream_client, call_id, case, payload)
            _stamp_result_digest(payload)
            resp = self._stream_client.patch(
                f"/simulate/api/alk-simulate/call-executions/{call_id}/result/",
                json=payload,
            )
            if resp.is_error:
                self._stream_failures[index] = {
                    "index": index,
                    "call_execution_id": call_id,
                    "status_code": resp.status_code,
                    "body": _safe_body(resp),
                }
                logger.warning(
                    "case submission http error",
                    extra={
                        "case_index": index,
                        "call_execution_id": call_id,
                        "status_code": resp.status_code,
                    },
                )
                return
            self._streamed_indices.add(index)
            self._stream_failures.pop(index, None)
        except Exception as exc:
            self._stream_failures[index] = {
                "index": index,
                "call_execution_id": call_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            logger.warning(
                "case submission failed",
                extra={
                    "case_index": index,
                    "call_execution_id": call_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    def case_started(self, index: int) -> None:
        """PATCH a pre-allocated CallExecution row to ONGOING the moment its case
        starts, so the platform shows progress instead of PENDING → terminal.

        Purely cosmetic and best-effort. Unlike ``submit_case`` this must NOT
        record into ``_stream_failures`` / ``_streamed_indices`` — those drive
        ``finalize_stream`` result reconciliation, and a missed status ping is not
        a missed result. The backend gates the update on PENDING, so a lost, late,
        or duplicate ping can never overwrite a terminal result; failures are
        swallowed.
        """
        if not self._streaming or self._stream_client is None:
            return
        if index >= len(self._stream_call_ids):
            return
        call_id = self._stream_call_ids[index]
        try:
            self._stream_client.patch(
                f"/simulate/api/alk-simulate/call-executions/{call_id}/status/",
                json={"status": "ongoing"},
            )
        except Exception:
            # Non-authoritative: never let a status ping disturb the run or the
            # reconciliation bookkeeping.
            pass

    def finalize_stream(self, report: SimulationReport) -> dict[str, Any]:
        """Reconcile any case the stream missed, then close the session.

        Runs after the engine returns, so no cases are in flight. Any index not
        already streamed (PATCH failure, or a whole-report failure that never fired
        callbacks) is retried here. ``submission.json`` records ``status:
        submitted`` when at least one case landed and ``failed`` when none did —
        ``child_entrypoint`` gates job success on that field.
        """
        run_directory = self._local.run_directory
        for index, case in enumerate(report.test_cases):
            if index in self._streamed_indices:
                continue
            self.submit_case(index, case)

        submitted = [self._stream_call_ids[i] for i in sorted(self._streamed_indices)]
        failed = [detail for _, detail in sorted(self._stream_failures.items())]
        # Every allocated row failing to submit is a failed submission — not a
        # green job with zero results landed (the batch path signalled this by
        # letting the exception propagate to ``submit``). Partial failures stay
        # "submitted": the cases that did land are real.
        all_failed = (
            bool(report.test_cases)
            and bool(self._stream_call_ids)
            and not self._streamed_indices
        )
        submission: dict[str, Any] = {
            "schema_version": "futureagi.submission.v1",
            "run_id": report.run_id,
            "report_hash": report.report_hash,
            "test_cases": len(report.test_cases),
            "events_recorded": self._event_count,
            "api_url": self._api_url,
            "run_test_id": self._run_test_id,
            "test_execution_id": self._test_execution_id,
            "streamed": True,
            "status": "failed" if all_failed else "submitted",
            "allocated_call_executions": list(self._stream_call_ids),
            "submitted_call_executions": submitted,
            "failed_call_executions": failed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if all_failed:
            submission["reason"] = "stream_all_cases_failed"
        if self._stream_client is not None:
            self._stream_client.close()
            self._stream_client = None
        self._streaming = False
        if run_directory is not None:
            _write_submission(run_directory, submission)
        return submission

    def submit(self, report: SimulationReport) -> dict[str, Any]:
        run_directory = self._local.run_directory
        if run_directory is None:
            raise RuntimeError("result_sink_not_prepared")

        api_key = _first_env(self._api_key_env)
        secret_key = _first_env(self._secret_key_env)
        internal_secret = _first_env(_INTERNAL_SECRET_ENV)

        submission: dict[str, Any] = {
            "schema_version": "futureagi.submission.v1",
            "run_id": report.run_id,
            "report_hash": report.report_hash,
            "test_cases": len(report.test_cases),
            "artifact_count": len(report.artifacts.entries),
            "events_recorded": self._event_count,
            "api_url": self._api_url,
            "run_test_id": self._run_test_id,
            "test_execution_id": self._test_execution_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        missing = _missing_config(
            api_url=self._api_url,
            api_key=api_key,
            secret_key=secret_key,
            internal_secret=internal_secret,
            run_test_id=self._run_test_id,
        )
        if missing:
            submission["status"] = "not_configured"
            submission["reason"] = "missing_config: " + ",".join(missing)
            logger.warning(
                "submission not configured", extra={"missing": ",".join(missing)}
            )
            _write_submission(run_directory, submission)
            return submission

        try:
            outcome = _submit_via_http(
                report=report,
                base_url=self._api_url,
                api_key=api_key,
                secret_key=secret_key,
                internal_secret=internal_secret,
                run_test_id=self._run_test_id,
                test_execution_id=self._test_execution_id,
            )
            submission.update(outcome)
            submission["status"] = "submitted"
            logger.info(
                "submission ok",
                extra={
                    "run_test_id": self._run_test_id,
                    "test_execution_id": self._test_execution_id,
                },
            )
        except Exception as exc:
            submission["status"] = "failed"
            submission["reason"] = f"submission_error: {exc.__class__.__name__}: {exc}"
            logger.error(
                "submission failed",
                extra={
                    "run_test_id": self._run_test_id,
                    "test_execution_id": self._test_execution_id,
                    "error": f"{exc.__class__.__name__}: {exc}",
                },
            )

        _write_submission(run_directory, submission)
        return submission


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _missing_config(
    *,
    api_url: str | None,
    api_key: str | None,
    secret_key: str | None,
    internal_secret: str | None,
    run_test_id: str | None,
) -> list[str]:
    missing: list[str] = []
    if not api_url:
        missing.append("api_url")
    if not internal_secret:
        if not api_key:
            missing.append("api_key")
        if not secret_key:
            missing.append("secret_key")
    if not run_test_id:
        missing.append("run_test_id")
    return missing


def _open_client(
    base_url: str,
    api_key: str | None,
    secret_key: str | None,
    internal_secret: str | None = None,
) -> httpx.Client:
    """Build the ALK ingestion HTTP client (shared by batch + streaming paths).

    No client-level Content-Type: httpx sets application/json for ``json=`` calls
    and multipart/form-data (with boundary) for the ``files=`` recording upload.
    A fixed application/json here silently breaks the multipart upload.
    """
    headers: dict[str, str] = {}
    if api_key and secret_key:
        headers.update({"x-api-key": api_key, "x-secret-key": secret_key})
    # These are alternative authentication modes.  A developer/hosted-runner
    # environment can legitimately contain both sets of variables; sending
    # both lets the backend's bearer authenticator win and loses the customer
    # organization carried by the API-key pair.  Prefer the explicit customer
    # identity whenever it is complete, and use the internal token only as the
    # service-to-service fallback.
    elif internal_secret:
        headers["Authorization"] = f"Bearer {internal_secret}"
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers=headers,
        timeout=_HTTP_TIMEOUT_SECONDS,
    )


def _ensure_test_execution(client: httpx.Client, run_test_id: str) -> str:
    """Create a TestExecution from the run test (local runs only)."""
    start = client.post(
        f"/simulate/api/alk-simulate/run-tests/{run_test_id}/test-executions/",
        json={},
    )
    start.raise_for_status()
    return _unwrap(start.json())["test_execution_id"]


def _allocate_call_ids(
    client: httpx.Client,
    test_execution_id: str,
    count: int | None = None,
) -> list[str]:
    """Claim exact local-report rows, or all pre-created hosted rows."""
    if count is not None and count <= 0:
        return []

    call_execution_ids: list[str] = []
    for _ in range(64):  # hard cap to prevent runaway
        remaining = count - len(call_execution_ids) if count is not None else None
        resp = client.post(
            f"/simulate/api/alk-simulate/test-executions/{test_execution_id}/batch/",
            json={"count": remaining} if remaining is not None else {},
        )
        resp.raise_for_status()
        body = _unwrap(resp.json())
        allocated = body["call_execution_ids"]
        if remaining is not None and len(allocated) > remaining:
            raise RuntimeError("backend allocated more call executions than requested")
        call_execution_ids.extend(allocated)
        if count is not None and len(call_execution_ids) == count:
            break
        if not allocated or not body.get("has_more"):
            if count is None:
                break
            raise RuntimeError(
                f"backend allocated {len(call_execution_ids)} of {count} required call executions"
            )

    if count is not None and len(call_execution_ids) != count:
        raise RuntimeError(
            f"backend allocated {len(call_execution_ids)} of {count} required call executions"
        )
    return call_execution_ids


def _submit_via_http(
    *,
    report: SimulationReport,
    base_url: str,
    api_key: str | None,
    secret_key: str | None,
    run_test_id: str,
    internal_secret: str | None = None,
    test_execution_id: str | None = None,
) -> dict[str, Any]:
    with _open_client(base_url, api_key, secret_key, internal_secret) as client:
        # Hosted runs submit into a TestExecution the platform pre-created; local
        # runs create one here from the run test.
        if not test_execution_id:
            test_execution_id = _ensure_test_execution(client, run_test_id)

        call_execution_ids = _allocate_call_ids(
            client,
            test_execution_id,
            len(report.test_cases),
        )

        submitted_ids: list[str] = []
        failed: list[dict[str, Any]] = []
        for call_id, case in zip(call_execution_ids, report.test_cases):
            payload = _build_result_payload(case)
            recording_url = _maybe_upload_recording(client, call_id, case)
            if recording_url:
                payload["recording_url"] = recording_url
            stereo_url = _maybe_upload_stereo_recording(client, call_id, case)
            if stereo_url:
                payload["stereo_recording_url"] = stereo_url
            _attach_channel_recordings(client, call_id, case, payload)
            _stamp_result_digest(payload)
            resp = client.patch(
                f"/simulate/api/alk-simulate/call-executions/{call_id}/result/",
                json=payload,
            )
            if resp.is_error:
                failed.append(
                    {
                        "call_execution_id": call_id,
                        "status_code": resp.status_code,
                        "body": _safe_body(resp),
                    }
                )
            else:
                submitted_ids.append(call_id)

    return {
        "test_execution_id": test_execution_id,
        "allocated_call_executions": call_execution_ids,
        "submitted_call_executions": submitted_ids,
        "failed_call_executions": failed,
    }


def _unwrap(body: Any) -> dict[str, Any]:
    if isinstance(body, dict) and "result" in body and isinstance(body["result"], dict):
        return body["result"]
    if isinstance(body, dict):
        return body
    raise ValueError(f"unexpected_response_shape: {body!r}")


def _safe_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text[:500]


def _build_result_payload(case) -> dict[str, Any]:
    """Map a SimulationTestCaseResult into the ALK ingestion PATCH body.

    Backend derives conversation metrics and CSAT from the transcript, so
    the SDK only ships what it directly observed.
    """
    payload: dict[str, Any] = {
        "status": _STATUS_MAP.get(case.status.value, "failed"),
    }

    started_at = case.started_at
    ended_at = case.ended_at
    # Case-level timestamps are unset for LiveKit runs (the engine does not
    # stamp them). Fall back to the observed speech timing carried on each
    # message so duration/start-time populate on the platform.
    if started_at is None or ended_at is None:
        speech_start, speech_end = _speech_bounds(case)
        started_at = started_at or speech_start
        ended_at = ended_at or speech_end

    if started_at is not None:
        payload["started_at"] = started_at.isoformat()
    if ended_at is not None:
        payload["ended_at"] = ended_at.isoformat()
    if started_at is not None and ended_at is not None:
        payload["duration_seconds"] = max(
            int((ended_at - started_at).total_seconds()), 0
        )

    if case.failure is not None:
        payload["ended_reason"] = case.failure.code
        payload["error_message"] = case.failure.message or ""

    result = case.result
    transcript_segments: list[dict[str, Any]] = []
    if result is not None:
        transcript_segments = _extract_transcript_segments(result)
        if transcript_segments:
            payload["transcript"] = transcript_segments
        if "ended_reason" not in payload:
            stop_reason = result.metadata.get("stop_reason")
            if isinstance(stop_reason, str) and stop_reason:
                payload["ended_reason"] = stop_reason

        recording_uri = _extract_recording_uri(result)
        if recording_uri:
            payload["recording_url"] = recording_uri

        provider_call_data: dict[str, Any] = {}
        existing_pcd = result.metadata.get("provider_call_data")
        if isinstance(existing_pcd, dict):
            provider_call_data = dict(existing_pcd)

        # Fold the target agent's provider-reported usage/cost (captured by the
        # SDK evidence layer — Vapi costBreakdown, Retell call_cost, LiveKit
        # usage) into provider_call_data under the normalized ``usage.llm``
        # shape the platform already reads for native voice. This is the
        # agent-under-test's real usage — not the FutureAGI simulator's.
        target = _target_provider_usage(case)
        if target is not None:
            provider_bucket = dict(provider_call_data.get(target.provider) or {})
            if target.usage:
                provider_bucket["usage"] = {
                    **(provider_bucket.get("usage") or {}),
                    "llm": target.usage,
                }
            if target.raw:
                provider_bucket.setdefault("costBreakdown", target.raw)
            if provider_bucket:
                provider_call_data[target.provider] = provider_bucket
            if target.cost_cents is not None:
                payload["costs"] = {"cost_cents": target.cost_cents}

        # Every LiveKit-engine run carries a truthy ``livekit`` marker so the
        # platform's SpeakerRoleResolver detects the provider as LiveKit — its
        # role map is direction-independent and already matches the SDK's
        # tested-agent-perspective transcript. Without this a black-box target
        # (no usage evidence) leaves ``provider_call_data`` empty, the platform
        # falls back to VAPI, and an inbound default swaps agent/customer labels.
        # A falsy ``{}`` is not enough — ``detect_provider`` treats it as absent.
        if str(
            result.metadata.get("engine")
        ) == "livekit" and not provider_call_data.get("livekit"):
            provider_call_data["livekit"] = {"engine": "livekit"}

        if provider_call_data:
            payload["provider_call_data"] = provider_call_data

        summary = result.metadata.get("call_summary") or result.metadata.get("summary")
        if isinstance(summary, str) and summary:
            payload["call_summary"] = summary

        call_metadata = {
            k: v
            for k, v in result.metadata.items()
            if k
            not in {
                "provider_call_data",
                "call_summary",
                "summary",
                "failure",
                "status",
                "test_case_id",
                "run_id",
            }
        }
        if call_metadata:
            payload["call_metadata"] = _json_safe(call_metadata)

    return payload


def _stamp_result_digest(payload: dict[str, Any]) -> None:
    """Bind an idempotency key to exactly the canonical payload being submitted."""
    payload.pop("result_digest", None)
    payload["result_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
    )


def _extract_transcript_segments(result) -> list[dict[str, Any]]:
    """Convert TestCaseResult.messages into ALK transcript segments.

    LiveKit engine emits each message with ``started_speaking_at`` and
    ``stopped_speaking_at`` (seconds since epoch, from ``ChatMessage.metrics``).
    We convert to millisecond offsets relative to the first speech timestamp
    so ``ConversationMetricsCalculator`` can compute overlap-based interruption
    counts, WPM and talk-ratio on the backend.
    """
    segments: list[dict[str, Any]] = []
    typed_messages = [msg for msg in result.messages if isinstance(msg, dict)]

    anchor = _first_speech_anchor(typed_messages)
    for msg in typed_messages:
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = _normalize_tool_calls(msg.get("tool_calls"))
        start_ms, end_ms = _resolve_message_timing_ms(msg, anchor)
        latency_ms = _message_latency_ms(msg)

        if role == "assistant":
            # An assistant turn can carry text, tool calls, or both. Tool-call
            # turns usually have empty content — emit them anyway as a
            # ``tool_calls`` segment so the agent's real tool activity survives
            # ingestion instead of being dropped by the empty-content guard.
            if isinstance(content, str) and content:
                segments.append(
                    _segment("assistant", content, start_ms, end_ms, latency_ms)
                )
            if tool_calls:
                segments.append(
                    _segment(
                        "tool_calls",
                        _render_tool_calls(tool_calls),
                        start_ms,
                        end_ms,
                        latency_ms,
                        tool_calls=tool_calls,
                    )
                )
            continue

        if role == "tool":
            if isinstance(content, str) and content:
                segments.append(
                    _segment(
                        "tool_call_result",
                        content,
                        start_ms,
                        end_ms,
                        None,
                        tool_call_id=msg.get("tool_call_id") or msg.get("id"),
                    )
                )
            continue

        if not isinstance(content, str) or not content:
            continue
        if role in {"user", "customer"}:
            speaker_role = "user"
        elif role == "system":
            speaker_role = "system"
        else:
            speaker_role = "unknown"
        segments.append(_segment(speaker_role, content, start_ms, end_ms, None))

    if segments:
        return segments

    if not result.transcript:
        return []
    for line in result.transcript.splitlines():
        if ":" not in line:
            continue
        role_label, content = line.split(":", 1)
        role_label = role_label.strip().lower()
        content = content.strip()
        if not content:
            continue
        if role_label in {"assistant", "agent", "bot"}:
            speaker_role = "assistant"
        elif role_label in {"customer", "user", "simulator", "caller"}:
            speaker_role = "user"
        else:
            speaker_role = "unknown"
        segments.append(
            {
                "speaker_role": speaker_role,
                "content": content,
                "start_time_ms": 0,
                "end_time_ms": 0,
            }
        )
    return segments


def _segment(
    speaker_role: str,
    content: str,
    start_ms: int,
    end_ms: int,
    latency_ms: int | None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    seg: dict[str, Any] = {
        "speaker_role": speaker_role,
        "content": content,
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
    }
    if latency_ms is not None:
        seg["latency_ms"] = latency_ms
    if tool_calls:
        seg["tool_calls"] = tool_calls
    if tool_call_id:
        seg["tool_call_id"] = tool_call_id
    return seg


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]] | None:
    """Coerce a message's tool_calls into a stable [{id, name, arguments}] shape.

    Accepts both the flat SDK shape (``{"name", "arguments", "id"}``) and the
    OpenAI/LiteLLM nested shape (``{"function": {"name", "arguments"}}``);
    ``arguments`` is JSON-decoded when the provider ships it as a string.
    """
    if not raw or not isinstance(raw, (list, tuple)):
        return None
    calls: list[dict[str, Any]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = tc.get("name") or fn.get("name")
        if not name:
            continue
        arguments = tc.get("arguments")
        if arguments is None:
            arguments = fn.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except (ValueError, TypeError):
                pass
        calls.append(
            {
                "id": tc.get("id") or name,
                "name": name,
                "arguments": arguments if arguments is not None else {},
            }
        )
    return calls or None


def _render_tool_calls(calls: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for call in calls:
        args = call.get("arguments")
        try:
            rendered = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            rendered = str(args)
        lines.append(f"{call['name']}({rendered})")
    return "\n".join(lines)


def _message_latency_ms(msg: dict[str, Any]) -> int | None:
    for key in ("latency_ms", "latency"):
        value = msg.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    metrics = msg.get("metrics")
    if isinstance(metrics, dict):
        for key in ("latency_ms", "latency"):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return None


def _maybe_upload_recording(
    client: httpx.Client, call_execution_id: str, case
) -> str | None:
    """Upload the case's audio file (if any) via a multipart POST.

    Prefers a combined/mixed WAV, falls back to output-only then input-only.
    Skips silently when no on-disk audio exists (e.g. ``record_audio=False``
    on the runner, or the SDK already surfaced an HTTPS URL via
    ``result.artifacts``). Returns the persisted ``recording_url`` to attach
    to the ingestion PATCH, or None.
    """
    if case.result is None:
        return None
    audio_path = _select_audio_path(case.result)
    if audio_path is None:
        return None

    filename = audio_path.name
    content_type = _CONTENT_TYPE_BY_EXT.get(
        audio_path.suffix.lower(), "application/octet-stream"
    )
    with audio_path.open("rb") as fh:
        files = {"file": (filename, fh, content_type)}
        data = {
            "filename": filename,
            "sha256": _sha256_file(audio_path),
            "kind": _recording_kind(case.result, audio_path),
        }
        resp = client.post(
            f"/simulate/api/alk-simulate/call-executions/{call_execution_id}/recording/",
            files=files,
            data=data,
            timeout=_RECORDING_UPLOAD_TIMEOUT_SECONDS,
        )
    if resp.is_error:
        return None
    body = _unwrap(resp.json())
    return body.get("recording_url")


def _maybe_upload_stereo_recording(
    client: httpx.Client, call_execution_id: str, case
) -> str | None:
    """Upload the case's 2-channel stereo WAV (ch0 customer, ch1 assistant).

    Uses the same multipart endpoint as ``_maybe_upload_recording`` and returns
    the persisted URL for ``stereo_recording_url``, or None when absent.
    """
    if case.result is None:
        return None
    stereo_path = getattr(case.result, "audio_stereo_path", None)
    if not stereo_path:
        return None
    path = Path(str(stereo_path)).expanduser()
    if not (path.exists() and path.is_file() and path.stat().st_size > 0):
        return None

    filename = path.name
    content_type = _CONTENT_TYPE_BY_EXT.get(
        path.suffix.lower(), "application/octet-stream"
    )
    with path.open("rb") as fh:
        files = {"file": (filename, fh, content_type)}
        data = {
            "filename": filename,
            "sha256": _sha256_file(path),
            "kind": "stereo",
        }
        resp = client.post(
            f"/simulate/api/alk-simulate/call-executions/{call_execution_id}/recording/",
            files=files,
            data=data,
            timeout=_RECORDING_UPLOAD_TIMEOUT_SECONDS,
        )
    if resp.is_error:
        return None
    body = _unwrap(resp.json())
    return body.get("recording_url")


def _upload_audio_file(
    client: httpx.Client, call_execution_id: str, path: Path, *, kind: str
) -> str | None:
    """POST a single on-disk WAV to the recording endpoint; return its URL."""
    filename = path.name
    content_type = _CONTENT_TYPE_BY_EXT.get(
        path.suffix.lower(), "application/octet-stream"
    )
    with path.open("rb") as fh:
        files = {"file": (filename, fh, content_type)}
        data = {"filename": filename, "sha256": _sha256_file(path), "kind": kind}
        resp = client.post(
            f"/simulate/api/alk-simulate/call-executions/{call_execution_id}/recording/",
            files=files,
            data=data,
            timeout=_RECORDING_UPLOAD_TIMEOUT_SECONDS,
        )
    if resp.is_error:
        return None
    return _unwrap(resp.json()).get("recording_url")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _upload_channel_recording(
    client: httpx.Client, call_execution_id: str, case, attr: str
) -> str | None:
    """Upload one per-speaker mono WAV named by ``attr`` on the case result."""
    if case.result is None:
        return None
    value = getattr(case.result, attr, None)
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not (path.exists() and path.is_file() and path.stat().st_size > 0):
        return None
    kind = "assistant" if attr == "audio_output_path" else "customer"
    return _upload_audio_file(client, call_execution_id, path, kind=kind)


def _attach_channel_recordings(
    client: httpx.Client, call_execution_id: str, case, payload: dict[str, Any]
) -> None:
    """Upload the per-speaker assistant/customer mono tracks and fold their URLs
    into ``provider_call_data.livekit.recording`` so evals mapped to
    ``call.assistant_recording`` / ``call.customer_recording`` resolve. LiveKit
    runs otherwise only produce combined + stereo, leaving the per-channel
    variables empty. ``audio_output_path`` is the target/assistant track,
    ``audio_input_path`` is the simulator/customer track (matching the stereo
    channel order ch0 customer, ch1 assistant)."""
    assistant_url = _upload_channel_recording(
        client, call_execution_id, case, "audio_output_path"
    )
    customer_url = _upload_channel_recording(
        client, call_execution_id, case, "audio_input_path"
    )
    recording: dict[str, str] = {}
    if assistant_url:
        recording["assistant"] = assistant_url
    if customer_url:
        recording["customer"] = customer_url
    if not recording:
        return
    provider_call_data = payload.setdefault("provider_call_data", {})
    livekit = provider_call_data.setdefault("livekit", {})
    if not isinstance(livekit, dict):
        return
    livekit["recording"] = {**(livekit.get("recording") or {}), **recording}


def _select_audio_path(result) -> Path | None:
    for candidate in (
        result.audio_combined_path,
        result.audio_output_path,
        result.audio_input_path,
    ):
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser()
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _recording_kind(result, path: Path) -> str:
    if (
        result.audio_combined_path
        and path == Path(str(result.audio_combined_path)).expanduser()
    ):
        return "combined"
    if (
        result.audio_output_path
        and path == Path(str(result.audio_output_path)).expanduser()
    ):
        return "assistant"
    return "customer"


_TARGET_PROVIDERS = ("vapi", "retell", "livekit")


class _TargetUsage:
    """Normalized target-agent usage extracted from one provider's evidence."""

    __slots__ = ("provider", "usage", "cost_cents", "raw")

    def __init__(
        self,
        provider: str,
        usage: dict[str, int] | None,
        cost_cents: int | None,
        raw: dict[str, Any] | None,
    ) -> None:
        self.provider = provider
        self.usage = usage
        self.cost_cents = cost_cents
        self.raw = raw


def _target_provider_usage(case) -> _TargetUsage | None:
    """Pull the target agent's provider-reported usage from case evidence.

    Provider-agnostic: dispatches to a per-provider extractor because each
    provider reports cost/tokens in a different shape (Vapi costBreakdown,
    Retell call_cost + llm_token_usage, LiveKit normalized usage). Returns a
    ``_TargetUsage`` with a normalized ``usage`` (``prompt_tokens`` /
    ``completion_tokens`` / ``total_tokens``) and ``cost_cents``, or None when
    no target evidence surfaced usage (e.g. a black-box self-hosted target).
    """
    evidence = getattr(case, "evidence", None) or []
    for source in evidence:
        metadata = getattr(source, "metadata", None) or {}
        provider = metadata.get("provider")
        if provider not in _TARGET_PROVIDERS:
            continue
        extractor = _PROVIDER_USAGE_EXTRACTORS.get(provider)
        if extractor is None:
            continue
        result = extractor(metadata)
        if result is not None:
            return result
    return None


def _vapi_usage(metadata: dict[str, Any]) -> _TargetUsage | None:
    cost = metadata.get("cost") if isinstance(metadata.get("cost"), dict) else {}
    breakdown = (
        cost.get("breakdown") if isinstance(cost.get("breakdown"), dict) else None
    )
    usage = None
    if breakdown:
        prompt = breakdown.get("llmPromptTokens", breakdown.get("promptTokens"))
        completion = breakdown.get(
            "llmCompletionTokens", breakdown.get("completionTokens")
        )
        usage = _normalized_usage(prompt, completion)
    cost_cents = _dollars_to_cents(cost.get("total"))
    if usage is None and cost_cents is None:
        return None
    return _TargetUsage("vapi", usage, cost_cents, breakdown)


def _retell_usage(metadata: dict[str, Any]) -> _TargetUsage | None:
    token_usage = metadata.get("usage")
    usage = None
    if isinstance(token_usage, dict):
        # Retell may report prompt/completion directly, or per-request `values`
        # (total tokens only, no split).
        prompt = token_usage.get("num_input_tokens", token_usage.get("prompt_tokens"))
        completion = token_usage.get(
            "num_output_tokens", token_usage.get("completion_tokens")
        )
        if prompt is not None or completion is not None:
            usage = _normalized_usage(prompt, completion)
        else:
            values = token_usage.get("values")
            if isinstance(values, list) and values:
                total = sum(_coerce_int(v) for v in values)
                if total:
                    usage = {"total_tokens": total}
    call_cost = metadata.get("cost") if isinstance(metadata.get("cost"), dict) else {}
    # Retell reports combined_cost already in cents.
    cost_cents = _coerce_int_or_none(call_cost.get("combined_cost"))
    if usage is None and cost_cents is None:
        return None
    return _TargetUsage("retell", usage, cost_cents, call_cost or None)


def _livekit_usage(metadata: dict[str, Any]) -> _TargetUsage | None:
    # A LiveKit target that reports a normalized usage blob back through the
    # evidence layer (self-hosted worker). Absent for black-box targets.
    usage_blob = metadata.get("usage")
    if not isinstance(usage_blob, dict):
        return None
    llm = (
        usage_blob.get("llm") if isinstance(usage_blob.get("llm"), dict) else usage_blob
    )
    prompt = llm.get("prompt_tokens", llm.get("promptTokens"))
    completion = llm.get("completion_tokens", llm.get("completionTokens"))
    usage = _normalized_usage(prompt, completion)
    cost_cents = _dollars_to_cents(
        (metadata.get("cost") or {}).get("total")
        if isinstance(metadata.get("cost"), dict)
        else None
    )
    if usage is None and cost_cents is None:
        return None
    return _TargetUsage("livekit", usage, cost_cents, None)


_PROVIDER_USAGE_EXTRACTORS = {
    "vapi": _vapi_usage,
    "retell": _retell_usage,
    "livekit": _livekit_usage,
}


def _normalized_usage(prompt: Any, completion: Any) -> dict[str, int] | None:
    if prompt is None and completion is None:
        return None
    prompt_i = _coerce_int(prompt)
    completion_i = _coerce_int(completion)
    return {
        "prompt_tokens": prompt_i,
        "completion_tokens": completion_i,
        "total_tokens": prompt_i + completion_i,
    }


def _dollars_to_cents(value: Any) -> int | None:
    dollars = _coerce_float(value)
    return int(round(dollars * 100)) if dollars is not None else None


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _speech_bounds(case) -> tuple[Any, Any]:
    """Return (start, end) datetimes from a case's message speech timing.

    LiveKit messages carry ``started_speaking_at`` / ``stopped_speaking_at``
    as epoch seconds; the earliest start and latest stop bound the actual
    conversation. Returns (None, None) when no timing is available.
    """
    if case.result is None:
        return None, None
    starts: list[float] = []
    ends: list[float] = []
    for msg in case.result.messages:
        if not isinstance(msg, dict):
            continue
        start = msg.get("started_speaking_at") or msg.get("created_at")
        stop = msg.get("stopped_speaking_at") or msg.get("created_at")
        if isinstance(start, (int, float)) and start > 0:
            starts.append(float(start))
        if isinstance(stop, (int, float)) and stop > 0:
            ends.append(float(stop))
    if not starts or not ends:
        return None, None
    start_dt = datetime.fromtimestamp(min(starts), tz=timezone.utc)
    end_dt = datetime.fromtimestamp(max(ends), tz=timezone.utc)
    if end_dt < start_dt:
        end_dt = start_dt
    return start_dt, end_dt


def _first_speech_anchor(messages: list[dict[str, Any]]) -> float | None:
    for msg in messages:
        for key in ("started_speaking_at", "created_at"):
            value = msg.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return None


def _resolve_message_timing_ms(
    msg: dict[str, Any], anchor: float | None
) -> tuple[int, int]:
    """Return (start_ms, end_ms) relative to the first-speech anchor.

    Falls back to ``created_at`` when speech-timing metrics are missing (text
    turns, providers that don't report the metric). Zero is used as the last
    resort — the backend metrics calculator degrades gracefully when timings
    collapse to zero-duration.
    """
    if anchor is None:
        return 0, 0

    start_raw = msg.get("started_speaking_at") or msg.get("created_at") or 0.0
    stop_raw = (
        msg.get("stopped_speaking_at") or msg.get("created_at") or start_raw or 0.0
    )
    start_ms = (
        max(int(round((float(start_raw) - anchor) * 1000)), 0) if start_raw else 0
    )
    end_ms = (
        max(int(round((float(stop_raw) - anchor) * 1000)), start_ms)
        if stop_raw
        else start_ms
    )
    return start_ms, end_ms


def _extract_recording_uri(result) -> str | None:
    for artifact in result.artifacts:
        artifact_type = getattr(artifact, "type", None)
        if artifact_type == "audio" and getattr(artifact, "uri", None):
            return artifact.uri
    for candidate in (
        result.audio_combined_path,
        result.audio_output_path,
        result.audio_input_path,
    ):
        if candidate and str(candidate).startswith(("http://", "https://")):
            return str(candidate)
    return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))


def _write_submission(run_directory: Path, payload: dict[str, Any]) -> None:
    submission_path = run_directory / "submission.json"
    submission_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = ["FutureAGIResultSink"]
