"""C3 dispatch-ack: the scheduler + call_runner half of the seam (Track B-ack).

These exercise the livekit-free side of the chain — the CallRunner structured-marker passthrough
and the scheduler's `except CallAborted` code-selection + `_CODE_DOMAIN`/fallback hardening
(c3-call-affinity.md v0.4 §4.5). The engine ladder itself lives in
`tests/runtime/test_livekit_dispatch_ack.py` (it needs the optional `livekit` dependency).

Convention (`test_call_runner.py`): `asyncio.run` drives every `async def` seam directly; no
pytest-asyncio.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from fi.alk.harness import call_runner as cr
from fi.alk.harness import hosted_scheduler as hs
from fi.alk.harness.hosted_scheduler import CallAborted
from fi.alk.harness.job import FailureDomain
from fi.simulate.runtime.run import TestCaseStatus

_UNACK = "voice_dispatch_unacknowledged"


# =================================================================================================
# CallAborted carries the structured marker (never string-matched from the message).
# =================================================================================================


def test_call_aborted_defaults_marker_to_none() -> None:
    exc = CallAborted("voice_call_not_completed: FAILED: boom")
    assert exc.marker is None


def test_call_aborted_carries_explicit_marker() -> None:
    exc = CallAborted("voice_call_not_completed: FAILED: boom", marker=_UNACK)
    assert exc.marker == _UNACK


# =================================================================================================
# Scheduler code-selection (C3 §4.5 step 3): marker -> receipt code, else byte-unchanged call_failed.
# =================================================================================================


def test_code_selection_picks_unacked_on_marker() -> None:
    exc = CallAborted("boom", marker=_UNACK)
    assert hs._call_aborted_code(exc) == _UNACK


def test_code_selection_falls_back_to_call_failed_without_marker() -> None:
    assert hs._call_aborted_code(CallAborted("boom")) == "call_failed"


def test_code_selection_ignores_unrelated_marker() -> None:
    # An unknown marker must NOT be trusted as a receipt code by this branch; only the one code
    # this branch owns is selected, everything else keeps call_failed.
    assert hs._call_aborted_code(CallAborted("boom", marker="something_else")) == "call_failed"


def test_code_selection_never_string_matches_the_message() -> None:
    # The message literally contains the marker text, but with no structured marker the branch
    # must still choose call_failed — proving it reads the field, not the string.
    exc = CallAborted(f"voice_call_not_completed: FAILED: {_UNACK} happened")
    assert hs._call_aborted_code(exc) == "call_failed"


# =================================================================================================
# _CODE_DOMAIN row + retry classification (C3 §4.5 step 4).
# =================================================================================================


def test_unacked_is_infrastructure_domain() -> None:
    assert hs._CODE_DOMAIN[_UNACK] is FailureDomain.INFRASTRUCTURE


def test_unacked_retries_once_like_call_failed() -> None:
    # Same retry posture as call_failed (retry once on a fresh world).
    assert hs._is_retryable(_UNACK) is True
    assert hs._is_retryable(_UNACK) == hs._is_retryable("call_failed")


def test_unacked_failure_receipt_has_infrastructure_domain() -> None:
    receipt = hs._failure(_UNACK, "not acked in 60s")
    assert receipt.code == _UNACK
    assert receipt.domain == FailureDomain.INFRASTRUCTURE.value


# =================================================================================================
# Unknown-code fallback hardening (C3 §4.5 step 5): never KeyError inside the CallAborted handler.
# =================================================================================================


def test_unknown_code_domain_defaults_to_infrastructure() -> None:
    assert hs._domain_for("a_code_that_does_not_exist") is FailureDomain.INFRASTRUCTURE


def test_unknown_code_is_retryable_without_keyerror() -> None:
    # Before hardening this indexed _CODE_DOMAIN[code] and KeyError'd.
    assert hs._is_retryable("totally_unmapped_code") is True


def test_unknown_code_failure_receipt_does_not_keyerror() -> None:
    receipt = hs._failure("totally_unmapped_code", "message")
    assert receipt.code == "totally_unmapped_code"
    assert receipt.domain == FailureDomain.INFRASTRUCTURE.value


def test_known_domains_are_unchanged_by_the_fallback() -> None:
    # The fallback must not disturb existing rows.
    assert hs._domain_for("world_unavailable") is FailureDomain.ENVIRONMENT
    assert hs._domain_for("driver_crashed") is FailureDomain.SIMULATOR
    assert hs._domain_for("call_failed") is FailureDomain.INFRASTRUCTURE


# =================================================================================================
# CallRunner passthrough (C3 §4.5 step 2): the structured marker rides through to CallAborted.
# `_translate_report` is exercised directly with duck-typed fakes — no real LiveKit call, no
# adapter uploads (result is None and no tool calls, so neither artifact path fires).
# =================================================================================================


def _runner() -> cr.CallRunnerImpl:
    runner = cr.CallRunnerImpl.__new__(cr.CallRunnerImpl)
    runner._adapter = None  # never touched: no calls, no result
    runner._collect_calls = lambda runtime: ()  # instance override
    return runner


def _report(*, status: TestCaseStatus, code: str | None) -> SimpleNamespace:
    failure = SimpleNamespace(code=code, message="dispatch not acked") if code else None
    case = SimpleNamespace(
        status=status,
        started_at=None,
        ended_at=None,
        result=None,
        failure=failure,
    )
    return SimpleNamespace(test_cases=[case])


def _translate(report: SimpleNamespace) -> CallAborted:
    runner = _runner()
    try:
        asyncio.run(
            runner._translate_report(
                report,
                runtime=object(),
                scenario_key="k1",
                started_at=datetime.now(timezone.utc),
            )
        )
    except CallAborted as exc:
        return exc
    raise AssertionError("expected CallAborted, nothing was raised")


def test_passthrough_sets_marker_for_unacked_dispatch() -> None:
    exc = _translate(_report(status=TestCaseStatus.FAILED, code=_UNACK))
    assert exc.marker == _UNACK
    # The message still carries the human-readable reason; the marker is the structured channel.
    assert "voice_call_not_completed" in str(exc)


def test_passthrough_leaves_marker_none_for_other_failures() -> None:
    exc = _translate(_report(status=TestCaseStatus.FAILED, code="livekit_case_failed"))
    assert exc.marker is None


def test_passthrough_leaves_marker_none_when_no_failure_object() -> None:
    exc = _translate(_report(status=TestCaseStatus.TIMED_OUT, code=None))
    assert exc.marker is None


def test_passthrough_end_to_end_receipt_code_is_unacked() -> None:
    # The two halves compose: the CallRunner marker feeds the scheduler code-selection, which the
    # scheduler turns into the voice_dispatch_unacknowledged receipt (infrastructure, retryable).
    exc = _translate(_report(status=TestCaseStatus.FAILED, code=_UNACK))
    code = hs._call_aborted_code(exc)
    receipt = hs._failure(code, str(exc))
    assert receipt.code == _UNACK
    assert receipt.domain == FailureDomain.INFRASTRUCTURE.value
    assert hs._is_retryable(code) is True
