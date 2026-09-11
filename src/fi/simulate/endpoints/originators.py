"""Provider-neutral call-origination surface — factory + cleanup helper.

Lives outside ``simulation/engines/livekit.py`` on purpose: that module
raises ``ImportError`` without the optional ``livekit`` extra (which this
SDK's dev venv lacks), so nothing inside it can be exercised by tests here;
this module has no such dependency, so it is fully testable (C3/C6).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from .retell import RetellCallOriginator
from .vapi import VapiCallOriginator


class CallOriginator(Protocol):
    """The lifecycle every provider originator satisfies (C2/C3)."""

    async def start(self) -> Any: ...

    async def stop(self, call_id: str) -> None: ...

    async def reconcile_and_stop(
        self, *, started_after_ms: int, ended_before_ms: int
    ) -> list[str]: ...

    async def close(self) -> None: ...


def build_call_originator(transport: Any) -> CallOriginator:
    """Construct the originator named by ``transport.inbound_call_originator``.

    This is the one place a provider name is compared to select a class;
    construction, start, stop and cleanup afterward stay provider-neutral.
    """
    name = transport.inbound_call_originator
    if name == "vapi":
        return VapiCallOriginator.from_env()
    if name == "retell":
        return RetellCallOriginator.from_env(transport)
    raise ValueError(f"unsupported_inbound_call_originator: {name}")


@dataclass(frozen=True)
class OriginatorFinalizeResult:
    """What cleanup did, so the engine can adapt it without deciding anything."""

    termination_source: str | None
    reconciled_call_ids: list[str]
    cleanup_errors: list[tuple[str, Exception]]


async def finalize_originator(
    originator: CallOriginator,
    *,
    provider_call_id: str | None,
    originator_name: str,
    case_started_at: datetime,
    cleanup_timeout: float,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> OriginatorFinalizeResult:
    """Best-effort originator cleanup: never raises (C3).

    ``stop()`` ends the call we know about; with no call id in hand (a
    cancelled ``asyncio.wait_for`` can still leave a live, billed call on
    the provider's side) ``reconcile_and_stop`` hunts for the orphan instead.
    ``close()`` always runs, even when the above raised.
    """
    termination_source: str | None = None
    reconciled_call_ids: list[str] = []
    cleanup_errors: list[tuple[str, Exception]] = []
    try:
        if provider_call_id is not None:
            try:
                await asyncio.wait_for(
                    originator.stop(provider_call_id), timeout=cleanup_timeout
                )
                termination_source = "sdk_originator_cleanup"
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup, never raises
                cleanup_errors.append((f"{originator_name}_call_stop", exc))
        else:
            # A naive datetime is read as local time by .timestamp(), silently
            # widening the blast-radius window; treat it as UTC instead.
            started = (
                case_started_at
                if case_started_at.tzinfo is not None
                else case_started_at.replace(tzinfo=timezone.utc)
            )
            started_after_ms = int(started.timestamp() * 1000)
            ended_before_ms = int(now().timestamp() * 1000)
            try:
                reconciled_call_ids = await asyncio.wait_for(
                    originator.reconcile_and_stop(
                        started_after_ms=started_after_ms,
                        ended_before_ms=ended_before_ms,
                    ),
                    timeout=cleanup_timeout,
                )
                if reconciled_call_ids:
                    termination_source = "sdk_originator_cleanup"
            except TypeError as exc:
                # A wrong call site (non-int epoch ms) fails loudly and
                # distinguishably from a swallowed HTTP failure.
                cleanup_errors.append(
                    (f"{originator_name}_call_reconcile_bad_arguments", exc)
                )
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup, never raises
                cleanup_errors.append((f"{originator_name}_call_reconcile", exc))
    finally:
        try:
            # Bounded like every other cleanup step, so a slow client can't
            # stall this case past its cleanup budget.
            await asyncio.wait_for(originator.close(), cleanup_timeout)
        except Exception as exc:  # noqa: BLE001 - best-effort cleanup, never raises
            cleanup_errors.append((f"{originator_name}_call_close", exc))
    return OriginatorFinalizeResult(
        termination_source=termination_source,
        reconciled_call_ids=reconciled_call_ids,
        cleanup_errors=cleanup_errors,
    )


__all__ = [
    "CallOriginator",
    "OriginatorFinalizeResult",
    "build_call_originator",
    "finalize_originator",
]
