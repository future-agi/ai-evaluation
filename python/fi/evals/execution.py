"""
Execution handles for async eval and composite runs.

An ``Execution`` is a lightweight view into a (possibly still-running) eval
on the backend (for single evals) or a background thread (for composite
evals, which the backend runs synchronously).

Typical usage::

    from fi.evals import Evaluator, EvalTemplateManager

    ev = Evaluator()
    mgr = EvalTemplateManager()

    # --- Single eval: real backend async via is_async=True ---
    handle = ev.submit("tone", {"output": "I love this!"})
    handle.wait()                # polls until completion
    print(handle.result.output)  # -> "love"

    # --- Composite eval: SDK-side threaded execution ---
    handle = mgr.submit_composite(composite_id, mapping={"output": "Hi!"})
    handle.wait()
    print(handle.result["aggregate_score"])

    # --- Resumable by ID (single eval only) ---
    other_handle = ev.get_execution(handle.id)
    other_handle.wait()

Note on composite executions: the handle lives in a background thread
inside the calling process. If the process dies or the handle is dropped,
the in-flight work is lost. Use single-eval async submissions when you
need cross-process resumability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


class ExecutionError(Exception):
    """Raised when an Execution that finished in the ``failed`` state is awaited."""


# Backend eval_status → SDK-normalized status.
_STATUS_MAP = {
    "pending": "pending",
    "PENDING": "pending",
    "processing": "processing",
    "PROCESSING": "processing",
    "running": "processing",
    "completed": "completed",
    "COMPLETED": "completed",
    "failed": "failed",
    "FAILED": "failed",
}


def _normalize_status(raw: Optional[str]) -> str:
    if not raw:
        return "pending"
    return _STATUS_MAP.get(raw, raw.lower())


@dataclass
class Execution:
    """
    Handle to an in-flight or completed eval execution.

    Attributes:
        id: Execution identifier. For single evals this is the server-side
            ``eval_id`` (a UUID) and is resumable from any process via
            :py:meth:`fi.evals.Evaluator.get_execution`. For composite
            evals this is a client-side UUID — see the module docstring
            for the caveat.
        kind: ``"eval"`` for a single eval execution, ``"composite"`` for
            a composite one.
        status: ``"pending"`` | ``"processing"`` | ``"completed"`` |
            ``"failed"``.
        result: Populated once ``status == "completed"``. An ``EvalResult``
            for single evals; a dict matching the composite execute
            response for composite ones.
        error_message: Populated when the execution failed.
        error_localizer: Populated for single evals when error
            localization was enabled and the analysis is available.
    """

    id: str
    kind: str
    status: str = "pending"
    result: Any = None
    error_message: Optional[str] = None
    error_localizer: Optional[Dict[str, Any]] = None

    # Closure that (re)fetches the latest state. Set by the factory
    # method on Evaluator / EvalTemplateManager. Excluded from repr.
    _refresher: Optional[Callable[[], "Execution"]] = field(
        default=None, repr=False, compare=False
    )

    def is_done(self) -> bool:
        """Return True if status is terminal (``completed`` or ``failed``)."""
        return self.status in ("completed", "failed")

    def refresh(self) -> "Execution":
        """
        Re-fetch the latest state from the source (backend for single
        evals, background thread for composites). Returns ``self``.
        """
        if self._refresher is None:
            return self
        updated = self._refresher()
        self.status = updated.status
        self.result = updated.result
        self.error_message = updated.error_message
        self.error_localizer = updated.error_localizer
        return self

    def wait(
        self,
        *,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
        raise_on_failure: bool = True,
    ) -> "Execution":
        """
        Block until the execution reaches a terminal state, refreshing
        every ``poll_interval`` seconds.

        Args:
            timeout: Maximum number of seconds to wait before giving up.
            poll_interval: Seconds between refreshes.
            raise_on_failure: If True (default) raise
                :class:`ExecutionError` when the run finished in
                ``"failed"`` state. If False, return the handle so the
                caller can inspect ``error_message`` themselves.

        Raises:
            TimeoutError: If the execution did not reach a terminal
                state within ``timeout`` seconds.
            ExecutionError: If the execution failed and
                ``raise_on_failure=True``.
        """
        if self.is_done():
            if self.status == "failed" and raise_on_failure:
                raise ExecutionError(
                    f"Execution {self.id} failed: {self.error_message}"
                )
            return self

        deadline = time.monotonic() + float(timeout)
        while True:
            time.sleep(poll_interval)
            self.refresh()
            if self.is_done():
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Execution {self.id} did not complete within {timeout}s "
                    f"(last status: {self.status})"
                )

        if self.status == "failed" and raise_on_failure:
            raise ExecutionError(
                f"Execution {self.id} failed: {self.error_message}"
            )
        return self
