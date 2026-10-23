"""Supervised JSON workers with private environments and process-group cleanup."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

MAX_RESULT_BYTES = 16 * 1024 * 1024
GRACEFUL_SHUTDOWN_SECONDS = 90.0


class IsolatedWorkerError(RuntimeError):
    """The worker exited without a valid result."""


async def _finish(task: asyncio.Task) -> bool:
    canceled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            canceled = True
    task.result()
    return canceled


def run_worker(main: Callable[[], Awaitable[None]]) -> None:
    """Translate termination into one cancellation so async cleanup can finish."""

    logging.basicConfig(
        level=logging.WARNING,
        format=f"%(asctime)s %(levelname)s job={os.environ.get('ALK_LOG_JOB_ID', '-')} %(name)s: %(message)s",
    )
    logging.getLogger("fi").setLevel(logging.INFO)

    async def supervised() -> None:
        task = asyncio.create_task(main())
        loop = asyncio.get_running_loop()
        termination_requested = False

        def cancel_once() -> None:
            nonlocal termination_requested
            if termination_requested:
                return
            termination_requested = True
            if not getattr(task, "cancelling", lambda: 0)():
                task.cancel()

        loop.add_signal_handler(signal.SIGTERM, cancel_once)
        try:
            await task
        finally:
            loop.remove_signal_handler(signal.SIGTERM)

    asyncio.run(supervised())


def _job_id() -> str:
    for handler in logging.getLogger().handlers:
        for log_filter in handler.filters:
            if isinstance(getattr(log_filter, "job_id", None), str):
                return log_filter.job_id
    return "-"


async def _reap(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float = GRACEFUL_SHUTDOWN_SECONDS,
) -> None:
    # Leave descendants alive while the leader finalizes remote calls and rooms.
    try:
        process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def run_json_worker(
    module: str,
    payload: Mapping[str, Any],
    *,
    environ: Mapping[str, str],
    work_directory: Path,
) -> dict[str, Any]:
    """Run a fresh interpreter; cancellation waits for process-group termination."""
    if os.name != "posix":
        raise IsolatedWorkerError("isolated workers require POSIX process groups")
    work_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="worker-", dir=work_directory) as directory:
        root = Path(directory)
        result_path = root / "result.json"
        child_env = dict(environ)
        # Runtime scratch is invocation-private, including libraries using tempfile.
        child_env.update(
            TMPDIR=str(root),
            TMP=str(root),
            TEMP=str(root),
            HOME=str(root),
            XDG_CACHE_HOME=str(root / "cache"),
        )
        # The worker logs into this process's stream, tagged with the same job id.
        child_env.setdefault("ALK_LOG_JOB_ID", _job_id())
        # It imports the same harness code as this process, wherever that code was loaded from.
        if os.environ.get("PYTHONPATH"):
            child_env.setdefault("PYTHONPATH", os.environ["PYTHONPATH"])
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                module,
                str(result_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=None,
                stderr=None,
                env=child_env,
                cwd=root,
                start_new_session=True,
            )
        )
        canceled = await _finish(creation)
        process = creation.result()
        try:
            if canceled:
                raise asyncio.CancelledError
            await process.communicate(json.dumps(payload).encode())
            if process.returncode != 0:
                logging.getLogger(__name__).error(
                    "isolated_worker_failed module=%s exit_status=%s result_present=%s",
                    module, process.returncode, result_path.is_file(),
                )
                raise IsolatedWorkerError(
                    f"isolated worker exited with status {process.returncode}"
                )
            if (
                result_path.is_symlink()
                or not result_path.is_file()
                or result_path.stat().st_size > MAX_RESULT_BYTES
            ):
                raise IsolatedWorkerError("isolated worker result missing or oversized")
            try:
                result = json.loads(result_path.read_text())
            except (ValueError, OSError) as exc:
                raise IsolatedWorkerError("isolated worker result invalid") from exc
            if not isinstance(result, dict):
                raise IsolatedWorkerError("isolated worker result must be an object")
            return result
        finally:
            cleanup = asyncio.create_task(_reap(process))
            # Do not allow repeated cancellation to release a world before cleanup.
            if await _finish(cleanup):
                raise asyncio.CancelledError
