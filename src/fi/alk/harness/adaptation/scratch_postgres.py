"""A throwaway PostgreSQL 16 cluster from the snapshot's own binaries, for adaptation's proofs.

Bootstrapped and launched with the managed-engine commands `process_runtime` uses for a world's
`world_db`, so proofs run on the same engine build a hosted run does.
"""

from __future__ import annotations

import os
import pwd
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..process_runtime import (
    EngineCredentials,
    postgres_bootstrap_argv,
    postgres_daemon_argv,
)

_BIN_DIRECTORIES = ("/usr/lib/postgresql/16/bin",)
# Root may not run postgres; the snapshot's data-engine user does, as it does for world stores.
_ENGINE_USER = "svc-data"
_READY_SECONDS = 60.0


def _binary(name: str) -> str:
    for directory in _BIN_DIRECTORIES:
        candidate = Path(directory) / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(f"postgres binary {name!r} is not installed")
    return found


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextmanager
def scratch_cluster(parent: Path | None = None) -> Iterator[str]:
    """Yield a superuser DSN (a role that may create databases); the cluster dies afterwards."""
    import psycopg
    from psycopg import conninfo

    root = Path(tempfile.mkdtemp(prefix="alk-adaptation-pg-", dir=parent))
    data_dir = root / "data"
    data_dir.mkdir(mode=0o700)
    credentials = EngineCredentials(
        username="harness", password=secrets.token_urlsafe(24)
    )
    user = pwd.getpwnam(_ENGINE_USER) if os.geteuid() == 0 else None
    ids = {"user": user.pw_uid, "group": user.pw_gid} if user else {}
    if user:
        for path in (root, data_dir):
            os.chown(path, user.pw_uid, user.pw_gid)
    pwfile = root / ".pwfile"
    descriptor = os.open(pwfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(credentials.password + "\n")
    if user:
        os.chown(pwfile, user.pw_uid, user.pw_gid)
    port = _free_port()
    process: subprocess.Popen | None = None
    try:
        argv = postgres_bootstrap_argv(
            data_dir=data_dir, credentials=credentials, pwfile=pwfile
        )
        argv[0] = _binary(argv[0])
        bootstrap = subprocess.run(
            argv, capture_output=True, text=True, check=False, **ids
        )
        if bootstrap.returncode != 0:
            raise RuntimeError(
                f"initdb exited {bootstrap.returncode}: {bootstrap.stderr[-2000:]}"
            )
        pwfile.unlink()
        argv = postgres_daemon_argv(data_dir=data_dir, port=port)
        argv[0] = _binary(argv[0])
        process = subprocess.Popen(
            argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **ids
        )
        dsn = conninfo.make_conninfo(
            host="localhost",
            port=port,
            user=credentials.username,
            password=credentials.password,
            dbname="postgres",
        )
        deadline = time.monotonic() + _READY_SECONDS
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"postgres exited {process.returncode} while starting"
                )
            try:
                with psycopg.connect(dsn, connect_timeout=2):
                    break
            except psycopg.OperationalError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.25)
        yield dsn
    finally:
        if process is not None and process.poll() is None:
            # SIGINT is postgres's fast shutdown.
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        shutil.rmtree(root, ignore_errors=True)
