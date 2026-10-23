from __future__ import annotations

import os
import re
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path

from fi._diagnostics import sanitize_diagnostic


def retain_process_log(
    path: Path, *, work: Path, destination: Path, secrets: Mapping[str, str] | None
) -> None:
    relative = path.relative_to(work)
    fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        log_fd = os.open(
            relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
        )
        with os.fdopen(log_fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return
            offset = max(0, info.st_size - 65536)
            stream.seek(offset)
            raw = stream.read(65536)
            if offset:
                # Drop the incomplete first record, including any split credential.
                raw = raw.partition(b"\n")[2]
        text = sanitize_diagnostic(
            raw.decode("utf-8", errors="replace"), secrets, limit=65536
        )[-8192:]
        if not text:
            return
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination_fd = os.open(
            destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            try:
                os.mkdir("process-logs", mode=0o700, dir_fd=destination_fd)
            except FileExistsError:
                pass
            root_fd = os.open(
                "process-logs",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=destination_fd,
            )
            try:
                name = "-".join(relative.parts[:-1])
                target = f"{name}-{uuid.uuid4().hex}.log"
                out = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=root_fd,
                )
                with os.fdopen(out, "w") as stream:
                    stream.write(text)
            finally:
                os.close(root_fd)
        finally:
            os.close(destination_fd)
    finally:
        os.close(fd)


async def upload_process_logs(root: Path, adapter, **kwargs) -> None:
    from .outbound import ArtifactKind

    directory = root / "process-logs"
    if not directory.is_dir() or directory.is_symlink():
        return
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in sorted(os.listdir(fd)):
            if not name.endswith(".log"):
                continue
            log_fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
            )
            with os.fdopen(log_fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 32768:
                    continue
                body = stream.read(32768)
            await adapter.upload_artifact(
                name.encode() + b"\n" + body, kind=ArtifactKind.LOG, **kwargs
            )
    finally:
        os.close(fd)


def retained_process_evidence(
    directory: Path, before: set[Path], secrets, *, process_name: str | None = None
) -> str:
    """Read only bounded, sanitized logs retained by the current validation attempt."""
    if not directory.is_dir() or directory.is_symlink():
        return ""
    pieces = []
    remaining = 12000
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in sorted(os.listdir(fd)):
            if not name.endswith(".log") or directory / name in before:
                continue
            if process_name is not None and not re.fullmatch(
                rf"worlds-w\d+-{re.escape(process_name)}-[0-9a-f]{{32}}\.log", name
            ):
                continue
            try:
                log_fd = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                )
            except OSError:
                continue
            with os.fdopen(log_fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 32768:
                    continue
                text = sanitize_diagnostic(
                    stream.read(32768).decode("utf-8", errors="replace"),
                    secrets,
                    limit=32768,
                )
            piece = (sanitize_diagnostic(name, secrets) + "\n" + text)[:remaining]
            pieces.append(piece)
            remaining -= len(piece)
            if not remaining:
                break
    finally:
        os.close(fd)
    return "\n".join(pieces)[:12000]
