"""Bounded native workspace access through directory descriptors."""

from __future__ import annotations

import os
import re
import secrets
import stat
from contextlib import contextmanager
from typing import NamedTuple
from pathlib import Path

from .errors import WorldStateTooLarge, WorldUsageError

FILE_BYTES = 1024 * 1024
SNAPSHOT_BYTES = 16 * 1024 * 1024
SNAPSHOT_ENTRIES = 2048


class FileSnapshot(NamedTuple):
    data: bytes | None
    mode: int


_RESERVED = {".git", ".venv", ".alk-runtime", "node_modules", "__pycache__"}


def relative_parts(path: str) -> list[str]:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise WorldUsageError("workspace path must be a nonempty relative POSIX path")
    parts = path.split("/")
    if len(parts) > 32 or len(path) > 1024:
        raise WorldUsageError("workspace path exceeds depth or length limit")
    if any(part in {"", ".", ".."} or part in _RESERVED for part in parts):
        raise WorldUsageError(
            "workspace path contains traversal or reserved runtime data"
        )
    return parts


def validate_bindings(bindings: dict[str, str]) -> dict[str, str]:
    if len(bindings) > 16:
        raise ValueError("at most 16 native workspaces are supported")
    paths = []
    for name, path in bindings.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError("workspace name must be a simple identifier")
        try:
            parts = tuple(relative_parts(path))
        except WorldUsageError as exc:
            raise ValueError(str(exc)) from exc
        if any(
            parts[: len(other)] == other or other[: len(parts)] == parts
            for other in paths
        ):
            raise ValueError("workspace roots must not overlap")
        paths.append(parts)
    return bindings


def _directory(parent: int, name: str, create: bool = False) -> int:
    if create:
        try:
            os.mkdir(name, 0o750, dir_fd=parent)
            owner = os.fstat(parent)
            os.chown(
                name, owner.st_uid, owner.st_gid, dir_fd=parent, follow_symlinks=False
            )
        except FileExistsError:
            pass
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


@contextmanager
def directory(path: Path, *, create: bool = False):
    """Never follow a symlink in any component of an absolute binding."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.absolute().parts[1:]:
            next_fd = _directory(fd, part, create)
            os.close(fd)
            fd = next_fd
        yield fd
    except OSError as exc:
        raise WorldUsageError(
            f"workspace directory unavailable: {exc.strerror}"
        ) from exc
    finally:
        os.close(fd)


def prepare_root(path: Path) -> None:
    with directory(path, create=True):
        pass


@contextmanager
def _parent(root: Path, path: str, *, create: bool = False):
    parts = relative_parts(path)
    with directory(root) as root_fd:
        fd = os.dup(root_fd)
        try:
            for part in parts[:-1]:
                next_fd = _directory(fd, part, create)
                os.close(fd)
                fd = next_fd
            yield fd, parts[-1]
        except OSError as exc:
            raise WorldUsageError(
                f"workspace file unavailable: {exc.strerror}"
            ) from exc
        finally:
            os.close(fd)


def _regular(info):
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise WorldUsageError("workspace files must be regular files with one link")


def read(root: Path, path: str) -> bytes:
    with _parent(root, path) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            _regular(os.fstat(fd))
            with os.fdopen(os.dup(fd), "rb") as stream:
                data = stream.read(FILE_BYTES + 1)
            if len(data) > FILE_BYTES:
                raise WorldStateTooLarge("workspace file exceeds 1 MiB")
            return data
        finally:
            os.close(fd)


def write(root: Path, path: str, data: bytes, *, mode: int | None = None) -> None:
    if len(data) > FILE_BYTES:
        raise WorldStateTooLarge("workspace file exceeds 1 MiB")
    with _parent(root, path, create=True) as (parent, name):
        try:
            original = os.stat(name, dir_fd=parent, follow_symlinks=False)
            _regular(original)
            if mode is None:
                mode = stat.S_IMODE(original.st_mode)
        except FileNotFoundError:
            pass
        temporary = f".alk-write-{secrets.token_hex(16)}"
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o640,
            dir_fd=parent,
        )
        try:
            owner = os.fstat(parent)
            os.fchown(fd, owner.st_uid, owner.st_gid)
            os.fchmod(fd, (mode if mode is not None else 0o640) & 0o777)
            with os.fdopen(os.dup(fd), "wb") as stream:
                stream.write(data)
            # Replacing the entry cannot mutate an outside inode linked in by a concurrent agent.
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            os.close(fd)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass


def snapshot(root: Path) -> dict[str, FileSnapshot]:
    result: dict[str, FileSnapshot] = {}
    size = 0

    def visit(fd, prefix):
        nonlocal size
        names = []
        with os.scandir(fd) as entries:
            for entry in entries:
                if len(result) + len(names) >= SNAPSHOT_ENTRIES:
                    raise WorldStateTooLarge("workspace snapshot exceeds 2048 entries")
                names.append(entry.name)
        for name in sorted(names):
            path = f"{prefix}/{name}" if prefix else name
            relative_parts(path)
            if len(result) >= SNAPSHOT_ENTRIES:
                raise WorldStateTooLarge("workspace snapshot exceeds 2048 entries")
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                result[path] = FileSnapshot(None, stat.S_IMODE(info.st_mode))
                child = _directory(fd, name)
                try:
                    visit(child, path)
                finally:
                    os.close(child)
            else:
                _regular(info)
                data = read(root, path)
                size += len(data)
                if size > SNAPSHOT_BYTES:
                    raise WorldStateTooLarge("workspace snapshot exceeds 16 MiB")
                result[path] = FileSnapshot(data, stat.S_IMODE(info.st_mode))

    with directory(root) as fd:
        result["."] = FileSnapshot(None, stat.S_IMODE(os.fstat(fd).st_mode))
        visit(fd, "")
    return result


def restore(root: Path, saved: dict[str, FileSnapshot]) -> None:
    current = snapshot(root)
    current.pop(".")
    for path in sorted(
        current, key=lambda value: (value.count("/"), value), reverse=True
    ):
        with _parent(root, path) as (parent, name):
            if current[path].data is None:
                os.rmdir(name, dir_fd=parent)
            else:
                os.unlink(name, dir_fd=parent)
    for path, entry in saved.items():
        if path == ".":
            continue
        if entry.data is None:
            with _parent(root, path, create=True) as (parent, name):
                fd = _directory(parent, name, True)
                os.close(fd)
        else:
            write(root, path, entry.data, mode=entry.mode)

    for path, entry in reversed(list(saved.items())):
        if entry.data is None:
            if path == ".":
                with directory(root) as fd:
                    os.fchmod(fd, entry.mode & 0o777)
                continue
            with _parent(root, path) as (parent, name):
                fd = _directory(parent, name)
                try:
                    os.fchmod(fd, entry.mode & 0o777)
                finally:
                    os.close(fd)
