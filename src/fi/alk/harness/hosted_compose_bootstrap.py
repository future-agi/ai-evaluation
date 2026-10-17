"""Freeze one trusted hosted-controller handoff before exposing Compose operations."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import socket
import stat
import struct
import tempfile
import time
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .compose_broker import (
    MAX_MESSAGE,
    BrokerConfig,
    BrokerError,
    ComposeBroker,
    _receive,
    _root_controlled,
    _send,
    validate_config,
)
from .job import HarnessJob
from .provision import source_fingerprint

_PROFILE_PATH = Path("/etc/alk/hosted-compose.json")
_INITIALIZE_SOCKET = Path("/run/alk-compose/initialize.sock")


class HostedComposeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    source: str = "/work/source"
    job: str = "/work/job.json"
    secrets: str = "/run/futureagi/secrets.json"
    private_root: str = "/var/lib/alk-compose"
    initialize_socket: str = str(_INITIALIZE_SOCKET)
    broker_socket: str = "/run/alk-compose/broker.sock"
    docker_socket: str = "/run/alk-docker/docker.sock"
    artifacts: str = "/run/alk-compose/artifacts"
    client_uid: int = Field(default=2000, gt=0)
    client_gid: int = Field(default=2000, gt=0)
    max_bytes: int = Field(default=2 * 1024**3, ge=1, le=8 * 1024**3)
    max_file_bytes: int = Field(default=512 * 1024**2, ge=1, le=2 * 1024**3)
    max_entries: int = Field(default=200000, ge=1, le=1000000)
    max_depth: int = Field(default=64, ge=1, le=128)
    copy_seconds: int = Field(default=300, ge=1, le=900)
    receive_seconds: int = Field(default=15, ge=1, le=60)
    lifetime_seconds: int = Field(default=14400, ge=60, le=86400)
    max_failures: int = Field(default=8, ge=1, le=32)


class InitializeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    operation: Literal["initialize"]
    job_id: str = Field(min_length=1, max_length=128)
    source_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


def _failure(code: str) -> BrokerError:
    return BrokerError("compose_bootstrap_" + code)


def _public_error_code(error: Exception) -> str:
    code = str(error).split(":", 1)[0]
    return code if re.fullmatch(r"compose_[a-z_]+", code) else "compose_bootstrap_request_failed"


@contextmanager
def _directory(path: Path):
    if not path.is_absolute() or ".." in path.parts:
        raise _failure("absolute_path_required")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _regular(info, uid, limit, *, private=False):
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != uid
        or info.st_nlink != 1
        or info.st_size > limit
        or info.st_mode & 0o022
        or (private and stat.S_IMODE(info.st_mode) != 0o600)
    ):
        raise _failure("control_file_invalid")


def _read_control(path: Path, uid: int, *, private=False) -> bytes:
    with _directory(path.parent) as parent:
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            before = os.fstat(descriptor)
            _regular(before, uid, MAX_MESSAGE, private=private)
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                raw = stream.read(MAX_MESSAGE + 1)
            if len(raw) > MAX_MESSAGE or _identity(before) != _identity(
                os.fstat(descriptor)
            ):
                raise _failure("control_file_changed")
            return raw
        finally:
            os.close(descriptor)


def _json(raw: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _failure("duplicate_json_key")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=unique)


def _load_profile() -> HostedComposeProfile | None:
    if not _PROFILE_PATH.exists() and not _PROFILE_PATH.is_symlink():
        if _INITIALIZE_SOCKET.exists() or _INITIALIZE_SOCKET.is_symlink():
            raise _failure("profile_required")
        return None
    _root_controlled(_PROFILE_PATH)
    raw = _read_control(_PROFILE_PATH, 0)
    _json(raw)
    return HostedComposeProfile.model_validate_json(raw)


def _trusted_socket(path: Path):
    _root_controlled(path.parent, directory=True)
    info = path.lstat()
    if info.st_uid != 0 or not stat.S_ISSOCK(info.st_mode) or info.st_mode & 0o007:
        raise _failure("private_socket_required")


def initialize_hosted_compose(job: HarnessJob, source: Path) -> Path | None:
    try:
        profile = _load_profile()
        if profile is None:
            return None
        if Path(source) != Path(profile.source):
            raise _failure("fixed_source_required")
        request = InitializeRequest(
            operation="initialize",
            job_id=job.job_id,
            source_digest=source_fingerprint(source),
        )
        path = Path(profile.initialize_socket)
        _trusted_socket(path)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(profile.lifetime_seconds + profile.receive_seconds)
            connection.connect(str(path))
            if (
                not hasattr(socket, "SO_PEERCRED")
                or struct.unpack(
                    "3i",
                    connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
                )[1]
                != 0
            ):
                raise _failure("root_peer_required")
            _send(connection, request.model_dump())
            response = _json(_receive(connection))
        expected = {"initialized": True, "source_digest": request.source_digest}
        if response != {"result": expected}:
            raise _failure("initialization_failed")
        broker_path = Path(profile.broker_socket)
        _trusted_socket(broker_path)
        return broker_path
    except BrokerError:
        raise
    except Exception:
        raise _failure("initialization_failed") from None


def _snapshot(profile: HostedComposeProfile, destination: Path, deadline: float):
    entries = 0
    copied = 0
    links = []

    def check_budget():
        if time.monotonic() >= deadline:
            raise _failure("copy_deadline")

    def visit(descriptor, target, depth):
        nonlocal entries, copied
        before = os.fstat(descriptor)
        if before.st_uid != profile.client_uid or depth > profile.max_depth:
            raise _failure("source_directory_invalid")
        with os.scandir(descriptor) as children:
            for entry in children:
                name = entry.name
                check_budget()
                entries += 1
                if (
                    entries > profile.max_entries
                    or len(os.fsencode(str(target / name))) > 4096
                ):
                    raise _failure("source_entry_limit")
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if observed.st_uid != profile.client_uid:
                    raise _failure("source_owner_invalid")
                output = target / name
                if stat.S_ISLNK(observed.st_mode):
                    link = os.readlink(name, dir_fd=descriptor)
                    if _identity(observed) != _identity(
                        os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    ):
                        raise _failure("source_changed")
                    links.append((output, link))
                    continue
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if stat.S_ISDIR(observed.st_mode):
                    flags |= os.O_DIRECTORY
                child = os.open(name, flags, dir_fd=descriptor)
                try:
                    opened = os.fstat(child)
                    if _identity(observed) != _identity(opened):
                        raise _failure("source_changed")
                    if stat.S_ISDIR(opened.st_mode):
                        output.mkdir(mode=0o700)
                        visit(child, output, depth + 1)
                    elif stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1:
                        if opened.st_size > profile.max_file_bytes:
                            raise _failure("source_file_limit")
                        file_bytes = 0
                        with output.open("xb") as stream:
                            while chunk := os.read(child, 1024 * 1024):
                                check_budget()
                                copied += len(chunk)
                                file_bytes += len(chunk)
                                if (
                                    copied > profile.max_bytes
                                    or file_bytes > profile.max_file_bytes
                                ):
                                    raise _failure("source_byte_limit")
                                stream.write(chunk)
                        output.chmod(0o555 if opened.st_mode & 0o111 else 0o444)
                        os.chown(output, 0, 0)
                    else:
                        raise _failure("source_file_invalid")
                    if _identity(opened) != _identity(os.fstat(child)):
                        raise _failure("source_changed")
                finally:
                    os.close(child)
        if _identity(before) != _identity(os.fstat(descriptor)):
            raise _failure("source_changed")

    with _directory(Path(profile.source)) as root:
        visit(root, destination, 0)
    for output, link in links:
        check_budget()
        output.symlink_to(link)
        os.chown(output, 0, 0, follow_symlinks=False)
    from .certified_compose_runtime import validate_source_tree

    validate_source_tree(destination)
    for directory in (
        p for p in destination.rglob("*") if not p.is_symlink() and p.is_dir()
    ):
        os.chown(directory, 0, 0)
        directory.chmod(0o555)


class _Registration:
    def __init__(self, profile: HostedComposeProfile):
        self.profile = profile
        self.deadline = time.monotonic() + profile.lifetime_seconds
        self.request: InitializeRequest | None = None
        self.broker: ComposeBroker | None = None
        self.failures = 0

    def initialize(self, request: InitializeRequest):
        if self.request is not None:
            if request != self.request:
                raise _failure("registration_conflict")
            return {"initialized": True, "source_digest": request.source_digest}
        if (
            self.failures >= self.profile.max_failures
            or time.monotonic() >= self.deadline
        ):
            raise _failure("registration_exhausted")
        self.failures += 1
        staging = None
        published = None
        directories = []
        try:
            profile = self.profile
            raw_job = _read_control(Path(profile.job), profile.client_uid)
            _json(raw_job)
            job = HarnessJob.model_validate_json(raw_job, strict=True)
            if job.job_id != request.job_id:
                raise _failure("job_identity_mismatch")
            aliases = {
                alias
                for alias, ref in job.agent.secret_refs.items()
                if ref.purpose == "target_provider"
            }
            try:
                raw = _json(
                    _read_control(
                        Path(profile.secrets), profile.client_uid, private=True
                    )
                )
            except FileNotFoundError:
                if aliases:
                    raise _failure("target_secret_missing") from None
                raw = {}
            if not isinstance(raw, dict) or any(
                not isinstance(value, str) for value in raw.values()
            ):
                raise _failure("secret_schema_invalid")
            if aliases - raw.keys() or any(not raw[alias] for alias in aliases):
                raise _failure("target_secret_missing")
            values = {alias: raw[alias] for alias in aliases}
            del raw
            staging = Path(
                tempfile.mkdtemp(prefix="staging-", dir=profile.private_root)
            )
            deadline = min(self.deadline, time.monotonic() + profile.copy_seconds)
            _snapshot(profile, staging, deadline)
            if source_fingerprint(staging) != request.source_digest:
                raise _failure("source_digest_mismatch")
            if any(
                admission.source_digest != request.source_digest
                for admission in [
                    *job.public_material_admissions,
                    *job.public_test_fixture_admissions,
                ]
            ):
                raise _failure("admission_source_mismatch")
            if time.monotonic() >= deadline:
                raise _failure("copy_deadline")
            destination = Path(profile.private_root) / "source"
            if destination.exists() or destination.is_symlink():
                raise _failure("registration_storage_conflict")
            staging.rename(destination)
            published = destination
            staging = None
            published.chmod(0o555)
            scratch = Path(profile.private_root) / "scratch"
            artifacts = Path(profile.artifacts)
            scratch.mkdir(mode=0o700)
            directories.append(scratch)
            artifacts.mkdir(mode=0o700)
            directories.append(artifacts)
            config = BrokerConfig(
                source=str(published),
                source_repository=job.source.repository,
                source_revision=job.source.commit_sha.lower()
                if job.source.commit_sha
                else None,
                scratch=str(scratch),
                artifacts=str(artifacts),
                socket=profile.broker_socket,
                docker_socket=profile.docker_socket,
                client_uid=profile.client_uid,
                client_gid=profile.client_gid,
                target_configuration=values,
                public_material_admissions=job.public_material_admissions,
                public_test_fixture_admissions=job.public_test_fixture_admissions,
                lifetime_seconds=profile.lifetime_seconds,
            )
            validate_config(config)
            self.broker = ComposeBroker(config)
            self.request = request
            return {"initialized": True, "source_digest": request.source_digest}
        except Exception as error:
            for path in (staging, published):
                if path is not None:
                    _remove_snapshot(path)
            for directory in reversed(directories):
                directory.rmdir()
            if isinstance(error, BrokerError):
                raise
            raise _failure("registration_failed") from None


def _remove_snapshot(path: Path):
    if path.exists():
        for directory in [
            path,
            *(p for p in path.rglob("*") if not p.is_symlink() and p.is_dir()),
        ]:
            directory.chmod(0o700)
        shutil.rmtree(path)


def _receive_before(connection, deadline):
    def recv(count):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _failure("receive_deadline")
        connection.settimeout(remaining)
        return connection.recv(count)

    return _receive(SimpleNamespace(recv=recv))


def _listener(path: str, gid: int):
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(path)
        os.chown(path, 0, gid)
        os.chmod(path, 0o620)
        listener.listen(4)
        return listener
    except BaseException:
        listener.close()
        raise


def serve_hosted_compose(profile: HostedComposeProfile) -> None:
    if os.geteuid() != 0 or not hasattr(socket, "SO_PEERCRED"):
        raise _failure("linux_root_required")
    for path in (
        profile.private_root,
        str(Path(profile.initialize_socket).parent),
        str(Path(profile.broker_socket).parent),
        str(Path(profile.artifacts).parent),
    ):
        _root_controlled(Path(path), directory=True)
    if Path(profile.private_root).stat().st_mode & 0o077:
        raise _failure("private_root_required")
    for path in (profile.initialize_socket, profile.broker_socket):
        if Path(path).exists() or Path(path).is_symlink():
            raise _failure("socket_already_exists")
    os.umask(0o077)
    registration = _Registration(profile)
    stop = False

    def shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    listeners = []
    try:
        with selectors.DefaultSelector() as selector:
            listener = _listener(profile.initialize_socket, profile.client_gid)
            listeners.append(listener)
            selector.register(listener, selectors.EVENT_READ, "initialize")
            while not stop and time.monotonic() < registration.deadline:
                for key, _ in selector.select(timeout=1):
                    connection, _ = key.fileobj.accept()
                    with connection:
                        connection.settimeout(profile.receive_seconds)
                        uid = struct.unpack(
                            "3i",
                            connection.getsockopt(
                                socket.SOL_SOCKET, socket.SO_PEERCRED, 12
                            ),
                        )[1]
                        if uid != profile.client_uid:
                            continue
                        try:
                            failures_before = registration.failures
                            payload = _receive_before(
                                connection,
                                min(
                                    registration.deadline,
                                    time.monotonic() + profile.receive_seconds,
                                ),
                            )
                            if key.data == "initialize":
                                _json(payload)
                                result = registration.initialize(
                                    InitializeRequest.model_validate_json(payload)
                                )
                                if len(listeners) == 1:
                                    broker_listener = _listener(
                                        profile.broker_socket, profile.client_gid
                                    )
                                    listeners.append(broker_listener)
                                    selector.register(
                                        broker_listener, selectors.EVENT_READ, "broker"
                                    )
                            else:
                                result = asyncio.run(
                                    registration.broker.dispatch(payload)
                                )
                            _send(connection, {"result": result})
                            if key.data == "initialize":
                                print(
                                    json.dumps(
                                        {"event": "compose_bootstrap_initialized"}
                                    ),
                                    flush=True,
                                )
                        except Exception as error:
                            if (
                                key.data == "initialize"
                                and registration.request is None
                                and registration.failures == failures_before
                            ):
                                registration.failures += 1
                            code = _public_error_code(error)
                            print(
                                json.dumps(
                                    {"event": "compose_bootstrap_error", "code": code}
                                ),
                                flush=True,
                            )
                            try:
                                _send(connection, {"error": code})
                            except OSError:
                                pass
                        finally:
                            if registration.broker is not None:
                                registration.broker._publish_logs()
    finally:
        try:
            if registration.broker is not None:
                asyncio.run(registration.broker.close())
                print(json.dumps({"event": "compose_bootstrap_closed"}), flush=True)
        finally:
            for listener in listeners:
                listener.close()
            Path(profile.initialize_socket).unlink(missing_ok=True)
            Path(profile.broker_socket).unlink(missing_ok=True)


def main():
    try:
        profile = _load_profile()
        if profile is None:
            raise _failure("profile_required")
        serve_hosted_compose(profile)
    except Exception:
        raise SystemExit("compose_bootstrap_service_failed") from None


if __name__ == "__main__":
    main()
