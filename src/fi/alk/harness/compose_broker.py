"""Per-job privileged Compose service with a path-free client protocol."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import socket
import stat
import struct
import time
from typing import Annotated, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .public_material import PublicMaterialAdmission, PublicTestFixtureAdmission
from .contract import Runtime
from .compose_image_lock import ResolvedComposeImage, selected_images, source_repository
from .bundle_v2 import EnvironmentBundleV2, RuntimeKindV2, seal_bundle_v2
from .certified_compose_runtime import (
    CertifiedComposeRuntimeProvider,
    normalize_compose,
)
from .process_runtime import EnvironmentRuntime
from .provision import source_fingerprint

MAX_MESSAGE = 2 * 1024 * 1024


class BrokerError(RuntimeError):
    pass


class BrokerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str
    source_repository: str | None = Field(default=None, max_length=2048)
    source_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    scratch: str
    artifacts: str
    socket: str
    docker_socket: str
    client_uid: int = Field(gt=0)
    client_gid: int = Field(gt=0)
    target_configuration: dict[str, str] = Field(default_factory=dict)
    public_material_admissions: list[PublicMaterialAdmission] = Field(
        default_factory=list, max_length=512
    )
    public_test_fixture_admissions: list[PublicTestFixtureAdmission] = Field(
        default_factory=list, max_length=128
    )
    lifetime_seconds: int = Field(default=14400, ge=60, le=86400)


class ProvisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["provision"]
    session: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    manifest: EnvironmentBundleV2
    document: str = Field(max_length=1024 * 1024)
    instances: int = Field(ge=1, le=4)


class ResolveImagesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["resolve_images"]
    manifest: EnvironmentBundleV2
    document: str = Field(max_length=1024 * 1024)
    runtime: Runtime


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["reset", "healthy", "close"]
    session: str = Field(pattern=r"^[a-f0-9]{32}$")
    world_index: int = Field(default=0, ge=0, le=3)


REQUEST = TypeAdapter(
    Annotated[ProvisionRequest | ResolveImagesRequest | SessionRequest, Field(discriminator="operation")]
)


def _receive(connection):
    def exact(count):
        result = bytearray()
        while len(result) < count:
            part = connection.recv(count - len(result))
            if not part:
                raise BrokerError("compose_broker_truncated_message")
            result.extend(part)
        return bytes(result)

    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise BrokerError("compose_broker_message_size")
    return exact(size)


def _send(connection, value):
    payload = json.dumps(value, separators=(",", ":")).encode()
    if len(payload) > MAX_MESSAGE:
        raise BrokerError("compose_broker_message_size")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _root_controlled(path: Path, *, directory=False):
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise BrokerError("compose_broker_root_owned_path_required")
    if directory and not path.is_dir():
        raise BrokerError("compose_broker_directory_required")


def validate_config(config: BrokerConfig):
    if os.geteuid() != 0 or not hasattr(socket, "SO_PEERCRED"):
        raise BrokerError("compose_broker_linux_root_required")
    paths = [
        Path(config.source),
        Path(config.scratch),
        Path(config.artifacts),
        Path(config.socket).parent,
    ]
    if any(not path.is_absolute() for path in paths):
        raise BrokerError("compose_broker_absolute_config_paths_required")
    for path in paths:
        _root_controlled(path, directory=True)
    source, scratch, artifacts, _ = paths
    if any(
        a.is_relative_to(b) or b.is_relative_to(a)
        for a, b in ((source, scratch), (source, artifacts), (scratch, artifacts))
    ):
        raise BrokerError("compose_broker_roots_must_be_separate")
    for path in source.rglob("*"):
        info = path.lstat()
        if info.st_uid != 0:
            raise BrokerError("compose_broker_immutable_source_required")
        if path.is_symlink():
            if not path.resolve().is_relative_to(source):
                raise BrokerError("compose_broker_source_symlink_escape")
        elif (
            info.st_mode & 0o022
            or not (path.is_dir() or path.is_file())
            or (path.is_file() and info.st_nlink != 1)
        ):
            raise BrokerError("compose_broker_immutable_source_required")
    if Path(config.socket).exists() or Path(config.socket).is_symlink():
        raise BrokerError("compose_broker_socket_already_exists")
    os.chown(artifacts, 0, config.client_gid)
    artifacts.chmod(0o750)


@dataclass
class Session:
    provider: CertifiedComposeRuntimeProvider
    directory: Path
    manifest: EnvironmentBundleV2
    instances: int
    runtimes: list[EnvironmentRuntime]


class ComposeBroker:
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.source = Path(config.source)
        self.source_digest = source_fingerprint(self.source)
        self.sessions: dict[str, Session] = {}
        self.created = 0
        self.image_resolutions = 0

    async def dispatch(self, payload: bytes):
        if len(payload) > MAX_MESSAGE:
            raise BrokerError("compose_broker_message_size")
        request = REQUEST.validate_json(payload)
        if isinstance(request, ResolveImagesRequest):
            return await self._resolve_images(request)
        if isinstance(request, ProvisionRequest):
            return await self._provision(request)
        session = self.sessions.get(request.session)
        if session is None:
            raise BrokerError("compose_broker_session_unknown")
        if request.operation == "close":
            await session.provider.close(work_directory=session.directory)
            del self.sessions[request.session]
            return {"closed": True}
        if request.world_index >= len(session.runtimes):
            raise BrokerError("compose_broker_world_unknown")
        runtime = session.runtimes[request.world_index]
        if request.operation == "reset":
            await session.provider.reset(runtime, work_directory=session.directory)
            return {"runtime": runtime.model_dump(mode="json")}
        healthy = await session.provider.healthy(
            runtime, work_directory=session.directory
        )
        return {"healthy": healthy, "runtime": runtime.model_dump(mode="json")}

    def _validated_document(self, request):
        manifest = request.manifest
        if (
            manifest.provenance.public_material_admissions
            != self.config.public_material_admissions
            or manifest.provenance.public_test_fixture_admissions
            != self.config.public_test_fixture_admissions
        ):
            raise BrokerError("compose_broker_public_material_admission_mismatch")
        if manifest.runtime.kind != RuntimeKindV2.COMPOSE or (
            manifest.seed and manifest.seed.stores
        ):
            raise BrokerError("compose_broker_native_compose_required")
        if (
            seal_bundle_v2(manifest) != manifest.digest
            or manifest.provenance.source_digest != self.source_digest
        ):
            raise BrokerError("compose_broker_digest_mismatch")
        raw = request.document.encode()
        records = [
            item for item in manifest.files if item.path == manifest.runtime.document
        ]
        if (
            len(records) != 1
            or records[0].size != len(raw)
            or records[0].sha256 != hashlib.sha256(raw).hexdigest()
        ):
            raise BrokerError("compose_broker_document_digest_mismatch")
        document = json.loads(request.document)
        if (
            not isinstance(document, dict)
            or not isinstance(document.get("services"), dict)
            or len(document["services"]) > 64
        ):
            raise BrokerError("compose_broker_service_limit")
        normalized = normalize_compose(
            document, self.source, capabilities=manifest.capabilities
        )
        return raw, normalized

    async def _resolve_images(self, request):
        self._validated_document(request)
        expected_repository = self.config.source_repository
        supplied_repository = request.manifest.provenance.repository
        if (
            (source_repository(expected_repository) if expected_repository else None)
            != (source_repository(supplied_repository) if supplied_repository else None)
            or self.config.source_revision != request.manifest.provenance.commit
        ):
            raise BrokerError("compose_broker_source_provenance_mismatch")
        selected_images(self.source, request.runtime, json.loads(request.document), control_service=request.manifest.runtime.control_service)
        if self.image_resolutions >= 32:
            raise BrokerError("compose_broker_image_resolution_limit")
        self.image_resolutions += 1
        provider = CertifiedComposeRuntimeProvider(socket_path=Path(self.config.docker_socket))
        identities = await asyncio.to_thread(
            provider.resolve_images, manifest=request.manifest, document=request.document,
            runtime=request.runtime, source=self.source,
        )
        return {"images": [item.model_dump(mode="json") for item in identities]}

    async def _provision(self, request):
        manifest = request.manifest
        raw, normalized = self._validated_document(request)
        for resource in normalized.get("x-alk-config-files", {}).values():
            if any(
                alias not in self.config.target_configuration
                for alias in resource.get("secret_aliases", {}).values()
            ):
                raise BrokerError("compose_broker_unknown_target_alias")
        for service in normalized["services"].values():
            for binding in service.get("x-alk-secret-files", []):
                if binding["alias"] not in self.config.target_configuration:
                    raise BrokerError("compose_broker_unknown_target_alias")
        identifier = request.session
        if identifier is not None:
            session = self.sessions.get(identifier)
            if (
                session is None
                or session.manifest.digest != manifest.digest
                or session.instances != request.instances
            ):
                raise BrokerError("compose_broker_session_mismatch")
        else:
            if (
                len(self.sessions) >= 2
                or self.created >= 32
                or sum(s.instances for s in self.sessions.values()) + request.instances
                > 4
            ):
                raise BrokerError("compose_broker_session_limit")
            identifier = uuid.uuid4().hex
            directory = Path(self.config.scratch) / identifier
            directory.mkdir(mode=0o700)
            bundle_directory = directory / "bundle"
            bundle_directory.mkdir(mode=0o700)
            # Only the verified runtime document is consumed by the privileged provider.
            document_path = bundle_directory / manifest.runtime.document
            document_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            document_path.write_bytes(raw)
            artifact_root = Path(self.config.artifacts) / identifier
            artifact_root.mkdir(mode=0o750)
            os.chown(artifact_root, 0, self.config.client_gid)
            provider = CertifiedComposeRuntimeProvider(
                socket_path=Path(self.config.docker_socket),
                configuration=self.config.target_configuration,
                artifact_root=artifact_root,
            )
            session = Session(provider, directory, manifest, request.instances, [])
            self.sessions[identifier] = session
            self.created += 1
        try:
            session.runtimes = await session.provider.provision(
                manifest,
                source=self.source,
                bundle_dir=session.directory / "bundle",
                work_directory=session.directory,
                instances=request.instances,
            )
        except BaseException:
            await session.provider.close(work_directory=session.directory)
            del self.sessions[identifier]
            raise
        finally:
            self._publish_logs()
        return {
            "session": identifier,
            "runtimes": [item.model_dump(mode="json") for item in session.runtimes],
        }

    def _publish_logs(self):
        for path in Path(self.config.artifacts).glob("*/*.log"):
            if path.is_symlink() or not path.is_file():
                raise BrokerError("compose_broker_artifact_invalid")
            os.chown(path, 0, self.config.client_gid)
            path.chmod(0o640)

    async def close(self):
        outcomes = []
        for identifier, session in self.sessions.items():
            try:
                await session.provider.close(work_directory=session.directory)
                outcomes.append({"session": identifier, "status": "closed"})
            except Exception:
                outcomes.append({"session": identifier, "status": "failed"})
        try:
            self._publish_logs()
            logs = "published"
        except Exception:
            logs = "failed"
        success = logs == "published" and all(
            item["status"] == "closed" for item in outcomes
        )
        receipt = Path(self.config.artifacts) / "cleanup.json"
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": 1,
                    "status": "closed" if success else "failed",
                    "sessions": outcomes,
                    "logs": logs,
                }
            )
        )
        os.chown(temporary, 0, 0)
        temporary.chmod(0o600)
        temporary.replace(receipt)
        if not success:
            raise BrokerError("compose_broker_cleanup_failed") from None
        self.sessions.clear()


def serve(config: BrokerConfig):
    validate_config(config)
    os.umask(0o077)
    broker = ComposeBroker(config)
    stop = False

    def shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    deadline = time.monotonic() + config.lifetime_seconds
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(config.socket)
            os.chown(config.socket, 0, config.client_gid)
            os.chmod(config.socket, 0o620)
            listener.listen(4)
            listener.settimeout(1)
            while not stop and time.monotonic() < deadline:
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(15)
                    _, uid, _ = struct.unpack(
                        "3i",
                        connection.getsockopt(
                            socket.SOL_SOCKET, socket.SO_PEERCRED, 12
                        ),
                    )
                    if uid != config.client_uid:
                        continue
                    try:
                        payload = _receive(connection)
                        result = asyncio.run(broker.dispatch(payload))
                        _send(connection, {"result": result})
                    except Exception as error:
                        code = str(error).split(":", 1)[0]
                        if not re.fullmatch(r"compose_[a-z_]+", code):
                            code = "compose_broker_request_failed"
                        with (Path(config.artifacts) / "events.jsonl").open("a") as log:
                            log.write(
                                json.dumps(
                                    {"error": code, "type": type(error).__name__}
                                )
                                + "\n"
                            )
                        os.chown(
                            Path(config.artifacts) / "events.jsonl",
                            0,
                            config.client_gid,
                        )
                        (Path(config.artifacts) / "events.jsonl").chmod(0o640)
                        try:
                            _send(connection, {"error": code})
                        except OSError:
                            pass
                    finally:
                        broker._publish_logs()
    finally:
        try:
            asyncio.run(broker.close())
        finally:
            Path(config.socket).unlink(missing_ok=True)


class ComposeBrokerClient:
    name = "certified-compose-broker"

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path
        self.session = None

    def _request(self, request):
        _root_controlled(self.socket_path.parent, directory=True)
        info = self.socket_path.lstat()
        if info.st_uid != 0 or not stat.S_ISSOCK(info.st_mode) or info.st_mode & 0o007:
            raise BrokerError("compose_broker_socket_required")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(86460)
            connection.connect(str(self.socket_path))
            if (
                not hasattr(socket, "SO_PEERCRED")
                or struct.unpack(
                    "3i",
                    connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
                )[1]
                != 0
            ):
                raise BrokerError("compose_broker_root_peer_required")
            _send(connection, request)
            response = json.loads(_receive(connection))
        if "error" in response:
            raise BrokerError(response["error"])
        return response["result"]

    def resolve_images(self, *, manifest, document, runtime, source):
        result = self._request({
            "operation": "resolve_images", "manifest": manifest.model_dump(mode="json"),
            "document": document, "runtime": Runtime.model_validate(runtime).model_dump(mode="json"),
        })
        if not isinstance(result.get("images"), list) or len(result["images"]) > 64:
            raise BrokerError("compose_broker_image_limit")
        return [ResolvedComposeImage.model_validate(item) for item in result["images"]]

    async def provision(
        self, bundle, *, source, bundle_dir, work_directory, contract=None, instances=1
    ):
        document = Path(bundle_dir) / bundle.runtime.document
        with document.open("rb") as stream:
            raw = stream.read(MAX_MESSAGE + 1)
        if len(raw) > MAX_MESSAGE:
            raise BrokerError("compose_broker_message_size")
        result = await asyncio.to_thread(
            self._request,
            {
                "operation": "provision",
                "session": self.session,
                "manifest": bundle.model_dump(mode="json"),
                "document": raw.decode(),
                "instances": instances,
            },
        )
        self.session = result["session"]
        return [
            EnvironmentRuntime.model_validate(value) for value in result["runtimes"]
        ]

    async def reset(self, runtime, *, work_directory, keep_processes=False):
        result = await asyncio.to_thread(
            self._request,
            {
                "operation": "reset",
                "session": self.session,
                "world_index": runtime.world_index,
            },
        )
        current = EnvironmentRuntime.model_validate(result["runtime"])
        runtime.state, runtime.endpoints, runtime.metadata = (
            current.state,
            current.endpoints,
            current.metadata,
        )

    async def healthy(self, runtime, *, work_directory):
        result = await asyncio.to_thread(
            self._request,
            {
                "operation": "healthy",
                "session": self.session,
                "world_index": runtime.world_index,
            },
        )
        current = EnvironmentRuntime.model_validate(result["runtime"])
        runtime.state, runtime.endpoints, runtime.metadata = (
            current.state,
            current.endpoints,
            current.metadata,
        )
        return result["healthy"]

    async def close(self, *, work_directory):
        if self.session:
            await asyncio.to_thread(
                self._request, {"operation": "close", "session": self.session}
            )
            self.session = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    _root_controlled(config_path)
    if (
        not config_path.is_file()
        or config_path.stat().st_mode & 0o077
        or config_path.stat().st_size > MAX_MESSAGE
    ):
        raise BrokerError("compose_broker_private_config_required")
    config = BrokerConfig.model_validate_json(config_path.read_bytes())
    serve(config)


if __name__ == "__main__":
    main()
