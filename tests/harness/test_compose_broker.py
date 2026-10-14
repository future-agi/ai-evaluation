import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import struct

import pytest
from pydantic import ValidationError

from fi.alk.harness import compose_broker as module
from fi.alk.harness.bundle_v2 import (
    BUNDLE_V2_SCHEMA_VERSION,
    BundleFileV2,
    BundleProvenanceV2,
    BundleRuntimeV2,
    CapabilityV2,
    EnvironmentBundleV2,
    seal_bundle_v2,
)
from fi.alk.harness.process_runtime import (
    EnvironmentRuntime,
    RuntimeEndpoint,
    RuntimeState,
)
from fi.alk.harness.provision import source_fingerprint


def request(source, document=None, **changes):
    document = document or {"services": {"chat": {"image": "example/chat:1"}}}
    raw = json.dumps(document)
    manifest = EnvironmentBundleV2(
        schema_version=BUNDLE_V2_SCHEMA_VERSION,
        digest="sha256:" + "0" * 64,
        name="test",
        runtime=BundleRuntimeV2(
            kind="compose",
            document="compose.json",
            control_service="chat",
            evidence_seam="tool_trace",
        ),
        capabilities={
            "target_http": CapabilityV2(
                protocol="http", service="chat", container_port=8000
            )
        },
        files=[
            BundleFileV2(
                path="compose.json",
                sha256=hashlib.sha256(raw.encode()).hexdigest(),
                size=len(raw.encode()),
            )
        ],
        provenance=BundleProvenanceV2(
            source_kind="local",
            source_digest=source_fingerprint(source),
            generator="test",
            generator_version="1",
        ),
    )
    manifest = manifest.model_copy(update={"digest": seal_bundle_v2(manifest)})
    result = {
        "operation": "provision",
        "manifest": manifest.model_dump(mode="json"),
        "document": raw,
        "instances": 1,
    }
    result.update(changes)
    return result


class FakeProvider:
    instances = []

    def __init__(self, **kwargs):
        self.config = kwargs
        self.runtimes = []
        self.closed = False
        type(self).instances.append(self)

    async def provision(
        self, manifest, *, source, bundle_dir, work_directory, instances
    ):
        assert source.name == "source"
        assert (bundle_dir / "compose.json").is_file()
        self.runtimes = [
            EnvironmentRuntime(
                runtime_id=f"runtime-{index}",
                world_index=index,
                bundle_digest=manifest.digest,
                state=RuntimeState.READY,
                endpoints={
                    "target_http": RuntimeEndpoint(
                        capability="target_http",
                        protocol="http",
                        address="http://127.0.0.1:12345",
                    )
                },
                metadata={"observation_mode": "native"},
            )
            for index in range(instances)
        ]
        return self.runtimes

    async def reset(self, runtime, **kwargs):
        runtime.endpoints["target_http"].address = "http://127.0.0.1:12346"

    async def healthy(self, runtime, **kwargs):
        return True

    async def close(self, **kwargs):
        self.closed = True
        for runtime in self.runtimes:
            runtime.state = RuntimeState.STOPPED


@pytest.fixture
def broker(tmp_path, monkeypatch):
    for name in ("source", "scratch", "artifacts"):
        (tmp_path / name).mkdir()
    (tmp_path / "source" / "app.py").write_text("print('hello')\n")
    FakeProvider.instances = []
    monkeypatch.setattr(module, "CertifiedComposeRuntimeProvider", FakeProvider)
    monkeypatch.setattr(module.os, "chown", lambda *args: None)
    config = module.BrokerConfig(
        source=str(tmp_path / "source"),
        scratch=str(tmp_path / "scratch"),
        artifacts=str(tmp_path / "artifacts"),
        socket=str(tmp_path / "broker.sock"),
        docker_socket="/root/docker.sock",
        client_uid=1000,
        client_gid=1000,
        target_configuration={"ADC": "secret-only-at-server"},
    )
    return module.ComposeBroker(config)


def dispatch(broker, payload):
    return asyncio.run(broker.dispatch(json.dumps(payload).encode()))


def test_lifecycle_and_precise_native_runtime(broker):
    created = dispatch(broker, request(broker.source, instances=2))
    identifier = created["session"]
    assert len(created["runtimes"]) == 2
    assert created["runtimes"][1]["metadata"] == {"observation_mode": "native"}
    assert created["runtimes"][1]["state"] == "ready"
    assert (
        dispatch(
            broker, {"operation": "healthy", "session": identifier, "world_index": 1}
        )["healthy"]
        is True
    )
    reset = dispatch(broker, {"operation": "reset", "session": identifier})
    assert (
        reset["runtime"]["endpoints"]["target_http"]["address"]
        == "http://127.0.0.1:12346"
    )
    assert (
        FakeProvider.instances[0].runtimes[1].endpoints["target_http"].address
        == "http://127.0.0.1:12345"
    )
    assert dispatch(broker, {"operation": "close", "session": identifier}) == {
        "closed": True
    }
    assert FakeProvider.instances[0].closed
    assert not broker.sessions
    second = dispatch(broker, request(broker.source))
    assert second["session"] != identifier
    asyncio.run(broker.close())
    assert FakeProvider.instances[1].closed


@pytest.mark.parametrize(
    "extra",
    [
        {"source": "/etc"},
        {"configuration": {"KEY": "value"}},
        {"command": ["sh"]},
        {"instances": 5},
    ],
)
def test_protocol_rejects_paths_commands_secrets_and_excess_worlds(broker, extra):
    with pytest.raises(ValidationError):
        dispatch(broker, {**request(broker.source), **extra})
    assert not FakeProvider.instances


def test_document_digest_and_source_digest_enforced(broker):
    payload = request(broker.source)
    payload["document"] += " "
    with pytest.raises(module.BrokerError, match="document_digest"):
        dispatch(broker, payload)
    payload = request(broker.source)
    payload["manifest"]["provenance"]["source_digest"] = "0" * 64
    with pytest.raises(module.BrokerError, match="digest_mismatch"):
        dispatch(broker, payload)
    assert not FakeProvider.instances


def test_unknown_target_secret_file_alias_rejected(broker):
    doc = {
        "services": {
            "chat": {
                "image": "example/chat:1",
                "x-alk-secret-files": [
                    {"alias": "SIMULATOR_KEY", "destination": "/tmp/key"}
                ],
            }
        }
    }
    with pytest.raises(module.BrokerError, match="unknown_target_alias"):
        dispatch(broker, request(broker.source, doc))
    assert not FakeProvider.instances


@pytest.mark.parametrize(
    "fields",
    [{"volumes": ["/etc:/host"]}, {"privileged": True}, {"network_mode": "host"}],
)
def test_host_control_rejected_before_provider(broker, fields):
    doc = {"services": {"chat": {"image": "example/chat:1", **fields}}}
    with pytest.raises(RuntimeError):
        dispatch(broker, request(broker.source, doc))
    assert not FakeProvider.instances


def test_malformed_and_oversized_requests_rejected(broker):
    with pytest.raises(ValidationError):
        asyncio.run(broker.dispatch(b"not json"))
    with pytest.raises(module.BrokerError, match="message_size"):
        asyncio.run(broker.dispatch(b" " * (module.MAX_MESSAGE + 1)))


def test_length_prefix_rejected_before_body_read():
    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack("!I", module.MAX_MESSAGE + 1))
        with pytest.raises(module.BrokerError, match="message_size"):
            module._receive(right)


def test_client_transports_no_host_paths_or_secret_values(
    broker, tmp_path, monkeypatch
):
    payload = request(broker.source)
    bundle = module.EnvironmentBundleV2.model_validate(payload["manifest"])
    bundle_dir = tmp_path / "client-bundle"
    bundle_dir.mkdir()
    (bundle_dir / "compose.json").write_text(payload["document"])
    client = module.ComposeBrokerClient(Path("/unused"))
    observed = []

    def transport(message):
        observed.append(message)
        return dispatch(broker, message)

    monkeypatch.setattr(client, "_request", transport)

    async def run():
        worlds = await client.provision(
            bundle,
            source=Path("/private/source"),
            bundle_dir=bundle_dir,
            work_directory=Path("/private/work"),
        )
        await client.reset(worlds[0], work_directory=tmp_path)
        assert worlds[0].endpoints["target_http"].address == "http://127.0.0.1:12346"
        assert await client.healthy(worlds[0], work_directory=tmp_path)
        await client.close(work_directory=tmp_path)

    asyncio.run(run())
    assert set(observed[0]) == {
        "operation",
        "session",
        "manifest",
        "document",
        "instances",
    }
    assert "secret-only-at-server" not in json.dumps(observed)
    assert "/private/" not in json.dumps(observed)
    assert FakeProvider.instances[0].config["configuration"] == {
        "ADC": "secret-only-at-server"
    }


def test_broker_opt_in_precedes_direct_provider(monkeypatch, tmp_path):
    from fi.alk.harness import compose_integration

    monkeypatch.setenv("ALK_COMPOSE_BROKER_SOCKET", "/run/job/broker.sock")
    monkeypatch.setenv("ALK_TRUSTED_COMPOSE_SOCKET", "/run/docker.sock")
    assert compose_integration.enabled()
    provider = compose_integration.provider({"KEY": "not-transmitted"}, tmp_path)
    assert isinstance(provider, module.ComposeBrokerClient)
    assert provider.socket_path == Path("/run/job/broker.sock")


def test_session_limits_and_unknown_world(broker):
    first = dispatch(broker, request(broker.source, instances=3))
    with pytest.raises(module.BrokerError, match="session_limit"):
        dispatch(broker, request(broker.source, instances=2))
    with pytest.raises(module.BrokerError, match="world_unknown"):
        dispatch(
            broker,
            {"operation": "healthy", "session": first["session"], "world_index": 3},
        )
    with pytest.raises(module.BrokerError, match="session_unknown"):
        dispatch(broker, {"operation": "healthy", "session": "0" * 32})
    with pytest.raises(module.BrokerError, match="session_mismatch"):
        dispatch(broker, request(broker.source, session=first["session"], instances=1))
    asyncio.run(broker.close())


def test_failed_provision_cleans_resources(broker, monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr(FakeProvider, "provision", fail)
    with pytest.raises(RuntimeError, match="build failed"):
        dispatch(broker, request(broker.source))
    assert FakeProvider.instances[0].closed
    assert not broker.sessions


def test_configuration_rejects_extra_launch_fields():
    with pytest.raises(ValidationError):
        module.BrokerConfig.model_validate({"command": "sh"})


def test_real_root_broker_unprivileged_client_and_shutdown(tmp_path):
    import os
    import subprocess
    import sys
    import tempfile
    import time

    root = os.environ.get("ALK_TEST_COMPOSE_BROKER_ROOT")
    docker_socket = os.environ.get("ALK_TEST_COMPOSE_SOCKET")
    image = os.environ.get("ALK_TEST_COMPOSE_BUSYBOX_IMAGE")
    if not root or not docker_socket or not image:
        pytest.skip(
            "requires Linux root broker directory, private Docker socket and pinned BusyBox image"
        )
    assert os.geteuid() == 0 and hasattr(socket, "SO_PEERCRED")
    assert "@sha256:" in image
    client_uid = int(os.environ["ALK_TEST_COMPOSE_CLIENT_UID"])
    client_gid = int(os.environ["ALK_TEST_COMPOSE_CLIENT_GID"])
    directory = Path(tempfile.mkdtemp(dir=root, prefix="broker-test-"))
    print(f"broker_canary_artifacts={directory}")
    directory.chmod(0o750)
    os.chown(directory, 0, client_gid)
    for name in ("source", "scratch", "artifacts"):
        (directory / name).mkdir(mode=0o755 if name == "source" else 0o700)
    source = directory / "source"
    (source / "index.html").write_text("broker-native-chat-ready")
    (source / "index.html").chmod(0o644)
    doc = {
        "services": {
            "chat": {
                "image": image,
                "command": ["httpd", "-f", "-p", "8000", "-h", "/www"],
                "volumes": ["./:/www:ro"],
            }
        }
    }
    payload = request(source, doc)
    config = module.BrokerConfig(
        source=str(source),
        scratch=str(directory / "scratch"),
        artifacts=str(directory / "artifacts"),
        socket=str(directory / "broker.sock"),
        docker_socket=docker_socket,
        client_uid=client_uid,
        client_gid=client_gid,
    )
    config_path = directory / "config.json"
    config_path.write_text(config.model_dump_json())
    config_path.chmod(0o600)
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    log = (directory / "controller.log").open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fi.alk.harness.compose_broker",
            "--config",
            str(config_path),
        ],
        env=environment,
        stdout=log,
        stderr=log,
    )
    try:
        deadline = time.monotonic() + 30
        while not Path(config.socket).exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.1)
        # The root controller is not the authorized harness identity.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unauthorized:
            unauthorized.settimeout(5)
            unauthorized.connect(config.socket)
            assert unauthorized.recv(1) == b""
        script = """
import json, sys, time, urllib.request
from pathlib import Path
from fi.alk.harness.compose_broker import ComposeBrokerClient
client = ComposeBrokerClient(Path(sys.argv[1]))
result = client._request(json.load(sys.stdin))
runtime = result["runtimes"][0]
url = runtime["endpoints"]["target_http"]["address"]
for attempt in range(50):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            assert response.read() == b"broker-native-chat-ready"
        break
    except OSError:
        if attempt == 49: raise
        time.sleep(0.1)
assert client._request({"operation":"healthy", "session":result["session"]})["healthy"]
print(json.dumps({"runtime_id":runtime["runtime_id"], "state":runtime["state"]}))
"""

        def identity():
            os.setgroups([])
            os.setgid(client_gid)
            os.setuid(client_uid)

        result = subprocess.run(
            [sys.executable, "-c", script, config.socket],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            env=environment,
            preexec_fn=identity,
            timeout=240,
        )
        assert result.returncode == 0, result.stderr
        record = json.loads(result.stdout)
        assert record["state"] == "ready"
        process.terminate()
        assert process.wait(timeout=180) == 0
        remaining = subprocess.run(
            [
                "docker",
                "--host",
                f"unix://{docker_socket}",
                "ps",
                "-aq",
                "--filter",
                f"label=com.docker.compose.project={record['runtime_id']}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert remaining.stdout.strip() == ""
        assert not Path(config.socket).exists()
        assert (
            json.loads((directory / "artifacts/cleanup.json").read_text())["status"]
            == "closed"
        )
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=180)
        log.close()


def test_failed_shutdown_writes_receipt_before_removing_socket(broker, monkeypatch):
    from unittest.mock import MagicMock

    dispatch(broker, request(broker.source))

    async def fail(self, **kwargs):
        raise RuntimeError("credential-must-not-appear")

    monkeypatch.setattr(FakeProvider, "close", fail)
    monkeypatch.setattr(module, "validate_config", lambda config: None)
    monkeypatch.setattr(module, "ComposeBroker", lambda config: broker)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    listener = MagicMock()
    listener.__enter__.return_value = listener
    listener.accept.side_effect = KeyboardInterrupt
    from types import SimpleNamespace

    monkeypatch.setattr(
        module,
        "socket",
        SimpleNamespace(
            socket=lambda *args: listener,
            AF_UNIX=socket.AF_UNIX,
            SOCK_STREAM=socket.SOCK_STREAM,
        ),
    )
    Path(broker.config.socket).touch()
    previous_umask = os.umask(0o077)
    try:
        with pytest.raises(module.BrokerError, match="cleanup_failed"):
            module.serve(broker.config)
    finally:
        os.umask(previous_umask)
    assert not Path(broker.config.socket).exists()
    path = Path(broker.config.artifacts) / "cleanup.json"
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "failed"
    assert receipt["sessions"][0]["status"] == "failed"
    assert receipt["logs"] == "published"
    assert "credential-must-not-appear" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600


def test_unknown_target_config_alias_rejected(broker):
    doc = {
        "x-alk-config-files": {".env": {"secret_aliases": {"KEY": "SIMULATOR_KEY"}}},
        "services": {"chat": {"image": "example/chat:1", "env_file": ["./.env"]}},
    }
    with pytest.raises(module.BrokerError, match="unknown_target_alias"):
        dispatch(broker, request(broker.source, doc))
    assert not FakeProvider.instances
