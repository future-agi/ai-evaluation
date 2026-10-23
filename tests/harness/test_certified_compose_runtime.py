from pathlib import Path
from types import SimpleNamespace
import asyncio
import json
import subprocess

import pytest

from fi.alk.harness.certified_compose_runtime import (
    CertifiedComposeRuntimeProvider,
    ComposeRuntimeError,
    normalize_compose,
    _copy_source,
)
from fi.alk.harness.bundle_v2 import CapabilityV2, RuntimeKindV2, ReadinessProbeV2
from fi.alk.harness.process_runtime import RuntimeState


def topology():
    return {
        "services": {
            "chat": {
                "image": "example/chat:1",
                "depends_on": ["cache"],
                "ports": ["0.0.0.0:80:8000"],
            },
            "cache": {"image": "redis:7", "volumes": ["data:/data"]},
        },
        "volumes": {"data": {}},
    }


def capabilities():
    return {"chat": CapabilityV2(protocol="http", service="chat", container_port=8000)}


def test_publish_only_declared_loopback_ports_and_project_owned_volume(tmp_path):
    original = topology()
    normalized = normalize_compose(original, tmp_path, capabilities=capabilities())
    assert normalized["services"]["chat"]["ports"] == [
        {"target": 8000, "host_ip": "127.0.0.1", "published": "0", "protocol": "tcp"}
    ]
    assert normalized["services"]["cache"]["ports"] == []
    assert normalized["services"]["cache"]["volumes"] == [
        {"type": "volume", "source": "data", "target": "/data", "read_only": False}
    ]
    assert original == topology()


@pytest.mark.parametrize(
    "key,value",
    [
        ("privileged", True),
        ("network_mode", "host"),
        ("pid", "host"),
        ("devices", ["/dev/kvm"]),
        ("cap_add", ["SYS_ADMIN"]),
        ("security_opt", ["seccomp=unconfined"]),
        ("volumes_from", ["another"]),
        ("extends", {"file": "other.yml"}),
        ("post_start", [{"command": "echo hi", "privileged": True}]),
    ],
)
def test_reject_host_control_fields(tmp_path, key, value):
    document = topology()
    document["services"]["chat"][key] = value
    with pytest.raises(ComposeRuntimeError, match="fields_forbidden"):
        normalize_compose(document, tmp_path, capabilities=capabilities())


@pytest.mark.parametrize(
    "mount",
    [
        "/var/run/docker.sock:/var/run/docker.sock",
        "../escape:/data",
        {
            "type": "bind",
            "source": ".",
            "target": "/data",
            "bind": {"propagation": "rshared"},
        },
        "${EVIL}:/data",
    ],
)
def test_reject_host_mounts(tmp_path, mount):
    document = topology()
    document["services"]["chat"]["volumes"] = [mount]
    with pytest.raises(ComposeRuntimeError):
        normalize_compose(document, tmp_path, capabilities=capabilities())


@pytest.mark.parametrize(
    "kind,spec",
    [
        ("volumes", {"external": True}),
        ("volumes", {"name": "shared"}),
        ("volumes", {"driver_opts": {"device": "/"}}),
        ("networks", {"external": True}),
        ("networks", {"driver": "host"}),
    ],
)
def test_reject_shared_or_host_resources(tmp_path, kind, spec):
    document = topology()
    document[kind] = {"data": spec}
    with pytest.raises(ComposeRuntimeError):
        normalize_compose(document, tmp_path, capabilities=capabilities())


def test_source_build_and_mount_rebased_inside_world(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM busybox\n")
    document = {"services": {"chat": {"build": ".", "volumes": ["./:/app:ro"]}}}
    normalized = normalize_compose(document, tmp_path, capabilities=capabilities())
    assert normalized["services"]["chat"]["build"]["context"] == str(tmp_path)
    assert normalized["services"]["chat"]["volumes"][0]["source"] == str(tmp_path)


def test_source_symlink_escape_never_reaches_daemon(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "outside").symlink_to(tmp_path)
    with pytest.raises(ComposeRuntimeError, match="symlink_escape"):
        _copy_source(source, tmp_path / "copy")


@pytest.mark.parametrize(
    "key",
    ["DOCKER_HOST", "COMPOSE_DISABLE_ENV_FILE", "BUILDX_CONFIG", "LD_PRELOAD", "PATH"],
)
def test_configuration_cannot_control_trusted_client(tmp_path, key):
    with pytest.raises(ComposeRuntimeError, match="reserved_configuration"):
        CertifiedComposeRuntimeProvider(
            socket_path=tmp_path / "socket", configuration={key: "bad"}
        )


def test_lifecycle_two_worlds_reset_and_cleanup(tmp_path, monkeypatch):
    source, bundle_dir, work = (tmp_path / name for name in ("src", "bundle", "work"))
    for path in (source, bundle_dir, work):
        path.mkdir()
    (source / "fixture.txt").write_text("baseline")
    (bundle_dir / "compose.json").write_text(json.dumps(topology()))
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(kind=RuntimeKindV2.COMPOSE, document="compose.json"),
        seed=None,
        digest="sha256:" + "a" * 64,
        capabilities=capabilities(),
        readiness=[],
    )
    provider = CertifiedComposeRuntimeProvider(socket_path=tmp_path / "socket")
    monkeypatch.setattr(provider, "_check_socket", lambda: None)
    commands = []

    def command(directory, project, *args, **kwargs):
        commands.append((project, args))
        if args[0] == "config":
            return json.dumps({"services": {}})
        if args[0] == "port":
            return "127.0.0.1:18000\n"
        if args[0] == "ps":
            return "chat\ncache\n"
        return ""

    monkeypatch.setattr(provider, "_command", command)

    async def run():
        worlds = await provider.provision(
            bundle,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=work,
            instances=2,
        )
        assert worlds[0].runtime_id != worlds[1].runtime_id
        assert all(world.metadata == {"observation_mode": "native"} for world in worlds)
        directories = [entry[1] for entry in provider._worlds.values()]
        (directories[0] / "source/fixture.txt").write_text("changed0")
        (directories[1] / "source/fixture.txt").write_text("changed1")
        await provider.reset(worlds[0], work_directory=work)
        assert (directories[0] / "source/fixture.txt").read_text() == "baseline"
        assert (directories[1] / "source/fixture.txt").read_text() == "changed1"
        assert await provider.healthy(worlds[0], work_directory=work)
        await provider.close(work_directory=work)
        assert all(world.state is RuntimeState.STOPPED for world in worlds)
        assert all(not directory.exists() for directory in directories)
        assert sum(args[0] == "down" for _, args in commands) == 3
        assert all("--volumes" in args for _, args in commands if args[0] == "down")

    asyncio.run(run())


def test_command_does_not_inherit_controller_secrets_and_redacts_logs(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CONTROLLER_SECRET", "never-forward")
    captured = {}

    def popen(command, **kwargs):
        captured.update(kwargs)
        captured["command"] = command
        kwargs["stdout"].write(b"credential: target-secret\n")
        return SimpleNamespace(wait=lambda **kwargs: 0, returncode=0)

    monkeypatch.setattr(subprocess, "Popen", popen)
    provider = CertifiedComposeRuntimeProvider(
        socket_path=tmp_path / "socket",
        configuration={"API_KEY": "target-secret", "LD_AUDIT": "/untrusted.so"},
        artifact_root=tmp_path / "logs",
    )
    provider._command(tmp_path, "project", "logs")
    assert "CONTROLLER_SECRET" not in captured["env"]
    assert "API_KEY" not in captured["env"]
    assert "LD_AUDIT" not in captured["env"]
    assert captured["command"][2:4] == [
        "--env-file",
        str(tmp_path / "configuration.env"),
    ]
    log = (tmp_path / "logs/project.log").read_text()
    assert "target-secret" not in log
    assert "[REDACTED]" in log


def test_only_contained_declared_env_files_are_read(tmp_path):
    (tmp_path / "config.env").write_text("SERVICE_TOKEN=fixture-token\n")
    document = topology()
    document["services"]["chat"]["env_file"] = [
        "config.env",
        {"path": "absent.env", "required": False},
    ]
    result = normalize_compose(document, tmp_path, capabilities=capabilities())
    assert result["services"]["chat"]["env_file"] == [
        {"path": str(tmp_path / "config.env"), "required": True}
    ]
    document["services"]["chat"]["env_file"] = "/etc/environment"
    with pytest.raises(ComposeRuntimeError, match="env_file_invalid"):
        normalize_compose(document, tmp_path, capabilities=capabilities())
    document["services"]["chat"]["env_file"] = "../escape.env"
    with pytest.raises(ComposeRuntimeError, match="outside_source"):
        normalize_compose(document, tmp_path, capabilities=capabilities())


def test_real_compose_chat_native_store_reset_and_cleanup(tmp_path):
    """Run explicitly inside the isolated Docker-capable certification VM."""
    import os
    import urllib.request

    socket_path = os.environ.get("ALK_TEST_COMPOSE_SOCKET")
    if not socket_path:
        pytest.skip("requires private Docker certification socket")
    python_image = os.environ["ALK_TEST_COMPOSE_PYTHON_IMAGE"]
    redis_image = os.environ["ALK_TEST_COMPOSE_REDIS_IMAGE"]
    assert "@sha256:" in python_image and "@sha256:" in redis_image
    source, bundle_dir, work = (tmp_path / name for name in ("src", "bundle", "work"))
    for path in (source, bundle_dir, work):
        path.mkdir()
    (source / "Dockerfile").write_text(
        f'FROM {python_image}\nWORKDIR /app\nCOPY server.py /app/server.py\nUSER 65534\nCMD ["python", "server.py"]\n'
    )
    (source / "server.py").write_text("""
import http.server, json, socket

def redis(*args):
    data = b"*%d\\r\\n" % len(args)
    for arg in args:
        raw = arg.encode()
        data += b"$%d\\r\\n" % len(raw) + raw + b"\\r\\n"
    with socket.create_connection(("store", 6379), timeout=5) as conn:
        conn.sendall(data)
        stream = conn.makefile("rb")
        line = stream.readline()
        if line == b"$-1\\r\\n":
            return None
        if line.startswith(b"$"):
            return stream.read(int(line[1:])).decode()
        return line.decode().strip()

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ready")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        value = redis("SET", body["key"], body["value"]) if self.path == "/seed" else redis("GET", body["message"])
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"response": value}).encode())
http.server.ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
""")
    document = {
        "services": {
            "chat": {
                "build": ".",
                "depends_on": {"store": {"condition": "service_healthy"}},
            },
            "store": {
                "image": redis_image,
                "command": ["redis-server", "--appendonly", "yes"],
                "healthcheck": {
                    "test": ["CMD", "redis-cli", "ping"],
                    "interval": "1s",
                    "timeout": "1s",
                    "retries": 30,
                },
                "volumes": ["data:/data"],
            },
        },
        "volumes": {"data": {}},
    }
    (bundle_dir / "compose.json").write_text(json.dumps(document))
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(kind=RuntimeKindV2.COMPOSE, document="compose.json"),
        seed=None,
        digest="sha256:" + "c" * 64,
        capabilities=capabilities(),
        readiness=[ReadinessProbeV2(capability="chat", path="/", timeout_seconds=30)],
    )
    provider = CertifiedComposeRuntimeProvider(
        socket_path=Path(socket_path), artifact_root=tmp_path / "logs"
    )

    def post(world, path, body):
        request = urllib.request.Request(
            world.endpoints["chat"].address + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)["response"]

    async def run():
        worlds = []
        try:
            worlds = await provider.provision(
                bundle,
                source=source,
                bundle_dir=bundle_dir,
                work_directory=work,
                instances=2,
            )
            assert (
                worlds[0].endpoints["chat"].address
                != worlds[1].endpoints["chat"].address
            )
            for index, world in enumerate(worlds):
                assert (
                    post(
                        world, "/seed", {"key": "question", "value": f"answer-{index}"}
                    )
                    == "+OK"
                )
                assert (
                    post(world, "/chat", {"message": "question"}) == f"answer-{index}"
                )
            await provider.reset(worlds[0], work_directory=work)
            assert post(worlds[0], "/chat", {"message": "question"}) is None
            assert post(worlds[1], "/chat", {"message": "question"}) == "answer-1"
            assert all(
                [await provider.healthy(world, work_directory=work) for world in worlds]
            )
        finally:
            projects = [world.runtime_id for world in worlds]
            await provider.close(work_directory=work)
        for project in projects:
            for kind in ("container", "network", "volume"):
                result = subprocess.run(
                    [
                        "docker",
                        "--host",
                        f"unix://{socket_path}",
                        kind,
                        "ls",
                        "--filter",
                        f"label=com.docker.compose.project={project}",
                        "--quiet",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                assert result.stdout.strip() == "", (project, kind)
        for world in worlds:
            with pytest.raises(OSError):
                urllib.request.urlopen(world.endpoints["chat"].address, timeout=2)

    asyncio.run(run())


@pytest.mark.parametrize("args", [["API_KEY"], {"API_KEY": None}])
def test_build_cannot_inherit_configuration(tmp_path, args):
    (tmp_path / "Dockerfile").write_text("FROM busybox\n")
    document = {"services": {"chat": {"build": {"context": ".", "args": args}}}}
    with pytest.raises(ComposeRuntimeError, match="build_passthrough_forbidden"):
        normalize_compose(document, tmp_path, capabilities=capabilities())


@pytest.mark.parametrize(
    "configuration_value",
    ["quoted'\n$literal", "trailing\\", "two\\\\slashes", "quote\\'", "${UNSET}"],
)
def test_mutable_source_env_file_cannot_change_control_or_cleanup(
    tmp_path, monkeypatch, configuration_value
):
    source, bundle_dir, work = (tmp_path / name for name in ("src", "bundle", "work"))
    for path in (source, bundle_dir, work):
        path.mkdir()
    (source / "config.env").write_text("TOKEN=original-token\n")
    document = topology()
    document["services"]["chat"].update(env_file="config.env", volumes=["./:/app"])
    (bundle_dir / "compose.json").write_text(json.dumps(document))
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(kind=RuntimeKindV2.COMPOSE, document="compose.json"),
        seed=None,
        digest="sha256:" + "a" * 64,
        capabilities=capabilities(),
        readiness=[],
    )
    provider = CertifiedComposeRuntimeProvider(
        socket_path=tmp_path / "socket", configuration={"API_KEY": configuration_value}
    )
    monkeypatch.setattr(provider, "_check_socket", lambda: None)
    seen = []

    def command(directory, project, *args, **kwargs):
        saved = json.loads((directory / "compose.json").read_text())
        env_path = Path(saved["services"]["chat"]["env_file"][0]["path"])
        assert env_path.parent == directory
        assert env_path.read_text() == "TOKEN=original-token\n"
        assert env_path.stat().st_mode & 0o777 == 0o600
        if args[0] == "up":
            mutable = directory / "source/config.env"
            mutable.unlink()
            mutable.symlink_to(tmp_path / "missing-controller-env")
            encoded = (directory / "configuration.env").read_text().split("=", 1)[1]
            assert json.loads(encoded) == configuration_value.replace("$", "$$")
            import shutil

            if shutil.which("docker"):
                result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "--env-file",
                        str(directory / "configuration.env"),
                        "-f",
                        str(directory / "compose.json"),
                        "config",
                        "--environment",
                    ],
                    env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                assert "API_KEY=" + configuration_value + "\n" in result.stdout
        seen.append(args[0])
        if args[0] == "config":
            return json.dumps(
                {"services": {"chat": {"environment": {"TOKEN": "original-token"}}}}
            )
        return "127.0.0.1:18000\n" if args[0] == "port" else ""

    monkeypatch.setattr(provider, "_command", command)

    async def run():
        await provider.provision(
            bundle, source=source, bundle_dir=bundle_dir, work_directory=work
        )
        assert "original-token" in provider._redactions
        await provider.close(work_directory=work)

    asyncio.run(run())
    assert seen == ["config", "up", "images", "port", "logs", "down"]


def test_timeout_logs_output_and_kills_cli_descendants(tmp_path, monkeypatch):
    import sys
    import time

    real_popen = subprocess.Popen
    marker = tmp_path / "survived"
    script = (
        "import subprocess,sys,time; "
        "print('failure target-secret',flush=True); "
        f"subprocess.Popen([sys.executable,'-c', {('import time,pathlib; time.sleep(1); pathlib.Path(' + repr(str(marker)) + ').touch()')!r}]); "
        "time.sleep(30)"
    )

    def popen(command, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    provider = CertifiedComposeRuntimeProvider(
        socket_path=tmp_path / "socket",
        configuration={"API_KEY": "target-secret"},
        artifact_root=tmp_path / "logs",
    )
    with pytest.raises(ComposeRuntimeError, match="command_timeout"):
        provider._command(tmp_path, "project", "up", timeout=0.3)
    log = (tmp_path / "logs/project.log").read_text()
    assert "failure [REDACTED]" in log
    assert "target-secret" not in log
    assert "timeout=True" in log
    time.sleep(1)
    assert not marker.exists()


@pytest.mark.parametrize(
    "assignment,expected",
    [
        ("TOKEN='secret\\\\suffix'\n", "secret\\\\suffix"),
        ('TOKEN="secret\\\\suffix"\n', "secret\\suffix"),
        ("TOKEN=${API_KEY}-suffix\n", "configured-secret-suffix"),
    ],
)
def test_real_compose_resolved_env_values_are_redacted(
    tmp_path, monkeypatch, assignment, expected
):
    import shutil

    if not shutil.which("docker"):
        pytest.skip("requires Docker Compose CLI; no daemon needed")
    source, bundle_dir, work = (tmp_path / name for name in ("src", "bundle", "work"))
    for path in (source, bundle_dir, work):
        path.mkdir()
    (source / "config.env").write_text(assignment)
    document = {"services": {"chat": {"image": "busybox", "env_file": "config.env"}}}
    (bundle_dir / "compose.json").write_text(json.dumps(document))
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(kind=RuntimeKindV2.COMPOSE, document="compose.json"),
        seed=None,
        digest="sha256:" + "a" * 64,
        capabilities={},
        readiness=[],
    )
    provider = CertifiedComposeRuntimeProvider(
        socket_path=Path("/tmp/alk-no-daemon.sock"),
        configuration={"API_KEY": "configured-secret"},
        artifact_root=tmp_path / "logs",
    )
    monkeypatch.setattr(provider, "_check_socket", lambda: None)
    real_command = provider._command

    def command(directory, project, *args, **kwargs):
        if args[0] == "config":
            # Docker Desktop installs Compose in the user plugin directory.
            plugin = Path.home() / ".docker/cli-plugins/docker-compose"
            if plugin.exists():
                plugins = directory / "docker-config/cli-plugins"
                plugins.mkdir(parents=True)
                (plugins / "docker-compose").symlink_to(plugin.resolve())
            return real_command(directory, project, *args, **kwargs)
        return ""

    monkeypatch.setattr(provider, "_command", command)

    async def run():
        worlds = await provider.provision(
            bundle, source=source, bundle_dir=bundle_dir, work_directory=work
        )
        assert expected in provider._redactions

        def popen(command, **kwargs):
            kwargs["stdout"].write(expected.encode())
            return SimpleNamespace(wait=lambda **kwargs: 0, returncode=0)

        monkeypatch.setattr(subprocess, "Popen", popen)
        real_command(work, worlds[0].runtime_id, "logs")
        log = (tmp_path / "logs" / f"{worlds[0].runtime_id}.log").read_text()
        assert expected not in log
        assert "[REDACTED]" in log
        assert '"services"' not in log
        await provider.close(work_directory=work)

    asyncio.run(run())


def test_config_failure_does_not_log_unresolved_secret(tmp_path, monkeypatch):
    def popen(command, **kwargs):
        kwargs["stdout"].write(b"unknown-secret-in-config-error")
        assert kwargs["stderr"] == subprocess.DEVNULL
        return SimpleNamespace(wait=lambda **kwargs: 1, returncode=1)

    monkeypatch.setattr(subprocess, "Popen", popen)
    provider = CertifiedComposeRuntimeProvider(
        socket_path=tmp_path / "socket", artifact_root=tmp_path / "logs"
    )
    with pytest.raises(
        ComposeRuntimeError, match="compose_command_failed: config exit=1"
    ):
        provider._command(
            tmp_path, "project", "config", "--format", "json", private_output=True
        )
    assert "unknown-secret" not in (tmp_path / "logs/project.log").read_text()


def test_failed_resolution_never_collects_unredacted_service_logs(
    tmp_path, monkeypatch
):
    source, bundle_dir, work = (tmp_path / name for name in ("src", "bundle", "work"))
    for path in (source, bundle_dir, work):
        path.mkdir()
    (bundle_dir / "compose.json").write_text(json.dumps(topology()))
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(kind=RuntimeKindV2.COMPOSE, document="compose.json"),
        seed=None,
        digest="sha256:" + "a" * 64,
        capabilities=capabilities(),
        readiness=[],
    )
    provider = CertifiedComposeRuntimeProvider(socket_path=tmp_path / "socket")
    monkeypatch.setattr(provider, "_check_socket", lambda: None)
    commands = []

    def command(directory, project, *args, **kwargs):
        commands.append(args[0])
        raise ComposeRuntimeError("compose_command_failed: config exit=1")

    monkeypatch.setattr(provider, "_command", command)
    with pytest.raises(ComposeRuntimeError, match="compose_command_failed: config"):
        asyncio.run(
            provider.provision(
                bundle, source=source, bundle_dir=bundle_dir, work_directory=work
            )
        )
    assert commands == ["config"]
    assert list(work.iterdir()) == []
