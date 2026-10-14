import json
from pathlib import Path

import pytest

from fi.alk.harness.certified_compose_runtime import (
    normalize_compose,
    ComposeRuntimeError,
)


def document(destination="/run/credentials/adc.json"):
    return {
        "services": {
            "app": {
                "image": "example/app",
                "x-alk-secret-files": [
                    {
                        "alias": "ADC",
                        "destination": destination,
                        "environment_variable": "GOOGLE_APPLICATION_CREDENTIALS",
                    }
                ],
            }
        }
    }


def test_file_bindings_are_validated_before_docker(tmp_path):
    normalized = normalize_compose(document(), tmp_path, capabilities={})
    assert normalized["services"]["app"]["x-alk-secret-files"][0]["alias"] == "ADC"


@pytest.mark.parametrize(
    "destination",
    [
        "/proc/key",
        "/etc/ld.so.preload",
        "/var/run/docker.sock",
        "/../key",
        "/",
        "/run/key/../key",
    ],
)
def test_secret_destinations_cannot_control_runtime(tmp_path, destination):
    with pytest.raises((ComposeRuntimeError, ValueError)):
        normalize_compose(document(destination), tmp_path, capabilities={})


def test_secret_destination_cannot_overlap_existing_mount(tmp_path):
    value = document("/app/key")
    value["services"]["app"]["volumes"] = ["./:/app"]
    with pytest.raises(ComposeRuntimeError, match="overlap"):
        normalize_compose(value, tmp_path, capabilities={})


def test_materialized_files_are_private_readonly_and_not_interpolated(
    tmp_path, monkeypatch
):
    import asyncio
    import os
    from types import SimpleNamespace
    from fi.alk.harness.certified_compose_runtime import CertifiedComposeRuntimeProvider
    from fi.alk.harness.bundle_v2 import RuntimeKindV2

    source, bundle_dir, work = (
        tmp_path / name for name in ("source", "bundle", "work")
    )
    for path in (source, bundle_dir, work):
        path.mkdir()
    (bundle_dir / "compose.json").write_text(json.dumps(document()))
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(kind=RuntimeKindV2.COMPOSE, document="compose.json"),
        seed=None,
        digest="sha256:" + "f" * 64,
        capabilities={},
        readiness=[],
    )
    owner = CertifiedComposeRuntimeProvider(
        socket_path=tmp_path / "socket", configuration={"ADC": "fake-file-secret"}
    )
    monkeypatch.setattr(owner, "_check_socket", lambda: None)
    files = []

    def command(directory, project, *args, **kwargs):
        current = json.loads((directory / "compose.json").read_text())
        assert "fake-file-secret" not in (directory / "configuration.env").read_text()
        assert "fake-file-secret" not in json.dumps(current)
        if args[0] == "config":
            return json.dumps(current)
        if args[0] == "run":
            assert not current["services"]["app"]["volumes"]
            return f"{os.getuid()}\n{os.getgid()}\n"
        if args[0] == "up":
            app = current["services"]["app"]
            assert app["environment"] == {
                "GOOGLE_APPLICATION_CREDENTIALS": "/run/credentials/adc.json"
            }
            mount = app["volumes"][0]
            assert mount["read_only"] is True
            path = Path(mount["source"])
            assert not path.is_relative_to(directory / "source")
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.read_text() == "fake-file-secret"
            files.append(path)
        return ""

    monkeypatch.setattr(owner, "_command", command)

    async def run():
        worlds = await owner.provision(
            bundle, source=source, bundle_dir=bundle_dir, work_directory=work
        )
        await owner.reset(worlds[0], work_directory=work)
        await owner.close(work_directory=work)

    asyncio.run(run())
    assert len(files) == 2
    assert not list(work.iterdir())
    assert all(not path.exists() for path in files)


def test_real_nonroot_container_reads_credential_readonly(tmp_path):
    import asyncio
    import os
    import urllib.request
    from types import SimpleNamespace
    from fi.alk.harness.certified_compose_runtime import CertifiedComposeRuntimeProvider
    from fi.alk.harness.bundle_v2 import RuntimeKindV2, CapabilityV2, ReadinessProbeV2

    socket = os.environ.get("ALK_TEST_COMPOSE_SOCKET")
    if not socket:
        pytest.skip("requires private Docker certification socket")
    image = os.environ["ALK_TEST_COMPOSE_PYTHON_IMAGE"]
    assert "@sha256:" in image
    source, bundle_dir, work = (
        tmp_path / name for name in ("source", "bundle", "work")
    )
    for path in (source, bundle_dir, work):
        path.mkdir()
    (source / "Dockerfile").write_text(
        f'FROM {image}\nCOPY server.py /server.py\nUSER 65534\nCMD ["python", "/server.py"]\n'
    )
    (source / "server.py").write_text("""import os, pathlib, http.server
p = pathlib.Path(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
assert p.read_text() == 'fake-file-secret'
assert 'ADC' not in os.environ
assert p.stat().st_mode & 0o777 == 0o600
try:
    p.write_text('changed')
except OSError:
    pass
else:
    raise RuntimeError('credential writable')
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'verified')
http.server.HTTPServer(('0.0.0.0', 8000), Handler).serve_forever()
""")
    value = document()
    value["services"]["app"].pop("image")
    value["services"]["app"]["build"] = "."
    (bundle_dir / "compose.json").write_text(json.dumps(value))
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(kind=RuntimeKindV2.COMPOSE, document="compose.json"),
        seed=None,
        digest="sha256:" + "f" * 64,
        capabilities={
            "chat": CapabilityV2(protocol="http", service="app", container_port=8000)
        },
        readiness=[ReadinessProbeV2(capability="chat", path="/", timeout_seconds=30)],
    )
    owner = CertifiedComposeRuntimeProvider(
        socket_path=Path(socket),
        configuration={"ADC": "fake-file-secret"},
        artifact_root=tmp_path / "logs",
    )

    async def run():
        try:
            worlds = await owner.provision(
                bundle, source=source, bundle_dir=bundle_dir, work_directory=work
            )
            for _ in range(2):
                with urllib.request.urlopen(
                    worlds[0].endpoints["chat"].address, timeout=10
                ) as response:
                    assert response.read() == b"verified"
                await owner.reset(worlds[0], work_directory=work)
        finally:
            await owner.close(work_directory=work)
        assert not list(work.iterdir())
        assert all(
            "fake-file-secret" not in path.read_text()
            for path in (tmp_path / "logs").glob("*.log")
        )

    asyncio.run(run())


@pytest.mark.parametrize("purpose", ["simulator_provider", None])
def test_compiler_rejects_secret_binding_without_target_reference(tmp_path, purpose):
    from types import SimpleNamespace
    from fi.alk.harness.compose_integration import compile_bundle

    source = tmp_path / "source"
    source.mkdir()
    (source / "compose.yaml").write_text("services:\n  app:\n    image: example/app\n")
    refs = {"ADC": SimpleNamespace(purpose=purpose)} if purpose else {}
    job = SimpleNamespace(agent=SimpleNamespace(secret_refs=refs))
    contract = {
        "modality": "chat",
        "runtime": {
            "execution": "compose",
            "compose_file": "compose.yaml",
            "service": "app",
            "interface": {"kind": "http", "port": 8000},
            "secret_files": {
                "app": document()["services"]["app"]["x-alk-secret-files"]
            },
        },
    }
    with pytest.raises(ValueError, match="target_reference_required"):
        compile_bundle(
            source=source,
            job=job,
            authoring=tmp_path,
            output=tmp_path / "output",
            contract=contract,
        )
