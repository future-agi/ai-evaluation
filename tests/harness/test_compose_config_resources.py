import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fi.alk.harness.certified_compose_runtime import (
    CertifiedComposeRuntimeProvider,
    normalize_compose,
    ComposeRuntimeError,
)
from fi.alk.harness.bundle_v2 import RuntimeKindV2
from fi.alk.harness.contract import Runtime, RuntimeConfigFile


def document():
    return {
        "x-alk-config-files": {
            ".env": {
                "values": {"PORT": "3000"},
                "secret_aliases": {"PASSWORD": "APP_KEY"},
            }
        },
        "x-alk-directories": ["data"],
        "services": {
            "app": {
                "image": "example/app",
                "env_file": ["./.env"],
                "volumes": ["./.env:/app/.env", "./data:/app/data"],
            }
        },
    }


def test_missing_resources_normalize_without_writing_original(tmp_path):
    normalized = normalize_compose(document(), tmp_path, capabilities={})
    assert normalized["services"]["app"]["volumes"][0]["read_only"] is True
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "path", ["../.env", "/.env", "app.py", "Dockerfile", ".git/a.env"]
)
def test_resource_paths_cannot_replace_code(path):
    with pytest.raises(ValueError):
        Runtime(config_files={path: RuntimeConfigFile()})


def test_resource_symlink_is_rejected(tmp_path):
    (tmp_path / "actual").write_text("unchanged")
    (tmp_path / ".env").symlink_to("actual")
    with pytest.raises(ComposeRuntimeError, match="symlink"):
        normalize_compose(document(), tmp_path, capabilities={})


def test_config_and_directories_are_private_reset_and_never_build_inputs(
    tmp_path, monkeypatch
):
    source, bundle_dir, work = [
        tmp_path / name for name in ("source", "bundle", "work")
    ]
    for path in (source, bundle_dir, work):
        path.mkdir()
    (source / "Dockerfile").write_text("FROM example/app\nCOPY . /app\n")
    (source / ".env").write_text("PORT=1234\n")
    doc = document()
    doc["services"]["app"]["build"] = "."
    (bundle_dir / "compose.json").write_text(json.dumps(doc))
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(kind=RuntimeKindV2.COMPOSE, document="compose.json"),
        seed=None,
        digest="sha256:" + "f" * 64,
        capabilities={},
        readiness=[],
    )
    owner = CertifiedComposeRuntimeProvider(
        socket_path=tmp_path / "socket", configuration={"APP_KEY": "private$value"}
    )
    monkeypatch.setattr(owner, "_check_socket", lambda: None)
    snapshots = []

    def command(directory, project, *args, **kwargs):
        current = json.loads((directory / "compose.json").read_text())
        assert all(
            b"private$value" not in p.read_bytes()
            for p in (directory / "source").rglob("*")
            if p.is_file()
        )
        app = current["services"]["app"]
        if args[0] == "config":
            return json.dumps(current)
        if args[0] == "run":
            assert app["environment"] == {}
            assert app["env_file"] == []
            assert len(app["volumes"]) == 1
            return f"{os.getuid()}\n{os.getgid()}\n"
        if args[0] == "up":
            mounts = {item["target"]: item for item in app["volumes"]}
            config = Path(mounts["/app/.env"]["source"])
            data = Path(mounts["/app/data"]["source"])
            assert not config.is_relative_to(directory / "source")
            assert config.stat().st_mode & 0o777 == 0o600
            assert config.read_text() == 'PASSWORD="private$value"\nPORT="3000"\n'
            assert mounts["/app/.env"]["read_only"] is True
            assert not (data / "state").exists()
            (data / "state").write_text("changed")
            snapshots.append(config)
        return ""

    monkeypatch.setattr(owner, "_command", command)

    async def run():
        worlds = await owner.provision(
            bundle, source=source, bundle_dir=bundle_dir, work_directory=work
        )
        await owner.reset(worlds[0], work_directory=work)
        await owner.close(work_directory=work)

    asyncio.run(run())
    assert len(snapshots) == 2
    assert (source / ".env").read_text() == "PORT=1234\n"
    assert not (source / "data").exists()
    assert not list(work.iterdir())


def test_missing_alias_fails_before_docker(tmp_path, monkeypatch):
    source, bundle_dir, work = [
        tmp_path / name for name in ("source", "bundle", "work")
    ]
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
        socket_path=tmp_path / "socket", configuration={}
    )
    monkeypatch.setattr(owner, "_check_socket", lambda: None)
    monkeypatch.setattr(
        owner, "_command", lambda *args, **kwargs: pytest.fail("Docker must not start")
    )
    with pytest.raises(ComposeRuntimeError, match="secret_missing"):
        asyncio.run(
            owner.provision(
                bundle, source=source, bundle_dir=bundle_dir, work_directory=work
            )
        )


@pytest.mark.parametrize("purpose", [None, "simulator_provider", "source_checkout"])
def test_compiler_config_alias_requires_target_purpose(tmp_path, purpose):
    from fi.alk.harness.compose_integration import compile_bundle

    refs = {} if purpose is None else {"KEY": SimpleNamespace(purpose=purpose)}
    job = SimpleNamespace(agent=SimpleNamespace(secret_refs=refs))
    contract = {
        "runtime": {"config_files": {".env": {"secret_aliases": {"PASSWORD": "KEY"}}}}
    }
    with pytest.raises(ValueError, match="target_reference_required"):
        compile_bundle(
            source=tmp_path,
            job=job,
            authoring=tmp_path,
            output=tmp_path / "output",
            contract=contract,
        )


@pytest.mark.parametrize("public", [False, True])
def test_bound_dotenv_interpolation_rejected_before_docker(
    tmp_path, monkeypatch, public
):
    source, bundle_dir, work = [
        tmp_path / name for name in ("source", "bundle", "work")
    ]
    for path in (source, bundle_dir, work):
        path.mkdir()
    doc = document()
    if public:
        doc["x-alk-config-files"][".env"]["values"]["VARIABLE"] = "${HOME}"
    (bundle_dir / "compose.json").write_text(json.dumps(doc))
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(kind=RuntimeKindV2.COMPOSE, document="compose.json"),
        seed=None,
        digest="sha256:" + "f" * 64,
        capabilities={},
        readiness=[],
    )
    owner = CertifiedComposeRuntimeProvider(
        socket_path=tmp_path / "socket",
        configuration={"APP_KEY": "safe" if public else "${HOME}"},
    )
    monkeypatch.setattr(owner, "_check_socket", lambda: None)
    monkeypatch.setattr(
        owner, "_command", lambda *args, **kwargs: pytest.fail("Docker must not start")
    )
    with pytest.raises(
        ComposeRuntimeError, match="bound_dotenv_interpolation_forbidden"
    ):
        asyncio.run(
            owner.provision(
                bundle, source=source, bundle_dir=bundle_dir, work_directory=work
            )
        )


def test_serialized_secret_values_are_redacted_from_actual_log_artifact(
    tmp_path, monkeypatch
):
    import subprocess

    value = 'a"b\\c$d'
    serialized = [
        value,
        json.dumps(value, ensure_ascii=False)[1:-1],
        json.dumps(value.replace("$", "$$"), ensure_ascii=False)[1:-1],
    ]

    def popen(command, **kwargs):
        kwargs["stdout"].write("\n".join(serialized).encode())
        return SimpleNamespace(wait=lambda **kwargs: 0, returncode=0)

    monkeypatch.setattr(subprocess, "Popen", popen)
    owner = CertifiedComposeRuntimeProvider(
        socket_path=tmp_path / "socket",
        configuration={"KEY": value},
        artifact_root=tmp_path / "logs",
    )
    owner._command(tmp_path, "project", "logs")
    log = (tmp_path / "logs/project.log").read_text()
    assert all(secret not in log for secret in serialized)
    assert log.count("[REDACTED]") == 3


def test_directory_ownership_includes_children_without_following_symlinks(
    tmp_path, monkeypatch
):
    from fi.alk.harness.certified_compose_runtime import _own_directory

    root = tmp_path / "data"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "file").write_text("fixture")
    external = tmp_path / "external"
    external.mkdir()
    (external / "file").write_text("external")
    (root / "link").symlink_to(external, target_is_directory=True)
    changed = []
    fchanged = []
    original = os.chown

    def chown(path, uid, gid, **kwargs):
        assert kwargs["follow_symlinks"] is False
        changed.append(path)
        original(path, os.getuid(), os.getgid(), **kwargs)

    monkeypatch.setattr(os, "chown", chown)
    monkeypatch.setattr(
        os,
        "fchown",
        lambda descriptor, uid, gid: fchanged.append(os.fstat(descriptor).st_ino),
    )
    _own_directory(root, os.getuid(), os.getgid())
    assert sorted(changed) == ["file", "nested"]
    assert set(fchanged) == {root.stat().st_ino, nested.stat().st_ino}
    assert external.stat().st_ino not in fchanged
