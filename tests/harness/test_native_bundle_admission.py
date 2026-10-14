import asyncio
import json
from types import SimpleNamespace

import pytest

from fi.alk.harness import cli
from fi.alk.harness.bundle import BundleError
from fi.alk.harness.bundle_author_v2 import author_bundle_v2
from fi.alk.harness.contract import AgentContract
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.process_preflight import preflight_bundle
from fi.alk.harness.world.snapshot import save
from fi.alk.harness.world.tools import world_tools


@pytest.fixture
def native_build(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Dockerfile").write_text("FROM python:3.12\n")
    (source / "compose.yaml").write_text(
        'services:\n  app:\n    build: .\n    ports: ["8080:8080"]\n'
    )
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    contract = AgentContract(
        agent="native",
        modality="chat",
        tools=[],
        runtime={
            "language": "python",
            "execution": "compose",
            "compose_file": "compose.yaml",
            "service": "app",
            "interface": {
                "kind": "http",
                "port": 8080,
                "path": "/chat",
                "protocol": "json_template",
                "request_template": {"message": "{{new_message_content}}"},
                "response_path": "answer",
            },
        },
    )
    (authoring / "contract.json").write_text(contract.model_dump_json())
    _, world = world_tools(
        contract, authoring, native_runtime=True, deferred_runtime=True
    )
    save(world, authoring)
    world.close()
    monkeypatch.setenv("ALK_COMPOSE_BROKER_SOCKET", "/private/alk/broker.sock")
    monkeypatch.setattr(
        cli, "build_stage", lambda *args, **kwargs: (SimpleNamespace(spent_usd=0), None)
    )

    async def converse(*args, **kwargs):
        pass

    monkeypatch.setattr(cli, "_converse", converse)
    monkeypatch.setattr(cli, "_guidance", lambda args: "")
    args = SimpleNamespace(
        out=str(authoring),
        path=str(source),
        name="native",
        external_runtime=False,
        skip_source_provision=True,
        interactive=False,
    )
    job = HarnessJob.model_validate(
        {
            "job_id": "native",
            "run_id": "native",
            "execution": "local",
            "source": {"kind": "local_repository", "local_path": str(source)},
            "agent": {"connector": "auto"},
            "scenario_count": 1,
            "runtime": {"cpu_units": 2, "memory_mb": 4096, "parallelism": 1},
        }
    )
    return source, authoring, args, job


def compile_native(source, authoring, job, output):
    scenario = authoring / "scenarios" / "one"
    (scenario / "checks").mkdir(parents=True)
    (scenario / "scenario.json").write_text(
        json.dumps({"name": "one", "instruction": "hello", "sub_goals": ["response"]})
    )
    for name in ("setup", "ready"):
        (scenario / f"{name}.py").write_text(f"def {name}(world):\n    return None\n")
    (scenario / "checks" / "response.py").write_text(
        "def check(world, calls):\n    return None\n"
    )
    return author_bundle_v2(source=source, authoring=authoring, job=job, output=output)


def test_native_build_then_v2_seal_preserves_contained_dangling_link(
    native_build, tmp_path
):
    source, authoring, args, job = native_build
    link = source / "testdata"
    link.symlink_to("missing-testdata")
    assert asyncio.run(cli._build(args)) == 0
    assert not (authoring / "environment-bundle").exists()
    output = tmp_path / "bundle"
    manifest = compile_native(source, authoring, job, output)
    assert manifest.runtime.kind.value == "compose"
    preflight_bundle(
        output, manifest, parallelism=1, secret_refs={}, allow_compose=True
    )
    assert link.is_symlink() and str(link.readlink()) == "missing-testdata"
    assert not (output / "services" / "source").exists()


@pytest.mark.parametrize("stage", ["build", "compile"])
def test_native_source_secret_is_rejected_before_runtime(native_build, tmp_path, stage):
    source, authoring, args, job = native_build
    (source / "example.txt").write_text(
        "-----BEGIN PRIVATE KEY-----\nactual-private-material\n-----END PRIVATE KEY-----\n"
    )
    if stage == "build":
        assert asyncio.run(cli._build(args)) == 1
    else:
        with pytest.raises(BundleError, match="secret_material_detected"):
            compile_native(source, authoring, job, tmp_path / "bundle")


def test_native_source_escape_rejected_without_reading_target(native_build, tmp_path):
    source, authoring, args, job = native_build
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (source / "escape").symlink_to("../outside.txt")
    assert asyncio.run(cli._build(args)) == 1
    assert outside.read_text() == "outside"
