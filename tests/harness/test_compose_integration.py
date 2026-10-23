import asyncio
import json
import random
from pathlib import Path

import pytest

from fi.alk.harness.bundle_author_v2 import author_bundle_v2
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.process_preflight import preflight_bundle, PreflightError
from fi.alk.harness.world.observation import ObservationWorld
from fi.alk.harness.world.errors import WorldUsageError


@pytest.mark.parametrize("included", [False, True])
@pytest.mark.parametrize("additional", [False, True])
@pytest.mark.parametrize("image_selected", [False, True])
def test_native_compiler_keeps_dependency_paths_and_requires_optin(
    tmp_path, monkeypatch, included, additional, image_selected, resolved_compose_images
):
    source = tmp_path / "source"
    (source / "deploy").mkdir(parents=True)
    (source / "Dockerfile").write_text("FROM node:22\n")
    (source / "config").mkdir()
    (source / "deploy" / "compose.yaml").write_text("""services:
  app:
    image: example/chat:1
    build:
      context: ..
      args: {LITERAL: "$$HOME"}
    profiles: [web]
    ports: ["3000:3000"]
    depends_on: [mongo]
    environment: ["OPTIONAL_URL=http://unrelated:8000"]
    volumes: ["../config:/app/config"]
  mongo:
    image: mongo:7
  unrelated:
    image: nginx:alpine
    profiles: [extra]
""")
    if image_selected:
        (source / "unused-test.py").write_text("TOKEN = 'ghp_" + "x" * 36 + "'\n")
    if included:
        (source / "deploy" / "parts").mkdir()
        (source / "deploy" / "parts" / "db.yaml").write_text(
            "services:\n  mongo:\n    image: mongo:7\n    volumes: [dbdata:/data/db]\nvolumes:\n  dbdata: {}\n"
        )
        compose_file = source / "deploy" / "compose.yaml"
        compose_file.write_text(
            "include: [parts/db.yaml]\n"
            + compose_file.read_text().replace("  mongo:\n    image: mongo:7\n", "")
        )
    authoring = tmp_path / "authoring"
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
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "chat",
                "runtime": {
                    "observations": {"history": {"capability": "target_http", "path": "/history", "body": {"operation": "read"}}},
                    "language": "javascript",
                    "execution": "compose",
                    "compose_file": "deploy/compose.yaml",
                    "workdir": ".",
                    "service": "app",
                    "additional_services": ["unrelated"] if additional else [],
                    "compose_image_services": ["app"] if image_selected else [],
                    "command": ["node", "server.js"],
                    "interface": {
                        "kind": "http",
                        "protocol": "json_template",
                        "request_template": {"message": "{{new_message_content}}"},
                        "port": 3000,
                        "path": "/chat",
                    },
                },
            }
        )
    )
    job = HarnessJob.model_validate(
        {
            "job_id": "compose",
            "run_id": "compose",
            "execution": "local",
            "source": {"kind": "local_repository", "local_path": str(source)},
            "agent": {"connector": "auto"},
            "scenario_count": 1,
            "runtime": {"cpu_units": 2, "memory_mb": 4096, "parallelism": 1},
        }
    )
    monkeypatch.setenv("ALK_TRUSTED_COMPOSE_SOCKET", "/private/alk/docker.sock")
    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=output
    )
    assert len(resolved_compose_images) == 1
    assert (output / "image-lock.json").is_file()
    assert "image-lock.json" in manifest.provenance.generated_files
    assert manifest.seed is None
    assert manifest.runtime.kind.value == "compose"
    document = json.loads((output / "compose.json").read_text())
    assert set(document["services"]) == (
        {"app", "mongo", "unrelated"} if additional else {"app", "mongo"}
    )
    assert manifest.runtime.control_service == "app"
    if additional:
        assert "profiles" not in document["services"]["unrelated"]
    if image_selected:
        assert "build" not in document["services"]["app"]
        assert document["services"]["app"]["image"].startswith("docker.io/example/chat@sha256:")
    else:
        assert document["services"]["app"]["build"]["context"] == "./."
        assert document["services"]["app"]["build"]["args"] == {"LITERAL": "$$HOME"}
    assert document["services"]["app"]["volumes"] == ["./config:/app/config"]
    assert "profiles" not in document["services"]["app"]
    if included:
        assert document["volumes"] == {"dbdata": {}}
    with pytest.raises(PreflightError, match="compose_not_hosted"):
        preflight_bundle(output, manifest, parallelism=1, secret_refs={})
    preflight_bundle(
        output, manifest, parallelism=1, secret_refs={}, allow_compose=True
    )
    from types import SimpleNamespace
    from fi.alk.harness.world.observation import ObservationWorldFactory

    factory = ObservationWorldFactory(output)
    (authoring / "contract.json").write_text("{}")
    world = asyncio.run(factory.create(SimpleNamespace(world_index=0, endpoints={
        "target_http": SimpleNamespace(address="http://127.0.0.1:12345", protocol="http")
    }), rng=random.Random(0)))
    assert world._execution_state()["observations"]["history"]["body"] == {"operation": "read"}
    sealed_contract = output / "contract.json"
    original = sealed_contract.read_text()
    sealed_contract.write_text(original.replace('"read"', '"delete"'))
    with pytest.raises(PreflightError):
        preflight_bundle(output, manifest, parallelism=1, secret_refs={}, allow_compose=True)
    sealed_contract.write_text(original)
    (output / "compose.json").write_text("{}")
    with pytest.raises(PreflightError):
        preflight_bundle(
            output, manifest, parallelism=1, secret_refs={}, allow_compose=True
        )


def test_observation_world_never_fabricates_empty_state():
    world = ObservationWorld(0, random.Random(1))
    for operation in (
        "state",
        "query",
        "put",
        "change",
        "drop",
        "checkpoint",
        "handle_tool_call",
    ):
        with pytest.raises(WorldUsageError, match="native_observation"):
            getattr(world.read_only(), operation)()
    assert world._execution_state()["kind"] == "native_observation"


def test_native_evidence_records_unobserved_state_as_not_applicable(tmp_path):
    from types import SimpleNamespace
    from fi.alk.harness.authoring_runtime_validation import _write_runtime_evidence
    from fi.alk.harness.action_certification import certify_actions
    from fi.alk.harness.tool_certification import ToolCertificationReport
    from fi.alk.harness.certification import CheckStatus, GenericHarnessArtifactStore

    digest = "sha256:" + "1" * 64
    (tmp_path / "contract.json").write_text("{}")
    (tmp_path / "scenarios").mkdir()
    _write_runtime_evidence(
        job=SimpleNamespace(metadata={}),
        authoring=tmp_path,
        manifest=SimpleNamespace(
            provenance=SimpleNamespace(source_digest=digest), digest=digest
        ),
        count=1,
        external_provider=False,
        native_observation=True,
        tool_report=ToolCertificationReport.create([]),
        action_report=certify_actions([], adapter=None),
        reset_equivalence=CheckStatus.NOT_APPLICABLE,
        world_isolation=CheckStatus.NOT_APPLICABLE,
        chat_ingress=CheckStatus.PASSED,
    )
    evidence = GenericHarnessArtifactStore(
        tmp_path / "generic-harness"
    ).read_runtime_evidence()
    assert evidence.checks.schema_and_seed is CheckStatus.NOT_APPLICABLE
    assert evidence.checks.source_invariants is CheckStatus.NOT_APPLICABLE
    assert evidence.checks.reset_equivalence is CheckStatus.NOT_APPLICABLE
    assert evidence.checks.chat_ingress is CheckStatus.PASSED
    assert any("opaque" in item for item in evidence.limitations)


def test_observation_phase_runs_in_worker_and_rejects_state_reads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from fi.alk.harness.hosted_scheduler import _run_phase
    from fi.alk.harness import process_runtime

    original = process_runtime._allowlisted_ambient_env
    monkeypatch.setattr(
        process_runtime,
        "_allowlisted_ambient_env",
        lambda env: {
            **original(env),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    world = ObservationWorld(0, random.Random(1))

    def ready(world):
        raise AssertionError("must run isolated source")

    ready._alk_entry = "ready"
    ready._alk_source = "def ready(world):\n    return world.state() == {}\n"
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = asyncio.run(
            _run_phase(ready, world, timeout=10, phase="ready", executor=executor)
        )
    assert result.failure is not None
    assert "WorldUsageError" in str(result.failure)


def test_provider_native_runtime_records_completed_tools_without_sql(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from fi.alk.harness.certified_compose_runtime import CertifiedComposeRuntimeProvider
    from fi.alk.harness.bundle_v2 import RuntimeKindV2, CapabilityV2
    from fi.alk.harness.chat_call_runner import _tool_world, _HostedChatTarget
    from fi.alk.harness.contract import AgentContract

    source, bundle_dir, work = (
        tmp_path / name for name in ("source", "bundle", "work")
    )
    source.mkdir()
    bundle_dir.mkdir()
    work.mkdir()
    (bundle_dir / "compose.json").write_text(
        json.dumps({"services": {"chat": {"image": "example/chat:1"}}})
    )
    from fi.alk.harness.bundle_v2 import BundleProvenanceV2
    from fi.alk.harness.provision import source_fingerprint

    bundle = SimpleNamespace(
        provenance=BundleProvenanceV2(source_kind="local", source_digest=source_fingerprint(source)),
        runtime=SimpleNamespace(kind=RuntimeKindV2.COMPOSE, document="compose.json"),
        seed=None,
        digest="sha256:" + "a" * 64,
        readiness=[],
        capabilities={
            "target_http": CapabilityV2(
                protocol="http", service="chat", container_port=8000
            )
        },
    )
    provider = CertifiedComposeRuntimeProvider(socket_path=tmp_path / "socket")
    monkeypatch.setattr(provider, "_check_socket", lambda: None)
    monkeypatch.setattr(
        provider,
        "_command",
        lambda directory, project, *args, **kwargs: (
            json.dumps({"services": {}})
            if args[0] == "config"
            else "127.0.0.1:18000\n"
            if args[0] == "port"
            else ""
        ),
    )
    contract = AgentContract.model_validate(
        {"tools": [{"name": "lookup", "args": ["key"]}]}
    )

    class NativeAgent:
        async def call(self, request):
            return SimpleNamespace(
                content="Stored value is 42.",
                metadata={},
                tool_calls=[
                    {"id": "call-1", "name": "lookup", "arguments": {"key": "answer"}}
                ],
                tool_responses=[
                    {"tool_call_id": "call-1", "result": {"value": 42}, "ok": True}
                ],
            )

    async def run():
        try:
            runtimes = await provider.provision(
                bundle, source=source, bundle_dir=bundle_dir, work_directory=work
            )
            (runtime,) = runtimes
            assert "world_db" not in runtime.endpoints
            world = _tool_world(bundle_dir, contract, runtime, source)
            assert isinstance(world, ObservationWorld)
            target = _HostedChatTarget(
                wrapper=NativeAgent(),
                contract=contract,
                world=world,
                scenario_key="lookup",
                scenario_id="lookup-1",
            )
            assert await target.say("Read answer") == "Stored value is 42."
            (call,) = world.calls
            assert (call.name, call.arguments, call.result, call.ok) == (
                "lookup",
                {"key": "answer"},
                {"value": 42},
                True,
            )
        finally:
            await provider.close(work_directory=work)

    asyncio.run(run())


@pytest.mark.parametrize("definition", ["base", "include", "overlay", "overlay_alias"])
def test_source_compose_definitions_keep_content_admission(
    tmp_path, monkeypatch, definition, resolved_compose_images
):
    from fi.alk.harness.bundle import BundleError
    from fi.alk.harness.compose_integration import compile_bundle

    source = tmp_path / "source"
    source.mkdir()
    base = source / "compose.yaml"
    base.write_text("services:\n  app:\n    image: example/chat:1\n    ports: ['3000:3000']\n")
    runtime = {
        "execution": "compose", "compose_file": "compose.yaml", "service": "app",
        "interface": {"kind": "http", "protocol": "json_template", "port": 3000,
                      "path": "/chat", "request_template": {"message": "{{new_message_content}}"}},
    }
    if definition == "include":
        base.write_text("include: [part.yaml]\n" + base.read_text())
        selected = source / "part.yaml"
        selected.write_text("services: {}\n")
    elif definition in {"overlay", "overlay_alias"}:
        runtime["compose_overrides"] = ["overlay.yaml"]
        selected = source / "overlay.yaml"
        selected.write_text("services: {}\n")
        if definition == "overlay_alias":
            target = source / "real-overlay.yaml"
            selected.rename(target)
            selected.symlink_to(target.name)
            selected = target
    else:
        selected = base
    selected.write_text(selected.read_text() + "# Public test token: ghp_" + "x" * 36 + "\n")
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    contract = {"modality": "chat", "runtime": runtime}
    (authoring / "contract.json").write_text(json.dumps(contract))
    job = HarnessJob.model_validate({
        "job_id": "compose", "run_id": "compose", "execution": "local",
        "source": {"kind": "local_repository", "local_path": str(source)},
        "agent": {"connector": "auto"},
    })
    monkeypatch.setenv("ALK_TRUSTED_COMPOSE_SOCKET", "/private/alk/docker.sock")
    with pytest.raises(BundleError, match="bundle_secret_material_detected"):
        compile_bundle(source=source, job=job, authoring=authoring,
                       output=tmp_path / "bundle", contract=contract, include_scenarios=False)


def test_compose_bundle_seals_declared_source_exclusions(
    tmp_path, monkeypatch, resolved_compose_images
):
    from fi.alk.harness.compose_integration import compile_bundle

    source = tmp_path / "source"
    source.mkdir()
    (source / "compose.yaml").write_text(
        "services:\n  app:\n    image: example/chat:1\n    ports: ['3000:3000']\n"
    )
    (source / "tests").mkdir()
    (source / "tests/fixture.ts").write_text("const token = 'ghp_" + "x" * 36 + "';\n")
    runtime = {
        "execution": "compose",
        "compose_file": "compose.yaml",
        "service": "app",
        "context_excludes": ["tests/**"],
        "interface": {
            "kind": "http",
            "protocol": "json_template",
            "port": 3000,
            "path": "/chat",
            "request_template": {"message": "{{new_message_content}}"},
        },
    }
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    contract = {"modality": "chat", "runtime": runtime}
    (authoring / "contract.json").write_text(json.dumps(contract))
    job = HarnessJob.model_validate(
        {
            "job_id": "compose",
            "run_id": "compose",
            "execution": "local",
            "source": {"kind": "local_repository", "local_path": str(source)},
            "agent": {"connector": "auto"},
        }
    )
    monkeypatch.setenv("ALK_TRUSTED_COMPOSE_SOCKET", "/private/alk/docker.sock")

    manifest = compile_bundle(
        source=source,
        job=job,
        authoring=authoring,
        output=tmp_path / "bundle",
        contract=contract,
        include_scenarios=False,
    )

    assert manifest.metadata["context_excludes"] == ["tests/**"]
