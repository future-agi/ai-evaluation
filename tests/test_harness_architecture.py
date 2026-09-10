from __future__ import annotations

from pathlib import Path
import asyncio
import json
from types import SimpleNamespace

import pytest

from fi.alk.harness.bundle import (
    BundleError,
    BundleProvenance,
    BundleRuntime,
    RuntimeKind,
    export_session_bundle,
    load_bundle,
    seal_bundle,
)
from fi.alk.harness.events import BufferedEventSink, EventOutbox
from fi.alk.harness.environment_plan import (
    ENVIRONMENT_PLAN_FILE,
    EnvironmentPlanError,
    load_environment_plan,
)
from fi.alk.harness.executor import (
    GitHubSourceAcquirer,
    HarnessExecutor,
    _failure_from_events,
)
from fi.alk.harness.job import FailureDomain, HarnessFailure, HarnessJob, HarnessStage
from fi.alk.harness.provision import source_fingerprint
from fi.simulate.runtime.spec import RuntimeIsolation
from fi.simulate.runtime.events import CanonicalEvent


def _manifest() -> dict:
    return {
        "name": "ride-environment",
        "digest": "sha256:" + "0" * 64,
        "runtime": BundleRuntime(kind=RuntimeKind.COMPOSE, document="compose.yaml"),
        "services": ["tools-api", "postgres"],
        "capabilities": {
            "ride_tools": {
                "protocol": "http",
                "service": "tools-api",
                "container_port": 8080,
                "configuration_name": "TOOLS_API_URL",
            }
        },
        "readiness": [
            {"capability": "ride_tools", "path": "/health", "timeout_seconds": 60}
        ],
        "provenance": BundleProvenance(
            source_kind="repository", source_digest="a" * 64
        ),
    }


def test_executor_understands_a_materialized_archive_as_a_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, str] = {}

    async def fake_auto(args) -> int:
        observed["kind"] = args.kind
        observed["path"] = args.path
        return 0

    monkeypatch.setattr("fi.alk.harness.cli._auto", fake_auto)
    source = tmp_path / "uploaded-agent"
    source.mkdir()
    output = tmp_path / "artifacts"
    job = HarnessJob(
        job_id="job-upload",
        run_id="run-upload",
        execution="hosted",
        source={"kind": "archive", "archive_artifact_id": "source-id"},
        agent={"connector": "auto"},
        metadata={"source_kind": "archive"},
    )

    status = asyncio.run(HarnessExecutor().run(job, source=source, output=output))

    assert status.stage.value == "completed"
    assert observed == {"kind": "repo", "path": str(source.resolve())}


def test_executor_freezes_the_resolved_github_commit_before_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "a" * 40
    observed: dict[str, str | None] = {}

    async def fake_auto(args) -> int:
        observed["commit"] = args.job.source.commit_sha
        return 0

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=commit + "\n", stderr="")

    monkeypatch.setattr("fi.alk.harness.cli._auto", fake_auto)
    monkeypatch.setattr("fi.alk.harness.executor.subprocess.run", fake_run)
    source = tmp_path / "repository"
    source.mkdir()
    job = HarnessJob(
        job_id="job-github",
        run_id="run-github",
        execution="hosted",
        source={
            "kind": "github",
            "visibility": "public",
            "repository": "customer/agent",
            "ref": "main",
        },
        agent={"connector": "auto"},
    )

    status = asyncio.run(
        HarnessExecutor().run(job, source=source, output=tmp_path / "artifacts")
    )

    assert status.stage.value == "completed"
    assert observed["commit"] == commit


def test_autonomous_pipeline_cleans_environment_when_a_stage_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fi.alk.harness import cli

    source = tmp_path / "agent"
    source.mkdir()
    output = tmp_path / "artifacts"
    cleaned = []

    async def failed_understanding(_args) -> int:
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(cli, "_understand", failed_understanding)
    monkeypatch.setattr(
        "fi.alk.harness.provision.stop", lambda destination: cleaned.append(destination)
    )

    with pytest.raises(RuntimeError, match="stage exploded"):
        asyncio.run(
            cli._auto(
                SimpleNamespace(
                    path=str(source),
                    name="agent",
                    kind="repo",
                    out=str(output),
                    count=1,
                    model=None,
                    run_model=None,
                )
            )
        )

    assert cleaned == [output.resolve()]


def test_hosted_authoring_uses_original_stages_without_running_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fi.alk.harness import cli

    source = tmp_path / "agent"
    source.mkdir()
    output = tmp_path / "authoring"
    observed: list[tuple[str, bool | None]] = []

    async def understand(_args) -> int:
        observed.append(("understand", None))
        return 0

    async def build(args) -> int:
        observed.append(("environment", args.skip_source_provision))
        return 0

    async def scenarios(_args) -> int:
        observed.append(("scenarios", None))
        return 0

    async def calls(_args) -> int:
        observed.append(("calls", None))
        return 0

    monkeypatch.setattr(cli, "_understand", understand)
    monkeypatch.setattr(cli, "_build", build)
    monkeypatch.setattr(cli, "_scenarios", scenarios)
    monkeypatch.setattr(cli, "_simulate", calls)
    monkeypatch.setattr(cli, "load_written", lambda _destination: [object(), object()])
    monkeypatch.setattr("fi.alk.harness.provision.stop", lambda _destination: False)

    status = asyncio.run(
        cli._auto(
            SimpleNamespace(
                path=str(source),
                name="agent",
                kind="repo",
                out=str(output),
                count=2,
                model=None,
                run_model=None,
                authoring_only=True,
                adjustments_path=None,
            )
        )
    )

    assert status == 0
    assert observed == [
        ("understand", None),
        ("environment", True),
        ("scenarios", None),
    ]


def test_hosted_authoring_repairs_partial_scenario_suite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fi.alk.harness import cli

    source = tmp_path / "agent"
    source.mkdir()
    output = tmp_path / "authoring"
    counts = iter((2, 3))
    observed_guidance: list[list[str]] = []

    async def ok(_args) -> int:
        return 0

    async def scenarios(args) -> int:
        observed_guidance.append(list(args.guidance))
        return 0

    monkeypatch.setattr(cli, "_understand", ok)
    monkeypatch.setattr(cli, "_build", ok)
    monkeypatch.setattr(cli, "_scenarios", scenarios)
    monkeypatch.setattr(
        cli,
        "load_written",
        lambda _destination: [object()] * next(counts),
    )
    monkeypatch.setattr("fi.alk.harness.provision.stop", lambda _destination: False)

    status = asyncio.run(
        cli._auto(
            SimpleNamespace(
                path=str(source),
                name="agent",
                kind="repo",
                out=str(output),
                count=3,
                model=None,
                run_model=None,
                authoring_only=True,
                adjustments_path=None,
            )
        )
    )

    assert status == 0
    assert len(observed_guidance) == 2
    assert observed_guidance[0] == []
    assert "only 2 are currently saved" in observed_guidance[1][0]
    assert "Add exactly 1" in observed_guidance[1][0]


def test_hosted_authoring_repairs_scenario_adjustment_without_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fi.alk.harness import cli

    source = tmp_path / "agent"
    source.mkdir()
    output = tmp_path / "authoring"
    adjustments = tmp_path / "adjustments.jsonl"
    adjustment_id = "adjustment-discount"
    adjustments.write_text(
        json.dumps(
            {
                "adjustment_id": adjustment_id,
                "instruction": "Add a scenario where the caller declines a high price.",
                "target_stage": "scenarios",
                "scenario_delta": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    written: list[SimpleNamespace] = []
    observed_guidance: list[list[str]] = []

    async def ok(_args) -> int:
        return 0

    async def scenarios(args) -> int:
        observed_guidance.append(list(args.guidance))
        if len(observed_guidance) == 1:
            written[:] = [
                SimpleNamespace(fixture={}),
                SimpleNamespace(fixture={}),
            ]
        else:
            written[:] = [
                SimpleNamespace(fixture={}),
                SimpleNamespace(fixture={"adjustment_ids": [adjustment_id]}),
            ]
        return 0

    monkeypatch.setattr(cli, "_understand", ok)
    monkeypatch.setattr(cli, "_build", ok)
    monkeypatch.setattr(cli, "_scenarios", scenarios)
    monkeypatch.setattr(cli, "load_written", lambda _destination: list(written))
    monkeypatch.setattr("fi.alk.harness.provision.stop", lambda _destination: False)

    status = asyncio.run(
        cli._auto(
            SimpleNamespace(
                path=str(source),
                name="agent",
                kind="repo",
                out=str(output),
                count=1,
                model=None,
                run_model=None,
                authoring_only=True,
                adjustments_path=str(adjustments),
            )
        )
    )

    assert status == 0
    assert len(observed_guidance) == 2
    assert adjustment_id in observed_guidance[0][0]
    assert "No saved scenario currently carries this marker" in observed_guidance[1][0]
    statuses = [
        json.loads(line)
        for line in (tmp_path / "adjustment-status.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert statuses[-1]["status"] == "applied"


def test_hosted_authoring_treats_an_extracted_archive_as_a_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fi.alk.harness import authoring_entrypoint

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    job = HarnessJob(
        job_id="job-author",
        run_id="run-author",
        execution="hosted",
        source={"kind": "archive", "archive_artifact_id": "source-id"},
        agent={"connector": "auto"},
        scenario_count=2,
        runtime={"isolation": "dedicated_vm"},
        metadata={"source_kind": "archive", "agent_name": "uploaded-agent"},
    )
    job_path = tmp_path / "job.json"
    job_path.write_text(job.model_dump_json(), encoding="utf-8")
    adjustments_path = tmp_path / "adjustments.jsonl"
    observed = {}

    async def fake_auto(args) -> int:
        observed.update(vars(args))
        return 0

    monkeypatch.setattr(authoring_entrypoint, "_auto", fake_auto)

    status = authoring_entrypoint.main(
        [
            str(job_path),
            "--source",
            str(source),
            "--output",
            str(output),
            "--adjustments",
            str(adjustments_path),
        ]
    )

    assert status == 0
    assert observed["kind"] == "repo"
    assert observed["authoring_only"] is True
    assert observed["count"] == 2
    assert observed["adjustments_path"] == str(adjustments_path)


def test_hosted_authoring_persists_adjusted_scenario_count_for_bundling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fi.alk.harness import authoring_entrypoint

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    job = HarnessJob(
        job_id="job-adjusted-count",
        run_id="run-adjusted-count",
        execution="hosted",
        source={"kind": "archive", "archive_artifact_id": "source-id"},
        agent={"connector": "auto"},
        scenario_count=1,
        runtime={"isolation": "dedicated_vm"},
        metadata={"source_kind": "archive", "agent_name": "uploaded-agent"},
    )
    job_path = tmp_path / "job.json"
    job_path.write_text(job.model_dump_json(), encoding="utf-8")

    async def fake_auto(args) -> int:
        del args
        return 0

    monkeypatch.setattr(authoring_entrypoint, "_auto", fake_auto)
    monkeypatch.setattr(authoring_entrypoint, "load_written", lambda _path: [1, 2])

    status = authoring_entrypoint.main(
        [str(job_path), "--source", str(source), "--output", str(output)]
    )

    assert status == 0
    persisted = HarnessJob.model_validate_json(job_path.read_text(encoding="utf-8"))
    assert persisted.scenario_count == 2


def test_hosted_authoring_inspects_imported_target_once_then_reuses_safe_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fi.alk.harness import authoring_entrypoint

    job = HarnessJob(
        job_id="job-provider-import",
        run_id="run-provider-import",
        execution="hosted",
        source={"kind": "archive", "archive_artifact_id": "source-id"},
        agent={
            "connector": "retell",
            "mode": "provider_import",
            "config": {"agent_id": "source-agent"},
            "secret_refs": {
                "RETELL_API_KEY": {
                    "manager": "platform-vault",
                    "key": "secret-id",
                    "purpose": "target_provider",
                }
            },
        },
        scenario_count=1,
        runtime={"isolation": "dedicated_vm"},
    )
    secrets = tmp_path / "target-secrets.json"
    secrets.write_text('{"RETELL_API_KEY":"retell-secret"}', encoding="utf-8")
    cache = tmp_path / "provider-profile.json"
    calls = []

    def inspect(provider, **kwargs):
        calls.append((provider, kwargs))
        return {"provider": "retell", "general_prompt": "Use the exact tool schema."}

    monkeypatch.setattr(authoring_entrypoint, "inspect_provider_target", inspect)

    first = authoring_entrypoint._load_provider_import_profile(job, secrets, cache)
    second = authoring_entrypoint._load_provider_import_profile(job, secrets, cache)

    assert first == second
    assert len(calls) == 1
    assert calls[0][1]["api_key"] == "retell-secret"
    assert not secrets.exists()
    assert "retell-secret" not in cache.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("connector", "secret_name", "config", "expected_provider", "expected_modality"),
    [
        ("vapi", "VAPI_API_KEY", {"assistant_id": "assistant-1"}, "vapi", "voice"),
        ("retell", "RETELL_API_KEY", {"agent_id": "agent-1"}, "retell", "voice"),
        (
            "retell_chat",
            "RETELL_API_KEY",
            {"agent_id": "agent-1"},
            "retell",
            "chat",
        ),
    ],
)
def test_connect_only_provider_authoring_inspects_the_real_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    connector: str,
    secret_name: str,
    config: dict[str, str],
    expected_provider: str,
    expected_modality: str,
) -> None:
    from fi.alk.harness import authoring_entrypoint

    job = HarnessJob(
        job_id=f"job-{connector}",
        run_id=f"run-{connector}",
        execution="hosted",
        source={"kind": "provider"},
        agent={
            "connector": connector,
            "mode": "connect_only",
            "config": config,
            "secret_refs": {
                secret_name: {
                    "manager": "platform-vault",
                    "key": "secret-id",
                    "purpose": "target_provider",
                }
            },
        },
        scenario_count=1,
        runtime={"isolation": "dedicated_vm"},
    )
    secrets = tmp_path / f"{connector}-secrets.json"
    secrets.write_text(json.dumps({secret_name: "provider-secret"}), encoding="utf-8")
    calls: list[tuple[str, dict[str, object]]] = []

    def inspect(provider: str, **kwargs: object) -> dict[str, object]:
        calls.append((provider, kwargs))
        return {"provider": provider}

    monkeypatch.setattr(authoring_entrypoint, "inspect_provider_target", inspect)

    profile = authoring_entrypoint._load_provider_import_profile(job, secrets)

    assert profile == {"provider": expected_provider}
    assert calls == [
        (
            expected_provider,
            {
                "source_target_id": next(iter(config.values())),
                "api_key": "provider-secret",
                "api_base_url": None,
                "target_modality": expected_modality,
            },
        )
    ]
    assert not secrets.exists()


@pytest.mark.parametrize("modality", ["voice", "chat"])
def test_deferred_hosted_authoring_never_starts_or_imports_the_submitted_runtime(
    tmp_path: Path,
    modality: str,
) -> None:
    from fi.alk.harness.contract import AgentContract, DataStore, Runtime, ToolSpec
    from fi.alk.harness.world.tools import world_tools

    contract = AgentContract(
        agent="hosted-voice-agent",
        modality=modality,
        tools=[ToolSpec(name="book_ride", args=["destination"])],
        real_use_cases=["book a ride"],
        data_store=DataStore(kind="postgres", database="rides"),
        runtime=Runtime(command=["python", "agent.py"]),
    )

    _server, world = world_tools(
        contract,
        tmp_path,
        source_root=str(tmp_path),
        deferred_runtime=True,
    )
    try:
        assert world.store.key == "sqlite"
        assert world.runtime_tools == {"book_ride"}
        assert "book_ride" not in world.handlers
    finally:
        world.close()


def test_bundle_is_content_addressed_and_reproducible(tmp_path: Path) -> None:
    (tmp_path / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    first = seal_bundle(tmp_path, _manifest())
    second = seal_bundle(tmp_path, first)

    assert first.digest == second.digest
    assert load_bundle(tmp_path).digest == first.digest
    assert first.files[0].path == "compose.yaml"


def test_bundle_detects_file_tampering(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services: {}\n", encoding="utf-8")
    seal_bundle(tmp_path, _manifest())
    compose.write_text("services: {unexpected: {}}\n", encoding="utf-8")

    with pytest.raises(BundleError, match="bundle_files_changed"):
        load_bundle(tmp_path)


def test_bundle_refuses_customer_secrets(tmp_path: Path) -> None:
    (tmp_path / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=customer-secret\n", encoding="utf-8")

    with pytest.raises(BundleError, match="bundle_secret_file_forbidden"):
        seal_bundle(tmp_path, _manifest())


def test_exported_bundle_survives_source_as_an_internal_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    session.mkdir()
    (source / "docker-compose.yml").write_text(
        "services: {tools-api: {image: busybox}}\n"
    )
    (source / "handler.py").write_text("def handle(): return 'ok'\n")
    (source / ".env.local").write_text("API_KEY=not-copied\n")
    (session / "contract.json").write_text('{"agent":"ride"}\n')

    root, bundle = export_session_bundle(source, session, name="ride-environment")

    assert bundle.runtime.document == "services/source/docker-compose.yml"
    assert (root / "services/source/handler.py").exists()
    assert not (root / "services/source/.env.local").exists()
    assert load_bundle(root).digest == bundle.digest
    plan = load_environment_plan(root, bundle=bundle)
    assert bundle.metadata["environment_plan_digest"] == plan.digest
    assert plan.source.source_digest == source_fingerprint(source)
    assert plan.runtime.document == "services/source/docker-compose.yml"


def test_environment_plan_is_reproducible_and_bound_to_its_bundle(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    (source / "docker-compose.yml").write_text(
        "services: {tools-api: {image: busybox}}\n", encoding="utf-8"
    )
    (source / "handler.py").write_text("VALUE = 1\n", encoding="utf-8")

    exports = [
        export_session_bundle(
            source, tmp_path / f"attempt-{attempt}", name="stable-environment"
        )
        for attempt in range(20)
    ]
    plan_digests = {
        load_environment_plan(root, bundle=bundle).digest for root, bundle in exports
    }
    bundle_digests = {bundle.digest for _root, bundle in exports}

    assert len(plan_digests) == 1
    assert len(bundle_digests) == 1

    second_root, _second_bundle = exports[-1]
    plan_path = second_root / ENVIRONMENT_PLAN_FILE
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    raw["metadata"]["managed"] = not raw["metadata"]["managed"]
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BundleError, match="bundle_files_changed"):
        load_bundle(second_root)


def test_environment_plan_rejects_a_validly_shaped_but_changed_decision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    (source / "docker-compose.yml").write_text(
        "services: {tools-api: {image: busybox}}\n", encoding="utf-8"
    )
    root, bundle = export_session_bundle(source, tmp_path / "session", name="stable")
    plan_path = root / ENVIRONMENT_PLAN_FILE
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    raw["services"] = ["unexpected"]
    plan_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(EnvironmentPlanError, match="digest_mismatch"):
        load_environment_plan(root, bundle=bundle)


def test_environment_up_from_bundle_ignores_a_later_source_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fi.alk.harness import cli

    source = tmp_path / "agent"
    source.mkdir()
    (source / "docker-compose.yml").write_text(
        "services: {tools-api: {image: busybox}}\n", encoding="utf-8"
    )
    handler = source / "handler.py"
    handler.write_text("VERSION = 'admitted'\n", encoding="utf-8")
    output = tmp_path / "session"
    bundle_root, _ = export_session_bundle(source, output, name="stable")
    handler.write_text("VERSION = 'changed-after-admission'\n", encoding="utf-8")
    observed: dict[str, str] = {}

    def fake_provision(selected_source, _destination, _contract):
        selected = Path(selected_source)
        observed["source"] = str(selected)
        observed["handler"] = (selected / "handler.py").read_text(encoding="utf-8")
        return SimpleNamespace(
            project="fagi-harness-frozen",
            services=["tools-api"],
            provision_seconds=0.1,
            overrides={},
        )

    monkeypatch.setattr("fi.alk.harness.provision.provision", fake_provision)
    result = asyncio.run(
        cli._environment(
            SimpleNamespace(
                action="up",
                path=str(source),
                bundle=str(bundle_root),
                out=str(output),
            )
        )
    )

    assert result == 0
    assert observed == {
        "source": str((bundle_root / "services/source").resolve()),
        "handler": "VERSION = 'admitted'\n",
    }


def test_environment_up_rejects_a_corrupt_bundle_before_provisioning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fi.alk.harness import cli

    source = tmp_path / "agent"
    source.mkdir()
    (source / "docker-compose.yml").write_text(
        "services: {tools-api: {image: busybox}}\n", encoding="utf-8"
    )
    (source / "handler.py").write_text("VERSION = 'sealed'\n", encoding="utf-8")
    output = tmp_path / "session"
    bundle_root, _ = export_session_bundle(source, output, name="stable")
    (bundle_root / "services/source/handler.py").write_text(
        "VERSION = 'tampered'\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "fi.alk.harness.provision.provision",
        lambda *_args, **_kwargs: pytest.fail(
            "corrupt bundle reached Docker provisioning"
        ),
    )

    result = asyncio.run(
        cli._environment(
            SimpleNamespace(
                action="up",
                path=str(source),
                bundle=str(bundle_root),
                out=str(output),
            )
        )
    )

    assert result == 1


def test_local_and_hosted_jobs_reject_the_other_sides_source() -> None:
    with pytest.raises(ValueError, match="hosted_job_cannot_use_local_path"):
        HarnessJob(
            job_id="job",
            run_id="run",
            execution="hosted",
            source={"kind": "local_repository", "local_path": "/private/agent"},
            agent={"connector": "http"},
        )

    with pytest.raises(
        ValueError, match="local_job_cannot_use_private_platform_github"
    ):
        HarnessJob(
            job_id="job",
            run_id="run",
            execution="local",
            source={
                "kind": "github",
                "installation_id": "installation",
                "repository": "customer/agent",
            },
            agent={"connector": "http"},
        )


def test_hosted_job_accepts_two_hundred_scenarios_and_rejects_more() -> None:
    common = {
        "job_id": "job",
        "run_id": "run",
        "execution": "hosted",
        "source": {"kind": "archive", "archive_artifact_id": "source-id"},
        "agent": {"connector": "auto"},
        "runtime": {"isolation": "dedicated_vm"},
    }

    assert HarnessJob(**common, scenario_count=200).scenario_count == 200
    with pytest.raises(ValueError, match="hosted_scenario_count_out_of_range"):
        HarnessJob(**common, scenario_count=201)


def test_job_carries_references_but_rejects_resolved_secrets() -> None:
    job = HarnessJob(
        job_id="job",
        run_id="run",
        execution="hosted",
        source={
            "kind": "github",
            "installation_id": "installation",
            "repository": "customer/agent",
            "ref": "main",
        },
        agent={
            "connector": "livekit",
            "secret_refs": {
                "api_key": {
                    "manager": "futureagi",
                    "key": "secret_livekit_key",
                    "purpose": "connect to agent",
                }
            },
        },
    )
    assert job.agent.secret_refs["api_key"].key == "secret_livekit_key"
    assert job.runtime.isolation is RuntimeIsolation.DEDICATED_VM

    with pytest.raises(ValueError, match="resolved_secret_forbidden"):
        HarnessJob(
            job_id="job",
            run_id="run",
            execution="hosted",
            source={
                "kind": "github",
                "installation_id": "installation",
                "repository": "customer/agent",
            },
            agent={"connector": "livekit", "config": {"api_key": "raw-key"}},
        )


def test_public_github_job_needs_no_installation_but_is_commit_pinnable() -> None:
    job = HarnessJob(
        job_id="job",
        run_id="run",
        execution="hosted",
        source={
            "kind": "github",
            "visibility": "public",
            "repository": "customer/public-agent",
            "ref": "main",
            "commit_sha": "a" * 40,
        },
        agent={"connector": "auto"},
    )

    assert job.source.installation_id is None
    assert job.source.commit_sha == "a" * 40


def test_public_github_branch_url_is_normalized_into_repository_and_ref() -> None:
    job = HarnessJob(
        job_id="job",
        run_id="run",
        execution="hosted",
        source={
            "kind": "github",
            "visibility": "public",
            "repository": "https://github.com/future-agi/future-agi/tree/feat/harness",
        },
        agent={"connector": "auto"},
    )

    assert job.source.repository == "future-agi/future-agi"
    assert job.source.ref == "feat/harness"


def test_github_branch_url_rejects_a_conflicting_explicit_ref() -> None:
    with pytest.raises(ValueError, match="github_ref_conflicts"):
        HarnessJob(
            job_id="job",
            run_id="run",
            execution="hosted",
            source={
                "kind": "github",
                "visibility": "public",
                "repository": "https://github.com/acme/agent/tree/release",
                "ref": "main",
            },
            agent={"connector": "auto"},
        )


@pytest.mark.parametrize(
    "security",
    [
        {"allow_privileged": True},
        {"allow_host_runtime_control": True},
        {"read_only_source": False},
    ],
)
def test_hosted_job_rejects_unsafe_security_policy(security: dict) -> None:
    with pytest.raises(ValueError, match="hosted_.*forbidden|hosted_source"):
        HarnessJob(
            job_id="job",
            run_id="run",
            execution="hosted",
            source={
                "kind": "github",
                "visibility": "public",
                "repository": "customer/public-agent",
            },
            agent={"connector": "auto"},
            security=security,
        )


def test_job_retry_policy_cannot_retry_agent_or_grading_failures() -> None:
    with pytest.raises(ValueError, match="retry_domain_unsafe"):
        HarnessJob(
            job_id="job",
            run_id="run",
            execution="hosted",
            source={
                "kind": "github",
                "visibility": "public",
                "repository": "customer/public-agent",
            },
            agent={"connector": "auto"},
            retry={"retryable_domains": ["agent", "grading"]},
        )


def test_failed_stage_is_reported_as_structured_non_retryable_failure(tmp_path: Path):
    (tmp_path / "harness-events.jsonl").write_text(
        '{"type":"harness.stage.failed","payload":'
        '{"stage":"scenarios","status":1,"detail":"invalid generated fixture"}}\n',
        encoding="utf-8",
    )

    failure = _failure_from_events(tmp_path)

    assert failure.domain.value == "simulator"
    assert failure.stage.value == "validating_scenarios"
    assert failure.retryable is False


def test_only_agent_failures_are_owned_by_the_customer() -> None:
    agent = HarnessFailure(
        domain=FailureDomain.AGENT,
        stage=HarnessStage.RUNNING,
        code="tool_result_wrong",
        message="The submitted agent returned the wrong record",
    )
    environment = HarnessFailure(
        domain=FailureDomain.ENVIRONMENT,
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        code="readiness_failed",
        message="internal endpoint and stack detail",
        retryable=True,
    )

    assert agent.for_customer().model_dump() == {
        "category": "agent_failure",
        "owner": "customer_agent",
        "code": "tool_result_wrong",
        "message": "The submitted agent returned the wrong record",
        "retryable": False,
    }
    public = environment.for_customer()
    assert public.category == "system_failure"
    assert public.owner.value == "futureagi"
    assert "internal endpoint" not in public.message
    assert public.retryable is True


def test_artifact_ingestion_has_its_own_futureagi_failure_domain(tmp_path: Path):
    (tmp_path / "harness-events.jsonl").write_text(
        '{"type":"harness.stage.failed","payload":'
        '{"stage":"uploading_artifacts","status":1,"detail":"hash mismatch"}}\n',
        encoding="utf-8",
    )

    failure = _failure_from_events(tmp_path)

    assert failure.domain is FailureDomain.ARTIFACT
    assert failure.owner.value == "futureagi"


class _Transport:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def send(self, _run_id: str, events: list[CanonicalEvent]) -> set[str]:
        self.seen.extend(event.event_id for event in events)
        return {event.event_id for event in events}


def test_event_delivery_is_durable_and_retryable(tmp_path: Path) -> None:
    outbox = EventOutbox(tmp_path, "run")
    offline = BufferedEventSink(outbox)
    events = [
        CanonicalEvent.create(
            run_id="run",
            test_case_id="harness",
            event_type="harness.stage.completed",
            source="harness",
            sequence=index,
            payload={"stage": stage},
        )
        for index, stage in enumerate(("environment", "scenarios"))
    ]
    for event in events:
        offline.write(event)
    assert [event.event_id for event in outbox.pending()] == [
        event.event_id for event in events
    ]

    transport = _Transport()
    online = BufferedEventSink(outbox, transport)
    assert online.flush() == 2
    assert outbox.pending() == []
    assert online.flush() == 0
    assert transport.seen == [event.event_id for event in events]


def test_hosted_github_checkout_keeps_token_out_of_process_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = "installation-secret-token"
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        Path(command[-1]).mkdir()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("fi.alk.harness.executor.subprocess.run", run)
    job = HarnessJob(
        job_id="job",
        run_id="run",
        execution="hosted",
        source={
            "kind": "github",
            "installation_id": "installation",
            "repository": "customer/private-agent",
            "ref": "main",
        },
        agent={"connector": "http"},
    )

    checkout = asyncio.run(
        GitHubSourceAcquirer(lambda _installation: token).acquire(job, tmp_path)
    )

    assert checkout == tmp_path / "repository"
    assert token not in " ".join(observed["command"])
    assert observed["environment"]["GIT_CONFIG_VALUE_0"].endswith(token)


def test_public_github_checkout_does_not_request_or_inject_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}

    def token(_installation):
        raise AssertionError("public checkout must not request an installation token")

    def run(command, **kwargs):
        observed["environment"] = kwargs["env"]
        Path(command[-1]).mkdir()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("fi.alk.harness.executor.subprocess.run", run)
    job = HarnessJob(
        job_id="job",
        run_id="run",
        execution="hosted",
        source={
            "kind": "github",
            "visibility": "public",
            "repository": "customer/public-agent",
        },
        agent={"connector": "auto"},
    )

    checkout = asyncio.run(GitHubSourceAcquirer(token).acquire(job, tmp_path))

    assert checkout == tmp_path / "repository"
    assert "GIT_CONFIG_VALUE_0" not in observed["environment"]


def test_public_github_branch_url_clones_the_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        Path(command[-1]).mkdir()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("fi.alk.harness.executor.subprocess.run", run)
    job = HarnessJob(
        job_id="job",
        run_id="run",
        execution="hosted",
        source={
            "kind": "github",
            "visibility": "public",
            "repository": "https://github.com/acme/agent/tree/feat/harness",
        },
        agent={"connector": "auto"},
    )

    asyncio.run(GitHubSourceAcquirer(lambda _: "").acquire(job, tmp_path))

    assert observed["command"] == [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "feat/harness",
        "https://github.com/acme/agent.git",
        str(tmp_path / "repository"),
    ]


def test_source_fingerprint_hashes_symlink_metadata_without_reading_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "outside-secret"
    secret.write_text("first secret", encoding="utf-8")
    (source / "link").symlink_to(secret)

    first = source_fingerprint(source)
    secret.write_text("different secret", encoding="utf-8")
    second = source_fingerprint(source)

    assert first == second
