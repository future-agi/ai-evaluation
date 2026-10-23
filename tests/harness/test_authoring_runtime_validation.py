import asyncio
import json
from types import SimpleNamespace

import pytest

from fi.alk.harness.authoring_runtime_validation import (
    _database_lanes,
    _fallback_diagnostic,
    _make_local_seed_files_readable,
    RuntimeValidationError,
    _setup_check_keys,
    _generic_candidate_hash,
    _world_isolation_status,
    _write_generic_certificate,
    _write_runtime_evidence,
    validate_and_repair,
    validate_once,
)
from fi.alk.harness.certification import (
    CertificationChecks,
    CheckStatus,
    GenericHarnessArtifactStore,
    RuntimeValidationEvidence,
)
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.diagnostics import HarnessDiagnostic
from fi.alk.harness.job import HarnessStage
from fi.alk.harness.repair_controller import (
    CandidateObservation,
    RepairAction,
    RepairBudgets,
    RepairController,
    RepairPhase,
)
from fi.alk.harness.repair_patch import (
    RepairPatchOp,
    RepairPatchOperation,
    WorldIRRepairPatch,
)
from fi.alk.harness.runtime_repair import RuntimePlanPatch, _contract_hash
from fi.alk.harness.contract import AgentContract, Runtime
from fi.alk.harness.source_model import (
    LogicalType,
    SourceColumn,
    SourceModel,
    SourceTable,
)
from fi.alk.harness.world_ir import (
    WorldIR,
    WorldRow,
    WorldTable,
    WorldValue,
    validate_world_ir,
)


def test_database_lanes_default_to_one_per_cpu(monkeypatch):
    from fi.alk.harness import process_runtime

    monkeypatch.delenv("ALK_VALIDATION_DB_LANES", raising=False)
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    monkeypatch.setattr(process_runtime, "_cgroup_cpu_quota", lambda: None)
    assert _database_lanes() == 3
    monkeypatch.setenv("ALK_VALIDATION_DB_LANES", "")
    assert _database_lanes() == 3
    monkeypatch.setattr(process_runtime, "_cgroup_cpu_quota", lambda: 2.0)
    assert _database_lanes() == 2
    monkeypatch.setenv("ALK_VALIDATION_DB_LANES", "16")
    assert _database_lanes() == 16
    monkeypatch.setenv("ALK_VALIDATION_DB_LANES", "abc")
    with pytest.raises(ValueError, match="ALK_VALIDATION_DB_LANES"):
        _database_lanes()


def test_local_seed_handoff_opens_only_generated_seed_files(tmp_path):
    bundle = tmp_path / "bundle"
    seed = bundle / "seed"
    seed.mkdir(parents=True)
    sql = seed / "source-schema.sql"
    sql.write_text("select 1;\n")
    secret = bundle / "private.txt"
    secret.write_text("private")
    bundle.chmod(0o700)
    seed.chmod(0o700)
    sql.chmod(0o600)
    secret.chmod(0o600)

    _make_local_seed_files_readable(bundle)

    assert bundle.stat().st_mode & 0o055 == 0o055
    assert seed.stat().st_mode & 0o055 == 0o055
    assert sql.stat().st_mode & 0o044 == 0o044
    assert secret.stat().st_mode & 0o077 == 0


def test_validation_repairs_then_revalidates_and_records_scope(tmp_path):
    calls = []

    async def validate(*args):
        calls.append("validate")
        if len(calls) == 1:
            raise RuntimeValidationError("environment", "foreign key mismatch")
        return 10

    async def repair(phase, guidance):
        assert phase == "environment"
        assert "foreign key mismatch" in guidance
        assert "disable database constraints" in guidance
        calls.append("repair")
        return 0

    asyncio.run(
        validate_and_repair(None, tmp_path, tmp_path, validate=validate, repair=repair)
    )
    assert calls == ["validate", "repair", "validate"]
    proof = json.loads((tmp_path / "runtime-validation.json").read_text())
    assert proof["setup_ready_scenarios"] == 10
    assert proof["reference_tools_proven"] is False


def test_validation_stops_after_two_repairs_without_false_certificate(tmp_path):
    repairs = []

    async def validate(*args):
        raise RuntimeValidationError("scenarios", "bad setup")

    async def repair(*args):
        repairs.append(args)
        return 0

    with pytest.raises(RuntimeValidationError, match="bad setup"):
        asyncio.run(
            validate_and_repair(
                None, tmp_path, tmp_path, validate=validate, repair=repair
            )
        )
    assert len(repairs) == 2
    assert not (tmp_path / "runtime-validation.json").exists()


def test_environment_repairs_do_not_exhaust_scenario_repair_budget(tmp_path):
    failures = iter(["environment", "scenarios", "environment", "scenarios", None])
    repairs = []

    async def validate(*args):
        phase = next(failures)
        if phase:
            raise RuntimeValidationError(phase, "invalid generated data")
        return 10

    async def repair(phase, guidance):
        repairs.append(phase)
        return 0

    asyncio.run(
        validate_and_repair(None, tmp_path, tmp_path, validate=validate, repair=repair)
    )
    assert repairs == ["environment", "scenarios", "environment", "scenarios"]
    assert (
        json.loads((tmp_path / "runtime-validation.json").read_text())["attempts"] == 5
    )


def test_infrastructure_failure_does_not_trigger_data_repair(tmp_path):
    async def validate(*args):
        raise RuntimeValidationError("infrastructure", "CERTIFICATE_VERIFY_FAILED")

    async def repair(*args):
        pytest.fail("Infrastructure trust errors must not rewrite generated data")

    with pytest.raises(RuntimeValidationError, match="CERTIFICATE_VERIFY_FAILED"):
        asyncio.run(
            validate_and_repair(
                None, tmp_path, tmp_path, validate=validate, repair=repair
            )
        )


def test_runtime_validation_error_carries_structured_diagnostics() -> None:
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        component="world_ir",
        code="required_value_missing",
        message="required value missing",
    )

    error = RuntimeValidationError(
        "environment", "safe summary", diagnostics=(diagnostic,)
    )

    assert error.diagnostics == (diagnostic,)


def test_generic_validation_uses_typed_patch_policy_and_persists_history(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    authoring = tmp_path / "authoring"
    source.mkdir()
    authoring.mkdir()
    (source / "agent.py").write_text("agent", encoding="utf-8")
    calls = []
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        component="world_ir",
        code="required_value_missing",
        message="missing value",
    )

    async def validate(*args):
        calls.append("validate")
        if len(calls) == 1:
            raise RuntimeValidationError(
                "environment", "invalid", diagnostics=(diagnostic,)
            )
        return 5

    async def repair(phase, guidance):
        assert phase == "environment"
        assert "required_value_missing@world_ir" in guidance
        assert "strategy=request_targeted_data_patch" in guidance
        assert "evidence=missing value" in guidance
        (authoring / "world.sqlite").write_text("changed", encoding="utf-8")
        calls.append("repair")
        return 0

    job = SimpleNamespace(metadata={"generic_harness_v1": True})
    asyncio.run(
        validate_and_repair(job, source, authoring, validate=validate, repair=repair)
    )

    assert calls == ["validate", "repair", "validate"]
    history = json.loads(
        (authoring / "generic-harness" / "repair-history.json").read_text()
    )
    assert [item["action"] for item in history["decisions"]] == [
        RepairAction.PATCH_ENVIRONMENT.value,
        RepairAction.CERTIFY.value,
    ]
    proof = json.loads((authoring / "runtime-validation.json").read_text())
    assert proof["generic_harness"] == "v1"
    assert proof["setup_ready_scenarios"] == 5


def test_generic_repair_without_typed_patch_keeps_validation_failure(
    tmp_path, monkeypatch
) -> None:
    from fi.alk.harness import repair_authoring

    source = tmp_path / "source"
    authoring = tmp_path / "authoring"
    artifacts = authoring / "generic-harness"
    source.mkdir()
    artifacts.mkdir(parents=True)
    (source / "agent.py").write_text("agent", encoding="utf-8")
    (artifacts / "source-model.json").write_text("source", encoding="utf-8")
    (artifacts / "world-ir.json").write_text("world", encoding="utf-8")
    monkeypatch.setattr(
        GenericHarnessArtifactStore, "read_source_model", lambda self: object()
    )
    monkeypatch.setattr(
        GenericHarnessArtifactStore, "read_world_ir", lambda self: object()
    )

    async def no_patch(*args):
        raise repair_authoring.RepairPatchNotSubmittedError("no patch submitted")

    monkeypatch.setattr(repair_authoring, "request_world_ir_patch", no_patch)
    calls = []

    async def validate(*args):
        calls.append("validate")
        raise RuntimeValidationError(
            "environment", "Source data invariant review did not finish; not certified"
        )

    job = SimpleNamespace(metadata={"generic_harness_v1": True})
    with pytest.raises(RuntimeValidationError, match="Source data invariant review"):
        asyncio.run(validate_and_repair(job, source, authoring, validate=validate))
    assert calls == ["validate"]
    history = json.loads((artifacts / "repair-history.json").read_text())
    assert history["results"][0]["outcome"] == "failed"


def test_generic_validation_applies_typed_world_patch_transactionally(tmp_path) -> None:
    source_root = tmp_path / "source"
    authoring = tmp_path / "authoring"
    source_root.mkdir()
    authoring.mkdir()
    (source_root / "agent.py").write_text("agent", encoding="utf-8")
    source_model = SourceModel.create(
        source_digest="sha256:" + "a" * 64,
        engine="postgres",
        tables=(
            SourceTable(
                name="accounts",
                columns=(
                    SourceColumn(
                        name="id",
                        logical_type=LogicalType.INTEGER,
                        native_type="integer",
                        nullable=False,
                        has_default=False,
                    ),
                ),
            ),
        ),
    )
    world = WorldIR.create(
        source_model_fingerprint=source_model.fingerprint,
        tables=(
            WorldTable(
                source_name="accounts",
                rows=(WorldRow(identity="account-1", values={}),),
            ),
        ),
    )
    store = GenericHarnessArtifactStore(authoring / "generic-harness")
    store.write_source_model(source_model)
    store.write_world_ir(world)
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        component="world_ir",
        code="required_value_missing",
        message="accounts.id is missing",
    )

    async def validate(*args):
        candidate = store.read_world_ir()
        try:
            validate_world_ir(candidate, source_model)
        except Exception:
            raise RuntimeValidationError(
                "environment", "invalid", diagnostics=(diagnostic,)
            ) from None
        return 1

    async def repair(*args):
        return WorldIRRepairPatch.create(
            base_world_ir_hash=world.fingerprint,
            operations=(
                RepairPatchOperation(
                    op=RepairPatchOp.SET_VALUE,
                    table="accounts",
                    row_identity="account-1",
                    column="id",
                    value=WorldValue.present(LogicalType.INTEGER, 1),
                    reason="required_value_missing",
                    evidence_refs=("source-model:accounts.id",),
                ),
            ),
        )

    job = SimpleNamespace(metadata={"generic_harness_v1": True})
    asyncio.run(
        validate_and_repair(
            job, source_root, authoring, validate=validate, repair=repair
        )
    )

    validate_world_ir(store.read_world_ir(), source_model)
    assert (authoring / "generic-harness" / "repair-patch-0001.json").is_file()
    history = store.read_repair_history()
    assert (
        history.results[0].after_candidate_hash
        != history.results[0].before_candidate_hash
    )


def test_generic_validation_rejects_unchanged_compiler_failure(tmp_path) -> None:
    source = tmp_path / "source"
    authoring = tmp_path / "authoring"
    source.mkdir()
    authoring.mkdir()
    (source / "agent.py").write_text("agent", encoding="utf-8")
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        component="world_import",
        code="array_shape_mismatch",
        message="malformed array",
    )
    calls = []

    async def validate(*args):
        calls.append("validate")
        raise RuntimeValidationError(
            "environment", "invalid", diagnostics=(diagnostic,)
        )

    async def repair(*args):
        pytest.fail("compiler failures must not invoke model repair")

    job = SimpleNamespace(metadata={"generic_harness_v1": True})
    with pytest.raises(RuntimeValidationError):
        asyncio.run(
            validate_and_repair(
                job, source, authoring, validate=validate, repair=repair
            )
        )

    assert calls == ["validate", "validate"]
    history = json.loads(
        (authoring / "generic-harness" / "repair-history.json").read_text()
    )
    assert [item["action"] for item in history["decisions"]] == [
        RepairAction.RECOMPILE.value,
        RepairAction.REJECT.value,
    ]


def test_generic_validation_retries_infrastructure_without_reauthoring(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    authoring = tmp_path / "authoring"
    source.mkdir()
    authoring.mkdir()
    (source / "agent.py").write_text("agent", encoding="utf-8")
    calls = []
    waits = []

    async def validate(*args):
        calls.append("validate")
        if len(calls) == 1:
            raise RuntimeValidationError("infrastructure", "temporary outage")
        return 1

    async def repair(*args):
        pytest.fail("infrastructure failures must not invoke model repair")

    async def sleeper(seconds):
        waits.append(seconds)

    controller = RepairController(
        RepairBudgets(
            infrastructure_retries_per_candidate=1,
            infrastructure_initial_backoff_seconds=0,
            infrastructure_jitter_ratio=0,
        )
    )
    job = SimpleNamespace(metadata={"generic_harness_v1": True})
    asyncio.run(
        validate_and_repair(
            job,
            source,
            authoring,
            validate=validate,
            repair=repair,
            controller=controller,
            sleeper=sleeper,
        )
    )

    assert calls == ["validate", "validate"]
    assert waits == [0]
    assert [item.action for item in controller.history.decisions] == [
        RepairAction.RETRY_INFRASTRUCTURE,
        RepairAction.CERTIFY,
    ]


def test_generic_candidate_hash_ignores_repository_metadata_and_own_artifacts(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    authoring = tmp_path / "authoring"
    (source / ".git").mkdir(parents=True)
    authoring.mkdir()
    (source / "agent.py").write_text("agent", encoding="utf-8")
    (authoring / "world.sqlite").write_text("world", encoding="utf-8")
    before = _generic_candidate_hash(source, authoring)

    (source / ".git" / "index").write_text("moving metadata", encoding="utf-8")
    (authoring / "generic-harness").mkdir()
    (authoring / "generic-harness" / "repair-history.json").write_text(
        "changing history", encoding="utf-8"
    )

    assert _generic_candidate_hash(source, authoring) == before


def test_runtime_evidence_reads_generic_models_when_world_isolation_not_run(
    tmp_path, monkeypatch
) -> None:
    from fi.alk.harness import certification

    captured = {}

    class Store:
        def __init__(self, _root):
            pass

        def read_source_model(self):
            return SimpleNamespace(fingerprint="sha256:" + "b" * 64, engine="none")

        def read_world_ir(self):
            return SimpleNamespace(fingerprint="sha256:" + "c" * 64)

        def write_runtime_evidence(self, evidence):
            captured["evidence"] = evidence

        def write_tool_certification(self, report):
            captured["report"] = report

        def write_action_certification(self, report):
            captured["action_report"] = report

    monkeypatch.setattr(certification, "GenericHarnessArtifactStore", Store)
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    (authoring / "contract.json").write_text("{}", encoding="utf-8")
    report = SimpleNamespace(certified_or_runtime_only=1, total=1)
    action_report = SimpleNamespace(passed=1, certified=1, executable=1, total=1)
    manifest = SimpleNamespace(
        provenance=SimpleNamespace(source_digest="a" * 64),
        digest="sha256:" + "d" * 64,
    )

    _write_runtime_evidence(
        job=SimpleNamespace(),
        authoring=authoring,
        manifest=manifest,
        count=1,
        external_provider=False,
        tool_report=report,
        action_report=action_report,
        reset_equivalence=CheckStatus.PASSED,
        world_isolation=CheckStatus.NOT_RUN,
    )

    evidence = captured["evidence"]
    assert evidence.source_schema_hash == "sha256:" + "b" * 64
    assert evidence.world_ir_hash == "sha256:" + "c" * 64
    assert evidence.compiler_version == "futureagi.no-state-compiler.v1"
    assert evidence.checks.schema_and_seed is CheckStatus.PASSED
    assert evidence.checks.world_isolation is CheckStatus.NOT_RUN


def test_fixed_port_single_world_marks_concurrent_isolation_not_applicable() -> None:
    assert (
        _world_isolation_status(
            {
                "conformance": None,
                "degrade_reason": "fixed_port",
                "effective_parallelism": 1,
            }
        )
        is CheckStatus.NOT_APPLICABLE
    )
    assert (
        _world_isolation_status(
            {
                "conformance": None,
                "degrade_reason": "fixed_port",
                "effective_parallelism": 2,
            }
        )
        is CheckStatus.NOT_RUN
    )
    assert _world_isolation_status({"conformance": True}) is CheckStatus.PASSED


def test_external_provider_runtime_evidence_marks_state_checks_not_applicable(
    tmp_path, monkeypatch
) -> None:
    from fi.alk.harness import certification

    captured = {}

    class Store:
        def __init__(self, _root):
            pass

        def write_runtime_evidence(self, evidence):
            captured["evidence"] = evidence

        def write_tool_certification(self, _report):
            pass

        def write_action_certification(self, _report):
            pass

    monkeypatch.setattr(certification, "GenericHarnessArtifactStore", Store)
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    (authoring / "contract.json").write_text("{}", encoding="utf-8")
    report = SimpleNamespace(certified_or_runtime_only=0, total=0)
    action_report = SimpleNamespace(certified=0, executable=0, total=0)
    manifest = SimpleNamespace(
        provenance=SimpleNamespace(source_digest="a" * 64),
        digest="sha256:" + "d" * 64,
    )

    _write_runtime_evidence(
        job=SimpleNamespace(),
        authoring=authoring,
        manifest=manifest,
        count=1,
        external_provider=True,
        tool_report=report,
        action_report=action_report,
        reset_equivalence=CheckStatus.NOT_APPLICABLE,
        world_isolation=CheckStatus.NOT_APPLICABLE,
    )

    checks = captured["evidence"].checks
    assert checks.schema_and_seed is CheckStatus.NOT_APPLICABLE
    assert checks.source_invariants is CheckStatus.NOT_APPLICABLE
    assert checks.reset_equivalence is CheckStatus.NOT_APPLICABLE
    assert checks.world_isolation is CheckStatus.NOT_APPLICABLE


def test_generic_certificate_combines_runtime_evidence_and_repair_history(
    tmp_path,
) -> None:
    authoring = tmp_path / "authoring"
    store = GenericHarnessArtifactStore(authoring / "generic-harness")

    def digest(character):
        return "sha256:" + character * 64

    store.write_runtime_evidence(
        RuntimeValidationEvidence(
            source_digest=digest("a"),
            source_schema_hash=digest("b"),
            world_ir_hash=digest("c"),
            compiler_version="compiler-v1",
            bundle_digest=digest("d"),
            contract_hash=digest("e"),
            scenario_set_hash=digest("f"),
            checks=CertificationChecks(
                static=CheckStatus.PASSED,
                schema_and_seed=CheckStatus.PASSED,
                scenario_setup_ready="2/2",
            ),
            limitations=("tool trajectories not run",),
        )
    )
    controller = RepairController()
    controller.decide(
        CandidateObservation(candidate_hash=digest("0"), phase=RepairPhase.ENVIRONMENT)
    )
    job = SimpleNamespace(
        source=SimpleNamespace(repository="future-agi/example", commit_sha="1" * 40),
        metadata={"daytona_snapshot": "snapshot-r1"},
    )

    _write_generic_certificate(job, authoring, controller.history)

    certificate = store.read_certification()
    assert certificate.status.value == "certified"
    assert certificate.source.repository == "future-agi/example"
    assert certificate.runtime.snapshot == "snapshot-r1"
    assert certificate.runtime.validation_attempts == 1
    assert certificate.checks.scenario_setup_ready == "2/2"
    assert (
        json.loads((authoring / "runtime-validation.json").read_text())["fingerprint"]
        == certificate.fingerprint
    )


@pytest.mark.parametrize("bad_setup", [False, True])
def test_runtime_gate_resets_each_scenario_and_preserves_execution_secrets(
    tmp_path, monkeypatch, bad_setup
):
    from fi.alk.harness import (
        bundle_author_v2,
        hosted_entrypoint,
        outbound,
        process_preflight,
        process_runtime,
        scenario_source,
        source_data_invariants,
    )

    original = tmp_path / "execution-secrets.json"
    original.write_text('{"TEST_SECRET":"target-secret-value"}')
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    calls = []

    async def invariants(*args, **kwargs):
        return []

    monkeypatch.setattr(source_data_invariants, "author_invariants", invariants)

    class Provider:
        def __init__(self, **kwargs):
            assert kwargs["secrets_path"] != original
            assert kwargs["secrets_path"].read_bytes() == original.read_bytes()

        async def provision(self, *args, **kwargs):
            actual_provider = provider_type(
                cpu_observer=lambda: None, mem_observer=lambda: None
            )
            assert actual_provider._run_admission(1, kwargs["work_directory"]) == 1
            calls.append("provision")
            return [SimpleNamespace(endpoints={})]

        async def reset(self, *args, **kwargs):
            calls.append("reset")

        async def close(self, **kwargs):
            calls.append("close")

    class World:
        def read_only(self):
            return self

    class Factory:
        def __init__(self, work):
            pass

        async def create(self, *args, **kwargs):
            return World()

    def setup(world):
        calls.append("setup")
        if bad_setup:
            raise ValueError("bad target-secret-value")

    provider_type = process_runtime.ProcessRuntimeProvider
    monkeypatch.setattr(process_runtime, "ProcessRuntimeProvider", Provider)
    monkeypatch.setattr(hosted_entrypoint, "ProcessWorldFactory", Factory)
    monkeypatch.setattr(bundle_author_v2, "author_bundle_v2", lambda **kwargs: object())
    monkeypatch.setattr(
        process_preflight, "preflight_bundle", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        outbound,
        "load_capabilities",
        lambda *, unlink: (
            SimpleNamespace(attempt_id="test", expires_at=None)
            if unlink is False
            else pytest.fail("validation consumed execution capabilities")
        ),
    )
    monkeypatch.setattr(
        scenario_source,
        "load_scenarios",
        lambda bundle: [
            SimpleNamespace(scenario_key=str(i), setup=setup, ready=lambda world: True)
            for i in range(2)
        ],
    )
    job = SimpleNamespace(
        agent=SimpleNamespace(secret_refs={}),
        scenario_count=2,
        seed=0,
        runtime=SimpleNamespace(cpu_units=4, memory_mb=8192),
    )
    if bad_setup:
        with pytest.raises(RuntimeValidationError) as error:
            asyncio.run(validate_once(job, tmp_path, authoring, secrets_path=original))
        assert "target-secret-value" not in str(error.value)
        assert "0: setup:" in str(error.value)
        assert "1: setup:" in str(error.value)
        assert calls == [
            "provision",
            "reset",
            "reset",
            "reset",
            "setup",
            "reset",
            "setup",
            "reset",
            "setup",
            "reset",
            "setup",
            "close",
        ]
    else:
        assert (
            asyncio.run(validate_once(job, tmp_path, authoring, secrets_path=original))
            == 2
        )
        assert calls == [
            "provision",
            "reset",
            "reset",
            "reset",
            "setup",
            "reset",
            "setup",
            "close",
        ]
    assert original.exists()


def test_connect_only_provider_validation_does_not_invent_source_data_review(
    tmp_path, monkeypatch
):
    from fi.alk.harness import (
        bundle_author_v2,
        hosted_entrypoint,
        outbound,
        process_preflight,
        process_runtime,
        scenario_source,
        source_data_invariants,
    )

    original = tmp_path / "execution-secrets.json"
    original.write_text('{"RETELL_API_KEY":"secret"}')
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    calls = []

    async def forbidden_review(*_args, **_kwargs):
        pytest.fail(
            "connect-only provider state cannot be reviewed as local source data"
        )

    monkeypatch.setattr(source_data_invariants, "author_invariants", forbidden_review)

    class Provider:
        def __init__(self, **_kwargs):
            pass

        async def provision(self, *_args, **_kwargs):
            calls.append("provision")
            return [SimpleNamespace(endpoints={})]

        async def reset(self, *_args, **_kwargs):
            calls.append("reset")

        async def close(self, **_kwargs):
            calls.append("close")

    class World:
        def read_only(self):
            return self

    class Factory:
        def __init__(self, _work):
            pass

        async def create(self, *_args, **_kwargs):
            return World()

    monkeypatch.setattr(process_runtime, "ProcessRuntimeProvider", Provider)
    monkeypatch.setattr(hosted_entrypoint, "ProcessWorldFactory", Factory)
    monkeypatch.setattr(
        bundle_author_v2, "author_bundle_v2", lambda **_kwargs: object()
    )
    monkeypatch.setattr(process_preflight, "preflight_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(
        outbound,
        "load_capabilities",
        lambda *, unlink: SimpleNamespace(attempt_id="test", expires_at=None),
    )
    monkeypatch.setattr(
        scenario_source,
        "load_scenarios",
        lambda _bundle: [
            SimpleNamespace(
                scenario_key="one",
                setup=lambda _world: None,
                ready=lambda _world: True,
            )
        ],
    )
    job = HarnessJob(
        job_id="job-provider",
        run_id="run-provider",
        execution="hosted",
        source={"kind": "provider"},
        agent={
            "connector": "retell",
            "mode": "connect_only",
            "config": {"agent_id": "agent_test"},
            "secret_refs": {
                "api_key": {
                    "manager": "platform-vault",
                    "key": "retell-key",
                    "purpose": "target_provider",
                }
            },
        },
        scenario_count=1,
        runtime={"isolation": "dedicated_vm"},
    )

    assert (
        asyncio.run(validate_once(job, tmp_path, authoring, secrets_path=original)) == 1
    )
    assert calls == ["provision", "reset", "close"]


def _sourced(entry, source):
    def function(*_args):
        return None

    function._alk_source = source
    function._alk_entry = entry
    return function


def _noop(entry):
    return _sourced(entry, f'def {entry}(world):\n    """Nothing beyond the base world."""\n')


def test_does_nothing_reads_only_a_bare_function_as_a_no_op():
    from fi.alk.harness.scenario_source import does_nothing

    assert does_nothing(_noop("setup"))
    assert does_nothing(_sourced("ready", "def ready(world):\n    return None\n"))
    assert does_nothing(_sourced("setup", "def setup(world):\n    pass\n"))
    for source in (
        "def setup(world):\n    world.insert('t', {})\n",
        "import os\ndef setup(world):\n    pass\n",
        "def setup(world, seed=print('x')):\n    pass\n",
        "def setup(world):\n    return True\n",
        "def other(world):\n    pass\n",
    ):
        assert not does_nothing(_sourced("setup", source)), source
    assert not does_nothing(lambda _world: None)


def test_connect_only_validation_walks_only_scenarios_whose_setup_does_something(
    tmp_path, monkeypatch
):
    from fi.alk.harness import (
        bundle_author_v2,
        hosted_entrypoint,
        outbound,
        process_preflight,
        process_runtime,
        scenario_source,
        source_data_invariants,
    )

    original = tmp_path / "execution-secrets.json"
    original.write_text('{"RETELL_API_KEY":"secret"}')
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    calls = []

    async def forbidden_review(*_args, **_kwargs):
        pytest.fail(
            "connect-only provider state cannot be reviewed as local source data"
        )

    monkeypatch.setattr(source_data_invariants, "author_invariants", forbidden_review)

    class Provider:
        def __init__(self, **_kwargs):
            pass

        async def provision(self, *_args, **_kwargs):
            calls.append("provision")
            return [SimpleNamespace(endpoints={})]

        async def reset(self, *_args, **_kwargs):
            calls.append("reset")

        async def close(self, **_kwargs):
            calls.append("close")

    class World:
        def read_only(self):
            return self

    class Factory:
        def __init__(self, _work):
            pass

        async def create(self, *_args, **_kwargs):
            return World()

    monkeypatch.setattr(process_runtime, "ProcessRuntimeProvider", Provider)
    monkeypatch.setattr(hosted_entrypoint, "ProcessWorldFactory", Factory)
    monkeypatch.setattr(
        bundle_author_v2, "author_bundle_v2", lambda **_kwargs: object()
    )
    monkeypatch.setattr(process_preflight, "preflight_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(
        outbound,
        "load_capabilities",
        lambda *, unlink: SimpleNamespace(attempt_id="test", expires_at=None),
    )
    monkeypatch.setattr(
        scenario_source,
        "load_scenarios",
        lambda _bundle: [
            SimpleNamespace(scenario_key=f"quiet-{index}", setup=_noop("setup"), ready=_noop("ready"))
            for index in range(3)
        ]
        + [
            SimpleNamespace(
                scenario_key="seeds",
                setup=_sourced("setup", "def setup(world):\n    world.insert('t', {})\n"),
                ready=_noop("ready"),
            )
        ],
    )
    job = HarnessJob(
        job_id="job-provider",
        run_id="run-provider",
        execution="hosted",
        source={"kind": "provider"},
        agent={
            "connector": "retell",
            "mode": "connect_only",
            "config": {"agent_id": "agent_test"},
            "secret_refs": {
                "api_key": {
                    "manager": "platform-vault",
                    "key": "retell-key",
                    "purpose": "target_provider",
                }
            },
        },
        scenario_count=4,
        runtime={"isolation": "dedicated_vm"},
    )

    assert (
        asyncio.run(validate_once(job, tmp_path, authoring, secrets_path=original)) == 4
    )
    assert calls == ["provision", "reset", "close"]


def test_database_clone_checks_use_the_lane_count(tmp_path, monkeypatch):
    from fi.alk.harness import (
        action_certification,
        authoring_runtime_validation,
        bundle_author_v2,
        certification,
        contract,
        hosted_entrypoint,
        outbound,
        process_preflight,
        process_runtime,
        scenario_source,
        source_data_invariants,
        tool_certification,
    )

    original = tmp_path / "execution-secrets.json"
    original.write_text("{}")
    authoring = tmp_path / "authoring"
    (authoring / "generic-harness").mkdir(parents=True)
    (authoring / "contract.json").write_text("{}")
    clones = []

    class Provider:
        def __init__(self, **kwargs):
            pass

        async def provision(self, *args, work_directory, **kwargs):
            (work_directory / "artifacts").mkdir()
            (work_directory / "artifacts" / "build.json").write_text("{}")
            return [SimpleNamespace(endpoints={})]

        async def clone_world_database(self, runtime, name):
            clones.append(name)
            return runtime

        async def reset(self, *args, **kwargs):
            pass

        async def close(self, **kwargs):
            pass

    class World:
        def read_only(self):
            return self

    class Factory:
        def __init__(self, work):
            pass

        async def create(self, *args, **kwargs):
            return World()

    class Store:
        RUNTIME_EVIDENCE = CERTIFICATION = "unused"

        def __init__(self, root):
            pass

        def read_source_model(self):
            return SimpleNamespace(actions=[])

        def write_action_certification(self, report):
            pass

    async def no_invariants(*args, **kwargs):
        return []

    for module, name, value in (
        (authoring_runtime_validation, "_database_lanes", lambda: 3),
        (authoring_runtime_validation, "_world_isolation_status", lambda output: None),
        (authoring_runtime_validation, "_world_state_digest", lambda world: "baseline"),
        (
            authoring_runtime_validation,
            "_GeneratedWorldActionAdapter",
            lambda world: None,
        ),
        (
            authoring_runtime_validation,
            "_setup_check_keys",
            lambda *args: [str(i) for i in range(10)],
        ),
        (
            authoring_runtime_validation,
            "_write_runtime_evidence",
            lambda **kwargs: None,
        ),
        (authoring_runtime_validation, "_freeze_certified_bundle", lambda *args: None),
        (process_runtime, "ProcessRuntimeProvider", Provider),
        (hosted_entrypoint, "ProcessWorldFactory", Factory),
        (certification, "GenericHarnessArtifactStore", Store),
        (
            contract,
            "AgentContract",
            SimpleNamespace(model_validate_json=lambda text: None),
        ),
        (
            tool_certification,
            "certify_tool_inventory",
            lambda *args, **kwargs: SimpleNamespace(tools=[]),
        ),
        (
            action_certification,
            "certify_actions",
            lambda *args, **kwargs: SimpleNamespace(actions=[]),
        ),
        (bundle_author_v2, "author_bundle_v2", lambda **kwargs: object()),
        (process_preflight, "preflight_bundle", lambda *args, **kwargs: None),
        (
            outbound,
            "load_capabilities",
            lambda *, unlink: SimpleNamespace(attempt_id="test", expires_at=None),
        ),
        (source_data_invariants, "author_invariants", no_invariants),
        (source_data_invariants, "check_invariants", no_invariants),
        (
            scenario_source,
            "load_scenarios",
            lambda bundle: [
                SimpleNamespace(
                    scenario_key=str(i),
                    setup=lambda world: None,
                    ready=lambda world: True,
                )
                for i in range(10)
            ],
        ),
    ):
        monkeypatch.setattr(module, name, value)
    job = SimpleNamespace(
        agent=SimpleNamespace(secret_refs={}),
        scenario_count=10,
        seed=0,
        runtime=SimpleNamespace(cpu_units=4, memory_mb=8192),
        metadata={"generic_harness_v1": True},
    )

    assert (
        asyncio.run(validate_once(job, tmp_path, authoring, secrets_path=original))
        == 10
    )
    assert sorted(set(clones)) == ["lane0", "lane1", "lane2"]


def test_local_runtime_validation_does_not_require_hosted_capabilities(
    tmp_path, monkeypatch
):
    from fi.alk.harness import (
        bundle_author_v2,
        hosted_entrypoint,
        outbound,
        process_preflight,
        process_runtime,
        scenario_source,
        source_data_invariants,
    )

    secrets = tmp_path / "secrets.json"
    secrets.write_text("{}", encoding="utf-8")
    authoring = tmp_path / "authoring"
    authoring.mkdir()

    monkeypatch.setattr(
        outbound,
        "load_capabilities",
        lambda **_kwargs: pytest.fail(
            "local validation must not load hosted capabilities"
        ),
    )

    class Provider:
        def __init__(self, **kwargs):
            assert kwargs["public_url_resolver"] is None
            assert kwargs["provider_attempt_id"] is None
            assert kwargs["provider_expires_at"] is None

        async def provision(self, *_args, **_kwargs):
            return [SimpleNamespace(endpoints={})]

        async def reset(self, *_args, **_kwargs):
            return None

        async def close(self, **_kwargs):
            return None

    class World:
        def read_only(self):
            return self

    class Factory:
        def __init__(self, _work):
            pass

        async def create(self, *_args, **_kwargs):
            return World()

    monkeypatch.setattr(process_runtime, "ProcessRuntimeProvider", Provider)
    monkeypatch.setattr(hosted_entrypoint, "ProcessWorldFactory", Factory)
    monkeypatch.setattr(
        bundle_author_v2, "author_bundle_v2", lambda **_kwargs: object()
    )
    monkeypatch.setattr(process_preflight, "preflight_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(
        scenario_source,
        "load_scenarios",
        lambda _bundle: [
            SimpleNamespace(
                scenario_key="one",
                setup=lambda _world: None,
                ready=lambda _world: True,
            )
        ],
    )

    async def author_invariants(*_args, **_kwargs):
        return []

    async def check_invariants(*_args, **_kwargs):
        return None

    monkeypatch.setattr(source_data_invariants, "author_invariants", author_invariants)
    monkeypatch.setattr(source_data_invariants, "check_invariants", check_invariants)
    job = SimpleNamespace(
        source=SimpleNamespace(kind=None),
        agent=SimpleNamespace(secret_refs={}),
        metadata={},
        scenario_count=1,
        seed=0,
    )

    assert (
        asyncio.run(
            validate_once(
                job,
                tmp_path,
                authoring,
                secrets_path=secrets,
                local_runtime=True,
            )
        )
        == 1
    )


def test_connect_only_provider_repair_preserves_external_runtime_mode(
    tmp_path, monkeypatch
):
    from fi.alk.harness import cli

    attempts = 0
    observed = []

    async def validate(*_args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeValidationError("environment", "repair me")
        return 1

    async def build(args):
        observed.append(args.external_runtime)
        return 0

    async def scenarios(_args):
        pytest.fail("an environment repair must not rewrite scenarios")

    monkeypatch.setattr(cli, "_build", build)
    monkeypatch.setattr(cli, "_scenarios", scenarios)
    job = HarnessJob(
        job_id="job-provider",
        run_id="run-provider",
        execution="hosted",
        source={"kind": "provider"},
        agent={
            "connector": "retell",
            "mode": "connect_only",
            "config": {"agent_id": "agent_test"},
        },
        scenario_count=1,
        runtime={"isolation": "dedicated_vm"},
    )

    asyncio.run(validate_and_repair(job, tmp_path, tmp_path, validate=validate))
    assert observed == [True]


@pytest.mark.parametrize("status", (400, 401, 403, 404))
def test_provider_lifecycle_definitive_rejection_is_source_owned(status):
    diagnostic = _fallback_diagnostic(
        RuntimeValidationError(
            "environment",
            "ProcessRuntimeError: provider_lifecycle/spawn_failed (agent): "
            f"provider_provision_failed: HTTP Error {status}: rejected",
        )
    )

    assert diagnostic.code == "external_service_request_rejected"
    assert diagnostic.owner.value == "source"
    assert diagnostic.retryable is False
    assert diagnostic.stage is HarnessStage.CONNECTING_AGENT


@pytest.mark.parametrize(
    "detail",
    (
        "HTTP Error 429: rate limited",
        "HTTP Error 503: unavailable",
        "ClientConnectorDNSError: name resolution failed",
    ),
)
def test_provider_lifecycle_transient_failure_is_retryable_infrastructure(detail):
    diagnostic = _fallback_diagnostic(
        RuntimeValidationError(
            "environment",
            "ProcessRuntimeError: provider_lifecycle/spawn_failed (agent): " + detail,
        )
    )

    assert diagnostic.code == "external_service_unavailable"
    assert diagnostic.owner.value == "infrastructure"
    assert diagnostic.retryable is True
    assert diagnostic.stage is HarnessStage.CONNECTING_AGENT


@pytest.mark.parametrize(
    "detail",
    (
        "GeneratedRuntimeError: generated_runtime_component_ambiguous",
        "ProcessRuntimeError: no runnable shipped entrypoint was identified",
        "docker build failed while installing dependencies",
        "agent process exited before readiness probe passed",
    ),
)
def test_runtime_construction_failure_routes_to_runtime_plan_repair(detail):
    diagnostic = _fallback_diagnostic(RuntimeValidationError("environment", detail))

    assert diagnostic.code == "generated_runtime_plan_invalid"
    assert diagnostic.owner.value == "authoring"
    assert diagnostic.repair_strategy == "inspect_repository_and_revise_runtime_plan"


def test_unknown_pre_provision_failure_routes_to_generic_runtime_inspection():
    diagnostic = _fallback_diagnostic(
        RuntimeValidationError("runtime", "an unfamiliar framework launcher failed")
    )

    assert diagnostic.code == "generated_runtime_plan_invalid"
    assert diagnostic.owner.value == "authoring"


def test_generic_validation_applies_runtime_plan_patch_and_revalidates(tmp_path):
    source = tmp_path / "source"
    authoring = tmp_path / "authoring"
    source.mkdir()
    authoring.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='agent'\nversion='1'\n")
    (source / "server.py").write_text("print('server')\n")
    contract = AgentContract(agent="example", runtime=Runtime(language="python"))
    (authoring / "contract.json").write_text(
        contract.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.BUILDING_ENVIRONMENT,
        component="runtime_validation",
        code="generated_runtime_plan_invalid",
        message="no runnable shipped entrypoint was identified",
    )
    calls = []

    async def validate(*_args):
        calls.append("validate")
        current = AgentContract.model_validate_json(
            (authoring / "contract.json").read_text(encoding="utf-8")
        )
        if not current.runtime or not current.runtime.command:
            raise RuntimeValidationError(
                "environment", "entrypoint missing", diagnostics=(diagnostic,)
            )
        return 1

    async def repair(phase, _guidance):
        assert phase == "environment"
        calls.append("repair")
        return RuntimePlanPatch(
            base_contract_hash=_contract_hash(contract),
            runtime=Runtime(language="python", command=["python", "server.py"]),
            evidence_paths=("pyproject.toml", "server.py"),
            diagnostic_codes=("generated_runtime_plan_invalid",),
            summary="Use the repository's server entrypoint.",
        )

    job = SimpleNamespace(metadata={"generic_harness_v1": True})
    asyncio.run(
        validate_and_repair(job, source, authoring, validate=validate, repair=repair)
    )

    repaired = AgentContract.model_validate_json(
        (authoring / "contract.json").read_text(encoding="utf-8")
    )
    assert repaired.runtime.command == ["python", "server.py"]
    assert calls == ["validate", "repair", "validate"]
    assert (authoring / "generic-harness" / "runtime-plan-repair-1.json").is_file()


def test_default_generic_repair_dispatches_unknown_runtime_to_repository_inspection(
    tmp_path, monkeypatch
):
    from fi.alk.harness import runtime_repair

    source = tmp_path / "source"
    authoring = tmp_path / "authoring"
    source.mkdir()
    authoring.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='agent'\nversion='1'\n")
    (source / "worker.py").write_text("print('worker')\n")
    contract = AgentContract(agent="unknown-framework", runtime=Runtime())
    (authoring / "contract.json").write_text(
        contract.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.BUILDING_ENVIRONMENT,
        component="runtime_validation",
        code="generated_runtime_plan_invalid",
        message="unfamiliar launcher",
    )
    inspected = []

    async def inspect(source_root, current_contract, diagnostics):
        inspected.append((source_root, current_contract.agent, diagnostics[0].code))
        return RuntimePlanPatch(
            base_contract_hash=_contract_hash(current_contract),
            runtime=Runtime(language="python", command=["python", "worker.py"]),
            evidence_paths=("pyproject.toml", "worker.py"),
            diagnostic_codes=("generated_runtime_plan_invalid",),
            summary="Use the discovered worker entrypoint.",
        )

    monkeypatch.setattr(runtime_repair, "request_runtime_plan_patch", inspect)

    async def validate(*_args):
        current = AgentContract.model_validate_json(
            (authoring / "contract.json").read_text(encoding="utf-8")
        )
        if not current.runtime or not current.runtime.command:
            raise RuntimeValidationError(
                "runtime", "unfamiliar launcher", diagnostics=(diagnostic,)
            )
        return 1

    job = SimpleNamespace(metadata={"generic_harness_v1": True})
    asyncio.run(validate_and_repair(job, source, authoring, validate=validate))

    assert inspected == [
        (source, "unknown-framework", "generated_runtime_plan_invalid")
    ]


def test_a_store_only_reset_is_offered_only_where_nothing_needs_respawning() -> None:
    from fi.alk.harness.process_runtime import _stores_reset_in_place

    class _Store:
        def __init__(self, strategy):
            self.baseline = type("B", (), {"strategy": strategy})()

    class _Manifest:
        def __init__(self, strategies):
            self.seed = (
                type("S", (), {"stores": [_Store(one) for one in strategies]})()
                if strategies is not None
                else None
            )

    from fi.alk.harness.process_runtime import BaselineStrategy

    assert _stores_reset_in_place(_Manifest([BaselineStrategy.TEMPLATE_DATABASE]))
    mixed = [BaselineStrategy.TEMPLATE_DATABASE, BaselineStrategy.DATADIR_COPY]
    assert not _stores_reset_in_place(_Manifest(mixed))
    assert not _stores_reset_in_place(_Manifest([]))
    assert not _stores_reset_in_place(_Manifest(None))


def test_declared_invariants_survive_a_review_that_forgot_to_finish(
    tmp_path, monkeypatch
) -> None:
    import sqlite3

    import fi.alk.harness.source_data_invariants as invariants

    source = tmp_path / "source"
    source.mkdir()
    (source / "lookup.py").write_text("return records[alias.target]\n")
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    database = sqlite3.connect(":memory:", check_same_thread=False)
    database.execute("CREATE TABLE aliases (target TEXT)")

    class World:
        def query(self, sql):
            return database.execute(sql).fetchall()

    class Stage:
        def __init__(self, spec, **kwargs):
            self.tools = {tool.name: tool.handler for tool in spec.servers["source_data"].tools}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def say(self, *args):
            await self.tools["declare_invariant"](
                {
                    "name": "aliases resolve",
                    "violations_sql": "SELECT target FROM aliases",
                    "evidence": [{"path": "lookup.py", "quote": "return records[alias.target]"}],
                }
            )

    monkeypatch.setattr(invariants, "Stage", Stage)
    checks = asyncio.run(invariants.author_invariants(source, authoring, World()))
    assert [check["name"] for check in checks] == ["aliases resolve"]


def test_setup_check_keys_change_only_with_what_a_check_depends_on(tmp_path):
    bundle = tmp_path / "bundle"
    for name in ("a", "b"):
        (bundle / "scenarios" / name).mkdir(parents=True)
        (bundle / "scenarios" / name / "setup.py").write_text(f"# {name}\n")
    (bundle / "seed").mkdir()
    (bundle / "seed" / "rows.sql").write_text("insert 1;\n")
    (bundle / "manifest.json").write_text("{}")

    def keys(invariants=(), baseline="sha256:0"):
        return _setup_check_keys(bundle, list(invariants), baseline)

    first = keys()
    assert len(set(first)) == 2
    (bundle / "manifest.json").write_text('{"digest": "new"}')
    assert keys() == first
    (bundle / "scenarios" / "b" / "setup.py").write_text("# b changed\n")
    changed = keys()
    assert changed[0] == first[0] and changed[1] != first[1]
    assert not set(keys([{"name": "x"}])) & set(changed)
    assert not set(keys(baseline="sha256:1")) & set(changed)
    (bundle / "seed" / "rows.sql").write_text("insert 2;\n")
    assert not set(keys()) & set(changed)


def test_rejected_scenario_repair_is_undone_and_repaired_again(
    tmp_path, monkeypatch
) -> None:
    from fi.alk.harness import scenario_repair, scenario_tools

    source = tmp_path / "source"
    authoring = tmp_path / "authoring"
    source.mkdir()
    authoring.mkdir()
    (source / "agent.py").write_text("agent", encoding="utf-8")
    suite = {"value": "original"}
    restored = []
    monkeypatch.setattr(scenario_tools, "load_scenarios", lambda path: suite["value"])
    monkeypatch.setattr(
        scenario_tools,
        "write_scenarios",
        lambda scenarios, path: restored.append(scenarios) or suite.update(value=scenarios),
    )

    def certify(before, after, **kwargs):
        if after == "weakened":
            raise ValueError("scenario_repair_weakened_checks: s1: check")
        return scenario_repair.ScenarioRepairReceipt.model_construct()

    monkeypatch.setattr(scenario_repair, "certify_scenario_repair", certify)
    monkeypatch.setattr(
        GenericHarnessArtifactStore, "write_scenario_repair", lambda *a, **k: None
    )
    attempts = iter(["weakened", "fixed"])
    guidance_seen = []

    async def repair(phase, guidance):
        assert phase == "scenarios"
        guidance_seen.append(guidance)
        suite["value"] = next(attempts)
        (authoring / "scenarios.txt").write_text(suite["value"], encoding="utf-8")
        return 0

    validations = []

    async def validate(*args):
        validations.append(suite["value"])
        if len(validations) == 1:
            raise RuntimeValidationError("scenarios", "bad setup")
        return 3

    job = SimpleNamespace(metadata={"generic_harness_v1": True}, scenario_count=3)
    asyncio.run(
        validate_and_repair(job, source, authoring, validate=validate, repair=repair)
    )

    assert restored == ["original"]
    assert len(guidance_seen) == 2
    assert "scenario_repair_weakened_checks" in guidance_seen[1]
    assert validations == ["original", "fixed"]


def test_scenario_repair_rejected_every_time_keeps_the_original_failure(
    tmp_path, monkeypatch
) -> None:
    from fi.alk.harness import scenario_repair, scenario_tools

    source = tmp_path / "source"
    authoring = tmp_path / "authoring"
    source.mkdir()
    authoring.mkdir()
    (source / "agent.py").write_text("agent", encoding="utf-8")
    suite = {"value": "original"}
    monkeypatch.setattr(scenario_tools, "load_scenarios", lambda path: suite["value"])
    monkeypatch.setattr(
        scenario_tools, "write_scenarios", lambda scenarios, path: suite.update(value=scenarios)
    )

    def certify(before, after, **kwargs):
        raise ValueError("scenario_repair_weakened_checks: s1: check")

    monkeypatch.setattr(scenario_repair, "certify_scenario_repair", certify)
    repairs = []

    async def repair(phase, guidance):
        repairs.append(guidance)
        suite["value"] = "weakened"
        return 0

    async def validate(*args):
        raise RuntimeValidationError("scenarios", "bad setup")

    job = SimpleNamespace(metadata={"generic_harness_v1": True}, scenario_count=3)
    with pytest.raises(ValueError, match="scenario_repair_weakened_checks"):
        asyncio.run(
            validate_and_repair(job, source, authoring, validate=validate, repair=repair)
        )
    assert len(repairs) == RepairBudgets().scenario_patches + 1
    assert suite["value"] == "original"
