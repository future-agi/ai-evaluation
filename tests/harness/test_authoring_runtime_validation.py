import asyncio
import json
from types import SimpleNamespace

import pytest

from fi.alk.harness.authoring_runtime_validation import (
    RuntimeValidationError,
    _generic_candidate_hash,
    _write_generic_certificate,
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
        agent=SimpleNamespace(secret_refs={}), scenario_count=2, seed=0
    )
    if bad_setup:
        with pytest.raises(RuntimeValidationError) as error:
            asyncio.run(validate_once(job, tmp_path, authoring, secrets_path=original))
        assert "target-secret-value" not in str(error.value)
        assert "0: setup:" in str(error.value)
        assert "1: setup:" in str(error.value)
        assert calls == ["provision", "reset", "setup", "reset", "setup", "close"]
    else:
        assert (
            asyncio.run(validate_once(job, tmp_path, authoring, secrets_path=original))
            == 2
        )
        assert calls == [
            "provision",
            "reset",
            "setup",
            "reset",
            "setup",
            "reset",
            "reset",
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
