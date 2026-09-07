import asyncio
import json
from types import SimpleNamespace

import pytest

from fi.alk.harness.authoring_runtime_validation import (
    RuntimeValidationError,
    validate_and_repair,
    validate_once,
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
