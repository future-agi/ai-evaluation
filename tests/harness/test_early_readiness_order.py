import asyncio
import json
from types import SimpleNamespace

import pytest

from fi.alk.harness import authoring_entrypoint, cli, local_certified_runtime
from fi.alk.harness.authoring_runtime_validation import RuntimeValidationError
from fi.alk.harness.job import HarnessJob, HarnessStage


def job_for(source):
    return HarnessJob(
        job_id="early-readiness",
        run_id="early-readiness",
        execution="local",
        source={"kind": "local_repository", "local_path": str(source)},
        agent={"connector": "auto"},
        scenario_count=2,
        metadata={"generic_harness_v1": True},
    )


@pytest.mark.parametrize("fails", [False, True])
def test_real_readiness_gates_public_generation(tmp_path, monkeypatch, fails):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    job = job_for(source)
    observed = []

    async def understand(args):
        observed.append("understand")
        (output / "contract.json").write_text(
            json.dumps({"agent": "example", "modality": "chat"})
        )
        return 0

    async def build(args):
        observed.append("build")
        return 0

    async def readiness(candidate, candidate_source, candidate_authoring):
        observed.append("internal-chat")
        assert candidate.scenario_count == 2
        assert candidate_source == source
        assert candidate_authoring == output
        assert not (output / "scenarios").exists()
        if fails:
            raise RuntimeValidationError("environment", "chat failed")

    async def scenarios(args):
        observed.append("public-scenarios")
        assert args.count == 2
        return 0

    monkeypatch.setattr(cli, "_understand", understand)
    monkeypatch.setattr(cli, "_build", build)
    monkeypatch.setattr(cli, "_scenarios", scenarios)
    monkeypatch.setattr(cli, "load_written", lambda _: [object(), object()])
    monkeypatch.setattr("fi.alk.harness.provision.stop", lambda _: None)
    args = SimpleNamespace(
        path=str(source),
        out=str(output),
        name="example",
        kind="repo",
        count=2,
        model=None,
        authoring_only=True,
        job=job,
        runtime_readiness=readiness,
    )
    if fails:
        with pytest.raises(RuntimeValidationError, match="chat failed"):
            asyncio.run(cli._auto(args))
        assert observed == ["understand", "build", "internal-chat"]
    else:
        assert asyncio.run(cli._auto(args)) == 0
        assert observed == ["understand", "build", "internal-chat", "public-scenarios"]
    assert job.scenario_count == 2
    assert not (output / "runtime-validation.json").exists()


@pytest.mark.parametrize("runtime_enabled", [False, True])
def test_hosted_execution_capability_controls_early_gate(
    tmp_path, monkeypatch, runtime_enabled
):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    job = job_for(source)
    job_path = tmp_path / "job.json"
    job_path.write_text(job.model_dump_json())
    observed = []

    async def validate(candidate, candidate_source, candidate_authoring, **kwargs):
        observed.append("readiness" if kwargs.get("readiness_only") else "final")
        assert candidate.scenario_count == 2

    async def auto(args):
        output.mkdir()
        if runtime_enabled:
            await args.runtime_readiness(job, source, output)
        else:
            assert args.runtime_readiness is None
        observed.append("public-scenarios")
        return 0

    monkeypatch.setattr(authoring_entrypoint, "_auto", auto)
    monkeypatch.setattr(
        authoring_entrypoint, "load_written", lambda _: [object(), object()]
    )
    monkeypatch.setattr(
        "fi.alk.harness.authoring_runtime_validation.validate_and_repair", validate
    )
    assert (
        authoring_entrypoint.main(
            [str(job_path), "--source", str(source), "--output", str(output)],
            validate_runtime=runtime_enabled,
        )
        == 0
    )
    assert observed == (
        ["readiness", "public-scenarios", "final"]
        if runtime_enabled
        else ["public-scenarios"]
    )


def test_local_readiness_uses_private_scoped_credential_handoff(tmp_path, monkeypatch):
    job = job_for(tmp_path)
    secrets_path = None

    async def once(candidate, source, authoring, **kwargs):
        nonlocal secrets_path
        secrets_path = kwargs["secrets_path"]
        assert kwargs["local_runtime"] is True
        assert kwargs["readiness_only"] is True
        assert secrets_path.stat().st_mode & 0o777 == 0o600
        assert json.loads(secrets_path.read_text()) == {"TARGET": "private-value"}
        return 0

    async def repair(candidate, source, authoring, **kwargs):
        assert kwargs["readiness_only"] is True
        await kwargs["validate"](candidate, source, authoring)

    monkeypatch.delenv("ALK_SANDBOX_UPLOAD_ROOT", raising=False)
    monkeypatch.setattr(
        local_certified_runtime,
        "_runtime_values",
        lambda _: {"TARGET": "private-value"},
    )
    monkeypatch.setattr(local_certified_runtime, "validate_once", once)
    monkeypatch.setattr(local_certified_runtime, "validate_and_repair", repair)
    asyncio.run(
        local_certified_runtime.check_local_runtime_readiness(
            job, tmp_path, tmp_path / "output"
        )
    )
    assert secrets_path is not None and not secrets_path.exists()


def test_executor_reports_early_failure_without_running_suite(tmp_path, monkeypatch):
    from fi.alk.harness.executor import HarnessExecutor

    async def auto(args):
        assert (
            args.runtime_readiness
            is local_certified_runtime.check_local_runtime_readiness
        )
        raise RuntimeValidationError("runtime", "installation failed")

    async def unexpected(*args, **kwargs):
        pytest.fail("suite execution must not run after failed early readiness")

    monkeypatch.setattr(cli, "_auto", auto)
    monkeypatch.setattr(local_certified_runtime, "run_local_certified", unexpected)
    result = asyncio.run(
        HarnessExecutor().run(
            job_for(tmp_path), source=tmp_path, output=tmp_path / "output"
        )
    )
    assert result.stage is HarnessStage.FAILED
    assert result.completed_scenarios == 0
    assert result.failure.code == "generic_runtime_validation_failed"
