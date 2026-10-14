from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fi.alk.harness import authoring_entrypoint, cli
from fi.alk.harness.job import HarnessJob


def _job(source: Path, *, required: bool = True) -> HarnessJob:
    return HarnessJob(
        job_id="workflow",
        run_id="workflow",
        execution="local",
        source={"kind": "local_repository", "local_path": str(source)},
        agent={"connector": "auto"},
        scenario_count=2,
        require_tool_workflow=required,
        guidance=["Test a failed lookup followed by recovery."],
    )


def _contract(*, executable: bool) -> dict:
    return {
        "agent": "example",
        "conversational": True,
        "tools": [{"name": "lookup", "args": []}] if executable else [],
        "tool_entrypoints": [
            {"tool": "lookup", "mode": "import", "module": "api", "callable": "lookup"}
        ]
        if executable
        else [],
    }


@pytest.mark.parametrize(
    "required,executable,expected",
    [(True, False, 1), (True, True, 0), (False, False, 0)],
)
def test_job_entrypoint_keeps_intent_through_stages_and_count_repair(
    tmp_path, monkeypatch, required, executable, expected
):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    job = _job(source, required=required)
    job_path = tmp_path / "job.json"
    job_path.write_text(job.model_dump_json())
    observed = []
    saved = []

    async def understand(args):
        observed.append(("understand", tuple(args.guidance)))
        (output / "contract.json").write_text(
            json.dumps(_contract(executable=executable))
        )
        return 0

    async def build(args):
        observed.append(("environment", tuple(args.guidance)))
        return 0

    async def scenarios(args):
        observed.append(("scenarios", tuple(args.guidance)))
        saved.append(object())
        return 0

    monkeypatch.setattr(cli, "_understand", understand)
    monkeypatch.setattr(cli, "_build", build)
    monkeypatch.setattr(cli, "_scenarios", scenarios)
    monkeypatch.setattr(cli, "load_written", lambda _: saved)
    monkeypatch.setattr(authoring_entrypoint, "load_written", lambda _: saved)
    monkeypatch.setattr("fi.alk.harness.provision.stop", lambda _: False)
    assert (
        authoring_entrypoint.main(
            [str(job_path), "--source", str(source), "--output", str(output)]
        )
        == expected
    )
    assert [label for label, _ in observed] == [
        "understand",
        "environment",
        "scenarios",
        "scenarios",
    ]
    for _, guidance in observed:
        assert "Test a failed lookup followed by recovery." in guidance
        assert any("require_tool_workflow" in item for item in guidance) is required


def test_declared_tool_without_executable_entry_is_rejected(tmp_path):
    from fi.alk.harness.contract import AgentContract, validate_tool_workflow

    contract = AgentContract.model_validate(
        {"agent": "example", "tools": [{"name": "lookup"}]}
    )
    with pytest.raises(ValueError, match="required_tool_workflow_unavailable"):
        validate_tool_workflow(contract)


def test_adjustment_rewind_preserves_original_requirements(tmp_path, monkeypatch):
    import asyncio

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    inbox = tmp_path / "adjustments.jsonl"
    observed = []

    async def understand(args):
        observed.append(tuple(args.guidance))
        (output / "contract.json").write_text(json.dumps(_contract(executable=True)))
        if len(observed) == 1:
            inbox.write_text(
                json.dumps(
                    {
                        "adjustment_id": "recheck",
                        "target_stage": "understand",
                        "instruction": "Recheck the configured endpoint.",
                    }
                )
                + "\n"
            )
        return 0

    async def ok(args):
        return 0

    monkeypatch.setattr(cli, "_understand", understand)
    monkeypatch.setattr(cli, "_build", ok)
    monkeypatch.setattr(cli, "_scenarios", ok)
    monkeypatch.setattr(cli, "load_written", lambda _: [object(), object()])
    monkeypatch.setattr("fi.alk.harness.provision.stop", lambda _: False)
    assert (
        asyncio.run(
            cli._auto(
                SimpleNamespace(
                    path=str(source),
                    name="example",
                    kind="repo",
                    out=str(output),
                    count=2,
                    model=None,
                    authoring_only=True,
                    job=_job(source),
                    adjustments_path=str(inbox),
                    guidance=["Keep existing lookup identifiers."],
                )
            )
        )
        == 0
    )
    assert len(observed) == 2
    for guidance in observed:
        assert "Test a failed lookup followed by recovery." in guidance
        assert "Keep existing lookup identifiers." in guidance
        assert any("require_tool_workflow" in item for item in guidance)
    assert "Recheck the configured endpoint." in observed[1]


def test_runtime_repair_keeps_job_intent(tmp_path):
    import asyncio
    from fi.alk.harness.authoring_runtime_validation import (
        RuntimeValidationError,
        validate_and_repair,
    )

    attempts = 0
    observed = []

    async def validate(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeValidationError("environment", "startup failed")
        return 2

    async def repair(phase, guidance):
        observed.append(guidance)
        return 0

    asyncio.run(
        validate_and_repair(
            _job(tmp_path), tmp_path, tmp_path, validate=validate, repair=repair
        )
    )
    assert len(observed) == 1
    assert "Test a failed lookup followed by recovery." in observed[0]
    assert "require_tool_workflow" in observed[0]


@pytest.mark.parametrize(
    "mode,fields,accepted",
    [
        ("unreachable", {}, False),
        ("import", {"module": "api"}, False),
        ("construct", {"module": "api", "callable": "lookup"}, False),
        (
            "construct",
            {"module": "api", "callable": "lookup", "factory": "Client()"},
            True,
        ),
        ("service", {"service": "catalog"}, False),
        ("service", {"service": "catalog", "endpoint": "/lookup"}, True),
    ],
)
def test_executable_tool_seams(mode, fields, accepted):
    from fi.alk.harness.contract import AgentContract, validate_tool_workflow

    contract = AgentContract.model_validate(
        {
            "agent": "example",
            "tools": [{"name": "lookup"}],
            "tool_entrypoints": [{"tool": "lookup", "mode": mode, **fields}],
        }
    )
    if accepted:
        validate_tool_workflow(contract)
    else:
        with pytest.raises(ValueError, match="required_tool_workflow_unavailable"):
            validate_tool_workflow(contract)


def test_runtime_validation_rechecks_tool_requirement(tmp_path):
    import asyncio
    from fi.alk.harness.authoring_runtime_validation import (
        RuntimeValidationError,
        validate_once,
    )

    (tmp_path / "contract.json").write_text(json.dumps(_contract(executable=False)))
    with pytest.raises(
        RuntimeValidationError, match="required_tool_workflow_unavailable"
    ):
        asyncio.run(validate_once(_job(tmp_path), tmp_path, tmp_path))


def test_job_requirement_is_in_serialized_schema(tmp_path):
    job = _job(tmp_path)
    restored = HarnessJob.model_validate_json(job.model_dump_json())
    assert restored.require_tool_workflow is True
    assert restored.guidance == ["Test a failed lookup followed by recovery."]
    assert (
        HarnessJob.model_json_schema()["properties"]["require_tool_workflow"]["default"]
        is False
    )
