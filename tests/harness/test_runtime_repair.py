from __future__ import annotations

import json

import pytest

from fi.alk.harness.contract import AgentContract, Runtime, RuntimeInterface
from fi.alk.harness.runtime_repair import (
    RuntimePlanPatch,
    _contract_hash,
    apply_runtime_plan_patch,
    validate_runtime_plan_patch,
)


def _patch(contract: AgentContract, **changes) -> RuntimePlanPatch:
    runtime = Runtime(
        language="python",
        workdir="service",
        command=["python", "app.py"],
        interface=RuntimeInterface(kind="http", port=8080, path="/chat"),
        **changes,
    )
    return RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=runtime,
        evidence_paths=("service/pyproject.toml", "service/app.py"),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Run the HTTP service declared by the submitted component.",
    )


def test_runtime_patch_changes_only_runtime_contract_metadata(tmp_path) -> None:
    source = tmp_path / "source"
    service = source / "service"
    service.mkdir(parents=True)
    (service / "pyproject.toml").write_text("[project]\nname='agent'\nversion='1'\n")
    (service / "app.py").write_text("# entrypoint\n")
    contract = AgentContract(
        agent="support-agent",
        hard_constraints=["Never disclose secrets"],
        runtime=Runtime(language="python"),
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(contract.model_dump_json(indent=2) + "\n")

    repaired = apply_runtime_plan_patch(
        source,
        contract_path,
        _patch(contract),
        allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
    )

    assert repaired.agent == contract.agent
    assert repaired.hard_constraints == contract.hard_constraints
    assert repaired.runtime.workdir == "service"
    assert repaired.runtime.command == ["python", "app.py"]
    assert json.loads(contract_path.read_text())["runtime"]["workdir"] == "service"
    assert (service / "app.py").read_text() == "# entrypoint\n"


def test_runtime_patch_rejects_missing_or_outside_evidence(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "service").mkdir(parents=True)
    contract = AgentContract(agent="agent", runtime=Runtime(language="python"))
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=Runtime(language="python", workdir="service"),
        evidence_paths=("service/missing.py",),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Use service.",
    )

    with pytest.raises(ValueError, match="evidence_missing"):
        validate_runtime_plan_patch(
            source,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )


def test_runtime_patch_rejects_stale_contract(tmp_path) -> None:
    source = tmp_path / "source"
    service = source / "service"
    service.mkdir(parents=True)
    (service / "pyproject.toml").write_text("[project]\n")
    (service / "app.py").write_text("# entrypoint\n")
    contract = AgentContract(agent="agent", runtime=Runtime(language="python"))
    patch = _patch(contract).model_copy(
        update={"base_contract_hash": "sha256:" + "0" * 64}
    )

    with pytest.raises(ValueError, match="base_contract_hash_mismatch"):
        validate_runtime_plan_patch(
            source,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )


def test_runtime_patch_rejects_model_authored_shell_commands(tmp_path) -> None:
    source = tmp_path / "source"
    service = source / "service"
    service.mkdir(parents=True)
    (service / "pyproject.toml").write_text("[project]\n")
    (service / "app.py").write_text("# entrypoint\n")
    contract = AgentContract(agent="agent", runtime=Runtime(language="python"))
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=Runtime(
            language="python",
            workdir="service",
            command=["bash", "-c", "python app.py"],
        ),
        evidence_paths=("service/pyproject.toml", "service/app.py"),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Shell launch.",
    )

    with pytest.raises(ValueError, match="shell_command_forbidden"):
        validate_runtime_plan_patch(
            source,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )


def test_compose_repair_path_is_relative_to_repository_not_workdir(tmp_path):
    (tmp_path / "service").mkdir()
    (tmp_path / "deployment").mkdir()
    (tmp_path / "service/app.py").write_text("print('source')")
    (tmp_path / "service/pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1"\n'
    )
    (tmp_path / "deployment/custom.yaml").write_text(
        "services:\n  dialogue:\n    build: ../service\n"
    )
    contract = AgentContract(agent="fixture")
    patch = _patch(
        contract,
        execution="compose",
        compose_file="deployment/custom.yaml",
        service="dialogue",
    )
    repaired = validate_runtime_plan_patch(
        tmp_path,
        contract,
        patch,
        allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
    )
    assert repaired.runtime.compose_file == "deployment/custom.yaml"
    assert repaired.runtime.service == "dialogue"
