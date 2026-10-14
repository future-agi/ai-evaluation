from pathlib import Path

import pytest

from fi.alk.harness.contract import AgentContract, Runtime
from fi.alk.harness.generated_runtime import (
    GeneratedRuntimeError,
    local_install_extra_claims,
    select_python_extras,
    validate_local_install_extras,
)
from fi.alk.harness.runtime_repair import (
    RuntimePlanPatch,
    _contract_hash,
    validate_runtime_plan_patch,
)


def workspace(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='shell'\nversion='1'\n[tool.uv.workspace]\nmembers=['sdk', 'tools']\n"
    )
    for name in ("sdk", "tools"):
        (root / name).mkdir()
        (root / name / "pyproject.toml").write_text(
            f"[project]\nname='{name}'\nversion='1'\n"
            + (
                "[project.optional-dependencies]\ncloud_sdk=['client']\n"
                if name == "sdk"
                else ""
            )
        )
    (root / "server.py").write_text("print('source')")


INSTALL = "pip install -e ./sdk[cloud-sdk] -e tools"


@pytest.mark.parametrize(
    "install",
    [
        "install cloud please",
        "pip install cloud[cloud-sdk]",
        "pip install -e sdk[cloud-sdk] && echo done",
        "pip install -e sdk[cloud-sdk] -r requirements.txt",
        "pip install -e sdk[cloud-sdk] broken",
        'pip install -e "broken',
        "pip install -e https://example.org/sdk[cloud-sdk]",
        "pip install -e ../sdk[cloud-sdk]",
        "pip install -e /sdk[cloud-sdk]",
        "pip install -e absent[cloud-sdk]",
        "pip install -e sdk[unknown]",
    ],
)
def test_unsupported_install_text_does_not_select_dependencies(tmp_path, install):
    workspace(tmp_path)
    assert local_install_extra_claims(tmp_path, install) == ()
    validate_local_install_extras(tmp_path, install, [])


@pytest.mark.parametrize("collision", ["root", "excluded", "duplicate", "symlink"])
def test_documentary_claim_requires_exact_selectable_owner(tmp_path, collision):
    workspace(tmp_path)
    manifest = tmp_path / "pyproject.toml"
    if collision == "root":
        with manifest.open("a") as stream:
            stream.write("[project.optional-dependencies]\ncloud_sdk=['other']\n")
    elif collision == "excluded":
        with manifest.open("a") as stream:
            stream.write("exclude=['sdk']\n")
        with (tmp_path / "tools/pyproject.toml").open("a") as stream:
            stream.write("[project.optional-dependencies]\ncloud_sdk=['other']\n")
    elif collision == "duplicate":
        with (tmp_path / "tools/pyproject.toml").open("a") as stream:
            stream.write("[project.optional-dependencies]\ncloud_sdk=['other']\n")
    else:
        (tmp_path / "sdk/pyproject.toml").unlink()
        outside = tmp_path.parent / "outside.toml"
        outside.write_text("[project.optional-dependencies]\ncloud_sdk=['other']\n")
        (tmp_path / "sdk/pyproject.toml").symlink_to(outside)
    if collision == "symlink":
        assert local_install_extra_claims(tmp_path, INSTALL) == ()
    else:
        with pytest.raises(
            GeneratedRuntimeError, match="extra_(ambiguous|owner_mismatch)"
        ):
            validate_local_install_extras(tmp_path, INSTALL, ["cloud-sdk"])


@pytest.mark.parametrize("compose", [False, True])
@pytest.mark.parametrize("relocate", [False, True])
def test_patch_preserves_original_claim_owner(tmp_path, compose, relocate):
    workspace(tmp_path / "component")
    workspace(tmp_path / "other")
    options = dict(
        language="python",
        execution="native",
        workdir="component",
        command=["python", "server.py"],
        install=INSTALL,
        interface={"kind": "http", "port": 8080, "path": "/chat"},
    )
    if compose:
        (tmp_path / "compose.yaml").write_text(
            "services:\n  api:\n    build: component\n    command: [python, server.py]\n"
            "  other:\n    build: other\n    command: [python, server.py]\n"
        )
        options.update(execution="compose", compose_file="compose.yaml", service="api")
    contract = AgentContract(agent="test", runtime=Runtime(**options))
    replacement = contract.runtime.model_copy(update={"install": "", "extras": []})
    if relocate:
        replacement = replacement.model_copy(
            update={"workdir": "other", **({"service": "other"} if compose else {})}
        )
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=replacement,
        evidence_paths=("component/pyproject.toml",),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Select dependency.",
    )
    with pytest.raises(GeneratedRuntimeError, match="extra_(omitted|owner_mismatch)"):
        validate_runtime_plan_patch(
            tmp_path,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )
    patch = patch.model_copy(
        update={"runtime": replacement.model_copy(update={"extras": ["cloud-sdk"]})}
    )
    if relocate:
        with pytest.raises(GeneratedRuntimeError, match="extra_owner_mismatch"):
            validate_runtime_plan_patch(
                tmp_path,
                contract,
                patch,
                allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
            )
    else:
        before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        repaired = validate_runtime_plan_patch(
            tmp_path,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )
        assert repaired.runtime.extras == ["cloud-sdk"]
        assert repaired.model_dump(exclude={"runtime"}) == contract.model_dump(
            exclude={"runtime"}
        )
        assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_unknown_and_requirements_only_extras_still_fail(tmp_path):
    (tmp_path / "requirements.txt").write_text("client")
    with pytest.raises(GeneratedRuntimeError, match="extra_unknown"):
        select_python_extras(tmp_path, ["cloud"])


def test_repair_schema_explains_executable_extras():
    fields = RuntimePlanPatch.model_json_schema()["$defs"]["Runtime"]["properties"]
    assert "never executed" in fields["install"]["description"]
    assert "Executable selection" in fields["extras"]["description"]


@pytest.mark.parametrize("compose", [False, True])
def test_selected_component_build_matches_certified_extras(tmp_path, compose):
    from fi.alk.harness.bundle_author_v2 import (
        BundleAuthorError,
        resolve_environment_plan,
    )
    from fi.alk.harness.job import HarnessJob

    workspace(tmp_path / "component")
    runtime = Runtime(
        language="python",
        execution="native",
        workdir="component",
        command=["python", "server.py"],
        install=INSTALL,
        interface={"kind": "http", "port": 8080, "path": "/chat"},
    )
    if compose:
        (tmp_path / "compose.yaml").write_text(
            "services:\n  api:\n    build: component\n    command: [python, server.py]\n"
        )
        runtime = runtime.model_copy(
            update={
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "api",
            }
        )
    job = HarnessJob.model_validate(
        {
            "job_id": "extra-test",
            "run_id": "extra-test",
            "execution": "hosted",
            "source": {"kind": "archive", "archive_artifact_id": "fixture"},
            "agent": {"connector": "http"},
            "scenario_count": 1,
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": 2,
                "memory_mb": 4096,
                "parallelism": 1,
            },
        }
    )
    with pytest.raises(BundleAuthorError, match="generated_runtime_extra_omitted"):
        resolve_environment_plan(
            tmp_path,
            job,
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime=runtime.model_dump(),
        )
    runtime = runtime.model_copy(update={"extras": ["cloud-sdk"]})
    plan = resolve_environment_plan(
        tmp_path,
        job,
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=runtime.model_dump(),
    )
    control = next(p for p in plan.processes if p.name == plan.control_service)
    command = control.build_commands[0]
    assert command[command.index("--all-packages") :] == [
        "--all-packages",
        "--extra",
        *runtime.extras,
    ]
    assert control.working_directory == "component"


def test_discovered_component_preserves_claim_in_patch(tmp_path):
    workspace(tmp_path / "component")
    (tmp_path / "component/server.py").rename(tmp_path / "component/agent.py")
    contract = AgentContract(agent="test", runtime=Runtime(install=INSTALL))
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=Runtime(),
        evidence_paths=("component/pyproject.toml",),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Keep component.",
    )
    with pytest.raises(GeneratedRuntimeError, match="generated_runtime_extra_omitted"):
        validate_runtime_plan_patch(
            tmp_path,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )


@pytest.mark.parametrize(
    "install", ["npm install", "Install dependencies as documented"]
)
def test_unsupported_prose_does_not_require_python_discovery_for_patch(
    tmp_path, install
):
    (tmp_path / "package.json").write_text("{}")
    contract = AgentContract(agent="test", runtime=Runtime(install=install))
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=Runtime(
            language="node", execution="native", command=["node", "server"]
        ),
        evidence_paths=("package.json",),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Use native server.",
    )
    assert (
        validate_runtime_plan_patch(
            tmp_path,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        ).runtime.language
        == "node"
    )


def test_invalid_original_selection_does_not_block_runtime_repair(tmp_path):
    workspace(tmp_path)
    contract = AgentContract(
        agent="test",
        runtime=Runtime(
            execution="compose",
            compose_file="missing.yaml",
            service="missing",
            install=INSTALL,
        ),
    )
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=Runtime(
            execution="native", command=["python", "server.py"], extras=["cloud-sdk"]
        ),
        evidence_paths=("pyproject.toml",),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Use native server.",
    )
    assert validate_runtime_plan_patch(
        tmp_path,
        contract,
        patch,
        allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
    ).runtime.extras == ["cloud-sdk"]


def test_excluded_only_extra_is_not_an_executable_claim(tmp_path):
    workspace(tmp_path)
    with (tmp_path / "pyproject.toml").open("a") as stream:
        stream.write("exclude=['sdk']\n")
    assert local_install_extra_claims(tmp_path, INSTALL) == ()
    with pytest.raises(GeneratedRuntimeError, match="generated_runtime_extra_unknown"):
        validate_local_install_extras(tmp_path, INSTALL, [])


@pytest.mark.parametrize("excluded", ["sdk", "sdk/pyproject.toml"])
@pytest.mark.parametrize("compiler", ["generated", "bundle"])
def test_compilers_reject_extra_owner_removed_from_context(tmp_path, excluded, compiler):
    from fi.alk.harness.bundle_author_v2 import BundleAuthorError, _plan_python
    from fi.alk.harness.generated_runtime import detect_generated_runtime

    workspace(tmp_path)
    runtime = Runtime(language="python", execution="native", command=["python", "server.py"], extras=["cloud-sdk"], install=INSTALL, context_excludes=[excluded])
    with pytest.raises((GeneratedRuntimeError, BundleAuthorError), match="extra_unknown"):
        if compiler == "generated":
            detect_generated_runtime(tmp_path, runtime)
        else:
            _plan_python(tmp_path, name="agent", root=tmp_path, entry="server.py", control=True, needs_secrets=False, extras=runtime.extras, install=runtime.install, context_excludes=tuple(runtime.context_excludes))
