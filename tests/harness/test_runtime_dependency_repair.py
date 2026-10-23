import pytest

from fi.alk.harness.bundle_author_v2 import BundleAuthorError, resolve_environment_plan
from fi.alk.harness.contract import (
    AgentContract,
    Dependency,
    Reached,
    Runtime,
    RuntimeInterface,
)
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.runtime_repair import (
    RuntimePlanPatch,
    _contract_hash,
    apply_runtime_plan_patch,
)


@pytest.fixture
def repair_case(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("print('agent')\n")
    (source / "pyproject.toml").write_text("[project]\nname='agent'\nversion='1'\n")
    loader = source / "component/src/client/config.py"
    loader.parent.mkdir(parents=True)
    loader.write_text("def load():\n    return {}\n")
    contract = AgentContract(
        agent="fixture",
        hard_constraints=["Preserve me"],
        runtime=Runtime(
            language="python",
            execution="native",
            command=["python", "app.py"],
            interface=RuntimeInterface(kind="http", port=8080, path="/chat"),
        ),
        runtime_dependencies=[
            Dependency(name="", kind="service", engine="remote-client"),
            Dependency(
                name="",
                kind="service",
                engine="other-client",
                reached=Reached(host="configured"),
            ),
        ],
    )
    path = tmp_path / "contract.json"
    path.write_text(contract.model_dump_json())
    replacement = contract.model_copy(deep=True).runtime_dependencies
    replacement[0].reached.loader_module = "client.config"
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=contract.runtime,
        runtime_dependencies=replacement,
        evidence_paths=("component/src/client/config.py",),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Record the source loader.",
    )
    return source, path, contract, patch


def apply(case, patch=None, codes=None):
    source, path, _, original_patch = case
    return apply_runtime_plan_patch(
        source,
        path,
        patch or original_patch,
        allowed_diagnostic_codes=codes or {"generated_runtime_plan_invalid"},
    )


def test_public_apply_repairs_anonymous_dependency_and_compiler_progresses(repair_case):
    source, path, contract, _ = repair_case
    job = HarnessJob.model_validate(
        dict(
            job_id="test-job",
            run_id="test-run",
            execution="hosted",
            source=dict(kind="archive", archive_artifact_id="source-1"),
            agent=dict(connector="auto"),
            scenario_count=1,
            runtime=dict(
                isolation="dedicated_vm", cpu_units=2, memory_mb=4096, parallelism=1
            ),
        )
    )

    def compile_contract(value):
        return resolve_environment_plan(
            source,
            job,
            contract_modality="chat",
            contract_runtime=value.runtime.model_dump(),
            runtime_dependencies=[
                item.model_dump() for item in value.runtime_dependencies
            ],
        )

    with pytest.raises(BundleAuthorError, match="has no configured connection seam"):
        compile_contract(contract)
    source_bytes = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    repaired = apply(repair_case)
    plan = compile_contract(repaired)
    assert plan.control_service == "agent"
    assert repaired.runtime_dependencies[0].reached.loader_module == "client.config"
    expected = contract.model_dump()
    expected["runtime_dependencies"][0]["reached"]["loader_module"] = "client.config"
    assert repaired.model_dump() == expected
    assert AgentContract.model_validate_json(path.read_text()) == repaired
    assert {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    } == source_bytes


@pytest.mark.parametrize(
    "change",
    [
        "remove",
        "reorder",
        "name",
        "engine",
        "kind",
        "what",
        "used_by",
        "version",
        "dsn_env",
        "host",
        "config_key",
        "port",
        "database",
        "user",
        "password_from",
        "loader_function",
        "unchanged",
        "configured",
        "engine_empty",
    ],
)
def test_dependency_patch_rejects_changes_outside_missing_loader(repair_case, change):
    source, path, contract, patch = repair_case
    dependencies = patch.model_copy(deep=True).runtime_dependencies
    if change == "remove":
        dependencies.pop()
    elif change == "reorder":
        dependencies.reverse()
    elif change == "unchanged":
        dependencies = contract.runtime_dependencies
    elif change == "configured":
        dependencies[1].reached.loader_module = "client.config"
    elif change == "engine_empty":
        contract.runtime_dependencies[0].engine = ""
        path.write_text(contract.model_dump_json())
        dependencies[0].engine = ""
        patch = patch.model_copy(
            update={"base_contract_hash": _contract_hash(contract)}
        )
    elif change in Dependency.model_fields:
        setattr(
            dependencies[0], change, ["changed"] if change == "used_by" else "changed"
        )
    else:
        setattr(
            dependencies[0].reached, change, 1234 if change == "port" else "changed"
        )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="runtime_repair_dependency_"):
        apply(
            repair_case, patch.model_copy(update={"runtime_dependencies": dependencies})
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "allowed,claimed",
    [
        ({"chat_ingress_failed"}, ("chat_ingress_failed",)),
        (
            {"generated_runtime_plan_invalid", "chat_ingress_failed"},
            ("generated_runtime_plan_invalid",),
        ),
        (
            {"generated_runtime_plan_invalid", "other"},
            ("generated_runtime_plan_invalid", "other"),
        ),
    ],
)
@pytest.mark.parametrize("unchanged", [False, True])
def test_dependency_field_rejected_for_readiness_or_mixed_codes(
    repair_case, allowed, claimed, unchanged
):
    _, path, contract, patch = repair_case
    patch = patch.model_copy(
        update={
            "diagnostic_codes": claimed,
            "runtime_dependencies": contract.runtime_dependencies
            if unchanged
            else patch.runtime_dependencies,
        }
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="dependency_diagnostic_not_allowed"):
        apply(repair_case, patch, allowed)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "module,evidence",
    [
        ("invented", "component/src/client/config.py"),
        ("client.config", "component/src/client"),
        ("../client.config", "component/src/client/config.py"),
        ("client/config", "component/src/client/config.py"),
        ("", "component/src/client/config.py"),
    ],
)
def test_dependency_loader_needs_matching_regular_python_file(
    repair_case, module, evidence
):
    _, path, _, patch = repair_case
    patch = patch.model_copy(deep=True, update={"evidence_paths": (evidence,)})
    patch.runtime_dependencies[0].reached.loader_module = module
    before = path.read_bytes()
    with pytest.raises(
        ValueError, match="dependency_(module_|requires_missing_loader)"
    ):
        apply(repair_case, patch)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "location,module",
    [
        ("client/config.py", "client.config"),
        ("component/src/client/__init__.py", "client"),
    ],
)
def test_dependency_loader_accepts_module_and_package_paths(
    repair_case, location, module
):
    source, _, _, patch = repair_case
    file = source / location
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text('raise RuntimeError("must not import source")\n')
    patch = patch.model_copy(deep=True, update={"evidence_paths": (location,)})
    patch.runtime_dependencies[0].reached.loader_module = module
    assert (
        apply(repair_case, patch).runtime_dependencies[0].reached.loader_module
        == module
    )


def test_dependency_evidence_symlink_cannot_escape_source(repair_case):
    source, path, _, patch = repair_case
    outside = source.parent / "config.py"
    outside.write_text("")
    loader = source / patch.evidence_paths[0]
    loader.unlink()
    loader.symlink_to(outside)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="evidence_escapes_repository"):
        apply(repair_case)
    assert path.read_bytes() == before


def test_old_patch_without_dependency_field_preserves_dependencies(repair_case):
    _, _, contract, patch = repair_case
    raw = patch.model_dump()
    del raw["runtime_dependencies"]
    repaired = apply(repair_case, RuntimePlanPatch.model_validate(raw))
    assert repaired.runtime_dependencies == contract.runtime_dependencies


@pytest.mark.parametrize("path", ["../config.py", "/tmp/config.py"])
def test_dependency_evidence_requires_portable_paths(repair_case, path):
    raw = repair_case[3].model_dump()
    raw["evidence_paths"] = [path]
    with pytest.raises(ValueError, match="evidence_path_must_be_relative"):
        RuntimePlanPatch.model_validate(raw)


def test_submission_forwards_dependency_patch(repair_case, monkeypatch):
    import asyncio
    from fi.alk.harness import runtime_repair
    from fi.alk.harness.diagnostics import HarnessDiagnostic
    from fi.alk.harness.job import HarnessStage

    source, _, contract, patch = repair_case

    class SubmittingStage:
        def __init__(self, spec, **kwargs):
            self.spec = spec
            assert (
                "runtime_dependencies"
                in spec.servers["runtime_repair"].tools[0].input_schema["properties"]
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def say(self, briefing, *, on_event=None):
            submit = self.spec.servers["runtime_repair"].tools[0].handler
            result = await submit(
                patch.model_dump(
                    mode="json", exclude={"base_contract_hash", "diagnostic_codes"}
                )
            )
            assert not result.get("is_error"), result

    monkeypatch.setattr(runtime_repair, "Stage", SubmittingStage)
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.CONNECTING_AGENT,
        component="runtime",
        code="generated_runtime_plan_invalid",
        message="missing connection seam",
    )
    result = asyncio.run(
        runtime_repair.request_runtime_plan_patch(source, contract, (diagnostic,))
    )
    assert result.runtime_dependencies == patch.runtime_dependencies
    assert result.base_contract_hash == _contract_hash(contract)


def test_submission_rejects_native_patch_that_leaves_dependency_seam_missing(
    repair_case, monkeypatch
):
    import asyncio
    from fi.alk.harness import runtime_repair
    from fi.alk.harness.diagnostics import HarnessDiagnostic
    from fi.alk.harness.job import HarnessStage

    source, _, contract, patch = repair_case
    rejected = []

    class SubmittingStage:
        def __init__(self, spec, **kwargs):
            self.spec = spec

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def say(self, briefing, *, on_event=None):
            submit = self.spec.servers["runtime_repair"].tools[0].handler
            submission = patch.model_dump(
                mode="json", exclude={"base_contract_hash", "diagnostic_codes"}
            )
            submission.pop("runtime_dependencies")
            rejected.append(await submit(submission))

    monkeypatch.setattr(runtime_repair, "Stage", SubmittingStage)
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.CONNECTING_AGENT,
        component="runtime",
        code="generated_runtime_plan_invalid",
        message="missing connection seam",
    )
    with pytest.raises(RuntimeError, match="typed_runtime_plan_patch_not_submitted"):
        asyncio.run(
            runtime_repair.request_runtime_plan_patch(source, contract, (diagnostic,))
        )
    assert len(rejected) == 2
    assert all(
        item["is_error"]
        and "runtime_repair_dependency_seam_unresolved" in item["content"][0]["text"]
        for item in rejected
    )


@pytest.mark.parametrize("field", ["dsn_env", "host", "config_key", "loader_module"])
def test_existing_compiler_seam_cannot_be_repaired(repair_case, field):
    _, path, contract, patch = repair_case
    setattr(contract.runtime_dependencies[0].reached, field, "existing")
    path.write_text(contract.model_dump_json())
    dependencies = contract.model_copy(deep=True).runtime_dependencies
    dependencies[0].reached.loader_module = "client.config"
    patch = patch.model_copy(
        update={
            "base_contract_hash": _contract_hash(contract),
            "runtime_dependencies": dependencies,
        }
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="dependency_seam_already_configured"):
        apply(repair_case, patch)
    assert path.read_bytes() == before


def test_comment_mention_is_not_module_evidence(repair_case):
    source, _, _, patch = repair_case
    (source / "app.py").write_text("# client.config provides connectivity")
    with pytest.raises(ValueError, match="dependency_module_not_evidenced"):
        apply(repair_case, patch.model_copy(update={"evidence_paths": ("app.py",)}))
