from pathlib import Path

import pytest

from fi.alk.harness.contract import AgentContract, Runtime
from fi.alk.harness.generated_runtime import (
    GeneratedRuntimeError,
    _context_excludes,
    _copy_sanitized_context,
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


def test_generated_context_exclude_glob_omits_only_matching_test_files(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='agent'\nversion='1'\n")
    (tmp_path / "server.py").write_text("print('ready')\n")
    (tmp_path / "agent.py").write_text("VALUE = 1\n")
    (tmp_path / "agent.test.py").write_text("TOKEN = 'ghp_" + "x" * 36 + "'\n")
    runtime = Runtime(
        language="python",
        execution="native",
        command=["python", "server.py"],
        context_excludes=["**/*.test.py"],
    )
    exclusions = _context_excludes(
        tmp_path, runtime, "pyproject.toml", runtime.command
    )
    destination = tmp_path / "copy"
    _copy_sanitized_context(tmp_path, destination, exclusions)

    assert (destination / "agent.py").is_file()
    assert not (destination / "agent.test.py").exists()


def test_generated_context_exclude_glob_cannot_remove_required_files(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='agent'\nversion='1'\n")
    (tmp_path / "server.py").write_text("print('ready')\n")
    runtime = Runtime(
        language="python",
        execution="native",
        command=["python", "server.py"],
        context_excludes=["**/*.py"],
    )

    with pytest.raises(GeneratedRuntimeError, match="context_exclude_required"):
        _context_excludes(tmp_path, runtime, "pyproject.toml", runtime.command)


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
            language="node", execution="native", command=["node", "server"],
            interface={"kind": "http", "port": 8000, "path": "/chat"},
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
            execution="native", command=["python", "server.py"], extras=["cloud-sdk"],
            interface={"kind": "http", "port": 8000, "path": "/chat"},
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
def test_compilers_reject_extra_owner_removed_from_context(
    tmp_path, excluded, compiler
):
    from fi.alk.harness.bundle_author_v2 import BundleAuthorError, _plan_python
    from fi.alk.harness.generated_runtime import detect_generated_runtime

    workspace(tmp_path)
    runtime = Runtime(
        language="python",
        execution="native",
        command=["python", "server.py"],
        extras=["cloud-sdk"],
        install=INSTALL,
        context_excludes=[excluded],
    )
    with pytest.raises(
        (GeneratedRuntimeError, BundleAuthorError), match="extra_unknown"
    ):
        if compiler == "generated":
            detect_generated_runtime(tmp_path, runtime)
        else:
            _plan_python(
                tmp_path,
                name="agent",
                root=tmp_path,
                entry="server.py",
                control=True,
                needs_secrets=False,
                extras=runtime.extras,
                install=runtime.install,
                context_excludes=tuple(runtime.context_excludes),
            )


def test_explicit_requirements_supplement_project_extras(tmp_path):
    from fi.alk.harness.bundle_author_v2 import _plan_python
    from fi.alk.harness.generated_runtime import detect_generated_runtime

    workspace(tmp_path)
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend/requirements.txt").write_text("tavily-python>=0.7.12\n")
    runtime = Runtime(
        language="python",
        command=["python", "server.py"],
        extras=["cloud-sdk"],
        requirements_files=["backend/requirements.txt"],
    )
    process = _plan_python(
        tmp_path,
        name="agent",
        root=tmp_path,
        entry="server.py",
        control=True,
        needs_secrets=False,
        extras=runtime.extras,
        requirements_files=runtime.requirements_files,
    )
    assert process.build_commands[-1] == [
        "uv",
        "pip",
        "check",
        "--python",
        ".venv/bin/python",
    ]
    assert process.build_commands[-2][0:2] == [".venv/bin/python", "-c"]
    assert process.build_commands[-2][-1] == "backend/requirements.txt"
    assert "cloud-sdk" in process.build_commands[0]
    generated = detect_generated_runtime(tmp_path, runtime)
    assert "backend/requirements.txt" in generated.dockerfile


@pytest.mark.parametrize(
    "case",
    [
        "escape",
        "absolute",
        "excluded",
        "nested_escape",
        "nested_excluded",
        "sensitive",
        "nested_sensitive",
        "cycle",
        "url",
        "option",
        "symlink",
    ],
)
def test_selected_requirements_reject_unsafe_graph(tmp_path, case):
    from fi.alk.harness.generated_runtime import select_python_requirements

    root = tmp_path / "source"
    root.mkdir()
    (root / "requirements.txt").write_text("client>=1\n")
    (tmp_path / "outside.txt").write_text("client>=1\n")
    names = ["requirements.txt"]
    excluded = ()
    if case == "escape":
        names = ["../outside.txt"]
    elif case == "absolute":
        names = [str(root / "requirements.txt")]
    elif case == "excluded":
        excluded = ("requirements.txt",)
    elif case == "nested_escape":
        (root / "requirements.txt").write_text("-r ../outside.txt\n")
    elif case == "nested_excluded":
        (root / "other.txt").write_text("client\n")
        (root / "requirements.txt").write_text("-r other.txt\n")
        excluded = ("other.txt",)
    elif case in {"sensitive", "nested_sensitive"}:
        (root / ".env.production").write_text("client>=1\n")
        if case == "sensitive":
            names = [".env.production"]
        else:
            (root / "requirements.txt").write_text("-r .env.production\n")
    elif case == "cycle":
        (root / "requirements.txt").write_text("-r requirements.txt\n")
    elif case == "url":
        (root / "requirements.txt").write_text("client @ file:///etc/client\n")
    elif case == "option":
        (root / "requirements.txt").write_text("--index-url https://example.com\n")
    else:
        (root / "requirements.txt").unlink()
        (root / "requirements.txt").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(GeneratedRuntimeError, match="generated_runtime_requirements"):
        select_python_requirements(root, names, context_excludes=excluded)


def test_selected_requirements_allow_nested_constraints_and_deduplicate(tmp_path):
    from fi.alk.harness.generated_runtime import select_python_requirements

    (tmp_path / "backend").mkdir()
    (tmp_path / "backend/requirements.txt").write_text(
        "-r ../common.txt\n-c ../constraints.txt\n"
    )
    (tmp_path / "common.txt").write_text("client>=1 # source comment\n")
    (tmp_path / "constraints.txt").write_text("client<3\n")
    assert select_python_requirements(tmp_path, ["backend/requirements.txt"] * 2) == (
        "backend/requirements.txt",
    )


def test_selected_requirements_accept_hashed_exports_without_rewriting(tmp_path):
    from fi.alk.harness.generated_runtime import select_python_requirements

    source = (
        '# Generated export\n-r dependencies.txt\n'
        'client[search]==1; python_version >= "3.11" \\\n'
        f'    --hash=sha256:{"a" * 64} \\\n'
        f'    --hash sha512:{"b" * 128} # wheel hashes\n'
    )
    (tmp_path / "requirements.txt").write_text(source)
    (tmp_path / "dependencies.txt").write_text(
        f'supplement==1 --hash=sha384:{"c" * 96}\n'
    )
    assert select_python_requirements(tmp_path, ["requirements.txt"]) == ("requirements.txt",)
    assert (tmp_path / "requirements.txt").read_text() == source


@pytest.mark.parametrize("entry", [
    "client==1 --hash=sha256:short",
    "client==1 --hash=md5:" + "a" * 32,
    "client==1 --hash=sha256:" + "a" * 64 + " --index-url https://example.com",
    "client==1 --config-settings=unsafe --hash=sha256:" + "a" * 64,
    "client @ file:///etc/client --hash=sha256:" + "a" * 64,
    "--hash=sha256:" + "a" * 64,
    "client==1 \\\n --trusted-host example.com",
    "# comment \\\nclient @ file:///etc/client",
    "   # comment \\\n--index-url https://example.com",
    "client==1\\\n# comment \\\n--index-url https://example.com",
])
def test_selected_hashed_requirements_reject_unsupported_entries(tmp_path, entry):
    from fi.alk.harness.generated_runtime import select_python_requirements

    (tmp_path / "requirements.txt").write_text(entry + "\n")
    with pytest.raises(GeneratedRuntimeError, match="generated_runtime_requirements"):
        select_python_requirements(tmp_path, ["requirements.txt"])


@pytest.mark.parametrize("baseline", ["wheel", "workspace"])
@pytest.mark.parametrize("requested_version, succeeds", [("1", True), ("2", False)])
@pytest.mark.parametrize("hash_mode", ["none", "valid", "invalid"])
def test_supplemental_requirements_preserve_installed_extra_resolution(
    tmp_path, requested_version, succeeds, baseline, hash_mode
):
    import hashlib
    import os
    import shutil
    import subprocess
    import sys
    import zipfile

    from fi.alk.harness.bundle_author_v2 import _plan_python

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv executable required for offline resolver regression")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name, version, extra in [
        ("client", "1", ""), ("client", "2", ""), ("supplement", "1", ""),
        ("agent", "1", 'Provides-Extra: search\nRequires-Dist: client<2; extra == "search"\n'),
    ]:
        dist = f"{name}-{version}.dist-info"
        with zipfile.ZipFile(wheels / f"{name}-{version}-py3-none-any.whl", "w") as wheel:
            wheel.writestr(f"{dist}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{extra}")
            wheel.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            wheel.writestr(f"{dist}/RECORD", "")
    env = dict(os.environ, UV_OFFLINE="1", UV_NO_INDEX="1", UV_FIND_LINKS=str(wheels), UV_CACHE_DIR=str(tmp_path / "cache"))
    subprocess.run([uv, "venv", "--python", sys.executable, ".venv"], cwd=tmp_path, env=env, check=True, capture_output=True)
    if baseline == "wheel":
        subprocess.run([uv, "pip", "install", "--python", ".venv/bin/python", "agent[search]"], cwd=tmp_path, env=env, check=True, capture_output=True)
    else:
        (tmp_path / "pyproject.toml").write_text('[project]\nname="root"\nversion="1"\n[tool.uv]\npackage=false\n[tool.uv.workspace]\nmembers=["sdk"]\n')
        (tmp_path / "sdk").mkdir()
        (tmp_path / "sdk/pyproject.toml").write_text('[project]\nname="sdk"\nversion="1"\n[project.optional-dependencies]\nsearch=["client<2"]\n[tool.uv]\npackage=false\n')
        subprocess.run([uv, "sync", "--all-packages", "--extra", "search"], cwd=tmp_path, env=env, check=True, capture_output=True)
    (tmp_path / "server.py").write_text("")
    requirements = []
    for name, version in [("client", requested_version), ("supplement", "1")]:
        requirement = f"{name}=={version}"
        if hash_mode != "none":
            digest = hashlib.sha256((wheels / f"{name}-{version}-py3-none-any.whl").read_bytes()).hexdigest()
            if hash_mode == "invalid" and name == "supplement":
                digest = "0" * 64
            requirement += f" \\\n    --hash=sha256:{digest}"
        requirements.append(requirement)
    (tmp_path / "supplement.txt").write_text("\n".join(requirements) + "\n")
    process = _plan_python(tmp_path, name="agent", root=tmp_path, entry="server.py", control=True, needs_secrets=False, requirements_files=["supplement.txt"])
    # The existing environment represents a successful source-owned sync, including extras.
    result = subprocess.run(process.build_commands[-2], cwd=tmp_path, env=env, capture_output=True, text=True)
    succeeds = succeeds and hash_mode != "invalid"
    assert (result.returncode == 0) is succeeds, result.stderr
    version = subprocess.check_output([str(tmp_path / ".venv/bin/python"), "-c", "from importlib.metadata import version; print(version('client'))"], text=True).strip()
    assert version == "1"
    if succeeds:
        assert subprocess.check_output([str(tmp_path / ".venv/bin/python"), "-c", "from importlib.metadata import version; print(version('supplement'))"], text=True).strip() == "1"
