from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest

from fi.alk.harness.bundle_author_v2 import (
    BundleAuthorError,
    _plan_python,
    _uv_sync_build_command,
    resolve_environment_plan,
)
from fi.alk.harness.generated_runtime import (
    GeneratedRuntimeError,
    _python_install,
    select_python_extras,
)
from fi.alk.harness.job import HarnessJob
from fi.simulate.runtime.spec import RuntimeIsolation


def project(path, extras=()):
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(
        f'[project]\nname="fixture-{path.name}"\nversion="1.0"\n'
        "[project.optional-dependencies]\n"
        + "".join(f"{extra}=[]\n" for extra in extras)
    )


def workspace(path):
    (path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers=["packages/*"]\nexclude=["packages/excluded"]\n'
    )
    project(path / "packages/core", ["cloud"])
    project(path / "packages/api", ["http"])
    project(path / "packages/excluded", ["cloud", "excluded"])


def test_workspace_extra_has_one_included_owner(tmp_path):
    workspace(tmp_path)
    selection = select_python_extras(tmp_path, ["cloud", "cloud"])
    assert selection.names == ("cloud",)
    assert selection.uv_arguments == ["--all-packages", "--extra", "cloud"]
    with pytest.raises(GeneratedRuntimeError, match="extra_unknown: excluded"):
        select_python_extras(tmp_path, ["excluded"])
    project(tmp_path / "packages/api", ["cloud"])
    with pytest.raises(GeneratedRuntimeError, match="extra_ambiguous: cloud"):
        select_python_extras(tmp_path, ["cloud"])


def test_root_extra_takes_priority_without_installing_workspace_members(tmp_path):
    workspace(tmp_path)
    with (tmp_path / "pyproject.toml").open("a") as handle:
        handle.write(
            '[project]\nname="fixture-root"\nversion="1.0"\n'
            "[project.optional-dependencies]\ncloud=[]\n"
        )
    assert select_python_extras(tmp_path, ["cloud"]).uv_arguments == [
        "--extra",
        "cloud",
    ]
    # Mixing scopes would activate both owners of cloud under --all-packages.
    with pytest.raises(GeneratedRuntimeError, match="extra_ambiguous: cloud"):
        select_python_extras(tmp_path, ["cloud", "http"])


@pytest.mark.parametrize("extra", ["--all-extras", "cloud;echo", "cloud/path", ""])
def test_invalid_extra_rejected_before_command_generation(tmp_path, extra):
    project(tmp_path, ["cloud"])
    with pytest.raises(GeneratedRuntimeError, match="extra_invalid"):
        select_python_extras(tmp_path, [extra])


def test_workspace_cannot_select_extra_outside_source(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers=["../outside"]\n'
    )
    with pytest.raises(GeneratedRuntimeError, match="workspace_path_invalid"):
        select_python_extras(tmp_path, ["cloud"])


def test_process_requirements_without_manifest_cannot_silently_drop_extras(tmp_path):
    (tmp_path / "requirements.txt").write_text("")
    with pytest.raises(BundleAuthorError, match="extra_unknown: cloud"):
        _plan_python(
            tmp_path,
            name="agent",
            root=tmp_path,
            entry=None,
            control=True,
            needs_secrets=False,
            extras=["cloud"],
        )


def test_existing_pip_paths_keep_exact_selected_project_extras(tmp_path):
    project(tmp_path, ["cloud", "other"])
    runtime = SimpleNamespace(extras=["cloud"])
    result = _python_install(tmp_path, runtime, [])
    assert result[1] == "pip-project"
    assert result[2] == ["RUN pip install --no-cache-dir '.[cloud]'"]
    (tmp_path / "requirements.txt").write_text("pinned-dependency==1.0\n")
    result = _python_install(tmp_path, runtime, [])
    assert result[2] == [
        "RUN pip install --no-cache-dir -r requirements.txt '.[cloud]'"
    ]
    assert result[3] == ["cloud"]


def test_non_uv_requirements_path_rejects_workspace_extra(tmp_path):
    workspace(tmp_path)
    (tmp_path / "requirements.txt").write_text("")
    with pytest.raises(GeneratedRuntimeError, match="workspace_extras_require_uv"):
        _python_install(tmp_path, SimpleNamespace(extras=["cloud"]), [])


def test_compiled_compose_extra_is_selected_process_only(tmp_path):
    project(tmp_path / "api", ["cloud"])
    project(tmp_path / "worker")
    (tmp_path / "compose.yaml").write_text(
        "services:\n  api:\n    build: ./api\n    command: [python, -m, example]\n"
        "    depends_on: [worker]\n  worker:\n    build: ./worker\n"
        "    command: [python, -m, worker]\n"
    )
    job = HarnessJob.model_validate(
        {
            "job_id": "fixture",
            "run_id": "fixture",
            "execution": "hosted",
            "source": {"kind": "archive", "archive_artifact_id": "fixture"},
            "agent": {"connector": "http"},
            "scenario_count": 1,
            "runtime": {"isolation": RuntimeIsolation.DEDICATED_VM.value},
        }
    )
    plan = resolve_environment_plan(
        tmp_path,
        job,
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={
            "execution": "compose",
            "compose_file": "compose.yaml",
            "service": "api",
            "extras": ["cloud"],
        },
    )
    commands = {p.name: p.build_commands for p in plan.processes if p.kind == "source"}
    assert commands["api"][0][-2:] == ["--extra", "cloud"]
    assert "--extra" not in commands["worker"][0]


def wheel(root, name):
    filename = root / f"{name}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(filename, "w") as archive:
        archive.writestr(f"{name}.py", 'VALUE = "offline dependency"\n')
        archive.writestr(
            f"{name}-1.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
        )
        archive.writestr(
            f"{name}-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{name}-1.0.dist-info/RECORD", "")


@pytest.mark.skipif(
    shutil.which("uv") is None, reason="requires installed uv; no downloads"
)
@pytest.mark.parametrize("locked", [False, True])
def test_real_uv_installs_only_selected_member_extra_offline(tmp_path, locked):
    workspace(tmp_path)
    for name in ("fixture_cloud", "fixture_other"):
        wheel(tmp_path, name)
    (tmp_path / "packages/core/pyproject.toml").write_text(
        '[project]\nname="fixture-core"\nversion="1.0"\nrequires-python=">=3.12"\n'
        '[project.optional-dependencies]\ncloud=["fixture-cloud"]\nother=["fixture-other"]\n'
        "[tool.uv]\npackage=false\n[tool.uv.sources]\n"
        'fixture-cloud={path="../../fixture_cloud-1.0-py3-none-any.whl"}\n'
        'fixture-other={path="../../fixture_other-1.0-py3-none-any.whl"}\n'
    )
    with (tmp_path / "packages/api/pyproject.toml").open("a") as handle:
        handle.write("[tool.uv]\npackage=false\n")
    environment = {**os.environ, "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never"}
    if locked:
        subprocess.run(
            ["uv", "lock", "--no-cache", "--python", sys.executable],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
        )
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    selection = select_python_extras(tmp_path, ["cloud"])
    command = _uv_sync_build_command(
        tmp_path, sys.executable, extra_arguments=selection.uv_arguments
    )
    subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            str(tmp_path / ".venv/bin/python"),
            "-c",
            "import fixture_cloud, importlib.util; "
            'assert fixture_cloud.VALUE == "offline dependency"; '
            'assert importlib.util.find_spec("fixture_other") is None',
        ],
        check=True,
        capture_output=True,
    )
    assert all((tmp_path / path).read_bytes() == data for path, data in before.items())


def test_unknown_extra_rejected_in_existing_pip_project(tmp_path):
    project(tmp_path, ["cloud"])
    with pytest.raises(GeneratedRuntimeError, match="extra_unknown: missing"):
        _python_install(tmp_path, SimpleNamespace(extras=["missing"]), [])


def test_unlocked_workspace_container_path_uses_selected_extra(tmp_path):
    workspace(tmp_path)
    result = _python_install(tmp_path, SimpleNamespace(extras=["cloud"]), [])
    assert result[2][1] == "RUN uv sync --no-cache --all-packages --extra cloud"
    assert result[3] == ["cloud"]


def test_noninstallable_pip_project_cannot_silently_drop_extras(tmp_path):
    project(tmp_path, ["cloud"])
    (tmp_path / "requirements.txt").write_text("")
    with (tmp_path / "pyproject.toml").open("a") as handle:
        handle.write("[tool.uv]\npackage=false\n")
    with pytest.raises(
        GeneratedRuntimeError, match="pip_extras_require_installable_project"
    ):
        _python_install(tmp_path, SimpleNamespace(extras=["cloud"]), [])
