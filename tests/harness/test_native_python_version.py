"""Explicit Python patch versions are interpreter selectors, not binary names."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fi.alk.harness.bundle_author_v2 import _plan_python


@pytest.mark.parametrize("requirements", [False, True])
def test_native_patch_version_runs_without_patch_named_binary(
    tmp_path: Path, requirements: bool
) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv is required to exercise native installation")
    version = ".".join(str(part) for part in sys.version_info[:3])
    (tmp_path / "agent.py").write_text("import sys; print(sys.version.split()[0])\n")
    if requirements:
        (tmp_path / "requirements.txt").write_text("")
    process = _plan_python(
        tmp_path, name="agent", root=tmp_path, entry="agent.py",
        control=True, needs_secrets=False, version=version,
    )
    environment = {
        **os.environ, "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
    }
    for command in process.build_commands:
        subprocess.run(command, cwd=tmp_path, env=environment, check=True, capture_output=True, text=True)
    result = subprocess.run(
        process.run_command, cwd=tmp_path, env=environment,
        check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == version


@pytest.mark.parametrize("compatible", [True, False])
def test_native_minor_selector_resolves_source_patch_requirement(
    tmp_path: Path, compatible: bool
) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv is required to exercise native installation")
    major, minor, patch = sys.version_info[:3]
    declared = f"{major}.{minor}"
    requires = f">={declared}.{patch},<{major}.{minor + 1}" if compatible else f">={major}.{minor + 1}"
    (tmp_path / "agent.py").write_text("import sys; print(sys.version.split()[0])\n")
    (tmp_path / "pyproject.toml").write_text(
        f"[project]\nname='agent'\nversion='1'\nrequires-python='{requires}'\n"
    )
    process = _plan_python(
        tmp_path, name="agent", root=tmp_path, entry="agent.py",
        control=True, needs_secrets=False, version=declared,
    )
    result = subprocess.run(
        process.build_commands[0], cwd=tmp_path,
        env={**os.environ, "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
             "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
        capture_output=True, text=True,
    )
    if compatible:
        assert result.returncode == 0, result.stderr
        run = subprocess.run(
            [str(tmp_path / ".venv/bin/python"), "agent.py"], cwd=tmp_path,
            check=True, capture_output=True, text=True,
        )
        assert run.stdout.strip() == f"{declared}.{patch}"
    else:
        assert result.returncode != 0
        assert "No interpreter found" in result.stderr
