from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def run_voice_case():
    acceptance_directory = Path(__file__).parents[1] / "oss" / "simulation-acceptance"
    path = acceptance_directory / "run_voice_case.py"
    spec = importlib.util.spec_from_file_location("acceptance_run_voice_case", path)
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(acceptance_directory))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(acceptance_directory))


@pytest.mark.parametrize(
    ("status", "evaluation_passed", "expected"),
    [
        ("completed", True, 0),
        ("completed", False, 1),
        ("failed", True, 1),
        ("failed", False, 1),
    ],
)
def test_result_exit_code_requires_transport_and_evaluation_success(
    run_voice_case,
    status: str,
    evaluation_passed: bool,
    expected: int,
) -> None:
    assert (
        run_voice_case._result_exit_code(
            status=status,
            evaluation_passed=evaluation_passed,
        )
        == expected
    )
