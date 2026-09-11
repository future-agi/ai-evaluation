from __future__ import annotations

import importlib.util
from pathlib import Path

from fi.alk import studio
from fi.simulate import cli


_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "build_delivery_support_suite.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("delivery_support_suite", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delivery_support_suite_is_deterministic_and_valid(tmp_path: Path) -> None:
    module = _module()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    module.write_suite(first)
    module.write_suite(second)

    assert first.read_bytes() == second.read_bytes()
    scenario = cli._build_scenario(
        {"scenario": {"source": first.name}},
        tmp_path,
    )
    assert len(scenario.dataset) == 10
    assert len({persona.version for persona in scenario.dataset}) == 10
    assert all(persona.is_typed for persona in scenario.dataset)
    assert all(studio.validate_persona(persona)["status"] == "valid" for persona in scenario.dataset)
    assert studio.bias_lint(scenario.dataset)["status"] == "passed"
    assert scenario.version == module.build_suite().version
