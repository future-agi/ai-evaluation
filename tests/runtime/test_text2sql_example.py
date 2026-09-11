from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example():
    path = Path(__file__).resolve().parents[2] / "examples" / "sdk_text2sql_world.py"
    spec = importlib.util.spec_from_file_location("text2sql_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text2sql_world_example_solves(tmp_path) -> None:
    module = _load_example()
    data = module.run(tmp_path / "text2sql.json")
    assert data["status"] == "passed"
    assert data["run_status"] == "completed"
    assert data["solved"] is True
    assert data["attempts"] == 1
    assert "[('A1',), ('A3',)]" in data["transcript"]
