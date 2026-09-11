from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "sdk_actor_source_tool_calling.py"
    )
    spec = importlib.util.spec_from_file_location("actor_tool_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_calling_actor_source_example_drives_the_world(tmp_path) -> None:
    module = _load_example()
    data = module.run(tmp_path / "actor-tool.json")
    assert data["status"] == "passed"
    assert data["run_status"] == "completed"
    assert data["tool_drove_world"] is True
    assert data["world_final_state"]["refund"]["status"] == "approved"
    assert "refund approved" in data["transcript"]
