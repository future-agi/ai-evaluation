"""Phase 11B — agentic-batch certification (7 probe shims + promotions)."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

AGENTIC = {
    "a2a": ("send_message", "dict"),
    "agno": ("run", "dict"),
    "beeai": ("run", "dict"),
    "claude_agent_sdk": ("query", "text"),
    "google_adk": ("run", "dict"),
    "instructor": ("chat", "dict"),
    "smolagents": ("run", "text"),
}

REAL_IMPORTS = {
    "a2a": ["import a2a", "from a2a"],
    "agno": ["import agno", "from agno"],
    "beeai": ["import beeai", "from beeai"],
    "claude_agent_sdk": ["import claude_agent_sdk", "from claude_agent_sdk"],
    "google_adk": ["import google.adk", "from google.adk", "import google_adk"],
    "instructor": ["import instructor", "from instructor"],
    "smolagents": ["import smolagents", "from smolagents"],
}


def _run_shim(framework: str) -> dict:
    module = importlib.import_module(f"sdk_framework_adapter_cert_{framework}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{framework}.json"
        module.run(out)
        return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("framework", sorted(AGENTIC))
def test_agentic_shims_resolve_preset_method(framework):
    from fi.simulate.agent.frameworks import FRAMEWORK_PRESETS

    method, input_mode = AGENTIC[framework]
    spec = FRAMEWORK_PRESETS[framework]
    assert spec.method == method
    assert spec.input_mode == input_mode

    saved = _run_shim(framework)
    assert saved["status"] == "passed"
    assert saved["method"] == method
    assert saved["input_mode"] == input_mode
    assert saved["summary"]["runtime_trace_count"] >= 1
    assert saved["summary"]["tool_call_count"] >= 1


@pytest.mark.parametrize("framework", sorted(AGENTIC))
def test_agentic_shims_deterministic(framework):
    first = _run_shim(framework)
    second = _run_shim(framework)
    assert first["method"] == second["method"]
    assert first["input_mode"] == second["input_mode"]
    assert sorted(first["summary"]) == sorted(second["summary"])


def test_instructor_emits_typed_output():
    saved = _run_shim("instructor")
    state_keys = saved["cases"][0]["response"]["state_keys"]
    assert "typed_output" in state_keys


def test_a2a_shim_crosslinks_live_lane():
    source = (EXAMPLES / "sdk_framework_adapter_cert_a2a.py").read_text(
        encoding="utf-8"
    )
    assert "a2a_lane.py" in source
    assert "sdk_framework_adapter_a2a_protocol_trace.py" in source
    saved = _run_shim("a2a")
    assert saved["status"] == "passed"


@pytest.mark.parametrize("framework", sorted(AGENTIC))
def test_no_real_framework_imported(framework):
    source = (
        EXAMPLES / f"sdk_framework_adapter_cert_{framework}.py"
    ).read_text(encoding="utf-8")
    for needle in REAL_IMPORTS[framework]:
        assert needle not in source, f"{framework} shim imports the real framework"


def test_agentic_promotions_select_preset_shape():
    families = {
        "keyword_inputs": {"agno", "beeai", "google_adk"},
        "message_history": {"claude_agent_sdk", "smolagents"},
        "typed_output": {"instructor"},
        "side_kwargs": {"a2a"},
    }
    for family, expected in families.items():
        module = importlib.import_module(
            f"sdk_framework_adapter_cert_{family}_promotion"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / f"{family}.json"
            result = module.run(out)
        for framework in expected:
            info = result["frameworks"][framework]
            adapter = info["selected_adapter"]
            method, input_mode = AGENTIC[framework]
            assert adapter["method"] == method
            assert adapter["input_mode"] == input_mode
            assert info["manifest_agent"]["framework"] == framework
