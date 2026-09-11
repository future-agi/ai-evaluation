"""Phase 11B — model-clients-batch certification (12 probe shims + promotions)."""

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

MODEL_CLIENTS = {
    "bedrock": ("invoke_model", "dict"),
    "cerebras": ("chat", "dict"),
    "cohere": ("chat", "dict"),
    "deepseek": ("chat", "dict"),
    "fireworks": ("chat", "dict"),
    "huggingface": ("__call__", "dict"),
    "litellm": ("completion", "dict"),
    "ollama": ("chat", "dict"),
    "portkey": ("chat", "dict"),
    "strands": ("__call__", "text"),
    "together": ("chat", "dict"),
    "xai": ("chat", "dict"),
}

REAL_IMPORTS = {
    "bedrock": ["import boto3", "import bedrock", "from boto3"],
    "cerebras": ["import cerebras", "from cerebras"],
    "cohere": ["import cohere", "from cohere"],
    "deepseek": ["import deepseek", "from deepseek"],
    "fireworks": ["import fireworks", "from fireworks"],
    "huggingface": ["import transformers", "from transformers", "import huggingface_hub"],
    "litellm": ["import litellm", "from litellm"],
    "ollama": ["import ollama", "from ollama"],
    "portkey": ["import portkey", "from portkey"],
    "strands": ["import strands", "from strands"],
    "together": ["import together", "from together"],
    "xai": ["import xai", "from xai"],
}


def _run_shim(framework: str) -> dict:
    module = importlib.import_module(f"sdk_framework_adapter_cert_{framework}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{framework}.json"
        module.run(out)
        return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("framework", sorted(MODEL_CLIENTS))
def test_model_client_shims_resolve_preset_method(framework):
    method, input_mode = MODEL_CLIENTS[framework]
    saved = _run_shim(framework)
    assert saved["status"] == "passed"
    assert saved["method"] == method
    assert saved["input_mode"] == input_mode
    assert saved["summary"]["tool_call_count"] >= 1


def test_model_client_input_mode_validity_not_equality():
    """chat/dict presets are VALID InputMode members, NOT discovery-equal.

    The §6 amendment: a strict chat == "text" discovery-equality check would
    falsely fail these presets, so it is deliberately NOT written.
    """
    from typing import get_args

    from fi.simulate.agent.frameworks import (
        _DISCOVERY_METHOD_INPUT_MODES,
        FRAMEWORK_PRESETS,
    )
    from fi.simulate.agent.generic import InputMode

    valid = set(get_args(InputMode))
    # discovery says chat -> text, but the presets pin chat -> dict (intentional)
    assert _DISCOVERY_METHOD_INPUT_MODES["chat"] == "text"
    for framework in ("cerebras", "cohere", "deepseek", "together", "xai"):
        spec = FRAMEWORK_PRESETS[framework]
        assert spec.method == "chat"
        assert spec.input_mode == "dict"
        assert spec.input_mode in valid  # validity, not discovery-equality
        assert spec.input_mode != _DISCOVERY_METHOD_INPUT_MODES["chat"]


def test_ollama_credential_free_no_daemon():
    from fi.alk import trinity

    lane = {row["framework"] for row in trinity.V1_FRAMEWORK_PRESET_LIVE_VALIDATION_LANE}
    assert "ollama" not in lane  # 11B-A9: local daemon is not a credential
    saved = _run_shim("ollama")  # runs with no daemon, no env var
    assert saved["status"] == "passed"


def test_bedrock_reconciliation_note_present():
    page = (ROOT / "docs/frameworks/bedrock.md").read_text(encoding="utf-8")
    assert "not in traceai" in page.lower()
    assert "inventory artifact" in page.lower()


@pytest.mark.parametrize("framework", sorted(MODEL_CLIENTS))
def test_no_real_provider_imported(framework):
    source = (
        EXAMPLES / f"sdk_framework_adapter_cert_{framework}.py"
    ).read_text(encoding="utf-8")
    for needle in REAL_IMPORTS[framework]:
        assert needle not in source, f"{framework} shim imports the real provider"


def test_model_client_promotions_select_preset_shape():
    families = {
        "provider_response": {
            "bedrock",
            "cerebras",
            "cohere",
            "deepseek",
            "fireworks",
            "litellm",
            "ollama",
            "portkey",
            "together",
            "xai",
        },
        "nested_method": {"huggingface"},
        "message_history": {"strands"},
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
            method, input_mode = MODEL_CLIENTS[framework]
            assert adapter["method"] == method
            assert adapter["input_mode"] == input_mode
            assert info["manifest_agent"]["framework"] == framework
