"""Full-matrix release gate — every cloud template × representative inputs.

Parametrized over templates auto-discovered from the live registry. New
templates added backend-side are automatically covered; removed ones are
automatically dropped. This is the script that gates dev → main.

Budget: ~4-6 min on dev at ~4-6s per eval (serial). Use pytest-xdist (-n auto)
to parallelize when merged.
"""
from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------
# Input synthesizers — per required-keys signature
# ---------------------------------------------------------------------

IMG_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg"
)
AUDIO_URL = "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"
PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

# Key-set → representative input generator. Order matters: first key-set
# that is a subset of required_keys wins.
_KEY_RECIPES: list[tuple[frozenset[str], dict]] = [
    # Multimedia first (most specific)
    (frozenset(["image", "caption"]),
     {"image": IMG_URL, "caption": "A yellow Labrador dog."}),
    (frozenset(["instruction", "images"]),
     {"instruction": "picture of a yellow Labrador", "images": [IMG_URL]}),
    (frozenset(["image"]),
     {"image": IMG_URL}),
    (frozenset(["audio", "generated_transcript"]),
     {"audio": AUDIO_URL, "generated_transcript": "baby elephant"}),
    (frozenset(["input_audio"]),
     {"input_audio": AUDIO_URL}),
    (frozenset(["text", "generated_audio"]),
     {"text": "baby elephant walk", "generated_audio": AUDIO_URL}),
    (frozenset(["input_pdf", "json_content"]),
     {"input_pdf": PDF_URL, "json_content": '{"title":"Dummy PDF","body":"Dummy PDF file"}'}),
    # Conversation-based
    (frozenset(["conversation"]),
     {"conversation": [
         {"role": "user", "content": "I need help with my order."},
         {"role": "assistant", "content": "Sure — what's the order number?"},
         {"role": "user", "content": "123456"},
         {"role": "assistant", "content": "Found it. Shipped yesterday."},
     ]}),
    (frozenset(["system_prompt", "conversation"]),
     {"system_prompt": "You are a helpful customer support agent.",
      "conversation": [
          {"role": "user", "content": "hours?"},
          {"role": "assistant", "content": "9-5 Mon-Fri"},
      ]}),
    # Prompt-instruction
    (frozenset(["output", "prompt"]),
     {"prompt": "Answer concisely.", "output": "Paris."}),
    # Ground-truth match
    (frozenset(["generated_value", "expected_value"]),
     {"generated_value": "Paris", "expected_value": "Paris"}),
    # RAG / context
    (frozenset(["input", "output", "context"]),
     {"input": "What is the capital of France?", "output": "Paris.",
      "context": "France's capital is Paris."}),
    (frozenset(["input", "context"]),
     {"input": "What is the capital of France?", "context": "France's capital is Paris."}),
    (frozenset(["output", "context"]),
     {"output": "Paris is the capital of France.",
      "context": "France's capital is Paris."}),
    (frozenset(["context", "output"]),
     {"output": "Paris is the capital of France.",
      "context": "France's capital is Paris."}),
    # Comparison
    (frozenset(["expected", "output"]),
     {"expected": "Paris", "output": "Paris"}),
    (frozenset(["reference", "hypothesis"]),
     {"reference": "the cat sat on the mat", "hypothesis": "the cat sat on the mat"}),
    # Input/output
    (frozenset(["input", "output"]),
     {"input": "What is the capital of France?", "output": "Paris."}),
    # Singles
    (frozenset(["output"]),
     {"output": "Paris is the capital of France."}),
    (frozenset(["input"]),
     {"input": "What is the capital of France?"}),
    (frozenset(["text"]),
     {"text": "The quick brown fox jumps over the lazy dog."}),
    (frozenset(["response"]),
     {"response": "Paris is the capital of France."}),
    # CustomCodeEval sub-signatures (text/string deterministic metrics)
    (frozenset(["expected_text", "text"]),
     {"expected_text": "Paris is the capital of France.",
      "text": "Paris is the capital of France."}),
    (frozenset(["expected_response", "response"]),
     {"expected_response": "Paris is the capital of France.",
      "response": "Paris is the capital of France."}),
    (frozenset(["actual_json", "expected_json"]),
     {"actual_json": '{"name":"Paris","country":"France"}',
      "expected_json": '{"name":"Paris","country":"France"}'}),
    (frozenset(["images", "text"]),
     {"images": [IMG_URL], "text": "A yellow Labrador dog."}),
    (frozenset(["fake_images", "real_images"]),
     {"fake_images": [IMG_URL], "real_images": [IMG_URL]}),
]


def _inputs_for(required_keys: list[str]) -> dict | None:
    """Pick the most specific recipe whose keys match required_keys."""
    needed = frozenset(required_keys or [])
    for keys, recipe in _KEY_RECIPES:
        if keys == needed:
            return recipe
    return None


def _model_for(required_keys: list[str]) -> str:
    """turing_flash by default; escalate to turing_large for audio/PDF."""
    heavy = {"input_audio", "audio", "generated_audio", "generated_transcript",
             "input_pdf"}
    if any(k in heavy for k in (required_keys or [])):
        return "turing_large"
    return "turing_flash"


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------

def _discover_templates() -> list[tuple[str, tuple[str, ...], str]]:
    """Pull the full registry from the live api.

    Returns ``[(name, required_keys, eval_type_id)]`` for every system-owned
    template — both ``AgentEvaluator`` (LLM-as-judge) and ``CustomCodeEval``
    (deterministic code-executor). Drafts and user-owned custom templates
    are skipped because they aren't portable across api environments.
    """
    from fi.evals.core.cloud_registry import load_registry

    api_key = os.environ.get("FI_API_KEY")
    secret_key = os.environ.get("FI_SECRET_KEY")
    base_url = os.environ.get("FI_BASE_URL")
    if not (api_key and secret_key and base_url):
        return []

    reg = load_registry(base_url, api_key, secret_key, force_refresh=True)
    out: list[tuple[str, tuple[str, ...], str]] = []
    for name, info in sorted(reg.items()):
        if info.get("owner") != "system":
            continue
        cfg = info.get("config", {}) or {}
        eval_type = cfg.get("eval_type_id") or ""
        if eval_type not in ("AgentEvaluator", "CustomCodeEval"):
            continue
        if name in SKIP_NAMES:
            continue
        rk = tuple(cfg.get("required_keys") or [])
        if not rk:  # meta-templates like deterministic_evals with empty required_keys
            continue
        out.append((name, rk, eval_type))
    return out


# Templates deliberately excluded from the matrix.
SKIP_NAMES = {
    # Backend bug: CustomCodeEval sandbox returns 500 on every call
    # regardless of input. Tracked separately — not an SDK issue.
    "fuzzy_match",
    # Meta-template with empty required_keys — not a real eval.
    "deterministic_evals",
}


_DISCOVERED = _discover_templates()


@pytest.fixture(scope="session")
def matrix_templates():
    return _DISCOVERED


# ---------------------------------------------------------------------
# The parametrized test
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "eval_name,required_keys,eval_type",
    _DISCOVERED,
    ids=[name for name, _, _ in _DISCOVERED] or ["no-templates-discovered"],
)
def test_template(
    evaluator,
    eval_name: str,
    required_keys: tuple[str, ...],
    eval_type: str,
):
    """Every system-owned template in the live registry must:
      * accept the canonical inputs for its required_keys signature, OR
      * return a concrete failed `EvalResult` with the api error text.
    Either way — never a silent empty `BatchRunResult`.
    """
    inputs = _inputs_for(list(required_keys))
    if inputs is None:
        pytest.skip(
            f"No canonical input recipe for required_keys={list(required_keys)}. "
            "Add one to _KEY_RECIPES if this template is customer-facing."
        )

    model = _model_for(list(required_keys))
    batch = evaluator.evaluate(
        eval_templates=eval_name,
        inputs=inputs,
        model_name=model,
        timeout=180,
    )

    # Silent-empty is a hard failure — this is the regression we protect.
    assert batch.eval_results, (
        f"silent empty BatchRunResult for {eval_name} — silent-empty regression"
    )

    r = batch.eval_results[0]
    assert r.name == eval_name, f"response name mismatch: got {r.name!r}"

    if r.output is None:
        # Backend rejected — ensure the reason explains why, not a silent None.
        assert r.reason, (
            f"{eval_name}: output=None and no reason — silent failure regression"
        )
        pytest.fail(
            f"{eval_name} rejected by backend: {r.reason[:200]}",
            pytrace=False,
        )
