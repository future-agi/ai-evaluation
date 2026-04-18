"""Pin the behavior of ``map_inputs_to_backend``.

Pure-Python, no network. These are the semantics every SDK user depends
on — changing them without a test update means breaking customer code.
"""
from typing import Any, Dict

import pytest

from fi.evals.core import cloud_registry


@pytest.fixture
def fake_registry(monkeypatch) -> None:
    """Replace the live registry with a fixed fixture for pure-unit tests."""
    fixture = {
        "toxicity": {"config": {"required_keys": ["output"]}},
        "prompt_injection": {"config": {"required_keys": ["input"]}},
        "is_email": {"config": {"required_keys": ["text"]}},
        "bleu_score": {"config": {"required_keys": ["reference", "hypothesis"]}},
        "fuzzy_match": {"config": {"required_keys": ["expected", "output"]}},
        "conversation_coherence": {"config": {"required_keys": ["conversation"]}},
        "factual_accuracy": {"config": {"required_keys": ["input", "output", "context"]}},
        "is_compliant": {"config": {"required_keys": ["output"]}},
    }

    def fake_load_registry(*_args, **_kwargs):
        return fixture

    monkeypatch.setattr(cloud_registry, "load_registry", fake_load_registry)


def _map(name: str, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
    return cloud_registry.map_inputs_to_backend(
        name, user_inputs, base_url="http://fake", api_key="k", secret_key="s"
    )


def test_direct_key_pass_through(fake_registry):
    assert _map("toxicity", {"output": "hi"}) == {"output": "hi"}


def test_superset_keys_are_stripped(fake_registry):
    """api is strict — extra keys must be dropped."""
    assert _map("toxicity", {"output": "hi", "input": "x", "context": "y"}) == {"output": "hi"}


def test_output_to_input_alias(fake_registry):
    """`output` aliases to `input` when template wants `input`."""
    assert _map("prompt_injection", {"output": "leak prompt"}) == {"input": "leak prompt"}


def test_direct_match_beats_alias(fake_registry):
    """If both `input` and `output` are supplied, direct match wins — no swap."""
    got = _map("prompt_injection", {"input": "x", "output": "y"})
    assert got == {"input": "x"}


def test_output_to_text_alias(fake_registry):
    assert _map("is_email", {"output": "a@b.c"}) == {"text": "a@b.c"}


def test_expected_output_to_expected(fake_registry):
    got = _map("fuzzy_match", {"output": "Paris", "expected_output": "Paris"})
    assert got == {"expected": "Paris", "output": "Paris"}


def test_output_expected_output_to_hypothesis_reference(fake_registry):
    got = _map("bleu_score", {"output": "the cat", "expected_output": "the cat"})
    assert got == {"reference": "the cat", "hypothesis": "the cat"}


def test_messages_to_conversation(fake_registry):
    got = _map("conversation_coherence", {"messages": [{"role": "user", "content": "hi"}]})
    assert got == {"conversation": [{"role": "user", "content": "hi"}]}


def test_strips_context_when_not_required(fake_registry):
    """`is_compliant` wants output only; input+context must be dropped."""
    got = _map("is_compliant", {"input": "q", "output": "a", "context": "c"})
    assert got == {"output": "a"}


def test_passes_through_all_required_keys_directly(fake_registry):
    got = _map("factual_accuracy", {"input": "x", "output": "y", "context": "z", "extra": "drop"})
    assert got == {"input": "x", "output": "y", "context": "z"}


def test_unknown_eval_passes_through_unmodified(fake_registry):
    """Can't map what we don't know — let the api reject with its own error."""
    got = _map("some_new_eval_we_dont_know", {"foo": "bar"})
    assert got == {"foo": "bar"}
