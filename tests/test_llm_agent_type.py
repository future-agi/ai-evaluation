"""agent.type=llm — instructions-driven LLM candidate (prompt-optimization unit)."""

from types import SimpleNamespace

import pytest

from fi.simulate.cli import ManifestError, _build_agent_callback


def test_llm_agent_requires_instructions(tmp_path):
    with pytest.raises(ManifestError, match="requires agent.instructions"):
        _build_agent_callback({"type": "llm"}, tmp_path)


def test_llm_agent_builds_messages_and_returns_completion(tmp_path, monkeypatch):
    captured = {}

    def fake_completion(self, model, messages, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        return "stubbed reply"

    monkeypatch.setattr(
        "fi.evals.llm.providers.litellm.LiteLLMProvider.get_completion",
        fake_completion,
    )

    cb = _build_agent_callback(
        {"type": "llm", "instructions": "Be terse.", "model": "gpt-4o-mini"},
        tmp_path,
    )
    out = cb(
        SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            new_message={"role": "user", "content": "hi"},
        )
    )

    assert out.content == "stubbed reply"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["messages"][0] == {"role": "system", "content": "Be terse."}
    # new_message equals the last history entry — must not be duplicated.
    assert captured["messages"].count({"role": "user", "content": "hi"}) == 1


def test_llm_agent_appends_new_message_when_not_in_history(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "fi.evals.llm.providers.litellm.LiteLLMProvider.get_completion",
        lambda self, model, messages, **kw: str(len(messages)),
    )
    cb = _build_agent_callback({"type": "prompt", "instructions": "x"}, tmp_path)
    out = cb(SimpleNamespace(messages=[], new_message={"role": "user", "content": "q"}))
    # system + appended new_message
    assert out.content == "2"
