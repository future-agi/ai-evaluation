"""The simulated caller should not deliberate before answering."""

from fi.simulate.simulation.livekit_models import _simulator_thinking


def test_a_level_taking_model_is_asked_for_the_least_it_accepts():
    # Gemini 3 takes a level, and the LiveKit plugin answers a budget it cannot use by
    # substituting its own "minimal", which Vertex rejects on every inference.
    assert _simulator_thinking("gemini-3.7-flash") == {"thinking_level": "low"}


def test_a_budget_taking_model_is_asked_for_none():
    assert _simulator_thinking("gemini-2.5-flash") == {"thinking_budget": 0}


def test_a_run_can_ask_for_thinking(monkeypatch):
    monkeypatch.setenv("SIMULATOR_LLM_THINKING", "low")
    assert _simulator_thinking("gemini-3.7-flash") == {"thinking_level": "low"}


def test_a_numeric_budget_is_passed_through(monkeypatch):
    monkeypatch.setenv("SIMULATOR_LLM_THINKING", "512")
    assert _simulator_thinking("gemini-2.5-flash") == {"thinking_budget": 512}


def test_a_numeric_budget_becomes_a_level_where_that_is_all_the_model_takes(monkeypatch):
    monkeypatch.setenv("SIMULATOR_LLM_THINKING", "512")
    assert _simulator_thinking("gemini-3.7-flash") == {"thinking_level": "low"}


def test_non_gemini_models_are_left_alone():
    assert _simulator_thinking("gpt-4o") is None
