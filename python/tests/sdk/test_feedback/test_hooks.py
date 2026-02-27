"""Tests for feedback integration hooks."""

import pytest
from fi.evals.feedback.hooks import (
    configure_feedback,
    get_default_store,
    retrieve_feedback_config,
)
from fi.evals.feedback.store import InMemoryFeedbackStore
from fi.evals.feedback.types import FeedbackEntry


class TestHooks:
    """Tests for feedback hooks."""

    def test_no_default_store(self):
        # Reset global state
        import fi.evals.feedback.hooks as hooks
        hooks._default_store = None
        assert get_default_store() is None

    def test_configure_feedback(self):
        import fi.evals.feedback.hooks as hooks
        store = InMemoryFeedbackStore()
        configure_feedback(store, max_examples=5)
        assert get_default_store() is store
        assert hooks._default_max_examples == 5
        # Clean up
        hooks._default_store = None

    def test_retrieve_feedback_config_no_store(self):
        import fi.evals.feedback.hooks as hooks
        hooks._default_store = None

        config = retrieve_feedback_config("faithfulness", {"output": "test"})
        assert config == {}

    def test_retrieve_feedback_config_with_explicit_store(self):
        store = InMemoryFeedbackStore()
        store.add(FeedbackEntry(
            eval_name="faithfulness",
            inputs={"output": "test"},
            correct_score=0.8,
            correct_reason="Good",
        ))

        config = retrieve_feedback_config(
            "faithfulness",
            {"output": "query"},
            store=store,
        )
        assert "few_shot_examples" in config
        assert len(config["few_shot_examples"]) == 1

    def test_retrieve_feedback_config_with_global_store(self):
        import fi.evals.feedback.hooks as hooks
        store = InMemoryFeedbackStore()
        store.add(FeedbackEntry(
            eval_name="test",
            inputs={"output": "a"},
            correct_score=0.5,
            correct_reason="Mid",
        ))
        hooks._default_store = store

        config = retrieve_feedback_config("test", {"output": "query"})
        assert "few_shot_examples" in config

        # Clean up
        hooks._default_store = None

    def test_explicit_store_overrides_global(self):
        import fi.evals.feedback.hooks as hooks
        global_store = InMemoryFeedbackStore()
        explicit_store = InMemoryFeedbackStore()

        explicit_store.add(FeedbackEntry(
            eval_name="test",
            inputs={"output": "a"},
            correct_score=0.5,
            correct_reason="From explicit",
        ))

        hooks._default_store = global_store

        config = retrieve_feedback_config("test", {"output": "q"}, store=explicit_store)
        assert "few_shot_examples" in config
        assert len(config["few_shot_examples"]) == 1

        # Global store has nothing
        config2 = retrieve_feedback_config("test", {"output": "q"})
        assert config2 == {} or "few_shot_examples" not in config2

        # Clean up
        hooks._default_store = None

    def test_merges_with_existing_config(self):
        store = InMemoryFeedbackStore()
        store.add(FeedbackEntry(
            eval_name="test",
            inputs={"output": "a"},
            correct_score=0.5,
            correct_reason="Mid",
        ))

        config = retrieve_feedback_config(
            "test",
            {"output": "q"},
            store=store,
            config={"existing": True},
        )
        assert config["existing"] is True
        assert "few_shot_examples" in config
