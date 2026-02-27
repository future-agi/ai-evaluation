"""Tests for feedback retrieval and few-shot formatting."""

import json
import pytest
from fi.evals.feedback.store import InMemoryFeedbackStore
from fi.evals.feedback.retriever import FeedbackRetriever
from fi.evals.feedback.types import FeedbackEntry


class TestFeedbackRetriever:
    """Tests for FeedbackRetriever."""

    @pytest.fixture
    def store(self):
        return InMemoryFeedbackStore()

    @pytest.fixture
    def retriever(self, store):
        return FeedbackRetriever(store=store, max_examples=3)

    def test_empty_store_returns_no_examples(self, retriever):
        examples = retriever.retrieve_few_shot_examples(
            "faithfulness", {"output": "test"}
        )
        assert examples == []

    def test_retrieves_corrected_entries(self, store, retriever):
        # Entry with correction -- should be retrieved
        store.add(FeedbackEntry(
            eval_name="faithfulness",
            inputs={"output": "Paris is in France", "context": "Paris is the capital of France"},
            correct_score=0.9,
            correct_reason="Faithful",
        ))
        # Entry without correction -- should be skipped
        store.add(FeedbackEntry(
            eval_name="faithfulness",
            inputs={"output": "something"},
        ))

        examples = retriever.retrieve_few_shot_examples(
            "faithfulness", {"output": "Berlin is in Germany"}
        )
        assert len(examples) == 1
        output = json.loads(examples[0]["output"])
        assert output["score"] == 0.9

    def test_respects_max_examples(self, store):
        for i in range(10):
            store.add(FeedbackEntry(
                eval_name="test",
                inputs={"output": f"item_{i}"},
                correct_score=float(i) / 10,
                correct_reason=f"Reason {i}",
            ))

        retriever = FeedbackRetriever(store=store, max_examples=2)
        examples = retriever.retrieve_few_shot_examples("test", {"output": "query"})
        assert len(examples) <= 2

    def test_inject_into_config_empty(self, retriever):
        config = retriever.inject_into_config("faithfulness", {"output": "test"})
        assert "few_shot_examples" not in config

    def test_inject_into_config_with_feedback(self, store, retriever):
        store.add(FeedbackEntry(
            eval_name="faithfulness",
            inputs={"output": "test"},
            correct_score=0.8,
            correct_reason="Good",
        ))

        config = retriever.inject_into_config(
            "faithfulness",
            {"output": "test query"},
            config={"existing_key": "value"},
        )
        assert config["existing_key"] == "value"
        assert len(config["few_shot_examples"]) == 1

    def test_inject_merges_existing_examples(self, store, retriever):
        store.add(FeedbackEntry(
            eval_name="test",
            inputs={"output": "a"},
            correct_score=0.5,
            correct_reason="Mid",
        ))

        existing_examples = [{"inputs": {"output": "existing"}, "output": '{"score": 1.0}'}]
        config = retriever.inject_into_config(
            "test",
            {"output": "query"},
            config={"few_shot_examples": existing_examples},
        )
        # Should have existing + retrieved
        assert len(config["few_shot_examples"]) == 2

    def test_does_not_mutate_input_config(self, store, retriever):
        store.add(FeedbackEntry(
            eval_name="test",
            inputs={"output": "a"},
            correct_score=0.5,
            correct_reason="Mid",
        ))

        original_config = {"key": "value"}
        new_config = retriever.inject_into_config("test", {"output": "q"}, config=original_config)
        assert "few_shot_examples" not in original_config
        assert "few_shot_examples" in new_config

    def test_build_query_text(self, retriever):
        text = retriever.build_query_text(
            "faithfulness",
            {"output": "hello", "context": "world", "extra": "ignored"},
        )
        assert "metric: faithfulness" in text
        assert "output: hello" in text
        assert "context: world" in text
