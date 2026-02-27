"""Tests for feedback storage backends."""

import pytest
from fi.evals.feedback.store import InMemoryFeedbackStore
from fi.evals.feedback.types import FeedbackEntry


class TestInMemoryFeedbackStore:
    """Tests for InMemoryFeedbackStore."""

    @pytest.fixture
    def store(self):
        return InMemoryFeedbackStore()

    def test_add_and_count(self, store):
        entry = FeedbackEntry(eval_name="faithfulness", inputs={"output": "test"})
        entry_id = store.add(entry)
        assert entry_id == entry.id
        assert store.count() == 1
        assert store.count("faithfulness") == 1
        assert store.count("other_metric") == 0

    def test_query_similar_returns_matching_metric(self, store):
        store.add(FeedbackEntry(eval_name="faithfulness", inputs={"output": "a"}))
        store.add(FeedbackEntry(eval_name="groundedness", inputs={"output": "b"}))
        store.add(FeedbackEntry(eval_name="faithfulness", inputs={"output": "c"}))

        results = store.query_similar("faithfulness", "query text")
        assert len(results) == 2
        assert all(r.eval_name == "faithfulness" for r in results)

    def test_query_similar_respects_n_results(self, store):
        for i in range(10):
            store.add(FeedbackEntry(eval_name="test", inputs={"output": f"item_{i}"}))

        results = store.query_similar("test", "query", n_results=3)
        assert len(results) == 3

    def test_get_by_metric(self, store):
        store.add(FeedbackEntry(eval_name="faithfulness", inputs={"output": "a"}))
        store.add(FeedbackEntry(eval_name="groundedness", inputs={"output": "b"}))

        entries = store.get_by_metric("faithfulness")
        assert len(entries) == 1
        assert entries[0].eval_name == "faithfulness"

    def test_delete(self, store):
        entry = FeedbackEntry(eval_name="test")
        store.add(entry)
        assert store.count() == 1

        assert store.delete(entry.id) is True
        assert store.count() == 0

    def test_delete_nonexistent(self, store):
        assert store.delete("nonexistent-id") is False

    def test_empty_store(self, store):
        assert store.count() == 0
        assert store.query_similar("any", "text") == []
        assert store.get_by_metric("any") == []
