"""Tests for FeedbackCollector."""

import pytest
from fi.evals.core.result import EvalResult
from fi.evals.feedback.collector import FeedbackCollector
from fi.evals.feedback.store import InMemoryFeedbackStore
from fi.evals.feedback.retriever import FeedbackRetriever


class TestFeedbackCollector:
    """Tests for FeedbackCollector."""

    @pytest.fixture
    def store(self):
        return InMemoryFeedbackStore()

    @pytest.fixture
    def collector(self, store):
        return FeedbackCollector(store)

    @pytest.fixture
    def sample_result(self):
        return EvalResult(
            eval_name="faithfulness",
            score=0.3,
            reason="Low faithfulness detected",
        )

    def test_submit_feedback(self, collector, store, sample_result):
        entry = collector.submit(
            sample_result,
            inputs={"output": "Paris is in Germany", "context": "Paris is in France"},
            correct_score=0.1,
            correct_reason="Even worse than detected",
        )
        assert entry.eval_name == "faithfulness"
        assert entry.original_score == 0.3
        assert entry.correct_score == 0.1
        assert store.count() == 1

    def test_submit_requires_correction(self, collector, sample_result):
        with pytest.raises(ValueError, match="at least one of"):
            collector.submit(
                sample_result,
                inputs={"output": "test"},
            )

    def test_submit_with_reason_only(self, collector, sample_result):
        entry = collector.submit(
            sample_result,
            inputs={"output": "test"},
            correct_reason="This was actually correct",
        )
        assert entry.correct_reason == "This was actually correct"
        assert entry.correct_score is None

    def test_submit_with_tags(self, collector, sample_result):
        entry = collector.submit(
            sample_result,
            inputs={"output": "test"},
            correct_score=0.8,
            tags=["rag", "production"],
        )
        assert entry.tags == ["rag", "production"]

    def test_confirm(self, collector, store, sample_result):
        entry = collector.confirm(
            sample_result,
            inputs={"output": "test"},
        )
        assert entry.correct_score == 0.3  # Same as original
        assert "confirmed" in entry.tags
        assert store.count() == 1

    def test_stats_empty(self, collector):
        stats = collector.stats("faithfulness")
        assert stats.total_entries == 0
        assert stats.agreement_rate == 0.0

    def test_stats_with_entries(self, collector, store):
        result1 = EvalResult(eval_name="faithfulness", score=0.3, reason="Low")
        result2 = EvalResult(eval_name="faithfulness", score=0.8, reason="High")

        collector.submit(
            result1,
            inputs={"output": "bad"},
            correct_score=0.1,
            correct_reason="Worse",
        )
        collector.confirm(
            result2,
            inputs={"output": "good"},
        )

        stats = collector.stats("faithfulness")
        assert stats.total_entries == 2
        assert stats.agreement_rate == 0.5  # one agrees (confirm), one doesn't

    def test_get_retriever(self, collector):
        retriever = collector.get_retriever(max_examples=5)
        assert isinstance(retriever, FeedbackRetriever)
        assert retriever.max_examples == 5

    def test_calibrate_insufficient_data(self, collector):
        with pytest.raises(ValueError, match="at least 5"):
            collector.calibrate("faithfulness")

    def test_calibrate_with_data(self, collector, store):
        # Submit enough feedback for calibration
        for i in range(10):
            original = 0.3 + (i * 0.05)
            correct = 0.8 if i >= 5 else 0.2  # Clear separation
            result = EvalResult(eval_name="test", score=original, reason=f"Score {original}")
            collector.submit(
                result,
                inputs={"output": f"item_{i}"},
                correct_score=correct,
                correct_reason=f"Corrected to {correct}",
            )

        profile = collector.calibrate("test")
        assert profile.eval_name == "test"
        assert profile.sample_size == 10
        assert 0.3 <= profile.optimal_threshold <= 0.9
        assert profile.accuracy_at_threshold > 0
