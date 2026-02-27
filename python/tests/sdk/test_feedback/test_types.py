"""Tests for feedback type definitions."""

import json
import pytest
from fi.evals.feedback.types import FeedbackEntry, CalibrationProfile, FeedbackStats


class TestFeedbackEntry:
    """Tests for FeedbackEntry dataclass."""

    def test_defaults(self):
        entry = FeedbackEntry()
        assert entry.eval_name == ""
        assert entry.inputs == {}
        assert entry.original_score is None
        assert entry.correct_score is None
        assert entry.id  # auto-generated UUID

    def test_to_few_shot_with_correction(self):
        entry = FeedbackEntry(
            eval_name="faithfulness",
            inputs={"output": "hello", "context": "world"},
            original_score=0.3,
            original_reason="Low faithfulness",
            correct_score=0.9,
            correct_reason="Actually faithful",
        )
        few_shot = entry.to_few_shot()
        assert few_shot["inputs"] == {"output": "hello", "context": "world"}
        output = json.loads(few_shot["output"])
        assert output["score"] == 0.9
        assert output["reason"] == "Actually faithful"

    def test_to_few_shot_falls_back_to_original(self):
        entry = FeedbackEntry(
            eval_name="faithfulness",
            inputs={"output": "hello"},
            original_score=0.5,
            original_reason="Medium score",
        )
        few_shot = entry.to_few_shot()
        output = json.loads(few_shot["output"])
        assert output["score"] == 0.5
        assert output["reason"] == "Medium score"

    def test_to_embedding_text(self):
        entry = FeedbackEntry(
            eval_name="faithfulness",
            inputs={"output": "hello world", "context": "some context", "extra": "ignored"},
        )
        text = entry.to_embedding_text()
        assert "metric: faithfulness" in text
        assert "output: hello world" in text
        assert "context: some context" in text
        # "extra" is not in the priority keys
        assert "extra" not in text

    def test_to_embedding_text_truncates(self):
        entry = FeedbackEntry(
            eval_name="test",
            inputs={"output": "x" * 1000},
        )
        text = entry.to_embedding_text()
        # Should truncate to 500 chars per field
        assert len(text) < 600

    def test_unique_ids(self):
        e1 = FeedbackEntry()
        e2 = FeedbackEntry()
        assert e1.id != e2.id


class TestCalibrationProfile:

    def test_defaults(self):
        profile = CalibrationProfile(
            eval_name="faithfulness",
            optimal_threshold=0.65,
            sample_size=20,
            accuracy_at_threshold=0.85,
        )
        assert profile.score_mean == 0.0
        assert profile.true_positives == 0


class TestFeedbackStats:

    def test_defaults(self):
        stats = FeedbackStats(eval_name="faithfulness")
        assert stats.total_entries == 0
        assert stats.agreement_rate == 0.0
        assert stats.score_distribution == {}
