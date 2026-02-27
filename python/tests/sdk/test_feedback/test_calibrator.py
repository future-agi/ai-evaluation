"""Tests for threshold calibration."""

import pytest
from fi.evals.feedback.calibrator import ThresholdCalibrator
from fi.evals.feedback.store import InMemoryFeedbackStore
from fi.evals.feedback.types import FeedbackEntry


class TestThresholdCalibrator:
    """Tests for ThresholdCalibrator."""

    @pytest.fixture
    def store(self):
        return InMemoryFeedbackStore()

    @pytest.fixture
    def calibrator(self, store):
        return ThresholdCalibrator(store)

    def _add_entries(self, store, entries):
        """Helper to add multiple (original_score, correct_score) pairs."""
        for orig, correct in entries:
            store.add(FeedbackEntry(
                eval_name="test",
                inputs={"output": f"item_{orig}"},
                original_score=orig,
                correct_score=correct,
            ))

    def test_insufficient_data_raises(self, calibrator):
        with pytest.raises(ValueError, match="at least 5"):
            calibrator.calibrate("test")

    def test_calibrate_perfect_separation(self, store, calibrator):
        # Low original scores that should pass, high ones that should fail
        # This creates a clear threshold at around 0.5
        self._add_entries(store, [
            (0.2, 0.1),  # low score, should fail
            (0.3, 0.2),  # low score, should fail
            (0.4, 0.3),  # low score, should fail
            (0.6, 0.8),  # high score, should pass
            (0.7, 0.9),  # high score, should pass
            (0.8, 0.95), # high score, should pass
        ])

        profile = calibrator.calibrate("test")
        assert profile.sample_size == 6
        assert profile.accuracy_at_threshold > 0.5

    def test_calibrate_f1_optimization(self, store):
        calibrator = ThresholdCalibrator(store, optimize_for="f1")

        self._add_entries(store, [
            (0.2, 0.1),
            (0.3, 0.2),
            (0.5, 0.6),
            (0.6, 0.8),
            (0.7, 0.9),
        ])

        profile = calibrator.calibrate("test")
        assert profile.sample_size == 5

    def test_confusion_matrix_values(self, store, calibrator):
        # All above threshold, all should pass
        self._add_entries(store, [
            (0.8, 0.9),
            (0.7, 0.8),
            (0.6, 0.7),
            (0.5, 0.6),
            (0.4, 0.5),
        ])

        profile = calibrator.calibrate("test", threshold_range=(0.3, 0.4), steps=1)
        # At threshold 0.35, all original_scores >= 0.35, all correct_passed = True
        assert profile.true_positives == 5
        assert profile.false_positives == 0

    def test_calibrate_returns_score_stats(self, store, calibrator):
        self._add_entries(store, [
            (0.5, 0.6),
            (0.5, 0.7),
            (0.5, 0.8),
            (0.5, 0.9),
            (0.5, 1.0),
        ])

        profile = calibrator.calibrate("test")
        assert profile.score_mean > 0
        assert profile.score_std >= 0

    def test_f1_helper(self):
        assert ThresholdCalibrator._f1(10, 0, 0) == 1.0
        assert ThresholdCalibrator._f1(0, 0, 0) == 0.0
        assert ThresholdCalibrator._f1(5, 5, 5) == pytest.approx(0.5)
