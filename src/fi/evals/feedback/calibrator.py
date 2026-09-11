"""Statistical threshold calibration based on feedback.

Optimizes pass/fail thresholds by computing confusion matrices against
developer-provided correct labels across a range of threshold values.
"""

import logging
import math
from typing import List, Tuple

from .store import FeedbackStore
from .types import CalibrationProfile, FeedbackEntry

logger = logging.getLogger(__name__)


class ThresholdCalibrator:
    """Optimizes pass/fail thresholds based on accumulated feedback.

    For each candidate threshold, computes a confusion matrix against
    the developer's correct_passed labels, and selects the threshold
    that maximizes agreement (accuracy) or F1 score.

    Args:
        store: FeedbackStore containing feedback entries.
        optimize_for: Metric to optimize. "accuracy" (default) or "f1".
    """

    def __init__(
        self,
        store: FeedbackStore,
        optimize_for: str = "accuracy",
    ):
        self.store = store
        self.optimize_for = optimize_for

    def calibrate(
        self,
        metric_name: str,
        threshold_range: Tuple[float, float] = (0.3, 0.9),
        steps: int = 13,
    ) -> CalibrationProfile:
        """Find the optimal pass/fail threshold for a metric.

        Args:
            metric_name: The metric to calibrate.
            threshold_range: (min_threshold, max_threshold) to search.
            steps: Number of evenly-spaced thresholds to try.

        Returns:
            CalibrationProfile with the optimal threshold and stats.

        Raises:
            ValueError: If insufficient feedback (< 5 entries with corrections).
        """
        entries = self.store.get_by_metric(metric_name)

        # Filter to entries with both a correct label and a score
        usable = [
            e for e in entries
            if e.correct_score is not None
            and e.original_score is not None
        ]

        if len(usable) < 5:
            raise ValueError(
                f"Need at least 5 feedback entries with corrections to calibrate "
                f"'{metric_name}', but only {len(usable)} found. Submit more feedback first."
            )

        # Derive correct_passed if not explicitly set
        for e in usable:
            if e.correct_passed is None:
                e.correct_passed = e.correct_score >= 0.5

        # Search over threshold space
        min_t, max_t = threshold_range
        best_score = -1.0
        best_threshold = 0.5
        best_matrix = (0, 0, 0, 0)

        for i in range(steps):
            t = min_t + (max_t - min_t) * i / (steps - 1) if steps > 1 else (min_t + max_t) / 2
            tp, fp, tn, fn = self._confusion_matrix(usable, t)

            if self.optimize_for == "f1":
                score = self._f1(tp, fp, fn)
            else:
                total = tp + fp + tn + fn
                score = (tp + tn) / total if total > 0 else 0.0

            if score > best_score:
                best_score = score
                best_threshold = t
                best_matrix = (tp, fp, tn, fn)

        # Compute score statistics
        scores = [e.correct_score for e in usable if e.correct_score is not None]
        mean = sum(scores) / len(scores) if scores else 0.0
        variance = sum((s - mean) ** 2 for s in scores) / len(scores) if scores else 0.0

        tp, fp, tn, fn = best_matrix

        profile = CalibrationProfile(
            eval_name=metric_name,
            optimal_threshold=round(best_threshold, 3),
            sample_size=len(usable),
            accuracy_at_threshold=best_score,
            score_mean=round(mean, 4),
            score_std=round(math.sqrt(variance), 4),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )

        logger.info(
            f"Calibrated '{metric_name}': threshold={profile.optimal_threshold} "
            f"accuracy={profile.accuracy_at_threshold:.1%} (n={profile.sample_size})"
        )

        return profile

    @staticmethod
    def _confusion_matrix(
        entries: List[FeedbackEntry],
        threshold: float,
    ) -> Tuple[int, int, int, int]:
        """Compute confusion matrix at a given threshold.

        Predicted positive = original_score >= threshold
        Actual positive = correct_passed is True

        Returns:
            (true_positives, false_positives, true_negatives, false_negatives)
        """
        tp = fp = tn = fn = 0
        for e in entries:
            predicted_pass = e.original_score >= threshold
            actual_pass = e.correct_passed

            if predicted_pass and actual_pass:
                tp += 1
            elif predicted_pass and not actual_pass:
                fp += 1
            elif not predicted_pass and not actual_pass:
                tn += 1
            else:
                fn += 1

        return tp, fp, tn, fn

    @staticmethod
    def _f1(tp: int, fp: int, fn: int) -> float:
        """Compute F1 score from confusion matrix components."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
