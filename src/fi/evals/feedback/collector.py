"""User-facing API for the feedback loop system.

Provides a clean interface for submitting feedback, querying statistics,
calibrating thresholds, and creating retrievers.
"""

import logging
from typing import Any, Dict, List, Optional

from ..core.result import EvalResult
from .store import FeedbackStore
from .types import FeedbackEntry, FeedbackStats
from .retriever import FeedbackRetriever
from .calibrator import ThresholdCalibrator

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Main user-facing class for the feedback loop system.

    Provides a clean API for:
    - Submitting feedback on metric results
    - Retrieving statistics on accumulated feedback
    - Calibrating thresholds based on feedback
    - Creating a retriever for pipeline integration

    Usage:
        from fi.evals.feedback import FeedbackCollector, InMemoryFeedbackStore

        store = InMemoryFeedbackStore()  # or ChromaFeedbackStore()
        feedback = FeedbackCollector(store)

        # After running a metric that gave wrong results:
        result = run_metric("faithfulness", output="...", context="...")

        # Submit correction
        feedback.submit(
            result,
            inputs={"output": "...", "context": "..."},
            correct_score=0.9,
            correct_reason="The response IS faithful via semantic equivalence.",
        )

        # Later, get a retriever for pipeline integration
        retriever = feedback.get_retriever()

    Args:
        store: The FeedbackStore backend.
    """

    def __init__(self, store: FeedbackStore):
        self.store = store

    def submit(
        self,
        result: EvalResult,
        *,
        inputs: Dict[str, Any],
        correct_score: Optional[float] = None,
        correct_passed: Optional[bool] = None,
        correct_reason: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackEntry:
        """Submit feedback on a metric result.

        Args:
            result: The EvalResult that needs correction.
            inputs: The original inputs that were used.
            correct_score: What the score SHOULD have been (0.0-1.0).
            correct_passed: What the pass/fail SHOULD have been.
            correct_reason: Why the original result was wrong.
            tags: Optional tags for organizing feedback.
            metadata: Optional metadata dict.

        Returns:
            The stored FeedbackEntry.

        Raises:
            ValueError: If neither correct_score nor correct_reason is provided.
        """
        if correct_score is None and not correct_reason:
            raise ValueError(
                "Feedback must include at least one of: correct_score, correct_reason. "
                "If the result was correct, use confirm() instead."
            )

        entry = FeedbackEntry(
            eval_name=result.eval_name,
            inputs=inputs,
            original_score=result.score,
            original_reason=result.reason,
            original_passed=result.passed,
            correct_score=correct_score,
            correct_passed=correct_passed,
            correct_reason=correct_reason,
            tags=tags or [],
            metadata=metadata or {},
        )

        self.store.add(entry)
        logger.info(
            f"Feedback submitted for '{result.eval_name}': "
            f"original={result.score} -> corrected={correct_score}"
        )
        return entry

    def confirm(
        self,
        result: EvalResult,
        *,
        inputs: Dict[str, Any],
        reason: str = "",
    ) -> FeedbackEntry:
        """Confirm that a metric result was correct.

        Records that the system got it right, which helps calibration
        accuracy calculations. These entries are stored but NOT injected
        as few-shot examples (since they don't correct anything).

        Args:
            result: The correct EvalResult.
            inputs: The original inputs.
            reason: Optional note on why this was correct.

        Returns:
            The stored FeedbackEntry.
        """
        entry = FeedbackEntry(
            eval_name=result.eval_name,
            inputs=inputs,
            original_score=result.score,
            original_reason=result.reason,
            original_passed=result.passed,
            correct_score=result.score,  # Same as original = confirmed correct
            correct_passed=result.passed,
            correct_reason=reason or "Confirmed correct by developer.",
            tags=["confirmed"],
        )
        self.store.add(entry)
        return entry

    def stats(self, metric_name: str) -> FeedbackStats:
        """Get aggregate statistics for feedback on a metric.

        Args:
            metric_name: The metric to get stats for.

        Returns:
            FeedbackStats with counts and agreement rates.
        """
        entries = self.store.get_by_metric(metric_name)

        if not entries:
            return FeedbackStats(eval_name=metric_name)

        total = len(entries)
        agreements = 0
        score_deltas = []

        for e in entries:
            if e.correct_score is not None and e.original_score is not None:
                delta = e.correct_score - e.original_score
                score_deltas.append(delta)
                # Agreement = within 0.1 of each other
                if abs(delta) < 0.1:
                    agreements += 1

        # Score distribution in 0.1 buckets
        distribution: Dict[str, int] = {}
        for e in entries:
            score = e.correct_score if e.correct_score is not None else e.original_score
            if score is not None:
                bucket = f"{int(score * 10) / 10:.1f}"
                distribution[bucket] = distribution.get(bucket, 0) + 1

        return FeedbackStats(
            eval_name=metric_name,
            total_entries=total,
            agreement_rate=agreements / total if total > 0 else 0.0,
            avg_score_delta=sum(score_deltas) / len(score_deltas) if score_deltas else 0.0,
            score_distribution=distribution,
        )

    def get_retriever(self, max_examples: int = 3) -> FeedbackRetriever:
        """Create a FeedbackRetriever wired to this collector's store.

        Args:
            max_examples: Max few-shot examples to retrieve per query.

        Returns:
            A FeedbackRetriever instance.
        """
        return FeedbackRetriever(store=self.store, max_examples=max_examples)

    def calibrate(
        self,
        metric_name: str,
        threshold_range: tuple = (0.3, 0.9),
        steps: int = 13,
    ):
        """Run threshold calibration for a metric.

        Args:
            metric_name: Metric to calibrate.
            threshold_range: (min, max) thresholds to search.
            steps: Number of threshold steps to try.

        Returns:
            CalibrationProfile with optimal threshold.
        """
        calibrator = ThresholdCalibrator(self.store)
        return calibrator.calibrate(metric_name, threshold_range, steps)
