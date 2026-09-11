"""Type definitions for the Feedback Loop system.

Provides dataclasses for storing developer feedback on evaluation results,
calibration profiles, and aggregate statistics.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class FeedbackEntry:
    """A single piece of developer feedback on an evaluation result."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # What was evaluated
    eval_name: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)

    # What the system produced
    original_score: Optional[float] = None
    original_reason: str = ""
    original_passed: Optional[bool] = None

    # What the developer says is correct
    correct_score: Optional[float] = None
    correct_passed: Optional[bool] = None
    correct_reason: str = ""

    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_few_shot(self) -> Dict[str, Any]:
        """Convert to the format expected by CustomLLMJudge few_shot_examples.

        Returns dict matching the template schema:
            {"inputs": {...}, "output": "<json with score/reason>"}
        """
        output_dict = {
            "score": self.correct_score if self.correct_score is not None else self.original_score,
            "reason": self.correct_reason or self.original_reason,
        }
        return {
            "inputs": self.inputs,
            "output": json.dumps(output_dict),
        }

    def to_embedding_text(self) -> str:
        """Create a text representation for embedding.

        Concatenates key input fields into a string suitable for
        semantic embedding. Prioritizes output, context, and input fields.
        """
        parts = [f"metric: {self.eval_name}"]
        for key in ("output", "response", "context", "input", "query"):
            val = self.inputs.get(key)
            if val:
                text = val if isinstance(val, str) else json.dumps(val, default=str)
                parts.append(f"{key}: {text[:500]}")
        return "\n".join(parts)


@dataclass
class CalibrationProfile:
    """Optimized threshold settings for a metric based on feedback."""

    eval_name: str
    optimal_threshold: float
    sample_size: int
    accuracy_at_threshold: float  # % of feedback entries that agree at this threshold

    # Distribution stats
    score_mean: float = 0.0
    score_std: float = 0.0

    # Confusion matrix at the optimal threshold
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackStats:
    """Aggregate statistics for feedback on a given metric."""

    eval_name: str
    total_entries: int = 0
    agreement_rate: float = 0.0  # How often original == correct
    avg_score_delta: float = 0.0  # avg(correct_score - original_score)
    score_distribution: Dict[str, int] = field(default_factory=dict)  # buckets
