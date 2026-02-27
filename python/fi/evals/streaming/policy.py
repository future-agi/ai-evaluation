"""Early stop policies for streaming evaluation.

Defines conditions under which streaming evaluation should stop early.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from .types import EarlyStopReason, EarlyStopCondition, ChunkResult


@dataclass
class PolicyState:
    """Tracks state for policy evaluation."""

    consecutive_failures: Dict[str, int] = field(default_factory=dict)
    total_failures: Dict[str, int] = field(default_factory=dict)
    triggered_conditions: List[str] = field(default_factory=list)


class EarlyStopPolicy:
    """
    Policy that determines when to stop streaming evaluation early.

    Manages a set of conditions that can trigger early stopping, such as
    toxicity thresholds, safety violations, or custom conditions.

    Example:
        policy = EarlyStopPolicy()
        policy.add_condition(
            name="high_toxicity",
            eval_name="toxicity",
            threshold=0.7,
            comparison="above",
        )

        for chunk_result in stream:
            should_stop, reason = policy.check(chunk_result)
            if should_stop:
                break
    """

    def __init__(self):
        """Initialize the policy."""
        self._conditions: List[EarlyStopCondition] = []
        self._custom_checks: List[Callable[[ChunkResult], Optional[EarlyStopReason]]] = []
        self._state = PolicyState()

    def add_condition(
        self,
        name: str,
        eval_name: str,
        threshold: float,
        comparison: str = "below",
        consecutive_chunks: int = 1,
    ) -> "EarlyStopPolicy":
        """
        Add a threshold-based stop condition.

        Args:
            name: Name for this condition
            eval_name: Name of the evaluation to check
            threshold: Threshold value
            comparison: "below" (stop if score < threshold) or "above" (stop if score > threshold)
            consecutive_chunks: Number of consecutive chunks that must fail

        Returns:
            Self for chaining
        """
        condition = EarlyStopCondition(
            name=name,
            eval_name=eval_name,
            threshold=threshold,
            comparison=comparison,
            consecutive_chunks=consecutive_chunks,
        )
        self._conditions.append(condition)
        return self

    def add_toxicity_stop(
        self,
        threshold: float = 0.7,
        consecutive: int = 1,
    ) -> "EarlyStopPolicy":
        """
        Add toxicity-based stop condition.

        Args:
            threshold: Stop if toxicity score exceeds this
            consecutive: Number of consecutive chunks

        Returns:
            Self for chaining
        """
        return self.add_condition(
            name="toxicity_stop",
            eval_name="toxicity",
            threshold=threshold,
            comparison="above",
            consecutive_chunks=consecutive,
        )

    def add_safety_stop(
        self,
        threshold: float = 0.3,
        consecutive: int = 1,
    ) -> "EarlyStopPolicy":
        """
        Add safety-based stop condition.

        Args:
            threshold: Stop if safety score drops below this
            consecutive: Number of consecutive chunks

        Returns:
            Self for chaining
        """
        return self.add_condition(
            name="safety_stop",
            eval_name="safety",
            threshold=threshold,
            comparison="below",
            consecutive_chunks=consecutive,
        )

    def add_quality_stop(
        self,
        threshold: float = 0.3,
        consecutive: int = 3,
    ) -> "EarlyStopPolicy":
        """
        Add quality-based stop condition.

        Args:
            threshold: Stop if quality score stays below this
            consecutive: Number of consecutive chunks

        Returns:
            Self for chaining
        """
        return self.add_condition(
            name="quality_stop",
            eval_name="quality",
            threshold=threshold,
            comparison="below",
            consecutive_chunks=consecutive,
        )

    def add_custom_check(
        self,
        check_fn: Callable[[ChunkResult], Optional[EarlyStopReason]],
    ) -> "EarlyStopPolicy":
        """
        Add a custom check function.

        Args:
            check_fn: Function that takes ChunkResult and returns
                      EarlyStopReason if should stop, None otherwise

        Returns:
            Self for chaining
        """
        self._custom_checks.append(check_fn)
        return self

    def check(self, chunk_result: ChunkResult) -> tuple:
        """
        Check if any stop conditions are triggered.

        Args:
            chunk_result: Result from evaluating a chunk

        Returns:
            Tuple of (should_stop: bool, reason: EarlyStopReason)
        """
        # Check threshold-based conditions
        for condition in self._conditions:
            if not condition.enabled:
                continue

            score = chunk_result.scores.get(condition.eval_name)
            if score is None:
                continue

            # Track consecutive failures
            key = condition.name
            if self._check_threshold(score, condition.threshold, condition.comparison):
                self._state.consecutive_failures[key] = (
                    self._state.consecutive_failures.get(key, 0) + 1
                )
                self._state.total_failures[key] = (
                    self._state.total_failures.get(key, 0) + 1
                )

                # Check if consecutive threshold met
                if condition.check(score, self._state.consecutive_failures[key]):
                    self._state.triggered_conditions.append(condition.name)
                    return True, self._get_reason_for_condition(condition)
            else:
                # Reset consecutive count
                self._state.consecutive_failures[key] = 0

        # Check custom conditions
        for check_fn in self._custom_checks:
            reason = check_fn(chunk_result)
            if reason is not None:
                return True, reason

        return False, EarlyStopReason.NONE

    def _check_threshold(
        self,
        score: float,
        threshold: float,
        comparison: str,
    ) -> bool:
        """Check if score triggers threshold."""
        if comparison == "below":
            return score < threshold
        else:
            return score > threshold

    def _get_reason_for_condition(
        self,
        condition: EarlyStopCondition,
    ) -> EarlyStopReason:
        """Get the appropriate stop reason for a condition."""
        name_lower = condition.name.lower()
        eval_lower = condition.eval_name.lower()

        if "toxic" in name_lower or "toxic" in eval_lower:
            return EarlyStopReason.TOXICITY
        elif "safe" in name_lower or "safe" in eval_lower:
            return EarlyStopReason.SAFETY
        elif "pii" in name_lower or "pii" in eval_lower:
            return EarlyStopReason.PII
        elif "jailbreak" in name_lower or "jailbreak" in eval_lower:
            return EarlyStopReason.JAILBREAK
        else:
            return EarlyStopReason.THRESHOLD

    def reset(self) -> None:
        """Reset policy state."""
        self._state = PolicyState()

    def enable_condition(self, name: str) -> None:
        """Enable a condition by name."""
        for condition in self._conditions:
            if condition.name == name:
                condition.enabled = True
                break

    def disable_condition(self, name: str) -> None:
        """Disable a condition by name."""
        for condition in self._conditions:
            if condition.name == name:
                condition.enabled = False
                break

    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics."""
        return {
            "conditions": [c.to_dict() for c in self._conditions],
            "consecutive_failures": dict(self._state.consecutive_failures),
            "total_failures": dict(self._state.total_failures),
            "triggered_conditions": list(self._state.triggered_conditions),
            "custom_checks": len(self._custom_checks),
        }

    @classmethod
    def default(cls) -> "EarlyStopPolicy":
        """
        Create a policy with sensible defaults.

        Returns:
            EarlyStopPolicy with toxicity and safety stops
        """
        policy = cls()
        policy.add_toxicity_stop(threshold=0.7, consecutive=1)
        policy.add_safety_stop(threshold=0.3, consecutive=1)
        return policy

    @classmethod
    def strict(cls) -> "EarlyStopPolicy":
        """
        Create a strict policy for high-risk applications.

        Returns:
            EarlyStopPolicy with strict thresholds
        """
        policy = cls()
        policy.add_toxicity_stop(threshold=0.5, consecutive=1)
        policy.add_safety_stop(threshold=0.5, consecutive=1)
        policy.add_quality_stop(threshold=0.4, consecutive=2)
        return policy

    @classmethod
    def permissive(cls) -> "EarlyStopPolicy":
        """
        Create a permissive policy that only stops on severe issues.

        Returns:
            EarlyStopPolicy with high thresholds
        """
        policy = cls()
        policy.add_toxicity_stop(threshold=0.9, consecutive=2)
        policy.add_safety_stop(threshold=0.1, consecutive=2)
        return policy
