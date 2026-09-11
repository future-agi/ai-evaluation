import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EarlyStoppingException(Exception):
    """
    Exception raised when early stopping criteria are met during optimization.

    This is a custom exception class to explicitly signal early termination,
    distinguishing it from built-in StopIteration which is intended for
    iterator protocol.

    Attributes:
        reason: Human-readable explanation of why early stopping was triggered
    """
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class EarlyStoppingConfig(BaseModel):
    """
    Configuration for early stopping criteria in optimization.

    All fields are optional - if all are None, early stopping is disabled.
    Multiple criteria can be configured simultaneously; optimization stops
    when ANY criterion is met.
    """

    patience: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Stop optimization after this many consecutive iterations "
            "without score improvement. None disables patience-based stopping."
        ),
    )

    min_score_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Stop optimization when average score reaches or exceeds this "
            "threshold (0.0-1.0). None disables threshold-based stopping."
        ),
    )

    min_delta: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Minimum score improvement to be considered as progress. "
            "If current_score > (best_score + min_delta), patience counter resets. "
            "None defaults to 0.0 (any improvement counts)."
        ),
    )

    max_evaluations: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Maximum number of dataset evaluations allowed. Counts total "
            "evaluations across all iterations. None disables budget-based stopping."
        ),
    )

    def is_enabled(self) -> bool:
        """
        Check if any early stopping criterion is configured.

        Returns:
            True if at least one stopping criterion is set, False otherwise.

        Note:
            min_delta is not checked here because it only modifies patience
            behavior and doesn't constitute a stopping criterion by itself.
        """
        return any(
            [
                self.patience is not None,
                self.min_score_threshold is not None,
                self.max_evaluations is not None,
            ]
        )


class EarlyStoppingChecker:
    """
    Stateful checker that tracks optimization progress and evaluates
    stopping conditions across iterations.

    This class maintains internal state about the best score achieved,
    iterations without improvement, and total evaluations performed.
    Call should_stop() after each iteration to check if optimization
    should terminate.

    Example:
        config = EarlyStoppingConfig(patience=3, min_delta=0.01)
        checker = EarlyStoppingChecker(config)

        for iteration in range(max_iterations):
            score = evaluate_current_prompt()

            if checker.should_stop(score, num_evaluations=10):
                print(f"Stopped: {checker.get_state()['stop_reason']}")
                break
    """

    def __init__(self, config: EarlyStoppingConfig):
        """
        Initialize early stopping checker.

        Args:
            config: Early stopping configuration
        """
        self.config = config

        # State tracking
        self._best_score: float = -1.0
        self._iterations_without_improvement: int = 0
        self._total_evaluations: int = 0
        self._stopped: bool = False
        self._stop_reason: Optional[str] = None

    def should_stop(
        self,
        current_score: float,
        num_evaluations: int = 1,
    ) -> bool:
        """
        Check if optimization should stop based on current iteration.

        This method updates internal state and evaluates all configured
        stopping criteria. Returns True if any criterion is met.

        Args:
            current_score: Average score from current iteration (0.0-1.0)
            num_evaluations: Number of dataset evaluations in this iteration

        Returns:
            True if any stopping criterion is met, False otherwise
        """
        if self._stopped:
            return True

        if not self.config.is_enabled():
            return False

        # Update evaluation count
        self._total_evaluations += num_evaluations

        # Check cost budget first (always check regardless of score)
        if self._check_cost_budget():
            return True

        # Check absolute threshold
        if self._check_score_threshold(current_score):
            return True

        # Update improvement tracking
        min_delta = self.config.min_delta if self.config.min_delta is not None else 0.0
        if current_score > (self._best_score + min_delta):
            # Improvement detected - reset patience
            self._best_score = current_score
            self._iterations_without_improvement = 0
            logger.debug(
                f"Early stopping: Improvement detected "
                f"(score={current_score:.4f}, best={self._best_score:.4f})"
            )
        else:
            # No improvement - increment patience counter
            self._iterations_without_improvement += 1
            logger.debug(
                f"Early stopping: No improvement "
                f"({self._iterations_without_improvement} iterations)"
            )

        # Check patience
        if self._check_patience():
            return True

        return False

    def _check_patience(self) -> bool:
        """Check patience criterion."""
        if self.config.patience is None:
            return False

        if self._iterations_without_improvement >= self.config.patience:
            self._stopped = True
            self._stop_reason = (
                f"Patience exceeded: no improvement for "
                f"{self._iterations_without_improvement} iterations "
                f"(best score: {self._best_score:.4f})"
            )
            return True

        return False

    def _check_score_threshold(self, score: float) -> bool:
        """Check absolute score threshold criterion."""
        if self.config.min_score_threshold is None:
            return False

        if score >= self.config.min_score_threshold:
            self._stopped = True
            self._stop_reason = (
                f"Score threshold reached: {score:.4f} >= "
                f"{self.config.min_score_threshold:.4f}"
            )
            return True

        return False

    def _check_cost_budget(self) -> bool:
        """Check cost budget criterion."""
        if self.config.max_evaluations is None:
            return False

        if self._total_evaluations >= self.config.max_evaluations:
            self._stopped = True
            self._stop_reason = (
                f"Evaluation budget exhausted: {self._total_evaluations} "
                f">= {self.config.max_evaluations} "
                f"(best score: {self._best_score:.4f})"
            )
            return True

        return False

    def get_state(self) -> Dict[str, Any]:
        """
        Get current checker state for debugging and logging.

        Returns:
            Dictionary containing:
                - best_score: Best score achieved so far
                - iterations_without_improvement: Current patience counter
                - total_evaluations: Total evaluations performed
                - stopped: Whether stopping criterion has been triggered
                - stop_reason: Reason for stopping (if stopped)
        """
        return {
            "best_score": self._best_score,
            "iterations_without_improvement": self._iterations_without_improvement,
            "total_evaluations": self._total_evaluations,
            "stopped": self._stopped,
            "stop_reason": self._stop_reason,
        }

    def reset(self) -> None:
        """
        Reset checker state for reuse across multiple optimization runs.

        This allows the same checker instance to be reused without
        creating a new object.
        """
        self._best_score = -1.0
        self._iterations_without_improvement = 0
        self._total_evaluations = 0
        self._stopped = False
        self._stop_reason = None
