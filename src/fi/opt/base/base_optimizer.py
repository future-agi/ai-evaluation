from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
from ..types import OptimizationResult


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization algorithms.
    Each concrete optimizer will implement its own `optimize` method,
    containing the full logic for its optimization loop.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def optimize(
        self,
        evaluator: Any,
        data_mapper: Any,  # We'll refine this later
        dataset: List[Dict[str, Any]],
        metric: Callable,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Runs the full optimization process.

        Args:
            evaluator: The user-provided evaluator instance.
            data_mapper: The user-provided data mapper.
            dataset: The dataset to use for evaluation.
            metric: The metric function to use for evaluation.
            **kwargs: Additional, optimizer-specific arguments. Common optional
                arguments include:
                - early_stopping (EarlyStoppingConfig): Configuration for early
                    stopping criteria. Supports patience-based stopping, score
                    thresholds, minimum improvement deltas, and cost budgets.
                    When configured, optimization may terminate before reaching
                    the maximum number of iterations.

        Returns:
            An OptimizationResult object with the best generator, iteration
            history, final score, and early stopping metadata (if applicable).
        """
        pass
