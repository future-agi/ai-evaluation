"""Condition types and operators for assertions."""

from enum import Enum
from typing import Union, Optional
from dataclasses import dataclass


class Operator(Enum):
    """Comparison operators for assertion conditions."""
    GTE = ">="      # Greater than or equal
    LTE = "<="      # Less than or equal
    GT = ">"        # Greater than
    LT = "<"        # Less than
    EQ = "=="       # Equal
    NEQ = "!="      # Not equal
    BETWEEN = "between"  # Between two values


class MetricType(Enum):
    """Types of metrics that can be evaluated in assertions."""
    PASS_RATE = "pass_rate"           # Percentage of passing evaluations
    AVG_SCORE = "avg_score"           # Average score (for numeric outputs)
    MIN_SCORE = "min_score"           # Minimum score
    MAX_SCORE = "max_score"           # Maximum score
    FAILED_COUNT = "failed_count"     # Number of failures
    PASSED_COUNT = "passed_count"     # Number of passes
    TOTAL_COUNT = "total_count"       # Total evaluations
    P50_SCORE = "p50_score"           # 50th percentile
    P90_SCORE = "p90_score"           # 90th percentile
    P95_SCORE = "p95_score"           # 95th percentile
    RUNTIME_AVG = "runtime_avg"       # Average runtime in ms
    RUNTIME_P95 = "runtime_p95"       # 95th percentile runtime
    # Global metrics (across all templates)
    TOTAL_PASS_RATE = "total_pass_rate"


@dataclass
class Condition:
    """A single assertion condition."""
    metric: MetricType
    operator: Operator
    value: Union[float, int]
    value2: Optional[Union[float, int]] = None  # For BETWEEN operator

    def evaluate(self, actual_value: float) -> bool:
        """Evaluate if the condition passes.

        Args:
            actual_value: The actual metric value to compare against.

        Returns:
            True if the condition passes, False otherwise.
        """
        if self.operator == Operator.GTE:
            return actual_value >= self.value
        elif self.operator == Operator.LTE:
            return actual_value <= self.value
        elif self.operator == Operator.GT:
            return actual_value > self.value
        elif self.operator == Operator.LT:
            return actual_value < self.value
        elif self.operator == Operator.EQ:
            return actual_value == self.value
        elif self.operator == Operator.NEQ:
            return actual_value != self.value
        elif self.operator == Operator.BETWEEN:
            if self.value2 is None:
                return False
            return self.value <= actual_value <= self.value2
        return False

    def __str__(self) -> str:
        """String representation of the condition."""
        if self.operator == Operator.BETWEEN:
            return f"{self.value} <= {self.metric.value} <= {self.value2}"
        return f"{self.metric.value} {self.operator.value} {self.value}"
