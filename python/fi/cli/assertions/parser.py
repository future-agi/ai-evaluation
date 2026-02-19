"""Parser for assertion condition strings."""

import re
from typing import List

from .conditions import Condition, Operator, MetricType


class ConditionParseError(ValueError):
    """Error raised when a condition string cannot be parsed."""
    pass


class ConditionParser:
    """Parse string conditions into Condition objects."""

    # Pattern: metric_name operator value
    PATTERN = re.compile(
        r'^(\w+)\s*(>=|<=|==|!=|>|<)\s*([\d.]+)$'
    )

    # Pattern for between: value <= metric <= value
    BETWEEN_PATTERN = re.compile(
        r'^([\d.]+)\s*<=\s*(\w+)\s*<=\s*([\d.]+)$'
    )

    METRIC_MAP = {
        'pass_rate': MetricType.PASS_RATE,
        'avg_score': MetricType.AVG_SCORE,
        'min_score': MetricType.MIN_SCORE,
        'max_score': MetricType.MAX_SCORE,
        'failed_count': MetricType.FAILED_COUNT,
        'passed_count': MetricType.PASSED_COUNT,
        'total_count': MetricType.TOTAL_COUNT,
        'p50_score': MetricType.P50_SCORE,
        'p90_score': MetricType.P90_SCORE,
        'p95_score': MetricType.P95_SCORE,
        'runtime_avg': MetricType.RUNTIME_AVG,
        'runtime_p95': MetricType.RUNTIME_P95,
        'total_pass_rate': MetricType.TOTAL_PASS_RATE,
    }

    OPERATOR_MAP = {
        '>=': Operator.GTE,
        '<=': Operator.LTE,
        '>': Operator.GT,
        '<': Operator.LT,
        '==': Operator.EQ,
        '!=': Operator.NEQ,
    }

    @classmethod
    def parse(cls, condition_str: str) -> Condition:
        """Parse a condition string into a Condition object.

        Supports two formats:
        1. Standard: "metric_name operator value" (e.g., "pass_rate >= 0.85")
        2. Between: "value1 <= metric_name <= value2" (e.g., "0.5 <= avg_score <= 1.0")

        Args:
            condition_str: The condition string to parse.

        Returns:
            A Condition object.

        Raises:
            ConditionParseError: If the condition string is invalid.
        """
        condition_str = condition_str.strip()

        # Try between pattern first
        between_match = cls.BETWEEN_PATTERN.match(condition_str)
        if between_match:
            value1, metric, value2 = between_match.groups()

            if metric not in cls.METRIC_MAP:
                raise ConditionParseError(
                    f"Unknown metric: {metric}. "
                    f"Available metrics: {', '.join(sorted(cls.METRIC_MAP.keys()))}"
                )

            return Condition(
                metric=cls.METRIC_MAP[metric],
                operator=Operator.BETWEEN,
                value=float(value1),
                value2=float(value2)
            )

        # Try standard pattern
        match = cls.PATTERN.match(condition_str)
        if not match:
            raise ConditionParseError(
                f"Invalid condition format: '{condition_str}'. "
                f"Expected format: 'metric operator value' (e.g., 'pass_rate >= 0.85')"
            )

        metric_str, operator_str, value_str = match.groups()

        if metric_str not in cls.METRIC_MAP:
            raise ConditionParseError(
                f"Unknown metric: {metric_str}. "
                f"Available metrics: {', '.join(sorted(cls.METRIC_MAP.keys()))}"
            )

        return Condition(
            metric=cls.METRIC_MAP[metric_str],
            operator=cls.OPERATOR_MAP[operator_str],
            value=float(value_str)
        )

    @classmethod
    def parse_many(cls, conditions: List[str]) -> List[Condition]:
        """Parse multiple condition strings.

        Args:
            conditions: List of condition strings to parse.

        Returns:
            List of Condition objects.
        """
        return [cls.parse(c) for c in conditions]

    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Get list of available metric names."""
        return sorted(cls.METRIC_MAP.keys())

    @classmethod
    def get_available_operators(cls) -> List[str]:
        """Get list of available operators."""
        return list(cls.OPERATOR_MAP.keys()) + ['between']
