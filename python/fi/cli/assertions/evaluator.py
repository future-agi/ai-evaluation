"""Assertion evaluator for checking conditions against evaluation results."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .conditions import Condition, MetricType
from .parser import ConditionParser, ConditionParseError


class AssertionResult(Enum):
    """Result status of an assertion evaluation."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class AssertionOutcome:
    """Result of evaluating a single assertion."""
    template: Optional[str]
    condition: str
    expected: str
    actual: float
    result: AssertionResult
    message: str


@dataclass
class AssertionReport:
    """Complete assertion evaluation report."""
    outcomes: List[AssertionOutcome] = field(default_factory=list)
    total_assertions: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    skipped: int = 0

    @property
    def all_passed(self) -> bool:
        """Check if all assertions passed (no failures)."""
        return self.failed == 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return self.warnings > 0


class AssertionEvaluator:
    """Evaluate assertions against evaluation results."""

    def __init__(self, results: Dict[str, Any], config: Dict[str, Any]):
        """Initialize the evaluator.

        Args:
            results: Dictionary containing evaluation results with 'eval_results' key.
            config: Configuration dictionary that may contain 'assertions' and 'thresholds'.
        """
        self.results = results
        self.config = config
        self.outcomes: List[AssertionOutcome] = []

    def compute_metrics(self, template: Optional[str] = None) -> Dict[MetricType, float]:
        """Compute all metrics for a template or globally.

        Args:
            template: If provided, compute metrics only for this template.
                     If None, compute global metrics across all results.

        Returns:
            Dictionary mapping MetricType to computed values.
        """
        if template:
            eval_results = [
                r for r in self.results.get('eval_results', [])
                if r.get('name') == template
            ]
        else:
            eval_results = self.results.get('eval_results', [])

        if not eval_results:
            return {}

        # Extract scores/outputs
        outputs = [r.get('output') for r in eval_results]
        runtimes = [r.get('runtime', 0) for r in eval_results if r.get('runtime')]

        # Calculate boolean pass/fail
        bool_outputs = [o for o in outputs if isinstance(o, bool)]
        numeric_outputs = [
            float(o) for o in outputs
            if isinstance(o, (int, float)) and not isinstance(o, bool)
        ]

        # Pass rate (boolean outputs: True = pass, numeric: >= 0.5 = pass)
        passes = sum(1 for o in bool_outputs if o is True)
        passes += sum(1 for o in numeric_outputs if o >= 0.5)
        total = len(outputs)

        metrics: Dict[MetricType, float] = {
            MetricType.PASS_RATE: passes / total if total > 0 else 0,
            MetricType.TOTAL_PASS_RATE: passes / total if total > 0 else 0,  # Alias for global
            MetricType.PASSED_COUNT: float(passes),
            MetricType.FAILED_COUNT: float(total - passes),
            MetricType.TOTAL_COUNT: float(total),
        }

        # Score metrics (only for numeric outputs)
        if numeric_outputs:
            metrics[MetricType.AVG_SCORE] = statistics.mean(numeric_outputs)
            metrics[MetricType.MIN_SCORE] = min(numeric_outputs)
            metrics[MetricType.MAX_SCORE] = max(numeric_outputs)

            sorted_outputs = sorted(numeric_outputs)
            n = len(sorted_outputs)

            # Percentiles
            metrics[MetricType.P50_SCORE] = self._percentile(sorted_outputs, 50)
            metrics[MetricType.P90_SCORE] = self._percentile(sorted_outputs, 90)
            metrics[MetricType.P95_SCORE] = self._percentile(sorted_outputs, 95)

        # Runtime metrics
        if runtimes:
            metrics[MetricType.RUNTIME_AVG] = statistics.mean(runtimes)
            metrics[MetricType.RUNTIME_P95] = self._percentile(sorted(runtimes), 95)

        return metrics

    def _percentile(self, sorted_data: List[float], p: float) -> float:
        """Calculate percentile from sorted data.

        Args:
            sorted_data: Pre-sorted list of values.
            p: Percentile to calculate (0-100).

        Returns:
            The calculated percentile value.
        """
        if not sorted_data:
            return 0.0

        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]

        # Linear interpolation
        k = (n - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < n else f

        if f == c:
            return sorted_data[f]

        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def evaluate_assertion(
        self,
        template: Optional[str],
        condition_str: str,
        on_fail: str = "error"
    ) -> AssertionOutcome:
        """Evaluate a single assertion condition.

        Args:
            template: Template name to evaluate against, or None for global.
            condition_str: The condition string to evaluate.
            on_fail: Action on failure - "error", "warn", or "skip".

        Returns:
            AssertionOutcome with the result.
        """
        try:
            condition = ConditionParser.parse(condition_str)
        except ConditionParseError as e:
            return AssertionOutcome(
                template=template,
                condition=condition_str,
                expected="valid condition",
                actual=0,
                result=AssertionResult.FAILED,
                message=f"Parse error: {e}"
            )

        metrics = self.compute_metrics(template)

        if condition.metric not in metrics:
            return AssertionOutcome(
                template=template,
                condition=condition_str,
                expected=f"{condition.metric.value} {condition.operator.value} {condition.value}",
                actual=0,
                result=AssertionResult.SKIPPED,
                message=f"Metric '{condition.metric.value}' not available for template '{template or 'global'}'"
            )

        actual_value = metrics[condition.metric]
        passed = condition.evaluate(actual_value)

        if passed:
            result = AssertionResult.PASSED
            message = "Assertion passed"
        elif on_fail == "warn":
            result = AssertionResult.WARNING
            message = (
                f"Warning: {condition.metric.value} is {actual_value:.4f}, "
                f"expected {condition.operator.value} {condition.value}"
            )
        elif on_fail == "skip":
            result = AssertionResult.SKIPPED
            message = "Assertion skipped"
        else:
            result = AssertionResult.FAILED
            message = (
                f"Failed: {condition.metric.value} is {actual_value:.4f}, "
                f"expected {condition.operator.value} {condition.value}"
            )

        return AssertionOutcome(
            template=template,
            condition=condition_str,
            expected=f"{condition.operator.value} {condition.value}",
            actual=actual_value,
            result=result,
            message=message
        )

    def evaluate_all(self) -> AssertionReport:
        """Evaluate all assertions from config.

        Returns:
            AssertionReport containing all outcomes and summary statistics.
        """
        assertions = self.config.get('assertions', [])
        thresholds = self.config.get('thresholds', {})

        outcomes: List[AssertionOutcome] = []

        # Evaluate explicit assertions
        for assertion in assertions:
            template = assertion.get('template')
            is_global = assertion.get('global', False)
            conditions = assertion.get('conditions', [])
            on_fail = assertion.get('on_fail', 'error')

            for condition_str in conditions:
                outcome = self.evaluate_assertion(
                    template=None if is_global else template,
                    condition_str=condition_str,
                    on_fail=on_fail
                )
                outcomes.append(outcome)

        # Evaluate threshold shortcuts
        default_threshold = thresholds.get('default_pass_rate')
        overrides = thresholds.get('overrides', {})

        if default_threshold is not None:
            templates = set(
                r.get('name') for r in self.results.get('eval_results', [])
                if r.get('name')
            )
            for template in templates:
                threshold = overrides.get(template, default_threshold)
                outcome = self.evaluate_assertion(
                    template=template,
                    condition_str=f"pass_rate >= {threshold}",
                    on_fail="error"
                )
                outcomes.append(outcome)

        # Build report
        passed = sum(1 for o in outcomes if o.result == AssertionResult.PASSED)
        failed = sum(1 for o in outcomes if o.result == AssertionResult.FAILED)
        warnings = sum(1 for o in outcomes if o.result == AssertionResult.WARNING)
        skipped = sum(1 for o in outcomes if o.result == AssertionResult.SKIPPED)

        return AssertionReport(
            outcomes=outcomes,
            total_assertions=len(outcomes),
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped
        )
