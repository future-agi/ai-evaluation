"""Tests for the assertions module."""

import pytest
from fi.cli.assertions import (
    Condition,
    Operator,
    MetricType,
    ConditionParser,
    AssertionEvaluator,
    AssertionResult,
    AssertionOutcome,
    AssertionReport,
    AssertionReporter,
    ExitCode,
)
from fi.cli.assertions.parser import ConditionParseError


class TestCondition:
    """Tests for Condition class."""

    def test_gte_operator(self):
        """Test greater than or equal operator."""
        condition = Condition(
            metric=MetricType.PASS_RATE,
            operator=Operator.GTE,
            value=0.8
        )
        assert condition.evaluate(0.85) is True
        assert condition.evaluate(0.8) is True
        assert condition.evaluate(0.79) is False

    def test_lte_operator(self):
        """Test less than or equal operator."""
        condition = Condition(
            metric=MetricType.FAILED_COUNT,
            operator=Operator.LTE,
            value=5
        )
        assert condition.evaluate(3) is True
        assert condition.evaluate(5) is True
        assert condition.evaluate(6) is False

    def test_gt_operator(self):
        """Test greater than operator."""
        condition = Condition(
            metric=MetricType.AVG_SCORE,
            operator=Operator.GT,
            value=0.7
        )
        assert condition.evaluate(0.71) is True
        assert condition.evaluate(0.7) is False
        assert condition.evaluate(0.69) is False

    def test_lt_operator(self):
        """Test less than operator."""
        condition = Condition(
            metric=MetricType.RUNTIME_AVG,
            operator=Operator.LT,
            value=1000
        )
        assert condition.evaluate(999) is True
        assert condition.evaluate(1000) is False
        assert condition.evaluate(1001) is False

    def test_eq_operator(self):
        """Test equal operator."""
        condition = Condition(
            metric=MetricType.TOTAL_COUNT,
            operator=Operator.EQ,
            value=100
        )
        assert condition.evaluate(100) is True
        assert condition.evaluate(99) is False
        assert condition.evaluate(101) is False

    def test_neq_operator(self):
        """Test not equal operator."""
        condition = Condition(
            metric=MetricType.FAILED_COUNT,
            operator=Operator.NEQ,
            value=0
        )
        assert condition.evaluate(1) is True
        assert condition.evaluate(0) is False

    def test_between_operator(self):
        """Test between operator."""
        condition = Condition(
            metric=MetricType.AVG_SCORE,
            operator=Operator.BETWEEN,
            value=0.5,
            value2=0.9
        )
        assert condition.evaluate(0.7) is True
        assert condition.evaluate(0.5) is True
        assert condition.evaluate(0.9) is True
        assert condition.evaluate(0.49) is False
        assert condition.evaluate(0.91) is False

    def test_between_without_value2(self):
        """Test between operator without value2 returns False."""
        condition = Condition(
            metric=MetricType.AVG_SCORE,
            operator=Operator.BETWEEN,
            value=0.5,
            value2=None
        )
        assert condition.evaluate(0.7) is False

    def test_str_representation(self):
        """Test string representation of conditions."""
        condition = Condition(
            metric=MetricType.PASS_RATE,
            operator=Operator.GTE,
            value=0.85
        )
        assert str(condition) == "pass_rate >= 0.85"

        between_condition = Condition(
            metric=MetricType.AVG_SCORE,
            operator=Operator.BETWEEN,
            value=0.5,
            value2=0.9
        )
        assert str(between_condition) == "0.5 <= avg_score <= 0.9"


class TestConditionParser:
    """Tests for ConditionParser class."""

    def test_parse_gte(self):
        """Test parsing >= condition."""
        condition = ConditionParser.parse("pass_rate >= 0.85")
        assert condition.metric == MetricType.PASS_RATE
        assert condition.operator == Operator.GTE
        assert condition.value == 0.85

    def test_parse_lte(self):
        """Test parsing <= condition."""
        condition = ConditionParser.parse("failed_count <= 5")
        assert condition.metric == MetricType.FAILED_COUNT
        assert condition.operator == Operator.LTE
        assert condition.value == 5.0

    def test_parse_gt(self):
        """Test parsing > condition."""
        condition = ConditionParser.parse("avg_score > 0.7")
        assert condition.metric == MetricType.AVG_SCORE
        assert condition.operator == Operator.GT
        assert condition.value == 0.7

    def test_parse_lt(self):
        """Test parsing < condition."""
        condition = ConditionParser.parse("runtime_avg < 1000")
        assert condition.metric == MetricType.RUNTIME_AVG
        assert condition.operator == Operator.LT
        assert condition.value == 1000.0

    def test_parse_eq(self):
        """Test parsing == condition."""
        condition = ConditionParser.parse("total_count == 100")
        assert condition.metric == MetricType.TOTAL_COUNT
        assert condition.operator == Operator.EQ
        assert condition.value == 100.0

    def test_parse_neq(self):
        """Test parsing != condition."""
        condition = ConditionParser.parse("failed_count != 0")
        assert condition.metric == MetricType.FAILED_COUNT
        assert condition.operator == Operator.NEQ
        assert condition.value == 0.0

    def test_parse_between(self):
        """Test parsing between condition."""
        condition = ConditionParser.parse("0.5 <= avg_score <= 0.9")
        assert condition.metric == MetricType.AVG_SCORE
        assert condition.operator == Operator.BETWEEN
        assert condition.value == 0.5
        assert condition.value2 == 0.9

    def test_parse_with_whitespace(self):
        """Test parsing with various whitespace."""
        condition = ConditionParser.parse("  pass_rate  >=  0.8  ")
        assert condition.metric == MetricType.PASS_RATE
        assert condition.operator == Operator.GTE
        assert condition.value == 0.8

    def test_parse_all_metrics(self):
        """Test parsing all metric types."""
        metrics = [
            ("pass_rate >= 0.8", MetricType.PASS_RATE),
            ("avg_score >= 0.7", MetricType.AVG_SCORE),
            ("min_score >= 0.5", MetricType.MIN_SCORE),
            ("max_score <= 1.0", MetricType.MAX_SCORE),
            ("failed_count <= 5", MetricType.FAILED_COUNT),
            ("passed_count >= 95", MetricType.PASSED_COUNT),
            ("total_count == 100", MetricType.TOTAL_COUNT),
            ("p50_score >= 0.6", MetricType.P50_SCORE),
            ("p90_score >= 0.8", MetricType.P90_SCORE),
            ("p95_score >= 0.85", MetricType.P95_SCORE),
            ("runtime_avg < 500", MetricType.RUNTIME_AVG),
            ("runtime_p95 < 1000", MetricType.RUNTIME_P95),
            ("total_pass_rate >= 0.9", MetricType.TOTAL_PASS_RATE),
        ]
        for condition_str, expected_metric in metrics:
            condition = ConditionParser.parse(condition_str)
            assert condition.metric == expected_metric

    def test_parse_invalid_format(self):
        """Test parsing invalid format raises error."""
        with pytest.raises(ConditionParseError) as exc_info:
            ConditionParser.parse("invalid condition")
        assert "Invalid condition format" in str(exc_info.value)

    def test_parse_unknown_metric(self):
        """Test parsing unknown metric raises error."""
        with pytest.raises(ConditionParseError) as exc_info:
            ConditionParser.parse("unknown_metric >= 0.5")
        assert "Unknown metric" in str(exc_info.value)

    def test_parse_many(self):
        """Test parsing multiple conditions."""
        conditions = ConditionParser.parse_many([
            "pass_rate >= 0.8",
            "avg_score >= 0.7",
            "failed_count <= 5"
        ])
        assert len(conditions) == 3
        assert conditions[0].metric == MetricType.PASS_RATE
        assert conditions[1].metric == MetricType.AVG_SCORE
        assert conditions[2].metric == MetricType.FAILED_COUNT

    def test_get_available_metrics(self):
        """Test getting available metrics."""
        metrics = ConditionParser.get_available_metrics()
        assert "pass_rate" in metrics
        assert "avg_score" in metrics
        assert len(metrics) == 13

    def test_get_available_operators(self):
        """Test getting available operators."""
        operators = ConditionParser.get_available_operators()
        assert ">=" in operators
        assert "between" in operators


class TestAssertionEvaluator:
    """Tests for AssertionEvaluator class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results."""
        return {
            "eval_results": [
                {"name": "groundedness", "output": True, "runtime": 100},
                {"name": "groundedness", "output": True, "runtime": 150},
                {"name": "groundedness", "output": False, "runtime": 120},
                {"name": "context_adherence", "output": 0.9, "runtime": 200},
                {"name": "context_adherence", "output": 0.8, "runtime": 180},
                {"name": "context_adherence", "output": 0.7, "runtime": 220},
            ]
        }

    @pytest.fixture
    def mixed_results(self):
        """Create results with mixed boolean and numeric outputs."""
        return {
            "eval_results": [
                {"name": "test_eval", "output": True, "runtime": 100},
                {"name": "test_eval", "output": 0.9, "runtime": 150},
                {"name": "test_eval", "output": 0.3, "runtime": 120},  # Below 0.5 threshold
                {"name": "test_eval", "output": False, "runtime": 180},
            ]
        }

    def test_compute_metrics_for_template(self, sample_results):
        """Test computing metrics for a specific template."""
        evaluator = AssertionEvaluator(sample_results, {})
        metrics = evaluator.compute_metrics("groundedness")

        assert metrics[MetricType.TOTAL_COUNT] == 3
        assert metrics[MetricType.PASSED_COUNT] == 2  # 2 True
        assert metrics[MetricType.FAILED_COUNT] == 1  # 1 False
        assert metrics[MetricType.PASS_RATE] == pytest.approx(2/3, rel=0.01)

    def test_compute_metrics_global(self, sample_results):
        """Test computing global metrics."""
        evaluator = AssertionEvaluator(sample_results, {})
        metrics = evaluator.compute_metrics()

        assert metrics[MetricType.TOTAL_COUNT] == 6
        # 2 True bools + 3 numeric >= 0.5
        assert metrics[MetricType.PASSED_COUNT] == 5
        assert metrics[MetricType.FAILED_COUNT] == 1

    def test_compute_numeric_metrics(self, sample_results):
        """Test computing numeric score metrics."""
        evaluator = AssertionEvaluator(sample_results, {})
        metrics = evaluator.compute_metrics("context_adherence")

        assert metrics[MetricType.AVG_SCORE] == pytest.approx(0.8, rel=0.01)
        assert metrics[MetricType.MIN_SCORE] == 0.7
        assert metrics[MetricType.MAX_SCORE] == 0.9

    def test_compute_runtime_metrics(self, sample_results):
        """Test computing runtime metrics."""
        evaluator = AssertionEvaluator(sample_results, {})
        metrics = evaluator.compute_metrics("groundedness")

        assert MetricType.RUNTIME_AVG in metrics
        assert metrics[MetricType.RUNTIME_AVG] == pytest.approx(123.33, rel=0.01)

    def test_compute_metrics_mixed_outputs(self, mixed_results):
        """Test computing metrics with mixed boolean and numeric outputs."""
        evaluator = AssertionEvaluator(mixed_results, {})
        metrics = evaluator.compute_metrics("test_eval")

        # True, 0.9 (pass), 0.3 (fail), False
        assert metrics[MetricType.TOTAL_COUNT] == 4
        assert metrics[MetricType.PASSED_COUNT] == 2  # True + 0.9
        assert metrics[MetricType.FAILED_COUNT] == 2  # False + 0.3

    def test_evaluate_assertion_pass(self, sample_results):
        """Test evaluating a passing assertion."""
        evaluator = AssertionEvaluator(sample_results, {})
        outcome = evaluator.evaluate_assertion(
            template="groundedness",
            condition_str="pass_rate >= 0.6"
        )

        assert outcome.result == AssertionResult.PASSED
        assert outcome.template == "groundedness"
        assert "passed" in outcome.message.lower()

    def test_evaluate_assertion_fail(self, sample_results):
        """Test evaluating a failing assertion."""
        evaluator = AssertionEvaluator(sample_results, {})
        outcome = evaluator.evaluate_assertion(
            template="groundedness",
            condition_str="pass_rate >= 0.9"
        )

        assert outcome.result == AssertionResult.FAILED
        assert "failed" in outcome.message.lower()

    def test_evaluate_assertion_warning(self, sample_results):
        """Test evaluating an assertion with warning on_fail."""
        evaluator = AssertionEvaluator(sample_results, {})
        outcome = evaluator.evaluate_assertion(
            template="groundedness",
            condition_str="pass_rate >= 0.9",
            on_fail="warn"
        )

        assert outcome.result == AssertionResult.WARNING
        assert "warning" in outcome.message.lower()

    def test_evaluate_assertion_skip(self, sample_results):
        """Test evaluating an assertion with skip on_fail."""
        evaluator = AssertionEvaluator(sample_results, {})
        outcome = evaluator.evaluate_assertion(
            template="groundedness",
            condition_str="pass_rate >= 0.9",
            on_fail="skip"
        )

        assert outcome.result == AssertionResult.SKIPPED

    def test_evaluate_assertion_unknown_metric(self, sample_results):
        """Test evaluating assertion for unavailable metric."""
        evaluator = AssertionEvaluator(sample_results, {})
        outcome = evaluator.evaluate_assertion(
            template="groundedness",
            condition_str="avg_score >= 0.7"  # groundedness has boolean outputs
        )

        assert outcome.result == AssertionResult.SKIPPED
        assert "not available" in outcome.message.lower()

    def test_evaluate_all_explicit_assertions(self, sample_results):
        """Test evaluating all explicit assertions."""
        config = {
            "assertions": [
                {
                    "template": "groundedness",
                    "conditions": ["pass_rate >= 0.6"],
                    "on_fail": "error"
                },
                {
                    "template": "context_adherence",
                    "conditions": ["avg_score >= 0.7", "pass_rate >= 0.9"],
                    "on_fail": "error"
                }
            ]
        }

        evaluator = AssertionEvaluator(sample_results, config)
        report = evaluator.evaluate_all()

        assert report.total_assertions == 3
        assert report.passed >= 2  # pass_rate >= 0.6 and avg_score >= 0.7

    def test_evaluate_all_global_assertions(self, sample_results):
        """Test evaluating global assertions."""
        config = {
            "assertions": [
                {
                    "global": True,
                    "conditions": ["total_pass_rate >= 0.7"],
                    "on_fail": "error"
                }
            ]
        }

        evaluator = AssertionEvaluator(sample_results, config)
        report = evaluator.evaluate_all()

        assert report.total_assertions == 1

    def test_evaluate_all_with_thresholds(self, sample_results):
        """Test evaluating threshold shortcuts."""
        config = {
            "thresholds": {
                "default_pass_rate": 0.5,
                "overrides": {
                    "context_adherence": 0.9  # Higher threshold for this template
                }
            }
        }

        evaluator = AssertionEvaluator(sample_results, config)
        report = evaluator.evaluate_all()

        # Should have assertions for both templates
        assert report.total_assertions == 2

    def test_evaluate_empty_results(self):
        """Test evaluating with empty results."""
        evaluator = AssertionEvaluator({"eval_results": []}, {})
        metrics = evaluator.compute_metrics()
        assert metrics == {}

    def test_evaluate_no_assertions(self, sample_results):
        """Test evaluating with no assertions configured."""
        evaluator = AssertionEvaluator(sample_results, {})
        report = evaluator.evaluate_all()

        assert report.total_assertions == 0
        assert report.all_passed is True


class TestAssertionReport:
    """Tests for AssertionReport class."""

    def test_all_passed_true(self):
        """Test all_passed when no failures."""
        report = AssertionReport(
            outcomes=[],
            total_assertions=5,
            passed=5,
            failed=0,
            warnings=0,
            skipped=0
        )
        assert report.all_passed is True

    def test_all_passed_false(self):
        """Test all_passed when there are failures."""
        report = AssertionReport(
            outcomes=[],
            total_assertions=5,
            passed=4,
            failed=1,
            warnings=0,
            skipped=0
        )
        assert report.all_passed is False

    def test_has_warnings(self):
        """Test has_warnings property."""
        report_with_warnings = AssertionReport(
            outcomes=[],
            total_assertions=5,
            passed=4,
            failed=0,
            warnings=1,
            skipped=0
        )
        assert report_with_warnings.has_warnings is True

        report_without_warnings = AssertionReport(
            outcomes=[],
            total_assertions=5,
            passed=5,
            failed=0,
            warnings=0,
            skipped=0
        )
        assert report_without_warnings.has_warnings is False


class TestAssertionReporter:
    """Tests for AssertionReporter class."""

    @pytest.fixture
    def sample_report(self):
        """Create sample assertion report."""
        return AssertionReport(
            outcomes=[
                AssertionOutcome(
                    template="groundedness",
                    condition="pass_rate >= 0.8",
                    expected=">= 0.8",
                    actual=0.85,
                    result=AssertionResult.PASSED,
                    message="Assertion passed"
                ),
                AssertionOutcome(
                    template="context_adherence",
                    condition="avg_score >= 0.9",
                    expected=">= 0.9",
                    actual=0.75,
                    result=AssertionResult.FAILED,
                    message="Failed: avg_score is 0.75, expected >= 0.9"
                ),
            ],
            total_assertions=2,
            passed=1,
            failed=1,
            warnings=0,
            skipped=0
        )

    def test_to_json(self, sample_report):
        """Test JSON conversion."""
        from rich.console import Console
        reporter = AssertionReporter(Console())
        json_data = reporter.to_json(sample_report)

        assert json_data["summary"]["total"] == 2
        assert json_data["summary"]["passed"] == 1
        assert json_data["summary"]["failed"] == 1
        assert json_data["summary"]["all_passed"] is False
        assert len(json_data["assertions"]) == 2
        assert json_data["assertions"][0]["template"] == "groundedness"
        assert json_data["assertions"][0]["result"] == "passed"

    def test_to_junit(self, sample_report):
        """Test JUnit XML conversion."""
        from rich.console import Console
        reporter = AssertionReporter(Console())
        junit_xml = reporter.to_junit(sample_report)

        assert '<?xml version="1.0" ?>' in junit_xml
        assert '<testsuites' in junit_xml
        assert 'tests="2"' in junit_xml
        assert 'failures="1"' in junit_xml
        assert '<testcase' in junit_xml
        assert 'groundedness' in junit_xml
        assert '<failure' in junit_xml


class TestExitCode:
    """Tests for ExitCode enum."""

    def test_success_code(self):
        """Test SUCCESS exit code."""
        assert ExitCode.SUCCESS == 0

    def test_assertion_failed_code(self):
        """Test ASSERTION_FAILED exit code."""
        assert ExitCode.ASSERTION_FAILED == 2

    def test_assertion_warning_code(self):
        """Test ASSERTION_WARNING exit code."""
        assert ExitCode.ASSERTION_WARNING == 3

    def test_all_codes_unique(self):
        """Test all exit codes are unique."""
        codes = [e.value for e in ExitCode]
        assert len(codes) == len(set(codes))
