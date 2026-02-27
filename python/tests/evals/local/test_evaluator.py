"""Tests for the local evaluator."""

import pytest

from fi.evals.local.evaluator import (
    LocalEvaluator,
    LocalEvaluatorConfig,
    LocalEvaluationResult,
    HybridEvaluator,
)
from fi.evals.local.execution_mode import ExecutionMode


class TestLocalEvaluatorConfig:
    """Tests for LocalEvaluatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LocalEvaluatorConfig()

        assert config.execution_mode == ExecutionMode.HYBRID
        assert config.fail_on_unsupported is False
        assert config.parallel_workers == 4
        assert config.timeout == 60

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LocalEvaluatorConfig(
            execution_mode=ExecutionMode.LOCAL,
            fail_on_unsupported=True,
            parallel_workers=8,
            timeout=120,
        )

        assert config.execution_mode == ExecutionMode.LOCAL
        assert config.fail_on_unsupported is True
        assert config.parallel_workers == 8
        assert config.timeout == 120


class TestLocalEvaluator:
    """Tests for the LocalEvaluator class."""

    def test_can_run_locally(self):
        """Test the can_run_locally method."""
        evaluator = LocalEvaluator()

        assert evaluator.can_run_locally("contains") is True
        assert evaluator.can_run_locally("is_json") is True
        assert evaluator.can_run_locally("groundedness") is False

    def test_list_available_metrics(self):
        """Test listing available metrics."""
        evaluator = LocalEvaluator()
        metrics = evaluator.list_available_metrics()

        assert "contains" in metrics
        assert "is_json" in metrics
        assert "bleu_score" in metrics
        assert isinstance(metrics, list)
        assert metrics == sorted(metrics)


class TestLocalEvaluatorEvaluate:
    """Tests for LocalEvaluator.evaluate method."""

    def test_evaluate_contains_metric(self):
        """Test evaluating the contains metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="contains",
            inputs=[{"response": "Hello world"}],
            config={"keyword": "world"},
        )

        assert len(result.results.eval_results) == 1
        assert result.results.eval_results[0].output == 1.0
        assert "contains" in result.executed_locally
        assert len(result.skipped) == 0

    def test_evaluate_contains_metric_not_found(self):
        """Test evaluating contains when keyword is not found."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="contains",
            inputs=[{"response": "Hello world"}],
            config={"keyword": "python"},
        )

        assert len(result.results.eval_results) == 1
        assert result.results.eval_results[0].output == 0.0

    def test_evaluate_is_json_metric(self):
        """Test evaluating the is_json metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="is_json",
            inputs=[
                {"response": '{"key": "value"}'},
                {"response": "not json"},
            ],
        )

        assert len(result.results.eval_results) == 2
        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0

    def test_evaluate_equals_metric(self):
        """Test evaluating the equals metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="equals",
            inputs=[
                {"response": "hello", "expected_response": "hello"},
                {"response": "hello", "expected_response": "HELLO"},
            ],
        )

        assert len(result.results.eval_results) == 2
        # Default is case insensitive
        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 1.0

    def test_evaluate_regex_metric(self):
        """Test evaluating the regex metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="regex",
            inputs=[
                {"response": "My email is test@example.com"},
                {"response": "No email here"},
            ],
            config={"pattern": r"\w+@\w+\.\w+"},
        )

        assert len(result.results.eval_results) == 2
        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0

    def test_evaluate_unsupported_metric_skipped(self):
        """Test that unsupported metrics are skipped by default."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="groundedness",
            inputs=[{"response": "test"}],
        )

        assert len(result.results.eval_results) == 1
        assert result.results.eval_results[0].output is None
        assert "groundedness" in result.skipped
        assert len(result.executed_locally) == 0

    def test_evaluate_unsupported_metric_raises_when_configured(self):
        """Test that unsupported metrics raise when fail_on_unsupported is True."""
        config = LocalEvaluatorConfig(fail_on_unsupported=True)
        evaluator = LocalEvaluator(config=config)

        with pytest.raises(ValueError, match="cannot run locally"):
            evaluator.evaluate(
                metric_name="groundedness",
                inputs=[{"response": "test"}],
            )

    def test_evaluate_with_invalid_input(self):
        """Test handling of invalid inputs."""
        evaluator = LocalEvaluator()

        # Missing required field
        result = evaluator.evaluate(
            metric_name="contains",
            inputs=[{"wrong_field": "test"}],
            config={"keyword": "test"},
        )

        assert len(result.results.eval_results) == 1
        # Should have an error reason
        assert result.results.eval_results[0].output == 0.0
        assert "validation failed" in result.results.eval_results[0].reason.lower()


class TestLocalEvaluatorBatch:
    """Tests for LocalEvaluator.evaluate_batch method."""

    def test_evaluate_batch_multiple_metrics(self):
        """Test evaluating multiple metrics in batch."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate_batch([
            {
                "metric_name": "contains",
                "inputs": [{"response": "Hello world"}],
                "config": {"keyword": "world"},
            },
            {
                "metric_name": "is_json",
                "inputs": [{"response": '{"key": "value"}'}],
            },
        ])

        assert len(result.results.eval_results) == 2
        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 1.0
        assert "contains" in result.executed_locally
        assert "is_json" in result.executed_locally

    def test_evaluate_batch_mixed_local_and_cloud(self):
        """Test batch with both local and cloud metrics."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate_batch([
            {
                "metric_name": "contains",
                "inputs": [{"response": "Hello world"}],
                "config": {"keyword": "world"},
            },
            {
                "metric_name": "groundedness",
                "inputs": [{"response": "test"}],
            },
        ])

        assert len(result.results.eval_results) == 2
        assert "contains" in result.executed_locally
        assert "groundedness" in result.skipped

    def test_evaluate_batch_empty_list(self):
        """Test batch evaluation with empty list."""
        evaluator = LocalEvaluator()
        result = evaluator.evaluate_batch([])

        assert len(result.results.eval_results) == 0


class TestHybridEvaluator:
    """Tests for the HybridEvaluator class."""

    def test_route_evaluation_local_metric(self):
        """Test routing a local-capable metric."""
        evaluator = HybridEvaluator()
        mode = evaluator.route_evaluation("contains")
        assert mode == ExecutionMode.LOCAL

    def test_route_evaluation_cloud_metric(self):
        """Test routing an LLM metric."""
        evaluator = HybridEvaluator()
        mode = evaluator.route_evaluation("groundedness")
        assert mode == ExecutionMode.CLOUD

    def test_partition_evaluations(self):
        """Test partitioning evaluations by mode."""
        evaluator = HybridEvaluator()

        evaluations = [
            {"metric_name": "contains", "inputs": [{"response": "test"}]},
            {"metric_name": "is_json", "inputs": [{"response": "{}"}]},
            {"metric_name": "groundedness", "inputs": [{"response": "test"}]},
            {"metric_name": "hallucination", "inputs": [{"response": "test"}]},
        ]

        partitions = evaluator.partition_evaluations(evaluations)

        assert len(partitions[ExecutionMode.LOCAL]) == 2
        assert len(partitions[ExecutionMode.CLOUD]) == 2

        local_metrics = [e["metric_name"] for e in partitions[ExecutionMode.LOCAL]]
        assert "contains" in local_metrics
        assert "is_json" in local_metrics

        cloud_metrics = [e["metric_name"] for e in partitions[ExecutionMode.CLOUD]]
        assert "groundedness" in cloud_metrics
        assert "hallucination" in cloud_metrics

    def test_evaluate_local_partition(self):
        """Test evaluating the local partition."""
        evaluator = HybridEvaluator()

        evaluations = [
            {
                "metric_name": "contains",
                "inputs": [{"response": "Hello world"}],
                "config": {"keyword": "world"},
            },
        ]

        result = evaluator.evaluate_local_partition(evaluations)

        assert len(result.results.eval_results) == 1
        assert result.results.eval_results[0].output == 1.0


class TestLengthMetrics:
    """Tests for length-based metrics."""

    def test_length_less_than(self):
        """Test length_less_than metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="length_less_than",
            inputs=[
                {"response": "short"},
                {"response": "this is a much longer response"},
            ],
            config={"max_length": 10},
        )

        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0

    def test_length_greater_than(self):
        """Test length_greater_than metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="length_greater_than",
            inputs=[
                {"response": "short"},
                {"response": "this is a much longer response"},
            ],
            config={"min_length": 10},
        )

        assert result.results.eval_results[0].output == 0.0
        assert result.results.eval_results[1].output == 1.0

    def test_length_between(self):
        """Test length_between metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="length_between",
            inputs=[
                {"response": "short"},  # 5 chars
                {"response": "medium text"},  # 11 chars
                {"response": "this is a much longer response"},  # 31 chars
            ],
            config={"min_length": 8, "max_length": 20},
        )

        assert result.results.eval_results[0].output == 0.0  # too short
        assert result.results.eval_results[1].output == 1.0  # in range
        assert result.results.eval_results[2].output == 0.0  # too long


class TestStringMetrics:
    """Tests for string-based metrics."""

    def test_starts_with(self):
        """Test starts_with metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="starts_with",
            inputs=[
                {"response": "Hello world", "expected_response": "Hello"},
                {"response": "Hello world", "expected_response": "World"},
            ],
        )

        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0

    def test_ends_with(self):
        """Test ends_with metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="ends_with",
            inputs=[
                {"response": "Hello world", "expected_response": "world"},
                {"response": "Hello world", "expected_response": "Hello"},
            ],
        )

        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0

    def test_one_line(self):
        """Test one_line metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="one_line",
            inputs=[
                {"response": "single line"},
                {"response": "line 1\nline 2"},
            ],
        )

        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0

    def test_contains_all(self):
        """Test contains_all metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="contains_all",
            inputs=[
                {"response": "The quick brown fox"},
                {"response": "The quick fox"},
            ],
            config={"keywords": ["quick", "brown", "fox"]},
        )

        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0

    def test_contains_any(self):
        """Test contains_any metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="contains_any",
            inputs=[
                {"response": "I love python"},
                {"response": "I love java"},
            ],
            config={"keywords": ["python", "rust"]},
        )

        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0

    def test_contains_none(self):
        """Test contains_none metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="contains_none",
            inputs=[
                {"response": "This is clean text"},
                {"response": "This has a bad word"},
            ],
            config={"keywords": ["bad", "evil"]},
        )

        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0


class TestJsonMetrics:
    """Tests for JSON-based metrics."""

    def test_contains_json(self):
        """Test contains_json metric."""
        evaluator = LocalEvaluator()

        result = evaluator.evaluate(
            metric_name="contains_json",
            inputs=[
                {"response": "The result is {\"key\": \"value\"} here"},
                {"response": "No JSON here"},
            ],
        )

        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0

    def test_json_schema_validation(self):
        """Test json_schema metric."""
        evaluator = LocalEvaluator()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }

        result = evaluator.evaluate(
            metric_name="json_schema",
            inputs=[
                {"response": '{"name": "John", "age": 30}', "schema": schema},
                {"response": '{"age": 30}', "schema": schema},  # missing required
            ],
        )

        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 0.0
