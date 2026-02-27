"""Tests for the execution mode selector."""

import pytest

from fi.evals.local.execution_mode import (
    ExecutionMode,
    LOCAL_CAPABLE_METRICS,
    can_run_locally,
    select_execution_mode,
)


class TestExecutionMode:
    """Tests for the ExecutionMode enum."""

    def test_execution_mode_values(self):
        """Test that execution modes have correct string values."""
        assert str(ExecutionMode.LOCAL) == "local"
        assert str(ExecutionMode.CLOUD) == "cloud"
        assert str(ExecutionMode.HYBRID) == "hybrid"

    def test_execution_mode_from_string(self):
        """Test creating execution mode from string."""
        assert ExecutionMode("local") == ExecutionMode.LOCAL
        assert ExecutionMode("cloud") == ExecutionMode.CLOUD
        assert ExecutionMode("hybrid") == ExecutionMode.HYBRID


class TestLocalCapableMetrics:
    """Tests for local capable metrics set."""

    def test_string_metrics_are_local_capable(self):
        """Test that string metrics are marked as local capable."""
        string_metrics = [
            "regex", "contains", "contains_all", "contains_any",
            "contains_none", "one_line", "equals", "starts_with",
            "ends_with", "length_less_than", "length_greater_than",
            "length_between",
        ]
        for metric in string_metrics:
            assert metric in LOCAL_CAPABLE_METRICS, f"{metric} should be local capable"

    def test_json_metrics_are_local_capable(self):
        """Test that JSON metrics are marked as local capable."""
        json_metrics = ["contains_json", "is_json", "json_schema"]
        for metric in json_metrics:
            assert metric in LOCAL_CAPABLE_METRICS, f"{metric} should be local capable"

    def test_similarity_metrics_are_local_capable(self):
        """Test that similarity metrics are marked as local capable."""
        similarity_metrics = [
            "bleu_score", "rouge_score", "recall_score",
            "levenshtein_similarity", "numeric_similarity",
            "embedding_similarity", "semantic_list_contains",
        ]
        for metric in similarity_metrics:
            assert metric in LOCAL_CAPABLE_METRICS, f"{metric} should be local capable"


class TestCanRunLocally:
    """Tests for the can_run_locally function."""

    def test_can_run_locally_with_local_metric(self):
        """Test that local capable metrics return True."""
        assert can_run_locally("contains") is True
        assert can_run_locally("is_json") is True
        assert can_run_locally("bleu_score") is True

    def test_can_run_locally_case_insensitive(self):
        """Test that metric name matching is case insensitive."""
        assert can_run_locally("Contains") is True
        assert can_run_locally("CONTAINS") is True
        assert can_run_locally("cOnTaInS") is True

    def test_cannot_run_locally_with_llm_metric(self):
        """Test that LLM metrics return False."""
        assert can_run_locally("groundedness") is False
        assert can_run_locally("context_adherence") is False
        assert can_run_locally("hallucination") is False


class TestSelectExecutionMode:
    """Tests for the select_execution_mode function."""

    def test_force_cloud_always_returns_cloud(self):
        """Test that force_cloud returns CLOUD regardless of other settings."""
        assert select_execution_mode(
            "contains", ExecutionMode.LOCAL, force_cloud=True
        ) == ExecutionMode.CLOUD

        assert select_execution_mode(
            "contains", ExecutionMode.HYBRID, force_cloud=True
        ) == ExecutionMode.CLOUD

    def test_force_local_with_capable_metric(self):
        """Test force_local with a local-capable metric."""
        assert select_execution_mode(
            "contains", ExecutionMode.CLOUD, force_local=True
        ) == ExecutionMode.LOCAL

    def test_force_local_with_incapable_metric_raises(self):
        """Test force_local raises ValueError for non-local metrics."""
        with pytest.raises(ValueError, match="cannot run locally"):
            select_execution_mode(
                "groundedness", ExecutionMode.CLOUD, force_local=True
            )

    def test_local_mode_with_capable_metric(self):
        """Test LOCAL mode with a local-capable metric."""
        assert select_execution_mode(
            "contains", ExecutionMode.LOCAL
        ) == ExecutionMode.LOCAL

    def test_local_mode_falls_back_to_cloud(self):
        """Test LOCAL mode falls back to CLOUD for non-local metrics."""
        assert select_execution_mode(
            "groundedness", ExecutionMode.LOCAL
        ) == ExecutionMode.CLOUD

    def test_hybrid_mode_prefers_local(self):
        """Test HYBRID mode prefers local for capable metrics."""
        assert select_execution_mode(
            "contains", ExecutionMode.HYBRID
        ) == ExecutionMode.LOCAL

    def test_hybrid_mode_uses_cloud_for_llm(self):
        """Test HYBRID mode uses cloud for LLM metrics."""
        assert select_execution_mode(
            "groundedness", ExecutionMode.HYBRID
        ) == ExecutionMode.CLOUD

    def test_cloud_mode_uses_cloud(self):
        """Test CLOUD mode always returns CLOUD."""
        assert select_execution_mode(
            "contains", ExecutionMode.CLOUD
        ) == ExecutionMode.CLOUD

        assert select_execution_mode(
            "groundedness", ExecutionMode.CLOUD
        ) == ExecutionMode.CLOUD
