"""Tests for the local metric registry."""

import pytest

from fi.evals.local.registry import (
    LocalMetricRegistry,
    get_registry,
)
from fi.evals.metrics.base_metric import BaseMetric


class TestLocalMetricRegistry:
    """Tests for the LocalMetricRegistry class."""

    def test_register_and_get_metric(self):
        """Test registering and retrieving a metric."""
        from fi.evals.metrics.heuristics.string_metrics import Contains

        registry = LocalMetricRegistry()
        registry.register("test_metric", Contains)

        assert registry.get("test_metric") is Contains

    def test_get_unregistered_metric_returns_none(self):
        """Test that getting an unregistered metric returns None."""
        registry = LocalMetricRegistry()
        assert registry.get("nonexistent") is None

    def test_is_registered(self):
        """Test the is_registered method."""
        from fi.evals.metrics.heuristics.string_metrics import Contains

        registry = LocalMetricRegistry()
        registry.register("test_metric", Contains)

        assert registry.is_registered("test_metric") is True
        assert registry.is_registered("nonexistent") is False

    def test_case_insensitive_lookup(self):
        """Test that metric lookup is case insensitive."""
        from fi.evals.metrics.heuristics.string_metrics import Contains

        registry = LocalMetricRegistry()
        registry.register("TestMetric", Contains)

        assert registry.get("testmetric") is Contains
        assert registry.get("TESTMETRIC") is Contains
        assert registry.get("TestMetric") is Contains

    def test_create_metric_instance(self):
        """Test creating a metric instance."""
        from fi.evals.metrics.heuristics.string_metrics import Contains

        registry = LocalMetricRegistry()
        registry.register("contains", Contains)

        metric = registry.create("contains", {"keyword": "test"})

        assert metric is not None
        assert isinstance(metric, Contains)
        assert metric.keyword == "test"

    def test_create_unregistered_returns_none(self):
        """Test that create returns None for unregistered metrics."""
        registry = LocalMetricRegistry()
        assert registry.create("nonexistent") is None

    def test_list_metrics(self):
        """Test listing all registered metrics."""
        from fi.evals.metrics.heuristics.string_metrics import Contains, Regex

        registry = LocalMetricRegistry()
        registry.register("contains", Contains)
        registry.register("regex", Regex)

        metrics = registry.list_metrics()

        assert "contains" in metrics
        assert "regex" in metrics
        assert metrics == sorted(metrics)  # Should be sorted


class TestLazyLoading:
    """Tests for lazy loading of metrics."""

    def test_register_lazy(self):
        """Test lazy registration of metrics."""
        registry = LocalMetricRegistry()
        load_count = [0]

        def loader():
            load_count[0] += 1
            from fi.evals.metrics.heuristics.string_metrics import Contains
            return Contains

        registry.register_lazy("lazy_metric", loader)

        # Should not load yet
        assert load_count[0] == 0
        assert registry.is_registered("lazy_metric") is True

        # Should load on first access
        metric_class = registry.get("lazy_metric")
        assert load_count[0] == 1
        assert metric_class is not None

        # Should not reload on second access
        metric_class2 = registry.get("lazy_metric")
        assert load_count[0] == 1
        assert metric_class2 is metric_class

    def test_list_includes_lazy_metrics(self):
        """Test that list_metrics includes lazy-loaded metrics."""
        registry = LocalMetricRegistry()

        def loader():
            from fi.evals.metrics.heuristics.string_metrics import Contains
            return Contains

        registry.register_lazy("lazy_metric", loader)

        metrics = registry.list_metrics()
        assert "lazy_metric" in metrics


class TestGlobalRegistry:
    """Tests for the global registry."""

    def test_get_registry_returns_same_instance(self):
        """Test that get_registry returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_builtin_metrics_are_registered(self):
        """Test that builtin metrics are registered in global registry."""
        registry = get_registry()

        # Check string metrics
        assert registry.is_registered("contains")
        assert registry.is_registered("regex")
        assert registry.is_registered("equals")

        # Check JSON metrics
        assert registry.is_registered("is_json")
        assert registry.is_registered("json_schema")

        # Check similarity metrics (lazy loaded)
        assert registry.is_registered("bleu_score")
        assert registry.is_registered("rouge_score")

    def test_can_create_builtin_metrics(self):
        """Test that builtin metrics can be created."""
        registry = get_registry()

        # Create a string metric
        contains = registry.create("contains", {"keyword": "test"})
        assert contains is not None

        # Create a JSON metric
        is_json = registry.create("is_json")
        assert is_json is not None
