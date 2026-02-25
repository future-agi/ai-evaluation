"""Execution mode selector for evaluations.

This module defines execution modes that determine how evaluations run:
- LOCAL: Run all evaluations locally using heuristic metrics (no API calls)
- CLOUD: Run all evaluations via the cloud API
- HYBRID: Automatically route each evaluation to local or cloud based on metric type
"""

from enum import Enum
from typing import Optional, Set


class RoutingMode(Enum):
    """Defines how evaluations should be executed."""

    LOCAL = "local"
    """Run evaluations locally using heuristic metrics only."""

    CLOUD = "cloud"
    """Run all evaluations via the cloud API."""

    HYBRID = "hybrid"
    """Automatically choose local or cloud based on metric capabilities."""

    def __str__(self) -> str:
        return self.value


# Set of metric names that can be run locally (heuristic metrics)
LOCAL_CAPABLE_METRICS: Set[str] = {
    # String metrics
    "regex",
    "contains",
    "contains_all",
    "contains_any",
    "contains_none",
    "one_line",
    "contains_email",
    "is_email",
    "contains_link",
    "contains_valid_link",
    "equals",
    "starts_with",
    "ends_with",
    "length_less_than",
    "length_greater_than",
    "length_between",

    # JSON metrics
    "contains_json",
    "is_json",
    "json_schema",

    # Similarity metrics
    "bleu_score",
    "rouge_score",
    "recall_score",
    "levenshtein_similarity",
    "numeric_similarity",
    "embedding_similarity",
    "semantic_list_contains",
}


def can_run_locally(metric_name: str) -> bool:
    """Check if a metric can be run locally.

    Args:
        metric_name: The name of the metric to check.

    Returns:
        True if the metric can run locally, False otherwise.
    """
    return metric_name.lower() in LOCAL_CAPABLE_METRICS


def select_routing_mode(
    metric_name: str,
    preferred_mode: RoutingMode,
    force_local: bool = False,
    force_cloud: bool = False,
) -> RoutingMode:
    """Select the execution mode for a metric based on preferences and capabilities.

    Args:
        metric_name: The name of the metric.
        preferred_mode: The user's preferred execution mode.
        force_local: If True, always try local execution (error if not possible).
        force_cloud: If True, always use cloud execution.

    Returns:
        The selected execution mode.

    Raises:
        ValueError: If force_local is True but the metric cannot run locally.
    """
    if force_cloud:
        return RoutingMode.CLOUD

    if force_local:
        if not can_run_locally(metric_name):
            raise ValueError(
                f"Metric '{metric_name}' cannot run locally. "
                f"Local-capable metrics: {sorted(LOCAL_CAPABLE_METRICS)}"
            )
        return RoutingMode.LOCAL

    if preferred_mode == RoutingMode.LOCAL:
        if can_run_locally(metric_name):
            return RoutingMode.LOCAL
        # Fall back to cloud if metric can't run locally
        return RoutingMode.CLOUD

    if preferred_mode == RoutingMode.HYBRID:
        # In hybrid mode, prefer local for capable metrics
        if can_run_locally(metric_name):
            return RoutingMode.LOCAL
        return RoutingMode.CLOUD

    # Default: CLOUD mode
    return RoutingMode.CLOUD
