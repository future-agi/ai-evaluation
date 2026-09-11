"""Registry for local metric classes.

This module provides a registry that maps metric names to their implementation classes,
enabling dynamic instantiation of metrics based on configuration.
"""

from typing import Any, Dict, Optional, Type, Callable

from ..metrics.base_metric import BaseMetric


class LocalMetricRegistry:
    """Registry for local metric implementations.

    The registry maps metric names to their implementation classes and allows
    for dynamic instantiation with configuration.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        self._lazy_loaders: Dict[str, Callable[[], Type[BaseMetric]]] = {}

    def register(self, name: str, metric_class: Type[BaseMetric]) -> None:
        """Register a metric class with a given name.

        Args:
            name: The name to register the metric under.
            metric_class: The metric class to register.
        """
        self._metrics[name.lower()] = metric_class

    def register_lazy(
        self, name: str, loader: Callable[[], Type[BaseMetric]]
    ) -> None:
        """Register a metric class with lazy loading.

        Use this for metrics with heavy dependencies (e.g., ML models).

        Args:
            name: The name to register the metric under.
            loader: A callable that returns the metric class when invoked.
        """
        self._lazy_loaders[name.lower()] = loader

    def get(self, name: str) -> Optional[Type[BaseMetric]]:
        """Get a metric class by name.

        Args:
            name: The name of the metric.

        Returns:
            The metric class, or None if not found.
        """
        name_lower = name.lower()

        # Check direct registry first
        if name_lower in self._metrics:
            return self._metrics[name_lower]

        # Check lazy loaders
        if name_lower in self._lazy_loaders:
            # Load and cache the metric class
            metric_class = self._lazy_loaders[name_lower]()
            self._metrics[name_lower] = metric_class
            del self._lazy_loaders[name_lower]
            return metric_class

        return None

    def create(
        self, name: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseMetric]:
        """Create an instance of a metric by name.

        Args:
            name: The name of the metric.
            config: Optional configuration for the metric.

        Returns:
            A metric instance, or None if the metric is not registered.
        """
        metric_class = self.get(name)
        if metric_class is None:
            return None
        return metric_class(config)

    def is_registered(self, name: str) -> bool:
        """Check if a metric is registered.

        Args:
            name: The name of the metric.

        Returns:
            True if the metric is registered, False otherwise.
        """
        name_lower = name.lower()
        return name_lower in self._metrics or name_lower in self._lazy_loaders

    def list_metrics(self) -> list:
        """List all registered metric names.

        Returns:
            A sorted list of registered metric names.
        """
        all_names = set(self._metrics.keys()) | set(self._lazy_loaders.keys())
        return sorted(all_names)


# Global registry instance
_registry: Optional[LocalMetricRegistry] = None


def get_registry() -> LocalMetricRegistry:
    """Get the global metric registry, initializing if needed.

    Returns:
        The global LocalMetricRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = LocalMetricRegistry()
        _register_builtin_metrics(_registry)
    return _registry


def _register_builtin_metrics(registry: LocalMetricRegistry) -> None:
    """Register all built-in local metrics.

    Args:
        registry: The registry to register metrics in.
    """
    # Import and register string metrics
    from ..metrics.heuristics.string_metrics import (
        Regex,
        Contains,
        ContainsAll,
        ContainsAny,
        ContainsNone,
        OneLine,
        ContainsEmail,
        IsEmail,
        ContainsLink,
        ContainsValidLink,
        Equals,
        StartsWith,
        EndsWith,
        LengthLessThan,
        LengthGreaterThan,
        LengthBetween,
    )

    registry.register("regex", Regex)
    registry.register("contains", Contains)
    registry.register("contains_all", ContainsAll)
    registry.register("contains_any", ContainsAny)
    registry.register("contains_none", ContainsNone)
    registry.register("one_line", OneLine)
    registry.register("contains_email", ContainsEmail)
    registry.register("is_email", IsEmail)
    registry.register("contains_link", ContainsLink)
    registry.register("contains_valid_link", ContainsValidLink)
    registry.register("equals", Equals)
    registry.register("starts_with", StartsWith)
    registry.register("ends_with", EndsWith)
    registry.register("length_less_than", LengthLessThan)
    registry.register("length_greater_than", LengthGreaterThan)
    registry.register("length_between", LengthBetween)

    # Import and register JSON metrics
    from ..metrics.heuristics.json_metrics import (
        ContainsJson,
        IsJson,
        JsonSchema,
    )

    registry.register("contains_json", ContainsJson)
    registry.register("is_json", IsJson)
    registry.register("json_schema", JsonSchema)

    # Register similarity metrics with lazy loading (they have heavy dependencies)
    def load_bleu_score():
        from ..metrics.heuristics.similarity_metrics import BLEUScore
        return BLEUScore

    def load_rouge_score():
        from ..metrics.heuristics.similarity_metrics import ROUGEScore
        return ROUGEScore

    def load_recall_score():
        from ..metrics.heuristics.similarity_metrics import RecallScore
        return RecallScore

    def load_levenshtein_similarity():
        from ..metrics.heuristics.similarity_metrics import LevenshteinSimilarity
        return LevenshteinSimilarity

    def load_numeric_similarity():
        from ..metrics.heuristics.similarity_metrics import NumericSimilarity
        return NumericSimilarity

    def load_embedding_similarity():
        from ..metrics.heuristics.similarity_metrics import EmbeddingSimilarity
        return EmbeddingSimilarity

    def load_semantic_list_contains():
        from ..metrics.heuristics.similarity_metrics import SemanticListContains
        return SemanticListContains

    # Import and register function calling metrics
    from ..metrics.function_calling.metrics import (
        FunctionNameMatch,
        ParameterValidation,
        FunctionCallAccuracy,
        FunctionCallExactMatch,
    )

    # Import and register agent evaluation metrics
    from ..metrics.agents.metrics import (
        TaskCompletion,
        StepEfficiency,
        ToolSelectionAccuracy,
        TrajectoryScore,
        GoalProgress,
        ActionSafety,
        ReasoningQuality,
    )

    registry.register("task_completion", TaskCompletion)
    registry.register("step_efficiency", StepEfficiency)
    registry.register("tool_selection_accuracy", ToolSelectionAccuracy)
    registry.register("trajectory_score", TrajectoryScore)
    registry.register("goal_progress", GoalProgress)
    registry.register("action_safety", ActionSafety)
    registry.register("reasoning_quality", ReasoningQuality)

    registry.register("function_name_match", FunctionNameMatch)
    registry.register("parameter_validation", ParameterValidation)
    registry.register("function_call_accuracy", FunctionCallAccuracy)
    registry.register("function_call_exact_match", FunctionCallExactMatch)

    # Register hallucination detection metrics (lazy — may load NLI models)
    def load_faithfulness():
        from ..metrics.hallucination.metrics import Faithfulness
        return Faithfulness

    def load_claim_support():
        from ..metrics.hallucination.metrics import ClaimSupport
        return ClaimSupport

    def load_factual_consistency():
        from ..metrics.hallucination.metrics import FactualConsistency
        return FactualConsistency

    def load_contradiction_detection():
        from ..metrics.hallucination.metrics import ContradictionDetection
        return ContradictionDetection

    def load_hallucination_score():
        from ..metrics.hallucination.metrics import HallucinationScore
        return HallucinationScore

    registry.register_lazy("faithfulness", load_faithfulness)
    registry.register_lazy("claim_support", load_claim_support)
    registry.register_lazy("factual_consistency", load_factual_consistency)
    registry.register_lazy("contradiction_detection", load_contradiction_detection)
    registry.register_lazy("hallucination_score", load_hallucination_score)

    registry.register_lazy("bleu_score", load_bleu_score)
    registry.register_lazy("rouge_score", load_rouge_score)
    registry.register_lazy("recall_score", load_recall_score)
    registry.register_lazy("levenshtein_similarity", load_levenshtein_similarity)
    registry.register_lazy("numeric_similarity", load_numeric_similarity)
    registry.register_lazy("embedding_similarity", load_embedding_similarity)
    registry.register_lazy("semantic_list_contains", load_semantic_list_contains)

    # Register RAG metrics (lazy — may load NLI models)
    def _rag(cls_name):
        """Helper to create lazy loaders for RAG metric classes."""
        def loader():
            import importlib
            # Retrieval metrics
            for mod_path in [
                "..metrics.rag.retrieval.context_recall",
                "..metrics.rag.retrieval.context_precision",
                "..metrics.rag.retrieval.context_entity_recall",
                "..metrics.rag.retrieval.noise_sensitivity",
                "..metrics.rag.retrieval.ranking",
                "..metrics.rag.generation.answer_relevancy",
                "..metrics.rag.generation.context_utilization",
                "..metrics.rag.generation.faithfulness",
                "..metrics.rag.generation.groundedness",
                "..metrics.rag.advanced.multi_hop",
                "..metrics.rag.advanced.source_attribution",
                "..metrics.rag.rag_score",
            ]:
                try:
                    mod = importlib.import_module(mod_path, package=__package__)
                    if hasattr(mod, cls_name):
                        return getattr(mod, cls_name)
                except ImportError:
                    continue
            raise ImportError(f"RAG metric class '{cls_name}' not found")
        return loader

    # RAG — retrieval
    registry.register_lazy("context_recall", _rag("ContextRecall"))
    registry.register_lazy("context_precision", _rag("ContextPrecision"))
    registry.register_lazy("context_entity_recall", _rag("ContextEntityRecall"))
    registry.register_lazy("noise_sensitivity", _rag("NoiseSensitivity"))
    registry.register_lazy("ndcg", _rag("NDCG"))
    registry.register_lazy("mrr", _rag("MRR"))
    registry.register_lazy("precision_at_k", _rag("PrecisionAtK"))
    registry.register_lazy("recall_at_k", _rag("RecallAtK"))
    # RAG — generation
    registry.register_lazy("answer_relevancy", _rag("AnswerRelevancy"))
    registry.register_lazy("context_utilization", _rag("ContextUtilization"))
    registry.register_lazy("context_relevance_to_response", _rag("ContextRelevanceToResponse"))
    registry.register_lazy("rag_faithfulness", _rag("RAGFaithfulness"))
    registry.register_lazy("rag_faithfulness_with_reference", _rag("RAGFaithfulnessWithReference"))
    registry.register_lazy("groundedness", _rag("Groundedness"))
    # RAG — advanced
    registry.register_lazy("multi_hop_reasoning", _rag("MultiHopReasoning"))
    registry.register_lazy("source_attribution", _rag("SourceAttribution"))
    registry.register_lazy("citation_presence", _rag("CitationPresence"))
    # RAG — composite
    registry.register_lazy("rag_score", _rag("RAGScore"))
    registry.register_lazy("rag_score_detailed", _rag("RAGScoreDetailed"))

    # Register structured output metrics (lazy)
    def _structured(cls_name):
        """Helper to create lazy loaders for Structured metric classes."""
        def loader():
            import importlib
            for mod_path in [
                "..metrics.structured.schema_compliance",
                "..metrics.structured.field_completeness",
                "..metrics.structured.hierarchy_score",
                "..metrics.structured.json_validation",
                "..metrics.structured.structured_output_score",
            ]:
                try:
                    mod = importlib.import_module(mod_path, package=__package__)
                    if hasattr(mod, cls_name):
                        return getattr(mod, cls_name)
                except ImportError:
                    continue
            raise ImportError(f"Structured metric class '{cls_name}' not found")
        return loader

    registry.register_lazy("schema_compliance", _structured("SchemaCompliance"))
    registry.register_lazy("type_compliance", _structured("TypeCompliance"))
    registry.register_lazy("field_completeness", _structured("FieldCompleteness"))
    registry.register_lazy("required_fields", _structured("RequiredFieldsOnly"))
    registry.register_lazy("field_coverage", _structured("FieldCoverage"))
    registry.register_lazy("hierarchy_score", _structured("HierarchyScore"))
    registry.register_lazy("tree_edit_distance", _structured("TreeEditDistance"))
    registry.register_lazy("json_validation", _structured("JSONValidation"))
    registry.register_lazy("json_syntax", _structured("JSONSyntaxOnly"))
    registry.register_lazy("structured_output_score", _structured("StructuredOutputScore"))
    registry.register_lazy("quick_structured_check", _structured("QuickStructuredCheck"))
