"""Local evaluator for running metrics without API calls.

This module provides the LocalEvaluator class which can run heuristic
metrics locally, enabling offline evaluation and faster feedback loops.

It also provides the HybridEvaluator which can intelligently route
evaluations between local execution, local LLM, and cloud APIs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import time
import logging

from ..types import BatchRunResult, EvalResult
from .execution_mode import RoutingMode, can_run_locally, select_routing_mode
from .registry import get_registry, LocalMetricRegistry

if TYPE_CHECKING:
    from .llm import OllamaLLM

logger = logging.getLogger(__name__)


@dataclass
class LocalEvaluatorConfig:
    """Configuration for the local evaluator.

    Attributes:
        execution_mode: The default execution mode.
        fail_on_unsupported: If True, raise an error when a metric can't run locally.
        parallel_workers: Number of parallel workers (for future use).
        timeout: Timeout in seconds for individual evaluations.
    """

    execution_mode: RoutingMode = RoutingMode.HYBRID
    fail_on_unsupported: bool = False
    parallel_workers: int = 4
    timeout: int = 60


@dataclass
class LocalEvaluationResult:
    """Result of a local evaluation operation.

    Attributes:
        results: The batch run results.
        executed_locally: Set of metric names that ran locally.
        skipped: Set of metric names that were skipped (not local-capable).
        errors: Dictionary mapping metric names to error messages.
    """

    results: BatchRunResult
    executed_locally: set = field(default_factory=set)
    skipped: set = field(default_factory=set)
    errors: Dict[str, str] = field(default_factory=dict)


class LocalEvaluator:
    """Evaluator that runs metrics locally without API calls.

    This evaluator can run heuristic metrics locally, providing fast feedback
    without requiring network access or API credentials.

    Example:
        >>> evaluator = LocalEvaluator()
        >>> result = evaluator.evaluate(
        ...     metric_name="contains",
        ...     inputs=[{"response": "Hello world", "keyword": "world"}],
        ...     config={"keyword": "world"}
        ... )
        >>> print(result.results.eval_results[0].output)
        1.0
    """

    def __init__(
        self,
        config: Optional[LocalEvaluatorConfig] = None,
        registry: Optional[LocalMetricRegistry] = None,
    ) -> None:
        """Initialize the local evaluator.

        Args:
            config: Configuration for the evaluator.
            registry: Optional metric registry (uses global if not provided).
        """
        self.config = config or LocalEvaluatorConfig()
        self.registry = registry or get_registry()

    def can_run_locally(self, metric_name: str) -> bool:
        """Check if a metric can be run locally.

        Args:
            metric_name: The name of the metric.

        Returns:
            True if the metric can run locally.
        """
        return can_run_locally(metric_name) and self.registry.is_registered(metric_name)

    def evaluate(
        self,
        metric_name: str,
        inputs: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> LocalEvaluationResult:
        """Evaluate a single metric on a batch of inputs.

        Args:
            metric_name: The name of the metric to run.
            inputs: List of input dictionaries for the metric.
            config: Optional configuration for the metric.

        Returns:
            LocalEvaluationResult with results and metadata.

        Raises:
            ValueError: If fail_on_unsupported is True and metric can't run locally.
        """
        result = LocalEvaluationResult(results=BatchRunResult(eval_results=[]))

        if not self.can_run_locally(metric_name):
            if self.config.fail_on_unsupported:
                raise ValueError(
                    f"Metric '{metric_name}' cannot run locally. "
                    f"Available local metrics: {self.registry.list_metrics()}"
                )
            result.skipped.add(metric_name)
            # Return empty results for skipped metrics
            for _ in inputs:
                result.results.eval_results.append(
                    EvalResult(
                        name=metric_name,
                        output=None,
                        reason=f"Metric '{metric_name}' cannot run locally",
                        runtime=0,
                    )
                )
            return result

        try:
            metric = self.registry.create(metric_name, config)
            if metric is None:
                raise ValueError(f"Failed to create metric '{metric_name}'")

            batch_result = metric.evaluate(inputs)
            result.results = batch_result
            result.executed_locally.add(metric_name)

        except Exception as e:
            result.errors[metric_name] = str(e)
            # Fill with error results
            for _ in inputs:
                result.results.eval_results.append(
                    EvalResult(
                        name=metric_name,
                        output=None,
                        reason=f"Error: {str(e)}",
                        runtime=0,
                    )
                )

        # Automatically enrich current span with evaluation results
        try:
            from fi.evals.otel.enrichment import enrich_span_with_batch_result, is_auto_enrichment_enabled
            if is_auto_enrichment_enabled():
                enriched_count = enrich_span_with_batch_result(result.results)
                if enriched_count > 0:
                    import logging
                    logging.debug(f"Enriched active span with {enriched_count} evaluation results")
        except ImportError:
            pass  # OTEL enrichment not available
        except Exception:
            pass  # Silently fail enrichment

        return result

    def evaluate_batch(
        self,
        evaluations: List[Dict[str, Any]],
    ) -> LocalEvaluationResult:
        """Evaluate multiple metrics on their respective inputs.

        Args:
            evaluations: List of evaluation specifications, each containing:
                - metric_name: Name of the metric
                - inputs: List of input dictionaries
                - config: Optional metric configuration

        Returns:
            LocalEvaluationResult with combined results.

        Example:
            >>> evaluator = LocalEvaluator()
            >>> result = evaluator.evaluate_batch([
            ...     {
            ...         "metric_name": "contains",
            ...         "inputs": [{"response": "hello world"}],
            ...         "config": {"keyword": "world"}
            ...     },
            ...     {
            ...         "metric_name": "is_json",
            ...         "inputs": [{"response": '{"key": "value"}'}]
            ...     }
            ... ])
        """
        combined_result = LocalEvaluationResult(
            results=BatchRunResult(eval_results=[])
        )

        for eval_spec in evaluations:
            metric_name = eval_spec.get("metric_name")
            inputs = eval_spec.get("inputs", [])
            config = eval_spec.get("config")

            if not metric_name:
                continue

            single_result = self.evaluate(metric_name, inputs, config)

            # Merge results
            combined_result.results.eval_results.extend(
                single_result.results.eval_results
            )
            combined_result.executed_locally.update(single_result.executed_locally)
            combined_result.skipped.update(single_result.skipped)
            combined_result.errors.update(single_result.errors)

        return combined_result

    def list_available_metrics(self) -> List[str]:
        """List all metrics available for local execution.

        Returns:
            Sorted list of available metric names.
        """
        return self.registry.list_metrics()


class HybridEvaluator:
    """Evaluator that routes metrics between local and cloud execution.

    This evaluator analyzes each metric and automatically routes it to
    either local heuristics, local LLM, or cloud execution based on the
    metric type and configuration.

    Example:
        >>> # Basic usage with automatic routing
        >>> evaluator = HybridEvaluator()
        >>> partitions = evaluator.partition_evaluations([
        ...     {"metric_name": "contains", "inputs": [{"response": "test"}]},
        ...     {"metric_name": "groundedness", "inputs": [{"response": "test"}]},
        ... ])
        >>> local_results = evaluator.evaluate_local_partition(partitions[RoutingMode.LOCAL])

        >>> # With local LLM for LLM-based evaluations
        >>> from fi.evals.local.llm import OllamaLLM
        >>> evaluator = HybridEvaluator(local_llm=OllamaLLM())
        >>> result = evaluator.evaluate(
        ...     template="custom_llm_judge",
        ...     inputs=[{"query": "What is AI?", "response": "AI is..."}]
        ... )
    """

    # LLM-based metrics that can use local LLM instead of cloud
    LLM_BASED_METRICS = {
        "groundedness",
        "hallucination",
        "relevance",
        "coherence",
        "context_relevance",
        "answer_relevance",
        "custom_llm_judge",
        "tone",
        "safety",
        "pii",
        "bias",
    }

    def __init__(
        self,
        config: Optional[LocalEvaluatorConfig] = None,
        local_evaluator: Optional[LocalEvaluator] = None,
        local_llm: Optional["OllamaLLM"] = None,
        cloud_evaluator: Optional[Any] = None,
        prefer_local: bool = True,
        fallback_to_cloud: bool = True,
        offline_mode: bool = False,
    ) -> None:
        """Initialize the hybrid evaluator.

        Args:
            config: Configuration for the evaluator.
            local_evaluator: Optional local evaluator instance for heuristic metrics.
            local_llm: Optional local LLM instance for LLM-based metrics.
            cloud_evaluator: Optional cloud evaluator instance (Evaluator).
            prefer_local: If True, prefer local execution when possible.
            fallback_to_cloud: If True, fall back to cloud when local fails.
            offline_mode: If True, never use cloud (raise error if metric requires it).
        """
        self.config = config or LocalEvaluatorConfig(execution_mode=RoutingMode.HYBRID)
        self.local_evaluator = local_evaluator or LocalEvaluator(self.config)
        self.local_llm = local_llm
        self.cloud_evaluator = cloud_evaluator
        self.prefer_local = prefer_local
        self.fallback_to_cloud = fallback_to_cloud
        self.offline_mode = offline_mode

    def set_local_llm(self, llm: "OllamaLLM") -> None:
        """Set the local LLM instance.

        Args:
            llm: OllamaLLM instance to use for LLM-based evaluations.
        """
        self.local_llm = llm

    def set_cloud_evaluator(self, evaluator: Any) -> None:
        """Set the cloud evaluator instance.

        Args:
            evaluator: Cloud Evaluator instance for API-based evaluations.
        """
        self.cloud_evaluator = evaluator

    def can_use_local_llm(self, metric_name: str) -> bool:
        """Check if a metric can use the local LLM.

        Args:
            metric_name: The name of the metric.

        Returns:
            True if the metric can use local LLM.
        """
        if self.local_llm is None:
            return False
        if not self.local_llm.is_available():
            return False
        return metric_name.lower() in self.LLM_BASED_METRICS

    def route_evaluation(
        self,
        metric_name: str,
        force_local: bool = False,
        force_cloud: bool = False,
    ) -> RoutingMode:
        """Determine the execution mode for a metric.

        Args:
            metric_name: The name of the metric.
            force_local: Force local execution.
            force_cloud: Force cloud execution.

        Returns:
            The recommended execution mode.
        """
        if force_cloud and not self.offline_mode:
            return RoutingMode.CLOUD
        if force_local:
            return RoutingMode.LOCAL

        # Check if it's a heuristic metric that can run locally
        if can_run_locally(metric_name):
            return RoutingMode.LOCAL

        # Check if it's an LLM metric and we have local LLM
        if self.can_use_local_llm(metric_name) and self.prefer_local:
            return RoutingMode.LOCAL

        # Default to cloud unless in offline mode
        if self.offline_mode:
            raise ValueError(
                f"Metric '{metric_name}' requires cloud execution but offline_mode is enabled"
            )
        return RoutingMode.CLOUD

    def partition_evaluations(
        self, evaluations: List[Dict[str, Any]]
    ) -> Dict[RoutingMode, List[Dict[str, Any]]]:
        """Partition evaluations by execution mode.

        Args:
            evaluations: List of evaluation specifications.

        Returns:
            Dictionary mapping execution modes to their evaluations.
        """
        partitions: Dict[RoutingMode, List[Dict[str, Any]]] = {
            RoutingMode.LOCAL: [],
            RoutingMode.CLOUD: [],
        }

        for eval_spec in evaluations:
            metric_name = eval_spec.get("metric_name", "")
            force_local = eval_spec.get("force_local", False)
            force_cloud = eval_spec.get("force_cloud", False)

            try:
                mode = self.route_evaluation(metric_name, force_local, force_cloud)
                partitions[mode].append(eval_spec)
            except ValueError as e:
                # Offline mode violation - add to errors
                logger.error(f"Routing error for {metric_name}: {e}")
                partitions[RoutingMode.CLOUD].append(eval_spec)

        return partitions

    def evaluate_local_partition(
        self, evaluations: List[Dict[str, Any]]
    ) -> LocalEvaluationResult:
        """Evaluate the local partition of evaluations.

        This handles both heuristic metrics (via LocalEvaluator) and
        LLM-based metrics (via local LLM).

        Args:
            evaluations: List of evaluation specifications to run locally.

        Returns:
            LocalEvaluationResult with results.
        """
        heuristic_evals = []
        llm_evals = []

        # Separate heuristic and LLM evaluations
        for eval_spec in evaluations:
            metric_name = eval_spec.get("metric_name", "")
            if can_run_locally(metric_name):
                heuristic_evals.append(eval_spec)
            elif self.can_use_local_llm(metric_name):
                llm_evals.append(eval_spec)
            else:
                heuristic_evals.append(eval_spec)  # Will be skipped

        # Run heuristic evaluations
        result = self.local_evaluator.evaluate_batch(heuristic_evals)

        # Run LLM evaluations if we have a local LLM
        if llm_evals and self.local_llm:
            llm_results = self._evaluate_with_local_llm(llm_evals)
            result.results.eval_results.extend(llm_results.results.eval_results)
            result.executed_locally.update(llm_results.executed_locally)
            result.skipped.update(llm_results.skipped)
            result.errors.update(llm_results.errors)

        return result

    def _evaluate_with_local_llm(
        self, evaluations: List[Dict[str, Any]]
    ) -> LocalEvaluationResult:
        """Run evaluations using the local LLM.

        Args:
            evaluations: List of LLM-based evaluation specifications.

        Returns:
            LocalEvaluationResult with LLM evaluation results.
        """
        result = LocalEvaluationResult(results=BatchRunResult(eval_results=[]))

        if not self.local_llm:
            for eval_spec in evaluations:
                metric_name = eval_spec.get("metric_name", "")
                result.skipped.add(metric_name)
                for _ in eval_spec.get("inputs", []):
                    result.results.eval_results.append(
                        EvalResult(
                            name=metric_name,
                            output=None,
                            reason="No local LLM configured",
                            runtime=0,
                        )
                    )
            return result

        for eval_spec in evaluations:
            metric_name = eval_spec.get("metric_name", "")
            inputs = eval_spec.get("inputs", [])
            config = eval_spec.get("config", {})

            for input_data in inputs:
                start_time = time.time()
                try:
                    # Build evaluation from input
                    judge_result = self.local_llm.judge(
                        query=input_data.get("input", input_data.get("query", "")),
                        response=input_data.get("response", input_data.get("output", "")),
                        criteria=config.get("criteria", f"Evaluate based on {metric_name}"),
                        context=input_data.get("context", input_data.get("contexts", "")),
                    )

                    runtime = int((time.time() - start_time) * 1000)
                    result.results.eval_results.append(
                        EvalResult(
                            name=metric_name,
                            output=judge_result.get("score", 0.0),
                            reason=judge_result.get("reason", ""),
                            runtime=runtime,
                            metrics=[{
                                "name": metric_name,
                                "value": judge_result.get("score", 0.0),
                            }],
                        )
                    )
                    result.executed_locally.add(metric_name)

                except Exception as e:
                    runtime = int((time.time() - start_time) * 1000)
                    result.errors[metric_name] = str(e)
                    result.results.eval_results.append(
                        EvalResult(
                            name=metric_name,
                            output=0.0,
                            reason=f"Local LLM error: {str(e)}",
                            runtime=runtime,
                        )
                    )

        return result

    def evaluate(
        self,
        template: str,
        inputs: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> LocalEvaluationResult:
        """Evaluate a single template with automatic routing.

        Args:
            template: The evaluation template/metric name.
            inputs: List of input dictionaries.
            config: Optional configuration for the evaluation.

        Returns:
            LocalEvaluationResult with evaluation results.
        """
        eval_spec = {
            "metric_name": template,
            "inputs": inputs,
            "config": config or {},
        }

        mode = self.route_evaluation(template)

        if mode == RoutingMode.LOCAL:
            result = self.evaluate_local_partition([eval_spec])
        else:
            result = self._evaluate_cloud([eval_spec])

        # Automatically enrich current span with evaluation results
        try:
            from fi.evals.otel.enrichment import enrich_span_with_batch_result, is_auto_enrichment_enabled
            if is_auto_enrichment_enabled():
                enrich_span_with_batch_result(result.results)
        except ImportError:
            pass
        except Exception:
            pass

        return result

    def evaluate_batch(
        self,
        evaluations: List[Dict[str, Any]],
    ) -> LocalEvaluationResult:
        """Evaluate multiple templates with automatic routing.

        Args:
            evaluations: List of evaluation specifications.

        Returns:
            Combined LocalEvaluationResult from all evaluations.
        """
        partitions = self.partition_evaluations(evaluations)

        # Run local evaluations
        local_result = self.evaluate_local_partition(partitions[RoutingMode.LOCAL])

        # Run cloud evaluations
        if partitions[RoutingMode.CLOUD]:
            cloud_result = self._evaluate_cloud(partitions[RoutingMode.CLOUD])

            # Merge results
            local_result.results.eval_results.extend(cloud_result.results.eval_results)
            local_result.executed_locally.update(cloud_result.executed_locally)
            local_result.skipped.update(cloud_result.skipped)
            local_result.errors.update(cloud_result.errors)

        return local_result

    def _evaluate_cloud(
        self, evaluations: List[Dict[str, Any]]
    ) -> LocalEvaluationResult:
        """Run evaluations via cloud API.

        Args:
            evaluations: List of evaluation specifications for cloud.

        Returns:
            LocalEvaluationResult with cloud evaluation results.
        """
        result = LocalEvaluationResult(results=BatchRunResult(eval_results=[]))

        if self.offline_mode:
            for eval_spec in evaluations:
                metric_name = eval_spec.get("metric_name", "")
                result.skipped.add(metric_name)
                for _ in eval_spec.get("inputs", []):
                    result.results.eval_results.append(
                        EvalResult(
                            name=metric_name,
                            output=None,
                            reason="Offline mode - cloud execution disabled",
                            runtime=0,
                        )
                    )
            return result

        if not self.cloud_evaluator:
            for eval_spec in evaluations:
                metric_name = eval_spec.get("metric_name", "")
                result.skipped.add(metric_name)
                for _ in eval_spec.get("inputs", []):
                    result.results.eval_results.append(
                        EvalResult(
                            name=metric_name,
                            output=None,
                            reason="No cloud evaluator configured",
                            runtime=0,
                        )
                    )
            return result

        # Route through fi.evals.evaluate() with Turing engine
        try:
            from fi.evals import evaluate as core_evaluate

            for eval_spec in evaluations:
                metric_name = eval_spec.get("metric_name", "")
                inputs_list = eval_spec.get("inputs", [])

                for input_data in inputs_list:
                    start_time = time.time()
                    try:
                        eval_result = core_evaluate(
                            metric_name,
                            engine="turing",
                            output=input_data.get("response", input_data.get("output", "")),
                            input=input_data.get("input", input_data.get("query", "")),
                            context=input_data.get("context", input_data.get("contexts", "")),
                        )

                        runtime = int((time.time() - start_time) * 1000)
                        result.results.eval_results.append(
                            EvalResult(
                                name=metric_name,
                                output=eval_result.score if hasattr(eval_result, "score") else eval_result.output,
                                reason=getattr(eval_result, "reason", ""),
                                runtime=runtime,
                                metrics=[{
                                    "name": metric_name,
                                    "value": eval_result.score if hasattr(eval_result, "score") else 0.0,
                                }],
                            )
                        )
                        result.executed_locally.add(metric_name)

                    except Exception as e:
                        runtime = int((time.time() - start_time) * 1000)
                        result.errors[metric_name] = str(e)
                        result.results.eval_results.append(
                            EvalResult(
                                name=metric_name,
                                output=None,
                                reason=f"Cloud evaluation error: {e}",
                                runtime=runtime,
                            )
                        )

        except ImportError:
            logger.error("fi.evals.evaluate not available for cloud routing")
            for eval_spec in evaluations:
                metric_name = eval_spec.get("metric_name", "")
                result.skipped.add(metric_name)
                result.errors[metric_name] = "fi.evals.evaluate not available"
        except Exception as e:
            logger.error(f"Cloud evaluation failed: {e}")
            for eval_spec in evaluations:
                result.errors[eval_spec.get("metric_name", "")] = str(e)

        return result
