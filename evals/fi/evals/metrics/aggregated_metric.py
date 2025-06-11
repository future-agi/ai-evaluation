import string
from typing import List, Union, Any, Dict, Optional, Tuple
import json

from fi.evals.templates import EvalTemplate
from fi.evals.types import EvalResult, EvalResultMetric, BatchRunResult
from fi.testcases import TestCase


class AggregatedMetric(EvalTemplate):
    """
    AggregatedMetric combines multiple metric evaluators into a single aggregated score.

    Possible values for aggregator:
        - 'average': Computes the mean of all metric scores.
        - 'weighted_average': Computes the weighted mean using provided weights.

    Example config:
        config = {
            "aggregator": "average",  # or "weighted_average"
            "weights": [0.5, 0.5],     # Only for weighted_average
            "metrics": [metric1, metric2]
        }

    Example usage:
        agg_metric = AggregatedMetric(config)
        agg_metric.validate_input(test_cases)
        result = agg_metric.evaluate(test_cases)
    """
    EVAL_ID = "82"
    NAME = "Aggregated Metric"
    DESCRIPTION = "Combines multiple metrics into a single aggregated score"
    EVAL_TAGS = ["metric", "aggregation", "composite", "multi-metric"]
    REQUIRED_KEYS = ["response", "expected_text"]
    OUTPUT = "score"
    EVAL_TYPE_ID = "metric"
    DEFAULT_CONFIG_SCHEMA = {
        "aggregator": "average",
        "weights": None,
        "metrics": []
    }
    CRITERIA = "Aggregates multiple evaluation metrics into a combined score"
    CHOICES = []
    MULTI_CHOICE = False
    SUPPORTED_AGGREGATORS = ["average", "weighted_average"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.aggregator = config.get("aggregator", self.DEFAULT_CONFIG_SCHEMA["aggregator"])
        self.weights = config.get("weights", self.DEFAULT_CONFIG_SCHEMA["weights"])
        self.metrics = config.get("metrics", self.DEFAULT_CONFIG_SCHEMA["metrics"])

        # Set required instance attributes for EvalTemplate
        self.name = self.NAME
        self.description = self.DESCRIPTION
        self.eval_id = self.EVAL_ID
        self.eval_tags = self.EVAL_TAGS
        self.required_keys = self.REQUIRED_KEYS
        self.output = self.OUTPUT
        self.eval_type_id = self.EVAL_TYPE_ID
        self.config_schema = self.DEFAULT_CONFIG_SCHEMA
        self.criteria = self.CRITERIA
        self.choices = self.CHOICES
        self.multi_choice = self.MULTI_CHOICE

        if not self.metrics:
            raise ValueError("AggregatedMetric requires at least one metric evaluator")

        if self.aggregator not in self.SUPPORTED_AGGREGATORS:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")
        if self.aggregator == "weighted_average":
            if self.weights is None or len(self.weights) != len(self.metrics):
                raise ValueError("Weights required for weighted_average and must match the number of metrics")

        self.metric_names = [getattr(m, 'NAME', getattr(m, 'name', str(m))) for m in self.metrics]

        super().__init__(config)

    def validate_input(self, inputs: List[TestCase]):
        for test_case in inputs:
            if not hasattr(test_case, "response") or test_case.response is None:
                raise ValueError("TestCase must have a 'response' field for AggregatedMetric evaluation")
            if not hasattr(test_case, "expected_text") or test_case.expected_text is None:
                raise ValueError("TestCase must have an 'expected_text' field for AggregatedMetric evaluation")
        for metric in self.metrics:
            if hasattr(metric, 'validate_input'):
                metric.validate_input(inputs)
        return True

    def evaluate(self, inputs: List[TestCase]) -> BatchRunResult:
        try:
            self.validate_input(inputs)
        except Exception as e:
            eval_result = EvalResult(
                data=[0.0, {}],
                failure=True,
                reason=f"Input validation failed: {str(e)}",
                runtime=0,
                metadata={
                    "aggregator": self.aggregator,
                    "metrics": self.metric_names,
                    "weights": self.weights,
                    "individual_scores": {}
                },
                metrics=[
                    EvalResultMetric(
                        id="aggregated_metric_score",
                        value=0.0
                    )
                ]
            )
            return BatchRunResult(eval_results=[eval_result])

        eval_results = []
        for test_case in inputs:
            try:
                metric_results = []
                for metric in self.metrics:
                    try:
                        if hasattr(metric, 'evaluate'):
                            result = metric.evaluate([test_case])
                            if result.eval_results and result.eval_results[0].metrics:
                                metric_value = result.eval_results[0].metrics[0].value
                                metric_results.append(self._normalize_score(metric_value))
                            else:
                                metric_results.append(0.0)
                        else:
                            metric_results.append(0.0)
                    except Exception as metric_exc:
                        metric_results.append(0.0)
                if self.aggregator == "average":
                    aggregated_score = sum(metric_results) / len(metric_results) if metric_results else 0.0
                elif self.aggregator == "weighted_average":
                    weighted_sum = sum(w * s for w, s in zip(self.weights, metric_results))
                    total_weight = sum(self.weights)
                    aggregated_score = weighted_sum / total_weight if total_weight > 0 else 0.0
                else:
                    aggregated_score = 0.0
                metric_details = {
                    name: score for name, score in zip(self.metric_names, metric_results)
                }
                failure = False
                reason = ""
            except Exception as e:
                aggregated_score = 0.0
                metric_details = {}
                failure = True
                reason = f"Evaluation failed: {str(e)}"
            eval_result = EvalResult(
                data=[aggregated_score, metric_details],
                failure=failure,
                reason=reason,
                runtime=0,
                metadata={
                    "aggregator": self.aggregator,
                    "metrics": self.metric_names,
                    "weights": self.weights,
                    "individual_scores": metric_details
                },
                metrics=[
                    EvalResultMetric(
                        id="aggregated_metric_score",
                        value=aggregated_score
                    )
                ]
            )
            eval_results.append(eval_result)
        return BatchRunResult(eval_results=eval_results)

    def _normalize_score(self, value: Any) -> float:
        """Convert various score types to a float between 0 and 1"""
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            float_value = float(value)
            return max(0.0, min(1.0, float_value))
        except (ValueError, TypeError):
            return 0.0