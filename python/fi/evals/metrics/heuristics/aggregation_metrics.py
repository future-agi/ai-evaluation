import json
from typing import Any, Dict, List, Optional, TypeVar

from ..base_metric import BaseMetric, BaseMetricInputType
from ...types import BaseMetricInput


class AggregatedMetric(BaseMetric[BaseMetricInputType]):
    """
    Combines multiple metric evaluators into a single aggregated score.

    This metric assumes all sub-metrics can operate on the same input type.

    Config:
        - aggregator (str): 'average' or 'weighted_average'.
        - metrics (List[BaseMetricInput]): A list of instantiated metric objects.
        - weights (List[float]): Required if aggregator is 'weighted_average'.
    """

    SUPPORTED_AGGREGATORS = ["average", "weighted_average"]

    @property
    def metric_name(self) -> str:
        return "aggregated_metric"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.aggregator = self.config.get("aggregator", "average")
        self.metrics: List[BaseMetric] = self.config.get("metrics", [])
        self.weights: List[float] = self.config.get("weights", [])

        if self.aggregator not in self.SUPPORTED_AGGREGATORS:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")
        if not self.metrics:
            raise ValueError(
                "AggregatedMetric requires at least one metric in its config."
            )
        if not all(isinstance(m, BaseMetric) for m in self.metrics):
            raise TypeError("All items in 'metrics' must be instances of BaseMetric.")
        
        # Explicitly set the input model based on the first sub-metric
        self.input_model = self.metrics[0].input_model

        if self.aggregator == "weighted_average":
            if not self.weights or len(self.weights) != len(self.metrics):
                raise ValueError(
                    "Weights are required for 'weighted_average' and must match the number of metrics."
                )

    def _normalize_score(self, value: Any) -> float:
        """Converts various score types to a float, clamping between 0 and 1."""
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            float_value = float(value)
            return max(0.0, min(1.0, float_value))
        except (ValueError, TypeError):
            return 0.0

    def compute_one(self, inputs: BaseMetricInputType) -> Dict[str, Any]:
        metric_scores = []
        metric_details = {}

        for metric in self.metrics:
            try:
                result_dict = metric.compute_one(inputs)
                score = self._normalize_score(result_dict.get("output", 0.0))
            except Exception as e:
                # If a sub-metric fails, record a score of 0.0 for it
                score = 0.0

            metric_scores.append(score)
            metric_details[metric.metric_name] = score

        if not metric_scores:
            return {"output": 0.0, "reason": "No metric scores were produced."}

        if self.aggregator == "average":
            aggregated_score = sum(metric_scores) / len(metric_scores)
        else:  # weighted_average
            weighted_sum = sum(w * s for w, s in zip(self.weights, metric_scores))
            total_weight = sum(self.weights)
            aggregated_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        reason = f"Aggregated score calculated using '{self.aggregator}'. Details: {json.dumps(metric_details)}"
        return {"output": aggregated_score, "reason": reason}
