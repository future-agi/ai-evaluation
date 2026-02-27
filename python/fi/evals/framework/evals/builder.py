"""Custom evaluation builder for creating evaluations without full classes.

This module provides a builder pattern and decorator-based approach
for defining custom evaluations with minimal boilerplate.

Example using decorator:
    from fi.evals.framework.evals.builder import custom_eval

    @custom_eval("sentiment_score")
    def evaluate_sentiment(inputs):
        text = inputs["text"]
        # Your evaluation logic here
        positive_words = ["good", "great", "excellent"]
        score = sum(1 for w in text.lower().split() if w in positive_words) / 10
        return {"score": min(1.0, score), "passed": score > 0.5}

Example using builder:
    from fi.evals.framework.evals.builder import EvalBuilder

    sentiment_eval = (
        EvalBuilder("sentiment_score")
        .version("1.0.0")
        .required_fields(["text"])
        .threshold(0.5)
        .evaluator(lambda inputs: {"score": 0.8, "passed": True})
        .build()
    )
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps

from ..protocols import BaseEvaluation, register_evaluation


@dataclass
class CustomEvalResult:
    """Result from a custom evaluation.

    This flexible result type supports any evaluation output format.

    Attributes:
        score: Evaluation score between 0 and 1
        passed: Whether the evaluation passed
        confidence: Confidence in the result (0-1)
        details: Additional evaluation details
    """

    score: float
    passed: bool
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomEvalResult":
        """Create result from dictionary."""
        return cls(
            score=data.get("score", 0.0),
            passed=data.get("passed", False),
            confidence=data.get("confidence", 1.0),
            details={k: v for k, v in data.items()
                    if k not in ("score", "passed", "confidence")},
        )


class CustomEvaluation(BaseEvaluation):
    """Dynamically created evaluation from builder or decorator."""

    def __init__(
        self,
        name: str,
        evaluator_fn: Callable[[Dict[str, Any]], Union[Dict[str, Any], CustomEvalResult]],
        version: str = "1.0.0",
        required_fields: Optional[List[str]] = None,
        threshold: float = 0.7,
        description: str = "",
        span_attributes_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ):
        """Initialize custom evaluation.

        Args:
            name: Unique evaluation name
            evaluator_fn: Function that performs the evaluation
            version: Evaluation version
            required_fields: List of required input fields
            threshold: Score threshold for passing
            description: Human-readable description
            span_attributes_fn: Optional function to extract span attributes
        """
        self._name = name
        self._version = version
        self._evaluator_fn = evaluator_fn
        self._required_fields = required_fields or []
        self._threshold = threshold
        self._description = description
        self._span_attributes_fn = span_attributes_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def description(self) -> str:
        return self._description

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate required inputs."""
        errors = []
        for field in self._required_fields:
            if field not in inputs:
                errors.append(f"Missing required field: {field}")
        return errors

    def evaluate(self, inputs: Dict[str, Any]) -> CustomEvalResult:
        """Run the evaluation.

        Args:
            inputs: Dictionary of evaluation inputs

        Returns:
            CustomEvalResult with evaluation outcome
        """
        result = self._evaluator_fn(inputs)

        if isinstance(result, CustomEvalResult):
            return result
        elif isinstance(result, dict):
            return CustomEvalResult.from_dict(result)
        else:
            raise ValueError(
                f"Evaluator function must return dict or CustomEvalResult, "
                f"got {type(result).__name__}"
            )

    def get_span_attributes(self, result: CustomEvalResult) -> Dict[str, Any]:
        """Get span attributes for tracing.

        Args:
            result: Evaluation result

        Returns:
            Dictionary of span attributes
        """
        if self._span_attributes_fn:
            return self._span_attributes_fn(result)

        return {
            "score": result.score,
            "passed": result.passed,
            "confidence": result.confidence,
            "threshold": self._threshold,
        }


class EvalBuilder:
    """Builder for creating custom evaluations.

    Provides a fluent interface for configuring evaluations.

    Example:
        eval = (
            EvalBuilder("my_eval")
            .version("2.0.0")
            .required_fields(["text", "reference"])
            .threshold(0.8)
            .description("Checks text quality")
            .evaluator(my_evaluation_function)
            .build()
        )
    """

    def __init__(self, name: str):
        """Initialize builder with evaluation name.

        Args:
            name: Unique evaluation name
        """
        self._name = name
        self._version = "1.0.0"
        self._required_fields: List[str] = []
        self._threshold = 0.7
        self._description = ""
        self._evaluator_fn: Optional[Callable] = None
        self._span_attributes_fn: Optional[Callable] = None
        self._register = False

    def version(self, version: str) -> "EvalBuilder":
        """Set evaluation version.

        Args:
            version: Semantic version string

        Returns:
            Self for chaining
        """
        self._version = version
        return self

    def required_fields(self, fields: List[str]) -> "EvalBuilder":
        """Set required input fields.

        Args:
            fields: List of field names

        Returns:
            Self for chaining
        """
        self._required_fields = fields
        return self

    def require(self, *fields: str) -> "EvalBuilder":
        """Add required input fields.

        Args:
            *fields: Field names to require

        Returns:
            Self for chaining
        """
        self._required_fields.extend(fields)
        return self

    def threshold(self, threshold: float) -> "EvalBuilder":
        """Set score threshold for passing.

        Args:
            threshold: Threshold value (0-1)

        Returns:
            Self for chaining
        """
        self._threshold = threshold
        return self

    def description(self, description: str) -> "EvalBuilder":
        """Set human-readable description.

        Args:
            description: Description text

        Returns:
            Self for chaining
        """
        self._description = description
        return self

    def evaluator(
        self,
        fn: Callable[[Dict[str, Any]], Union[Dict[str, Any], CustomEvalResult]]
    ) -> "EvalBuilder":
        """Set the evaluator function.

        The function should take an inputs dict and return either:
        - A dict with at least "score" and "passed" keys
        - A CustomEvalResult object

        Args:
            fn: Evaluator function

        Returns:
            Self for chaining
        """
        self._evaluator_fn = fn
        return self

    def span_attributes(
        self,
        fn: Callable[[CustomEvalResult], Dict[str, Any]]
    ) -> "EvalBuilder":
        """Set custom span attributes extractor.

        Args:
            fn: Function to extract span attributes from result

        Returns:
            Self for chaining
        """
        self._span_attributes_fn = fn
        return self

    def auto_register(self, register: bool = True) -> "EvalBuilder":
        """Enable automatic registration with EvalRegistry.

        Args:
            register: Whether to auto-register

        Returns:
            Self for chaining
        """
        self._register = register
        return self

    def build(self) -> CustomEvaluation:
        """Build the custom evaluation.

        Returns:
            Configured CustomEvaluation instance

        Raises:
            ValueError: If evaluator function not set
        """
        if self._evaluator_fn is None:
            raise ValueError("Evaluator function must be set via .evaluator()")

        evaluation = CustomEvaluation(
            name=self._name,
            evaluator_fn=self._evaluator_fn,
            version=self._version,
            required_fields=self._required_fields,
            threshold=self._threshold,
            description=self._description,
            span_attributes_fn=self._span_attributes_fn,
        )

        if self._register:
            register_evaluation(evaluation.__class__)

        return evaluation


def custom_eval(
    name: str,
    version: str = "1.0.0",
    required_fields: Optional[List[str]] = None,
    threshold: float = 0.7,
    description: str = "",
    auto_register: bool = False,
) -> Callable:
    """Decorator for creating custom evaluations from functions.

    The decorated function should take an inputs dict and return either:
    - A dict with at least "score" and "passed" keys
    - A CustomEvalResult object

    Args:
        name: Unique evaluation name
        version: Evaluation version
        required_fields: List of required input fields
        threshold: Score threshold for passing
        description: Human-readable description
        auto_register: Whether to register with EvalRegistry

    Returns:
        Decorator function

    Example:
        @custom_eval("sentiment", required_fields=["text"])
        def evaluate_sentiment(inputs):
            text = inputs["text"]
            score = analyze_sentiment(text)
            return {"score": score, "passed": score > 0.5}

        # Use it
        result = evaluate_sentiment.evaluate({"text": "Great product!"})
    """
    def decorator(fn: Callable) -> CustomEvaluation:
        evaluation = CustomEvaluation(
            name=name,
            evaluator_fn=fn,
            version=version,
            required_fields=required_fields or [],
            threshold=threshold,
            description=description or fn.__doc__ or "",
        )

        if auto_register:
            # Create a class to register
            eval_class = type(
                f"Custom_{name}",
                (CustomEvaluation,),
                {"name": name, "version": version}
            )
            register_evaluation(eval_class)

        return evaluation

    return decorator


def simple_eval(
    name: str,
    scorer: Callable[[Dict[str, Any]], float],
    threshold: float = 0.7,
    required_fields: Optional[List[str]] = None,
) -> CustomEvaluation:
    """Create a simple evaluation from a scoring function.

    This is the simplest way to create a custom evaluation when you
    only need to compute a score.

    Args:
        name: Unique evaluation name
        scorer: Function that takes inputs and returns a score (0-1)
        threshold: Score threshold for passing
        required_fields: List of required input fields

    Returns:
        CustomEvaluation instance

    Example:
        word_count_eval = simple_eval(
            "word_count",
            scorer=lambda inputs: min(1.0, len(inputs["text"].split()) / 100),
            threshold=0.5,
            required_fields=["text"],
        )
    """
    def evaluator(inputs: Dict[str, Any]) -> Dict[str, Any]:
        score = scorer(inputs)
        return {
            "score": score,
            "passed": score >= threshold,
        }

    return CustomEvaluation(
        name=name,
        evaluator_fn=evaluator,
        required_fields=required_fields or [],
        threshold=threshold,
    )


def comparison_eval(
    name: str,
    comparator: Callable[[Any, Any], float],
    source_field: str = "response",
    target_field: str = "reference",
    threshold: float = 0.7,
) -> CustomEvaluation:
    """Create an evaluation that compares two fields.

    This is useful for evaluations that compare a response to a reference.

    Args:
        name: Unique evaluation name
        comparator: Function that compares source and target, returns score (0-1)
        source_field: Name of the source field
        target_field: Name of the target field
        threshold: Score threshold for passing

    Returns:
        CustomEvaluation instance

    Example:
        length_match_eval = comparison_eval(
            "length_match",
            comparator=lambda src, tgt: 1 - abs(len(src) - len(tgt)) / max(len(src), len(tgt), 1),
            threshold=0.8,
        )
    """
    def evaluator(inputs: Dict[str, Any]) -> Dict[str, Any]:
        source = inputs.get(source_field, "")
        target = inputs.get(target_field, "")
        score = comparator(source, target)
        return {
            "score": score,
            "passed": score >= threshold,
            "source_field": source_field,
            "target_field": target_field,
        }

    return CustomEvaluation(
        name=name,
        evaluator_fn=evaluator,
        required_fields=[source_field, target_field],
        threshold=threshold,
    )


def threshold_eval(
    name: str,
    metric_fn: Callable[[Dict[str, Any]], float],
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
    required_fields: Optional[List[str]] = None,
) -> CustomEvaluation:
    """Create an evaluation based on min/max thresholds.

    Pass if the metric is within the specified range.

    Args:
        name: Unique evaluation name
        metric_fn: Function that computes the metric value
        min_threshold: Minimum acceptable value (inclusive)
        max_threshold: Maximum acceptable value (inclusive)
        required_fields: List of required input fields

    Returns:
        CustomEvaluation instance

    Example:
        length_eval = threshold_eval(
            "response_length",
            metric_fn=lambda inputs: len(inputs["response"]),
            min_threshold=10,
            max_threshold=500,
            required_fields=["response"],
        )
    """
    def evaluator(inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = metric_fn(inputs)

        passed = True
        if min_threshold is not None and value < min_threshold:
            passed = False
        if max_threshold is not None and value > max_threshold:
            passed = False

        # Normalize score to 0-1
        if min_threshold is not None and max_threshold is not None:
            range_val = max_threshold - min_threshold
            if range_val > 0:
                score = max(0, min(1, (value - min_threshold) / range_val))
            else:
                score = 1.0 if value == min_threshold else 0.0
        elif min_threshold is not None:
            score = 1.0 if value >= min_threshold else value / min_threshold
        elif max_threshold is not None:
            score = 1.0 if value <= max_threshold else max_threshold / value
        else:
            score = 1.0

        return {
            "score": score,
            "passed": passed,
            "value": value,
            "min_threshold": min_threshold,
            "max_threshold": max_threshold,
        }

    return CustomEvaluation(
        name=name,
        evaluator_fn=evaluator,
        required_fields=required_fields or [],
        threshold=0.5,  # Not used, but required
    )


def pattern_match_eval(
    name: str,
    patterns: List[str],
    field: str = "response",
    mode: str = "any",  # "any", "all", "none"
    case_sensitive: bool = False,
) -> CustomEvaluation:
    """Create an evaluation based on pattern matching.

    Args:
        name: Unique evaluation name
        patterns: List of regex patterns
        field: Field to check patterns against
        mode: "any" (pass if any match), "all" (pass if all match),
              "none" (pass if none match)
        case_sensitive: Whether patterns are case-sensitive

    Returns:
        CustomEvaluation instance

    Example:
        has_greeting = pattern_match_eval(
            "has_greeting",
            patterns=[r"\\b(hello|hi|hey)\\b"],
            field="response",
            mode="any",
        )
    """
    import re
    flags = 0 if case_sensitive else re.IGNORECASE

    compiled_patterns = [re.compile(p, flags) for p in patterns]

    def evaluator(inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = str(inputs.get(field, ""))
        matches = [bool(p.search(text)) for p in compiled_patterns]

        if mode == "any":
            passed = any(matches)
            score = sum(matches) / len(matches) if matches else 0
        elif mode == "all":
            passed = all(matches)
            score = sum(matches) / len(matches) if matches else 0
        elif mode == "none":
            passed = not any(matches)
            score = 1 - (sum(matches) / len(matches)) if matches else 1
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {
            "score": score,
            "passed": passed,
            "mode": mode,
            "matches": matches,
            "pattern_count": len(patterns),
        }

    return CustomEvaluation(
        name=name,
        evaluator_fn=evaluator,
        required_fields=[field],
        threshold=0.5,
    )
