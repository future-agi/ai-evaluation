import re
from typing import List, Union, Any, Dict, Optional

from fi.evals.templates import EvalTemplate
from fi.evals.types import EvalResult, EvalResultMetric, BatchRunResult
from fi.testcases import TestCase


class NumericDiff(EvalTemplate):
    """
    NumericDiff evaluates the numerical difference between a model's response and an expected value.

    Attributes:
        EVAL_ID (str): Unique identifier for the evaluation metric.
        NAME (str): Name of the evaluation metric.
        DESCRIPTION (str): Description of the metric's purpose.
        EVAL_TAGS (List[str]): Tags categorizing the metric.
        REQUIRED_KEYS (List[str]): Required fields in the test case.
        OUTPUT (str): Output type of the metric.
        EVAL_TYPE_ID (str): Type identifier for the metric.
        CONFIG_SCHEMA (Dict[str, Any]): Default configuration schema.
        CRITERIA (str): Criteria for the metric.
        CHOICES (List[str]): Choices for multi-choice metrics (empty here).
        MULTI_CHOICE (bool): Whether the metric is multi-choice.

    Args:
        config (Optional[Dict[str, Any]]): Configuration dictionary.

    Possible values for config:
        extract_numeric (bool): Whether to extract numeric values from text. Default: True
        normalized_result (bool): Whether to normalize the result. Default: True

    Example:
        >>> test_case = TestCase(response="The answer is 42", expected_text="40")
        >>> metric = NumericDiff()
        >>> metric.evaluate([test_case])
        # Returns a BatchRunResult with normalized difference score
    """
    EVAL_ID = "78"
    NAME = "Numeric Difference"
    DESCRIPTION = "Calculates the difference between numeric values, optionally extracting them from text"
    EVAL_TAGS = ["metric", "numerical", "comparison"]
    REQUIRED_KEYS = ["response", "expected_text"]
    OUTPUT = "score"
    EVAL_TYPE_ID = "metric"
    CONFIG_SCHEMA = {
        "extract_numeric": True,
        "normalized_result": True
    }
    CRITERIA = "Calculate numerical difference between generated value and reference value"
    CHOICES = []
    MULTI_CHOICE = False
    DEFAULT_DIFF_ON_ERROR = 1.0
    DEFAULT_DIFF_ON_ERROR_NONORM = float('inf')
    DEFAULT_DENOMINATOR = 1.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.extract_numeric = config.get("extract_numeric", self.CONFIG_SCHEMA["extract_numeric"])
        self.normalized_result = config.get("normalized_result", self.CONFIG_SCHEMA["normalized_result"])

        # Set required instance attributes for EvalTemplate
        self.name = self.NAME
        self.description = self.DESCRIPTION
        self.eval_id = self.EVAL_ID
        self.eval_tags = self.EVAL_TAGS
        self.required_keys = self.REQUIRED_KEYS
        self.output = self.OUTPUT
        self.eval_type_id = self.EVAL_TYPE_ID
        self.config_schema = self.CONFIG_SCHEMA
        self.criteria = self.CRITERIA
        self.choices = self.CHOICES
        self.multi_choice = self.MULTI_CHOICE

        super().__init__(config)

    def validate_input(self, inputs: List[TestCase]):
        for test_case in inputs:
            if not hasattr(test_case, "response") or test_case.response is None:
                raise ValueError(f"TestCase must have a '{self.REQUIRED_KEYS[0]}' field for NumericDiff evaluation")
            if not hasattr(test_case, "expected_text") or test_case.expected_text is None:
                raise ValueError(f"TestCase must have a '{self.REQUIRED_KEYS[1]}' field for NumericDiff evaluation")
        return True

    def evaluate(self, inputs: List[TestCase]) -> BatchRunResult:
        self.validate_input(inputs)
        eval_results = []
        for test_case in inputs:
            try:
                diff_score = self._calculate_numeric_diff(test_case)
                failure = False
                reason = ""
            except Exception as e:
                diff_score = self.DEFAULT_DIFF_ON_ERROR if self.normalized_result else self.DEFAULT_DIFF_ON_ERROR_NONORM
                failure = True
                reason = str(e)
            eval_result = EvalResult(
                data=[diff_score],
                failure=failure,
                reason=reason,
                runtime=0,
                metadata={},
                metrics=[
                    EvalResultMetric(
                        id="numeric_diff",
                        value=diff_score
                    )
                ]
            )
            eval_results.append(eval_result)
        return BatchRunResult(eval_results=eval_results)

    def _calculate_numeric_diff(self, test_case: TestCase) -> float:
        prediction = test_case.response
        reference = test_case.expected_text
        if prediction is None or reference is None:
            raise ValueError("Both response and expected_text must be provided.")
        pred_num = self._to_number(prediction)
        ref_num = self._to_number(reference)
        diff = abs(pred_num - ref_num)
        if self.normalized_result:
            denom = abs(ref_num) if ref_num != 0 else self.DEFAULT_DENOMINATOR
            return min(diff / denom, 1.0)
        else:
            return diff

    def _to_number(self, value: Union[str, float, int, None]) -> float:
        if value is None:
            raise ValueError("Value cannot be None for numeric conversion.")
        if isinstance(value, (int, float)):
            return float(value)
        if self.extract_numeric:
            match = re.search(r'-?\d+\.?\d*', str(value))
            if match:
                return float(match.group())
            else:
                raise ValueError(f"No numeric value found in: {value}")
        else:
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Cannot convert to number: {value}")