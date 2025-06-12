import string
from typing import List, Union, Any, Dict, Optional

import Levenshtein

from fi.evals.templates import EvalTemplate
from fi.evals.types import EvalResult, EvalResultMetric, BatchRunResult
from fi.testcases import TestCase


class LevenshteinDistance(EvalTemplate):
    """
    Calculates the normalized Levenshtein (edit) distance between strings.

    Possible config values:
        - case_insensitive (bool): If True, ignores case when comparing strings. Default: False
        - remove_punctuation (bool): If True, removes punctuation before comparison. Default: False

    Required TestCase fields:
        - response (str): The generated or predicted text.
        - expected_text (str): The reference or ground truth text.

    Example usage:
        >>> config = {"case_insensitive": True, "remove_punctuation": True}
        >>> eval = LevenshteinDistance(config)
        >>> test_cases = [TestCase(response="Hello, World!", expected_text="hello world")]
        >>> result = eval.evaluate(test_cases)
        >>> print(result.eval_results[0].metrics[0].value)
        0.0
    """
    # Class-level constants
    EVAL_ID = "79"
    NAME = "Levenshtein Distance"
    DESCRIPTION = "Calculates the normalized Levenshtein (edit) distance between strings"
    EVAL_TAGS = ["metric", "string", "similarity", "nlp"]
    REQUIRED_KEYS = ["response", "expected_text"]
    OUTPUT = "score"
    EVAL_TYPE_ID = "metric"
    CONFIG_SCHEMA = {
        "case_insensitive": False,
        "remove_punctuation": False
    }
    CRITERIA = "Calculate edit distance between generated text and reference text"
    CHOICES = []
    MULTI_CHOICE = False
    METRIC_ID = "levenshtein_distance"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.case_insensitive = config.get("case_insensitive", self.CONFIG_SCHEMA["case_insensitive"])
        self.remove_punctuation = config.get("remove_punctuation", self.CONFIG_SCHEMA["remove_punctuation"])

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
                raise ValueError(f"TestCase must have a '{self.REQUIRED_KEYS[0]}' field for Levenshtein evaluation")
            if not hasattr(test_case, "expected_text") or test_case.expected_text is None:
                raise ValueError(f"TestCase must have an '{self.REQUIRED_KEYS[1]}' field for Levenshtein evaluation")
        return True

    def evaluate(self, inputs: List[TestCase]) -> BatchRunResult:
        self.validate_input(inputs)
        eval_results = []
        for test_case in inputs:
            try:
                distance_score = self._calculate_levenshtein_distance(test_case)
                failure = False
                reason = ""
            except Exception as e:
                distance_score = 1.0
                failure = True
                reason = str(e)
            eval_result = EvalResult(
                data=[distance_score],
                failure=failure,
                reason=reason,
                runtime=0,
                metadata={},
                metrics=[
                    EvalResultMetric(
                        id=self.METRIC_ID,
                        value=distance_score
                    )
                ]
            )
            eval_results.append(eval_result)
        return BatchRunResult(eval_results=eval_results)

    def _calculate_levenshtein_distance(self, test_case: TestCase) -> float:
        # Validate input for this single test case
        self.validate_input([test_case])
        prediction = test_case.response if test_case.response is not None else ""
        reference = test_case.expected_text if test_case.expected_text is not None else ""
        pred_proc = self._preprocess(prediction)
        ref_proc = self._preprocess(reference)
        max_len = max(len(pred_proc), len(ref_proc), 1)  # to prevent divide-by-zero
        try:
            distance = Levenshtein.distance(pred_proc, ref_proc)
        except Exception as e:
            raise RuntimeError(f"Error computing Levenshtein distance: {e}")
        return distance / max_len

    def _preprocess(self, text: Optional[str]) -> str:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if self.case_insensitive:
            text = text.lower()
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        return text