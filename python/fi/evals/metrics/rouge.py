from typing import Any, Dict, List, Optional, Union
from rouge_score import rouge_scorer

from fi.evals.templates import EvalTemplate
from fi.evals.types import BatchRunResult, EvalResult, EvalResultMetric
from fi.testcases import TestCase


class ROUGEScore(EvalTemplate):
    """
    ROUGEScore evaluates the similarity between a generated text (hypothesis) and a reference text using ROUGE metrics.

    Possible values for config:
        - rouge_type: str, one of ["rouge1", "rouge2", "rougeL"] (default: "rouge1")
        - use_stemmer: bool, whether to use stemming (default: True)

    Example usage:
        ROUGEScore(config={"rouge_type": "rouge2", "use_stemmer": False})

    Required TestCase fields:
        - response: str (the generated text)
        - expected_text: str (the reference text)
    """
    EVAL_ID = "77"
    NAME = "ROUGE Score"
    DESCRIPTION = "Calculates ROUGE score between a generated text and reference(s)"
    EVAL_TAGS = ["metric", "summarization", "nlp"]
    REQUIRED_KEYS = ["response", "expected_text"]
    OUTPUT = "score"
    EVAL_TYPE_ID = "metric"
    CONFIG_SCHEMA = {
        "rouge_type": "rouge1",
        "use_stemmer": True
    }
    CRITERIA = "Calculate ROUGE score between generated text and reference text"
    CHOICES = []
    MULTI_CHOICE = False
    VALID_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
    METRIC_KEYS = ["precision", "recall", "fmeasure"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.rouge_type = config.get("rouge_type", self.CONFIG_SCHEMA["rouge_type"])
        self.use_stemmer = config.get("use_stemmer", self.CONFIG_SCHEMA["use_stemmer"])

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

        if self.rouge_type not in self.VALID_ROUGE_TYPES:
            raise ValueError(f"Invalid rouge_type: {self.rouge_type}. Must be one of {self.VALID_ROUGE_TYPES}")

        try:
            self.scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=self.use_stemmer)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RougeScorer: {e}")

        super().__init__(config)

    def validate_input(self, inputs: List[TestCase]):
        for test_case in inputs:
            if not hasattr(test_case, "response") or test_case.response is None:
                raise ValueError("TestCase must have a 'response' field for ROUGE evaluation")
            if not hasattr(test_case, "expected_text") or test_case.expected_text is None:
                raise ValueError("TestCase must have an 'expected_text' field for ROUGE evaluation")
        return True

    def evaluate(self, inputs: List[TestCase]) -> BatchRunResult:
        """
        Evaluate a batch of test cases using ROUGE.
        Args:
            inputs (List[TestCase]): List of test cases with 'response' and 'expected_text'.
        Returns:
            BatchRunResult: Results for each test case.
        """
        try:
            self.validate_input(inputs)
        except Exception as e:
            return BatchRunResult(eval_results=[EvalResult(
                data=None,
                failure=True,
                reason=str(e),
                runtime=0,
                metadata={},
                metrics=[]
            )])

        eval_results = []
        for test_case in inputs:
            try:
                scores = self._calculate_rouge_score(test_case)
                metrics = [
                    EvalResultMetric(
                        id=f"rouge_{self.rouge_type}_{metric}",
                        value=scores.get(metric, 0.0)
                    ) for metric in self.METRIC_KEYS
                ]
                eval_result = EvalResult(
                    data=[scores],
                    failure=False,
                    reason="",
                    runtime=0,
                    metadata={},
                    metrics=metrics
                )
            except Exception as e:
                eval_result = EvalResult(
                    data=None,
                    failure=True,
                    reason=f"Exception during evaluation: {e}",
                    runtime=0,
                    metadata={},
                    metrics=[]
                )
            eval_results.append(eval_result)
        return BatchRunResult(eval_results=eval_results)

    def _calculate_rouge_score(self, test_case: TestCase) -> Dict[str, float]:
        """
        Calculate ROUGE score for a single test case.
        Args:
            test_case (TestCase): Test case with 'response' and 'expected_text'.
        Returns:
            Dict[str, float]: Dictionary with precision, recall, and fmeasure.
        """
        hypothesis = getattr(test_case, "response", None)
        reference = getattr(test_case, "expected_text", None)
        if hypothesis is None or reference is None:
            raise ValueError("Both 'response' and 'expected_text' must be provided in TestCase.")
        try:
            if not isinstance(hypothesis, str) or not isinstance(reference, str):
                raise TypeError("Both 'response' and 'expected_text' must be strings.")
            if not hypothesis.strip() or not reference.strip():
                return {k: 0.0 for k in self.METRIC_KEYS}
            scores = self.scorer.score(reference, hypothesis)
            rouge_scores = scores[self.rouge_type]
            return {
                "precision": rouge_scores.precision,
                "recall": rouge_scores.recall,
                "fmeasure": rouge_scores.fmeasure
            }
        except Exception as e:
            return {k: 0.0 for k in self.METRIC_KEYS}

    def _preprocess_text(self, text: Union[str, List[str]]) -> str:
        """
        Preprocess text by joining list of strings or returning the string as is.
        Args:
            text (Union[str, List[str]]): Text or list of text.
        Returns:
            str: Preprocessed text.
        """
        if isinstance(text, list):
            return " ".join(text)
        return text