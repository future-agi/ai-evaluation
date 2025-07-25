import logging
import re
from typing import Any, Dict, List, Optional, Union

import nltk
from nltk.translate.bleu_score import (
    sentence_bleu,
    corpus_bleu,
    SmoothingFunction
)

from fi.evals.templates import EvalTemplate
from fi.evals.types import EvalResult, EvalResultMetric, BatchRunResult
from fi.testcases import TestCase


class BLEUScore(EvalTemplate):
    """
    BLEUScore Evaluation Metric

    Calculates the BLEU score between a generated translation and reference(s).

    Possible values for config:
        mode: "sentence" or "corpus" (default: "sentence")
        max_n_gram: int, maximum n-gram to use (default: 4)
        smooth: str, smoothing method from nltk.translate.bleu_score.SmoothingFunction (default: "method1")
        weights: Optional[List[float]], custom n-gram weights (default: uniform)

    Example usage:
        config = {
            "mode": "sentence",
            "max_n_gram": 4,
            "smooth": "method1"
        }
        bleu = BLEUScore(config)
        bleu.evaluate([TestCase(response="a cat sits", expected_text="a cat is sitting")])
    """
    EVAL_ID = "76"
    NAME = "BLEU Score"
    DESCRIPTION = "Calculates BLEU score between a generated translation and reference(s)"
    EVAL_TAGS = ["metric", "translation", "nlp"]
    REQUIRED_KEYS = ["response", "expected_text"]
    OUTPUT = "score"
    EVAL_TYPE_ID = "metric"
    DEFAULT_MODE = "sentence"
    DEFAULT_MAX_N_GRAM = 4
    DEFAULT_SMOOTH = "method1"
    DEFAULT_CRITERIA = "Calculate BLEU score between generated text and reference text"
    DEFAULT_CHOICES = []
    DEFAULT_MULTI_CHOICE = False
    CONFIG_SCHEMA = {
        "mode": DEFAULT_MODE,
        "max_n_gram": DEFAULT_MAX_N_GRAM,
        "smooth": DEFAULT_SMOOTH
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.mode = config.get("mode", self.DEFAULT_MODE)
        self.max_n_gram = config.get("max_n_gram", self.DEFAULT_MAX_N_GRAM)
        self.weights = config.get("weights") or self._default_weights(self.max_n_gram)
        self.smooth = config.get("smooth", self.DEFAULT_SMOOTH)

        # Set required instance attributes for EvalTemplate
        self.name = self.NAME
        self.description = self.DESCRIPTION
        self.eval_id = self.EVAL_ID
        self.eval_tags = self.EVAL_TAGS
        self.required_keys = self.REQUIRED_KEYS
        self.output = self.OUTPUT
        self.eval_type_id = self.EVAL_TYPE_ID
        self.config_schema = self.CONFIG_SCHEMA
        self.criteria = self.DEFAULT_CRITERIA
        self.choices = self.DEFAULT_CHOICES
        self.multi_choice = self.DEFAULT_MULTI_CHOICE

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            raise RuntimeError(f"Error loading NLTK punkt tokenizer: {e}")

        super().__init__(config)

    def _default_weights(self, max_n: int) -> List[float]:
        if max_n <= 0:
            raise ValueError("max_n_gram must be a positive integer")
        return [1.0 / max_n] * max_n

    def validate_input(self, inputs: List[TestCase]):
        for test_case in inputs:
            if not hasattr(test_case, "response") or test_case.response is None:
                raise ValueError("TestCase must have a 'response' field for BLEU evaluation")
            if not hasattr(test_case, "expected_text") or test_case.expected_text is None:
                raise ValueError("TestCase must have an 'expected_text' field for BLEU evaluation")
        return True

    def evaluate(self, inputs: List[TestCase]) -> BatchRunResult:
        self.validate_input(inputs)
        eval_results = []
        for test_case in inputs:
            try:
                score = self._calculate_bleu_score(test_case)
                eval_result = EvalResult(
                    name="bleu_score",
                    output=score,
                    reason=None,
                    runtime=0,
                    output_type="score",
                    eval_id=None,
                )
            except Exception as e:
                eval_result = EvalResult(
                    name="bleu_score",
                    output=0.0,
                    reason=str(e),
                    runtime=0,
                    output_type="score",
                    eval_id=None,
                )
            eval_results.append(eval_result)
        return BatchRunResult(eval_results=eval_results)

    def _calculate_bleu_score(self, test_case: TestCase) -> float:
        if not hasattr(test_case, "response") or not hasattr(test_case, "expected_text"):
            raise ValueError("TestCase must have 'response' and 'expected_text' fields.")
        if test_case.response is None or test_case.expected_text is None:
            raise ValueError("TestCase 'response' and 'expected_text' cannot be None.")
        smooth_func = getattr(SmoothingFunction(), self.smooth, None)
        if smooth_func is None:
            raise ValueError(f"Invalid smoothing function: {self.smooth}")
        if self.mode == self.DEFAULT_MODE:
            # Sentence BLEU
            if isinstance(test_case.expected_text, str):
                reference = [test_case.expected_text.split()]
            elif isinstance(test_case.expected_text, list):
                reference = [r.split() if isinstance(r, str) else r for r in test_case.expected_text if isinstance(r, (str, list))]
                if not reference:
                    reference = [[]]
            else:
                reference = [[]]
            if not isinstance(test_case.response, str):
                raise ValueError("TestCase 'response' must be a string for BLEU evaluation.")
            prediction = test_case.response.split()
            try:
                score = sentence_bleu(
                    reference,
                    prediction,
                    weights=self.weights,
                    smoothing_function=smooth_func
                )
                score = float(score) if isinstance(score, (float, int)) else 0.0
            except Exception as e:
                raise RuntimeError(f"Error computing sentence BLEU: {e}")
            return score
        elif self.mode == "corpus":
            # Corpus BLEU
            if not isinstance(test_case.response, str):
                raise ValueError("TestCase 'response' must be a string for BLEU evaluation.")
            prediction = test_case.response.split()
            if isinstance(test_case.expected_text, str):
                references = [[test_case.expected_text.split()]]
            elif isinstance(test_case.expected_text, list):
                references = [[r.split() if isinstance(r, str) else r] for r in test_case.expected_text if isinstance(r, (str, list))]
                if not references:
                    references = [[[]]]
            else:
                references = [[[]]]
            try:
                score = corpus_bleu(
                    references,
                    [prediction],
                    weights=self.weights,
                    smoothing_function=smooth_func
                )
                score = float(score) if isinstance(score, (float, int)) else 0.0
            except Exception as e:
                raise RuntimeError(f"Error computing corpus BLEU: {e}")
            return score
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported modes are 'sentence' and 'corpus'.")