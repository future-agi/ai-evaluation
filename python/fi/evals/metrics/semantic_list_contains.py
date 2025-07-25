import logging
import re
from typing import List, Union, Any, Dict, Optional, Tuple
import json
import string
import numpy as np
from scipy.spatial.distance import cosine

from fi.evals.templates import EvalTemplate
from fi.evals.types import EvalResult, EvalResultMetric, BatchRunResult
from fi.testcases import TestCase

class SemanticListContains(EvalTemplate):
    """
    SemanticListContains evaluates if a text contains phrases semantically similar to reference phrases.

    Attributes:
        eval_id (str): Unique identifier for the evaluation. Example: '81'.
        name (str): Name of the evaluation. Example: 'Semantic List Contains'.
        description (str): Description of the evaluation.
        eval_tags (List[str]): Tags for categorization. Example: ['metric', 'semantic', 'containment', 'nlp', 'embedding'].
        required_keys (List[str]): Required fields in TestCase. Example: ['response', 'expected_text'].
        output (str): Output type. Example: 'score'.
        eval_type_id (str): Type identifier. Example: 'metric'.
        config_schema (Dict[str, Any]): Default configuration schema.
            - case_insensitive (bool): Whether to ignore case. Example: True.
            - remove_punctuation (bool): Whether to remove punctuation. Example: True.
            - match_all (bool): Require all phrases to match. Example: False.
            - similarity_threshold (float): Similarity threshold. Example: 0.7.
            - model_name (str): Embedding model name. Example: 'all-MiniLM-L6-v2'.
        criteria (str): Evaluation criteria.
        choices (List): Choices for evaluation (unused).
        multi_choice (bool): If multiple choices are allowed (unused).

    Example usage:
        config = {
            'case_insensitive': True,
            'remove_punctuation': True,
            'match_all': False,
            'similarity_threshold': 0.7,
            'model_name': 'all-MiniLM-L6-v2'
        }
        evaluator = SemanticListContains(config)
        result = evaluator.evaluate([TestCase(response='...', expected_text='...')])
    """
    eval_id = "81"
    name = "Semantic List Contains"
    description = "Determines if a text contains phrases semantically similar to reference phrases"
    eval_tags = ["metric", "semantic", "containment", "nlp", "embedding"]
    required_keys = ["response", "expected_text"]
    output = "score"
    eval_type_id = "metric"
    config_schema = {
        "case_insensitive": True,
        "remove_punctuation": True,
        "match_all": False,
        "similarity_threshold": 0.7,
        "model_name": "all-MiniLM-L6-v2"
    }
    criteria = "Check if text contains phrases semantically similar to reference phrases"
    choices = []
    multi_choice = False

    # Class-level variable names for keys
    RESPONSE_KEY = "response"
    EXPECTED_TEXT_KEY = "expected_text"
    METRIC_ID = "semantic_list_contains"
    EMPTY_RESPONSE_REASON = "Empty response"
    NO_EXPECTED_REASON = "No expected phrases to match"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.case_insensitive = config.get("case_insensitive", self.config_schema["case_insensitive"])
        self.remove_punctuation = config.get("remove_punctuation", self.config_schema["remove_punctuation"])
        self.match_all = config.get("match_all", self.config_schema["match_all"])
        self.similarity_threshold = config.get("similarity_threshold", self.config_schema["similarity_threshold"])
        self.model_name = config.get("model_name", self.config_schema["model_name"])

        # Set required instance attributes for EvalTemplate
        self.name = self.__class__.name
        self.description = self.__class__.description
        self.eval_id = self.__class__.eval_id
        self.eval_tags = self.__class__.eval_tags
        self.required_keys = self.__class__.required_keys
        self.output = self.__class__.output
        self.eval_type_id = self.__class__.eval_type_id
        self.config_schema = self.__class__.config_schema
        self.criteria = self.__class__.criteria
        self.choices = self.__class__.choices
        self.multi_choice = self.__class__.multi_choice

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "SemanticListContains requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {e}")

        super().__init__(config)

    def validate_input(self, inputs: List[TestCase]):
        for test_case in inputs:
            if not hasattr(test_case, self.RESPONSE_KEY) or getattr(test_case, self.RESPONSE_KEY) is None:
                raise ValueError(f"TestCase must have a '{self.RESPONSE_KEY}' field for SemanticListContains evaluation")
            if not hasattr(test_case, self.EXPECTED_TEXT_KEY) or getattr(test_case, self.EXPECTED_TEXT_KEY) is None:
                raise ValueError(f"TestCase must have an '{self.EXPECTED_TEXT_KEY}' field for SemanticListContains evaluation")
        return True

    def evaluate(self, inputs: List[TestCase]) -> BatchRunResult:
        try:
            self.validate_input(inputs)
        except Exception as e:
            eval_result = EvalResult(
                name="semantic_list_contains",
                output=0.0,
                reason=str(e),
                runtime=0,
                output_type="score",
                eval_id=None,
            )
            return BatchRunResult(eval_results=[eval_result])

        eval_results = []
        for test_case in inputs:
            try:
                result, details = self._check_semantic_containment(test_case)
                score = 1.0 if result else 0.0
                failure = False
                reason = ""
            except Exception as e:
                score = 0.0
                failure = True
                reason = str(e)
                details = {}
            eval_result = EvalResult(
                name="semantic_list_contains",
                output=score,
                reason=reason,
                runtime=0,
                output_type="score",
                eval_id=None,
            )
            eval_results.append(eval_result)
        return BatchRunResult(eval_results=eval_results)

    def _check_semantic_containment(self, test_case: TestCase) -> Tuple[bool, Dict[str, Any]]:
        try:
            response = getattr(test_case, self.RESPONSE_KEY, None)
            expected_phrases = self._get_expected_phrases(getattr(test_case, self.EXPECTED_TEXT_KEY, None))

            if not isinstance(response, str) or not response or not response.strip():
                return False, {"reason": self.EMPTY_RESPONSE_REASON}
            if not expected_phrases:
                return False, {"reason": self.NO_EXPECTED_REASON}

            response_proc = self._preprocess(response)
            phrases_proc = [self._preprocess(phrase) for phrase in expected_phrases]

            try:
                resp_embedding = self.model.encode(response_proc)
                phrase_embeddings = self.model.encode(phrases_proc)
            except Exception as e:
                raise RuntimeError(f"Embedding model failed: {e}")

            matches = []
            similarities = {}
            for i, phrase in enumerate(expected_phrases):
                try:
                    similarity = 1.0 - cosine(resp_embedding, phrase_embeddings[i])
                except Exception as e:
                    similarity = 0.0
                matches.append(similarity >= self.similarity_threshold)
                similarities[phrase] = similarity

            result = all(matches) if self.match_all else any(matches)
            details = {
                "matches": matches,
                "similarities": similarities,
                "threshold": self.similarity_threshold,
                "match_all": self.match_all
            }
            return result, details
        except Exception as e:
            raise RuntimeError(f"Error in semantic containment check: {e}")

    def _preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        if self.case_insensitive:
            text = text.lower()
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()

    def _get_expected_phrases(self, expected_text) -> List[str]:
        if expected_text is None:
            return []
        if isinstance(expected_text, str):
            if (expected_text.startswith('[') and expected_text.endswith(']')) or \
               (expected_text.startswith('{') and expected_text.endswith('}')):
                try:
                    parsed = json.loads(expected_text)
                    if isinstance(parsed, list):
                        return parsed
                    return [expected_text]
                except Exception:
                    return [expected_text]
            return [expected_text]
        elif isinstance(expected_text, list):
            return expected_text
        else:
            return [str(expected_text)]
