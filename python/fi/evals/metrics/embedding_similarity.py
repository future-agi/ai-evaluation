from typing import List, Union, Any, Dict, Optional, Literal
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock

from fi.evals.templates import EvalTemplate
from fi.evals.types import EvalResult, EvalResultMetric, BatchRunResult
from fi.testcases import TestCase

class EmbeddingSimilarity(EvalTemplate):
    """
    Calculates semantic similarity between texts using sentence embeddings.

    Attributes:
        EVAL_ID (str): Unique identifier for the evaluation template. Example: "80"
        NAME (str): Name of the evaluation. Example: "Embedding Similarity"
        DESCRIPTION (str): Description of the evaluation.
        EVAL_TAGS (List[str]): Tags for categorizing the evaluation.
        REQUIRED_KEYS (List[str]): Required fields in the test case. Example: ["response", "expected_text"]
        OUTPUT (str): Output type. Example: "score"
        EVAL_TYPE_ID (str): Type identifier. Example: "metric"
        CONFIG_SCHEMA (Dict[str, Any]): Default configuration schema.
        CRITERIA (str): Criteria for evaluation.
        CHOICES (List[str]): Choices for multi-choice evaluations (empty here).
        MULTI_CHOICE (bool): Whether the evaluation is multi-choice.
        DEFAULT_SIMILARITY_METHOD (str): Default similarity method. Possible values: "cosine", "euclidean", "manhattan"
        DEFAULT_NORMALIZE (bool): Whether to normalize embeddings by default.
        DEFAULT_MODEL_NAME (str): Default model name for sentence embeddings.

    Possible values for similarity_method:
        - "cosine": Cosine similarity (default)
        - "euclidean": Euclidean distance
        - "manhattan": Manhattan (cityblock) distance

    Example usage:
        config = {
            "similarity_method": "cosine",
            "normalize": True,
            "model_name": "all-MiniLM-L6-v2"
        }
        metric = EmbeddingSimilarity(config)
        metric.evaluate([TestCase(response="A", expected_text="B")])
    """
    EVAL_ID = "80"
    NAME = "Embedding Similarity"
    DESCRIPTION = "Calculates semantic similarity between texts using sentence embeddings"
    EVAL_TAGS = ["metric", "semantic", "similarity", "nlp", "embedding"]
    REQUIRED_KEYS = ["response", "expected_text"]
    OUTPUT = "score"
    EVAL_TYPE_ID = "metric"
    CONFIG_SCHEMA = {
        "similarity_method": "cosine",
        "normalize": True,
        "model_name": "all-MiniLM-L6-v2"
    }
    CRITERIA = "Calculate semantic similarity between generated text and reference text"
    CHOICES = []
    MULTI_CHOICE = False
    DEFAULT_SIMILARITY_METHOD = "cosine"
    DEFAULT_NORMALIZE = True
    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
    SUPPORTED_METHODS = ["cosine", "euclidean", "manhattan"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.similarity_method = config.get("similarity_method", self.DEFAULT_SIMILARITY_METHOD)
        self.normalize = config.get("normalize", self.DEFAULT_NORMALIZE)
        self.model_name = config.get("model_name", self.DEFAULT_MODEL_NAME)

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

        if self.similarity_method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported similarity method: {self.similarity_method}. Supported methods: {self.SUPPORTED_METHODS}")

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError as e:
            raise ImportError(
                "EmbeddingSimilarity requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}")

        super().__init__(config)

    def validate_input(self, inputs: List[Any]):
        """
        Validate the input list for required fields.
        Args:
            inputs (List[TestCase]): List of test cases to validate.
        Returns:
            bool: True if valid, raises exception otherwise.
        """
        # Type ignore because parent expects LLMTestCase, but we support TestCase
        super().validate_input(inputs)  # type: ignore
        for test_case in inputs:
            if not hasattr(test_case, "response") or test_case.response is None:
                from fi.utils.errors import MissingRequiredConfigForEvalTemplate
                raise MissingRequiredConfigForEvalTemplate("response", self.name)
            if not hasattr(test_case, "expected_text") or test_case.expected_text is None:
                from fi.utils.errors import MissingRequiredConfigForEvalTemplate
                raise MissingRequiredConfigForEvalTemplate("expected_text", self.name)
        return True

    def evaluate(self, inputs: List[TestCase]) -> BatchRunResult:
        self.validate_input(inputs)
        eval_results = []
        for test_case in inputs:
            try:
                similarity_score = self._calculate_embedding_similarity(test_case)
                failure = False
                reason = ""
            except Exception as e:
                similarity_score = 0.0
                failure = True
                reason = str(e)
            eval_result = EvalResult(
                data=[similarity_score],
                failure=failure,
                reason=reason,
                runtime=0,
                metadata={},
                metrics=[
                    EvalResultMetric(
                        id="embedding_similarity",
                        value=similarity_score
                    )
                ]
            )
            eval_results.append(eval_result)
        return BatchRunResult(eval_results=eval_results)

    def _calculate_embedding_similarity(self, test_case: TestCase) -> float:
        prediction = str(getattr(test_case, "response", ""))
        reference = str(getattr(test_case, "expected_text", ""))
        if not prediction.strip() or not reference.strip():
            return 0.0
        try:
            embeddings = self.model.encode(
                [prediction, reference],
                normalize_embeddings=self.normalize
            )
        except Exception as e:
            raise RuntimeError(f"Failed to encode embeddings: {e}")
        return self._compute_similarity(embeddings[0], embeddings[1])

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        try:
            if self.similarity_method == 'cosine':
                return 1.0 - cosine(emb1, emb2)
            elif self.similarity_method == 'euclidean':
                dist = euclidean(emb1, emb2)
                return 1.0 / (1.0 + dist)
            elif self.similarity_method == 'manhattan':
                dist = cityblock(emb1, emb2)
                return 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unsupported similarity method: {self.similarity_method}")
        except ImportError as e:
            raise ImportError("scipy is required for similarity computation. Install with: pip install scipy") from e
        except Exception as e:
            raise RuntimeError(f"Failed to compute similarity: {e}")