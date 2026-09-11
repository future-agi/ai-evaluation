import ast
import json
import re
import string
from typing import Any, Dict, List, Optional, Set, Union
import Levenshtein
from rouge_score import rouge_scorer

import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

from ..base_metric import BaseMetric
from ...types import TextMetricInput


class BLEUScore(BaseMetric[TextMetricInput]):
    """Calculates the BLEU score between a generated translation and reference(s)."""

    @property
    def metric_name(self) -> str:
        return "bleu_score"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.mode = self.config.get("mode", "sentence")
        self.max_n_gram = self.config.get("max_n_gram", 4)
        self.weights = self.config.get("weights") or self._default_weights(
            self.max_n_gram
        )
        self.smooth_method = self.config.get("smooth", "method1")
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    def _default_weights(self, max_n: int) -> List[float]:
        if max_n <= 0:
            raise ValueError("max_n_gram must be a positive integer")
        return [1.0 / max_n] * max_n

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        if inputs.expected_response is None:
            raise ValueError("BLEUScore requires 'expected_response' to be provided.")
        smooth_func = getattr(SmoothingFunction(), self.smooth_method, None)
        if smooth_func is None:
            raise ValueError(f"Invalid smoothing function: {self.smooth_method}")
        prediction_tokens = inputs.response.split()
        if isinstance(inputs.expected_response, str):
            reference_tokens = [inputs.expected_response.split()]
        else:
            reference_tokens = [ref.split() for ref in inputs.expected_response]
        if self.mode == "sentence":
            score = sentence_bleu(
                reference_tokens,
                prediction_tokens,
                weights=self.weights,
                smoothing_function=smooth_func,
            )
        elif self.mode == "corpus":
            score = corpus_bleu(
                [reference_tokens],
                [prediction_tokens],
                weights=self.weights,
                smoothing_function=smooth_func,
            )
        else:
            raise ValueError(
                f"Unsupported mode: {self.mode}. Use 'sentence' or 'corpus'."
            )
        return {
            "output": float(score),
            "reason": f"BLEU score calculated using mode: {self.mode}",
        }


class ROUGEScore(BaseMetric[TextMetricInput]):
    """Calculates ROUGE score between a generated text and a reference text."""

    VALID_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    @property
    def metric_name(self) -> str:
        return "rouge_score"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rouge_type = self.config.get("rouge_type", "rouge1")
        self.use_stemmer = self.config.get("use_stemmer", True)
        if self.rouge_type not in self.VALID_ROUGE_TYPES:
            raise ValueError(f"Invalid rouge_type: {self.rouge_type}.")
        self.scorer = rouge_scorer.RougeScorer(
            [self.rouge_type], use_stemmer=self.use_stemmer
        )

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        if inputs.expected_response is None:
            raise ValueError("ROUGEScore requires 'expected_response' to be provided.")
        if not isinstance(inputs.expected_response, str):
            raise TypeError("ROUGE score requires a single string for 'expected_response'.")
        if not inputs.response.strip() or not inputs.expected_response.strip():
            scores = {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
        else:
            rouge_scores = self.scorer.score(inputs.expected_response, inputs.response)
            result = rouge_scores[self.rouge_type]
            scores = {
                "precision": result.precision,
                "recall": result.recall,
                "fmeasure": result.fmeasure,
            }
        # The main output will be the F-measure, but we include all scores in the reason.
        return {
            "output": scores["fmeasure"],
            "reason": f"Scores for {self.rouge_type}: Precision={scores['precision']:.3f}, Recall={scores['recall']:.3f}, F-measure={scores['fmeasure']:.3f}",
        }


class RecallScore(BaseMetric[TextMetricInput]):
    """
    Calculates recall for retrieved items against ground truth items.
    Recall = (Number of overlapping items) / (Total number of ground truth items)
    """

    @property
    def metric_name(self) -> str:
        return "recall_score"

    def _parse_to_set(self, data: Union[str, List, Set]) -> Set[Any]:
        """Robustly parses input into a set."""
        if isinstance(data, set):
            return data
        if isinstance(data, list):
            return set(data)
        if isinstance(data, str):
            try:
                # First, try to parse as a JSON list
                parsed_data = json.loads(data)
                if isinstance(parsed_data, list):
                    return set(parsed_data)
            except (json.JSONDecodeError, TypeError):
                try:
                    # If JSON fails, try Python's literal evaluation (for lists, sets, etc.)
                    parsed_data = ast.literal_eval(data)
                    if isinstance(parsed_data, (list, set, tuple)):
                        return set(parsed_data)
                except (ValueError, SyntaxError):
                    # If all else fails, treat as a single-item list
                    return {data}
        # Fallback for other types
        return {data}

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        if inputs.expected_response is None:
            raise ValueError("RecallScore requires 'expected_response' to be provided.")
        retrieved_set = self._parse_to_set(inputs.response)
        ground_truth_set = self._parse_to_set(inputs.expected_response)

        if not ground_truth_set:
            # If there are no ground truth items, recall is vacuously perfect.
            return {
                "output": 1.0,
                "reason": "Recall is 1.0 as there were no ground truth items.",
            }

        true_positives = len(retrieved_set.intersection(ground_truth_set))
        total_relevant = len(ground_truth_set)

        recall = true_positives / total_relevant

        reason = f"Found {true_positives} of {total_relevant} ground truth items. Recall: {recall:.3f}"
        return {"output": recall, "reason": reason}


class LevenshteinSimilarity(BaseMetric[TextMetricInput]):
    """Calculates normalized Levenshtein similarity (1.0 - normalized distance)."""

    @property
    def metric_name(self) -> str:
        return "levenshtein_similarity"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.case_insensitive = self.config.get("case_insensitive", True)
        self.remove_punctuation = self.config.get("remove_punctuation", True)

    def _preprocess(self, text: str) -> str:
        if self.case_insensitive:
            text = text.lower()
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        if inputs.expected_response is None:
            raise ValueError("LevenshteinSimilarity requires 'expected_response' to be provided.")
        if not isinstance(inputs.expected_response, str):
            raise TypeError(
                "LevenshteinSimilarity requires 'expected_response' to be a string."
            )
        pred_proc = self._preprocess(inputs.response)
        ref_proc = self._preprocess(inputs.expected_response)
        max_len = max(len(pred_proc), len(ref_proc), 1)
        distance = Levenshtein.distance(pred_proc, ref_proc)
        normalized_distance = distance / max_len
        similarity = 1.0 - normalized_distance
        return {
            "output": similarity,
            "reason": f"Levenshtein distance was {distance} over a max length of {max_len}.",
        }


class NumericSimilarity(BaseMetric[TextMetricInput]):
    """Calculates similarity between numbers extracted from response and expected_response."""

    @property
    def metric_name(self) -> str:
        return "numeric_similarity"

    def _to_number(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"-?\d+\.?\d*", str(value))
        if match:
            return float(match.group())
        raise ValueError(f"No numeric value found in '{str(value)}'.")

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        if inputs.expected_response is None:
            raise ValueError("NumericSimilarity requires 'expected_response' to be provided.")
        if not isinstance(inputs.expected_response, str):
            raise TypeError(
                "NumericSimilarity requires 'expected_response' to be a string."
            )
        pred_num = self._to_number(inputs.response)
        ref_num = self._to_number(inputs.expected_response)
        diff = abs(pred_num - ref_num)
        max_abs = max(abs(pred_num), abs(ref_num), 1e-9)
        normalized_diff = diff / max_abs
        similarity = max(0.0, 1.0 - normalized_diff)
        return {
            "output": similarity,
            "reason": f"Numeric Diff: |{pred_num} - {ref_num}| = {diff:.3f}, Similarity: {similarity:.3f}",
        }


class EmbeddingSimilarity(BaseMetric[TextMetricInput]):
    """Calculates semantic similarity between texts using sentence embeddings."""

    SUPPORTED_METHODS = ["cosine", "euclidean", "manhattan"]

    @property
    def metric_name(self) -> str:
        return "embedding_similarity"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.similarity_method = self.config.get("similarity_method", "cosine")
        self.normalize = self.config.get("normalize", True)
        self.model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
        if self.similarity_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported similarity method: {self.similarity_method}."
            )
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        except ImportError as e:
            raise ImportError(
                "EmbeddingSimilarity requires `sentence-transformers`. Install with: pip install sentence-transformers"
            ) from e

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        from scipy.spatial.distance import cosine, euclidean, cityblock
        
        if inputs.expected_response is None:
            raise ValueError("EmbeddingSimilarity requires 'expected_response' to be provided.")
        if not isinstance(inputs.expected_response, str):
            raise TypeError(
                "EmbeddingSimilarity requires a single string for 'expected_response'."
            )
        if not inputs.response.strip() or not inputs.expected_response.strip():
            return {"output": 0.0, "reason": "Response or expected text is empty."}

        embeddings = self.model.encode(
            [inputs.response, inputs.expected_response], normalize_embeddings=self.normalize
        )
        emb1, emb2 = embeddings[0], embeddings[1]

        if self.similarity_method == "cosine":
            score = 1.0 - cosine(emb1, emb2)
            reason = f"Cosine similarity: {score:.3f}"
        elif self.similarity_method == "euclidean":
            dist = euclidean(emb1, emb2)
            score = 1.0 / (1.0 + dist)
            reason = f"Euclidean similarity: {score:.3f} (distance={dist:.3f})"
        else:  # manhattan
            dist = cityblock(emb1, emb2)
            score = 1.0 / (1.0 + dist)
            reason = f"Manhattan similarity: {score:.3f} (distance={dist:.3f})"
        return {"output": score, "reason": reason}


class SemanticListContains(BaseMetric[TextMetricInput]):
    """Determines if a text contains phrases semantically similar to reference phrases."""

    @property
    def metric_name(self) -> str:
        return "semantic_list_contains"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.case_insensitive = self.config.get("case_insensitive", True)
        self.remove_punctuation = self.config.get("remove_punctuation", True)
        self.match_all = self.config.get("match_all", False)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        except ImportError as e:
            raise ImportError(
                "SemanticListContains requires `sentence-transformers`. Install with: pip install sentence-transformers"
            ) from e

    def _preprocess(self, text: str) -> str:
        if self.case_insensitive:
            text = text.lower()
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        return text.strip()

    def _get_expected_phrases(self, expected_text: Union[str, List[str]]) -> List[str]:
        if isinstance(expected_text, list):
            return [str(item) for item in expected_text]
        if isinstance(expected_text, str):
            try:  # Try parsing string as JSON list
                parsed = json.loads(expected_text)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (json.JSONDecodeError, TypeError):
                pass
            return [
                expected_text
            ]  # Fallback to treating the whole string as one phrase
        return [str(expected_text)]

    def compute_one(self, inputs: TextMetricInput) -> Dict[str, Any]:
        from scipy.spatial.distance import cosine

        if inputs.expected_response is None:
            raise ValueError("SemanticListContains requires 'expected_response' to be provided.")
        if not inputs.response or not inputs.response.strip():
            return {"output": 0.0, "reason": "Response is empty."}

        expected_phrases = self._get_expected_phrases(inputs.expected_response)
        if not expected_phrases:
            return {"output": 0.0, "reason": "No expected phrases to match against."}

        response_proc = self._preprocess(inputs.response)
        phrases_proc = [self._preprocess(phrase) for phrase in expected_phrases]

        resp_embedding = self.model.encode(response_proc)
        phrase_embeddings = self.model.encode(phrases_proc)

        matches = []
        similarities = {}
        for i, phrase in enumerate(expected_phrases):
            similarity = 1.0 - cosine(resp_embedding, phrase_embeddings[i])
            matches.append(similarity >= self.similarity_threshold)
            similarities[phrase] = round(similarity, 3)

        result = all(matches) if self.match_all else any(matches)
        reason = f"Matched {sum(matches)}/{len(matches)} phrases. Similarities: {json.dumps(similarities)}. Threshold: {self.similarity_threshold}."
        return {"output": 1.0 if result else 0.0, "reason": reason}
