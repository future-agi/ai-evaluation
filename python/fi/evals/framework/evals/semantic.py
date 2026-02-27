"""
Semantic evaluation implementations.

Provides evaluations for semantic analysis of text:
- CoherenceEval: Evaluate logical coherence of a response
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..protocols import BaseEvaluation, register_evaluation


@dataclass
class SemanticEvalResult:
    """Result from semantic evaluation."""
    score: float  # 0.0 to 1.0
    passed: bool
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


@register_evaluation
class CoherenceEval(BaseEvaluation):
    """
    Evaluate logical coherence of a response.

    Checks if the response is internally consistent and logically sound
    by measuring sentence-to-sentence similarity.

    Required inputs:
        - response: The text to evaluate

    Optional inputs:
        - context: Additional context for coherence check

    Returns:
        SemanticEvalResult with coherence score

    Example:
        eval = CoherenceEval()
        result = eval.evaluate({
            "response": "The sun rises in the east. It sets in the west."
        })
        print(f"Coherence: {result.score}")
    """

    name = "coherence"
    version = "1.0.0"

    def __init__(
        self,
        threshold: float = 0.6,
        model_name: Optional[str] = None,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self._model = None

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        errors = []
        if "response" not in inputs:
            errors.append("Missing required field: response")
        return errors

    def get_span_attributes(self, result: SemanticEvalResult) -> Dict[str, Any]:
        return {
            "score": result.score,
            "passed": result.passed,
            "confidence": result.confidence,
            "threshold": self.threshold,
        }

    def evaluate(self, inputs: Dict[str, Any]) -> SemanticEvalResult:
        response = inputs["response"]
        context = inputs.get("context", "")

        sentences = self._split_sentences(response)

        if len(sentences) <= 1:
            return SemanticEvalResult(
                score=1.0,
                passed=True,
                details={"sentence_count": len(sentences)},
            )

        try:
            score = self._compute_coherence_score(sentences, context)
        except ImportError:
            score = self._compute_heuristic_coherence(sentences)

        return SemanticEvalResult(
            score=score,
            passed=score >= self.threshold,
            details={
                "sentence_count": len(sentences),
                "method": "embedding" if self._model else "heuristic",
            },
        )

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_coherence_score(self, sentences: List[str], context: str) -> float:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            if self._model is None:
                self._model = SentenceTransformer(self.model_name or "all-MiniLM-L6-v2")

            embeddings = self._model.encode(sentences)

            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(sim)

            return float(np.mean(similarities)) if similarities else 1.0
        except ImportError:
            raise

    def _compute_heuristic_coherence(self, sentences: List[str]) -> float:
        if len(sentences) <= 1:
            return 1.0

        scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())

            if not words1 or not words2:
                continue

            overlap = len(words1 & words2) / len(words1 | words2)
            scores.append(overlap)

        return sum(scores) / len(scores) if scores else 0.5
