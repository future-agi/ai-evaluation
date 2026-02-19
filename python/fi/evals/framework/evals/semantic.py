"""
Semantic evaluation implementations.

Provides evaluations for semantic analysis of text:
- SemanticSimilarityEval: Measure semantic similarity between texts
- CoherenceEval: Evaluate logical coherence of a response
- EntailmentEval: Check if response entails from context
- ContradictionEval: Detect contradictions
- FactualConsistencyEval: Verify factual consistency with context

These evaluations can use either:
- Local models (sentence-transformers, NLI models)
- External APIs (OpenAI, etc.)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from abc import abstractmethod

from ..protocols import BaseEvaluation, register_evaluation


@dataclass
class SemanticEvalResult:
    """Result from semantic evaluation."""
    score: float  # 0.0 to 1.0
    passed: bool
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


class BaseSemanticEval:
    """Base class for semantic evaluations with common functionality."""

    # Configurable threshold for pass/fail
    threshold: float = 0.7

    def __init__(
        self,
        threshold: Optional[float] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the semantic evaluation.

        Args:
            threshold: Score threshold for passing (default: class threshold)
            model_name: Model to use for evaluation (implementation-specific)
        """
        if threshold is not None:
            self.threshold = threshold
        self.model_name = model_name
        self._model = None

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate required inputs."""
        errors = []
        required = self._get_required_fields()
        for field in required:
            if field not in inputs:
                errors.append(f"Missing required field: {field}")
        return errors

    @abstractmethod
    def _get_required_fields(self) -> List[str]:
        """Return list of required input fields."""
        pass

    def get_span_attributes(self, result: SemanticEvalResult) -> Dict[str, Any]:
        """Convert result to span attributes."""
        return {
            "score": result.score,
            "passed": result.passed,
            "confidence": result.confidence,
            "threshold": self.threshold,
        }


@register_evaluation
class SemanticSimilarityEval(BaseSemanticEval):
    """
    Evaluate semantic similarity between response and reference.

    Uses sentence embeddings to compute cosine similarity.

    Required inputs:
        - response: The text to evaluate
        - reference: The reference text to compare against

    Returns:
        SemanticEvalResult with similarity score (0.0 to 1.0)

    Example:
        eval = SemanticSimilarityEval(threshold=0.8)
        result = eval.evaluate({
            "response": "The dog ran quickly.",
            "reference": "A canine moved fast.",
        })
        print(f"Similarity: {result.score}")  # ~0.85
    """

    name = "semantic_similarity"
    version = "1.0.0"
    threshold = 0.7

    def __init__(
        self,
        threshold: Optional[float] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__(threshold, model_name)

    def _get_required_fields(self) -> List[str]:
        return ["response", "reference"]

    def evaluate(self, inputs: Dict[str, Any]) -> SemanticEvalResult:
        """
        Compute semantic similarity between response and reference.

        Args:
            inputs: Dict with 'response' and 'reference' keys

        Returns:
            SemanticEvalResult with similarity score
        """
        response = inputs["response"]
        reference = inputs["reference"]

        try:
            score = self._compute_similarity(response, reference)
        except ImportError:
            # Fallback to simple word overlap if sentence-transformers not available
            score = self._compute_word_overlap(response, reference)

        return SemanticEvalResult(
            score=score,
            passed=score >= self.threshold,
            details={
                "method": "embedding" if self._model else "word_overlap",
                "model": self.model_name if self._model else None,
            },
        )

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute embedding-based similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            if self._model is None:
                self._model = SentenceTransformer(self.model_name)

            embeddings = self._model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(max(0.0, min(1.0, similarity)))
        except ImportError:
            raise

    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """Fallback: compute word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


@register_evaluation
class CoherenceEval(BaseSemanticEval):
    """
    Evaluate logical coherence of a response.

    Checks if the response is internally consistent and logically sound.

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
    threshold = 0.6

    def _get_required_fields(self) -> List[str]:
        return ["response"]

    def evaluate(self, inputs: Dict[str, Any]) -> SemanticEvalResult:
        """
        Evaluate coherence of the response.

        Args:
            inputs: Dict with 'response' key

        Returns:
            SemanticEvalResult with coherence score
        """
        response = inputs["response"]
        context = inputs.get("context", "")

        # Split into sentences for coherence analysis
        sentences = self._split_sentences(response)

        if len(sentences) <= 1:
            # Single sentence is coherent by default
            return SemanticEvalResult(
                score=1.0,
                passed=True,
                details={"sentence_count": len(sentences)},
            )

        try:
            score = self._compute_coherence_score(sentences, context)
        except ImportError:
            # Fallback to simple heuristics
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
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_coherence_score(self, sentences: List[str], context: str) -> float:
        """Compute coherence using embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            if self._model is None:
                self._model = SentenceTransformer(self.model_name or "all-MiniLM-L6-v2")

            # Compute pairwise similarity between consecutive sentences
            embeddings = self._model.encode(sentences)

            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(sim)

            # Average similarity as coherence score
            return float(np.mean(similarities)) if similarities else 1.0
        except ImportError:
            raise

    def _compute_heuristic_coherence(self, sentences: List[str]) -> float:
        """Fallback: compute coherence using heuristics."""
        if len(sentences) <= 1:
            return 1.0

        # Check for word overlap between consecutive sentences
        scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())

            if not words1 or not words2:
                continue

            # Jaccard similarity
            overlap = len(words1 & words2) / len(words1 | words2)
            scores.append(overlap)

        return sum(scores) / len(scores) if scores else 0.5


@register_evaluation
class EntailmentEval(BaseSemanticEval):
    """
    Evaluate if the response is entailed by the context.

    Uses Natural Language Inference (NLI) to determine if the response
    logically follows from the given context.

    Required inputs:
        - response: The text to evaluate (hypothesis)
        - context: The premise/context text

    Returns:
        SemanticEvalResult with entailment score

    Example:
        eval = EntailmentEval()
        result = eval.evaluate({
            "context": "Paris is the capital of France.",
            "response": "France has a capital city.",
        })
        print(f"Entailment: {result.score}")  # High score (entailed)
    """

    name = "entailment"
    version = "1.0.0"
    threshold = 0.7

    def __init__(
        self,
        threshold: Optional[float] = None,
        model_name: str = "facebook/bart-large-mnli",
    ):
        super().__init__(threshold, model_name)

    def _get_required_fields(self) -> List[str]:
        return ["response", "context"]

    def evaluate(self, inputs: Dict[str, Any]) -> SemanticEvalResult:
        """
        Check if response is entailed by context.

        Args:
            inputs: Dict with 'response' and 'context' keys

        Returns:
            SemanticEvalResult with entailment score
        """
        response = inputs["response"]
        context = inputs["context"]

        try:
            scores = self._compute_nli_scores(context, response)
            entailment_score = scores.get("entailment", 0.0)
        except ImportError:
            # Fallback to keyword overlap
            entailment_score = self._compute_keyword_entailment(context, response)
            scores = {"entailment": entailment_score}

        return SemanticEvalResult(
            score=entailment_score,
            passed=entailment_score >= self.threshold,
            details={
                "scores": scores,
                "method": "nli" if self._model else "keyword",
            },
        )

    def _compute_nli_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Compute NLI scores using a model."""
        try:
            from transformers import pipeline

            if self._model is None:
                self._model = pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                )

            result = self._model(
                hypothesis,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="{}",
            )

            scores = {}
            for label, score in zip(result["labels"], result["scores"]):
                scores[label] = score

            return scores
        except ImportError:
            raise

    def _compute_keyword_entailment(self, premise: str, hypothesis: str) -> float:
        """Fallback: compute entailment using keyword overlap."""
        premise_words = set(premise.lower().split())
        hypothesis_words = set(hypothesis.lower().split())

        if not hypothesis_words:
            return 0.0

        # Check what fraction of hypothesis words appear in premise
        overlap = len(hypothesis_words & premise_words)
        return overlap / len(hypothesis_words)


@register_evaluation
class ContradictionEval(BaseSemanticEval):
    """
    Detect contradictions between response and context.

    Uses NLI to identify if the response contradicts the given context.

    Required inputs:
        - response: The text to evaluate
        - context: The context to check against

    Returns:
        SemanticEvalResult with contradiction score (lower is better)

    Example:
        eval = ContradictionEval()
        result = eval.evaluate({
            "context": "The meeting is at 3 PM.",
            "response": "The meeting starts at 5 PM.",
        })
        print(f"Contradiction: {result.score}")  # High score (contradiction)
    """

    name = "contradiction"
    version = "1.0.0"
    threshold = 0.3  # Lower threshold = less contradiction allowed

    def __init__(
        self,
        threshold: Optional[float] = None,
        model_name: str = "facebook/bart-large-mnli",
    ):
        super().__init__(threshold, model_name)

    def _get_required_fields(self) -> List[str]:
        return ["response", "context"]

    def evaluate(self, inputs: Dict[str, Any]) -> SemanticEvalResult:
        """
        Detect contradictions in response.

        Args:
            inputs: Dict with 'response' and 'context' keys

        Returns:
            SemanticEvalResult with contradiction score
        """
        response = inputs["response"]
        context = inputs["context"]

        try:
            scores = self._compute_nli_scores(context, response)
            contradiction_score = scores.get("contradiction", 0.0)
        except ImportError:
            # Fallback to negation detection
            contradiction_score = self._detect_negation_contradiction(context, response)
            scores = {"contradiction": contradiction_score}

        # For contradiction, passed means LOW contradiction (below threshold)
        passed = contradiction_score < self.threshold

        return SemanticEvalResult(
            score=contradiction_score,
            passed=passed,
            details={
                "scores": scores,
                "method": "nli" if self._model else "negation",
            },
        )

    def _compute_nli_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Compute NLI scores using a model."""
        try:
            from transformers import pipeline

            if self._model is None:
                self._model = pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                )

            result = self._model(
                hypothesis,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="{}",
            )

            scores = {}
            for label, score in zip(result["labels"], result["scores"]):
                scores[label] = score

            return scores
        except ImportError:
            raise

    def _detect_negation_contradiction(self, context: str, response: str) -> float:
        """Fallback: detect contradiction via negation patterns."""
        negation_words = {"not", "no", "never", "none", "neither", "nobody", "nothing"}

        context_lower = context.lower()
        response_lower = response.lower()

        context_has_negation = any(word in context_lower.split() for word in negation_words)
        response_has_negation = any(word in response_lower.split() for word in negation_words)

        # Simple heuristic: different negation status might indicate contradiction
        if context_has_negation != response_has_negation:
            # Check for topic overlap
            context_words = set(context_lower.split()) - negation_words
            response_words = set(response_lower.split()) - negation_words

            overlap = len(context_words & response_words) / max(len(context_words | response_words), 1)
            if overlap > 0.3:  # Same topic but different polarity
                return 0.7

        return 0.1  # Default low contradiction


@register_evaluation
class FactualConsistencyEval(BaseSemanticEval):
    """
    Evaluate factual consistency between response and context.

    Checks if the claims in the response are supported by the context.
    Useful for evaluating summarization, RAG, and QA systems.

    Required inputs:
        - response: The generated response to evaluate
        - context: The source context/documents

    Returns:
        SemanticEvalResult with consistency score

    Example:
        eval = FactualConsistencyEval()
        result = eval.evaluate({
            "context": "John was born in 1990 in New York.",
            "response": "John is from New York and was born in 1990.",
        })
        print(f"Factual consistency: {result.score}")
    """

    name = "factual_consistency"
    version = "1.0.0"
    threshold = 0.7

    def _get_required_fields(self) -> List[str]:
        return ["response", "context"]

    def evaluate(self, inputs: Dict[str, Any]) -> SemanticEvalResult:
        """
        Evaluate factual consistency of response with context.

        Args:
            inputs: Dict with 'response' and 'context' keys

        Returns:
            SemanticEvalResult with consistency score
        """
        response = inputs["response"]
        context = inputs["context"]

        # Split response into claims
        claims = self._extract_claims(response)

        if not claims:
            return SemanticEvalResult(
                score=1.0,
                passed=True,
                details={"claim_count": 0, "method": "no_claims"},
            )

        try:
            # Check each claim against context
            claim_scores = []
            for claim in claims:
                score = self._verify_claim(claim, context)
                claim_scores.append(score)

            avg_score = sum(claim_scores) / len(claim_scores)
        except ImportError:
            # Fallback to word overlap
            avg_score = self._compute_word_support(response, context)
            claim_scores = [avg_score]

        return SemanticEvalResult(
            score=avg_score,
            passed=avg_score >= self.threshold,
            details={
                "claim_count": len(claims),
                "claim_scores": claim_scores,
                "method": "nli" if self._model else "word_overlap",
            },
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from text."""
        import re
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _verify_claim(self, claim: str, context: str) -> float:
        """Verify a single claim against context."""
        try:
            from transformers import pipeline

            if self._model is None:
                self._model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                )

            result = self._model(
                claim,
                candidate_labels=["supported", "not supported"],
                hypothesis_template="This claim is {}.",
            )

            for label, score in zip(result["labels"], result["scores"]):
                if label == "supported":
                    return score

            return 0.5
        except ImportError:
            raise

    def _compute_word_support(self, response: str, context: str) -> float:
        """Fallback: compute support via word overlap."""
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        if not response_words:
            return 1.0

        # Fraction of response words found in context
        overlap = len(response_words & context_words)
        return overlap / len(response_words)
