"""Multi-modal evaluation implementations.

This module provides evaluations for multi-modal content including:
- Image quality and relevance
- Visual question answering
- Image-text consistency
- Caption quality

All evaluations use local heuristics by default with optional
external model support for enhanced accuracy.

Example:
    from fi.evals.framework import Evaluator, ExecutionMode
    from fi.evals.framework.evals.multimodal import (
        ImageTextConsistencyEval,
        CaptionQualityEval,
    )

    evaluator = Evaluator(
        evaluations=[
            ImageTextConsistencyEval(),
            CaptionQualityEval(),
        ],
        mode=ExecutionMode.BLOCKING,
    )

    result = evaluator.run({
        "image_description": "A cat sitting on a couch",
        "caption": "A fluffy orange cat relaxing on a sofa",
    })
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import re
import base64
import hashlib

from ..protocols import BaseEvaluation, register_evaluation


@dataclass
class MultiModalEvalResult:
    """Result from a multi-modal evaluation.

    Attributes:
        score: Evaluation score between 0 and 1
        passed: Whether the evaluation passed threshold
        confidence: Confidence in the result (0-1)
        modalities: List of modalities evaluated
        details: Additional evaluation details
    """

    score: float
    passed: bool
    confidence: float = 1.0
    modalities: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class BaseMultiModalEval(BaseEvaluation, ABC):
    """Base class for multi-modal evaluations.

    Provides common functionality for evaluations that handle
    multiple modalities (text, image, audio, video).
    """

    version = "1.0.0"

    def __init__(self, threshold: float = 0.7):
        """Initialize evaluation.

        Args:
            threshold: Score threshold for passing (default 0.7)
        """
        self.threshold = threshold

    @property
    @abstractmethod
    def required_fields(self) -> List[str]:
        """Return required input fields."""
        pass

    @property
    @abstractmethod
    def supported_modalities(self) -> List[str]:
        """Return supported modalities."""
        pass

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate required inputs are present.

        Args:
            inputs: Input dictionary

        Returns:
            List of validation error messages
        """
        errors = []
        for field in self.required_fields:
            if field not in inputs:
                errors.append(f"Missing required field: {field}")
        return errors

    def get_span_attributes(self, result: MultiModalEvalResult) -> Dict[str, Any]:
        """Get span attributes for tracing.

        Args:
            result: Evaluation result

        Returns:
            Dictionary of span attributes
        """
        return {
            "score": result.score,
            "passed": result.passed,
            "confidence": result.confidence,
            "threshold": self.threshold,
            "modalities": ",".join(result.modalities),
        }


@register_evaluation
class ImageTextConsistencyEval(BaseMultiModalEval):
    """Evaluate consistency between image description and text.

    This evaluation checks whether a text (caption, description, etc.)
    is consistent with an image description. Uses word overlap and
    semantic similarity heuristics.

    Required inputs:
        - image_description: Description of the image content
        - text: Text to check for consistency (caption, response, etc.)

    Example:
        eval = ImageTextConsistencyEval()
        result = eval.evaluate({
            "image_description": "A dog playing in a park",
            "text": "The happy puppy is running in the grass",
        })
    """

    name = "image_text_consistency"

    @property
    def required_fields(self) -> List[str]:
        return ["image_description", "text"]

    @property
    def supported_modalities(self) -> List[str]:
        return ["image", "text"]

    def evaluate(self, inputs: Dict[str, Any]) -> MultiModalEvalResult:
        """Evaluate image-text consistency.

        Args:
            inputs: Dictionary with 'image_description' and 'text'

        Returns:
            MultiModalEvalResult with consistency score
        """
        image_desc = inputs["image_description"].lower()
        text = inputs["text"].lower()

        # Tokenize
        desc_words = set(re.findall(r'\b\w+\b', image_desc))
        text_words = set(re.findall(r'\b\w+\b', text))

        # Remove common stopwords
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of'}
        desc_words -= stopwords
        text_words -= stopwords

        if not desc_words or not text_words:
            return MultiModalEvalResult(
                score=0.5,
                passed=False,
                confidence=0.5,
                modalities=self.supported_modalities,
                details={"reason": "insufficient_content"},
            )

        # Calculate overlap (Jaccard similarity)
        intersection = desc_words & text_words
        union = desc_words | text_words
        overlap = len(intersection) / len(union) if union else 0

        # Check for key concept overlap (nouns typically)
        # Using a simple heuristic: longer words are more likely to be key concepts
        key_desc_words = {w for w in desc_words if len(w) > 3}
        key_text_words = {w for w in text_words if len(w) > 3}
        key_overlap = len(key_desc_words & key_text_words) / len(key_desc_words) if key_desc_words else 0

        # Combined score
        score = 0.6 * key_overlap + 0.4 * overlap

        return MultiModalEvalResult(
            score=score,
            passed=score >= self.threshold,
            confidence=0.7,
            modalities=self.supported_modalities,
            details={
                "overlap_score": overlap,
                "key_concept_overlap": key_overlap,
                "matched_words": list(intersection),
            },
        )


@register_evaluation
class CaptionQualityEval(BaseMultiModalEval):
    """Evaluate the quality of image captions.

    Assesses caption quality based on:
    - Descriptiveness (sufficient detail)
    - Grammar/fluency (basic checks)
    - Informativeness (unique information)

    Required inputs:
        - caption: The caption to evaluate

    Optional inputs:
        - image_description: Reference description for comparison

    Example:
        eval = CaptionQualityEval()
        result = eval.evaluate({
            "caption": "A beautiful sunset over the ocean with orange and purple colors",
        })
    """

    name = "caption_quality"

    @property
    def required_fields(self) -> List[str]:
        return ["caption"]

    @property
    def supported_modalities(self) -> List[str]:
        return ["text"]

    def evaluate(self, inputs: Dict[str, Any]) -> MultiModalEvalResult:
        """Evaluate caption quality.

        Args:
            inputs: Dictionary with 'caption' and optional 'image_description'

        Returns:
            MultiModalEvalResult with quality score
        """
        caption = inputs["caption"]
        reference = inputs.get("image_description", "")

        # Length score (penalize too short or too long)
        words = caption.split()
        word_count = len(words)

        if word_count < 3:
            length_score = 0.3
        elif word_count < 5:
            length_score = 0.6
        elif word_count <= 20:
            length_score = 1.0
        elif word_count <= 30:
            length_score = 0.8
        else:
            length_score = 0.6

        # Descriptiveness score (presence of adjectives, details)
        descriptive_patterns = [
            r'\b(beautiful|bright|dark|large|small|old|new|colorful|peaceful)\b',
            r'\b(sitting|standing|walking|running|playing|looking)\b',
            r'\b(on|in|under|above|near|beside|between)\b',
        ]
        descriptive_count = sum(
            1 for pattern in descriptive_patterns
            if re.search(pattern, caption.lower())
        )
        descriptive_score = min(1.0, descriptive_count / 2)

        # Basic grammar check (starts with capital, ends with punctuation)
        grammar_score = 1.0
        if not caption[0].isupper():
            grammar_score -= 0.2
        if not caption.rstrip()[-1] in '.!?':
            grammar_score -= 0.1

        # Calculate relevance if reference provided
        if reference:
            ref_words = set(re.findall(r'\b\w+\b', reference.lower()))
            cap_words = set(re.findall(r'\b\w+\b', caption.lower()))
            stopwords = {'a', 'an', 'the', 'is', 'are', 'in', 'on', 'of'}
            ref_words -= stopwords
            cap_words -= stopwords
            relevance_score = len(ref_words & cap_words) / len(ref_words) if ref_words else 0.5
        else:
            relevance_score = 0.7  # Default if no reference

        # Combined score
        score = (
            0.25 * length_score +
            0.25 * descriptive_score +
            0.2 * grammar_score +
            0.3 * relevance_score
        )

        return MultiModalEvalResult(
            score=score,
            passed=score >= self.threshold,
            confidence=0.75,
            modalities=self.supported_modalities,
            details={
                "word_count": word_count,
                "length_score": length_score,
                "descriptive_score": descriptive_score,
                "grammar_score": grammar_score,
                "relevance_score": relevance_score,
            },
        )


@register_evaluation
class VisualQAEval(BaseMultiModalEval):
    """Evaluate visual question answering responses.

    Assesses whether an answer to a visual question is:
    - Relevant to the question
    - Consistent with image context
    - Sufficiently detailed

    Required inputs:
        - question: The visual question asked
        - answer: The response to evaluate

    Optional inputs:
        - image_description: Description of the image for context

    Example:
        eval = VisualQAEval()
        result = eval.evaluate({
            "question": "What color is the car in the image?",
            "answer": "The car is red.",
            "image_description": "A red sports car parked on the street",
        })
    """

    name = "visual_qa"

    @property
    def required_fields(self) -> List[str]:
        return ["question", "answer"]

    @property
    def supported_modalities(self) -> List[str]:
        return ["image", "text"]

    def evaluate(self, inputs: Dict[str, Any]) -> MultiModalEvalResult:
        """Evaluate visual QA response.

        Args:
            inputs: Dictionary with 'question', 'answer', and optional 'image_description'

        Returns:
            MultiModalEvalResult with VQA score
        """
        question = inputs["question"].lower()
        answer = inputs["answer"].lower()
        image_desc = inputs.get("image_description", "").lower()

        # Question type detection
        question_type = "unknown"
        if question.startswith(("what color", "what colour")):
            question_type = "color"
        elif question.startswith("how many"):
            question_type = "count"
        elif question.startswith("where"):
            question_type = "location"
        elif question.startswith("what is") or question.startswith("what are"):
            question_type = "identification"
        elif question.startswith(("is there", "are there", "is it", "is the")):
            question_type = "yes_no"

        # Relevance: answer should relate to question topic
        question_words = set(re.findall(r'\b\w+\b', question))
        answer_words = set(re.findall(r'\b\w+\b', answer))
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of', 'what', 'where', 'how', 'many'}
        question_words -= stopwords
        answer_words -= stopwords

        # Check for question topic coverage
        topic_words = {w for w in question_words if len(w) > 2}
        topic_coverage = len(topic_words & answer_words) / len(topic_words) if topic_words else 0

        # Answer format appropriateness
        format_score = 0.7  # Default
        if question_type == "yes_no":
            if re.search(r'\b(yes|no|not)\b', answer):
                format_score = 1.0
        elif question_type == "count":
            if re.search(r'\b(\d+|one|two|three|four|five|several|many|none|no)\b', answer):
                format_score = 1.0
        elif question_type == "color":
            colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'brown', 'gray', 'grey']
            if any(c in answer for c in colors):
                format_score = 1.0

        # Consistency with image description if provided
        consistency_score = 0.7  # Default
        if image_desc:
            desc_words = set(re.findall(r'\b\w+\b', image_desc)) - stopwords
            consistency = len(desc_words & answer_words) / len(answer_words) if answer_words else 0
            consistency_score = 0.5 + 0.5 * consistency

        # Answer length (should be appropriate)
        answer_word_count = len(answer.split())
        if answer_word_count < 1:
            length_score = 0.0
        elif answer_word_count <= 3:
            length_score = 0.8 if question_type == "yes_no" else 0.6
        elif answer_word_count <= 15:
            length_score = 1.0
        else:
            length_score = 0.7

        # Combined score
        score = (
            0.25 * topic_coverage +
            0.25 * format_score +
            0.25 * consistency_score +
            0.25 * length_score
        )

        return MultiModalEvalResult(
            score=score,
            passed=score >= self.threshold,
            confidence=0.7,
            modalities=self.supported_modalities,
            details={
                "question_type": question_type,
                "topic_coverage": topic_coverage,
                "format_score": format_score,
                "consistency_score": consistency_score,
                "length_score": length_score,
            },
        )


@register_evaluation
class ImageSafetyEval(BaseMultiModalEval):
    """Evaluate image safety based on metadata and heuristics.

    Checks for potential safety issues in images:
    - File format validation
    - Size constraints
    - Metadata analysis (when available)

    Note: This is a lightweight check. For comprehensive NSFW detection,
    use the guardrails module or external APIs.

    Required inputs:
        - image_data: Base64 encoded image OR file path OR URL

    Optional inputs:
        - filename: Original filename for format checking

    Example:
        eval = ImageSafetyEval()
        result = eval.evaluate({
            "image_data": "base64encodedstring...",
            "filename": "photo.jpg",
        })
    """

    name = "image_safety"

    # Maximum file size (10MB)
    MAX_SIZE = 10 * 1024 * 1024

    # Allowed formats
    ALLOWED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

    def __init__(self, threshold: float = 0.7, max_size: int = None):
        """Initialize evaluation.

        Args:
            threshold: Score threshold for passing
            max_size: Maximum allowed file size in bytes
        """
        super().__init__(threshold)
        self.max_size = max_size or self.MAX_SIZE

    @property
    def required_fields(self) -> List[str]:
        return ["image_data"]

    @property
    def supported_modalities(self) -> List[str]:
        return ["image"]

    def evaluate(self, inputs: Dict[str, Any]) -> MultiModalEvalResult:
        """Evaluate image safety.

        Args:
            inputs: Dictionary with 'image_data' and optional 'filename'

        Returns:
            MultiModalEvalResult with safety score
        """
        image_data = inputs["image_data"]
        filename = inputs.get("filename", "")

        issues = []
        checks_passed = 0
        total_checks = 0

        # Format check
        total_checks += 1
        if filename:
            ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
            if ext in self.ALLOWED_FORMATS:
                checks_passed += 1
            else:
                issues.append(f"Unsupported format: {ext}")
        else:
            checks_passed += 0.5  # Partial credit if no filename

        # Size check (if base64 encoded)
        total_checks += 1
        if image_data.startswith('data:image'):
            # Data URL format
            try:
                header, data = image_data.split(',', 1)
                decoded_size = len(base64.b64decode(data))
                if decoded_size <= self.max_size:
                    checks_passed += 1
                else:
                    issues.append(f"Image too large: {decoded_size} bytes")
            except Exception:
                issues.append("Could not decode base64 data")
        elif len(image_data) < 100:
            # Likely a path or URL
            checks_passed += 1  # Assume OK for paths
        else:
            # Raw base64
            try:
                decoded_size = len(base64.b64decode(image_data))
                if decoded_size <= self.max_size:
                    checks_passed += 1
                else:
                    issues.append(f"Image too large: {decoded_size} bytes")
            except Exception:
                checks_passed += 0.5  # Might not be base64

        # Magic bytes check (basic format validation)
        total_checks += 1
        try:
            if image_data.startswith('data:image'):
                header, data = image_data.split(',', 1)
                raw = base64.b64decode(data[:100])
            elif len(image_data) > 100:
                raw = base64.b64decode(image_data[:100])
            else:
                raw = None

            if raw:
                # Check for common image magic bytes
                if (raw[:3] == b'\xff\xd8\xff' or  # JPEG
                    raw[:8] == b'\x89PNG\r\n\x1a\n' or  # PNG
                    raw[:6] in (b'GIF87a', b'GIF89a') or  # GIF
                    raw[:4] == b'RIFF'):  # WEBP
                    checks_passed += 1
                else:
                    issues.append("Unknown image format")
            else:
                checks_passed += 0.5  # Can't verify
        except Exception:
            checks_passed += 0.5  # Assume OK if can't check

        # Calculate score
        score = checks_passed / total_checks if total_checks > 0 else 0

        return MultiModalEvalResult(
            score=score,
            passed=score >= self.threshold and len(issues) == 0,
            confidence=0.8,
            modalities=self.supported_modalities,
            details={
                "issues": issues,
                "checks_passed": checks_passed,
                "total_checks": total_checks,
            },
        )


@register_evaluation
class CrossModalConsistencyEval(BaseMultiModalEval):
    """Evaluate consistency across multiple modalities.

    Checks that information is consistent across different representations:
    - Text description vs. structured data
    - Multiple text sources
    - Caption vs. detailed description

    Required inputs:
        - source: Primary source content
        - target: Content to check for consistency

    Optional inputs:
        - source_modality: Type of source ('text', 'structured')
        - target_modality: Type of target

    Example:
        eval = CrossModalConsistencyEval()
        result = eval.evaluate({
            "source": "John is 25 years old and lives in New York",
            "target": {"name": "John", "age": 25, "city": "New York"},
        })
    """

    name = "cross_modal_consistency"

    @property
    def required_fields(self) -> List[str]:
        return ["source", "target"]

    @property
    def supported_modalities(self) -> List[str]:
        return ["text", "structured"]

    def evaluate(self, inputs: Dict[str, Any]) -> MultiModalEvalResult:
        """Evaluate cross-modal consistency.

        Args:
            inputs: Dictionary with 'source' and 'target'

        Returns:
            MultiModalEvalResult with consistency score
        """
        source = inputs["source"]
        target = inputs["target"]

        # Convert both to comparable text form
        source_text = self._to_text(source)
        target_text = self._to_text(target)

        # Extract key-value pairs from text
        source_facts = self._extract_facts(source_text)
        target_facts = self._extract_facts(target_text)

        # Check consistency
        consistent_facts = 0
        total_facts = 0
        inconsistencies = []

        # Compare overlapping keys
        common_keys = set(source_facts.keys()) & set(target_facts.keys())
        for key in common_keys:
            total_facts += 1
            if source_facts[key].lower() == target_facts[key].lower():
                consistent_facts += 1
            else:
                inconsistencies.append({
                    "key": key,
                    "source_value": source_facts[key],
                    "target_value": target_facts[key],
                })

        # If no overlapping facts found, use word overlap
        if total_facts == 0:
            source_words = set(re.findall(r'\b\w+\b', source_text.lower()))
            target_words = set(re.findall(r'\b\w+\b', target_text.lower()))
            stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on'}
            source_words -= stopwords
            target_words -= stopwords

            if source_words and target_words:
                overlap = len(source_words & target_words)
                score = overlap / max(len(source_words), len(target_words))
            else:
                score = 0.5
        else:
            score = consistent_facts / total_facts

        return MultiModalEvalResult(
            score=score,
            passed=score >= self.threshold,
            confidence=0.75,
            modalities=self.supported_modalities,
            details={
                "consistent_facts": consistent_facts,
                "total_facts": total_facts,
                "inconsistencies": inconsistencies,
                "source_facts": source_facts,
                "target_facts": target_facts,
            },
        )

    def _to_text(self, content: Any) -> str:
        """Convert content to text representation."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            parts = []
            for k, v in content.items():
                parts.append(f"{k}: {v}")
            return ", ".join(parts)
        elif isinstance(content, (list, tuple)):
            return ", ".join(str(item) for item in content)
        else:
            return str(content)

    def _extract_facts(self, text: str) -> Dict[str, str]:
        """Extract key-value facts from text."""
        facts = {}

        # Pattern: "X is Y" or "X: Y"
        patterns = [
            r'(\w+)\s+is\s+(\w+)',
            r'(\w+):\s*(\w+)',
            r'(\w+)\s*=\s*(\w+)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                key = match.group(1).lower()
                value = match.group(2)
                facts[key] = value

        # Extract numbers with context
        number_pattern = r'(\w+)\s+(\d+)'
        for match in re.finditer(number_pattern, text):
            key = match.group(1).lower()
            value = match.group(2)
            if key not in facts:
                facts[key] = value

        return facts
