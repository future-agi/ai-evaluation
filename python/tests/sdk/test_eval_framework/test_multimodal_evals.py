"""Tests for fi.evals.framework.evals.multimodal module."""

import pytest
import base64
from fi.evals.framework.evals.multimodal import (
    ImageTextConsistencyEval,
    CaptionQualityEval,
    VisualQAEval,
    ImageSafetyEval,
    CrossModalConsistencyEval,
    MultiModalEvalResult,
)
from fi.evals.framework.protocols import EvalRegistry


class TestMultiModalEvalResult:
    """Tests for MultiModalEvalResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = MultiModalEvalResult(
            score=0.85,
            passed=True,
        )

        assert result.score == 0.85
        assert result.passed is True
        assert result.confidence == 1.0
        assert result.modalities == []
        assert result.details == {}

    def test_with_modalities(self):
        """Test result with modalities."""
        result = MultiModalEvalResult(
            score=0.9,
            passed=True,
            modalities=["image", "text"],
            details={"method": "embedding"},
        )

        assert result.modalities == ["image", "text"]
        assert result.details == {"method": "embedding"}


class TestImageTextConsistencyEval:
    """Tests for ImageTextConsistencyEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = ImageTextConsistencyEval()

        assert eval.name == "image_text_consistency"
        assert eval.version == "1.0.0"

    def test_default_threshold(self):
        """Test default threshold."""
        eval = ImageTextConsistencyEval()
        assert eval.threshold == 0.7

    def test_custom_threshold(self):
        """Test custom threshold."""
        eval = ImageTextConsistencyEval(threshold=0.9)
        assert eval.threshold == 0.9

    def test_required_fields(self):
        """Test required field validation."""
        eval = ImageTextConsistencyEval()
        errors = eval.validate_inputs({})

        assert "image_description" in str(errors)
        assert "text" in str(errors)

    def test_supported_modalities(self):
        """Test supported modalities."""
        eval = ImageTextConsistencyEval()
        assert "image" in eval.supported_modalities
        assert "text" in eval.supported_modalities

    def test_evaluate_identical_content(self):
        """Test with identical description and text."""
        eval = ImageTextConsistencyEval()
        result = eval.evaluate({
            "image_description": "A dog playing in the park",
            "text": "A dog playing in the park",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert result.score >= 0.8
        assert result.passed is True

    def test_evaluate_similar_content(self):
        """Test with similar content."""
        eval = ImageTextConsistencyEval()
        result = eval.evaluate({
            "image_description": "A cat sitting on a couch",
            "text": "A fluffy cat relaxing on a sofa",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert 0.0 <= result.score <= 1.0
        assert "overlap_score" in result.details

    def test_evaluate_different_content(self):
        """Test with unrelated content."""
        eval = ImageTextConsistencyEval()
        result = eval.evaluate({
            "image_description": "A mountain landscape",
            "text": "A busy city street with cars",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert result.score < 0.5

    def test_get_span_attributes(self):
        """Test span attributes generation."""
        eval = ImageTextConsistencyEval()
        result = MultiModalEvalResult(
            score=0.85,
            passed=True,
            modalities=["image", "text"],
        )

        attrs = eval.get_span_attributes(result)

        assert attrs["score"] == 0.85
        assert attrs["passed"] is True
        assert attrs["threshold"] == 0.7


class TestCaptionQualityEval:
    """Tests for CaptionQualityEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = CaptionQualityEval()

        assert eval.name == "caption_quality"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = CaptionQualityEval()
        errors = eval.validate_inputs({})

        assert "caption" in str(errors)

    def test_evaluate_good_caption(self):
        """Test with a well-formed caption."""
        eval = CaptionQualityEval()
        result = eval.evaluate({
            "caption": "A beautiful sunset over the ocean with bright orange and purple colors.",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert result.score >= 0.6
        assert "word_count" in result.details

    def test_evaluate_short_caption(self):
        """Test with a very short caption."""
        eval = CaptionQualityEval()
        result = eval.evaluate({
            "caption": "Dog.",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert result.details["length_score"] < 0.7

    def test_evaluate_with_reference(self):
        """Test with reference description."""
        eval = CaptionQualityEval()
        result = eval.evaluate({
            "caption": "A red car parked on the street.",
            "image_description": "A red sports car is parked on a city street.",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert "relevance_score" in result.details

    def test_evaluate_grammar(self):
        """Test grammar scoring."""
        eval = CaptionQualityEval()

        # Good grammar
        result_good = eval.evaluate({
            "caption": "The dog is running in the park.",
        })

        # Poor grammar (no capital, no period)
        result_poor = eval.evaluate({
            "caption": "the dog is running in the park",
        })

        assert result_good.details["grammar_score"] > result_poor.details["grammar_score"]


class TestVisualQAEval:
    """Tests for VisualQAEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = VisualQAEval()

        assert eval.name == "visual_qa"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = VisualQAEval()
        errors = eval.validate_inputs({})

        assert "question" in str(errors)
        assert "answer" in str(errors)

    def test_evaluate_color_question(self):
        """Test with color question."""
        eval = VisualQAEval()
        result = eval.evaluate({
            "question": "What color is the car?",
            "answer": "The car is red.",
            "image_description": "A red sports car on the road",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert result.details["question_type"] == "color"
        assert result.details["format_score"] == 1.0  # Contains color

    def test_evaluate_yes_no_question(self):
        """Test with yes/no question."""
        eval = VisualQAEval()
        result = eval.evaluate({
            "question": "Is there a dog in the image?",
            "answer": "Yes, there is a dog.",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert result.details["question_type"] == "yes_no"
        assert result.details["format_score"] == 1.0

    def test_evaluate_count_question(self):
        """Test with counting question."""
        eval = VisualQAEval()
        result = eval.evaluate({
            "question": "How many people are in the photo?",
            "answer": "There are three people in the photo.",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert result.details["question_type"] == "count"
        assert result.details["format_score"] == 1.0

    def test_evaluate_with_image_context(self):
        """Test with image description context."""
        eval = VisualQAEval()
        result = eval.evaluate({
            "question": "What is the person doing?",
            "answer": "The person is reading a book.",
            "image_description": "A person sitting on a bench reading a book in a park",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert result.details["consistency_score"] > 0.5


class TestImageSafetyEval:
    """Tests for ImageSafetyEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = ImageSafetyEval()

        assert eval.name == "image_safety"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = ImageSafetyEval()
        errors = eval.validate_inputs({})

        assert "image_data" in str(errors)

    def test_evaluate_valid_format(self):
        """Test with valid filename format."""
        eval = ImageSafetyEval()
        result = eval.evaluate({
            "image_data": "/path/to/image.jpg",
            "filename": "photo.jpg",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert "issues" in result.details

    def test_evaluate_invalid_format(self):
        """Test with invalid filename format."""
        eval = ImageSafetyEval()
        result = eval.evaluate({
            "image_data": "/path/to/file.exe",
            "filename": "file.exe",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert len(result.details["issues"]) > 0

    def test_evaluate_base64_jpeg(self):
        """Test with base64 encoded JPEG."""
        # Minimal JPEG header
        jpeg_header = bytes([0xFF, 0xD8, 0xFF, 0xE0])
        fake_image = jpeg_header + b'\x00' * 100
        b64_data = base64.b64encode(fake_image).decode()

        eval = ImageSafetyEval()
        result = eval.evaluate({
            "image_data": b64_data,
            "filename": "image.jpg",
        })

        assert isinstance(result, MultiModalEvalResult)
        assert result.score > 0.5

    def test_custom_max_size(self):
        """Test with custom max size."""
        eval = ImageSafetyEval(max_size=1024)
        assert eval.max_size == 1024


class TestCrossModalConsistencyEval:
    """Tests for CrossModalConsistencyEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = CrossModalConsistencyEval()

        assert eval.name == "cross_modal_consistency"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = CrossModalConsistencyEval()
        errors = eval.validate_inputs({})

        assert "source" in str(errors)
        assert "target" in str(errors)

    def test_evaluate_text_to_dict_consistent(self):
        """Test text to dict consistency with explicit key-value patterns."""
        eval = CrossModalConsistencyEval()
        result = eval.evaluate({
            "source": "name: John, age: 25",
            "target": {"name": "John", "age": "25"},
        })

        assert isinstance(result, MultiModalEvalResult)
        assert result.score >= 0.5  # At least partial match

    def test_evaluate_text_to_dict_inconsistent(self):
        """Test text to dict inconsistency with explicit key-value patterns."""
        eval = CrossModalConsistencyEval()
        result = eval.evaluate({
            "source": "name: John, age: 25",
            "target": {"name": "John", "age": "30"},
        })

        assert isinstance(result, MultiModalEvalResult)
        # Should detect age inconsistency
        assert result.score < 1.0

    def test_evaluate_text_to_text(self):
        """Test text to text consistency using word overlap."""
        eval = CrossModalConsistencyEval()
        result = eval.evaluate({
            "source": "The red car parked street sunny day",
            "target": "red car parked street day",
        })

        assert isinstance(result, MultiModalEvalResult)
        # High word overlap should give good score
        assert result.score > 0.5

    def test_evaluate_list_source(self):
        """Test with list as source."""
        eval = CrossModalConsistencyEval()
        result = eval.evaluate({
            "source": ["apple", "banana", "orange"],
            "target": "The fruits include apple and banana",
        })

        assert isinstance(result, MultiModalEvalResult)


class TestIntegrationWithFramework:
    """Tests for integration with the evaluation framework."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_use_with_evaluator(self):
        """Test using multi-modal evals with Evaluator."""
        from fi.evals.framework import Evaluator, ExecutionMode

        evaluator = Evaluator(
            evaluations=[
                ImageTextConsistencyEval(),
                CaptionQualityEval(),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "image_description": "A dog in a park",
            "text": "A happy dog playing in the park",
            "caption": "A playful dog enjoying the sunny day in the park.",
        })

        assert len(result.results) == 2
        for r in result.results:
            assert hasattr(r.value, "score")
            assert hasattr(r.value, "passed")

    def test_use_with_async_evaluator(self):
        """Test using multi-modal evals with async evaluator."""
        from fi.evals.framework import async_evaluator

        evaluator = async_evaluator(
            ImageTextConsistencyEval(),
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "image_description": "A cat on a couch",
            "text": "A cat resting on a couch",
        })

        batch = result.wait()
        assert len(batch.results) == 1
        assert batch.results[0].value.score > 0.5

        evaluator.shutdown()

    def test_span_attributes_format(self):
        """Test span attributes are OTEL-compatible."""
        eval = ImageTextConsistencyEval()
        result = MultiModalEvalResult(
            score=0.85,
            passed=True,
            confidence=0.95,
            modalities=["image", "text"],
        )

        attrs = eval.get_span_attributes(result)

        # All values should be OTEL-compatible types
        for key, value in attrs.items():
            assert isinstance(value, (str, int, float, bool))

    def test_multiple_multimodal_evals(self):
        """Test running multiple multi-modal evals together."""
        from fi.evals.framework import Evaluator, ExecutionMode

        evaluator = Evaluator(
            evaluations=[
                ImageTextConsistencyEval(),
                VisualQAEval(),
                CrossModalConsistencyEval(),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "image_description": "A red car on the street",
            "text": "A red vehicle parked on the road",
            "question": "What color is the car?",
            "answer": "The car is red.",
            "source": "The car is red and parked.",
            "target": "A red car is parked.",
        })

        assert len(result.results) == 3
        assert result.success_rate == 1.0
