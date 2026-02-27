"""Comprehensive tests for fi.evals.types module."""

import pytest
from fi.evals.types import (
    OutputType,
    RequiredKeys,
    EvalTags,
    Comparator,
    EvalResult,
    BatchRunResult,
    EvalResultMetric,
    DatapointFieldAnnotation,
    TextMetricInput,
    JsonMetricInput,
    ConfigParam,
    ConfigPossibleValues,
)


class TestOutputType:
    """Tests for OutputType enum."""

    def test_output_type_values(self):
        """Test that all expected output types exist."""
        assert OutputType.SCORE.value == "score"
        assert OutputType.BOOLEAN.value == "boolean"
        assert OutputType.JSON.value == "json"
        assert OutputType.TEXT.value == "text"

    def test_output_type_count(self):
        """Test that we have exactly 4 output types."""
        assert len(OutputType) == 4


class TestRequiredKeys:
    """Tests for RequiredKeys enum."""

    def test_required_keys_basic(self):
        """Test basic required keys."""
        assert RequiredKeys.text.value == "text"
        assert RequiredKeys.response.value == "response"
        assert RequiredKeys.query.value == "query"
        assert RequiredKeys.context.value == "context"

    def test_required_keys_expected(self):
        """Test expected response keys."""
        assert RequiredKeys.expected_response.value == "expected_response"
        assert RequiredKeys.expected_text.value == "expected_text"

    def test_required_keys_io(self):
        """Test input/output keys."""
        assert RequiredKeys.input.value == "input"
        assert RequiredKeys.output.value == "output"
        assert RequiredKeys.prompt.value == "prompt"

    def test_required_keys_images(self):
        """Test image-related keys."""
        assert RequiredKeys.image_url.value == "image_url"
        assert RequiredKeys.input_image_url.value == "input_image_url"
        assert RequiredKeys.output_image_url.value == "output_image_url"

    def test_required_keys_json(self):
        """Test JSON-related keys."""
        assert RequiredKeys.actual_json.value == "actual_json"
        assert RequiredKeys.expected_json.value == "expected_json"

    def test_required_keys_messages(self):
        """Test messages key."""
        assert RequiredKeys.messages.value == "messages"
        assert RequiredKeys.document.value == "document"


class TestEvalTags:
    """Tests for EvalTags enum."""

    def test_eval_tags_values(self):
        """Test all evaluation tags."""
        assert EvalTags.CONVERSATION.value == "CONVERSATION"
        assert EvalTags.HALLUCINATION.value == "HALLUCINATION"
        assert EvalTags.RAG.value == "RAG"
        assert EvalTags.FUTURE_EVALS.value == "FUTURE_EVALS"
        assert EvalTags.LLMS.value == "LLMS"
        assert EvalTags.CUSTOM.value == "CUSTOM"
        assert EvalTags.FUNCTION.value == "FUNCTION"
        assert EvalTags.IMAGE.value == "IMAGE"
        assert EvalTags.SAFETY.value == "SAFETY"
        assert EvalTags.TEXT.value == "TEXT"

    def test_eval_tags_count(self):
        """Test that we have exactly 10 eval tags."""
        assert len(EvalTags) == 10


class TestComparator:
    """Tests for Comparator enum."""

    def test_comparator_values(self):
        """Test all comparator values."""
        assert Comparator.COSINE.value == "CosineSimilarity"
        assert Comparator.LEVENSHTEIN.value == "NormalisedLevenshteinSimilarity"
        assert Comparator.JARO_WINKLER.value == "JaroWincklerSimilarity"
        assert Comparator.JACCARD.value == "JaccardSimilarity"
        assert Comparator.SORENSEN_DICE.value == "SorensenDiceSimilarity"

    def test_comparator_count(self):
        """Test that we have exactly 5 comparators."""
        assert len(Comparator) == 5


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_eval_result_creation(self):
        """Test creating an EvalResult."""
        result = EvalResult(
            name="test_eval",
            output="PASS",
            reason="Test passed successfully",
            runtime=1234,
            output_type="boolean",
            eval_id="eval-123"
        )
        assert result.name == "test_eval"
        assert result.output == "PASS"
        assert result.reason == "Test passed successfully"
        assert result.runtime == 1234
        assert result.output_type == "boolean"
        assert result.eval_id == "eval-123"

    def test_eval_result_defaults(self):
        """Test EvalResult with default values."""
        result = EvalResult(name="minimal_eval")
        assert result.name == "minimal_eval"
        assert result.output is None
        assert result.reason is None
        assert result.runtime == 0
        assert result.output_type is None
        assert result.eval_id is None

    def test_eval_result_with_numeric_output(self):
        """Test EvalResult with numeric output."""
        result = EvalResult(name="score_eval", output=0.85, reason="High score")
        assert result.output == 0.85

    def test_eval_result_with_dict_output(self):
        """Test EvalResult with dict output."""
        result = EvalResult(name="json_eval", output={"key": "value"})
        assert result.output == {"key": "value"}


class TestBatchRunResult:
    """Tests for BatchRunResult model."""

    def test_batch_run_result_creation(self):
        """Test creating a BatchRunResult."""
        eval1 = EvalResult(name="eval1", output="PASS")
        eval2 = EvalResult(name="eval2", output="FAIL")
        batch = BatchRunResult(eval_results=[eval1, eval2])
        assert len(batch.eval_results) == 2
        assert batch.eval_results[0].name == "eval1"
        assert batch.eval_results[1].name == "eval2"

    def test_batch_run_result_empty(self):
        """Test empty BatchRunResult."""
        batch = BatchRunResult(eval_results=[])
        assert len(batch.eval_results) == 0

    def test_batch_run_result_with_none(self):
        """Test BatchRunResult with None values."""
        batch = BatchRunResult(eval_results=[None, EvalResult(name="valid"), None])
        assert len(batch.eval_results) == 3
        assert batch.eval_results[0] is None
        assert batch.eval_results[1].name == "valid"


class TestEvalResultMetric:
    """Tests for EvalResultMetric model."""

    def test_eval_result_metric_string_id(self):
        """Test with string ID."""
        metric = EvalResultMetric(id="metric-1", value=0.95)
        assert metric.id == "metric-1"
        assert metric.value == 0.95

    def test_eval_result_metric_int_id(self):
        """Test with integer ID."""
        metric = EvalResultMetric(id=123, value="passed")
        assert metric.id == 123
        assert metric.value == "passed"

    def test_eval_result_metric_list_value(self):
        """Test with list value."""
        metric = EvalResultMetric(id="multi", value=[1, 2, 3])
        assert metric.value == [1, 2, 3]


class TestDatapointFieldAnnotation:
    """Tests for DatapointFieldAnnotation model."""

    def test_datapoint_field_annotation(self):
        """Test creating a datapoint field annotation."""
        annotation = DatapointFieldAnnotation(
            field_name="response",
            text="This is the model output",
            annotation_type="highlight",
            annotation_note="Important section"
        )
        assert annotation.field_name == "response"
        assert annotation.text == "This is the model output"
        assert annotation.annotation_type == "highlight"
        assert annotation.annotation_note == "Important section"


class TestTextMetricInput:
    """Tests for TextMetricInput model."""

    def test_text_metric_input_basic(self):
        """Test basic TextMetricInput."""
        input_data = TextMetricInput(response="Hello world")
        assert input_data.response == "Hello world"
        assert input_data.expected_response is None

    def test_text_metric_input_with_expected(self):
        """Test with expected response."""
        input_data = TextMetricInput(
            response="Generated response",
            expected_response="Expected response"
        )
        assert input_data.response == "Generated response"
        assert input_data.expected_response == "Expected response"

    def test_text_metric_input_with_list_expected(self):
        """Test with list of expected responses."""
        input_data = TextMetricInput(
            response="Result",
            expected_response=["Option 1", "Option 2", "Option 3"]
        )
        assert len(input_data.expected_response) == 3


class TestJsonMetricInput:
    """Tests for JsonMetricInput model."""

    def test_json_metric_input_string(self):
        """Test with string JSON."""
        input_data = JsonMetricInput(response='{"key": "value"}')
        assert input_data.response == '{"key": "value"}'

    def test_json_metric_input_dict(self):
        """Test with dict response."""
        input_data = JsonMetricInput(response={"key": "value"})
        assert input_data.response == {"key": "value"}

    def test_json_metric_input_with_schema(self):
        """Test with schema."""
        input_data = JsonMetricInput(
            response={"name": "test"},
            schema={"type": "object", "properties": {"name": {"type": "string"}}}
        )
        assert input_data.schema is not None


class TestConfigParam:
    """Tests for ConfigParam model."""

    def test_config_param_required(self):
        """Test required config param."""
        param = ConfigParam(type="string")
        assert param.type == "string"
        assert param.default is None

    def test_config_param_with_default(self):
        """Test config param with default."""
        param = ConfigParam(type="int", default=10)
        assert param.type == "int"
        assert param.default == 10


class TestConfigPossibleValues:
    """Tests for ConfigPossibleValues model."""

    def test_config_possible_values_length(self):
        """Test length configuration."""
        config = ConfigPossibleValues(min_length=10, max_length=100)
        assert config.min_length == 10
        assert config.max_length == 100

    def test_config_possible_values_keywords(self):
        """Test keywords configuration."""
        config = ConfigPossibleValues(
            keywords=["test", "example"],
            keyword="single",
            case_sensitive=True
        )
        assert config.keywords == ["test", "example"]
        assert config.keyword == "single"
        assert config.case_sensitive is True

    def test_config_possible_values_model(self):
        """Test model configuration."""
        config = ConfigPossibleValues(
            model="gpt-4",
            system_prompt="You are a helpful assistant"
        )
        assert config.model == "gpt-4"
        assert config.system_prompt == "You are a helpful assistant"

    def test_config_possible_values_url(self):
        """Test URL configuration."""
        config = ConfigPossibleValues(
            url="https://api.example.com",
            headers={"Authorization": "Bearer token"},
            payload={"key": "value"}
        )
        assert config.url == "https://api.example.com"
        assert config.headers == {"Authorization": "Bearer token"}

    def test_config_possible_values_grading(self):
        """Test grading configuration."""
        config = ConfigPossibleValues(
            grading_criteria="Accuracy and relevance",
            choices=["A", "B", "C", "D"],
            multi_choice=True
        )
        assert config.grading_criteria == "Accuracy and relevance"
        assert config.choices == ["A", "B", "C", "D"]
        assert config.multi_choice is True
