"""Comprehensive tests for fi.evals.evaluator module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fi.evals.evaluator import (
    Evaluator,
    EvalResponseHandler,
    EvalInfoResponseHandler,
    list_evaluations,
)
from fi.evals.types import BatchRunResult, EvalResult
from fi.evals.templates import Groundedness, Toxicity
from fi.utils.errors import InvalidAuthError


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Set mock API keys for testing."""
    monkeypatch.setenv("FI_API_KEY", "test_api_key")
    monkeypatch.setenv("FI_SECRET_KEY", "test_secret_key")


@pytest.fixture
def evaluator(mock_api_keys):
    """Create an evaluator instance with mock keys."""
    return Evaluator()


class TestEvaluatorInit:
    """Tests for Evaluator initialization."""

    def test_init_with_env_vars(self, mock_api_keys):
        """Test initialization with environment variables."""
        evaluator = Evaluator()
        assert evaluator is not None

    def test_init_with_explicit_keys(self):
        """Test initialization with explicit API keys."""
        evaluator = Evaluator(
            fi_api_key="explicit_api_key",
            fi_secret_key="explicit_secret_key"
        )
        assert evaluator is not None

    def test_init_with_custom_base_url(self, mock_api_keys):
        """Test initialization with custom base URL."""
        evaluator = Evaluator(fi_base_url="https://custom.api.com")
        assert evaluator is not None

    def test_init_with_max_workers(self, mock_api_keys):
        """Test initialization with custom max_workers."""
        evaluator = Evaluator(max_workers=16)
        assert evaluator._max_workers == 16

    def test_init_default_max_workers(self, mock_api_keys):
        """Test default max_workers value."""
        evaluator = Evaluator()
        assert evaluator._max_workers == 8

    def test_init_with_timeout(self, mock_api_keys):
        """Test initialization with custom timeout."""
        evaluator = Evaluator(timeout=300)
        assert evaluator._default_timeout == 300

    def test_init_with_langfuse_credentials(self, mock_api_keys):
        """Test initialization with Langfuse credentials."""
        evaluator = Evaluator(
            langfuse_secret_key="langfuse_secret",
            langfuse_public_key="langfuse_public",
            langfuse_host="https://langfuse.example.com"
        )
        assert evaluator.langfuse_secret_key == "langfuse_secret"
        assert evaluator.langfuse_public_key == "langfuse_public"
        assert evaluator.langfuse_host == "https://langfuse.example.com"


class TestEvaluate:
    """Tests for Evaluator.evaluate method."""

    @patch.object(Evaluator, 'request')
    def test_evaluate_with_string_template(self, mock_request, evaluator):
        """Test evaluate with string template name."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": [{
                "evaluations": [{
                    "name": "groundedness",
                    "output": "GROUNDED",
                    "reason": "The response is grounded in context",
                    "runtime": 1500,
                    "outputType": "boolean",
                    "evalId": "eval-123"
                }]
            }]
        }
        mock_request.return_value = BatchRunResult(eval_results=[
            EvalResult(
                name="groundedness",
                output="GROUNDED",
                reason="The response is grounded in context",
                runtime=1500
            )
        ])

        result = evaluator.evaluate(
            eval_templates="groundedness",
            inputs={"context": "Test context", "output": "Test output"},
            model_name="turing_flash"
        )

        assert result is not None
        assert isinstance(result, BatchRunResult)

    @patch.object(Evaluator, 'request')
    def test_evaluate_with_template_class(self, mock_request, evaluator):
        """Test evaluate with template class."""
        mock_request.return_value = BatchRunResult(eval_results=[
            EvalResult(name="groundedness", output="GROUNDED")
        ])

        result = evaluator.evaluate(
            eval_templates=Groundedness,
            inputs={"context": "Test context", "output": "Test output"},
            model_name="turing_flash"
        )

        assert result is not None

    @patch.object(Evaluator, 'request')
    def test_evaluate_with_template_instance(self, mock_request, evaluator):
        """Test evaluate with template instance."""
        mock_request.return_value = BatchRunResult(eval_results=[
            EvalResult(name="toxicity", output="SAFE")
        ])

        result = evaluator.evaluate(
            eval_templates=Toxicity(),
            inputs={"text": "This is a safe message"},
            model_name="protect_flash"
        )

        assert result is not None

    def test_evaluate_invalid_template_type(self, evaluator):
        """Test evaluate with invalid template type."""
        with pytest.raises(TypeError):
            evaluator.evaluate(
                eval_templates=12345,  # Invalid type
                inputs={"text": "test"},
                model_name="turing_flash"
            )

    @patch.object(Evaluator, 'request')
    def test_evaluate_with_async_flag(self, mock_request, evaluator):
        """Test evaluate with is_async flag."""
        mock_request.return_value = BatchRunResult(eval_results=[])

        result = evaluator.evaluate(
            eval_templates="groundedness",
            inputs={"context": "test", "output": "test"},
            model_name="turing_flash",
            is_async=True
        )

        assert result is not None

    @patch.object(Evaluator, 'request')
    def test_evaluate_with_error_localizer(self, mock_request, evaluator):
        """Test evaluate with error_localizer flag."""
        mock_request.return_value = BatchRunResult(eval_results=[])

        result = evaluator.evaluate(
            eval_templates="groundedness",
            inputs={"context": "test", "output": "test"},
            model_name="turing_flash",
            error_localizer=True
        )

        assert result is not None


class TestListEvaluations:
    """Tests for Evaluator.list_evaluations method."""

    @patch.object(Evaluator, 'request')
    def test_list_evaluations(self, mock_request, evaluator):
        """Test listing available evaluations."""
        mock_request.return_value = [
            {"name": "groundedness", "eval_id": "1"},
            {"name": "toxicity", "eval_id": "2"}
        ]

        result = evaluator.list_evaluations()

        assert isinstance(result, list)
        assert len(result) == 2


class TestGetEvalResult:
    """Tests for Evaluator.get_eval_result method."""

    @patch.object(Evaluator, 'request')
    def test_get_eval_result(self, mock_request, evaluator):
        """Test getting evaluation result by ID."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "eval_id": "eval-123",
            "status": "completed",
            "result": {"output": "PASS"}
        }
        mock_request.return_value = mock_response

        result = evaluator.get_eval_result("eval-123")

        assert result is not None


class TestEvaluatePipeline:
    """Tests for Evaluator.evaluate_pipeline method."""

    @patch.object(Evaluator, 'request')
    def test_evaluate_pipeline(self, mock_request, evaluator):
        """Test pipeline evaluation."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "submitted"}
        mock_request.return_value = mock_response

        result = evaluator.evaluate_pipeline(
            project_name="test_project",
            version="v1.0",
            eval_data=[{"input": "test", "output": "result"}]
        )

        assert result is not None


class TestGetPipelineResults:
    """Tests for Evaluator.get_pipeline_results method."""

    @patch.object(Evaluator, 'request')
    def test_get_pipeline_results(self, mock_request, evaluator):
        """Test getting pipeline results."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [{"version": "v1.0", "status": "completed"}]
        }
        mock_request.return_value = mock_response

        result = evaluator.get_pipeline_results(
            project_name="test_project",
            versions=["v1.0", "v1.1"]
        )

        assert result is not None

    def test_get_pipeline_results_invalid_versions(self, evaluator):
        """Test get_pipeline_results with invalid versions type."""
        with pytest.raises(TypeError):
            evaluator.get_pipeline_results(
                project_name="test_project",
                versions="v1.0"  # Should be a list
            )

    def test_get_pipeline_results_invalid_version_items(self, evaluator):
        """Test get_pipeline_results with non-string version items."""
        with pytest.raises(TypeError):
            evaluator.get_pipeline_results(
                project_name="test_project",
                versions=[1, 2, 3]  # Should be strings
            )


class TestEvalResponseHandler:
    """Tests for EvalResponseHandler."""

    def test_parse_success(self):
        """Test parsing successful response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": [{
                "evaluations": [{
                    "name": "test_eval",
                    "output": "PASS",
                    "reason": "Test passed",
                    "runtime": 100,
                    "outputType": "boolean",
                    "evalId": "eval-123"
                }]
            }]
        }

        result = EvalResponseHandler._parse_success(mock_response)

        assert isinstance(result, BatchRunResult)
        assert len(result.eval_results) == 1
        assert result.eval_results[0].name == "test_eval"
        assert result.eval_results[0].output == "PASS"

    def test_parse_success_with_metadata(self):
        """Test parsing response with metadata."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": [{
                "evaluations": [{
                    "name": "test_eval",
                    "output": 0.95,
                    "reason": "High score",
                    "runtime": 200,
                    "metadata": '{"usage": {"tokens": 100}, "cost": {"usd": 0.01}}'
                }]
            }]
        }

        result = EvalResponseHandler._parse_success(mock_response)

        assert result.eval_results[0].output == 0.95

    def test_parse_success_with_dict_metadata(self):
        """Test parsing response with dict metadata."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": [{
                "evaluations": [{
                    "name": "test_eval",
                    "output": "PASS",
                    "reason": "Test passed",
                    "runtime": 100,
                    "metadata": {"usage": {"tokens": 100}, "cost": {"usd": 0.01}}
                }]
            }]
        }

        result = EvalResponseHandler._parse_success(mock_response)

        assert len(result.eval_results) == 1

    def test_handle_error_400(self):
        """Test handling 400 error."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with pytest.raises(Exception, match="400 Bad Request"):
            EvalResponseHandler._handle_error(mock_response)

    def test_handle_error_403(self):
        """Test handling 403 error."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        with pytest.raises(InvalidAuthError):
            EvalResponseHandler._handle_error(mock_response)

    def test_handle_error_500(self):
        """Test handling 500 error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with pytest.raises(Exception, match="500"):
            EvalResponseHandler._handle_error(mock_response)


class TestEvalInfoResponseHandler:
    """Tests for EvalInfoResponseHandler."""

    def test_parse_success(self):
        """Test parsing successful info response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": [
                {"name": "groundedness", "eval_id": "1"},
                {"name": "toxicity", "eval_id": "2"}
            ]
        }

        result = EvalInfoResponseHandler._parse_success(mock_response)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_success_no_result(self):
        """Test parsing response without result key."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "No result"}

        with pytest.raises(Exception, match="Failed to get evaluation info"):
            EvalInfoResponseHandler._parse_success(mock_response)

    def test_handle_error_403(self):
        """Test handling 403 error."""
        mock_response = Mock()
        mock_response.status_code = 403

        with pytest.raises(InvalidAuthError):
            EvalInfoResponseHandler._handle_error(mock_response)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch('fi.evals.evaluator.Evaluator')
    def test_list_evaluations_convenience(self, MockEvaluator):
        """Test list_evaluations convenience function."""
        mock_evaluator = Mock()
        mock_evaluator.list_evaluations.return_value = []
        MockEvaluator.return_value = mock_evaluator

        result = list_evaluations()

        mock_evaluator.list_evaluations.assert_called_once()


class TestGetEvalInfo:
    """Tests for _get_eval_info method."""

    @patch.object(Evaluator, 'request')
    def test_get_eval_info_found(self, mock_request, evaluator):
        """Test getting eval info for existing template."""
        mock_request.return_value = [
            {"name": "groundedness", "eval_id": "47", "description": "Check grounding"},
            {"name": "toxicity", "eval_id": "15", "description": "Check toxicity"}
        ]

        result = evaluator._get_eval_info("groundedness")

        assert result["name"] == "groundedness"
        assert result["eval_id"] == "47"

    @patch.object(Evaluator, 'request')
    def test_get_eval_info_not_found(self, mock_request, evaluator):
        """Test getting eval info for non-existent template."""
        mock_request.return_value = [
            {"name": "groundedness", "eval_id": "47"}
        ]

        with pytest.raises(KeyError, match="not found"):
            evaluator._get_eval_info("nonexistent_template")

    @patch.object(Evaluator, 'request')
    def test_get_eval_info_cached(self, mock_request, evaluator):
        """Test that eval info is cached."""
        mock_request.return_value = [
            {"name": "groundedness", "eval_id": "47"}
        ]

        # Call twice
        evaluator._get_eval_info("groundedness")
        evaluator._get_eval_info("groundedness")

        # Should only make one request due to caching
        assert mock_request.call_count == 1


class TestConfigureEvaluations:
    """Tests for _configure_evaluations method."""

    def test_configure_evaluations_requires_platform(self, evaluator):
        """Test that platform configuration requires specific arguments."""
        with pytest.raises(ValueError, match="Invalid arguments"):
            evaluator.evaluate(
                eval_templates="groundedness",
                inputs={"context": "test"},
                platform="langfuse"
                # Missing custom_eval_name
            )


class TestTraceEvaluation:
    """Tests for trace evaluation functionality."""

    @patch.object(Evaluator, 'request')
    def test_trace_eval_without_custom_name(self, mock_request, evaluator, caplog):
        """Test that trace_eval requires custom_eval_name."""
        mock_request.return_value = BatchRunResult(eval_results=[])

        import logging
        with caplog.at_level(logging.WARNING):
            evaluator.evaluate(
                eval_templates="groundedness",
                inputs={"context": "test", "output": "test"},
                model_name="turing_flash",
                trace_eval=True
                # Missing custom_eval_name
            )

        # Should log a warning
        assert "custom_eval_name" in caplog.text or mock_request.called
