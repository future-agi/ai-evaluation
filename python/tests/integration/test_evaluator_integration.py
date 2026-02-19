"""
Integration tests for the Evaluator class against real backend.

These tests require the backend to be running. See conftest.py for setup instructions.

Run with:
    pytest tests/integration/ -v -m integration

NOTE: Some tests require the model serving service to be running, which is not
available in local test environments. Tests that require model inference are
marked with @pytest.mark.requires_model_serving and will be skipped if the
backend returns "Unable to run standalone evaluation".
"""

import pytest


@pytest.mark.integration
class TestEvaluatorConnection:
    """Test basic connectivity to backend."""

    def test_list_evaluations(self, evaluator):
        """Test listing available evaluation templates."""
        templates = evaluator.list_evaluations()

        assert isinstance(templates, list)
        assert len(templates) > 0

        # Check template structure
        template = templates[0]
        assert "name" in template
        assert "description" in template or "eval_tags" in template

    def test_evaluator_initialization(self, evaluator, backend_url):
        """Test that evaluator is properly configured."""
        assert evaluator is not None
        # Check base URL is set (implementation dependent)


@pytest.mark.integration
class TestAPIRequestValidation:
    """Test that API requests are properly validated."""

    def test_invalid_template_returns_error(self, evaluator):
        """Test that invalid template name returns error (doesn't raise)."""
        result = evaluator.evaluate(
            eval_templates="nonexistent_template_xyz",
            inputs={"text": "test"},
            model_name="turing_flash",
        )
        # SDK logs errors but returns empty results instead of raising
        assert result is not None
        assert len(result.eval_results) == 0

    def test_missing_required_inputs_returns_error(self, evaluator):
        """Test that missing required inputs returns error (doesn't raise)."""
        result = evaluator.evaluate(
            eval_templates="groundedness",
            inputs={"response": "test"},  # Missing 'context'
            model_name="turing_flash",
        )
        # SDK logs errors but returns empty results instead of raising
        assert result is not None
        assert len(result.eval_results) == 0


@pytest.mark.integration
class TestAsyncEvaluation:
    """Test async evaluation mode."""

    def test_async_evaluation_returns_result(self, evaluator):
        """Test running evaluation in async mode returns a result object."""
        result = evaluator.evaluate(
            eval_templates="groundedness",
            inputs={
                "context": "Python is a programming language.",
                "response": "Python is a programming language.",
            },
            model_name="turing_flash",
            is_async=True,
        )

        # Async mode should return a result object (even if evaluation fails)
        assert result is not None


# Tests below require model serving service (not available in local test env)
# They demonstrate the expected API usage when full infrastructure is available


@pytest.mark.integration
@pytest.mark.requires_model_serving
class TestGroundednessEvaluation:
    """Test groundedness evaluation.

    NOTE: These tests require the model serving service to be running.
    In local test environments, these will be skipped.
    """

    def test_groundedness_grounded_response(self, evaluator):
        """Test groundedness with a grounded response."""
        result = evaluator.evaluate(
            eval_templates="groundedness",
            inputs={
                "context": "The Eiffel Tower is 324 meters tall and located in Paris, France.",
                "response": "The Eiffel Tower is 324 meters tall.",
            },
            model_name="turing_flash",
        )

        assert result is not None
        assert len(result.eval_results) > 0

        eval_result = result.eval_results[0]
        assert eval_result.name == "groundedness"
        assert eval_result.output is not None

    def test_groundedness_ungrounded_response(self, evaluator):
        """Test groundedness with an ungrounded response."""
        result = evaluator.evaluate(
            eval_templates="groundedness",
            inputs={
                "context": "The Eiffel Tower is 324 meters tall.",
                "response": "The Eiffel Tower is 500 meters tall and made of gold.",
            },
            model_name="turing_flash",
        )

        assert result is not None
        assert len(result.eval_results) > 0


@pytest.mark.integration
@pytest.mark.requires_model_serving
class TestSafetyEvaluations:
    """Test safety-related evaluations.

    NOTE: These tests require the model serving service to be running.
    """

    def test_toxicity_safe_text(self, evaluator):
        """Test toxicity detection with safe text."""
        result = evaluator.evaluate(
            eval_templates="toxicity",
            inputs={
                "text": "Hello, how can I help you today?",
            },
            model_name="turing_flash",
        )

        assert result is not None
        assert len(result.eval_results) > 0

    @pytest.mark.skip(reason="PII detection uses external HuggingFace endpoint that may be paused")
    def test_pii_detection(self, evaluator):
        """Test PII detection.

        NOTE: This test relies on an external HuggingFace endpoint that may be
        temporarily unavailable. Skip if endpoint returns 'paused' error.
        """
        result = evaluator.evaluate(
            eval_templates="pii",
            inputs={
                "text": "My email is test@example.com and my SSN is 123-45-6789.",
            },
            model_name="turing_flash",
        )

        assert result is not None
        assert len(result.eval_results) > 0


# NOTE: TestToneEvaluations (is_polite, is_helpful) removed because
# those evaluator classes don't exist in the backend codebase.
