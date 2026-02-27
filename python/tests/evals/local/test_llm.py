"""Tests for the local LLM module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from fi.evals.local.llm import (
    LocalLLMConfig,
    OllamaLLM,
    LocalLLMFactory,
)


class TestLocalLLMConfig:
    """Tests for LocalLLMConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LocalLLMConfig()

        assert config.model == "llama3.2"
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.timeout == 120

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LocalLLMConfig(
            model="mistral",
            base_url="http://custom:8080",
            temperature=0.7,
            max_tokens=2048,
            timeout=60,
        )

        assert config.model == "mistral"
        assert config.base_url == "http://custom:8080"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.timeout == 60


class TestOllamaLLMInit:
    """Tests for OllamaLLM initialization."""

    @patch('requests.get')
    def test_init_with_available_ollama(self, mock_get):
        """Test initialization when Ollama is available."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "mistral:latest"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        llm = OllamaLLM()

        assert llm.is_available() is True
        assert "llama3.2:latest" in llm.list_models()
        assert "mistral:latest" in llm.list_models()

    @patch('requests.get')
    def test_init_with_unavailable_ollama(self, mock_get):
        """Test initialization when Ollama is not running."""
        mock_get.side_effect = Exception("Connection refused")

        llm = OllamaLLM()

        assert llm.is_available() is False
        assert llm.list_models() == []

    @patch('requests.get')
    def test_init_without_auto_check(self, mock_get):
        """Test initialization with auto_check disabled."""
        llm = OllamaLLM(auto_check=False)

        # Should not call requests until is_available() is called
        mock_get.assert_not_called()
        assert llm._available is None


class TestOllamaLLMGenerate:
    """Tests for OllamaLLM.generate method."""

    @patch('requests.post')
    @patch('requests.get')
    def test_generate_success(self, mock_get, mock_post):
        """Test successful text generation."""
        # Setup availability check
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )

        # Setup generation response
        mock_post.return_value = Mock(
            json=lambda: {"response": "The answer is 4."},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        response = llm.generate("What is 2+2?")

        assert response == "The answer is 4."
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "api/generate" in call_args[0][0]

    @patch('requests.get')
    def test_generate_unavailable_raises_error(self, mock_get):
        """Test that generate raises when Ollama unavailable."""
        mock_get.side_effect = Exception("Connection refused")

        llm = OllamaLLM()

        with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
            llm.generate("What is 2+2?")

    @patch('requests.post')
    @patch('requests.get')
    def test_generate_with_system_prompt(self, mock_get, mock_post):
        """Test generation with system prompt."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )
        mock_post.return_value = Mock(
            json=lambda: {"response": "Hello!"},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        llm.generate("Hi", system="You are a helpful assistant.")

        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "system" in payload
        assert payload["system"] == "You are a helpful assistant."


class TestOllamaLLMChat:
    """Tests for OllamaLLM.chat method."""

    @patch('requests.post')
    @patch('requests.get')
    def test_chat_success(self, mock_get, mock_post):
        """Test successful chat completion."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )
        mock_post.return_value = Mock(
            json=lambda: {"message": {"content": "I'm doing well!"}},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        response = llm.chat([
            {"role": "user", "content": "Hello!"},
        ])

        assert response == "I'm doing well!"
        call_args = mock_post.call_args
        assert "api/chat" in call_args[0][0]


class TestOllamaLLMJudge:
    """Tests for OllamaLLM.judge method."""

    @patch('requests.post')
    @patch('requests.get')
    def test_judge_valid_json_response(self, mock_get, mock_post):
        """Test judge with valid JSON response."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )
        mock_post.return_value = Mock(
            json=lambda: {
                "response": json.dumps({
                    "score": 0.9,
                    "passed": True,
                    "reason": "The response correctly answers the question."
                })
            },
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        result = llm.judge(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            criteria="Evaluate if the response correctly answers the question.",
        )

        assert result["score"] == 0.9
        assert result["passed"] is True
        assert "correctly answers" in result["reason"]

    @patch('requests.post')
    @patch('requests.get')
    def test_judge_json_in_code_block(self, mock_get, mock_post):
        """Test judge with JSON wrapped in markdown code block."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )
        mock_post.return_value = Mock(
            json=lambda: {
                "response": '''```json
{
    "score": 0.8,
    "passed": true,
    "reason": "Good answer"
}
```'''
            },
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        result = llm.judge(
            query="test query",
            response="test response",
            criteria="test criteria",
        )

        assert result["score"] == 0.8
        assert result["passed"] is True

    @patch('requests.post')
    @patch('requests.get')
    def test_judge_json_embedded_in_text(self, mock_get, mock_post):
        """Test judge with JSON embedded in surrounding text."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )
        mock_post.return_value = Mock(
            json=lambda: {
                "response": '''Based on my analysis, here is my evaluation:
{"score": 0.7, "passed": true, "reason": "Decent answer"}
That's my evaluation.'''
            },
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        result = llm.judge(
            query="test query",
            response="test response",
            criteria="test criteria",
        )

        assert result["score"] == 0.7
        assert result["passed"] is True

    @patch('requests.post')
    @patch('requests.get')
    def test_judge_score_normalization(self, mock_get, mock_post):
        """Test that scores > 1 are normalized."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )
        mock_post.return_value = Mock(
            json=lambda: {
                "response": json.dumps({
                    "score": 8,  # Score on 1-10 scale
                    "passed": True,
                    "reason": "Good"
                })
            },
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        result = llm.judge(
            query="test query",
            response="test response",
            criteria="test criteria",
        )

        assert result["score"] == 0.8  # Normalized to 0-1

    @patch('requests.post')
    @patch('requests.get')
    def test_judge_fallback_parsing(self, mock_get, mock_post):
        """Test judge fallback parsing when JSON fails."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )
        # Use format that matches the regex: "score: 0.75" or "score 0.75"
        mock_post.return_value = Mock(
            json=lambda: {
                "response": "Based on my analysis, the score: 0.75. It's a good answer."
            },
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        result = llm.judge(
            query="test query",
            response="test response",
            criteria="test criteria",
        )

        assert result["score"] == 0.75
        assert result.get("parse_error") is True

    @patch('requests.post')
    @patch('requests.get')
    def test_judge_with_context(self, mock_get, mock_post):
        """Test judge with context parameter."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )
        mock_post.return_value = Mock(
            json=lambda: {
                "response": json.dumps({
                    "score": 0.95,
                    "passed": True,
                    "reason": "Response matches context"
                })
            },
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        result = llm.judge(
            query="What color is the sky?",
            response="The sky is blue.",
            criteria="Evaluate accuracy based on context.",
            context="Scientific fact: The sky appears blue due to Rayleigh scattering.",
        )

        assert result["score"] == 0.95
        # Verify context was included in the prompt
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "Context" in payload["prompt"] or "context" in payload["prompt"].lower()


class TestOllamaLLMBatchJudge:
    """Tests for OllamaLLM.batch_judge method."""

    @patch('requests.post')
    @patch('requests.get')
    def test_batch_judge_multiple_evaluations(self, mock_get, mock_post):
        """Test batch evaluation with multiple items."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )

        # Return different responses for each call
        responses = [
            {"response": json.dumps({"score": 0.9, "passed": True, "reason": "Good"})},
            {"response": json.dumps({"score": 0.6, "passed": True, "reason": "OK"})},
            {"response": json.dumps({"score": 0.3, "passed": False, "reason": "Poor"})},
        ]
        mock_post.return_value.json.side_effect = responses
        mock_post.return_value.raise_for_status = Mock()

        llm = OllamaLLM()
        results = llm.batch_judge([
            {"query": "Q1", "response": "R1", "criteria": "C1"},
            {"query": "Q2", "response": "R2", "criteria": "C2"},
            {"query": "Q3", "response": "R3", "criteria": "C3"},
        ])

        assert len(results) == 3
        assert results[0]["score"] == 0.9
        assert results[1]["score"] == 0.6
        assert results[2]["score"] == 0.3


class TestLocalLLMFactory:
    """Tests for LocalLLMFactory."""

    @patch('requests.get')
    def test_create_ollama_backend(self, mock_get):
        """Test creating an Ollama backend."""
        mock_get.side_effect = Exception("Not running")  # Skip availability check

        llm = LocalLLMFactory.create(backend="ollama", auto_check=False)

        assert isinstance(llm, OllamaLLM)

    def test_create_unsupported_backend_raises(self):
        """Test that unsupported backends raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported LLM backend"):
            LocalLLMFactory.create(backend="unsupported")

    @patch('requests.get')
    def test_from_string_with_model(self, mock_get):
        """Test creating LLM from string spec with model."""
        mock_get.side_effect = Exception("Not running")

        llm = LocalLLMFactory.from_string("ollama/mistral")

        assert isinstance(llm, OllamaLLM)
        assert llm.config.model == "mistral"

    @patch('requests.get')
    def test_from_string_without_model(self, mock_get):
        """Test creating LLM from string spec without model."""
        mock_get.side_effect = Exception("Not running")

        llm = LocalLLMFactory.from_string("ollama")

        assert isinstance(llm, OllamaLLM)
        assert llm.config.model == "llama3.2"  # Default


class TestJudgeResultValidation:
    """Tests for judge result validation logic."""

    @patch('requests.get')
    def test_validate_string_score(self, mock_get):
        """Test validation handles string scores."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        result = llm._validate_judge_result({"score": "0.85", "reason": "test"})

        assert result["score"] == 0.85
        assert isinstance(result["score"], float)

    @patch('requests.get')
    def test_validate_string_passed(self, mock_get):
        """Test validation handles string passed values."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()

        result = llm._validate_judge_result({"score": 0.7, "passed": "true"})
        assert result["passed"] is True

        result = llm._validate_judge_result({"score": 0.7, "passed": "yes"})
        assert result["passed"] is True

        result = llm._validate_judge_result({"score": 0.7, "passed": "false"})
        assert result["passed"] is False

    @patch('requests.get')
    def test_validate_missing_passed(self, mock_get):
        """Test validation derives passed from score when missing."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()

        result = llm._validate_judge_result({"score": 0.7, "reason": "test"})
        assert result["passed"] is True  # 0.7 >= 0.5

        result = llm._validate_judge_result({"score": 0.3, "reason": "test"})
        assert result["passed"] is False  # 0.3 < 0.5

    @patch('requests.get')
    def test_validate_score_clamping(self, mock_get):
        """Test that scores are clamped and normalized correctly."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()

        # Negative scores get clamped to 0
        result = llm._validate_judge_result({"score": -0.5, "reason": "test"})
        assert result["score"] == 0.0  # Clamped to min

        # Scores > 1 and <= 10 get normalized (divided by 10)
        result = llm._validate_judge_result({"score": 8, "reason": "test"})
        assert result["score"] == 0.8  # Normalized from 1-10 scale

        # Scores > 10 get normalized (divided by 100)
        result = llm._validate_judge_result({"score": 85, "reason": "test"})
        assert result["score"] == 0.85  # Normalized from 1-100 scale

        # Normal 0-1 scores stay as-is
        result = llm._validate_judge_result({"score": 0.7, "reason": "test"})
        assert result["score"] == 0.7

    @patch('requests.get')
    def test_validate_explanation_fallback(self, mock_get):
        """Test that 'explanation' is used as fallback for 'reason'."""
        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()

        result = llm._validate_judge_result({"score": 0.7, "explanation": "This is why"})
        assert result["reason"] == "This is why"


class TestHybridEvaluatorWithLocalLLM:
    """Tests for HybridEvaluator with local LLM integration."""

    @patch('requests.get')
    def test_can_use_local_llm(self, mock_get):
        """Test can_use_local_llm detection."""
        from fi.evals.local import HybridEvaluator

        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        evaluator = HybridEvaluator(local_llm=llm)

        # LLM-based metrics should be detectable
        assert evaluator.can_use_local_llm("groundedness") is True
        assert evaluator.can_use_local_llm("hallucination") is True
        assert evaluator.can_use_local_llm("relevance") is True

        # Heuristic metrics should not
        assert evaluator.can_use_local_llm("contains") is False
        assert evaluator.can_use_local_llm("is_json") is False

    @patch('requests.get')
    def test_hybrid_routing_with_local_llm(self, mock_get):
        """Test that hybrid evaluator routes LLM metrics to local when LLM available."""
        from fi.evals.local import HybridEvaluator, ExecutionMode

        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        evaluator = HybridEvaluator(local_llm=llm, prefer_local=True)

        # Heuristic metrics -> LOCAL
        assert evaluator.route_evaluation("contains") == ExecutionMode.LOCAL

        # LLM metrics with local LLM -> LOCAL
        assert evaluator.route_evaluation("groundedness") == ExecutionMode.LOCAL

    def test_hybrid_routing_without_local_llm(self):
        """Test that hybrid evaluator routes LLM metrics to cloud without local LLM."""
        from fi.evals.local import HybridEvaluator, ExecutionMode

        evaluator = HybridEvaluator(local_llm=None)

        # Heuristic metrics -> LOCAL
        assert evaluator.route_evaluation("contains") == ExecutionMode.LOCAL

        # LLM metrics without local LLM -> CLOUD
        assert evaluator.route_evaluation("groundedness") == ExecutionMode.CLOUD

    def test_offline_mode_raises_for_cloud_metric(self):
        """Test that offline mode raises for metrics requiring cloud."""
        from fi.evals.local import HybridEvaluator

        evaluator = HybridEvaluator(local_llm=None, offline_mode=True)

        with pytest.raises(ValueError, match="requires cloud execution"):
            evaluator.route_evaluation("groundedness")

    @patch('requests.get')
    def test_partition_with_local_llm(self, mock_get):
        """Test partitioning with local LLM available."""
        from fi.evals.local import HybridEvaluator, ExecutionMode

        mock_get.return_value = Mock(
            json=lambda: {"models": [{"name": "llama3.2"}]},
            raise_for_status=Mock(),
        )

        llm = OllamaLLM()
        evaluator = HybridEvaluator(local_llm=llm, prefer_local=True)

        evaluations = [
            {"metric_name": "contains", "inputs": [{"response": "test"}]},
            {"metric_name": "groundedness", "inputs": [{"response": "test"}]},
            {"metric_name": "is_json", "inputs": [{"response": "{}"}]},
        ]

        partitions = evaluator.partition_evaluations(evaluations)

        # All should be local with local LLM
        assert len(partitions[ExecutionMode.LOCAL]) == 3
        assert len(partitions[ExecutionMode.CLOUD]) == 0
