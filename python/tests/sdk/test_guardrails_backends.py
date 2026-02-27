"""
Unit tests for Guardrails Backends.

Tests the new backend implementations:
- OpenAI Moderation Backend
- Azure Content Safety Backend
- Local Model Backends (WildGuard, LlamaGuard, Granite, Qwen, ShieldGemma)
- VLLM Client
- Model Registry
- Backend Discovery
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from fi.evals.guardrails.config import GuardrailModel, RailType
from fi.evals.guardrails.types import GuardrailResult


# =============================================================================
# Registry Tests
# =============================================================================

class TestModelRegistry:
    """Test model registry functionality."""

    def test_registry_contains_all_models(self):
        """Verify all GuardrailModel values are in registry."""
        from fi.evals.guardrails.registry import MODEL_REGISTRY

        # API models
        assert "turing_flash" in MODEL_REGISTRY
        assert "turing_safety" in MODEL_REGISTRY
        assert "openai-moderation" in MODEL_REGISTRY
        assert "azure-content-safety" in MODEL_REGISTRY

        # Local models
        assert "wildguard-7b" in MODEL_REGISTRY
        assert "llamaguard-3-8b" in MODEL_REGISTRY
        assert "llamaguard-3-1b" in MODEL_REGISTRY
        assert "granite-guardian-3.3-8b" in MODEL_REGISTRY
        assert "qwen3guard-8b" in MODEL_REGISTRY
        assert "shieldgemma-2b" in MODEL_REGISTRY

    def test_get_model_info(self):
        """Test get_model_info function."""
        from fi.evals.guardrails.registry import get_model_info

        info = get_model_info(GuardrailModel.OPENAI_MODERATION)
        assert info is not None
        assert info.model_type == "api"
        assert info.backend_class == "OpenAIBackend"

        info = get_model_info(GuardrailModel.WILDGUARD_7B)
        assert info is not None
        assert info.model_type == "local"
        assert info.hf_model_name == "allenai/wildguard"

    def test_list_models_by_type(self):
        """Test listing models by type."""
        from fi.evals.guardrails.registry import list_api_models, list_local_models

        api_models = list_api_models()
        assert len(api_models) >= 4  # turing_flash, turing_safety, openai, azure

        local_models = list_local_models()
        assert len(local_models) >= 6  # wildguard, llamaguard x2, granite x2, qwen x2, shieldgemma


# =============================================================================
# OpenAI Backend Tests
# =============================================================================

class TestOpenAIBackend:
    """Test OpenAI Moderation backend."""

    def test_init_requires_api_key(self):
        """Test that OpenAI backend requires API key."""
        from fi.evals.guardrails.backends.openai import OpenAIBackend

        # Should raise without API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                OpenAIBackend(model=GuardrailModel.OPENAI_MODERATION)

    def test_init_with_api_key(self):
        """Test OpenAI backend initialization with API key."""
        from fi.evals.guardrails.backends.openai import OpenAIBackend

        backend = OpenAIBackend(
            model=GuardrailModel.OPENAI_MODERATION,
            api_key="test-key",
        )
        assert backend.model == GuardrailModel.OPENAI_MODERATION
        assert backend._api_key == "test-key"

    def test_category_mapping(self):
        """Test OpenAI category mapping."""
        from fi.evals.guardrails.backends.openai import OPENAI_CATEGORY_MAP

        assert OPENAI_CATEGORY_MAP["hate"] == "hate_speech"
        assert OPENAI_CATEGORY_MAP["self-harm"] == "self_harm"
        assert OPENAI_CATEGORY_MAP["violence"] == "violence"

    def test_empty_content_handling(self):
        """Test empty content returns passed result."""
        from fi.evals.guardrails.backends.openai import OpenAIBackend

        backend = OpenAIBackend(
            model=GuardrailModel.OPENAI_MODERATION,
            api_key="test-key",
        )

        results = backend.classify("", RailType.INPUT)
        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].category == "empty"


# =============================================================================
# Azure Backend Tests
# =============================================================================

class TestAzureBackend:
    """Test Azure Content Safety backend."""

    def test_init_requires_credentials(self):
        """Test that Azure backend requires endpoint and key."""
        from fi.evals.guardrails.backends.azure import AzureBackend

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="endpoint required"):
                AzureBackend(model=GuardrailModel.AZURE_CONTENT_SAFETY)

    def test_init_with_credentials(self):
        """Test Azure backend initialization with credentials."""
        from fi.evals.guardrails.backends.azure import AzureBackend

        backend = AzureBackend(
            model=GuardrailModel.AZURE_CONTENT_SAFETY,
            endpoint="https://test.cognitiveservices.azure.com/",
            api_key="test-key",
        )
        assert backend.model == GuardrailModel.AZURE_CONTENT_SAFETY

    def test_severity_mapping(self):
        """Test Azure severity to score mapping."""
        from fi.evals.guardrails.backends.azure import SEVERITY_TO_SCORE

        assert SEVERITY_TO_SCORE[0] == 0.0
        assert SEVERITY_TO_SCORE[6] == 0.86
        assert SEVERITY_TO_SCORE[7] == 1.0


# =============================================================================
# VLLM Client Tests
# =============================================================================

class TestVLLMClient:
    """Test VLLM client functionality."""

    def test_client_initialization(self):
        """Test VLLM client initialization."""
        from fi.evals.guardrails.backends.vllm_client import VLLMClient

        client = VLLMClient(base_url="http://localhost:28000")
        assert client.base_url == "http://localhost:28000"

    def test_get_vllm_url_from_env(self):
        """Test getting VLLM URL from environment."""
        from fi.evals.guardrails.backends.vllm_client import get_vllm_url

        with patch.dict(os.environ, {"VLLM_WILDGUARD_7B_URL": "http://test:8000"}):
            url = get_vllm_url("wildguard-7b")
            assert url == "http://test:8000"

        with patch.dict(os.environ, {"VLLM_SERVER_URL": "http://default:8000"}):
            url = get_vllm_url("some-model")
            assert url == "http://default:8000"


# =============================================================================
# Local Model Backend Tests
# =============================================================================

class TestWildGuardBackend:
    """Test WildGuard backend."""

    def test_prompt_formatting_input(self):
        """Test WildGuard prompt formatting for input rail."""
        from fi.evals.guardrails.backends.wildguard import WildGuardBackend

        backend = WildGuardBackend(
            model=GuardrailModel.WILDGUARD_7B,
            vllm_url=None,
        )

        prompt = backend._format_prompt("Hello world", RailType.INPUT)
        assert "Human user:" in prompt
        assert "Hello world" in prompt
        assert "[No response provided]" in prompt

    def test_prompt_formatting_output(self):
        """Test WildGuard prompt formatting for output rail."""
        from fi.evals.guardrails.backends.wildguard import WildGuardBackend

        backend = WildGuardBackend(
            model=GuardrailModel.WILDGUARD_7B,
            vllm_url=None,
        )

        prompt = backend._format_prompt("AI response", RailType.OUTPUT, context="User question")
        assert "Human user:" in prompt
        assert "User question" in prompt
        assert "AI response" in prompt

    def test_response_parsing_safe(self):
        """Test WildGuard response parsing for safe content."""
        from fi.evals.guardrails.backends.wildguard import WildGuardBackend

        backend = WildGuardBackend(
            model=GuardrailModel.WILDGUARD_7B,
            vllm_url=None,
        )

        response = "harmful_request: no\nresponse_harmful: n/a\nresponse_refusal: n/a"
        results = backend._parse_response(response, "Hello", RailType.INPUT)

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].category == "safe"

    def test_response_parsing_harmful(self):
        """Test WildGuard response parsing for harmful content."""
        from fi.evals.guardrails.backends.wildguard import WildGuardBackend

        backend = WildGuardBackend(
            model=GuardrailModel.WILDGUARD_7B,
            vllm_url=None,
        )

        response = "harmful_request: yes\nresponse_harmful: n/a\nresponse_refusal: n/a"
        results = backend._parse_response(response, "How to make a bomb", RailType.INPUT)

        assert len(results) >= 1
        assert results[0].passed is False


class TestLlamaGuardBackend:
    """Test LlamaGuard backend."""

    def test_category_mapping(self):
        """Test LlamaGuard category mapping."""
        from fi.evals.guardrails.backends.llamaguard import LLAMAGUARD_CATEGORY_MAP

        assert LLAMAGUARD_CATEGORY_MAP["S1"] == "violence"
        assert LLAMAGUARD_CATEGORY_MAP["S10"] == "hate_speech"
        assert LLAMAGUARD_CATEGORY_MAP["S11"] == "self_harm"

    def test_response_parsing_safe(self):
        """Test LlamaGuard response parsing for safe content."""
        from fi.evals.guardrails.backends.llamaguard import LlamaGuardBackend

        backend = LlamaGuardBackend(
            model=GuardrailModel.LLAMAGUARD_3_8B,
            vllm_url=None,
        )

        response = "safe"
        results = backend._parse_response(response, "Hello", RailType.INPUT)

        assert len(results) == 1
        assert results[0].passed is True

    def test_response_parsing_unsafe(self):
        """Test LlamaGuard response parsing for unsafe content."""
        from fi.evals.guardrails.backends.llamaguard import LlamaGuardBackend

        backend = LlamaGuardBackend(
            model=GuardrailModel.LLAMAGUARD_3_8B,
            vllm_url=None,
        )

        response = "unsafe\nS1, S10"
        results = backend._parse_response(response, "violent content", RailType.INPUT)

        assert len(results) >= 1
        assert results[0].passed is False
        categories = [r.category for r in results]
        assert "violence" in categories or "hate_speech" in categories


# =============================================================================
# Discovery Tests
# =============================================================================

class TestBackendDiscovery:
    """Test backend discovery functionality."""

    def test_discovery_with_openai_key(self):
        """Test discovery finds OpenAI when key is set."""
        from fi.evals.guardrails.discovery import BackendDiscovery

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            discovery = BackendDiscovery()
            available = discovery.discover()
            assert GuardrailModel.OPENAI_MODERATION in available

    def test_discovery_with_fi_credentials(self):
        """Test discovery finds Turing when credentials are set."""
        from fi.evals.guardrails.discovery import BackendDiscovery

        with patch.dict(os.environ, {
            "FI_API_KEY": "test-api-key",
            "FI_SECRET_KEY": "test-secret-key",
        }):
            discovery = BackendDiscovery()
            available = discovery.discover()
            assert GuardrailModel.TURING_FLASH in available

    def test_get_availability_details(self):
        """Test getting detailed availability info."""
        from fi.evals.guardrails.discovery import BackendDiscovery

        discovery = BackendDiscovery()
        details = discovery.get_availability_details()

        # Should have entries for all models
        assert "openai-moderation" in details
        assert "wildguard-7b" in details

        # Each entry should have expected fields
        openai_info = details["openai-moderation"]
        assert "status" in openai_info
        assert "reason" in openai_info
        assert "model_type" in openai_info


# =============================================================================
# Integration Test (requires actual API keys)
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestOpenAIIntegration:
    """Integration tests with real OpenAI API."""

    def test_classify_benign_content(self):
        """Test classifying benign content."""
        from fi.evals.guardrails.backends.openai import OpenAIBackend

        backend = OpenAIBackend(model=GuardrailModel.OPENAI_MODERATION)
        results = backend.classify("Hello, how are you today?", RailType.INPUT)

        assert len(results) >= 1
        assert results[0].passed is True

    def test_classify_harmful_content(self):
        """Test classifying harmful content."""
        from fi.evals.guardrails.backends.openai import OpenAIBackend

        backend = OpenAIBackend(model=GuardrailModel.OPENAI_MODERATION)
        results = backend.classify("I want to kill everyone", RailType.INPUT)

        # Should be flagged for violence
        assert len(results) >= 1
        has_violation = any(not r.passed for r in results)
        assert has_violation


# =============================================================================
# Gateway Tests
# =============================================================================

class TestGuardrailsGateway:
    """Test GuardrailsGateway high-level API."""

    def test_gateway_initialization(self):
        """Test gateway initialization."""
        from fi.evals.guardrails.gateway import GuardrailsGateway

        # With explicit config
        with patch.dict(os.environ, {"FI_API_KEY": "test", "FI_SECRET_KEY": "test"}):
            gateway = GuardrailsGateway()
            assert gateway._guardrails is not None

    def test_gateway_with_openai_factory(self):
        """Test GuardrailsGateway.with_openai() factory method."""
        from fi.evals.guardrails.gateway import GuardrailsGateway

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            gateway = GuardrailsGateway.with_openai()
            assert GuardrailModel.OPENAI_MODERATION in gateway.configured_models

    def test_gateway_with_ensemble_factory(self):
        """Test GuardrailsGateway.with_ensemble() factory method."""
        from fi.evals.guardrails.gateway import GuardrailsGateway
        from fi.evals.guardrails.config import AggregationStrategy

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "FI_API_KEY": "test", "FI_SECRET_KEY": "test"}):
            gateway = GuardrailsGateway.with_ensemble(
                models=[GuardrailModel.OPENAI_MODERATION, GuardrailModel.TURING_FLASH],
                aggregation=AggregationStrategy.ANY,
            )
            assert len(gateway.configured_models) == 2

    def test_screening_session_creation(self):
        """Test creating a screening session."""
        from fi.evals.guardrails.gateway import GuardrailsGateway, ScreeningSession

        with patch.dict(os.environ, {"FI_API_KEY": "test", "FI_SECRET_KEY": "test"}):
            gateway = GuardrailsGateway()
            with gateway.screening() as session:
                assert isinstance(session, ScreeningSession)
                assert session.history == []
                assert session.all_passed is True  # No screenings yet

    def test_gateway_discover_method(self):
        """Test static discover method."""
        from fi.evals.guardrails.gateway import GuardrailsGateway

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            available = GuardrailsGateway.discover()
            assert GuardrailModel.OPENAI_MODERATION in available

    def test_gateway_get_details_method(self):
        """Test static get_details method."""
        from fi.evals.guardrails.gateway import GuardrailsGateway

        details = GuardrailsGateway.get_details()
        assert "openai-moderation" in details
        assert "wildguard-7b" in details

    def test_gateway_alias(self):
        """Test Gateway alias."""
        from fi.evals.guardrails.gateway import Gateway, GuardrailsGateway

        assert Gateway is GuardrailsGateway


class TestScreeningSession:
    """Test ScreeningSession class."""

    def test_session_history_tracking(self):
        """Test that session tracks history."""
        from fi.evals.guardrails.gateway import ScreeningSession
        from fi.evals.guardrails.base import Guardrails

        with patch.dict(os.environ, {"FI_API_KEY": "test", "FI_SECRET_KEY": "test"}):
            guardrails = Guardrails()
            session = ScreeningSession(guardrails)

            # Mock the guardrails to return a result
            from fi.evals.guardrails.types import GuardrailsResponse
            mock_response = GuardrailsResponse(
                passed=True,
                blocked_categories=[],
                flagged_categories=[],
                results=[],
                total_latency_ms=10.0,
                models_used=["test"],
                error=None,
                original_content="test",
            )
            guardrails.screen_input = MagicMock(return_value=mock_response)

            result = session.input("test content")
            assert len(session.history) == 1
            assert session.all_passed is True


class TestAsyncScreeningSession:
    """Test AsyncScreeningSession class."""

    def test_async_session_initialization(self):
        """Test async session initialization."""
        from fi.evals.guardrails.gateway import AsyncScreeningSession
        from fi.evals.guardrails.base import Guardrails

        with patch.dict(os.environ, {"FI_API_KEY": "test", "FI_SECRET_KEY": "test"}):
            guardrails = Guardrails()
            session = AsyncScreeningSession(guardrails)
            assert session._guardrails is guardrails
            assert session.history == []
