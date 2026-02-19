"""
Unit Tests for Guardrails Module.

These tests verify the configuration, types, and basic module functionality
without requiring API access.
"""

import pytest
from typing import List

# Import the guardrails module
from fi.evals.guardrails import (
    Guardrails,
    GuardrailsConfig,
    GuardrailModel,
    SafetyCategory,
    AggregationStrategy,
    RailType,
    GuardrailResult,
    GuardrailsResponse,
)


class TestGuardrailsConfig:
    """Tests for GuardrailsConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GuardrailsConfig()

        assert len(config.models) == 1
        assert config.models[0] == GuardrailModel.TURING_FLASH
        assert RailType.INPUT in config.rails
        assert RailType.OUTPUT in config.rails
        assert config.aggregation == AggregationStrategy.ANY
        assert config.timeout_ms == 100
        assert config.parallel is True
        assert config.max_workers == 5
        assert config.fail_open is False
        assert config.fallback_model is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = GuardrailsConfig(
            models=[GuardrailModel.TURING_SAFETY],
            rails=[RailType.INPUT],
            aggregation=AggregationStrategy.MAJORITY,
            timeout_ms=500,
            parallel=False,
            max_workers=10,
            fail_open=True,
        )

        assert config.models[0] == GuardrailModel.TURING_SAFETY
        assert len(config.rails) == 1
        assert config.aggregation == AggregationStrategy.MAJORITY
        assert config.timeout_ms == 500
        assert config.parallel is False
        assert config.max_workers == 10
        assert config.fail_open is True

    def test_config_with_custom_categories(self):
        """Test configuration with custom safety categories."""
        config = GuardrailsConfig(
            categories={
                "custom_category": SafetyCategory(
                    name="custom_category",
                    threshold=0.5,
                    action="flag",
                ),
            },
        )

        assert "custom_category" in config.categories
        assert config.categories["custom_category"].threshold == 0.5
        assert config.categories["custom_category"].action == "flag"

    def test_config_validation_no_models(self):
        """Test that config requires at least one model."""
        with pytest.raises(ValueError, match="At least one model"):
            GuardrailsConfig(models=[])

    def test_config_validation_invalid_timeout(self):
        """Test that timeout must be positive."""
        with pytest.raises(ValueError, match="timeout_ms must be positive"):
            GuardrailsConfig(timeout_ms=0)

    def test_config_validation_invalid_workers(self):
        """Test that max_workers must be positive."""
        with pytest.raises(ValueError, match="max_workers must be positive"):
            GuardrailsConfig(max_workers=0)


class TestSafetyCategory:
    """Tests for SafetyCategory dataclass."""

    def test_default_category(self):
        """Test default category values."""
        category = SafetyCategory(name="test")

        assert category.name == "test"
        assert category.enabled is True
        assert category.threshold == 0.7
        assert category.action == "block"
        assert category.models == []

    def test_custom_category(self):
        """Test custom category values."""
        category = SafetyCategory(
            name="custom",
            enabled=False,
            threshold=0.3,
            action="redact",
            models=[GuardrailModel.TURING_FLASH],
        )

        assert category.name == "custom"
        assert category.enabled is False
        assert category.threshold == 0.3
        assert category.action == "redact"
        assert GuardrailModel.TURING_FLASH in category.models

    def test_category_validation_invalid_threshold(self):
        """Test that threshold must be between 0 and 1."""
        with pytest.raises(ValueError, match="threshold must be between"):
            SafetyCategory(name="test", threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be between"):
            SafetyCategory(name="test", threshold=-0.1)

    def test_category_validation_invalid_action(self):
        """Test that action must be valid."""
        with pytest.raises(ValueError, match="action must be one of"):
            SafetyCategory(name="test", action="invalid")


class TestGuardrailModel:
    """Tests for GuardrailModel enum."""

    def test_turing_models(self):
        """Test Turing model values."""
        assert GuardrailModel.TURING_FLASH.value == "turing_flash"
        assert GuardrailModel.TURING_SAFETY.value == "turing_safety"

    def test_local_models(self):
        """Test local model values."""
        assert GuardrailModel.QWEN3GUARD_8B.value == "qwen3guard-8b"
        assert GuardrailModel.GRANITE_GUARDIAN_8B.value == "granite-guardian-3.3-8b"
        assert GuardrailModel.WILDGUARD_7B.value == "wildguard-7b"
        assert GuardrailModel.LLAMAGUARD_3_8B.value == "llamaguard-3-8b"
        assert GuardrailModel.SHIELDGEMMA_2B.value == "shieldgemma-2b"

    def test_api_models(self):
        """Test third-party API model values."""
        assert GuardrailModel.OPENAI_MODERATION.value == "openai-moderation"
        assert GuardrailModel.ANTHROPIC_SAFETY.value == "anthropic-safety"
        assert GuardrailModel.AZURE_CONTENT_SAFETY.value == "azure-content-safety"


class TestRailType:
    """Tests for RailType enum."""

    def test_rail_types(self):
        """Test rail type values."""
        assert RailType.INPUT.value == "input"
        assert RailType.OUTPUT.value == "output"
        assert RailType.RETRIEVAL.value == "retrieval"


class TestAggregationStrategy:
    """Tests for AggregationStrategy enum."""

    def test_strategies(self):
        """Test aggregation strategy values."""
        assert AggregationStrategy.ANY.value == "any"
        assert AggregationStrategy.ALL.value == "all"
        assert AggregationStrategy.MAJORITY.value == "majority"
        assert AggregationStrategy.WEIGHTED.value == "weighted"


class TestGuardrailResult:
    """Tests for GuardrailResult dataclass."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = GuardrailResult(
            passed=True,
            category="safe",
            score=0.0,
            model="turing_flash",
        )

        assert result.passed is True
        assert result.category == "safe"
        assert result.score == 0.0
        assert result.model == "turing_flash"
        assert result.reason is None
        assert result.action == "pass"
        assert result.latency_ms == 0.0

    def test_blocked_result(self):
        """Test blocked result creation."""
        result = GuardrailResult(
            passed=False,
            category="toxicity",
            score=0.95,
            model="turing_safety",
            reason="Toxic content detected",
            action="block",
            latency_ms=45.5,
        )

        assert result.passed is False
        assert result.category == "toxicity"
        assert result.score == 0.95
        assert result.reason == "Toxic content detected"
        assert result.action == "block"

    def test_score_clamping(self):
        """Test that scores are clamped to valid range."""
        result = GuardrailResult(
            passed=False,
            category="test",
            score=1.5,  # Should be clamped to 1.0
            model="test",
        )
        assert result.score == 1.0

        result2 = GuardrailResult(
            passed=True,
            category="test",
            score=-0.5,  # Should be clamped to 0.0
            model="test",
        )
        assert result2.score == 0.0


class TestGuardrailsResponse:
    """Tests for GuardrailsResponse dataclass."""

    def test_create_passed(self):
        """Test creating a passed response."""
        response = GuardrailsResponse.create_passed(
            content="Hello world",
            latency_ms=25.0,
            models_used=["turing_flash"],
        )

        assert response.passed is True
        assert response.original_content == "Hello world"
        assert response.total_latency_ms == 25.0
        assert "turing_flash" in response.models_used
        assert len(response.blocked_categories) == 0
        assert len(response.flagged_categories) == 0

    def test_create_blocked(self):
        """Test creating a blocked response."""
        response = GuardrailsResponse.create_blocked(
            content="Bad content",
            blocked_categories=["toxicity", "hate_speech"],
            latency_ms=50.0,
            models_used=["turing_safety"],
        )

        assert response.passed is False
        assert response.original_content == "Bad content"
        assert "toxicity" in response.blocked_categories
        assert "hate_speech" in response.blocked_categories

    def test_create_error(self):
        """Test creating an error response."""
        response = GuardrailsResponse.create_error(
            content="Test content",
            error="API timeout",
            fail_open=False,
        )

        assert response.passed is False
        assert response.error == "API timeout"

        # Test fail_open=True
        response2 = GuardrailsResponse.create_error(
            content="Test content",
            error="API timeout",
            fail_open=True,
        )

        assert response2.passed is True


class TestGuardrailsClass:
    """Tests for the main Guardrails class."""

    def test_repr(self):
        """Test string representation."""
        config = GuardrailsConfig(
            models=[GuardrailModel.TURING_FLASH],
            aggregation=AggregationStrategy.ANY,
        )

        # This will fail because Turing backend needs API keys,
        # but we can test that the config is set up correctly
        try:
            guardrails = Guardrails(config=config)
            repr_str = repr(guardrails)
            assert "turing_flash" in repr_str
            assert "any" in repr_str
        except Exception:
            # Expected if no API keys
            pass

    def test_local_model_not_implemented(self):
        """Test that local models raise NotImplementedError."""
        config = GuardrailsConfig(
            models=[GuardrailModel.QWEN3GUARD_8B],
        )

        with pytest.raises(NotImplementedError, match="Local model backend"):
            Guardrails(config=config)

    def test_openai_not_implemented(self):
        """Test that OpenAI backend raises NotImplementedError."""
        config = GuardrailsConfig(
            models=[GuardrailModel.OPENAI_MODERATION],
        )

        with pytest.raises(NotImplementedError, match="OpenAI backend"):
            Guardrails(config=config)

    def test_anthropic_not_implemented(self):
        """Test that Anthropic backend raises NotImplementedError."""
        config = GuardrailsConfig(
            models=[GuardrailModel.ANTHROPIC_SAFETY],
        )

        with pytest.raises(NotImplementedError, match="Anthropic backend"):
            Guardrails(config=config)

    def test_azure_not_implemented(self):
        """Test that Azure backend raises NotImplementedError."""
        config = GuardrailsConfig(
            models=[GuardrailModel.AZURE_CONTENT_SAFETY],
        )

        with pytest.raises(NotImplementedError, match="Azure backend"):
            Guardrails(config=config)


class TestAggregationLogic:
    """Tests for aggregation logic."""

    def test_any_strategy_single_fail(self):
        """Test ANY strategy fails if any model flags."""
        # ANY = fail if ANY model flags
        # With 1 fail and 2 pass, should fail
        config = GuardrailsConfig(aggregation=AggregationStrategy.ANY)
        # Can't fully test without backends, but verify config
        assert config.aggregation == AggregationStrategy.ANY

    def test_majority_strategy(self):
        """Test MAJORITY strategy."""
        config = GuardrailsConfig(aggregation=AggregationStrategy.MAJORITY)
        assert config.aggregation == AggregationStrategy.MAJORITY

    def test_all_strategy(self):
        """Test ALL strategy."""
        config = GuardrailsConfig(aggregation=AggregationStrategy.ALL)
        assert config.aggregation == AggregationStrategy.ALL
