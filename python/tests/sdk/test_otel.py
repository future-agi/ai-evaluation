"""
Tests for OpenTelemetry Integration.

Tests cover:
- Types and configurations
- Semantic conventions
- Span processors
- Cost calculation
- Instrumentors
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch
import time


class TestOtelTypes:
    """Tests for OTEL type definitions."""

    def test_exporter_type_enum(self):
        """Test ExporterType enum values."""
        from fi.evals.otel import ExporterType

        assert ExporterType.OTLP_GRPC.value == "otlp_grpc"
        assert ExporterType.OTLP_HTTP.value == "otlp_http"
        assert ExporterType.CONSOLE.value == "console"
        assert ExporterType.JAEGER.value == "jaeger"
        assert ExporterType.ZIPKIN.value == "zipkin"
        assert ExporterType.ARIZE.value == "arize"
        assert ExporterType.LANGFUSE.value == "langfuse"

    def test_span_kind_enum(self):
        """Test SpanKind enum values."""
        from fi.evals.otel import SpanKind

        assert SpanKind.LLM.value == "llm"
        assert SpanKind.RETRIEVER.value == "retriever"
        assert SpanKind.EMBEDDING.value == "embedding"
        assert SpanKind.AGENT.value == "agent"

    def test_processor_type_enum(self):
        """Test ProcessorType enum values."""
        from fi.evals.otel import ProcessorType

        assert ProcessorType.LLM.value == "llm"
        assert ProcessorType.EVALUATION.value == "evaluation"
        assert ProcessorType.COST.value == "cost"

    def test_token_pricing(self):
        """Test TokenPricing dataclass."""
        from fi.evals.otel import TokenPricing

        pricing = TokenPricing(
            model="gpt-4",
            input_per_1k=0.03,
            output_per_1k=0.06,
        )

        assert pricing.model == "gpt-4"
        assert pricing.input_per_1k == 0.03
        assert pricing.output_per_1k == 0.06
        assert pricing.input_per_token == pytest.approx(0.00003, rel=1e-6)
        assert pricing.output_per_token == pytest.approx(0.00006, rel=1e-6)

    def test_span_attributes_to_dict(self):
        """Test SpanAttributes.to_dict()."""
        from fi.evals.otel import SpanAttributes

        attrs = SpanAttributes(
            system="openai",
            request_model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            temperature=0.7,
            cost_total_usd=0.01,
        )

        d = attrs.to_dict()

        assert d["gen_ai.system"] == "openai"
        assert d["gen_ai.request.model"] == "gpt-4"
        assert d["gen_ai.usage.input_tokens"] == 100
        assert d["gen_ai.usage.output_tokens"] == 50
        assert d["gen_ai.request.temperature"] == 0.7
        assert d["llm.cost.total_usd"] == 0.01

    def test_evaluation_result(self):
        """Test EvaluationResult dataclass."""
        from fi.evals.otel import EvaluationResult

        result = EvaluationResult(
            metric="relevance",
            score=0.85,
            reason="Response is highly relevant",
            latency_ms=150.0,
        )

        assert result.metric == "relevance"
        assert result.score == 0.85
        assert result.reason == "Response is highly relevant"
        assert result.latency_ms == 150.0

    def test_trace_context(self):
        """Test TraceContext dataclass."""
        from fi.evals.otel import TraceContext

        ctx = TraceContext(
            trace_id="abc123",
            span_id="def456",
            parent_span_id="parent123",
        )

        assert ctx.is_valid
        assert ctx.is_sampled  # Default trace_flags=1

        # Test invalid context
        invalid_ctx = TraceContext(trace_id="", span_id="")
        assert not invalid_ctx.is_valid


class TestSemanticConventions:
    """Tests for OTEL semantic conventions."""

    def test_genai_attributes(self):
        """Test GenAIAttributes constants."""
        from fi.evals.otel import GenAIAttributes

        assert GenAIAttributes.SYSTEM == "gen_ai.system"
        assert GenAIAttributes.OPERATION_NAME == "gen_ai.operation.name"
        assert GenAIAttributes.REQUEST_MODEL == "gen_ai.request.model"
        assert GenAIAttributes.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"

    def test_genai_attribute_templates(self):
        """Test GenAIAttributes template methods."""
        from fi.evals.otel import GenAIAttributes

        assert GenAIAttributes.prompt_content(0) == "gen_ai.prompt.0.content"
        assert GenAIAttributes.prompt_content(1) == "gen_ai.prompt.1.content"
        assert GenAIAttributes.prompt_role(0) == "gen_ai.prompt.0.role"
        assert GenAIAttributes.completion_content(0) == "gen_ai.completion.0.content"

    def test_evaluation_attributes(self):
        """Test EvaluationAttributes."""
        from fi.evals.otel import EvaluationAttributes

        assert EvaluationAttributes.score("relevance") == "eval.relevance"
        assert EvaluationAttributes.reason("relevance") == "eval.relevance.reason"
        assert EvaluationAttributes.latency("relevance") == "eval.relevance.latency_ms"

    def test_cost_attributes(self):
        """Test LLMCostAttributes."""
        from fi.evals.otel import LLMCostAttributes

        assert LLMCostAttributes.INPUT_COST_USD == "llm.cost.input_usd"
        assert LLMCostAttributes.OUTPUT_COST_USD == "llm.cost.output_usd"
        assert LLMCostAttributes.TOTAL_COST_USD == "llm.cost.total_usd"

    def test_normalize_system_name(self):
        """Test provider name normalization."""
        from fi.evals.otel import normalize_system_name

        # Direct mappings
        assert normalize_system_name("openai") == "openai"
        assert normalize_system_name("anthropic") == "anthropic"
        assert normalize_system_name("cohere") == "cohere"

        # Inferred from model names
        assert normalize_system_name("gpt-4") == "openai"
        assert normalize_system_name("claude-3") == "anthropic"
        assert normalize_system_name("gemini") == "google"
        assert normalize_system_name("mistral-large") == "mistral"
        assert normalize_system_name("llama-2") == "meta"

        # Case insensitive
        assert normalize_system_name("OPENAI") == "openai"
        assert normalize_system_name("Claude") == "anthropic"

        # Unknown -> custom
        assert normalize_system_name("unknown-provider") == "custom"

    def test_create_llm_span_attributes(self):
        """Test helper function for creating LLM span attributes."""
        from fi.evals.otel import create_llm_span_attributes, GenAIAttributes

        attrs = create_llm_span_attributes(
            system="openai",
            model="gpt-4",
            operation="chat",
            input_tokens=100,
            output_tokens=50,
            temperature=0.7,
        )

        assert attrs[GenAIAttributes.SYSTEM] == "openai"
        assert attrs[GenAIAttributes.REQUEST_MODEL] == "gpt-4"
        assert attrs[GenAIAttributes.OPERATION_NAME] == "chat"
        assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 100
        assert attrs[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 50
        assert attrs[GenAIAttributes.USAGE_TOTAL_TOKENS] == 150
        assert attrs[GenAIAttributes.REQUEST_TEMPERATURE] == 0.7

    def test_create_evaluation_attributes(self):
        """Test helper function for creating evaluation attributes."""
        from fi.evals.otel import create_evaluation_attributes, EvaluationAttributes

        attrs = create_evaluation_attributes(
            metric="relevance",
            score=0.85,
            reason="Good match",
            latency_ms=100.0,
        )

        assert attrs[EvaluationAttributes.score("relevance")] == 0.85
        assert attrs[EvaluationAttributes.reason("relevance")] == "Good match"
        assert attrs[EvaluationAttributes.latency("relevance")] == 100.0

    def test_span_names(self):
        """Test standard span names."""
        from fi.evals.otel import SpanNames

        assert SpanNames.LLM_CHAT == "llm.chat"
        assert SpanNames.LLM_COMPLETION == "llm.completion"
        assert SpanNames.LLM_EMBEDDING == "llm.embedding"
        assert SpanNames.RAG_RETRIEVE == "rag.retrieve"
        assert SpanNames.AGENT_STEP == "agent.step"


class TestConfiguration:
    """Tests for OTEL configuration."""

    def test_exporter_config(self):
        """Test ExporterConfig dataclass."""
        from fi.evals.otel import ExporterConfig, ExporterType

        config = ExporterConfig(
            type=ExporterType.OTLP_GRPC,
            endpoint="localhost:4317",
            headers={"Authorization": "Bearer token"},
            timeout_ms=5000,
        )

        assert config.type == ExporterType.OTLP_GRPC
        assert config.endpoint == "localhost:4317"
        assert config.headers["Authorization"] == "Bearer token"
        assert config.timeout_ms == 5000

    def test_trace_config_defaults(self):
        """Test TraceConfig default values."""
        from fi.evals.otel import TraceConfig, ExporterType, SamplingStrategy

        config = TraceConfig()

        assert config.service_name == "llm-service"
        assert config.deployment_environment == "development"
        assert config.enabled is True
        assert config.sampling_strategy == SamplingStrategy.ALWAYS_ON
        assert len(config.exporters) == 1
        assert config.exporters[0].type == ExporterType.CONSOLE

    def test_trace_config_development(self):
        """Test TraceConfig.development() factory."""
        from fi.evals.otel import TraceConfig, ExporterType

        config = TraceConfig.development(service_name="test-service")

        assert config.service_name == "test-service"
        assert config.deployment_environment == "development"
        assert config.debug is True
        assert config.evaluation.sample_rate == 1.0
        assert config.evaluation.async_evaluation is False

    def test_trace_config_production(self):
        """Test TraceConfig.production() factory."""
        from fi.evals.otel import TraceConfig, ExporterType, SamplingStrategy

        config = TraceConfig.production(
            service_name="prod-service",
            otlp_endpoint="otlp.example.com:4317",
            service_version="1.0.0",
            eval_sample_rate=0.1,
        )

        assert config.service_name == "prod-service"
        assert config.service_version == "1.0.0"
        assert config.deployment_environment == "production"
        assert config.exporters[0].type == ExporterType.OTLP_GRPC
        assert config.exporters[0].endpoint == "otlp.example.com:4317"
        assert config.evaluation.sample_rate == 0.1
        assert config.sampling_strategy == SamplingStrategy.RATIO
        assert config.content.redact_pii is True

    def test_trace_config_multi_backend(self):
        """Test TraceConfig.multi_backend() factory."""
        from fi.evals.otel import TraceConfig, ExporterType

        config = TraceConfig.multi_backend(
            service_name="multi-service",
            backends=[
                {"type": "otlp_grpc", "endpoint": "localhost:4317"},
                {"type": "console"},
                {"type": "jaeger", "endpoint": "localhost:6831"},
            ],
        )

        assert config.service_name == "multi-service"
        assert len(config.exporters) == 3
        assert config.exporters[0].type == ExporterType.OTLP_GRPC
        assert config.exporters[1].type == ExporterType.CONSOLE
        assert config.exporters[2].type == ExporterType.JAEGER

    def test_trace_config_chaining(self):
        """Test TraceConfig method chaining."""
        from fi.evals.otel import TraceConfig, ExporterType

        config = (
            TraceConfig(service_name="chained-service")
            .add_exporter(ExporterType.OTLP_GRPC, endpoint="localhost:4317")
            .with_evaluation(metrics=["relevance", "coherence"], sample_rate=0.5)
            .with_cost_tracking(alert_threshold=1.0)
        )

        assert config.service_name == "chained-service"
        assert len(config.exporters) == 2  # Console (default) + OTLP
        assert config.evaluation.metrics == ["relevance", "coherence"]
        assert config.evaluation.sample_rate == 0.5
        assert config.cost.enabled is True
        assert config.cost.alert_threshold_usd == 1.0

    def test_exporter_presets(self):
        """Test exporter presets."""
        from fi.evals.otel import get_exporter_preset, ExporterType

        jaeger = get_exporter_preset("jaeger")
        assert jaeger.type == ExporterType.JAEGER
        assert "6831" in jaeger.endpoint

        with pytest.raises(ValueError):
            get_exporter_preset("unknown-preset")


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_calculate_cost_openai(self):
        """Test cost calculation for OpenAI models."""
        from fi.evals.otel import calculate_cost

        cost = calculate_cost(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
        )

        # gpt-4: $0.03/1K input, $0.06/1K output
        assert cost["input_cost"] == pytest.approx(0.03, rel=0.01)
        assert cost["output_cost"] == pytest.approx(0.03, rel=0.01)
        assert cost["total_cost"] == pytest.approx(0.06, rel=0.01)

    def test_calculate_cost_anthropic(self):
        """Test cost calculation for Anthropic models."""
        from fi.evals.otel import calculate_cost

        cost = calculate_cost(
            model="claude-3-sonnet-20240229",
            input_tokens=1000,
            output_tokens=500,
        )

        # Claude 3 Sonnet: $0.003/1K input, $0.015/1K output
        assert cost["input_cost"] == pytest.approx(0.003, rel=0.01)
        assert cost["output_cost"] == pytest.approx(0.0075, rel=0.01)

    def test_calculate_cost_with_custom_pricing(self):
        """Test cost calculation with custom pricing."""
        from fi.evals.otel import calculate_cost, TokenPricing

        custom_pricing = {
            "my-model": TokenPricing("my-model", 0.01, 0.02),
        }

        cost = calculate_cost(
            model="my-model",
            input_tokens=1000,
            output_tokens=1000,
            custom_pricing=custom_pricing,
        )

        assert cost["input_cost"] == pytest.approx(0.01, rel=0.01)
        assert cost["output_cost"] == pytest.approx(0.02, rel=0.01)
        assert cost["total_cost"] == pytest.approx(0.03, rel=0.01)

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model."""
        from fi.evals.otel import calculate_cost

        cost = calculate_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should return zeros for unknown model
        assert cost["input_cost"] == 0.0
        assert cost["output_cost"] == 0.0
        assert cost["total_cost"] == 0.0

    def test_calculate_cost_model_alias(self):
        """Test cost calculation with model aliases."""
        from fi.evals.otel import calculate_cost

        # These should resolve to the same pricing
        cost1 = calculate_cost("gpt-4", 1000, 500)
        cost2 = calculate_cost("gpt-4-0613", 1000, 500)

        assert cost1["total_cost"] == cost2["total_cost"]


class TestProcessors:
    """Tests for span processors."""

    def test_base_processor_lifecycle(self):
        """Test BaseSpanProcessor enable/disable."""
        from fi.evals.otel import LLMSpanProcessor

        processor = LLMSpanProcessor()

        assert processor.enabled is True

        processor.disable()
        assert processor.enabled is False

        processor.enable()
        assert processor.enabled is True

        processor.shutdown()
        assert processor.enabled is False

    def test_composite_processor(self):
        """Test CompositeSpanProcessor."""
        from fi.evals.otel import CompositeSpanProcessor, LLMSpanProcessor, CostSpanProcessor

        llm_proc = LLMSpanProcessor()
        cost_proc = CostSpanProcessor()

        composite = CompositeSpanProcessor([llm_proc, cost_proc])

        assert len(composite._processors) == 2

        # Add another
        from fi.evals.otel import EvaluationSpanProcessor
        eval_proc = EvaluationSpanProcessor(metrics=["relevance"])
        composite.add_processor(eval_proc)
        assert len(composite._processors) == 3

        # Remove
        composite.remove_processor(eval_proc)
        assert len(composite._processors) == 2

    def test_filtering_processor(self):
        """Test FilteringSpanProcessor."""
        from fi.evals.otel import FilteringSpanProcessor, LLMSpanProcessor

        # Create a mock span
        mock_span = MagicMock()
        mock_span.attributes = {"gen_ai.system": "openai"}

        # Filter for OpenAI only
        delegate = MagicMock()

        processor = FilteringSpanProcessor(
            filter_fn=lambda s: s.attributes.get("gen_ai.system") == "openai",
            delegate=delegate,
        )

        # Should process OpenAI span
        assert processor.should_process(mock_span) is True

        # Change to Anthropic - should not process
        mock_span.attributes = {"gen_ai.system": "anthropic"}
        assert processor.should_process(mock_span) is False

    def test_llm_processor_pattern_matching(self):
        """Test LLMSpanProcessor span detection."""
        from fi.evals.otel import LLMSpanProcessor

        processor = LLMSpanProcessor()

        # Create mock spans
        llm_span = MagicMock()
        llm_span.name = "openai.chat.completions.create"

        non_llm_span = MagicMock()
        non_llm_span.name = "http.request"

        assert processor.should_process(llm_span) is True
        assert processor.should_process(non_llm_span) is False

    def test_cost_processor_tracking(self):
        """Test CostSpanProcessor cumulative tracking."""
        from fi.evals.otel import CostSpanProcessor, GenAIAttributes

        processor = CostSpanProcessor()

        # Create mock span with token usage
        mock_span = MagicMock()
        mock_span.attributes = {
            GenAIAttributes.REQUEST_MODEL: "gpt-4",
            GenAIAttributes.USAGE_INPUT_TOKENS: 100,
            GenAIAttributes.USAGE_OUTPUT_TOKENS: 50,
        }
        mock_span.get_span_context.return_value = MagicMock(
            trace_id=12345,
            span_id=67890,
        )

        processor.on_end(mock_span)

        assert processor.total_calls == 1
        assert processor.total_cost_usd > 0

        # Get summary
        summary = processor.get_summary()
        assert summary["total_calls"] == 1
        assert summary["currency"] == "USD"

        # Reset
        processor.reset_totals()
        assert processor.total_calls == 0
        assert processor.total_cost_usd == 0.0

    def test_cost_processor_alert(self):
        """Test CostSpanProcessor cost alerting."""
        from fi.evals.otel import CostSpanProcessor, GenAIAttributes

        alert_callback = MagicMock()

        processor = CostSpanProcessor(
            alert_threshold_usd=0.001,  # Very low threshold
            on_cost_alert=alert_callback,
        )

        mock_span = MagicMock()
        mock_span.attributes = {
            GenAIAttributes.REQUEST_MODEL: "gpt-4",
            GenAIAttributes.USAGE_INPUT_TOKENS: 1000,  # Will exceed threshold
            GenAIAttributes.USAGE_OUTPUT_TOKENS: 1000,
        }
        mock_span.get_span_context.return_value = MagicMock(
            trace_id=12345,
            span_id=67890,
        )

        processor.on_end(mock_span)

        # Alert should have been called
        alert_callback.assert_called_once()

    def test_evaluation_processor_caching(self):
        """Test EvaluationSpanProcessor result caching."""
        from fi.evals.otel import EvaluationSpanProcessor

        processor = EvaluationSpanProcessor(
            metrics=["relevance"],
            cache_enabled=True,
            cache_ttl_seconds=60,
        )

        # Compute cache key
        key1 = processor._compute_cache_key("prompt1", "completion1", ["relevance"])
        key2 = processor._compute_cache_key("prompt1", "completion1", ["relevance"])
        key3 = processor._compute_cache_key("prompt2", "completion1", ["relevance"])

        # Same inputs should produce same key
        assert key1 == key2
        # Different inputs should produce different key
        assert key1 != key3

    def test_evaluation_processor_sampling(self):
        """Test EvaluationSpanProcessor sampling."""
        from fi.evals.otel import EvaluationSpanProcessor, GenAIAttributes

        processor = EvaluationSpanProcessor(
            metrics=["relevance"],
            sample_rate=0.0,  # 0% sampling
        )

        mock_span = MagicMock()
        mock_span.attributes = {
            GenAIAttributes.SYSTEM: "openai",
            GenAIAttributes.completion_content(0): "test response",
        }

        # With 0% sampling, should never process
        # (statistically, over many runs this holds)
        processed = sum(
            1 for _ in range(100)
            if processor.should_process(mock_span)
        )
        assert processed == 0


class TestInstrumentors:
    """Tests for LLM client instrumentors."""

    def test_instrumentor_manager(self):
        """Test InstrumentorManager."""
        from fi.evals.otel import InstrumentorManager, OpenAIInstrumentor

        manager = InstrumentorManager()

        # Add instrumentor
        openai_inst = OpenAIInstrumentor()
        manager.add(openai_inst)

        # Get instrumentor
        assert manager.get("openai") is openai_inst

        # Remove instrumentor
        manager.remove("openai")
        assert manager.get("openai") is None

    def test_openai_instrumentor_creation(self):
        """Test OpenAIInstrumentor initialization."""
        from fi.evals.otel import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor(
            capture_prompts=True,
            capture_completions=True,
            capture_streaming=False,
        )

        assert instrumentor.system_name == "openai"
        assert instrumentor.library_name == "openai"
        assert instrumentor.is_instrumented is False

    def test_anthropic_instrumentor_creation(self):
        """Test AnthropicInstrumentor initialization."""
        from fi.evals.otel import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor(
            capture_prompts=True,
            capture_completions=True,
        )

        assert instrumentor.system_name == "anthropic"
        assert instrumentor.library_name == "anthropic"
        assert instrumentor.is_instrumented is False

    def test_convenience_functions(self):
        """Test instrumentor convenience functions."""
        from fi.evals.otel import (
            get_instrumented_libraries,
            is_instrumented,
        )

        # Initially nothing instrumented (without actually importing libs)
        libs = get_instrumented_libraries()
        assert isinstance(libs, list)

        # Check specific library
        assert is_instrumented("openai") is False
        assert is_instrumented("anthropic") is False


class TestTracerSetup:
    """Tests for tracer setup functions."""

    def test_trace_config_resource_sync(self):
        """Test that TraceConfig syncs resource attributes."""
        from fi.evals.otel import TraceConfig

        config = TraceConfig(
            service_name="test-service",
            service_version="1.0.0",
            deployment_environment="staging",
        )

        # Resource should be synced
        assert config.resource.service_name == "test-service"
        assert config.resource.service_version == "1.0.0"
        assert config.resource.deployment_environment == "staging"

    def test_default_pricing_coverage(self):
        """Test that DEFAULT_PRICING covers common models."""
        from fi.evals.otel import DEFAULT_PRICING

        # OpenAI models
        assert "gpt-4" in DEFAULT_PRICING
        assert "gpt-4o" in DEFAULT_PRICING
        assert "gpt-3.5-turbo" in DEFAULT_PRICING

        # Anthropic models
        assert "claude-3-opus-20240229" in DEFAULT_PRICING
        assert "claude-3-sonnet-20240229" in DEFAULT_PRICING
        assert "claude-3-haiku-20240307" in DEFAULT_PRICING

        # Google models
        assert "gemini-1.5-pro" in DEFAULT_PRICING
        assert "gemini-pro" in DEFAULT_PRICING

        # Embedding models
        assert "text-embedding-3-large" in DEFAULT_PRICING
        assert "text-embedding-ada-002" in DEFAULT_PRICING


class TestIntegration:
    """Integration tests for OTEL module."""

    def test_full_import(self):
        """Test that all exports are importable."""
        from fi.evals.otel import (
            # Core setup
            setup_tracing,
            get_tracer,
            trace_llm_call,

            # Configuration
            TraceConfig,
            ExporterConfig,
            ExporterType,

            # Types
            SpanAttributes,
            EvaluationResult,
            TokenPricing,

            # Conventions
            GenAIAttributes,
            normalize_system_name,

            # Processors
            LLMSpanProcessor,
            EvaluationSpanProcessor,
            CostSpanProcessor,
            CompositeSpanProcessor,

            # Instrumentors
            OpenAIInstrumentor,
            AnthropicInstrumentor,
            instrument_all,

            # Constants
            SYSTEM_OPENAI,
            OPERATION_CHAT,
            FINISH_STOP,
        )

        # All imports should work
        assert True

    def test_module_version(self):
        """Test module has version."""
        from fi.evals.otel import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_otel_available_flag(self):
        """Test OTEL_AVAILABLE flag."""
        from fi.evals.otel import OTEL_AVAILABLE

        # Should be a boolean
        assert isinstance(OTEL_AVAILABLE, bool)

    def test_end_to_end_config_to_attributes(self):
        """Test end-to-end flow from config to attributes."""
        from fi.evals.otel import (
            TraceConfig,
            SpanAttributes,
            create_llm_span_attributes,
            calculate_cost,
            GenAIAttributes,
        )

        # Create production config
        config = TraceConfig.production(
            service_name="e2e-test",
            otlp_endpoint="localhost:4317",
            eval_sample_rate=0.5,
        )

        # Simulate an LLM call
        span_attrs = create_llm_span_attributes(
            system="openai",
            model="gpt-4",
            operation="chat",
            input_tokens=500,
            output_tokens=200,
            temperature=0.7,
            finish_reason="stop",
        )

        # Calculate cost
        cost = calculate_cost("gpt-4", 500, 200)

        # Verify attributes
        assert span_attrs[GenAIAttributes.SYSTEM] == "openai"
        assert span_attrs[GenAIAttributes.REQUEST_MODEL] == "gpt-4"
        assert span_attrs[GenAIAttributes.USAGE_TOTAL_TOKENS] == 700
        assert cost["total_cost"] > 0

        # Create SpanAttributes object
        attrs = SpanAttributes(
            system="openai",
            request_model="gpt-4",
            input_tokens=500,
            output_tokens=200,
            cost_total_usd=cost["total_cost"],
        )

        d = attrs.to_dict()
        assert "gen_ai.system" in d
        assert "llm.cost.total_usd" in d


class TestAutoEnrichment:
    """Tests for automatic span enrichment."""

    def test_enrichment_enabled_by_default(self):
        """Test that auto-enrichment is enabled by default."""
        from fi.evals.otel import is_auto_enrichment_enabled

        assert is_auto_enrichment_enabled() is True

    def test_enable_disable_enrichment(self):
        """Test enabling/disabling auto-enrichment."""
        from fi.evals.otel import (
            enable_auto_enrichment,
            disable_auto_enrichment,
            is_auto_enrichment_enabled,
        )

        # Disable
        disable_auto_enrichment()
        assert is_auto_enrichment_enabled() is False

        # Re-enable
        enable_auto_enrichment()
        assert is_auto_enrichment_enabled() is True

    def test_enrich_span_with_evaluation(self):
        """Test enriching span with evaluation data."""
        from fi.evals.otel import enrich_span_with_evaluation

        # Without an active span, should return False but not error
        result = enrich_span_with_evaluation(
            metric_name="relevance",
            score=0.85,
            reason="Good response",
            latency_ms=100.0,
        )

        # Should gracefully handle no active span
        assert result is False  # No span to enrich

    def test_enrich_span_with_eval_result(self):
        """Test enriching span with EvalResult object."""
        from fi.evals.otel import enrich_span_with_eval_result

        # Create a mock EvalResult
        class MockEvalResult:
            name = "coherence"
            output = 0.9
            reason = "Well structured"
            runtime = 50

        result = enrich_span_with_eval_result(MockEvalResult())
        # Should handle gracefully even without active span
        assert result is False  # No span to enrich

    def test_enrich_span_with_batch_result(self):
        """Test enriching span with BatchRunResult."""
        from fi.evals.otel import enrich_span_with_batch_result

        # Create a mock BatchRunResult
        class MockEvalResult:
            def __init__(self, name, output, reason):
                self.name = name
                self.output = output
                self.reason = reason
                self.runtime = 10

        class MockBatchResult:
            eval_results = [
                MockEvalResult("relevance", 0.8, "Good"),
                MockEvalResult("coherence", 0.9, "Clear"),
            ]

        count = enrich_span_with_batch_result(MockBatchResult())
        # Should handle gracefully even without active span
        assert count == 0  # No span to enrich

    def test_evaluation_span_context(self):
        """Test EvaluationSpanContext manager."""
        from fi.evals.otel import EvaluationSpanContext

        with EvaluationSpanContext("test_metric") as ctx:
            # Simulate evaluation work
            result = ctx.record_result(score=0.75, reason="Test")

        # Should not error even without OTEL
        assert True

    def test_enrichment_with_bool_score(self):
        """Test enrichment handles boolean scores."""
        from fi.evals.otel import enrich_span_with_evaluation

        # Boolean True -> 1.0
        result = enrich_span_with_evaluation("is_valid", True)
        assert result is False  # No span, but shouldn't error

        # Boolean False -> 0.0
        result = enrich_span_with_evaluation("is_valid", False)
        assert result is False


class TestRealWorldScenarios:
    """Real-world scenario tests."""

    def test_multi_provider_cost_tracking(self):
        """Test tracking costs across multiple providers."""
        from fi.evals.otel import CostSpanProcessor, GenAIAttributes

        processor = CostSpanProcessor()

        # Simulate OpenAI call
        openai_span = MagicMock()
        openai_span.attributes = {
            GenAIAttributes.REQUEST_MODEL: "gpt-4",
            GenAIAttributes.USAGE_INPUT_TOKENS: 1000,
            GenAIAttributes.USAGE_OUTPUT_TOKENS: 500,
        }
        openai_span.get_span_context.return_value = MagicMock(trace_id=1, span_id=1)

        # Simulate Anthropic call
        anthropic_span = MagicMock()
        anthropic_span.attributes = {
            GenAIAttributes.REQUEST_MODEL: "claude-3-sonnet-20240229",
            GenAIAttributes.USAGE_INPUT_TOKENS: 1000,
            GenAIAttributes.USAGE_OUTPUT_TOKENS: 500,
        }
        anthropic_span.get_span_context.return_value = MagicMock(trace_id=2, span_id=2)

        processor.on_end(openai_span)
        processor.on_end(anthropic_span)

        summary = processor.get_summary()
        assert summary["total_calls"] == 2
        assert summary["total_cost_usd"] > 0

    def test_evaluation_workflow(self):
        """Test evaluation processor workflow."""
        from fi.evals.otel import (
            EvaluationSpanProcessor,
            EvaluationResult,
            GenAIAttributes,
        )

        results_collected = []

        def callback(span_id, results):
            results_collected.extend(results)

        processor = EvaluationSpanProcessor(
            metrics=["relevance"],
            sample_rate=1.0,
            async_evaluation=False,
            cache_enabled=False,
            on_evaluation_complete=callback,
        )

        # Mock span with content
        mock_span = MagicMock()
        mock_span.attributes = {
            GenAIAttributes.SYSTEM: "openai",
            GenAIAttributes.prompt_content(0): "What is the capital of France?",
            GenAIAttributes.completion_content(0): "The capital of France is Paris.",
        }
        mock_span.get_span_context.return_value = MagicMock(trace_id=123, span_id=456)

        # Note: Actual evaluation would require fi.evals.Evaluator
        # This test verifies the workflow structure
        assert processor.should_process(mock_span) is True

    def test_content_redaction(self):
        """Test content redaction in LLM processor."""
        from fi.evals.otel import LLMSpanProcessor

        processor = LLMSpanProcessor(
            redact_patterns=[
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                r"sk-[a-zA-Z0-9]{48}",  # API key pattern
            ],
        )

        # Test content processing
        content = "Contact me at user@example.com with key sk-" + "a" * 48
        processed = processor._process_content(content)

        assert "user@example.com" not in processed
        assert "[REDACTED]" in processed

    def test_attribute_extraction_resilience(self):
        """Test processor resilience to malformed data."""
        from fi.evals.otel import LLMSpanProcessor

        processor = LLMSpanProcessor()

        # Test with various malformed inputs
        test_cases = [
            {},  # Empty
            {"gen_ai.system": None},  # None value
            {"gen_ai.usage.input_tokens": "not_a_number"},  # Invalid type
            {"gen_ai.request.temperature": "0.7"},  # String instead of float
        ]

        for attrs in test_cases:
            # Should not raise
            result = processor._extract_llm_attributes(attrs)
            assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
