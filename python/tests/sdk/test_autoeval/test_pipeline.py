"""Tests for AutoEvalPipeline."""

import pytest
from fi.evals.autoeval.pipeline import AutoEvalPipeline
from fi.evals.autoeval.config import AutoEvalConfig, EvalConfig, ScannerConfig
from fi.evals.autoeval.types import AppCategory, RiskLevel


class TestAutoEvalPipelineCreation:
    """Tests for creating AutoEvalPipeline instances."""

    def test_create_from_config(self):
        """Should create pipeline from config."""
        config = AutoEvalConfig(
            name="test_pipeline",
            evaluations=[EvalConfig(name="answer_relevancy")],
            scanners=[ScannerConfig(name="JailbreakScanner")],
        )
        pipeline = AutoEvalPipeline.from_config(config)
        assert pipeline.config.name == "test_pipeline"
        assert len(pipeline.config.evaluations) == 1
        assert len(pipeline.config.scanners) == 1

    def test_create_from_template(self):
        """Should create pipeline from template."""
        pipeline = AutoEvalPipeline.from_template("customer_support")
        assert pipeline.config.name == "customer_support"
        assert len(pipeline.config.evaluations) > 0
        assert len(pipeline.config.scanners) > 0

    def test_create_from_invalid_template_raises(self):
        """Should raise error for invalid template."""
        with pytest.raises(ValueError) as excinfo:
            AutoEvalPipeline.from_template("nonexistent")
        assert "not found" in str(excinfo.value).lower()

    def test_create_from_description_rule_based(self):
        """Should create pipeline from description using rule-based analysis."""
        pipeline = AutoEvalPipeline.from_description(
            "A customer support chatbot for healthcare.",
            llm_provider=None,  # Force rule-based
        )
        assert pipeline.config is not None
        assert pipeline.analysis is not None
        # Healthcare should be detected
        assert pipeline.analysis.domain_sensitivity.value in ["healthcare", "general"]


class TestAutoEvalPipelineCustomization:
    """Tests for customizing AutoEvalPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a basic pipeline for testing."""
        return AutoEvalPipeline.from_template("customer_support")

    def test_add_eval_config(self, pipeline):
        """Should add evaluation to pipeline."""
        initial_count = len(pipeline.config.evaluations)
        pipeline.add(EvalConfig(name="NewEval", threshold=0.9))
        assert len(pipeline.config.evaluations) == initial_count + 1

    def test_add_scanner_config(self, pipeline):
        """Should add scanner to pipeline."""
        initial_count = len(pipeline.config.scanners)
        pipeline.add(ScannerConfig(name="NewScanner", action="flag"))
        assert len(pipeline.config.scanners) == initial_count + 1

    def test_add_returns_self_for_chaining(self, pipeline):
        """Add should return self for method chaining."""
        result = pipeline.add(EvalConfig(name="TestEval"))
        assert result is pipeline

    def test_remove_evaluation(self, pipeline):
        """Should remove evaluation by name."""
        initial_count = len(pipeline.config.evaluations)
        # Get first eval name
        first_eval = pipeline.config.evaluations[0].name
        pipeline.remove(first_eval)
        assert len(pipeline.config.evaluations) == initial_count - 1

    def test_remove_scanner(self, pipeline):
        """Should remove scanner by name."""
        initial_count = len(pipeline.config.scanners)
        first_scanner = pipeline.config.scanners[0].name
        pipeline.remove(first_scanner)
        assert len(pipeline.config.scanners) == initial_count - 1

    def test_remove_returns_self_for_chaining(self, pipeline):
        """Remove should return self for method chaining."""
        first_eval = pipeline.config.evaluations[0].name
        result = pipeline.remove(first_eval)
        assert result is pipeline

    def test_set_threshold_for_eval(self, pipeline):
        """Should set threshold for evaluation."""
        first_eval = pipeline.config.evaluations[0].name
        pipeline.set_threshold(first_eval, 0.95)
        updated = pipeline.config.get_eval(first_eval)
        assert updated.threshold == 0.95

    def test_set_threshold_for_scanner(self, pipeline):
        """Should set threshold for scanner."""
        first_scanner = pipeline.config.scanners[0].name
        pipeline.set_threshold(first_scanner, 0.85)
        updated = pipeline.config.get_scanner(first_scanner)
        assert updated.threshold == 0.85

    def test_enable_disable_eval(self, pipeline):
        """Should enable/disable evaluations."""
        first_eval = pipeline.config.evaluations[0].name
        pipeline.disable(first_eval)
        assert pipeline.config.get_eval(first_eval).enabled is False

        pipeline.enable(first_eval)
        assert pipeline.config.get_eval(first_eval).enabled is True

    def test_method_chaining(self, pipeline):
        """Should support fluent method chaining."""
        result = (
            pipeline
            .add(EvalConfig(name="ChainedEval"))
            .set_threshold("ChainedEval", 0.9)
            .disable("ChainedEval")
        )
        assert result is pipeline
        eval_config = pipeline.config.get_eval("ChainedEval")
        assert eval_config is not None
        assert eval_config.threshold == 0.9
        assert eval_config.enabled is False


class TestAutoEvalPipelineExplain:
    """Tests for pipeline explanation."""

    def test_explain_returns_string(self):
        """Should return explanation string."""
        pipeline = AutoEvalPipeline.from_template("rag_system")
        explanation = pipeline.explain()
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explain_includes_config_info(self):
        """Explanation should include config info."""
        pipeline = AutoEvalPipeline.from_template("customer_support")
        explanation = pipeline.explain()
        assert "customer_support" in explanation

    def test_summary_returns_brief_string(self):
        """Should return brief summary."""
        pipeline = AutoEvalPipeline.from_template("code_assistant")
        summary = pipeline.summary()
        assert isinstance(summary, str)
        assert "code_assistant" in summary

    def test_repr_is_informative(self):
        """Repr should show useful info."""
        pipeline = AutoEvalPipeline.from_template("agent_workflow")
        repr_str = repr(pipeline)
        assert "AutoEvalPipeline" in repr_str
        assert "agent_workflow" in repr_str


class TestAutoEvalPipelineIntegration:
    """Integration tests for AutoEvalPipeline (no actual LLM/scanner calls)."""

    def test_full_workflow_from_template(self):
        """Should support full workflow from template."""
        # Create pipeline
        pipeline = AutoEvalPipeline.from_template("rag_system")

        # Customize
        pipeline.set_threshold("answer_relevancy", 0.9)
        pipeline.add(ScannerConfig(name="PIIScanner", action="redact"))

        # Verify config
        assert pipeline.config.get_eval("answer_relevancy").threshold == 0.9
        assert pipeline.config.get_scanner("PIIScanner") is not None

    def test_full_workflow_from_description(self):
        """Should support full workflow from description."""
        # Create from description (rule-based)
        pipeline = AutoEvalPipeline.from_description(
            "A document Q&A system using RAG for a financial company."
        )

        # Should detect relevant settings
        assert pipeline.analysis is not None

        # Should have recommendations
        assert len(pipeline.config.evaluations) > 0 or len(pipeline.config.scanners) > 0

        # Can customize
        pipeline.add(EvalConfig(name="CustomEval", threshold=0.8))
        assert pipeline.config.get_eval("CustomEval") is not None
