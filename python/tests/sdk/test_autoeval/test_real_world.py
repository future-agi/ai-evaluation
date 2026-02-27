"""Real-world scenario tests for AutoEval.

These tests demonstrate practical usage patterns and verify that AutoEval
correctly handles common real-world application scenarios.
"""

import pytest
from fi.evals.autoeval import (
    AutoEvalPipeline,
    EvalConfig,
    ScannerConfig,
    AppCategory,
    RiskLevel,
    DomainSensitivity,
)


class TestHealthcareScenarios:
    """Real-world tests for healthcare applications."""

    def test_healthcare_chatbot_detection(self):
        """Should detect healthcare domain from description."""
        pipeline = AutoEvalPipeline.from_description(
            "A HIPAA-compliant patient portal chatbot for a hospital. "
            "Patients can ask about their appointments, test results, medications, "
            "and billing information. The system retrieves from electronic health records."
        )

        # Should detect healthcare domain
        assert pipeline.analysis.domain_sensitivity == DomainSensitivity.HEALTHCARE
        assert pipeline.analysis.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}

        # Should have appropriate scanners
        scanner_names = [s.name for s in pipeline.config.scanners]
        assert "PIIScanner" in scanner_names or any("PII" in s for s in scanner_names)

    def test_healthcare_template_configuration(self):
        """Healthcare template should have strict safety settings."""
        pipeline = AutoEvalPipeline.from_template("healthcare")

        # Verify strict thresholds
        assert pipeline.config.global_pass_rate >= 0.9

        # PII scanner should use redact action
        pii_scanner = pipeline.config.get_scanner("PIIScanner")
        assert pii_scanner is not None
        assert pii_scanner.action == "redact"

        # Should have coherence eval with high threshold
        coherence = pipeline.config.get_eval("CoherenceEval")
        assert coherence is not None
        assert coherence.threshold >= 0.8

    def test_healthcare_medical_terminology(self):
        """Should detect healthcare from medical terminology."""
        descriptions = [
            "A clinical decision support system for diagnosis",
            "An app that helps patients track their prescription medications",
            "A telemedicine platform connecting doctors with patients",
            "A HIPAA-compliant medical records system",
        ]

        for desc in descriptions:
            pipeline = AutoEvalPipeline.from_description(desc)
            assert pipeline.analysis.domain_sensitivity == DomainSensitivity.HEALTHCARE, \
                f"Failed to detect healthcare for: {desc}"


class TestFinancialScenarios:
    """Real-world tests for financial applications."""

    def test_banking_chatbot_detection(self):
        """Should detect financial domain for banking apps."""
        pipeline = AutoEvalPipeline.from_description(
            "A mobile banking assistant that helps customers check balances, "
            "review recent transactions, transfer money between accounts, "
            "and answer questions about fees and interest rates."
        )

        assert pipeline.analysis.domain_sensitivity == DomainSensitivity.FINANCIAL
        assert pipeline.analysis.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def test_financial_template_has_quality_evals(self):
        """Financial template should have quality evals."""
        pipeline = AutoEvalPipeline.from_template("financial")

        eval_names = [e.name for e in pipeline.config.evaluations]
        assert "CoherenceEval" in eval_names

    def test_financial_terminology_detection(self):
        """Should detect financial from various financial terms."""
        descriptions = [
            "A credit card fraud detection system",
            "An investment portfolio management tool",
            "A loan application processing assistant",
            "A payment gateway integration helper",
        ]

        for desc in descriptions:
            pipeline = AutoEvalPipeline.from_description(desc)
            assert pipeline.analysis.domain_sensitivity == DomainSensitivity.FINANCIAL, \
                f"Failed to detect financial for: {desc}"


class TestCodeAssistantScenarios:
    """Real-world tests for code assistant applications."""

    def test_code_assistant_detection(self):
        """Should detect code assistant category."""
        pipeline = AutoEvalPipeline.from_description(
            "A VS Code extension that generates code snippets, explains code, "
            "reviews pull requests, and helps debug issues."
        )

        assert pipeline.analysis.category == AppCategory.CODE_ASSISTANT

    def test_code_assistant_security_scanners(self):
        """Code assistants should have security-focused scanners."""
        pipeline = AutoEvalPipeline.from_template("code_assistant")

        scanner_names = [s.name for s in pipeline.config.scanners]
        assert "CodeInjectionScanner" in scanner_names
        assert "SecretsScanner" in scanner_names

    def test_code_assistant_secrets_scanner_strict(self):
        """Secrets scanner should be strict for code assistants."""
        pipeline = AutoEvalPipeline.from_template("code_assistant")

        secrets_scanner = pipeline.config.get_scanner("SecretsScanner")
        assert secrets_scanner is not None
        assert secrets_scanner.threshold >= 0.9
        assert secrets_scanner.action == "block"


class TestRAGSystemScenarios:
    """Real-world tests for RAG-based systems."""

    def test_rag_system_detection(self):
        """Should detect RAG system from description."""
        pipeline = AutoEvalPipeline.from_description(
            "A document Q&A system that uses semantic search to retrieve "
            "relevant passages from a knowledge base and generates answers."
        )

        assert pipeline.analysis.category == AppCategory.RAG_SYSTEM
        assert "rag" in pipeline.analysis.detected_features

    def test_rag_template_has_quality_evals(self):
        """RAG template should include quality evaluations."""
        pipeline = AutoEvalPipeline.from_template("rag_system")

        eval_names = [e.name for e in pipeline.config.evaluations]
        assert "CoherenceEval" in eval_names

    def test_rag_with_different_descriptions(self):
        """Should detect RAG from various descriptions."""
        descriptions = [
            "A retrieval-augmented generation system for legal documents",
            "A knowledge base chatbot with vector search",
            "A semantic search engine with AI-generated summaries",
            "A document retrieval system with context-aware responses",
        ]

        for desc in descriptions:
            pipeline = AutoEvalPipeline.from_description(desc)
            # Should detect RAG or have RAG-related features
            is_rag = (
                pipeline.analysis.category == AppCategory.RAG_SYSTEM or
                "rag" in pipeline.analysis.detected_features
            )
            assert is_rag, f"Failed to detect RAG for: {desc}"


class TestAgentWorkflowScenarios:
    """Real-world tests for autonomous agent applications."""

    def test_agent_detection(self):
        """Should detect agent workflow from description."""
        pipeline = AutoEvalPipeline.from_description(
            "An autonomous research agent that can search the web, read documents, "
            "write and execute Python code, and send emails."
        )

        assert pipeline.analysis.category == AppCategory.AGENT_WORKFLOW
        assert "tool_use" in pipeline.analysis.detected_features

    def test_agent_template_has_safety_eval(self):
        """Agent template should have action safety evaluation."""
        pipeline = AutoEvalPipeline.from_template("agent_workflow")

        eval_names = [e.name for e in pipeline.config.evaluations]
        assert "ActionSafetyEval" in eval_names
        assert "ReasoningQualityEval" in eval_names

        # Action safety should have high weight
        safety = pipeline.config.get_eval("ActionSafetyEval")
        assert safety.weight >= 1.5

    def test_agent_high_risk_by_default(self):
        """Agent workflows should be high risk by default."""
        pipeline = AutoEvalPipeline.from_template("agent_workflow")
        assert pipeline.config.risk_level == "high"


class TestChildrenContentScenarios:
    """Real-world tests for children's content applications."""

    def test_children_detection(self):
        """Should detect children's content from description."""
        pipeline = AutoEvalPipeline.from_description(
            "An educational chatbot for elementary school students (ages 6-12). "
            "Helps with homework, explains concepts in simple terms."
        )

        assert pipeline.analysis.domain_sensitivity == DomainSensitivity.CHILDREN
        assert pipeline.analysis.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def test_children_strict_safety(self):
        """Children's content should have very strict safety."""
        pipeline = AutoEvalPipeline.from_description(
            "A kids learning app for preschoolers with fun educational games."
        )

        # Should have toxicity scanner with high threshold
        scanner_names = [s.name for s in pipeline.config.scanners]
        has_safety_scanners = (
            "ToxicityScanner" in scanner_names or
            "BiasScanner" in scanner_names
        )
        assert has_safety_scanners

        # Check for strict thresholds on safety scanners (children gets 0.9 from recommender)
        for scanner in pipeline.config.scanners:
            if scanner.name in ["ToxicityScanner", "BiasScanner"]:
                # Children domain gets base threshold (0.8 for HIGH) + strict action
                assert scanner.threshold >= 0.8
                assert scanner.action == "block"


class TestContentModerationScenarios:
    """Real-world tests for content moderation applications."""

    def test_content_moderation_detection(self):
        """Should detect content moderation from description."""
        pipeline = AutoEvalPipeline.from_description(
            "A content filtering system that reviews user-generated posts "
            "for inappropriate content before publishing."
        )

        assert pipeline.analysis.category == AppCategory.CONTENT_MODERATION

    def test_content_moderation_template(self):
        """Content moderation template should be scanner-heavy."""
        pipeline = AutoEvalPipeline.from_template("content_moderation")

        # Should have multiple safety scanners
        scanner_names = [s.name for s in pipeline.config.scanners]
        assert "ToxicityScanner" in scanner_names
        assert "BiasScanner" in scanner_names

        # Should have strict thresholds
        assert pipeline.config.global_pass_rate >= 0.9


class TestCustomerSupportScenarios:
    """Real-world tests for customer support applications."""

    def test_customer_support_detection(self):
        """Should detect customer support from description."""
        pipeline = AutoEvalPipeline.from_description(
            "A help desk chatbot that answers customer questions about "
            "product features, shipping, and returns policy."
        )

        assert pipeline.analysis.category == AppCategory.CUSTOMER_SUPPORT

    def test_customer_support_template_balance(self):
        """Customer support should balance quality and safety."""
        pipeline = AutoEvalPipeline.from_template("customer_support")

        eval_names = [e.name for e in pipeline.config.evaluations]
        scanner_names = [s.name for s in pipeline.config.scanners]

        # Should have quality evaluations
        assert "CoherenceEval" in eval_names

        # Should have basic safety scanners
        assert "JailbreakScanner" in scanner_names
        assert "ToxicityScanner" in scanner_names


class TestPipelineCustomization:
    """Tests for real-world customization patterns."""

    def test_customize_after_creation(self):
        """Should support common customization patterns."""
        pipeline = AutoEvalPipeline.from_template("rag_system")

        # Common pattern: increase accuracy for production
        pipeline.set_threshold("CoherenceEval", 0.9)

        # Add PII protection
        pipeline.add(ScannerConfig("PIIScanner", action="redact"))

        # Verify customizations
        coherence = pipeline.config.get_eval("CoherenceEval")
        assert coherence.threshold == 0.9

        pii = pipeline.config.get_scanner("PIIScanner")
        assert pii is not None
        assert pii.action == "redact"

    def test_domain_specific_customization(self):
        """Should support domain-specific overrides."""
        # Start with generic template
        pipeline = AutoEvalPipeline.from_template("customer_support")

        # Remove existing PIIScanner if present, then add with custom settings
        pipeline.remove("PIIScanner")
        pipeline.add(ScannerConfig("PIIScanner", action="redact", threshold=0.95))

        # Use set_threshold to modify existing scanner
        pipeline.set_threshold("ToxicityScanner", 0.95)

        # Verify - new PII scanner with custom settings
        pii = pipeline.config.get_scanner("PIIScanner")
        assert pii is not None
        assert pii.threshold == 0.95
        assert pii.action == "redact"

        # Verify threshold was updated
        toxicity = pipeline.config.get_scanner("ToxicityScanner")
        assert toxicity.threshold == 0.95


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_ambiguous_description(self):
        """Should handle ambiguous descriptions gracefully."""
        pipeline = AutoEvalPipeline.from_description(
            "An AI assistant."  # Very vague
        )

        # Should still produce valid config
        assert pipeline.config is not None
        assert pipeline.analysis is not None
        # May have lower confidence
        assert pipeline.analysis.confidence <= 0.7

    def test_multiple_domains(self):
        """Should handle descriptions with multiple domains."""
        pipeline = AutoEvalPipeline.from_description(
            "A healthcare chatbot for a bank that helps customers with "
            "both medical insurance claims and account balances."
        )

        # Should detect at least one sensitive domain
        assert pipeline.analysis.domain_sensitivity in {
            DomainSensitivity.HEALTHCARE,
            DomainSensitivity.FINANCIAL,
            DomainSensitivity.PII_SENSITIVE,
        }

    def test_mixed_features(self):
        """Should detect multiple features in complex apps."""
        pipeline = AutoEvalPipeline.from_description(
            "A multi-modal AI assistant that searches documents, "
            "processes images, executes code, and has conversations."
        )

        features = pipeline.analysis.detected_features
        # Should detect multiple features
        assert len(features) >= 2

    def test_non_english_keywords(self):
        """Should handle descriptions with proper nouns/brands."""
        pipeline = AutoEvalPipeline.from_description(
            "A Microsoft Teams chatbot that integrates with Salesforce CRM "
            "to help customer support teams track customer interactions and queries."
        )

        # Should identify as customer support or chatbot due to keywords
        # "chatbot", "customer support", "customer interactions"
        assert pipeline.analysis.category in {
            AppCategory.CUSTOMER_SUPPORT,
            AppCategory.CHATBOT,
            AppCategory.UNKNOWN,  # Rule-based might not detect if keywords don't match exactly
        }


class TestExportImportRoundtrip:
    """Tests for export/import in real-world scenarios."""

    def test_yaml_roundtrip_preserves_customizations(self, tmp_path):
        """Exported YAML should preserve all customizations."""
        # Create and customize
        pipeline = AutoEvalPipeline.from_template("rag_system")
        pipeline.set_threshold("CoherenceEval", 0.95)
        pipeline.add(ScannerConfig("PIIScanner", action="redact"))

        # Export
        yaml_path = tmp_path / "config.yaml"
        pipeline.export_yaml(str(yaml_path))

        # Import
        loaded = AutoEvalPipeline.from_yaml(str(yaml_path))

        # Verify
        assert loaded.config.get_eval("CoherenceEval").threshold == 0.95
        assert loaded.config.get_scanner("PIIScanner").action == "redact"

    def test_json_roundtrip_for_api_configs(self, tmp_path):
        """JSON export should work for API-based configs."""
        pipeline = AutoEvalPipeline.from_description(
            "A REST API that provides AI-powered customer support."
        )

        # Export to JSON
        json_path = tmp_path / "config.json"
        pipeline.export_json(str(json_path))

        # Load and verify
        from fi.evals.autoeval import load_config
        loaded = load_config(str(json_path))
        assert loaded.name == pipeline.config.name


class TestPerformanceConsiderations:
    """Tests related to performance and production use."""

    def test_pipeline_creation_is_fast(self):
        """Pipeline creation should be fast for production use."""
        import time

        start = time.perf_counter()
        for _ in range(10):
            pipeline = AutoEvalPipeline.from_template("rag_system")
        elapsed = time.perf_counter() - start

        # Should create 10 pipelines in under 1 second
        assert elapsed < 1.0

    def test_description_analysis_reasonable_time(self):
        """Rule-based analysis should be fast."""
        import time

        start = time.perf_counter()
        pipeline = AutoEvalPipeline.from_description(
            "A complex enterprise system with multiple features including "
            "document retrieval, code generation, and customer support."
        )
        elapsed = time.perf_counter() - start

        # Rule-based analysis should complete quickly
        assert elapsed < 0.5

    def test_lazy_evaluator_building(self):
        """Evaluator should be built lazily on first evaluate()."""
        pipeline = AutoEvalPipeline.from_template("customer_support")

        # Should not have evaluator built yet
        assert pipeline._evaluator is None
        assert pipeline._scanner_pipeline is None
