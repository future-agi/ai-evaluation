"""Tests for pre-built templates."""

import pytest
from fi.evals.autoeval.templates import (
    TEMPLATES,
    get_template,
    list_templates,
    get_template_names,
)
from fi.evals.autoeval.config import AutoEvalConfig


class TestTemplates:
    """Tests for template definitions."""

    def test_templates_exist(self):
        """Should have templates defined."""
        assert len(TEMPLATES) > 0

    def test_core_templates_defined(self):
        """Should have core templates defined."""
        expected = [
            "customer_support",
            "rag_system",
            "code_assistant",
            "content_moderation",
            "agent_workflow",
        ]
        for name in expected:
            assert name in TEMPLATES, f"Missing template: {name}"

    def test_all_templates_are_valid_configs(self):
        """All templates should be valid AutoEvalConfig instances."""
        for name, config in TEMPLATES.items():
            assert isinstance(config, AutoEvalConfig), f"Invalid config: {name}"
            assert config.name == name
            assert len(config.description) > 0

    def test_customer_support_template(self):
        """Customer support template should have appropriate config."""
        config = TEMPLATES["customer_support"]
        eval_names = [e.name for e in config.evaluations]
        scanner_names = [s.name for s in config.scanners]

        assert "CoherenceEval" in eval_names
        assert "JailbreakScanner" in scanner_names
        assert "ToxicityScanner" in scanner_names

    def test_rag_system_template(self):
        """RAG system template should have quality evals."""
        config = TEMPLATES["rag_system"]
        eval_names = [e.name for e in config.evaluations]

        assert "CoherenceEval" in eval_names

    def test_code_assistant_template(self):
        """Code assistant template should have security scanners."""
        config = TEMPLATES["code_assistant"]
        scanner_names = [s.name for s in config.scanners]

        assert "CodeInjectionScanner" in scanner_names
        assert "SecretsScanner" in scanner_names

    def test_content_moderation_template(self):
        """Content moderation template should have safety scanners."""
        config = TEMPLATES["content_moderation"]
        scanner_names = [s.name for s in config.scanners]

        assert "ToxicityScanner" in scanner_names
        assert "BiasScanner" in scanner_names
        # Content moderation is scanner-focused
        assert len(config.scanners) > 0

    def test_agent_workflow_template(self):
        """Agent workflow template should have safety evals."""
        config = TEMPLATES["agent_workflow"]
        eval_names = [e.name for e in config.evaluations]

        assert "ActionSafetyEval" in eval_names
        assert "ReasoningQualityEval" in eval_names


class TestGetTemplate:
    """Tests for get_template function."""

    def test_get_existing_template(self):
        """Should return copy of existing template."""
        config = get_template("customer_support")
        assert config is not None
        assert isinstance(config, AutoEvalConfig)
        assert config.name == "customer_support"

    def test_get_nonexistent_template(self):
        """Should return None for nonexistent template."""
        config = get_template("nonexistent_template")
        assert config is None

    def test_get_template_returns_copy(self):
        """Should return a copy, not the original."""
        config1 = get_template("rag_system")
        config2 = get_template("rag_system")

        # Modify one, shouldn't affect the other
        config1.evaluations.append(config1.evaluations[0].copy())
        assert len(config1.evaluations) != len(config2.evaluations)


class TestListTemplates:
    """Tests for list_templates function."""

    def test_list_templates_returns_dict(self):
        """Should return dictionary of names to descriptions."""
        templates = list_templates()
        assert isinstance(templates, dict)
        assert len(templates) > 0

    def test_list_templates_has_descriptions(self):
        """All templates should have descriptions."""
        templates = list_templates()
        for name, desc in templates.items():
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_list_templates_matches_templates(self):
        """Listed templates should match TEMPLATES keys."""
        templates = list_templates()
        assert set(templates.keys()) == set(TEMPLATES.keys())


class TestGetTemplateNames:
    """Tests for get_template_names function."""

    def test_get_template_names_returns_list(self):
        """Should return list of template names."""
        names = get_template_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_template_names_matches_templates(self):
        """Names should match TEMPLATES keys."""
        names = get_template_names()
        assert set(names) == set(TEMPLATES.keys())


class TestHealthcareTemplate:
    """Tests for healthcare-specific template."""

    def test_healthcare_template_exists(self):
        """Should have healthcare template."""
        assert "healthcare" in TEMPLATES

    def test_healthcare_has_high_risk(self):
        """Healthcare should be high risk."""
        config = TEMPLATES["healthcare"]
        assert config.risk_level == "high"
        assert config.domain_sensitivity == "healthcare"

    def test_healthcare_has_pii_scanner(self):
        """Healthcare should have PII scanner with redact action."""
        config = TEMPLATES["healthcare"]
        pii_scanner = None
        for s in config.scanners:
            if s.name == "PIIScanner":
                pii_scanner = s
                break

        assert pii_scanner is not None
        assert pii_scanner.action == "redact"

    def test_healthcare_has_high_thresholds(self):
        """Healthcare should have high thresholds."""
        config = TEMPLATES["healthcare"]
        for eval_config in config.evaluations:
            # Healthcare evals should have threshold >= 0.8
            assert eval_config.threshold >= 0.8
        assert config.global_pass_rate >= 0.9


class TestFinancialTemplate:
    """Tests for financial-specific template."""

    def test_financial_template_exists(self):
        """Should have financial template."""
        assert "financial" in TEMPLATES

    def test_financial_has_high_risk(self):
        """Financial should be high risk."""
        config = TEMPLATES["financial"]
        assert config.risk_level == "high"
        assert config.domain_sensitivity == "financial"

    def test_financial_has_coherence(self):
        """Financial should have coherence eval."""
        config = TEMPLATES["financial"]
        eval_names = [e.name for e in config.evaluations]
        assert "CoherenceEval" in eval_names
