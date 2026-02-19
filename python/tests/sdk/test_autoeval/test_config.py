"""Tests for AutoEval configuration classes."""

import pytest
from fi.evals.autoeval.config import (
    EvalConfig,
    ScannerConfig,
    AutoEvalConfig,
)


class TestEvalConfig:
    """Tests for EvalConfig dataclass."""

    def test_create_eval_config(self):
        """Should create eval config with all fields."""
        config = EvalConfig(
            name="CoherenceEval",
            enabled=True,
            threshold=0.8,
            weight=1.5,
            params={"strict": True},
        )
        assert config.name == "CoherenceEval"
        assert config.enabled is True
        assert config.threshold == 0.8
        assert config.weight == 1.5
        assert config.params == {"strict": True}

    def test_eval_config_defaults(self):
        """Should have sensible defaults."""
        config = EvalConfig(name="TestEval")
        assert config.enabled is True
        assert config.threshold == 0.7
        assert config.weight == 1.0
        assert config.params == {}

    def test_eval_config_to_dict(self):
        """Should convert to dictionary."""
        config = EvalConfig(
            name="TestEval",
            threshold=0.85,
            params={"key": "value"},
        )
        data = config.to_dict()
        assert data["name"] == "TestEval"
        assert data["threshold"] == 0.85
        assert data["params"] == {"key": "value"}

    def test_eval_config_from_dict(self):
        """Should create from dictionary."""
        data = {
            "name": "FromDictEval",
            "threshold": 0.9,
            "weight": 2.0,
            "enabled": False,
        }
        config = EvalConfig.from_dict(data)
        assert config.name == "FromDictEval"
        assert config.threshold == 0.9
        assert config.weight == 2.0
        assert config.enabled is False

    def test_eval_config_copy(self):
        """Should create independent copy."""
        original = EvalConfig(
            name="Original",
            params={"nested": {"key": "value"}},
        )
        copy = original.copy()
        copy.params["nested"]["key"] = "modified"
        assert original.params["nested"]["key"] == "value"


class TestScannerConfig:
    """Tests for ScannerConfig dataclass."""

    def test_create_scanner_config(self):
        """Should create scanner config with all fields."""
        config = ScannerConfig(
            name="JailbreakScanner",
            enabled=True,
            threshold=0.9,
            action="block",
            params={"patterns": ["test"]},
        )
        assert config.name == "JailbreakScanner"
        assert config.enabled is True
        assert config.threshold == 0.9
        assert config.action == "block"

    def test_scanner_config_defaults(self):
        """Should have sensible defaults."""
        config = ScannerConfig(name="TestScanner")
        assert config.enabled is True
        assert config.threshold == 0.7
        assert config.action == "block"
        assert config.params == {}

    def test_scanner_config_actions(self):
        """Should support different actions."""
        for action in ["block", "flag", "warn", "redact"]:
            config = ScannerConfig(name="Test", action=action)
            assert config.action == action

    def test_scanner_config_to_dict(self):
        """Should convert to dictionary."""
        config = ScannerConfig(
            name="PIIScanner",
            action="redact",
        )
        data = config.to_dict()
        assert data["name"] == "PIIScanner"
        assert data["action"] == "redact"

    def test_scanner_config_from_dict(self):
        """Should create from dictionary."""
        data = {
            "name": "FromDictScanner",
            "threshold": 0.95,
            "action": "flag",
        }
        config = ScannerConfig.from_dict(data)
        assert config.name == "FromDictScanner"
        assert config.threshold == 0.95
        assert config.action == "flag"


class TestAutoEvalConfig:
    """Tests for AutoEvalConfig dataclass."""

    def test_create_full_config(self):
        """Should create config with all fields."""
        config = AutoEvalConfig(
            name="test_pipeline",
            description="Test pipeline",
            app_category="rag_system",
            risk_level="high",
            domain_sensitivity="healthcare",
            evaluations=[
                EvalConfig(name="FactualConsistencyEval", threshold=0.9),
            ],
            scanners=[
                ScannerConfig(name="PIIScanner", action="redact"),
            ],
            execution_mode="blocking",
            parallel_workers=8,
            timeout_seconds=60,
            fail_fast=True,
            global_pass_rate=0.9,
        )
        assert config.name == "test_pipeline"
        assert config.app_category == "rag_system"
        assert len(config.evaluations) == 1
        assert len(config.scanners) == 1
        assert config.parallel_workers == 8

    def test_config_defaults(self):
        """Should have sensible defaults."""
        config = AutoEvalConfig(name="default_test")
        assert config.description == ""
        assert config.version == "1.0.0"
        assert config.app_category == "unknown"
        assert config.risk_level == "medium"
        assert config.execution_mode == "non_blocking"
        assert config.parallel_workers == 4
        assert config.global_pass_rate == 0.8

    def test_config_to_dict(self):
        """Should convert to dictionary."""
        config = AutoEvalConfig(
            name="export_test",
            description="For export",
            app_category="chatbot",
            evaluations=[EvalConfig(name="TestEval")],
        )
        data = config.to_dict()
        assert data["name"] == "export_test"
        assert data["description"] == "For export"
        assert data["metadata"]["app_category"] == "chatbot"
        assert data["metadata"]["generated_by"] == "autoeval"
        assert len(data["evaluations"]) == 1

    def test_config_from_dict(self):
        """Should create from dictionary."""
        data = {
            "name": "imported",
            "description": "Imported config",
            "metadata": {
                "app_category": "rag_system",
                "risk_level": "high",
            },
            "evaluations": [
                {"name": "Eval1", "threshold": 0.8},
            ],
            "scanners": [
                {"name": "Scanner1", "action": "block"},
            ],
            "execution": {
                "mode": "blocking",
                "parallel_workers": 2,
            },
            "thresholds": {
                "global_pass_rate": 0.85,
            },
        }
        config = AutoEvalConfig.from_dict(data)
        assert config.name == "imported"
        assert config.app_category == "rag_system"
        assert config.risk_level == "high"
        assert len(config.evaluations) == 1
        assert config.evaluations[0].threshold == 0.8
        assert config.execution_mode == "blocking"
        assert config.global_pass_rate == 0.85

    def test_config_copy(self):
        """Should create independent copy."""
        original = AutoEvalConfig(
            name="original",
            evaluations=[EvalConfig(name="Eval1")],
        )
        copy = original.copy()
        copy.evaluations.append(EvalConfig(name="Eval2"))
        assert len(original.evaluations) == 1
        assert len(copy.evaluations) == 2

    def test_config_get_eval(self):
        """Should get eval by name."""
        config = AutoEvalConfig(
            name="test",
            evaluations=[
                EvalConfig(name="Eval1"),
                EvalConfig(name="Eval2"),
            ],
        )
        eval_config = config.get_eval("Eval1")
        assert eval_config is not None
        assert eval_config.name == "Eval1"

        missing = config.get_eval("NonExistent")
        assert missing is None

    def test_config_get_scanner(self):
        """Should get scanner by name."""
        config = AutoEvalConfig(
            name="test",
            scanners=[
                ScannerConfig(name="Scanner1"),
                ScannerConfig(name="Scanner2"),
            ],
        )
        scanner_config = config.get_scanner("Scanner2")
        assert scanner_config is not None
        assert scanner_config.name == "Scanner2"

    def test_config_summary(self):
        """Should generate summary string."""
        config = AutoEvalConfig(
            name="summary_test",
            app_category="rag_system",
            risk_level="high",
            evaluations=[EvalConfig(name="Eval1")],
            scanners=[ScannerConfig(name="Scanner1")],
        )
        summary = config.summary()
        assert "summary_test" in summary
        assert "rag_system" in summary
        assert "high" in summary
        assert "Eval1" in summary
        assert "Scanner1" in summary
