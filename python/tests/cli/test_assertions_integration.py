"""Integration tests for assertions in the run command."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from fi.cli.main import app
from fi.cli.assertions import ExitCode


runner = CliRunner()


@pytest.fixture
def config_with_assertions(tmp_path: Path):
    """Create a config file with assertions."""
    # Create test data first so we know the path
    test_data = [
        {"input": "test1", "output": "output1", "context": "context1"},
        {"input": "test2", "output": "output2", "context": "context2"},
    ]
    data_path = tmp_path / "test_data.json"
    data_path.write_text(json.dumps(test_data))

    # Use absolute path in config
    config_content = f"""
version: "1.0"

evaluations:
  - name: "rag_quality"
    templates:
      - "groundedness"
      - "context_adherence"
    data: "{data_path}"

assertions:
  - template: "groundedness"
    conditions:
      - "pass_rate >= 0.6"
    on_fail: "error"

  - template: "context_adherence"
    conditions:
      - "avg_score >= 0.7"
    on_fail: "warn"
"""
    config_path = tmp_path / "fi-evaluation.yaml"
    config_path.write_text(config_content)

    return config_path, data_path


@pytest.fixture
def config_with_global_assertions(tmp_path: Path):
    """Create a config file with global assertions."""
    test_data = [{"input": "test1", "output": "output1", "context": "context1"}]
    data_path = tmp_path / "test_data.json"
    data_path.write_text(json.dumps(test_data))

    config_content = f"""
version: "1.0"

evaluations:
  - name: "test_suite"
    templates:
      - "groundedness"
    data: "{data_path}"

assertions:
  - global: true
    conditions:
      - "total_pass_rate >= 0.5"
    on_fail: "error"
"""
    config_path = tmp_path / "fi-evaluation.yaml"
    config_path.write_text(config_content)

    return config_path, data_path


@pytest.fixture
def config_with_thresholds(tmp_path: Path):
    """Create a config file with threshold shortcuts."""
    test_data = [{"input": "test1", "output": "output1", "context": "context1"}]
    data_path = tmp_path / "test_data.json"
    data_path.write_text(json.dumps(test_data))

    config_content = f"""
version: "1.0"

evaluations:
  - name: "test_suite"
    templates:
      - "groundedness"
      - "context_adherence"
    data: "{data_path}"

thresholds:
  default_pass_rate: 0.5
  fail_fast: false
  overrides:
    groundedness: 0.8
"""
    config_path = tmp_path / "fi-evaluation.yaml"
    config_path.write_text(config_content)

    return config_path, data_path


class TestAssertionConfigLoading:
    """Tests for loading assertion config from YAML."""

    def test_load_config_with_assertions(self, config_with_assertions):
        """Test loading config with assertions section."""
        config_path, _ = config_with_assertions

        from fi.cli.config.loader import load_config
        config = load_config(config_path)

        assert config.assertions is not None
        assert len(config.assertions) == 2

        # Check first assertion
        assert config.assertions[0].template == "groundedness"
        assert "pass_rate >= 0.6" in config.assertions[0].conditions
        assert config.assertions[0].on_fail == "error"

        # Check second assertion
        assert config.assertions[1].template == "context_adherence"
        assert config.assertions[1].on_fail == "warn"

    def test_load_config_with_global_assertions(self, config_with_global_assertions):
        """Test loading config with global assertions."""
        config_path, _ = config_with_global_assertions

        from fi.cli.config.loader import load_config
        config = load_config(config_path)

        assert config.assertions is not None
        assert len(config.assertions) == 1
        assert config.assertions[0].is_global is True
        assert config.assertions[0].template is None

    def test_load_config_with_thresholds(self, config_with_thresholds):
        """Test loading config with threshold shortcuts."""
        config_path, _ = config_with_thresholds

        from fi.cli.config.loader import load_config
        config = load_config(config_path)

        assert config.thresholds is not None
        assert config.thresholds.default_pass_rate == 0.5
        assert config.thresholds.fail_fast is False
        assert config.thresholds.overrides == {"groundedness": 0.8}


class TestAssertionIntegration:
    """Integration tests for assertions with the run command."""

    def test_build_assertion_config(self, config_with_assertions):
        """Test building assertion config from loaded config."""
        config_path, _ = config_with_assertions

        from fi.cli.config.loader import load_config
        from fi.cli.commands.run import _build_assertion_config

        config = load_config(config_path)
        assertion_config = _build_assertion_config(config, fail_fast=False)

        assert "assertions" in assertion_config
        assert len(assertion_config["assertions"]) == 2

    def test_build_assertion_config_with_thresholds(self, config_with_thresholds):
        """Test building assertion config with thresholds."""
        config_path, _ = config_with_thresholds

        from fi.cli.config.loader import load_config
        from fi.cli.commands.run import _build_assertion_config

        config = load_config(config_path)
        assertion_config = _build_assertion_config(config, fail_fast=True)

        assert "thresholds" in assertion_config
        assert assertion_config["thresholds"]["default_pass_rate"] == 0.5
        assert assertion_config["thresholds"]["fail_fast"] is True  # CLI override

    def test_run_with_assertions_dry_run(self, config_with_assertions, monkeypatch):
        """Test dry run mode with assertions config."""
        config_path, _ = config_with_assertions

        monkeypatch.setenv("FI_API_KEY", "test-key")
        monkeypatch.setenv("FI_SECRET_KEY", "test-secret")

        result = runner.invoke(app, [
            "run",
            "-c", str(config_path),
            "--dry-run"
        ])

        # Dry run should succeed even with assertions configured
        assert result.exit_code == 0
        assert "Configuration valid" in result.stdout


class TestAssertionEvaluatorIntegration:
    """Tests for assertion evaluator with real-like data."""

    def test_evaluator_with_passing_assertions(self):
        """Test evaluator when all assertions pass."""
        from fi.cli.assertions import AssertionEvaluator

        results = {
            "eval_results": [
                {"name": "groundedness", "output": True, "runtime": 100},
                {"name": "groundedness", "output": True, "runtime": 150},
                {"name": "groundedness", "output": True, "runtime": 120},
            ]
        }

        config = {
            "assertions": [
                {
                    "template": "groundedness",
                    "conditions": ["pass_rate >= 0.9"],
                    "on_fail": "error"
                }
            ]
        }

        evaluator = AssertionEvaluator(results, config)
        report = evaluator.evaluate_all()

        assert report.all_passed is True
        assert report.passed == 1
        assert report.failed == 0

    def test_evaluator_with_failing_assertions(self):
        """Test evaluator when assertions fail."""
        from fi.cli.assertions import AssertionEvaluator, AssertionResult

        results = {
            "eval_results": [
                {"name": "groundedness", "output": True, "runtime": 100},
                {"name": "groundedness", "output": False, "runtime": 150},
                {"name": "groundedness", "output": False, "runtime": 120},
            ]
        }

        config = {
            "assertions": [
                {
                    "template": "groundedness",
                    "conditions": ["pass_rate >= 0.8"],  # Only 33% pass rate
                    "on_fail": "error"
                }
            ]
        }

        evaluator = AssertionEvaluator(results, config)
        report = evaluator.evaluate_all()

        assert report.all_passed is False
        assert report.failed == 1
        assert report.outcomes[0].result == AssertionResult.FAILED

    def test_evaluator_with_warnings(self):
        """Test evaluator with warning-level assertions."""
        from fi.cli.assertions import AssertionEvaluator, AssertionResult

        results = {
            "eval_results": [
                {"name": "context_adherence", "output": 0.6, "runtime": 100},
                {"name": "context_adherence", "output": 0.7, "runtime": 150},
            ]
        }

        config = {
            "assertions": [
                {
                    "template": "context_adherence",
                    "conditions": ["avg_score >= 0.9"],  # 0.65 avg, below 0.9
                    "on_fail": "warn"
                }
            ]
        }

        evaluator = AssertionEvaluator(results, config)
        report = evaluator.evaluate_all()

        assert report.all_passed is True  # No hard failures
        assert report.has_warnings is True
        assert report.warnings == 1
        assert report.outcomes[0].result == AssertionResult.WARNING

    def test_evaluator_with_global_assertions(self):
        """Test evaluator with global assertions."""
        from fi.cli.assertions import AssertionEvaluator

        results = {
            "eval_results": [
                {"name": "groundedness", "output": True, "runtime": 100},
                {"name": "context_adherence", "output": True, "runtime": 150},
                {"name": "context_adherence", "output": False, "runtime": 120},
            ]
        }

        config = {
            "assertions": [
                {
                    "global": True,
                    "conditions": ["total_pass_rate >= 0.6"],
                    "on_fail": "error"
                }
            ]
        }

        evaluator = AssertionEvaluator(results, config)
        report = evaluator.evaluate_all()

        # 2/3 = 66.7% pass rate, >= 60%
        assert report.all_passed is True

    def test_evaluator_with_thresholds(self):
        """Test evaluator with threshold shortcuts."""
        from fi.cli.assertions import AssertionEvaluator

        results = {
            "eval_results": [
                {"name": "groundedness", "output": True, "runtime": 100},
                {"name": "groundedness", "output": True, "runtime": 100},
                {"name": "context_adherence", "output": True, "runtime": 150},
                {"name": "context_adherence", "output": False, "runtime": 120},
            ]
        }

        config = {
            "thresholds": {
                "default_pass_rate": 0.5,  # Both should pass
                "overrides": {}
            }
        }

        evaluator = AssertionEvaluator(results, config)
        report = evaluator.evaluate_all()

        # Both templates have >= 50% pass rate
        assert report.total_assertions == 2
        assert report.all_passed is True

    def test_evaluator_with_threshold_overrides(self):
        """Test evaluator with per-template threshold overrides."""
        from fi.cli.assertions import AssertionEvaluator

        results = {
            "eval_results": [
                {"name": "groundedness", "output": True, "runtime": 100},
                {"name": "groundedness", "output": False, "runtime": 100},
                {"name": "context_adherence", "output": True, "runtime": 150},
            ]
        }

        config = {
            "thresholds": {
                "default_pass_rate": 0.5,
                "overrides": {
                    "groundedness": 0.9  # Will fail (50% < 90%)
                }
            }
        }

        evaluator = AssertionEvaluator(results, config)
        report = evaluator.evaluate_all()

        # groundedness: 50% < 90% = FAIL
        # context_adherence: 100% >= 50% = PASS
        assert report.total_assertions == 2
        assert report.failed == 1
        assert report.passed == 1


class TestAssertionReporterIntegration:
    """Tests for assertion reporter output formats."""

    def test_reporter_json_output(self):
        """Test JSON output from reporter."""
        from rich.console import Console
        from fi.cli.assertions import AssertionReporter, AssertionReport, AssertionOutcome, AssertionResult

        console = Console()
        reporter = AssertionReporter(console)

        report = AssertionReport(
            outcomes=[
                AssertionOutcome(
                    template="groundedness",
                    condition="pass_rate >= 0.8",
                    expected=">= 0.8",
                    actual=0.85,
                    result=AssertionResult.PASSED,
                    message="Assertion passed"
                ),
            ],
            total_assertions=1,
            passed=1,
            failed=0,
            warnings=0,
            skipped=0
        )

        json_output = reporter.to_json(report)

        assert json_output["summary"]["total"] == 1
        assert json_output["summary"]["passed"] == 1
        assert json_output["summary"]["all_passed"] is True
        assert len(json_output["assertions"]) == 1
        assert json_output["assertions"][0]["result"] == "passed"

    def test_reporter_junit_output(self):
        """Test JUnit XML output from reporter."""
        from rich.console import Console
        from fi.cli.assertions import AssertionReporter, AssertionReport, AssertionOutcome, AssertionResult

        console = Console()
        reporter = AssertionReporter(console)

        report = AssertionReport(
            outcomes=[
                AssertionOutcome(
                    template="groundedness",
                    condition="pass_rate >= 0.8",
                    expected=">= 0.8",
                    actual=0.75,
                    result=AssertionResult.FAILED,
                    message="Failed: pass_rate is 0.75"
                ),
            ],
            total_assertions=1,
            passed=0,
            failed=1,
            warnings=0,
            skipped=0
        )

        junit_output = reporter.to_junit(report)

        assert '<?xml version="1.0"' in junit_output
        assert '<testsuites' in junit_output
        assert 'failures="1"' in junit_output
        assert '<failure' in junit_output
        assert 'groundedness' in junit_output


class TestExitCodeIntegration:
    """Tests for exit code behavior with assertions."""

    def test_exit_code_success(self):
        """Test SUCCESS exit code value."""
        assert ExitCode.SUCCESS == 0

    def test_exit_code_assertion_failed(self):
        """Test ASSERTION_FAILED exit code value."""
        assert ExitCode.ASSERTION_FAILED == 2

    def test_exit_code_assertion_warning(self):
        """Test ASSERTION_WARNING exit code value."""
        assert ExitCode.ASSERTION_WARNING == 3

    def test_exit_codes_for_ci(self):
        """Test that exit codes are suitable for CI/CD use."""
        # Non-zero codes should signal failures
        assert ExitCode.SUCCESS == 0
        assert ExitCode.EVALUATION_ERROR > 0
        assert ExitCode.ASSERTION_FAILED > 0
        assert ExitCode.ASSERTION_WARNING > 0
        assert ExitCode.CONFIG_ERROR > 0
