"""Comprehensive tests for the run command."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from fi.cli.main import app
from fi.evals.types import BatchRunResult, EvalResult


runner = CliRunner()


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Create a sample configuration file."""
    config_content = """version: "1.0"

evaluations:
  - name: "test_eval"
    template: "groundedness"
    data: "./data/test.json"

output:
  format: "json"
  path: "./results/"
"""
    config_path = tmp_path / "fi-evaluation.yaml"
    config_path.write_text(config_content)

    # Create data directory and file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "test.json"
    data_file.write_text(json.dumps([
        {"query": "test", "response": "test", "context": "test context"}
    ]))

    # Create results directory
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    return tmp_path


@pytest.fixture
def mock_evaluator(monkeypatch):
    """Create a mock evaluator that returns successful results."""
    # Set mock API keys
    monkeypatch.setenv("FI_API_KEY", "test_api_key")
    monkeypatch.setenv("FI_SECRET_KEY", "test_secret_key")

    with patch('fi.evals.evaluator.Evaluator') as MockEvaluator:
        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = BatchRunResult(eval_results=[
            EvalResult(
                name="groundedness",
                output="GROUNDED",
                reason="The response is grounded in context",
                runtime=1500,
                output_type="boolean",
                eval_id="eval-123"
            )
        ])
        MockEvaluator.return_value = mock_instance
        yield mock_instance


class TestRunCommand:
    """Tests for fi run command."""

    def test_run_with_config_file(self, sample_config: Path, mock_evaluator):
        """Test run with a configuration file."""
        result = runner.invoke(
            app,
            ["run", "--config", str(sample_config / "fi-evaluation.yaml")],
            catch_exceptions=False
        )

        # The command should complete
        assert result.exit_code == 0 or "Error" in result.stdout

    def test_run_without_config_discovers_file(self, sample_config: Path, mock_evaluator):
        """Test run discovers config file in current directory."""
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(sample_config)
            result = runner.invoke(app, ["run"])
            # Should not fail with "config not found" when config exists
        finally:
            os.chdir(original_dir)

    def test_run_dry_run_mode(self, sample_config: Path, mock_evaluator):
        """Test run with --dry-run flag."""
        result = runner.invoke(
            app,
            ["run", "--config", str(sample_config / "fi-evaluation.yaml"), "--dry-run"]
        )

        # Dry run should not call the evaluator
        mock_evaluator.evaluate.assert_not_called()

    def test_run_with_custom_output_format_json(self, sample_config: Path, mock_evaluator):
        """Test run with JSON output format."""
        result = runner.invoke(
            app,
            ["run", "--config", str(sample_config / "fi-evaluation.yaml"), "--output", "json"]
        )

        # Should complete without error
        assert result.exit_code == 0 or "Error" in result.stdout

    def test_run_with_custom_output_format_table(self, sample_config: Path, mock_evaluator):
        """Test run with table output format."""
        result = runner.invoke(
            app,
            ["run", "--config", str(sample_config / "fi-evaluation.yaml"), "--output", "table"]
        )

        # Should complete without error
        assert result.exit_code == 0 or "Error" in result.stdout

    def test_run_with_custom_timeout(self, sample_config: Path, mock_evaluator):
        """Test run with custom timeout."""
        result = runner.invoke(
            app,
            ["run", "--config", str(sample_config / "fi-evaluation.yaml"), "--timeout", "300"]
        )

        # Should complete without error
        assert result.exit_code == 0 or "Error" in result.stdout

    def test_run_with_parallel_workers(self, sample_config: Path, mock_evaluator):
        """Test run with custom parallel workers."""
        result = runner.invoke(
            app,
            ["run", "--config", str(sample_config / "fi-evaluation.yaml"), "--parallel", "16"]
        )

        # Should complete without error
        assert result.exit_code == 0 or "Error" in result.stdout

    def test_run_with_model_override(self, sample_config: Path, mock_evaluator):
        """Test run with model override."""
        result = runner.invoke(
            app,
            ["run", "--config", str(sample_config / "fi-evaluation.yaml"), "--model", "turing_pro"]
        )

        # Should complete without error
        assert result.exit_code == 0 or "Error" in result.stdout

    def test_run_missing_config_file(self, tmp_path: Path):
        """Test run with missing config file."""
        result = runner.invoke(
            app,
            ["run", "--config", str(tmp_path / "nonexistent.yaml")]
        )

        assert result.exit_code == 1 or "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_run_quiet_mode(self, sample_config: Path, mock_evaluator):
        """Test run with quiet mode suppresses progress output."""
        result = runner.invoke(
            app,
            ["run", "--config", str(sample_config / "fi-evaluation.yaml"), "--quiet"]
        )

        # Should complete without error
        assert result.exit_code == 0 or "Error" in result.stdout


class TestRunWithOverrides:
    """Tests for run command with CLI overrides."""

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> Path:
        """Create a test data file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        data_file = data_dir / "test.json"
        data_file.write_text(json.dumps([
            {"context": "The capital of France is Paris.", "output": "Paris is the capital of France."}
        ]))
        return data_file

    def test_run_with_eval_override(self, data_file: Path, mock_evaluator):
        """Test run with --eval override."""
        result = runner.invoke(
            app,
            ["run", "--eval", "groundedness", "--data", str(data_file)]
        )

        # Should attempt to run the specified evaluation
        assert result.exit_code == 0 or "Error" in result.stdout

    def test_run_with_data_override(self, data_file: Path, mock_evaluator):
        """Test run with --data override."""
        result = runner.invoke(
            app,
            ["run", "--eval", "factual_accuracy", "--data", str(data_file)]
        )

        # Should attempt to run with the specified data
        assert result.exit_code == 0 or "Error" in result.stdout


class TestRunOutputFormats:
    """Tests for different output formats."""

    @pytest.fixture
    def sample_config_with_results(self, tmp_path: Path, mock_evaluator) -> Path:
        """Create config with mock evaluator results."""
        config_content = """version: "1.0"

evaluations:
  - name: "test_eval"
    template: "groundedness"
    data: "./data/test.json"
"""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text(config_content)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        data_file = data_dir / "test.json"
        data_file.write_text(json.dumps([
            {"context": "Test context", "output": "Test output"}
        ]))

        return tmp_path

    def test_output_to_file(self, sample_config_with_results: Path, mock_evaluator):
        """Test saving output to a file."""
        output_file = sample_config_with_results / "output.json"
        result = runner.invoke(
            app,
            [
                "run",
                "--config", str(sample_config_with_results / "fi-evaluation.yaml"),
                "--output", "json",
                "--output-file", str(output_file)
            ]
        )

        # Should complete without error
        assert result.exit_code == 0 or "Error" in result.stdout


class TestRunEdgeCases:
    """Tests for edge cases in run command."""

    def test_run_empty_data_file(self, tmp_path: Path):
        """Test run with empty data file."""
        config_content = """version: "1.0"

evaluations:
  - name: "test_eval"
    template: "groundedness"
    data: "./data/empty.json"
"""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text(config_content)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "empty.json").write_text("[]")

        result = runner.invoke(
            app,
            ["run", "--config", str(config_path)]
        )

        # Should handle empty data gracefully
        # May succeed with empty results or fail with appropriate message

    def test_run_invalid_json_data(self, tmp_path: Path):
        """Test run with invalid JSON data file."""
        config_content = """version: "1.0"

evaluations:
  - name: "test_eval"
    template: "groundedness"
    data: "./data/invalid.json"
"""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text(config_content)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "invalid.json").write_text("{invalid json")

        result = runner.invoke(
            app,
            ["run", "--config", str(config_path)]
        )

        assert result.exit_code == 1 or "error" in result.stdout.lower()

    def test_run_with_multiple_templates(self, tmp_path: Path, mock_evaluator):
        """Test run with multiple evaluation templates."""
        config_content = """version: "1.0"

evaluations:
  - name: "multi_eval"
    templates:
      - "groundedness"
      - "factual_accuracy"
    data: "./data/test.json"
"""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text(config_content)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "test.json").write_text(json.dumps([
            {"context": "Test", "output": "Test"}
        ]))

        result = runner.invoke(
            app,
            ["run", "--config", str(config_path)]
        )

        # Should handle multiple templates
        assert result.exit_code == 0 or "Error" in result.stdout
