"""Tests for the export command."""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner

from fi.cli.main import app
from fi.cli.storage import RunHistory


runner = CliRunner()


@pytest.fixture
def sample_run(tmp_path: Path):
    """Create a sample run in history."""
    storage_dir = tmp_path / ".fi" / "runs"
    history = RunHistory(storage_dir)

    from fi.evals.types import BatchRunResult, EvalResult

    results = BatchRunResult(
        eval_results=[
            EvalResult(
                name="groundedness",
                output=True,
                reason="All claims are grounded",
                runtime=150,
                output_type="boolean",
                eval_id="test-1",
            ),
            EvalResult(
                name="context_adherence",
                output=0.85,
                reason="High adherence",
                runtime=200,
                output_type="float",
                eval_id="test-2",
            ),
        ]
    )

    record = history.save_run(
        results=results,
        config_file="test-config.yaml",
        templates=["groundedness", "context_adherence"],
    )

    return history, record


class TestExportCommand:
    """Tests for fi export command."""

    def test_export_json(self, sample_run, tmp_path: Path, monkeypatch):
        """Test exporting to JSON format."""
        history, record = sample_run
        output_file = tmp_path / "export.json"

        monkeypatch.setattr(
            "fi.cli.commands.export.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, [
            "export", "--last", "-o", str(output_file), "-f", "json"
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)

        assert data["run_id"] == record.run_id
        assert "eval_results" in data
        assert len(data["eval_results"]) == 2

    def test_export_csv(self, sample_run, tmp_path: Path, monkeypatch):
        """Test exporting to CSV format."""
        history, record = sample_run
        output_file = tmp_path / "export.csv"

        monkeypatch.setattr(
            "fi.cli.commands.export.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, [
            "export", "--last", "-o", str(output_file), "-f", "csv"
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify CSV content
        content = output_file.read_text()
        assert "run_id" in content
        assert "groundedness" in content
        assert "context_adherence" in content

    def test_export_html(self, sample_run, tmp_path: Path, monkeypatch):
        """Test exporting to HTML format."""
        history, record = sample_run
        output_file = tmp_path / "export.html"

        monkeypatch.setattr(
            "fi.cli.commands.export.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, [
            "export", "--last", "-o", str(output_file), "-f", "html"
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify HTML content
        content = output_file.read_text()
        assert "<html>" in content
        assert "Evaluation Results" in content
        assert record.run_id in content

    def test_export_junit(self, sample_run, tmp_path: Path, monkeypatch):
        """Test exporting to JUnit XML format."""
        history, record = sample_run
        output_file = tmp_path / "export.xml"

        monkeypatch.setattr(
            "fi.cli.commands.export.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, [
            "export", "--last", "-o", str(output_file), "-f", "junit"
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify XML content
        content = output_file.read_text()
        assert "<testsuites" in content
        assert "<testsuite" in content
        assert "<testcase" in content
        assert "groundedness" in content

    def test_export_specific_run(self, sample_run, tmp_path: Path, monkeypatch):
        """Test exporting a specific run by ID."""
        history, record = sample_run
        output_file = tmp_path / "export.json"

        monkeypatch.setattr(
            "fi.cli.commands.export.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, [
            "export", record.run_id, "-o", str(output_file)
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_export_nonexistent_run(self, sample_run, tmp_path: Path, monkeypatch):
        """Test exporting a non-existent run."""
        history, _ = sample_run
        output_file = tmp_path / "export.json"

        monkeypatch.setattr(
            "fi.cli.commands.export.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, [
            "export", "nonexistent-run", "-o", str(output_file)
        ])

        assert result.exit_code == 1
        assert "Run not found" in result.stdout

    def test_export_invalid_format(self, sample_run, tmp_path: Path, monkeypatch):
        """Test exporting with invalid format."""
        history, _ = sample_run
        output_file = tmp_path / "export.xyz"

        monkeypatch.setattr(
            "fi.cli.commands.export.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, [
            "export", "--last", "-o", str(output_file), "-f", "invalid"
        ])

        assert result.exit_code == 1
        assert "Unsupported format" in result.stdout

    def test_export_no_run_specified(self, sample_run, tmp_path: Path, monkeypatch):
        """Test export without specifying a run."""
        history, _ = sample_run
        output_file = tmp_path / "export.json"

        monkeypatch.setattr(
            "fi.cli.commands.export.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, [
            "export", "-o", str(output_file)
        ])

        assert result.exit_code == 1
        assert "specify a run ID" in result.stdout or "--last" in result.stdout


class TestJUnitExport:
    """Tests for JUnit XML export format."""

    def test_junit_includes_failures(self, tmp_path: Path):
        """Test that JUnit export marks failures correctly."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        from fi.evals.types import BatchRunResult, EvalResult

        # Create run with failures
        results = BatchRunResult(
            eval_results=[
                EvalResult(name="passing", output=True, reason="Passed", runtime=100, output_type="boolean", eval_id="1"),
                EvalResult(name="failing", output=False, reason="Failed check", runtime=100, output_type="boolean", eval_id="2"),
            ]
        )

        record = history.save_run(results)
        output_file = tmp_path / "results.xml"

        from fi.cli.commands.export import _export_junit

        results_data = history.load_results(record.run_id)
        _export_junit(record, results_data, output_file)

        content = output_file.read_text()
        assert '<failure' in content
        assert 'Failed check' in content

    def test_junit_numeric_scores(self, tmp_path: Path):
        """Test JUnit export with numeric scores."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        from fi.evals.types import BatchRunResult, EvalResult

        # Create run with numeric scores
        results = BatchRunResult(
            eval_results=[
                EvalResult(name="high_score", output=0.9, reason="High", runtime=100, output_type="float", eval_id="1"),
                EvalResult(name="low_score", output=0.3, reason="Low", runtime=100, output_type="float", eval_id="2"),
            ]
        )

        record = history.save_run(results)
        output_file = tmp_path / "results.xml"

        from fi.cli.commands.export import _export_junit

        results_data = history.load_results(record.run_id)
        _export_junit(record, results_data, output_file)

        content = output_file.read_text()
        # Low scores (< 0.5) should be marked as failures
        assert '<failure' in content
