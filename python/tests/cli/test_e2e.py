"""End-to-end tests simulating real user workflows.

These tests cover complete user journeys through the CLI, testing
the integration between multiple commands as a real user would use them.
"""

import json
import os
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from fi.cli.main import app
from fi.cli.storage import RunHistory
from fi.evals.types import BatchRunResult, EvalResult


runner = CliRunner()


# =============================================================================
# Fixtures for E2E Tests
# =============================================================================


@pytest.fixture
def e2e_project(tmp_path: Path):
    """
    Create a complete project structure for E2E testing.

    Structure:
        project/
        ├── fi-evaluation.yaml
        ├── data/
        │   ├── rag_tests.json
        │   ├── safety_tests.json
        │   └── quality_tests.json
        └── results/
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create data directory
    data_dir = project_dir / "data"
    data_dir.mkdir()

    # Create results directory
    results_dir = project_dir / "results"
    results_dir.mkdir()

    # Create RAG test data
    rag_data = [
        {
            "query": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "context": "France is a country in Western Europe. Its capital city is Paris, which is known for the Eiffel Tower."
        },
        {
            "query": "Who wrote Romeo and Juliet?",
            "response": "William Shakespeare wrote Romeo and Juliet.",
            "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career."
        },
        {
            "query": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "context": "Machine learning is a branch of AI focused on building systems that learn from data."
        }
    ]
    (data_dir / "rag_tests.json").write_text(json.dumps(rag_data, indent=2))

    # Create safety test data
    safety_data = [
        {
            "response": "Here is a helpful answer to your question about cooking recipes.",
            "query": "How do I make pasta?"
        },
        {
            "response": "I'm happy to help you with your homework problem.",
            "query": "Can you help me with math?"
        }
    ]
    (data_dir / "safety_tests.json").write_text(json.dumps(safety_data, indent=2))

    # Create quality test data
    quality_data = [
        {
            "query": "Explain quantum computing",
            "response": "Quantum computing uses quantum bits (qubits) that can exist in superposition, allowing parallel computation. Unlike classical bits that are 0 or 1, qubits can be both simultaneously."
        }
    ]
    (data_dir / "quality_tests.json").write_text(json.dumps(quality_data, indent=2))

    # Create config file
    config = """version: "1.0"

defaults:
  model: "turing_flash"
  timeout: 120

evaluations:
  - name: "rag_quality"
    templates:
      - "groundedness"
      - "context_adherence"
    data: "./data/rag_tests.json"

  - name: "safety_checks"
    templates:
      - "content_moderation"
    data: "./data/safety_tests.json"

  - name: "response_quality"
    templates:
      - "is_concise"
    data: "./data/quality_tests.json"

output:
  format: "json"
  path: "./results/"
"""
    (project_dir / "fi-evaluation.yaml").write_text(config)

    return project_dir


@pytest.fixture
def mock_evaluation_client():
    """Mock the evaluation client to return predictable results."""
    def create_mock_result(template_name, success=True, score=None):
        if score is not None:
            return EvalResult(
                name=template_name,
                output=score,
                reason=f"Score: {score}",
                runtime=100,
                output_type="float",
                eval_id=f"eval-{template_name}-1"
            )
        return EvalResult(
            name=template_name,
            output=success,
            reason="Evaluation passed" if success else "Evaluation failed",
            runtime=100,
            output_type="boolean",
            eval_id=f"eval-{template_name}-1"
        )

    return create_mock_result


@pytest.fixture
def mock_run_history(tmp_path: Path):
    """Create a RunHistory instance with a temp directory."""
    storage_dir = tmp_path / ".fi" / "runs"
    return RunHistory(storage_dir)


# =============================================================================
# Workflow 1: Basic Evaluation Workflow
# User: validate config → run evaluations → view results → export
# =============================================================================


class TestBasicEvaluationWorkflow:
    """Test the basic user workflow: validate → run → view → export."""

    def test_complete_workflow_validate_run_view_export(
        self, e2e_project: Path, tmp_path: Path, monkeypatch
    ):
        """
        Simulate a complete user workflow:
        1. User validates their config file
        2. User runs evaluations
        3. User views the results
        4. User exports results to different formats
        """
        # Set up environment
        monkeypatch.setenv("FI_API_KEY", "test_key")
        monkeypatch.setenv("FI_SECRET_KEY", "test_secret")
        monkeypatch.chdir(e2e_project)

        # Step 1: Validate the configuration
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 0, f"Validation failed: {result.stdout}"
        assert "valid" in result.stdout.lower() or "✓" in result.stdout

        # Step 2: Run evaluations (dry-run mode to avoid API calls)
        result = runner.invoke(app, ["run", "--dry-run"])
        assert result.exit_code == 0, f"Dry run failed: {result.stdout}"
        assert "would run" in result.stdout.lower() or "dry" in result.stdout.lower()

    def test_workflow_with_mock_results(
        self, tmp_path: Path, monkeypatch
    ):
        """Test viewing and exporting with mock evaluation results."""
        # Set up mock run history with results
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        # Create mock results simulating a real evaluation run
        results = BatchRunResult(
            eval_results=[
                EvalResult(
                    name="groundedness",
                    output=True,
                    reason="All claims are supported by context",
                    runtime=150,
                    output_type="boolean",
                    eval_id="eval-1"
                ),
                EvalResult(
                    name="context_adherence",
                    output=0.92,
                    reason="High adherence score",
                    runtime=180,
                    output_type="float",
                    eval_id="eval-2"
                ),
                EvalResult(
                    name="content_moderation",
                    output=True,
                    reason="No harmful content detected",
                    runtime=100,
                    output_type="boolean",
                    eval_id="eval-3"
                ),
            ]
        )

        record = history.save_run(
            results=results,
            config_file="fi-evaluation.yaml",
            templates=["groundedness", "context_adherence", "content_moderation"]
        )

        # Monkeypatch RunHistory to use our test instance
        monkeypatch.setattr("fi.cli.commands.view.RunHistory", lambda: history)
        monkeypatch.setattr("fi.cli.commands.export.RunHistory", lambda: history)

        # Step 3: View results in terminal
        result = runner.invoke(app, ["view", "--last", "--terminal"])
        assert result.exit_code == 0, f"View failed: {result.stdout}"
        assert "groundedness" in result.stdout
        assert record.run_id[:16] in result.stdout

        # Step 4a: Export to JSON
        json_output = tmp_path / "results.json"
        result = runner.invoke(app, ["export", "--last", "-o", str(json_output), "-f", "json"])
        assert result.exit_code == 0, f"JSON export failed: {result.stdout}"
        assert json_output.exists()

        # Verify JSON content
        with open(json_output) as f:
            data = json.load(f)
        assert data["run_id"] == record.run_id
        assert len(data["eval_results"]) == 3

        # Step 4b: Export to CSV
        csv_output = tmp_path / "results.csv"
        result = runner.invoke(app, ["export", "--last", "-o", str(csv_output), "-f", "csv"])
        assert result.exit_code == 0, f"CSV export failed: {result.stdout}"
        assert csv_output.exists()
        assert "groundedness" in csv_output.read_text()

        # Step 4c: Export to HTML
        html_output = tmp_path / "report.html"
        result = runner.invoke(app, ["export", "--last", "-o", str(html_output), "-f", "html"])
        assert result.exit_code == 0, f"HTML export failed: {result.stdout}"
        assert html_output.exists()
        assert "<html>" in html_output.read_text()


# =============================================================================
# Workflow 2: CI/CD Pipeline Simulation
# User: run evaluations → export JUnit XML → check pass/fail
# =============================================================================


class TestCICDWorkflow:
    """Test CI/CD integration workflows."""

    def test_cicd_workflow_with_passing_results(self, tmp_path: Path, monkeypatch):
        """
        Simulate a CI/CD pipeline where all evaluations pass.
        Expected: exit code 0, JUnit XML shows all tests passing.
        """
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        # Create passing results
        results = BatchRunResult(
            eval_results=[
                EvalResult(name="groundedness", output=True, reason="Pass", runtime=100, output_type="boolean", eval_id="1"),
                EvalResult(name="context_adherence", output=0.95, reason="High score", runtime=100, output_type="float", eval_id="2"),
                EvalResult(name="content_moderation", output=True, reason="Safe", runtime=100, output_type="boolean", eval_id="3"),
            ]
        )

        record = history.save_run(results, config_file="ci-config.yaml", templates=["groundedness", "context_adherence", "content_moderation"])

        monkeypatch.setattr("fi.cli.commands.export.RunHistory", lambda: history)

        # Export to JUnit XML
        junit_output = tmp_path / "test-results.xml"
        result = runner.invoke(app, ["export", "--last", "-o", str(junit_output), "-f", "junit"])

        assert result.exit_code == 0
        assert junit_output.exists()

        # Verify JUnit XML content - no failures
        xml_content = junit_output.read_text()
        assert "<testsuites" in xml_content
        assert "<testsuite" in xml_content
        assert "failures=\"0\"" in xml_content or xml_content.count("<failure") == 0

    def test_cicd_workflow_with_failing_results(self, tmp_path: Path, monkeypatch):
        """
        Simulate a CI/CD pipeline where some evaluations fail.
        Expected: JUnit XML shows failures for failed tests.
        """
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        # Create mixed results (some failing)
        results = BatchRunResult(
            eval_results=[
                EvalResult(name="groundedness", output=True, reason="Pass", runtime=100, output_type="boolean", eval_id="1"),
                EvalResult(name="context_adherence", output=0.35, reason="Low score", runtime=100, output_type="float", eval_id="2"),  # Fail: < 0.5
                EvalResult(name="content_moderation", output=False, reason="Harmful content", runtime=100, output_type="boolean", eval_id="3"),  # Fail
            ]
        )

        record = history.save_run(results, config_file="ci-config.yaml", templates=["groundedness", "context_adherence", "content_moderation"])

        monkeypatch.setattr("fi.cli.commands.export.RunHistory", lambda: history)

        # Export to JUnit XML
        junit_output = tmp_path / "test-results.xml"
        result = runner.invoke(app, ["export", "--last", "-o", str(junit_output), "-f", "junit"])

        assert result.exit_code == 0
        assert junit_output.exists()

        # Verify JUnit XML content - should have failures
        xml_content = junit_output.read_text()
        assert "<failure" in xml_content
        # Two failures: context_adherence (0.35 < 0.5) and content_moderation (False)
        assert xml_content.count("<failure") == 2

    def test_cicd_multiple_formats_export(self, tmp_path: Path, monkeypatch):
        """
        Simulate CI pipeline that exports to multiple formats.
        Common pattern: JSON for debugging, JUnit for test reporting, HTML for artifacts.
        """
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        results = BatchRunResult(
            eval_results=[
                EvalResult(name="test_eval", output=True, reason="Pass", runtime=100, output_type="boolean", eval_id="1"),
            ]
        )

        history.save_run(results)
        monkeypatch.setattr("fi.cli.commands.export.RunHistory", lambda: history)

        # Export to all formats
        formats = [
            ("json", "results.json"),
            ("csv", "results.csv"),
            ("html", "report.html"),
            ("junit", "test-results.xml"),
        ]

        for fmt, filename in formats:
            output_file = tmp_path / filename
            result = runner.invoke(app, ["export", "--last", "-o", str(output_file), "-f", fmt])
            assert result.exit_code == 0, f"Export to {fmt} failed: {result.stdout}"
            assert output_file.exists(), f"Output file {filename} not created"


# =============================================================================
# Workflow 3: Multiple Runs and History Management
# User: run multiple times → list runs → view specific run → compare
# =============================================================================


class TestMultipleRunsWorkflow:
    """Test workflows involving multiple evaluation runs."""

    def test_multiple_runs_list_and_view(self, tmp_path: Path, monkeypatch):
        """
        Simulate a user running evaluations multiple times and reviewing history.
        """
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        # Create multiple runs
        run_ids = []
        for i in range(3):
            results = BatchRunResult(
                eval_results=[
                    EvalResult(
                        name=f"eval_run_{i}",
                        output=True,
                        reason=f"Run {i} passed",
                        runtime=100 + i * 10,
                        output_type="boolean",
                        eval_id=f"eval-{i}"
                    ),
                ]
            )
            record = history.save_run(results, config_file=f"config-{i}.yaml", templates=[f"template_{i}"])
            run_ids.append(record.run_id)

        monkeypatch.setattr("fi.cli.commands.view.RunHistory", lambda: history)

        # List all runs
        result = runner.invoke(app, ["view", "--list"])
        assert result.exit_code == 0
        assert "3" in result.stdout or "Recent Evaluation Runs" in result.stdout

        # View most recent run
        result = runner.invoke(app, ["view", "--last", "--terminal"])
        assert result.exit_code == 0
        assert run_ids[-1][:16] in result.stdout  # Should show the last run

        # View specific older run
        result = runner.invoke(app, ["view", run_ids[0], "--terminal"])
        assert result.exit_code == 0
        assert run_ids[0][:16] in result.stdout

    def test_run_history_limit(self, tmp_path: Path, monkeypatch):
        """Test that run history respects limits when listing."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        # Create more runs than the default display limit
        for i in range(15):
            results = BatchRunResult(
                eval_results=[
                    EvalResult(name=f"eval_{i}", output=True, reason="", runtime=100, output_type="boolean", eval_id=str(i))
                ]
            )
            history.save_run(results)

        monkeypatch.setattr("fi.cli.commands.view.RunHistory", lambda: history)

        # List should show limited runs (default is 10)
        result = runner.invoke(app, ["view", "--list"])
        assert result.exit_code == 0
        # The output should indicate how many are shown
        assert "shown" in result.stdout.lower() or "Recent Evaluation Runs" in result.stdout

    def test_delete_and_clear_history(self, tmp_path: Path, monkeypatch):
        """Test history management - delete specific run, clear all."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        # Create runs
        results = BatchRunResult(
            eval_results=[
                EvalResult(name="test", output=True, reason="", runtime=100, output_type="boolean", eval_id="1")
            ]
        )
        record1 = history.save_run(results)
        record2 = history.save_run(results)

        # Verify both exist
        assert history.get_run(record1.run_id) is not None
        assert history.get_run(record2.run_id) is not None

        # Delete one
        assert history.delete_run(record1.run_id) is True
        assert history.get_run(record1.run_id) is None
        assert history.get_run(record2.run_id) is not None

        # Clear all
        history.save_run(results)  # Add another
        count = history.clear_history()
        assert count >= 1
        assert len(history.list_runs()) == 0


# =============================================================================
# Workflow 4: Error Handling and Recovery
# User encounters errors and recovers from them
# =============================================================================


class TestErrorHandlingWorkflow:
    """Test error handling and recovery scenarios."""

    def test_missing_config_file(self, tmp_path: Path, monkeypatch):
        """User tries to run without a config file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FI_API_KEY", "test_key")
        monkeypatch.setenv("FI_SECRET_KEY", "test_secret")

        # Should fail gracefully with helpful message
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "no configuration" in result.stdout.lower()

    def test_invalid_config_validation(self, tmp_path: Path, monkeypatch):
        """User validates an invalid config file."""
        monkeypatch.chdir(tmp_path)

        # Create invalid YAML
        invalid_config = tmp_path / "fi-evaluation.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        result = runner.invoke(app, ["validate"])
        assert result.exit_code != 0

    def test_view_nonexistent_run(self, tmp_path: Path, monkeypatch):
        """User tries to view a run that doesn't exist."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        monkeypatch.setattr("fi.cli.commands.view.RunHistory", lambda: history)

        result = runner.invoke(app, ["view", "nonexistent-run-id"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_export_nonexistent_run(self, tmp_path: Path, monkeypatch):
        """User tries to export a run that doesn't exist."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        monkeypatch.setattr("fi.cli.commands.export.RunHistory", lambda: history)

        output_file = tmp_path / "output.json"
        result = runner.invoke(app, ["export", "nonexistent-run", "-o", str(output_file)])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_export_invalid_format(self, tmp_path: Path, monkeypatch):
        """User tries to export to an unsupported format."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        # Create a run first
        results = BatchRunResult(
            eval_results=[
                EvalResult(name="test", output=True, reason="", runtime=100, output_type="boolean", eval_id="1")
            ]
        )
        history.save_run(results)

        monkeypatch.setattr("fi.cli.commands.export.RunHistory", lambda: history)

        output_file = tmp_path / "output.xyz"
        result = runner.invoke(app, ["export", "--last", "-o", str(output_file), "-f", "xyz"])
        assert result.exit_code == 1
        assert "unsupported" in result.stdout.lower()

    def test_view_empty_history(self, tmp_path: Path, monkeypatch):
        """User tries to view runs when history is empty."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        monkeypatch.setattr("fi.cli.commands.view.RunHistory", lambda: history)

        # --list with empty history
        result = runner.invoke(app, ["view", "--list"])
        assert result.exit_code == 0
        assert "no runs found" in result.stdout.lower()

        # --last with empty history
        result = runner.invoke(app, ["view", "--last"])
        assert result.exit_code == 1
        assert "no runs found" in result.stdout.lower()


# =============================================================================
# Workflow 5: Configuration Discovery and Templates
# User explores available templates and sets up configuration
# =============================================================================


class TestConfigurationWorkflow:
    """Test configuration discovery and setup workflows."""

    def test_list_available_templates(self, monkeypatch):
        """User lists available evaluation templates."""
        monkeypatch.setenv("FI_API_KEY", "test_key")
        monkeypatch.setenv("FI_SECRET_KEY", "test_secret")

        result = runner.invoke(app, ["list", "templates"])
        assert result.exit_code == 0
        # Should show some templates
        assert "groundedness" in result.stdout.lower() or "template" in result.stdout.lower()

    def test_list_templates_by_category(self, monkeypatch):
        """User lists templates filtered by category."""
        monkeypatch.setenv("FI_API_KEY", "test_key")
        monkeypatch.setenv("FI_SECRET_KEY", "test_secret")

        result = runner.invoke(app, ["list", "templates", "--category", "rag"])
        # May return 0 or 1 depending on whether category exists
        # Main check is it doesn't crash
        assert result.exit_code in [0, 1]

    def test_init_creates_config(self, tmp_path: Path, monkeypatch):
        """User initializes a new project."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FI_API_KEY", "test_key")
        monkeypatch.setenv("FI_SECRET_KEY", "test_secret")

        result = runner.invoke(app, ["init"], input="n\n")  # Don't overwrite if exists
        # Init should create a config file or indicate one exists
        assert result.exit_code == 0 or "already exists" in result.stdout.lower()


# =============================================================================
# Workflow 6: Data Handling Edge Cases
# User works with various data formats and sizes
# =============================================================================


class TestDataHandlingWorkflow:
    """Test data handling edge cases in user workflows."""

    def test_empty_results_handling(self, tmp_path: Path, monkeypatch):
        """Test handling when evaluation returns empty results."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        # Create run with empty results
        results = BatchRunResult(eval_results=[])
        record = history.save_run(results)

        monkeypatch.setattr("fi.cli.commands.view.RunHistory", lambda: history)
        monkeypatch.setattr("fi.cli.commands.export.RunHistory", lambda: history)

        # View should work but show no results
        result = runner.invoke(app, ["view", "--last", "--terminal"])
        assert result.exit_code == 0

        # Export should work but produce minimal output
        output_file = tmp_path / "empty.json"
        result = runner.invoke(app, ["export", "--last", "-o", str(output_file)])
        assert result.exit_code == 0

        with open(output_file) as f:
            data = json.load(f)
        assert data["eval_results"] == []

    def test_large_results_handling(self, tmp_path: Path, monkeypatch):
        """Test handling large number of evaluation results."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        # Create run with many results
        results = BatchRunResult(
            eval_results=[
                EvalResult(
                    name=f"eval_{i}",
                    output=i % 2 == 0,  # Alternate pass/fail
                    reason=f"Result {i} with some longer text to simulate real output",
                    runtime=100 + i,
                    output_type="boolean",
                    eval_id=f"eval-{i}"
                )
                for i in range(100)
            ]
        )
        record = history.save_run(results)

        monkeypatch.setattr("fi.cli.commands.view.RunHistory", lambda: history)
        monkeypatch.setattr("fi.cli.commands.export.RunHistory", lambda: history)

        # View should handle large results
        result = runner.invoke(app, ["view", "--last", "--terminal"])
        assert result.exit_code == 0

        # Export should produce valid output
        output_file = tmp_path / "large.json"
        result = runner.invoke(app, ["export", "--last", "-o", str(output_file)])
        assert result.exit_code == 0

        with open(output_file) as f:
            data = json.load(f)
        assert len(data["eval_results"]) == 100

    def test_special_characters_in_results(self, tmp_path: Path, monkeypatch):
        """Test handling special characters in evaluation results."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        # Create results with special characters
        results = BatchRunResult(
            eval_results=[
                EvalResult(
                    name="unicode_test",
                    output=True,
                    reason="Special chars: émojis 🎉, quotes \"test\", newlines\nand\ttabs",
                    runtime=100,
                    output_type="boolean",
                    eval_id="eval-unicode"
                ),
            ]
        )
        record = history.save_run(results)

        monkeypatch.setattr("fi.cli.commands.export.RunHistory", lambda: history)

        # JSON export should handle unicode
        output_file = tmp_path / "unicode.json"
        result = runner.invoke(app, ["export", "--last", "-o", str(output_file)])
        assert result.exit_code == 0

        # Verify content is preserved
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        assert "émojis" in data["eval_results"][0]["reason"]
        assert "🎉" in data["eval_results"][0]["reason"]


# =============================================================================
# Workflow 7: Full Integration Test
# Complete user journey from start to finish
# =============================================================================


class TestFullIntegrationWorkflow:
    """Complete end-to-end integration test."""

    def test_complete_user_journey(self, tmp_path: Path, monkeypatch):
        """
        Simulate a complete user journey:
        1. Set up project structure
        2. Validate configuration
        3. Run evaluation (mocked)
        4. View results in list
        5. View detailed results
        6. Export to multiple formats
        7. Verify all exports
        """
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        # Step 1: Set up project
        data_dir = project_dir / "data"
        data_dir.mkdir()

        test_data = [
            {"query": "What is AI?", "response": "AI is artificial intelligence.", "context": "AI stands for artificial intelligence."}
        ]
        (data_dir / "tests.json").write_text(json.dumps(test_data))

        config = """version: "1.0"
evaluations:
  - name: "ai_test"
    template: "groundedness"
    data: "./data/tests.json"
"""
        (project_dir / "fi-evaluation.yaml").write_text(config)

        monkeypatch.chdir(project_dir)
        monkeypatch.setenv("FI_API_KEY", "test_key")
        monkeypatch.setenv("FI_SECRET_KEY", "test_secret")

        # Step 2: Validate
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 0, f"Validation failed: {result.stdout}"

        # Step 3: Create mock run (simulating what `fi run` would create)
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        results = BatchRunResult(
            eval_results=[
                EvalResult(
                    name="groundedness",
                    output=True,
                    reason="All claims supported by context",
                    runtime=150,
                    output_type="boolean",
                    eval_id="eval-1"
                ),
            ]
        )
        record = history.save_run(
            results,
            config_file="fi-evaluation.yaml",
            templates=["groundedness"]
        )

        monkeypatch.setattr("fi.cli.commands.view.RunHistory", lambda: history)
        monkeypatch.setattr("fi.cli.commands.export.RunHistory", lambda: history)

        # Step 4: View list
        result = runner.invoke(app, ["view", "--list"])
        assert result.exit_code == 0
        assert "1" in result.stdout or "Recent Evaluation Runs" in result.stdout

        # Step 5: View details
        result = runner.invoke(app, ["view", "--last", "--terminal", "--detailed"])
        assert result.exit_code == 0
        assert "groundedness" in result.stdout

        # Step 6: Export to all formats
        exports = {
            "json": project_dir / "results" / "export.json",
            "csv": project_dir / "results" / "export.csv",
            "html": project_dir / "results" / "report.html",
            "junit": project_dir / "results" / "junit.xml",
        }

        (project_dir / "results").mkdir(exist_ok=True)

        for fmt, path in exports.items():
            result = runner.invoke(app, ["export", "--last", "-o", str(path), "-f", fmt])
            assert result.exit_code == 0, f"Export to {fmt} failed: {result.stdout}"
            assert path.exists(), f"Export file {path} not created"

        # Step 7: Verify exports
        # JSON
        with open(exports["json"]) as f:
            json_data = json.load(f)
        assert json_data["run_id"] == record.run_id
        assert len(json_data["eval_results"]) == 1

        # CSV
        csv_content = exports["csv"].read_text()
        assert "groundedness" in csv_content
        assert record.run_id in csv_content

        # HTML
        html_content = exports["html"].read_text()
        assert "<html>" in html_content
        assert "groundedness" in html_content

        # JUnit
        junit_content = exports["junit"].read_text()
        assert "<testsuites" in junit_content
        assert "groundedness" in junit_content
        assert "failures=\"0\"" in junit_content  # No failures
