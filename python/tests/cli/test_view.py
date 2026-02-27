"""Tests for the view command."""

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
    # Set up storage in temp directory
    storage_dir = tmp_path / ".fi" / "runs"
    history = RunHistory(storage_dir)

    # Create mock results
    from fi.evals.types import BatchRunResult, EvalResult

    results = BatchRunResult(
        eval_results=[
            EvalResult(
                name="groundedness",
                output=True,
                reason="All claims are grounded in context",
                runtime=150,
                output_type="boolean",
                eval_id="test-1",
            ),
            EvalResult(
                name="context_adherence",
                output=0.85,
                reason="High adherence to context",
                runtime=200,
                output_type="float",
                eval_id="test-2",
            ),
            EvalResult(
                name="content_moderation",
                output=False,
                reason="Potential harmful content detected",
                runtime=100,
                output_type="boolean",
                eval_id="test-3",
            ),
        ]
    )

    record = history.save_run(
        results=results,
        config_file="test-config.yaml",
        templates=["groundedness", "context_adherence", "content_moderation"],
    )

    return history, record


class TestViewCommand:
    """Tests for fi view command."""

    def test_view_list_shows_runs(self, sample_run, tmp_path: Path, monkeypatch):
        """Test that --list shows available runs."""
        history, record = sample_run

        # Monkeypatch RunHistory to use our test storage
        monkeypatch.setattr(
            "fi.cli.commands.view.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, ["view", "--list"])

        assert result.exit_code == 0
        # Rich may truncate long run IDs in the table, so check for the beginning
        run_id_prefix = record.run_id[:16]  # e.g., "20260123-192003-"
        assert run_id_prefix in result.stdout

    def test_view_list_empty(self, tmp_path: Path, monkeypatch):
        """Test --list with no runs."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        monkeypatch.setattr(
            "fi.cli.commands.view.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, ["view", "--list"])

        assert result.exit_code == 0
        assert "No runs found" in result.stdout

    def test_view_last_terminal(self, sample_run, monkeypatch):
        """Test viewing last run in terminal mode."""
        history, record = sample_run

        monkeypatch.setattr(
            "fi.cli.commands.view.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, ["view", "--last", "--terminal"])

        assert result.exit_code == 0
        assert record.run_id in result.stdout
        assert "groundedness" in result.stdout

    def test_view_last_detailed(self, sample_run, monkeypatch):
        """Test viewing last run with detailed output."""
        history, record = sample_run

        monkeypatch.setattr(
            "fi.cli.commands.view.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, ["view", "--last", "--terminal", "--detailed"])

        assert result.exit_code == 0
        assert "Evaluation Results" in result.stdout

    def test_view_specific_run(self, sample_run, monkeypatch):
        """Test viewing a specific run by ID."""
        history, record = sample_run

        monkeypatch.setattr(
            "fi.cli.commands.view.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, ["view", record.run_id, "--terminal"])

        assert result.exit_code == 0
        assert record.run_id in result.stdout

    def test_view_nonexistent_run(self, sample_run, monkeypatch):
        """Test viewing a non-existent run."""
        history, _ = sample_run

        monkeypatch.setattr(
            "fi.cli.commands.view.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, ["view", "nonexistent-run-id"])

        assert result.exit_code == 1
        assert "Run not found" in result.stdout

    def test_view_no_args_shows_help(self, sample_run, monkeypatch):
        """Test that view with no args prompts for action."""
        history, _ = sample_run

        monkeypatch.setattr(
            "fi.cli.commands.view.RunHistory",
            lambda: history
        )

        result = runner.invoke(app, ["view"])

        assert result.exit_code == 1
        assert "specify a run ID" in result.stdout or "--last" in result.stdout


class TestRunHistory:
    """Tests for RunHistory storage."""

    def test_save_and_load_run(self, tmp_path: Path):
        """Test saving and loading a run."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        from fi.evals.types import BatchRunResult, EvalResult

        results = BatchRunResult(
            eval_results=[
                EvalResult(
                    name="test_eval",
                    output=True,
                    reason="Test passed",
                    runtime=100,
                    output_type="boolean",
                    eval_id="test-1",
                ),
            ]
        )

        record = history.save_run(results)

        # Verify record
        assert record.run_id is not None
        assert record.total_evaluations == 1
        assert record.successful == 1

        # Load and verify
        loaded = history.get_run(record.run_id)
        assert loaded is not None
        assert loaded.run_id == record.run_id

    def test_get_latest_run(self, tmp_path: Path):
        """Test getting the latest run."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        from fi.evals.types import BatchRunResult, EvalResult

        # Save two runs
        results1 = BatchRunResult(eval_results=[
            EvalResult(name="first", output=True, reason="", runtime=100, output_type="boolean", eval_id="1")
        ])
        results2 = BatchRunResult(eval_results=[
            EvalResult(name="second", output=True, reason="", runtime=100, output_type="boolean", eval_id="2")
        ])

        record1 = history.save_run(results1)
        record2 = history.save_run(results2)

        # Latest should be the second one
        latest = history.get_latest_run()
        assert latest is not None
        assert latest.run_id == record2.run_id

    def test_list_runs(self, tmp_path: Path):
        """Test listing runs."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        from fi.evals.types import BatchRunResult, EvalResult

        # Save multiple runs
        for i in range(5):
            results = BatchRunResult(eval_results=[
                EvalResult(name=f"test_{i}", output=True, reason="", runtime=100, output_type="boolean", eval_id=str(i))
            ])
            history.save_run(results)

        runs = history.list_runs(limit=3)
        assert len(runs) == 3

    def test_delete_run(self, tmp_path: Path):
        """Test deleting a run."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        from fi.evals.types import BatchRunResult, EvalResult

        results = BatchRunResult(eval_results=[
            EvalResult(name="test", output=True, reason="", runtime=100, output_type="boolean", eval_id="1")
        ])

        record = history.save_run(results)

        # Delete
        success = history.delete_run(record.run_id)
        assert success is True

        # Verify deleted
        loaded = history.get_run(record.run_id)
        assert loaded is None

    def test_clear_history(self, tmp_path: Path):
        """Test clearing all history."""
        storage_dir = tmp_path / ".fi" / "runs"
        history = RunHistory(storage_dir)

        from fi.evals.types import BatchRunResult, EvalResult

        # Save some runs
        for i in range(3):
            results = BatchRunResult(eval_results=[
                EvalResult(name=f"test_{i}", output=True, reason="", runtime=100, output_type="boolean", eval_id=str(i))
            ])
            history.save_run(results)

        # Clear
        count = history.clear_history()
        assert count == 3

        # Verify empty
        runs = history.list_runs()
        assert len(runs) == 0
