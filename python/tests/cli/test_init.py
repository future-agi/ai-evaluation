"""Tests for the init command."""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from fi.cli.main import app


runner = CliRunner()


class TestInitCommand:
    """Tests for fi init command."""

    def test_init_creates_config_file(self, tmp_path: Path):
        """Test that init creates a configuration file."""
        result = runner.invoke(app, ["init", str(tmp_path)])

        assert result.exit_code == 0
        assert (tmp_path / "fi-evaluation.yaml").exists()

    def test_init_creates_data_directory(self, tmp_path: Path):
        """Test that init creates a data directory."""
        result = runner.invoke(app, ["init", str(tmp_path)])

        assert result.exit_code == 0
        assert (tmp_path / "data").is_dir()

    def test_init_creates_results_directory(self, tmp_path: Path):
        """Test that init creates a results directory."""
        result = runner.invoke(app, ["init", str(tmp_path)])

        assert result.exit_code == 0
        assert (tmp_path / "results").is_dir()

    def test_init_creates_sample_data(self, tmp_path: Path):
        """Test that init creates sample data file."""
        result = runner.invoke(app, ["init", str(tmp_path)])

        assert result.exit_code == 0
        assert (tmp_path / "data" / "test_cases.json").exists()

    def test_init_with_rag_template(self, tmp_path: Path):
        """Test init with RAG template."""
        result = runner.invoke(app, ["init", str(tmp_path), "--template", "rag"])

        assert result.exit_code == 0

        config_content = (tmp_path / "fi-evaluation.yaml").read_text()
        assert "groundedness" in config_content

    def test_init_with_safety_template(self, tmp_path: Path):
        """Test init with safety template."""
        result = runner.invoke(app, ["init", str(tmp_path), "--template", "safety"])

        assert result.exit_code == 0

        config_content = (tmp_path / "fi-evaluation.yaml").read_text()
        assert "content_moderation" in config_content

    def test_init_fails_if_config_exists(self, tmp_path: Path):
        """Test that init fails if config already exists."""
        # Create existing config
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text("existing: config")

        result = runner.invoke(app, ["init", str(tmp_path)])

        assert result.exit_code == 1
        assert "already exists" in result.stdout

    def test_init_force_overwrites(self, tmp_path: Path):
        """Test that init --force overwrites existing config."""
        # Create existing config
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text("existing: config")

        result = runner.invoke(app, ["init", str(tmp_path), "--force"])

        assert result.exit_code == 0

        new_content = config_path.read_text()
        assert "existing" not in new_content

    def test_init_unknown_template(self, tmp_path: Path):
        """Test that init fails with unknown template."""
        result = runner.invoke(app, ["init", str(tmp_path), "--template", "unknown"])

        assert result.exit_code == 1
        assert "Unknown template" in result.stdout

    def test_init_creates_nonexistent_directory(self, tmp_path: Path):
        """Test that init creates the directory if it doesn't exist."""
        new_dir = tmp_path / "new_project"

        result = runner.invoke(app, ["init", str(new_dir)])

        assert result.exit_code == 0
        assert new_dir.exists()
        assert (new_dir / "fi-evaluation.yaml").exists()
