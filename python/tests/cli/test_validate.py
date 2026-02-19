"""Tests for the validate command."""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from fi.cli.main import app


runner = CliRunner()


VALID_CONFIG = """version: "1.0"

evaluations:
  - name: "test_eval"
    template: "groundedness"
    data: "./data/test.json"
"""

INVALID_YAML = """version: "1.0"
evaluations
  - name: "test"
"""

MISSING_REQUIRED = """version: "1.0"
# Missing evaluations
"""

UNKNOWN_TEMPLATE = """version: "1.0"

evaluations:
  - name: "test_eval"
    template: "unknown_template"
    data: "./data/test.json"
"""


class TestValidateCommand:
    """Tests for fi validate command."""

    def test_validate_valid_config(self, tmp_path: Path):
        """Test validation of a valid config."""
        # Create config
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text(VALID_CONFIG)

        # Create data file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        data_file = data_dir / "test.json"
        data_file.write_text('[{"query": "test", "response": "test"}]')

        result = runner.invoke(app, ["validate", "--config", str(config_path)])

        assert result.exit_code == 0
        assert "Validation passed" in result.stdout

    def test_validate_invalid_yaml(self, tmp_path: Path):
        """Test validation fails for invalid YAML."""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text(INVALID_YAML)

        result = runner.invoke(app, ["validate", "--config", str(config_path)])

        assert result.exit_code == 1

    def test_validate_missing_config(self, tmp_path: Path):
        """Test validation fails when config doesn't exist."""
        config_path = tmp_path / "nonexistent.yaml"

        result = runner.invoke(app, ["validate", "--config", str(config_path)])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_validate_unknown_template(self, tmp_path: Path):
        """Test validation fails for unknown template."""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text(UNKNOWN_TEMPLATE)

        # Create data file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        data_file = data_dir / "test.json"
        data_file.write_text('[{"query": "test"}]')

        result = runner.invoke(app, ["validate", "--config", str(config_path)])

        assert result.exit_code == 1
        assert "Unknown template" in result.stdout

    def test_validate_missing_data_file(self, tmp_path: Path):
        """Test validation fails when data file is missing."""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text(VALID_CONFIG)

        # Don't create data file

        result = runner.invoke(app, ["validate", "--config", str(config_path)])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_validate_strict_mode_fails_on_warnings(self, tmp_path: Path, monkeypatch):
        """Test that strict mode fails on warnings."""
        # Create valid config
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text(VALID_CONFIG)

        # Create data file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        data_file = data_dir / "test.json"
        data_file.write_text('[{"query": "test", "response": "test"}]')

        # Remove API keys to trigger warning
        monkeypatch.delenv("FI_API_KEY", raising=False)
        monkeypatch.delenv("FI_SECRET_KEY", raising=False)

        result = runner.invoke(app, ["validate", "--config", str(config_path), "--strict"])

        assert result.exit_code == 1
        assert "warning" in result.stdout.lower()
