"""Tests for the config loader."""

import pytest
from pathlib import Path

from fi.cli.config.loader import load_config, find_config_file, load_test_data
from fi.cli.config.schema import FIEvaluationConfig


VALID_CONFIG = """version: "1.0"

api:
  base_url: "https://api.futureagi.com"

defaults:
  model: "gpt-4o"
  timeout: 200
  parallel_workers: 8

evaluations:
  - name: "test_eval"
    template: "groundedness"
    data: "./data/test.json"

output:
  format: "json"
  path: "./results/"
"""


class TestFindConfigFile:
    """Tests for find_config_file function."""

    def test_finds_config_in_current_dir(self, tmp_path: Path):
        """Test finding config in current directory."""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text("version: '1.0'")

        result = find_config_file(tmp_path)

        assert result == config_path

    def test_finds_config_with_yml_extension(self, tmp_path: Path):
        """Test finding config with .yml extension."""
        config_path = tmp_path / "fi-evaluation.yml"
        config_path.write_text("version: '1.0'")

        result = find_config_file(tmp_path)

        assert result == config_path

    def test_finds_hidden_config(self, tmp_path: Path):
        """Test finding hidden config file."""
        config_path = tmp_path / ".fi-evaluation.yaml"
        config_path.write_text("version: '1.0'")

        result = find_config_file(tmp_path)

        assert result == config_path

    def test_returns_none_when_no_config(self, tmp_path: Path):
        """Test returns None when no config exists."""
        result = find_config_file(tmp_path)

        assert result is None

    def test_finds_config_in_parent_dir(self, tmp_path: Path):
        """Test finding config in parent directory."""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text("version: '1.0'")

        child_dir = tmp_path / "child"
        child_dir.mkdir()

        result = find_config_file(child_dir)

        assert result == config_path


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_valid_config(self, tmp_path: Path):
        """Test loading a valid config file."""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text(VALID_CONFIG)

        config = load_config(config_path)

        assert isinstance(config, FIEvaluationConfig)
        assert config.version == "1.0"
        assert len(config.evaluations) == 1

    def test_raises_on_missing_file(self, tmp_path: Path):
        """Test raises FileNotFoundError for missing file."""
        config_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_config(config_path)

    def test_raises_on_invalid_yaml(self, tmp_path: Path):
        """Test raises ValueError for invalid YAML."""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text("invalid: yaml: content:")

        with pytest.raises(ValueError):
            load_config(config_path)

    def test_raises_on_empty_file(self, tmp_path: Path):
        """Test raises ValueError for empty file."""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text("")

        with pytest.raises(ValueError):
            load_config(config_path)

    def test_raises_on_missing_evaluations(self, tmp_path: Path):
        """Test raises ValueError when evaluations missing."""
        config_path = tmp_path / "fi-evaluation.yaml"
        config_path.write_text("version: '1.0'")

        with pytest.raises(ValueError):
            load_config(config_path)


class TestLoadTestData:
    """Tests for load_test_data function."""

    def test_loads_json_file(self, tmp_path: Path):
        """Test loading JSON test data."""
        data_path = tmp_path / "test.json"
        data_path.write_text('[{"query": "test", "response": "answer"}]')

        data = load_test_data(data_path)

        assert len(data) == 1
        assert data[0]["query"] == "test"

    def test_loads_jsonl_file(self, tmp_path: Path):
        """Test loading JSONL test data."""
        data_path = tmp_path / "test.jsonl"
        data_path.write_text('{"query": "test1"}\n{"query": "test2"}')

        data = load_test_data(data_path)

        assert len(data) == 2

    def test_loads_csv_file(self, tmp_path: Path):
        """Test loading CSV test data."""
        data_path = tmp_path / "test.csv"
        data_path.write_text("query,response\ntest1,answer1\ntest2,answer2")

        data = load_test_data(data_path)

        assert len(data) == 2
        assert data[0]["query"] == "test1"

    def test_raises_on_missing_file(self, tmp_path: Path):
        """Test raises FileNotFoundError for missing file."""
        data_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_test_data(data_path)

    def test_raises_on_unsupported_format(self, tmp_path: Path):
        """Test raises ValueError for unsupported format."""
        data_path = tmp_path / "test.txt"
        data_path.write_text("some text")

        with pytest.raises(ValueError):
            load_test_data(data_path)

    def test_wraps_single_object_in_list(self, tmp_path: Path):
        """Test that single JSON object is wrapped in list."""
        data_path = tmp_path / "test.json"
        data_path.write_text('{"query": "test", "response": "answer"}')

        data = load_test_data(data_path)

        assert isinstance(data, list)
        assert len(data) == 1
