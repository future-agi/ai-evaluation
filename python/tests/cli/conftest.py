"""Pytest configuration and fixtures for CLI tests."""

import os
import pytest
from pathlib import Path


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Set mock API keys for testing."""
    monkeypatch.setenv("FI_API_KEY", "test_api_key")
    monkeypatch.setenv("FI_SECRET_KEY", "test_secret_key")


@pytest.fixture
def clean_env(monkeypatch):
    """Remove API keys from environment."""
    monkeypatch.delenv("FI_API_KEY", raising=False)
    monkeypatch.delenv("FI_SECRET_KEY", raising=False)


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Create a sample configuration file."""
    config_content = """version: "1.0"

evaluations:
  - name: "test_eval"
    template: "groundedness"
    data: "./data/test.json"
"""
    config_path = tmp_path / "fi-evaluation.yaml"
    config_path.write_text(config_content)

    # Create data directory and file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "test.json"
    data_file.write_text('[{"query": "test", "response": "test", "context": "test context"}]')

    return config_path


@pytest.fixture
def sample_test_data(tmp_path: Path) -> Path:
    """Create sample test data file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / "test_cases.json"
    data_path.write_text("""[
  {
    "query": "What is machine learning?",
    "response": "Machine learning is a subset of AI.",
    "context": "Machine learning is a branch of artificial intelligence."
  }
]""")
    return data_path
