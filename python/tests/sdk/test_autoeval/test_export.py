"""Tests for export/import functionality."""

import json
import tempfile
import pytest
from pathlib import Path

from fi.evals.autoeval.export import (
    export_json,
    load_json,
    load_config,
    to_json_string,
    from_json_string,
)
from fi.evals.autoeval.config import AutoEvalConfig, EvalConfig, ScannerConfig


class TestJsonExport:
    """Tests for JSON export functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return AutoEvalConfig(
            name="test_export",
            description="Test configuration",
            app_category="rag_system",
            risk_level="high",
            domain_sensitivity="healthcare",
            evaluations=[
                EvalConfig(name="Eval1", threshold=0.8),
                EvalConfig(name="Eval2", threshold=0.9, enabled=False),
            ],
            scanners=[
                ScannerConfig(name="Scanner1", action="block"),
                ScannerConfig(name="Scanner2", action="redact"),
            ],
            execution_mode="blocking",
            parallel_workers=8,
        )

    def test_export_json_to_file(self, sample_config):
        """Should export config to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            export_json(sample_config, path)
            assert path.exists()

            # Verify content is valid JSON
            with open(path) as f:
                data = json.load(f)
            assert data["name"] == "test_export"

    def test_load_json_from_file(self, sample_config):
        """Should load config from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            export_json(sample_config, path)

            loaded = load_json(path)
            assert loaded.name == sample_config.name
            assert len(loaded.evaluations) == len(sample_config.evaluations)
            assert len(loaded.scanners) == len(sample_config.scanners)

    def test_to_json_string(self, sample_config):
        """Should convert config to JSON string."""
        json_str = to_json_string(sample_config)
        assert isinstance(json_str, str)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["name"] == "test_export"

    def test_from_json_string(self, sample_config):
        """Should create config from JSON string."""
        json_str = to_json_string(sample_config)
        loaded = from_json_string(json_str)
        assert loaded.name == sample_config.name
        assert loaded.app_category == sample_config.app_category

    def test_json_roundtrip_preserves_data(self, sample_config):
        """Roundtrip should preserve all data."""
        json_str = to_json_string(sample_config)
        loaded = from_json_string(json_str)

        # Check all fields
        assert loaded.name == sample_config.name
        assert loaded.description == sample_config.description
        assert loaded.app_category == sample_config.app_category
        assert loaded.risk_level == sample_config.risk_level
        assert loaded.domain_sensitivity == sample_config.domain_sensitivity
        assert loaded.execution_mode == sample_config.execution_mode
        assert loaded.parallel_workers == sample_config.parallel_workers

        # Check evaluations
        assert len(loaded.evaluations) == len(sample_config.evaluations)
        for orig, load in zip(sample_config.evaluations, loaded.evaluations):
            assert load.name == orig.name
            assert load.threshold == orig.threshold
            assert load.enabled == orig.enabled

        # Check scanners
        assert len(loaded.scanners) == len(sample_config.scanners)
        for orig, load in zip(sample_config.scanners, loaded.scanners):
            assert load.name == orig.name
            assert load.action == orig.action


class TestLoadConfig:
    """Tests for auto-detecting file format."""

    @pytest.fixture
    def sample_config(self):
        return AutoEvalConfig(name="autodetect_test")

    def test_load_config_json(self, sample_config):
        """Should auto-detect JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            export_json(sample_config, path)

            loaded = load_config(path)
            assert loaded.name == sample_config.name

    def test_load_config_invalid_extension_raises(self):
        """Should raise error for unknown file extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.txt"
            path.write_text("{}")

            with pytest.raises(ValueError) as excinfo:
                load_config(path)
            assert "unknown file format" in str(excinfo.value).lower()


class TestYamlExport:
    """Tests for YAML export (if PyYAML is available)."""

    @pytest.fixture
    def sample_config(self):
        return AutoEvalConfig(
            name="yaml_test",
            evaluations=[EvalConfig(name="Eval1")],
        )

    def test_yaml_export_import_if_available(self, sample_config):
        """Should export/import YAML if PyYAML is available."""
        try:
            from fi.evals.autoeval.export import (
                export_yaml,
                load_yaml,
                to_yaml_string,
                from_yaml_string,
            )
        except ImportError:
            pytest.skip("PyYAML not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            export_yaml(sample_config, path)
            assert path.exists()

            loaded = load_yaml(path)
            assert loaded.name == sample_config.name

    def test_yaml_string_roundtrip_if_available(self, sample_config):
        """Should roundtrip YAML string if PyYAML is available."""
        try:
            from fi.evals.autoeval.export import to_yaml_string, from_yaml_string
        except ImportError:
            pytest.skip("PyYAML not installed")

        yaml_str = to_yaml_string(sample_config)
        loaded = from_yaml_string(yaml_str)
        assert loaded.name == sample_config.name

    def test_load_config_yaml_if_available(self, sample_config):
        """Should auto-detect YAML format if PyYAML available."""
        try:
            from fi.evals.autoeval.export import export_yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            export_yaml(sample_config, path)

            loaded = load_config(path)
            assert loaded.name == sample_config.name
