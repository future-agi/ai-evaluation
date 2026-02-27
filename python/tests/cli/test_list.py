"""Tests for the list command."""

import json
import pytest
from typer.testing import CliRunner

from fi.cli.main import app


runner = CliRunner()


class TestListCommand:
    """Tests for fi list command."""

    def test_list_templates_table(self):
        """Test listing templates in table format."""
        result = runner.invoke(app, ["list", "templates"])

        assert result.exit_code == 0
        assert "groundedness" in result.stdout
        assert "context_adherence" in result.stdout

    def test_list_templates_json(self):
        """Test listing templates in JSON format."""
        result = runner.invoke(app, ["list", "templates", "--format", "json"])

        assert result.exit_code == 0

        # Parse JSON output
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) > 0

        # Check structure
        template_names = [t["name"] for t in data]
        assert "groundedness" in template_names

    def test_list_templates_by_category(self):
        """Test listing templates filtered by category."""
        result = runner.invoke(app, ["list", "templates", "--category", "rag"])

        assert result.exit_code == 0
        assert "groundedness" in result.stdout
        assert "context_adherence" in result.stdout

    def test_list_templates_unknown_category(self):
        """Test listing templates with unknown category."""
        result = runner.invoke(app, ["list", "templates", "--category", "unknown"])

        assert result.exit_code == 1
        assert "Unknown category" in result.stdout

    def test_list_categories(self):
        """Test listing categories."""
        result = runner.invoke(app, ["list", "categories"])

        assert result.exit_code == 0
        assert "rag" in result.stdout
        assert "safety" in result.stdout

    def test_list_categories_json(self):
        """Test listing categories in JSON format."""
        result = runner.invoke(app, ["list", "categories", "--format", "json"])

        assert result.exit_code == 0

        data = json.loads(result.stdout)
        assert isinstance(data, list)

        category_names = [c["name"] for c in data]
        assert "rag" in category_names
        assert "safety" in category_names

    def test_list_unknown_resource(self):
        """Test listing unknown resource."""
        result = runner.invoke(app, ["list", "unknown"])

        assert result.exit_code == 1
        assert "Unknown resource" in result.stdout
