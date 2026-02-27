"""
Tests for Structured Output Validation Metrics.

Tests cover:
- JSON validation and schema compliance
- YAML validation
- Field completeness
- Hierarchy comparison
- Composite metrics
- Real-world use cases
"""

import pytest
import json


class TestJSONValidator:
    """Test JSONValidator class."""

    def test_valid_json_syntax(self):
        """Test basic JSON syntax validation."""
        from fi.evals.metrics.structured import JSONValidator

        validator = JSONValidator()
        result = validator.validate_syntax('{"name": "Alice", "age": 30}')

        assert result.syntax_valid is True
        assert result.valid is True
        assert result.parsed == {"name": "Alice", "age": 30}

    def test_invalid_json_syntax(self):
        """Test invalid JSON detection."""
        from fi.evals.metrics.structured import JSONValidator

        validator = JSONValidator()
        result = validator.validate_syntax('{"name": "Alice", "age": }')

        assert result.syntax_valid is False
        assert result.valid is False
        assert len(result.errors) > 0
        assert result.errors[0].error_type == "syntax"

    def test_json_schema_validation_valid(self):
        """Test JSON Schema validation with valid data."""
        from fi.evals.metrics.structured import JSONValidator, ValidationMode

        validator = JSONValidator()
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        result = validator.validate_schema(
            '{"name": "Alice", "age": 30}',
            schema,
            ValidationMode.COERCE,
        )

        assert result.schema_valid is True
        assert result.completeness == 1.0

    def test_json_schema_validation_missing_field(self):
        """Test schema validation with missing required field."""
        from fi.evals.metrics.structured import JSONValidator, ValidationMode

        validator = JSONValidator()
        schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
        }

        result = validator.validate_schema(
            '{"name": "Alice", "age": 30}',
            schema,
            ValidationMode.COERCE,
        )

        assert result.schema_valid is False
        assert result.completeness == pytest.approx(2 / 3, rel=0.01)

    def test_json_schema_validation_type_error(self):
        """Test schema validation with type error."""
        from fi.evals.metrics.structured import JSONValidator, ValidationMode

        validator = JSONValidator()
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
            },
        }

        result = validator.validate_schema(
            '{"age": "thirty"}',
            schema,
            ValidationMode.STRICT,
        )

        assert result.schema_valid is False
        type_errors = [e for e in result.errors if e.error_type == "type"]
        assert len(type_errors) > 0

    def test_compare_equal_values(self):
        """Test comparison with matching values."""
        from fi.evals.metrics.structured import JSONValidator, ValidationMode

        validator = JSONValidator()
        expected = {"name": "Alice", "score": 95}

        result = validator.compare(
            '{"name": "Alice", "score": 95}',
            expected,
            ValidationMode.COERCE,
        )

        assert result.valid is True
        assert len(result.errors) == 0

    def test_compare_value_mismatch(self):
        """Test comparison with mismatched values."""
        from fi.evals.metrics.structured import JSONValidator, ValidationMode

        validator = JSONValidator()
        expected = {"name": "Alice", "score": 95}

        result = validator.compare(
            '{"name": "Bob", "score": 95}',
            expected,
            ValidationMode.STRICT,
        )

        assert result.valid is False
        value_errors = [e for e in result.errors if e.error_type == "value"]
        assert len(value_errors) > 0


class TestYAMLValidator:
    """Test YAMLValidator class."""

    def test_valid_yaml_syntax(self):
        """Test basic YAML syntax validation."""
        pytest.importorskip("yaml")
        from fi.evals.metrics.structured import YAMLValidator

        validator = YAMLValidator()
        yaml_content = """
name: Alice
age: 30
skills:
  - python
  - javascript
"""
        result = validator.validate_syntax(yaml_content)

        assert result.syntax_valid is True
        assert result.parsed["name"] == "Alice"
        assert result.parsed["age"] == 30
        assert "python" in result.parsed["skills"]

    def test_invalid_yaml_syntax(self):
        """Test invalid YAML detection."""
        pytest.importorskip("yaml")
        from fi.evals.metrics.structured import YAMLValidator

        validator = YAMLValidator()
        result = validator.validate_syntax("name: 'unclosed string")

        assert result.syntax_valid is False
        assert len(result.errors) > 0

    def test_yaml_schema_validation(self):
        """Test YAML validation against JSON Schema."""
        pytest.importorskip("yaml")
        from fi.evals.metrics.structured import YAMLValidator, ValidationMode

        validator = YAMLValidator()
        schema = {
            "type": "object",
            "required": ["name", "config"],
            "properties": {
                "name": {"type": "string"},
                "config": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                    },
                },
            },
        }

        yaml_content = """
name: my-service
config:
  enabled: true
  port: 8080
"""
        result = validator.validate_schema(yaml_content, schema, ValidationMode.COERCE)

        assert result.schema_valid is True
        assert result.completeness == 1.0


class TestPydanticValidator:
    """Test PydanticValidator class."""

    def test_pydantic_validation_valid(self):
        """Test Pydantic model validation with valid data."""
        from pydantic import BaseModel
        from fi.evals.metrics.structured import PydanticValidator

        class User(BaseModel):
            name: str
            age: int
            email: str = "default@example.com"

        validator = PydanticValidator(model_class=User)
        result = validator.validate_model(
            '{"name": "Alice", "age": 30}',
            User,
        )

        assert result.valid is True
        assert result.parsed["name"] == "Alice"
        assert result.parsed["email"] == "default@example.com"

    def test_pydantic_validation_missing_required(self):
        """Test Pydantic validation with missing required field."""
        from pydantic import BaseModel
        from fi.evals.metrics.structured import PydanticValidator

        class User(BaseModel):
            name: str
            age: int

        validator = PydanticValidator(model_class=User)
        result = validator.validate_model('{"name": "Alice"}', User)

        assert result.valid is False
        missing_errors = [e for e in result.errors if e.error_type == "missing"]
        assert len(missing_errors) > 0

    def test_pydantic_nested_model(self):
        """Test Pydantic validation with nested models."""
        from pydantic import BaseModel
        from typing import List
        from fi.evals.metrics.structured import PydanticValidator

        class Address(BaseModel):
            city: str
            country: str

        class Person(BaseModel):
            name: str
            addresses: List[Address]

        validator = PydanticValidator(model_class=Person)
        result = validator.validate_model(
            '{"name": "Alice", "addresses": [{"city": "NYC", "country": "USA"}]}',
            Person,
        )

        assert result.valid is True
        assert result.parsed["addresses"][0]["city"] == "NYC"


class TestJSONValidationMetric:
    """Test JSONValidation metric."""

    def test_valid_json_full_score(self):
        """Test that valid JSON matching schema gets full score."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        result = metric.evaluate([{
            "response": '{"name": "Alice", "age": 30}',
            "schema": {
                "type": "object",
                "required": ["name", "age"],
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        }])

        assert result.eval_results[0].output == 1.0

    def test_invalid_json_zero_score(self):
        """Test that invalid JSON syntax gets zero score."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        result = metric.evaluate([{
            "response": '{"name": "Alice", age: 30}',  # Missing quotes
            "schema": {"type": "object"},
        }])

        assert result.eval_results[0].output == 0.0

    def test_partial_compliance_partial_score(self):
        """Test that partial schema compliance gets partial score."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        result = metric.evaluate([{
            "response": '{"name": "Alice"}',
            "schema": {
                "type": "object",
                "required": ["name", "age"],
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        }])

        score = result.eval_results[0].output
        assert 0.0 < score < 1.0


class TestSchemaCompliance:
    """Test SchemaCompliance metric."""

    def test_full_compliance(self):
        """Test full schema compliance."""
        from fi.evals.metrics.structured import SchemaCompliance

        metric = SchemaCompliance()
        result = metric.evaluate([{
            "response": '{"id": 123, "name": "test", "active": true}',
            "format": "json",
            "schema": {
                "type": "object",
                "required": ["id", "name", "active"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "active": {"type": "boolean"},
                },
            },
        }])

        assert result.eval_results[0].output == 1.0

    def test_compliance_breakdown(self):
        """Test compliance breakdown is included in result."""
        from fi.evals.metrics.structured import SchemaCompliance

        metric = SchemaCompliance()
        result = metric.evaluate([{
            "response": '{"id": "not-an-int", "name": "test"}',
            "format": "json",
            "schema": {
                "type": "object",
                "required": ["id", "name", "active"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "active": {"type": "boolean"},
                },
            },
        }])

        # Should have partial score due to type error and missing field
        score = result.eval_results[0].output
        assert 0.0 < score < 1.0


class TestFieldCompleteness:
    """Test FieldCompleteness metric."""

    def test_all_required_present(self):
        """Test when all required fields are present."""
        from fi.evals.metrics.structured import FieldCompleteness

        metric = FieldCompleteness()
        result = metric.evaluate([{
            "response": '{"id": 1, "name": "Alice", "email": "a@b.com"}',
            "format": "json",
            "schema": {
                "type": "object",
                "required": ["id", "name", "email"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
            },
        }])

        assert result.eval_results[0].output >= 0.8  # High score

    def test_missing_required_fields(self):
        """Test when some required fields are missing."""
        from fi.evals.metrics.structured import FieldCompleteness

        metric = FieldCompleteness()
        result = metric.evaluate([{
            "response": '{"id": 1}',
            "format": "json",
            "schema": {
                "type": "object",
                "required": ["id", "name", "email"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
            },
        }])

        score = result.eval_results[0].output
        assert score < 0.5  # Low score due to missing fields


class TestHierarchyScore:
    """Test HierarchyScore metric."""

    def test_identical_structure(self):
        """Test identical structure gets perfect score."""
        from fi.evals.metrics.structured import HierarchyScore

        metric = HierarchyScore()
        expected = {"user": {"name": "Alice"}, "items": [1, 2]}

        result = metric.evaluate([{
            "response": '{"user": {"name": "Bob"}, "items": [3, 4]}',
            "expected": expected,
        }])

        # Structure is identical (different values don't matter)
        assert result.eval_results[0].output >= 0.8

    def test_different_structure(self):
        """Test different structure gets lower score."""
        from fi.evals.metrics.structured import HierarchyScore

        metric = HierarchyScore()
        expected = {"user": {"name": "Alice", "email": "a@b.com"}, "orders": []}

        result = metric.evaluate([{
            "response": '{"profile": {"username": "Bob"}}',
            "expected": expected,
        }])

        # Very different structure
        score = result.eval_results[0].output
        assert score < 0.5


class TestTreeEditDistance:
    """Test TreeEditDistance metric."""

    def test_identical_trees_zero_distance(self):
        """Test identical trees have zero edit distance."""
        from fi.evals.metrics.structured import TreeEditDistance

        metric = TreeEditDistance()
        expected = {"a": 1, "b": 2}

        result = metric.evaluate([{
            "response": '{"a": 1, "b": 2}',
            "expected": expected,
        }])

        # Identical = 0 distance
        assert result.eval_results[0].output == 0.0

    def test_different_trees_positive_distance(self):
        """Test different trees have positive edit distance."""
        from fi.evals.metrics.structured import TreeEditDistance

        metric = TreeEditDistance()
        expected = {"a": 1, "b": 2, "c": 3}

        result = metric.evaluate([{
            "response": '{"a": 1, "d": 4}',
            "expected": expected,
        }])

        # Different structure = positive distance
        score = result.eval_results[0].output
        assert score > 0.0


class TestStructuredOutputScore:
    """Test StructuredOutputScore composite metric."""

    def test_perfect_output(self):
        """Test perfect structured output gets high score."""
        from fi.evals.metrics.structured import StructuredOutputScore

        metric = StructuredOutputScore()
        result = metric.evaluate([{
            "response": '{"name": "Alice", "age": 30}',
            "format": "json",
            "schema": {
                "type": "object",
                "required": ["name", "age"],
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        }])

        assert result.eval_results[0].output >= 0.9

    def test_invalid_syntax_zero_score(self):
        """Test invalid syntax gets zero score."""
        from fi.evals.metrics.structured import StructuredOutputScore

        metric = StructuredOutputScore()
        result = metric.evaluate([{
            "response": 'not valid json at all',
            "format": "json",
        }])

        assert result.eval_results[0].output == 0.0

    def test_breakdown_included(self):
        """Test that score breakdown is included."""
        from fi.evals.metrics.structured import StructuredOutputScore

        metric = StructuredOutputScore()
        result = metric.evaluate([{
            "response": '{"name": "Alice"}',
            "format": "json",
            "schema": {
                "type": "object",
                "required": ["name", "age"],
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        }])

        # Result should have partial score
        score = result.eval_results[0].output
        assert 0.0 < score < 1.0


class TestQuickStructuredCheck:
    """Test QuickStructuredCheck metric."""

    def test_valid_json_passes(self):
        """Test valid JSON passes quick check."""
        from fi.evals.metrics.structured import QuickStructuredCheck

        metric = QuickStructuredCheck()
        result = metric.evaluate([{
            "response": '{"key": "value"}',
        }])

        assert result.eval_results[0].output >= 0.5

    def test_invalid_json_fails(self):
        """Test invalid JSON fails quick check."""
        from fi.evals.metrics.structured import QuickStructuredCheck

        metric = QuickStructuredCheck()
        result = metric.evaluate([{
            "response": '{invalid}',
        }])

        assert result.eval_results[0].output == 0.0


# ============================================================================
# Real-World Use Cases
# ============================================================================


class TestRealWorldAPIResponse:
    """Test validation of real-world API response formats."""

    def test_rest_api_user_response(self):
        """Test validating a typical REST API user response."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        api_response = json.dumps({
            "id": 12345,
            "username": "john_doe",
            "email": "john@example.com",
            "profile": {
                "first_name": "John",
                "last_name": "Doe",
                "avatar_url": "https://example.com/avatar.jpg",
            },
            "settings": {
                "notifications_enabled": True,
                "theme": "dark",
            },
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-20T14:45:00Z",
        })

        user_schema = {
            "type": "object",
            "required": ["id", "username", "email"],
            "properties": {
                "id": {"type": "integer"},
                "username": {"type": "string", "minLength": 3},
                "email": {"type": "string"},
                "profile": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "avatar_url": {"type": "string"},
                    },
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "notifications_enabled": {"type": "boolean"},
                        "theme": {"type": "string"},
                    },
                },
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"},
            },
        }

        result = metric.evaluate([{
            "response": api_response,
            "schema": user_schema,
        }])

        assert result.eval_results[0].output == 1.0

    def test_graphql_response_validation(self):
        """Test validating a GraphQL-style response."""
        from fi.evals.metrics.structured import StructuredOutputScore

        metric = StructuredOutputScore()
        graphql_response = json.dumps({
            "data": {
                "user": {
                    "id": "user_123",
                    "name": "Alice",
                    "posts": [
                        {"id": "post_1", "title": "Hello World"},
                        {"id": "post_2", "title": "Second Post"},
                    ],
                },
            },
            "errors": None,
        })

        schema = {
            "type": "object",
            "required": ["data"],
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "required": ["id", "name"],
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "posts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "title": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "errors": {},
            },
        }

        result = metric.evaluate([{
            "response": graphql_response,
            "format": "json",
            "schema": schema,
        }])

        assert result.eval_results[0].output >= 0.9


class TestRealWorldLLMOutputs:
    """Test validation of common LLM output formats."""

    def test_function_calling_output(self):
        """Test validating LLM function calling output."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        function_call = json.dumps({
            "name": "get_weather",
            "arguments": {
                "location": "San Francisco, CA",
                "unit": "celsius",
            },
        })

        schema = {
            "type": "object",
            "required": ["name", "arguments"],
            "properties": {
                "name": {"type": "string"},
                "arguments": {
                    "type": "object",
                    "required": ["location"],
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                },
            },
        }

        result = metric.evaluate([{
            "response": function_call,
            "schema": schema,
        }])

        assert result.eval_results[0].output == 1.0

    def test_chain_of_thought_structured_output(self):
        """Test validating structured chain-of-thought output."""
        from fi.evals.metrics.structured import FieldCompleteness

        metric = FieldCompleteness()
        cot_output = json.dumps({
            "thinking": [
                "First, I need to understand the problem.",
                "The key insight is that we can use dynamic programming.",
                "Time complexity will be O(n^2).",
            ],
            "answer": 42,
            "confidence": 0.95,
            "reasoning_type": "mathematical",
        })

        schema = {
            "type": "object",
            "required": ["thinking", "answer", "confidence"],
            "properties": {
                "thinking": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "answer": {},
                "confidence": {"type": "number"},
                "reasoning_type": {"type": "string"},
            },
        }

        result = metric.evaluate([{
            "response": cot_output,
            "format": "json",
            "schema": schema,
        }])

        assert result.eval_results[0].output >= 0.9

    def test_llm_classification_output(self):
        """Test validating LLM classification output."""
        from fi.evals.metrics.structured import SchemaCompliance

        metric = SchemaCompliance()
        classification = json.dumps({
            "label": "positive",
            "confidence": 0.87,
            "all_scores": {
                "positive": 0.87,
                "negative": 0.08,
                "neutral": 0.05,
            },
        })

        schema = {
            "type": "object",
            "required": ["label", "confidence"],
            "properties": {
                "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "all_scores": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
            },
        }

        result = metric.evaluate([{
            "response": classification,
            "format": "json",
            "schema": schema,
        }])

        assert result.eval_results[0].output == 1.0

    def test_entity_extraction_output(self):
        """Test validating entity extraction output."""
        from fi.evals.metrics.structured import HierarchyScore

        metric = HierarchyScore()

        llm_output = json.dumps({
            "entities": [
                {"text": "Apple Inc.", "type": "ORG", "start": 0, "end": 10},
                {"text": "Tim Cook", "type": "PERSON", "start": 15, "end": 23},
                {"text": "California", "type": "LOC", "start": 40, "end": 50},
            ],
            "relationships": [
                {"subject": "Tim Cook", "predicate": "CEO_OF", "object": "Apple Inc."},
            ],
        })

        expected_structure = {
            "entities": [
                {"text": "", "type": "", "start": 0, "end": 0},
            ],
            "relationships": [
                {"subject": "", "predicate": "", "object": ""},
            ],
        }

        result = metric.evaluate([{
            "response": llm_output,
            "expected": expected_structure,
        }])

        # Structure should match well
        assert result.eval_results[0].output >= 0.7


class TestRealWorldConfigFiles:
    """Test validation of configuration file formats."""

    def test_yaml_kubernetes_config(self):
        """Test validating Kubernetes-style YAML config."""
        pytest.importorskip("yaml")
        from fi.evals.metrics.structured import SchemaCompliance

        metric = SchemaCompliance()
        k8s_config = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    spec:
      containers:
      - name: web
        image: nginx:latest
        ports:
        - containerPort: 80
"""

        schema = {
            "type": "object",
            "required": ["apiVersion", "kind", "metadata", "spec"],
            "properties": {
                "apiVersion": {"type": "string"},
                "kind": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "labels": {"type": "object"},
                    },
                },
                "spec": {
                    "type": "object",
                    "properties": {
                        "replicas": {"type": "integer"},
                        "selector": {"type": "object"},
                        "template": {"type": "object"},
                    },
                },
            },
        }

        result = metric.evaluate([{
            "response": k8s_config,
            "format": "yaml",
            "schema": schema,
        }])

        assert result.eval_results[0].output >= 0.9

    def test_json_package_config(self):
        """Test validating package.json-style config."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        package_json = json.dumps({
            "name": "my-package",
            "version": "1.0.0",
            "description": "A sample package",
            "main": "index.js",
            "scripts": {
                "test": "jest",
                "build": "tsc",
            },
            "dependencies": {
                "lodash": "^4.17.21",
            },
            "devDependencies": {
                "typescript": "^5.0.0",
            },
        })

        schema = {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string", "pattern": "^[a-z0-9-]+$"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "main": {"type": "string"},
                "scripts": {"type": "object"},
                "dependencies": {"type": "object"},
                "devDependencies": {"type": "object"},
            },
        }

        result = metric.evaluate([{
            "response": package_json,
            "schema": schema,
        }])

        assert result.eval_results[0].output == 1.0


class TestRealWorldECommerceScenarios:
    """Test validation in e-commerce scenarios."""

    def test_product_catalog_response(self):
        """Test validating product catalog API response."""
        from fi.evals.metrics.structured import StructuredOutputScore

        metric = StructuredOutputScore()
        product_response = json.dumps({
            "products": [
                {
                    "id": "prod_123",
                    "name": "Wireless Headphones",
                    "price": {
                        "amount": 99.99,
                        "currency": "USD",
                    },
                    "in_stock": True,
                    "categories": ["electronics", "audio"],
                    "ratings": {
                        "average": 4.5,
                        "count": 128,
                    },
                },
            ],
            "pagination": {
                "page": 1,
                "per_page": 20,
                "total": 156,
            },
        })

        schema = {
            "type": "object",
            "required": ["products", "pagination"],
            "properties": {
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "name", "price"],
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "price": {
                                "type": "object",
                                "required": ["amount", "currency"],
                                "properties": {
                                    "amount": {"type": "number"},
                                    "currency": {"type": "string"},
                                },
                            },
                            "in_stock": {"type": "boolean"},
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "ratings": {
                                "type": "object",
                                "properties": {
                                    "average": {"type": "number"},
                                    "count": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
                "pagination": {
                    "type": "object",
                    "required": ["page", "total"],
                    "properties": {
                        "page": {"type": "integer"},
                        "per_page": {"type": "integer"},
                        "total": {"type": "integer"},
                    },
                },
            },
        }

        result = metric.evaluate([{
            "response": product_response,
            "format": "json",
            "schema": schema,
        }])

        assert result.eval_results[0].output >= 0.95

    def test_order_submission_validation(self):
        """Test validating order submission structure."""
        from fi.evals.metrics.structured import FieldCompleteness

        metric = FieldCompleteness()
        order = json.dumps({
            "order_id": "ord_abc123",
            "customer": {
                "id": "cust_456",
                "email": "customer@example.com",
            },
            "items": [
                {"product_id": "prod_123", "quantity": 2, "price": 99.99},
            ],
            "shipping": {
                "address": {
                    "line1": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip": "94102",
                    "country": "US",
                },
                "method": "standard",
            },
            "payment": {
                "method": "card",
                "status": "paid",
            },
            "total": 199.98,
        })

        schema = {
            "type": "object",
            "required": ["order_id", "customer", "items", "shipping", "payment", "total"],
            "properties": {
                "order_id": {"type": "string"},
                "customer": {
                    "type": "object",
                    "required": ["id", "email"],
                    "properties": {
                        "id": {"type": "string"},
                        "email": {"type": "string"},
                    },
                },
                "items": {
                    "type": "array",
                    "minItems": 1,
                },
                "shipping": {
                    "type": "object",
                    "required": ["address", "method"],
                },
                "payment": {
                    "type": "object",
                    "required": ["method", "status"],
                },
                "total": {"type": "number"},
            },
        }

        result = metric.evaluate([{
            "response": order,
            "format": "json",
            "schema": schema,
        }])

        # All required fields are present (8/8), but no optional fields defined
        # With required_weight=0.8, optional_weight=0.2, score = 0.8 * 1.0 + 0.2 * 1.0 = 0.8
        # when there are no optional fields to count
        assert result.eval_results[0].output >= 0.8


class TestRealWorldMLScenarios:
    """Test validation in ML/AI scenarios."""

    def test_model_prediction_output(self):
        """Test validating ML model prediction output."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        prediction = json.dumps({
            "model_id": "sentiment-v2",
            "model_version": "2.1.0",
            "prediction": {
                "class": "positive",
                "probabilities": {
                    "positive": 0.92,
                    "negative": 0.05,
                    "neutral": 0.03,
                },
            },
            "metadata": {
                "latency_ms": 45,
                "tokens_processed": 128,
            },
        })

        schema = {
            "type": "object",
            "required": ["model_id", "prediction"],
            "properties": {
                "model_id": {"type": "string"},
                "model_version": {"type": "string"},
                "prediction": {
                    "type": "object",
                    "required": ["class", "probabilities"],
                    "properties": {
                        "class": {"type": "string"},
                        "probabilities": {"type": "object"},
                    },
                },
                "metadata": {"type": "object"},
            },
        }

        result = metric.evaluate([{
            "response": prediction,
            "schema": schema,
        }])

        assert result.eval_results[0].output == 1.0

    def test_rag_retrieval_output(self):
        """Test validating RAG retrieval output."""
        from fi.evals.metrics.structured import SchemaCompliance

        metric = SchemaCompliance()
        rag_output = json.dumps({
            "query": "What is machine learning?",
            "retrieved_documents": [
                {
                    "id": "doc_1",
                    "content": "Machine learning is a subset of AI...",
                    "score": 0.95,
                    "metadata": {"source": "wikipedia", "date": "2024-01-01"},
                },
                {
                    "id": "doc_2",
                    "content": "ML algorithms learn from data...",
                    "score": 0.88,
                    "metadata": {"source": "textbook", "date": "2023-06-15"},
                },
            ],
            "generated_answer": "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        })

        schema = {
            "type": "object",
            "required": ["query", "retrieved_documents", "generated_answer"],
            "properties": {
                "query": {"type": "string"},
                "retrieved_documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "content", "score"],
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "score": {"type": "number"},
                            "metadata": {"type": "object"},
                        },
                    },
                },
                "generated_answer": {"type": "string"},
            },
        }

        result = metric.evaluate([{
            "response": rag_output,
            "format": "json",
            "schema": schema,
        }])

        assert result.eval_results[0].output == 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_response(self):
        """Test handling of empty response."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        result = metric.evaluate([{"response": "", "schema": {"type": "object"}}])

        assert result.eval_results[0].output == 0.0

    def test_whitespace_only_response(self):
        """Test handling of whitespace-only response."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        result = metric.evaluate([{"response": "   \n\t  ", "schema": {"type": "object"}}])

        assert result.eval_results[0].output == 0.0

    def test_deeply_nested_structure(self):
        """Test handling of deeply nested structures."""
        from fi.evals.metrics.structured import HierarchyScore

        metric = HierarchyScore()
        deep_nested = json.dumps({
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {"value": "deep"},
                        },
                    },
                },
            },
        })

        expected = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {"value": "expected"},
                        },
                    },
                },
            },
        }

        result = metric.evaluate([{
            "response": deep_nested,
            "expected": expected,
        }])

        # Structure matches even if values differ
        assert result.eval_results[0].output >= 0.8

    def test_large_array_handling(self):
        """Test handling of large arrays."""
        from fi.evals.metrics.structured import FieldCoverage

        metric = FieldCoverage()
        large_array = json.dumps({
            "items": list(range(1000)),
            "total": 1000,
        })

        expected = {
            "items": list(range(1000)),
            "total": 1000,
        }

        result = metric.evaluate([{
            "response": large_array,
            "expected": expected,
        }])

        assert result.eval_results[0].output >= 0.9

    def test_unicode_content(self):
        """Test handling of Unicode content."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        unicode_json = json.dumps({
            "greeting": "こんにちは",
            "emoji": "🎉",
            "arabic": "مرحبا",
            "math": "∑∏∫",
        })

        result = metric.evaluate([{
            "response": unicode_json,
        }])

        assert result.eval_results[0].output == 1.0

    def test_null_values_handling(self):
        """Test proper handling of null values."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        with_nulls = json.dumps({
            "name": "Test",
            "optional_field": None,
            "nested": {"value": None},
        })

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "optional_field": {"type": ["string", "null"]},
                "nested": {"type": "object"},
            },
        }

        result = metric.evaluate([{
            "response": with_nulls,
            "schema": schema,
        }])

        assert result.eval_results[0].output >= 0.8

    def test_batch_processing(self):
        """Test batch processing of multiple inputs."""
        from fi.evals.metrics.structured import JSONValidation

        metric = JSONValidation()
        inputs = [
            {"response": '{"valid": true}'},
            {"response": '{"also": "valid"}'},
            {"response": 'invalid json'},
            {"response": '{"another": 1}'},
        ]

        result = metric.evaluate(inputs)

        assert len(result.eval_results) == 4
        assert result.eval_results[0].output == 1.0
        assert result.eval_results[1].output == 1.0
        assert result.eval_results[2].output == 0.0
        assert result.eval_results[3].output == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
