"""
JSON Schema Validation Metric.

Evaluates whether LLM-generated JSON conforms to a schema.
"""

from typing import Any, Dict, Optional

from ..base_metric import BaseMetric
from .types import JSONInput, ValidationMode
from .validators import JSONValidator


class JSONValidation(BaseMetric[JSONInput]):
    """
    Evaluates JSON output against a JSON Schema.

    This metric checks:
    1. Syntax validity (is it parseable JSON?)
    2. Schema compliance (does it match the schema?)
    3. Value correctness (optional, if expected provided)

    Score: 0.0 (invalid) to 1.0 (fully valid)

    Example:
        >>> metric = JSONValidation()
        >>> result = metric.evaluate([{
        ...     "response": '{"name": "John", "age": 30}',
        ...     "schema": {
        ...         "type": "object",
        ...         "required": ["name", "age"],
        ...         "properties": {
        ...             "name": {"type": "string"},
        ...             "age": {"type": "integer"}
        ...         }
        ...     }
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "json_validation"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.validator = JSONValidator()
        self.syntax_weight = self.config.get("syntax_weight", 0.3)
        self.schema_weight = self.config.get("schema_weight", 0.5)
        self.completeness_weight = self.config.get("completeness_weight", 0.2)

    def compute_one(self, inputs: JSONInput) -> Dict[str, Any]:
        response = inputs.response
        schema = inputs.schema
        expected = inputs.expected
        mode_str = inputs.mode

        # Convert mode string to enum
        try:
            mode = ValidationMode(mode_str)
        except ValueError:
            mode = ValidationMode.COERCE

        if not response or not response.strip():
            return {
                "output": 0.0,
                "reason": "Empty response",
            }

        # First check syntax
        syntax_result = self.validator.validate_syntax(response)
        if not syntax_result.syntax_valid:
            errors = [e.dict() for e in syntax_result.errors]
            return {
                "output": 0.0,
                "reason": f"JSON syntax error: {syntax_result.errors[0].message if syntax_result.errors else 'Unknown'}",
                "syntax_valid": False,
                "schema_valid": False,
                "errors": errors,
            }

        # Schema validation if schema provided
        if schema:
            schema_result = self.validator.validate_schema(response, schema, mode)

            if not schema_result.schema_valid:
                errors = [e.dict() for e in schema_result.errors]
                # Calculate partial score based on completeness
                score = self.syntax_weight + (self.completeness_weight * schema_result.completeness)

                return {
                    "output": round(score, 4),
                    "reason": f"Schema validation failed: {len(schema_result.errors)} error(s)",
                    "syntax_valid": True,
                    "schema_valid": False,
                    "completeness": schema_result.completeness,
                    "errors": errors,
                    "parsed": schema_result.parsed,
                }

            # Full schema compliance
            score = self.syntax_weight + self.schema_weight + (
                self.completeness_weight * schema_result.completeness
            )

            return {
                "output": round(score, 4),
                "reason": "Valid JSON matching schema",
                "syntax_valid": True,
                "schema_valid": True,
                "completeness": schema_result.completeness,
                "parsed": schema_result.parsed,
            }

        # Value comparison if expected provided
        if expected is not None:
            compare_result = self.validator.compare(response, expected, mode)

            if not compare_result.valid:
                errors = [e.dict() for e in compare_result.errors]
                # Partial score for syntax-valid but wrong values
                score = self.syntax_weight

                return {
                    "output": round(score, 4),
                    "reason": f"Value mismatch: {len(compare_result.errors)} error(s)",
                    "syntax_valid": True,
                    "value_match": False,
                    "errors": errors,
                    "parsed": compare_result.parsed,
                }

            return {
                "output": 1.0,
                "reason": "Exact value match",
                "syntax_valid": True,
                "value_match": True,
                "parsed": compare_result.parsed,
            }

        # Only syntax checked (no schema or expected)
        return {
            "output": 1.0,
            "reason": "Valid JSON (no schema to validate against)",
            "syntax_valid": True,
            "parsed": syntax_result.parsed,
        }


class JSONSyntaxOnly(BaseMetric[JSONInput]):
    """
    Simple metric that only checks JSON syntax validity.

    Returns 1.0 for valid JSON, 0.0 for invalid.

    Example:
        >>> metric = JSONSyntaxOnly()
        >>> result = metric.evaluate([{"response": '{"valid": true}'}])
    """

    @property
    def metric_name(self) -> str:
        return "json_syntax"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.validator = JSONValidator()

    def compute_one(self, inputs: JSONInput) -> Dict[str, Any]:
        response = inputs.response

        if not response or not response.strip():
            return {
                "output": 0.0,
                "reason": "Empty response",
            }

        result = self.validator.validate_syntax(response)

        if result.syntax_valid:
            return {
                "output": 1.0,
                "reason": "Valid JSON syntax",
                "parsed": result.parsed,
            }
        else:
            return {
                "output": 0.0,
                "reason": result.errors[0].message if result.errors else "Invalid JSON",
                "errors": [e.dict() for e in result.errors],
            }
