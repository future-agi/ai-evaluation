"""
Structured Output Score - Composite Metric.

Combines multiple structured validation aspects into a single score.
"""

from typing import Any, Dict, Optional

from ..base_metric import BaseMetric
from .types import StructuredInput, JSONInput, ValidationMode
from .validators import JSONValidator, YAMLValidator


class StructuredOutputScore(BaseMetric[StructuredInput]):
    """
    Comprehensive structured output evaluation.

    Combines multiple aspects:
    - Syntax validity (parseability)
    - Schema compliance (matches expected structure)
    - Field completeness (required fields present)
    - Type correctness (values have correct types)
    - Value accuracy (optional, if expected provided)

    Score: 0.0 to 1.0 weighted combination of all aspects.

    Example:
        >>> metric = StructuredOutputScore()
        >>> result = metric.evaluate([{
        ...     "response": '{"name": "Alice", "age": 25}',
        ...     "format": "json",
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
        return "structured_output_score"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.json_validator = JSONValidator()
        self._yaml_validator = None

        # Weights for different aspects
        self.syntax_weight = self.config.get("syntax_weight", 0.2)
        self.schema_weight = self.config.get("schema_weight", 0.3)
        self.completeness_weight = self.config.get("completeness_weight", 0.25)
        self.type_weight = self.config.get("type_weight", 0.15)
        self.value_weight = self.config.get("value_weight", 0.1)

    @property
    def yaml_validator(self):
        if self._yaml_validator is None:
            try:
                self._yaml_validator = YAMLValidator()
            except ImportError:
                return None
        return self._yaml_validator

    def _get_validator(self, format_name: str):
        """Get validator for format."""
        if format_name.lower() == "yaml":
            if self.yaml_validator is None:
                raise ImportError("PyYAML is required for YAML validation")
            return self.yaml_validator
        return self.json_validator

    def compute_one(self, inputs: StructuredInput) -> Dict[str, Any]:
        response = inputs.response
        format_name = inputs.format
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
                "breakdown": {
                    "syntax": 0.0,
                    "schema": 0.0,
                    "completeness": 0.0,
                    "types": 0.0,
                    "values": 0.0,
                },
            }

        # Get validator
        try:
            validator = self._get_validator(format_name)
        except ImportError as e:
            return {
                "output": 0.0,
                "reason": str(e),
            }

        # Initialize scores
        scores = {
            "syntax": 0.0,
            "schema": 0.0,
            "completeness": 0.0,
            "types": 0.0,
            "values": 0.0,
        }

        # 1. Syntax validation
        syntax_result = validator.validate_syntax(response)
        if not syntax_result.syntax_valid:
            return {
                "output": 0.0,
                "reason": f"Syntax error: {syntax_result.errors[0].message if syntax_result.errors else 'Unknown'}",
                "breakdown": scores,
                "errors": [e.dict() for e in syntax_result.errors],
            }

        scores["syntax"] = 1.0
        parsed = syntax_result.parsed

        # 2. Schema validation (if schema provided)
        if schema:
            schema_result = validator.validate_schema(response, schema, mode)
            scores["completeness"] = schema_result.completeness

            if schema_result.schema_valid:
                scores["schema"] = 1.0
                scores["types"] = 1.0
            else:
                # Analyze errors for partial scores
                type_errors = sum(1 for e in schema_result.errors if e.error_type == "type")
                schema_errors = len(schema_result.errors) - type_errors

                # Estimate total expected validations
                total_fields = self._count_schema_fields(schema)

                if total_fields > 0:
                    scores["schema"] = max(0.0, 1.0 - schema_errors / total_fields)
                    scores["types"] = max(0.0, 1.0 - type_errors / total_fields)
        else:
            # No schema: give full marks for schema-related aspects
            scores["schema"] = 1.0
            scores["completeness"] = 1.0
            scores["types"] = 1.0

        # 3. Value accuracy (if expected provided)
        if expected is not None:
            compare_result = validator.compare(response, expected, mode)
            if compare_result.valid:
                scores["values"] = 1.0
            else:
                # Calculate partial value match
                value_errors = len(compare_result.errors)
                total_values = self._count_values(expected)
                scores["values"] = max(0.0, 1.0 - value_errors / max(total_values, 1))
        else:
            # No expected: give full marks for values
            scores["values"] = 1.0

        # Calculate weighted overall score
        overall = (
            self.syntax_weight * scores["syntax"] +
            self.schema_weight * scores["schema"] +
            self.completeness_weight * scores["completeness"] +
            self.type_weight * scores["types"] +
            self.value_weight * scores["values"]
        )

        return {
            "output": round(overall, 4),
            "reason": self._generate_reason(scores),
            "breakdown": {k: round(v, 4) for k, v in scores.items()},
            "parsed": parsed,
        }

    def _count_schema_fields(self, schema: Dict[str, Any], depth: int = 0) -> int:
        """Count fields in schema."""
        if depth > 10:
            return 1

        count = 0
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            count = len(properties)
            for prop_schema in properties.values():
                count += self._count_schema_fields(prop_schema, depth + 1)
        elif schema.get("type") == "array":
            items = schema.get("items", {})
            count = 1 + self._count_schema_fields(items, depth + 1)

        return max(count, 1)

    def _count_values(self, data: Any) -> int:
        """Count total values in data structure."""
        if isinstance(data, dict):
            return sum(self._count_values(v) for v in data.values())
        elif isinstance(data, list):
            return sum(self._count_values(item) for item in data)
        else:
            return 1

    def _generate_reason(self, scores: Dict[str, float]) -> str:
        """Generate human-readable reason."""
        issues = []
        if scores["syntax"] < 1.0:
            issues.append("syntax errors")
        if scores["schema"] < 1.0:
            issues.append("schema violations")
        if scores["completeness"] < 1.0:
            issues.append("missing fields")
        if scores["types"] < 1.0:
            issues.append("type errors")
        if scores["values"] < 1.0:
            issues.append("value mismatches")

        if not issues:
            return "Fully valid structured output"
        return f"Issues: {', '.join(issues)}"


class QuickStructuredCheck(BaseMetric[JSONInput]):
    """
    Fast, lightweight structured output check.

    Quick validation that just checks:
    1. Is it valid JSON?
    2. Does it have the expected keys?
    3. Are the types roughly correct?

    Score: 0.0 to 1.0
    """

    @property
    def metric_name(self) -> str:
        return "quick_structured_check"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.validator = JSONValidator()

    def compute_one(self, inputs: JSONInput) -> Dict[str, Any]:
        response = inputs.response
        schema = inputs.schema
        expected = inputs.expected

        if not response or not response.strip():
            return {"output": 0.0, "reason": "Empty response"}

        # Quick syntax check
        syntax_result = self.validator.validate_syntax(response)
        if not syntax_result.syntax_valid:
            return {"output": 0.0, "reason": "Invalid JSON"}

        parsed = syntax_result.parsed
        score = 0.5  # Base score for valid JSON

        # Quick schema check
        if schema:
            required = schema.get("required", [])
            if required and isinstance(parsed, dict):
                present = sum(1 for f in required if f in parsed)
                schema_score = present / len(required)
                score = 0.5 + (0.5 * schema_score)
            else:
                score = 1.0
        elif expected is not None:
            # Quick key comparison
            if isinstance(expected, dict) and isinstance(parsed, dict):
                expected_keys = set(expected.keys())
                actual_keys = set(parsed.keys())
                if expected_keys:
                    overlap = len(expected_keys & actual_keys) / len(expected_keys)
                    score = 0.5 + (0.5 * overlap)
                else:
                    score = 1.0
            else:
                score = 1.0 if type(expected) == type(parsed) else 0.5

        return {
            "output": round(score, 4),
            "reason": "Valid JSON" + (" with expected structure" if score >= 0.8 else ""),
            "parsed": parsed,
        }
