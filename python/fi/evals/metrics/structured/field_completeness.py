"""
Field Completeness Metric.

Measures the presence of required and optional fields in structured output.
"""

from typing import Any, Dict, List, Optional, Set

from ..base_metric import BaseMetric
from .types import StructuredInput, JSONInput
from .validators import JSONValidator


class FieldCompleteness(BaseMetric[StructuredInput]):
    """
    Evaluates field completeness in structured output.

    Measures:
    - Required field presence (weighted heavily)
    - Optional field presence (weighted lightly)
    - Nested field coverage

    Score: 0.0 (no required fields) to 1.0 (all fields present)

    Example:
        >>> metric = FieldCompleteness()
        >>> result = metric.evaluate([{
        ...     "response": '{"name": "Alice", "email": "alice@example.com"}',
        ...     "format": "json",
        ...     "schema": {
        ...         "type": "object",
        ...         "required": ["name", "email", "age"],
        ...         "properties": {
        ...             "name": {"type": "string"},
        ...             "email": {"type": "string"},
        ...             "age": {"type": "integer"},
        ...             "phone": {"type": "string"}
        ...         }
        ...     }
        ... }])
        # Result: 0.67 (2/3 required fields present)
    """

    @property
    def metric_name(self) -> str:
        return "field_completeness"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.validator = JSONValidator()
        self.required_weight = self.config.get("required_weight", 0.8)
        self.optional_weight = self.config.get("optional_weight", 0.2)
        self.include_nested = self.config.get("include_nested", True)

    def compute_one(self, inputs: StructuredInput) -> Dict[str, Any]:
        response = inputs.response
        schema = inputs.schema

        if not response or not response.strip():
            return {
                "output": 0.0,
                "reason": "Empty response",
            }

        if not schema:
            return {
                "output": 0.0,
                "reason": "No schema provided for field analysis",
            }

        # Parse response
        syntax_result = self.validator.validate_syntax(response)
        if not syntax_result.syntax_valid:
            return {
                "output": 0.0,
                "reason": "Cannot parse response",
                "errors": [e.dict() for e in syntax_result.errors],
            }

        parsed = syntax_result.parsed

        # Analyze field presence
        analysis = self._analyze_fields(parsed, schema, "$")

        # Calculate score
        required_score = analysis["required_present"] / max(analysis["required_total"], 1)
        optional_score = analysis["optional_present"] / max(analysis["optional_total"], 1)

        # Weight the scores — only include components that exist
        has_required = analysis["required_total"] > 0
        has_optional = analysis["optional_total"] > 0

        if has_required and has_optional:
            score = (
                self.required_weight * required_score +
                self.optional_weight * optional_score
            )
        elif has_required:
            score = required_score
        elif has_optional:
            score = optional_score
        else:
            score = 1.0

        return {
            "output": round(score, 4),
            "reason": f"{analysis['required_present']}/{analysis['required_total']} required, {analysis['optional_present']}/{analysis['optional_total']} optional fields",
            "required_fields": {
                "present": analysis["required_present"],
                "total": analysis["required_total"],
                "missing": analysis["missing_required"],
            },
            "optional_fields": {
                "present": analysis["optional_present"],
                "total": analysis["optional_total"],
                "missing": analysis["missing_optional"],
            },
            "completeness": required_score,
            "parsed": parsed,
        }

    def _analyze_fields(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> Dict[str, Any]:
        """Analyze field presence recursively."""
        result = {
            "required_present": 0,
            "required_total": 0,
            "optional_present": 0,
            "optional_total": 0,
            "missing_required": [],
            "missing_optional": [],
        }

        if schema.get("type") != "object" or not isinstance(data, dict):
            return result

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for field, field_schema in properties.items():
            field_path = f"{path}.{field}"
            is_required = field in required
            is_present = field in data

            if is_required:
                result["required_total"] += 1
                if is_present:
                    result["required_present"] += 1
                else:
                    result["missing_required"].append(field_path)
            else:
                result["optional_total"] += 1
                if is_present:
                    result["optional_present"] += 1
                else:
                    result["missing_optional"].append(field_path)

            # Recursively analyze nested objects
            if self.include_nested and is_present and field_schema.get("type") == "object":
                nested_result = self._analyze_fields(
                    data[field], field_schema, field_path
                )
                result["required_present"] += nested_result["required_present"]
                result["required_total"] += nested_result["required_total"]
                result["optional_present"] += nested_result["optional_present"]
                result["optional_total"] += nested_result["optional_total"]
                result["missing_required"].extend(nested_result["missing_required"])
                result["missing_optional"].extend(nested_result["missing_optional"])

        return result


class RequiredFieldsOnly(BaseMetric[StructuredInput]):
    """
    Simple metric that only checks required field presence.

    Returns the fraction of required fields present.

    Score: 0.0 to 1.0
    """

    @property
    def metric_name(self) -> str:
        return "required_fields"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.validator = JSONValidator()

    def compute_one(self, inputs: StructuredInput) -> Dict[str, Any]:
        response = inputs.response
        schema = inputs.schema

        if not response or not response.strip():
            return {"output": 0.0, "reason": "Empty response"}

        if not schema:
            return {"output": 0.0, "reason": "No schema provided"}

        # Parse response
        syntax_result = self.validator.validate_syntax(response)
        if not syntax_result.syntax_valid:
            return {"output": 0.0, "reason": "Cannot parse response"}

        parsed = syntax_result.parsed

        # Get required fields
        required = schema.get("required", [])
        if not required:
            return {
                "output": 1.0,
                "reason": "No required fields in schema",
                "parsed": parsed,
            }

        # Check presence
        present = [f for f in required if isinstance(parsed, dict) and f in parsed]
        missing = [f for f in required if f not in present]

        score = len(present) / len(required)

        return {
            "output": round(score, 4),
            "reason": f"{len(present)}/{len(required)} required fields present",
            "present_fields": present,
            "missing_fields": missing,
            "parsed": parsed,
        }


class FieldCoverage(BaseMetric[JSONInput]):
    """
    Measures field coverage comparing response to expected output.

    Compares actual fields present vs expected fields without validating values.

    Score: 0.0 to 1.0 (fraction of expected fields present)
    """

    @property
    def metric_name(self) -> str:
        return "field_coverage"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.validator = JSONValidator()
        self.include_nested = self.config.get("include_nested", True)

    def compute_one(self, inputs: JSONInput) -> Dict[str, Any]:
        response = inputs.response
        expected = inputs.expected

        if not response or not response.strip():
            return {"output": 0.0, "reason": "Empty response"}

        if expected is None:
            return {"output": 0.0, "reason": "No expected output provided"}

        # Parse response
        syntax_result = self.validator.validate_syntax(response)
        if not syntax_result.syntax_valid:
            return {"output": 0.0, "reason": "Cannot parse response"}

        parsed = syntax_result.parsed

        # Extract fields from both
        expected_fields = self._extract_fields(expected, "$")
        actual_fields = self._extract_fields(parsed, "$")

        # Calculate coverage
        if not expected_fields:
            return {
                "output": 1.0,
                "reason": "No fields expected",
                "parsed": parsed,
            }

        covered = expected_fields & actual_fields
        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields

        score = len(covered) / len(expected_fields)

        return {
            "output": round(score, 4),
            "reason": f"{len(covered)}/{len(expected_fields)} expected fields present",
            "covered_fields": list(covered),
            "missing_fields": list(missing),
            "extra_fields": list(extra),
            "parsed": parsed,
        }

    def _extract_fields(self, data: Any, path: str) -> Set[str]:
        """Extract all field paths from data."""
        fields = set()

        if isinstance(data, dict):
            for key, value in data.items():
                field_path = f"{path}.{key}"
                fields.add(field_path)

                if self.include_nested:
                    fields.update(self._extract_fields(value, field_path))

        elif isinstance(data, list) and self.include_nested:
            for i, item in enumerate(data):
                fields.update(self._extract_fields(item, f"{path}[{i}]"))

        return fields
