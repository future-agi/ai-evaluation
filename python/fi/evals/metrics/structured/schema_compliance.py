"""
Schema Compliance Metric.

Generic schema compliance evaluation supporting multiple formats.
"""

from typing import Any, Dict, Optional, List

from ..base_metric import BaseMetric
from .types import StructuredInput, ValidationMode, ValidationError
from .validators import JSONValidator, YAMLValidator, BaseValidator


class SchemaCompliance(BaseMetric[StructuredInput]):
    """
    Evaluates structured output compliance with a schema.

    Supports multiple formats (JSON, YAML) and provides detailed
    compliance breakdown:
    - Field presence (completeness)
    - Type correctness
    - Value constraints
    - Structural compliance

    Score: 0.0 (completely non-compliant) to 1.0 (fully compliant)

    Example:
        >>> metric = SchemaCompliance()
        >>> result = metric.evaluate([{
        ...     "response": '{"user": {"name": "Alice", "age": 25}}',
        ...     "format": "json",
        ...     "schema": {
        ...         "type": "object",
        ...         "required": ["user"],
        ...         "properties": {
        ...             "user": {
        ...                 "type": "object",
        ...                 "required": ["name"],
        ...                 "properties": {
        ...                     "name": {"type": "string"},
        ...                     "age": {"type": "integer"}
        ...                 }
        ...             }
        ...         }
        ...     }
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "schema_compliance"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.validators = {
            "json": JSONValidator(),
            "yaml": YAMLValidator() if self._yaml_available() else None,
        }
        # Weights for different compliance aspects
        self.syntax_weight = self.config.get("syntax_weight", 0.2)
        self.type_weight = self.config.get("type_weight", 0.3)
        self.completeness_weight = self.config.get("completeness_weight", 0.3)
        self.constraint_weight = self.config.get("constraint_weight", 0.2)

    def _yaml_available(self) -> bool:
        try:
            import yaml
            return True
        except ImportError:
            return False

    def _get_validator(self, format_name: str) -> Optional[BaseValidator]:
        """Get validator for format."""
        validator = self.validators.get(format_name.lower())
        if validator is None and format_name.lower() == "yaml":
            raise ImportError("PyYAML is required for YAML validation")
        return validator

    def compute_one(self, inputs: StructuredInput) -> Dict[str, Any]:
        response = inputs.response
        format_name = inputs.format
        schema = inputs.schema
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

        if not schema:
            return {
                "output": 0.0,
                "reason": "No schema provided for validation",
            }

        # Get appropriate validator
        validator = self._get_validator(format_name)
        if validator is None:
            return {
                "output": 0.0,
                "reason": f"Unsupported format: {format_name}",
            }

        # Validate
        result = validator.validate_schema(response, schema, mode)

        if not result.syntax_valid:
            return {
                "output": 0.0,
                "reason": f"Syntax error: {result.errors[0].message if result.errors else 'Unknown'}",
                "syntax_valid": False,
                "errors": [e.dict() for e in result.errors],
            }

        # Calculate detailed compliance score
        compliance_breakdown = self._calculate_compliance_breakdown(result.errors, schema)

        if result.schema_valid:
            return {
                "output": 1.0,
                "reason": "Fully compliant with schema",
                "syntax_valid": True,
                "schema_valid": True,
                "completeness": result.completeness,
                "compliance_breakdown": compliance_breakdown,
                "parsed": result.parsed,
            }

        # Calculate partial compliance score
        score = self._calculate_partial_score(compliance_breakdown)

        return {
            "output": round(score, 4),
            "reason": f"Partial compliance: {len(result.errors)} error(s)",
            "syntax_valid": True,
            "schema_valid": False,
            "completeness": result.completeness,
            "error_count": len(result.errors),
            "errors": [e.dict() for e in result.errors],
            "compliance_breakdown": compliance_breakdown,
            "parsed": result.parsed,
        }

    def _calculate_compliance_breakdown(
        self,
        errors: List[ValidationError],
        schema: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate detailed compliance breakdown by error type."""
        # Count errors by type
        error_counts = {
            "syntax": 0,
            "type": 0,
            "missing": 0,
            "extra": 0,
            "constraint": 0,
            "value": 0,
        }

        for error in errors:
            error_type = error.error_type
            if error_type in error_counts:
                error_counts[error_type] += 1
            else:
                error_counts["value"] += 1

        # Estimate expected fields from schema
        expected_fields = self._count_schema_fields(schema)

        # Calculate compliance ratios
        breakdown = {
            "syntax_compliance": 1.0 if error_counts["syntax"] == 0 else 0.0,
            "type_compliance": 1.0 if error_counts["type"] == 0 else max(0.0, 1.0 - error_counts["type"] / max(expected_fields, 1)),
            "field_compliance": 1.0 if error_counts["missing"] == 0 else max(0.0, 1.0 - error_counts["missing"] / max(expected_fields, 1)),
            "constraint_compliance": 1.0 if error_counts["constraint"] == 0 else max(0.0, 1.0 - error_counts["constraint"] / max(expected_fields, 1)),
        }

        return breakdown

    def _count_schema_fields(self, schema: Dict[str, Any], depth: int = 0) -> int:
        """Count expected fields in schema."""
        if depth > 10:  # Prevent infinite recursion
            return 1

        count = 0
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            count += len(properties)
            for prop_schema in properties.values():
                count += self._count_schema_fields(prop_schema, depth + 1)
        elif schema.get("type") == "array":
            items = schema.get("items", {})
            count += self._count_schema_fields(items, depth + 1)

        return max(count, 1)

    def _calculate_partial_score(self, breakdown: Dict[str, float]) -> float:
        """Calculate weighted partial compliance score."""
        return (
            self.syntax_weight * breakdown["syntax_compliance"] +
            self.type_weight * breakdown["type_compliance"] +
            self.completeness_weight * breakdown["field_compliance"] +
            self.constraint_weight * breakdown["constraint_compliance"]
        )


class TypeCompliance(BaseMetric[StructuredInput]):
    """
    Evaluates only type compliance (ignoring extra fields, constraints).

    Useful for lenient validation where types matter but structure is flexible.

    Score: 0.0 to 1.0 (fraction of correctly typed fields)
    """

    @property
    def metric_name(self) -> str:
        return "type_compliance"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.json_validator = JSONValidator()

    def compute_one(self, inputs: StructuredInput) -> Dict[str, Any]:
        response = inputs.response
        schema = inputs.schema

        if not response or not response.strip():
            return {"output": 0.0, "reason": "Empty response"}

        if not schema:
            return {"output": 0.0, "reason": "No schema provided"}

        result = self.json_validator.validate_schema(
            response, schema, ValidationMode.STRICT
        )

        if not result.syntax_valid:
            return {
                "output": 0.0,
                "reason": "Syntax error",
                "errors": [e.dict() for e in result.errors],
            }

        # Count type errors specifically
        type_errors = [e for e in result.errors if e.error_type == "type"]
        total_fields = self._count_schema_fields(schema)

        if not type_errors:
            return {
                "output": 1.0,
                "reason": "All types correct",
                "parsed": result.parsed,
            }

        score = max(0.0, 1.0 - len(type_errors) / max(total_fields, 1))

        return {
            "output": round(score, 4),
            "reason": f"{len(type_errors)} type error(s)",
            "type_errors": [e.dict() for e in type_errors],
            "parsed": result.parsed,
        }

    def _count_schema_fields(self, schema: Dict[str, Any]) -> int:
        """Count fields in schema."""
        count = 0
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            count = len(properties)
            for prop_schema in properties.values():
                count += self._count_schema_fields(prop_schema)
        return max(count, 1)
