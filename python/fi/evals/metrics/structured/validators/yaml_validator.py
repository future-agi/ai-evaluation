"""
YAML validation with JSON Schema support.
"""

import json
from typing import Any, Dict, List
from ..types import ValidationResult, ValidationError, ValidationMode
from .base import BaseValidator

# Optional yaml import
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Optional jsonschema import
try:
    import jsonschema
    from jsonschema import Draft7Validator
    _JSONSCHEMA_AVAILABLE = True
except ImportError:
    _JSONSCHEMA_AVAILABLE = False


class YAMLValidator(BaseValidator):
    """
    Validator for YAML output.

    Features:
    - Syntax validation (yaml.safe_load)
    - JSON Schema validation (YAML is a superset of JSON)
    - Support for multi-document YAML
    - Handles common LLM YAML mistakes
    """

    format_name = "yaml"

    def __init__(self, allow_multi_doc: bool = False):
        """
        Initialize YAML validator.

        Args:
            allow_multi_doc: Allow multiple YAML documents (---) in content
        """
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAMLValidator")
        self.allow_multi_doc = allow_multi_doc

    def validate_syntax(self, content: str) -> ValidationResult:
        """Check if content is valid YAML."""
        try:
            # Try to fix common LLM YAML mistakes
            content = self._preprocess_yaml(content)

            if self.allow_multi_doc:
                parsed = list(yaml.safe_load_all(content))
                if len(parsed) == 1:
                    parsed = parsed[0]
            else:
                parsed = yaml.safe_load(content)

            return ValidationResult(
                valid=True,
                syntax_valid=True,
                parsed=parsed,
            )
        except yaml.YAMLError as e:
            error_msg = str(e)
            line = getattr(e, 'problem_mark', None)
            if line:
                path = f"$.line[{line.line + 1}]:col[{line.column + 1}]"
            else:
                path = "$"

            return ValidationResult(
                valid=False,
                syntax_valid=False,
                errors=[ValidationError(
                    path=path,
                    message=f"YAML syntax error: {error_msg}",
                    error_type="syntax",
                )]
            )

    def validate_schema(
        self,
        content: str,
        schema: Dict[str, Any],
        mode: ValidationMode = ValidationMode.COERCE,
    ) -> ValidationResult:
        """Validate YAML against JSON Schema."""
        # First check syntax
        syntax_result = self.validate_syntax(content)
        if not syntax_result.syntax_valid:
            return syntax_result

        parsed = syntax_result.parsed

        if not _JSONSCHEMA_AVAILABLE:
            # Fallback to basic type checking
            return self._validate_schema_basic(parsed, schema, mode)

        # Use jsonschema for validation (YAML parses to same types as JSON)
        errors = []
        validator = Draft7Validator(schema)

        for error in validator.iter_errors(parsed):
            path = "$" + "".join(
                f".{p}" if isinstance(p, str) else f"[{p}]"
                for p in error.absolute_path
            )
            errors.append(ValidationError(
                path=path,
                message=error.message,
                error_type=self._classify_schema_error(error),
                expected=error.schema.get("type") if hasattr(error, 'schema') else None,
                actual=type(error.instance).__name__ if error.instance is not None else None,
            ))

        # Calculate completeness
        required_fields = schema.get("required", [])
        if required_fields and isinstance(parsed, dict):
            present = sum(1 for f in required_fields if f in parsed)
            completeness = present / len(required_fields)
        else:
            completeness = 1.0

        return ValidationResult(
            valid=len(errors) == 0,
            syntax_valid=True,
            schema_valid=len(errors) == 0,
            errors=errors,
            completeness=completeness,
            parsed=parsed,
        )

    def parse(self, content: str) -> Any:
        """Parse YAML string to Python object."""
        content = self._preprocess_yaml(content)
        return yaml.safe_load(content)

    def _preprocess_yaml(self, content: str) -> str:
        """Fix common LLM YAML mistakes."""
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            # Fix tabs (YAML doesn't allow tabs for indentation)
            if '\t' in line:
                # Replace tabs with 2 spaces
                line = line.replace('\t', '  ')

            # Remove trailing whitespace
            line = line.rstrip()

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _classify_schema_error(self, error) -> str:
        """Classify jsonschema error into our error types."""
        validator = error.validator
        if validator == "type":
            return "type"
        elif validator == "required":
            return "missing"
        elif validator == "additionalProperties":
            return "extra"
        elif validator in ("enum", "const"):
            return "value"
        elif validator in ("minLength", "maxLength", "minimum", "maximum"):
            return "constraint"
        else:
            return "schema"

    def _validate_schema_basic(
        self,
        parsed: Any,
        schema: Dict[str, Any],
        mode: ValidationMode,
    ) -> ValidationResult:
        """Basic schema validation without jsonschema library."""
        errors = []

        def check_type(value: Any, expected_type: str, path: str):
            type_map = {
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "array": list,
                "object": dict,
                "null": type(None),
            }

            expected = type_map.get(expected_type)
            if expected and not isinstance(value, expected):
                if mode == ValidationMode.STRICT:
                    errors.append(ValidationError(
                        path=path,
                        message=f"Expected {expected_type}, got {type(value).__name__}",
                        error_type="type",
                        expected=expected_type,
                        actual=type(value).__name__,
                    ))

        def validate_object(obj: Any, obj_schema: Dict, path: str):
            if "type" in obj_schema:
                check_type(obj, obj_schema["type"], path)

            if isinstance(obj, dict) and "properties" in obj_schema:
                # Check required fields
                for field in obj_schema.get("required", []):
                    if field not in obj:
                        errors.append(ValidationError(
                            path=f"{path}.{field}",
                            message=f"Missing required field: {field}",
                            error_type="missing",
                            expected=field,
                        ))

                # Validate properties
                for key, prop_schema in obj_schema.get("properties", {}).items():
                    if key in obj:
                        validate_object(obj[key], prop_schema, f"{path}.{key}")

            if isinstance(obj, list) and "items" in obj_schema:
                for i, item in enumerate(obj):
                    validate_object(item, obj_schema["items"], f"{path}[{i}]")

        validate_object(parsed, schema, "$")

        # Calculate completeness
        required_fields = schema.get("required", [])
        if required_fields and isinstance(parsed, dict):
            present = sum(1 for f in required_fields if f in parsed)
            completeness = present / len(required_fields)
        else:
            completeness = 1.0

        return ValidationResult(
            valid=len(errors) == 0,
            syntax_valid=True,
            schema_valid=len(errors) == 0,
            errors=errors,
            completeness=completeness,
            parsed=parsed,
        )
