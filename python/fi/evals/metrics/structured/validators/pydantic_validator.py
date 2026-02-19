"""
Pydantic model validation for LLM outputs.
"""

import json
from typing import Any, Dict, Optional, Type, List
from ..types import ValidationResult, ValidationError, ValidationMode
from .base import BaseValidator

try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    BaseModel = None


class PydanticValidator(BaseValidator):
    """
    Validator using Pydantic models.

    Features:
    - Full Pydantic validation with type coercion
    - Detailed error paths
    - Support for nested models
    - Custom validators respected
    """

    format_name = "pydantic"

    def __init__(self, model_class: Optional[Type] = None):
        """
        Initialize with optional model class.

        Args:
            model_class: Pydantic model class to validate against
        """
        if not _PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic is required for PydanticValidator")
        self.model_class = model_class

    def validate_syntax(self, content: str) -> ValidationResult:
        """Check if content is valid JSON (required for Pydantic)."""
        try:
            parsed = json.loads(content)
            return ValidationResult(
                valid=True,
                syntax_valid=True,
                parsed=parsed,
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                syntax_valid=False,
                errors=[ValidationError(
                    path=f"$.char[{e.pos}]",
                    message=f"JSON syntax error: {e.msg}",
                    error_type="syntax",
                )]
            )

    def validate_schema(
        self,
        content: str,
        schema: Dict[str, Any] = None,
        mode: ValidationMode = ValidationMode.COERCE,
    ) -> ValidationResult:
        """Validate using Pydantic model."""
        if self.model_class is None:
            raise ValueError("No model class set for validation")

        return self.validate_model(content, self.model_class, mode)

    def validate_model(
        self,
        content: str,
        model_class: Type,
        mode: ValidationMode = ValidationMode.COERCE,
    ) -> ValidationResult:
        """
        Validate content against a Pydantic model.

        Args:
            content: JSON string
            model_class: Pydantic model class
            mode: Validation mode

        Returns:
            ValidationResult with Pydantic validation details
        """
        # Check syntax first
        syntax_result = self.validate_syntax(content)
        if not syntax_result.syntax_valid:
            return syntax_result

        parsed = syntax_result.parsed

        # Configure validation based on mode
        try:
            if mode == ValidationMode.STRICT:
                # Pydantic v2 strict mode
                instance = model_class.model_validate(
                    parsed,
                    strict=True,
                )
            else:
                # Default: allow coercion
                instance = model_class.model_validate(parsed)

            return ValidationResult(
                valid=True,
                syntax_valid=True,
                schema_valid=True,
                completeness=1.0,
                parsed=instance.model_dump(),
            )
        except PydanticValidationError as e:
            return self._convert_pydantic_errors(e, parsed, model_class)

    def _convert_pydantic_errors(
        self,
        pydantic_error: "PydanticValidationError",
        parsed: Any,
        model_class: Type,
    ) -> ValidationResult:
        """Convert Pydantic validation errors to our format."""
        errors = []

        for error in pydantic_error.errors():
            # Build path from location
            loc = error.get("loc", ())
            path = "$" + "".join(
                f".{p}" if isinstance(p, str) else f"[{p}]"
                for p in loc
            )

            # Classify error type
            error_type = error.get("type", "validation")
            if "missing" in error_type:
                classified = "missing"
            elif "type" in error_type:
                classified = "type"
            elif "extra" in error_type:
                classified = "extra"
            else:
                classified = "validation"

            errors.append(ValidationError(
                path=path,
                message=error.get("msg", str(error)),
                error_type=classified,
                expected=error.get("ctx", {}).get("expected") if error.get("ctx") else None,
            ))

        # Calculate completeness based on missing field errors
        missing_count = sum(1 for e in errors if e.error_type == "missing")
        total_fields = len(model_class.model_fields) if hasattr(model_class, 'model_fields') else 1
        completeness = 1.0 - (missing_count / max(total_fields, 1))

        return ValidationResult(
            valid=False,
            syntax_valid=True,
            schema_valid=False,
            errors=errors,
            completeness=completeness,
            parsed=parsed,
        )

    def parse(self, content: str) -> Any:
        """Parse JSON string."""
        return json.loads(content)

    @staticmethod
    def get_schema_from_model(model_class: Type) -> Dict[str, Any]:
        """Extract JSON Schema from a Pydantic model."""
        if hasattr(model_class, 'model_json_schema'):
            return model_class.model_json_schema()
        return {}
