"""
Base validator interface for structured output validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from ..types import ValidationResult, ValidationError, ValidationMode


class BaseValidator(ABC):
    """Abstract base class for format validators."""

    format_name: str = "unknown"

    @abstractmethod
    def validate_syntax(self, content: str) -> ValidationResult:
        """
        Validate syntax only (is it parseable?).

        Returns:
            ValidationResult with syntax_valid set
        """
        pass

    @abstractmethod
    def validate_schema(
        self,
        content: str,
        schema: Dict[str, Any],
        mode: ValidationMode = ValidationMode.COERCE,
    ) -> ValidationResult:
        """
        Validate against a schema.

        Args:
            content: Raw string content
            schema: Schema to validate against
            mode: Validation strictness

        Returns:
            ValidationResult with full validation details
        """
        pass

    @abstractmethod
    def parse(self, content: str) -> Any:
        """
        Parse content into Python object.

        Raises:
            ValueError: If content cannot be parsed
        """
        pass

    def compare(
        self,
        content: str,
        expected: Any,
        mode: ValidationMode = ValidationMode.COERCE,
    ) -> ValidationResult:
        """
        Compare parsed content against expected value.

        Default implementation - can be overridden.
        """
        try:
            parsed = self.parse(content)
        except Exception as e:
            return ValidationResult(
                valid=False,
                syntax_valid=False,
                errors=[ValidationError(
                    path="$",
                    message=str(e),
                    error_type="syntax",
                )]
            )

        errors = self._compare_values(parsed, expected, "$", mode)

        return ValidationResult(
            valid=len(errors) == 0,
            syntax_valid=True,
            errors=errors,
            parsed=parsed,
        )

    def _compare_values(
        self,
        actual: Any,
        expected: Any,
        path: str,
        mode: ValidationMode,
    ) -> List[ValidationError]:
        """Recursively compare values."""
        errors = []

        # Handle None cases
        if expected is None and actual is None:
            return errors
        if expected is None or actual is None:
            if expected != actual:
                errors.append(ValidationError(
                    path=path,
                    message="Value mismatch (None vs non-None)",
                    error_type="value",
                    expected=expected,
                    actual=actual,
                ))
            return errors

        # Type comparison
        if type(actual) != type(expected):
            if mode == ValidationMode.STRICT:
                errors.append(ValidationError(
                    path=path,
                    message=f"Type mismatch",
                    error_type="type",
                    expected=type(expected).__name__,
                    actual=type(actual).__name__,
                ))
                return errors
            elif mode == ValidationMode.COERCE:
                # Try to coerce
                try:
                    actual = type(expected)(actual)
                except (ValueError, TypeError):
                    errors.append(ValidationError(
                        path=path,
                        message=f"Cannot coerce {type(actual).__name__} to {type(expected).__name__}",
                        error_type="type",
                        expected=type(expected).__name__,
                        actual=type(actual).__name__,
                    ))
                    return errors

        # Dict comparison
        if isinstance(expected, dict):
            # Check for missing keys
            for key in expected:
                if key not in actual:
                    errors.append(ValidationError(
                        path=f"{path}.{key}",
                        message=f"Missing required field",
                        error_type="missing",
                        expected=key,
                    ))
                else:
                    errors.extend(self._compare_values(
                        actual[key], expected[key], f"{path}.{key}", mode
                    ))

            # Check for extra keys in strict mode
            if mode == ValidationMode.STRICT:
                for key in actual:
                    if key not in expected:
                        errors.append(ValidationError(
                            path=f"{path}.{key}",
                            message=f"Unexpected field",
                            error_type="extra",
                            actual=key,
                        ))

        # List comparison
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                errors.append(ValidationError(
                    path=path,
                    message=f"Array length mismatch",
                    error_type="length",
                    expected=len(expected),
                    actual=len(actual),
                ))
            else:
                for i, (a, e) in enumerate(zip(actual, expected)):
                    errors.extend(self._compare_values(a, e, f"{path}[{i}]", mode))

        # Scalar comparison
        else:
            if actual != expected:
                errors.append(ValidationError(
                    path=path,
                    message=f"Value mismatch",
                    error_type="value",
                    expected=expected,
                    actual=actual,
                ))

        return errors
