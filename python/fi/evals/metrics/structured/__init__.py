"""
Structured Output Validation Metrics.

This module provides comprehensive metrics for evaluating LLM-generated
structured outputs (JSON, YAML, Pydantic models).

Metrics:
- JSONValidation: Validates JSON against JSON Schema
- JSONSyntaxOnly: Simple JSON syntax check
- SchemaCompliance: Generic schema compliance with detailed breakdown
- TypeCompliance: Type-only validation
- FieldCompleteness: Field presence evaluation
- RequiredFieldsOnly: Required field check
- FieldCoverage: Coverage comparison with expected
- HierarchyScore: Tree-based structural similarity
- TreeEditDistance: Structural edit distance
- StructuredOutputScore: Composite metric combining all aspects
- QuickStructuredCheck: Fast lightweight validation

Validators:
- JSONValidator: JSON validation with JSON Schema support
- YAMLValidator: YAML validation with JSON Schema support
- PydanticValidator: Pydantic model validation

Example:
    >>> from fi.evals.metrics.structured import (
    ...     StructuredOutputScore,
    ...     JSONValidation,
    ...     HierarchyScore,
    ... )
    >>>
    >>> # Comprehensive validation
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

# Types
from .types import (
    ValidationMode,
    JSONInput,
    PydanticInput,
    YAMLInput,
    StructuredInput,
    ValidationError,
    ValidationResult,
)

# Validators
from .validators import (
    BaseValidator,
    JSONValidator,
    PydanticValidator,
    YAMLValidator,
    get_validator,
)

# Metrics - JSON
from .json_validation import JSONValidation, JSONSyntaxOnly

# Metrics - Schema
from .schema_compliance import SchemaCompliance, TypeCompliance

# Metrics - Fields
from .field_completeness import (
    FieldCompleteness,
    RequiredFieldsOnly,
    FieldCoverage,
)

# Metrics - Hierarchy
from .hierarchy_score import HierarchyScore, TreeEditDistance

# Metrics - Composite
from .structured_output_score import StructuredOutputScore, QuickStructuredCheck

__all__ = [
    # Types
    "ValidationMode",
    "JSONInput",
    "PydanticInput",
    "YAMLInput",
    "StructuredInput",
    "ValidationError",
    "ValidationResult",
    # Validators
    "BaseValidator",
    "JSONValidator",
    "PydanticValidator",
    "YAMLValidator",
    "get_validator",
    # Metrics
    "JSONValidation",
    "JSONSyntaxOnly",
    "SchemaCompliance",
    "TypeCompliance",
    "FieldCompleteness",
    "RequiredFieldsOnly",
    "FieldCoverage",
    "HierarchyScore",
    "TreeEditDistance",
    "StructuredOutputScore",
    "QuickStructuredCheck",
]
