"""
Input types for structured output validation metrics.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ValidationMode(Enum):
    """Validation strictness modes."""
    STRICT = "strict"      # Exact match, no extra fields, correct types
    COERCE = "coerce"      # Allow type coercion (str -> int, etc.)
    LENIENT = "lenient"    # Allow extra fields, flexible types


class JSONInput(BaseModel):
    """Input for JSON validation metrics."""
    model_config = ConfigDict(extra="allow")

    response: str = Field(..., description="LLM-generated JSON string")
    schema: Optional[Dict[str, Any]] = Field(
        None, description="JSON Schema to validate against"
    )
    expected: Optional[Dict[str, Any]] = Field(
        None, description="Expected JSON object for comparison"
    )
    mode: str = Field(
        "coerce", description="Validation strictness: strict, coerce, lenient"
    )


class PydanticInput(BaseModel):
    """Input for Pydantic model validation."""
    model_config = ConfigDict(extra="allow")

    response: str = Field(..., description="LLM-generated JSON string")
    model_class: Optional[str] = Field(
        None, description="Fully qualified Pydantic model class name"
    )
    model_schema: Optional[Dict[str, Any]] = Field(
        None, description="JSON Schema derived from Pydantic model"
    )
    mode: str = Field("coerce")


class YAMLInput(BaseModel):
    """Input for YAML validation metrics."""
    model_config = ConfigDict(extra="allow")

    response: str = Field(..., description="LLM-generated YAML string")
    schema: Optional[Dict[str, Any]] = Field(
        None, description="JSON Schema to validate against"
    )
    expected: Optional[Dict[str, Any]] = Field(
        None, description="Expected YAML content as dict"
    )


class StructuredInput(BaseModel):
    """Generic input for any structured format."""
    model_config = ConfigDict(extra="allow")

    response: str = Field(..., description="LLM-generated structured output")
    format: str = Field("json", description="Format: json, xml, yaml, toml")
    schema: Optional[Dict[str, Any]] = Field(None, description="Schema to validate")
    expected: Optional[Any] = Field(None, description="Expected output")
    mode: str = Field("coerce")


class ValidationError(BaseModel):
    """Single validation error."""
    path: str = Field(..., description="JSON path to error (e.g., '$.user.name')")
    message: str = Field(..., description="Error description")
    error_type: str = Field(..., description="Error type: syntax, type, missing, extra")
    expected: Optional[Any] = Field(None, description="Expected value/type")
    actual: Optional[Any] = Field(None, description="Actual value/type")

    def dict(self, **kwargs):
        """Convert to dictionary."""
        return {
            "path": self.path,
            "message": self.message,
            "error_type": self.error_type,
            "expected": self.expected,
            "actual": self.actual,
        }


class ValidationResult(BaseModel):
    """Complete validation result."""
    model_config = ConfigDict(extra="allow")

    valid: bool = Field(..., description="Overall validity")
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationError] = Field(default_factory=list)

    # Detailed scores
    syntax_valid: bool = True
    schema_valid: bool = True
    type_valid: bool = True
    completeness: float = 1.0  # 0-1, fraction of required fields present

    # Parsed output (if successful)
    parsed: Optional[Any] = None
