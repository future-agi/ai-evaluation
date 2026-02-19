"""
Validators for structured output formats.
"""

from .base import BaseValidator
from .json_validator import JSONValidator
from .pydantic_validator import PydanticValidator
from .yaml_validator import YAMLValidator

__all__ = [
    "BaseValidator",
    "JSONValidator",
    "PydanticValidator",
    "YAMLValidator",
]

# Validator registry for easy lookup
VALIDATORS = {
    "json": JSONValidator,
    "yaml": YAMLValidator,
    "pydantic": PydanticValidator,
}


def get_validator(format_name: str) -> BaseValidator:
    """Get validator instance by format name."""
    validator_class = VALIDATORS.get(format_name.lower())
    if validator_class is None:
        raise ValueError(f"Unknown format: {format_name}. Available: {list(VALIDATORS.keys())}")
    return validator_class()
