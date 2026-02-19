"""
Types for Function Calling Evaluation.

These types support the evaluation of LLM function/tool calling
capabilities with AST-based comparison.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from ...types import BaseMetricInput


class ParameterSpec(BaseModel):
    """Specification for a function parameter."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Expected type (string, integer, number, boolean, array, object)")
    required: bool = Field(default=True, description="Whether the parameter is required")
    enum: Optional[List[Any]] = Field(default=None, description="Allowed values if constrained")
    default: Optional[Any] = Field(default=None, description="Default value if not required")


class FunctionDefinition(BaseModel):
    """Definition of an expected function signature."""

    name: str = Field(..., description="Function name")
    parameters: List[ParameterSpec] = Field(
        default_factory=list,
        description="List of parameter specifications"
    )
    description: Optional[str] = Field(default=None, description="Function description")


class FunctionCall(BaseModel):
    """Represents a function call from the LLM."""

    name: str = Field(..., description="Name of the function called")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the function"
    )


class FunctionCallInput(BaseMetricInput):
    """
    Input for function calling evaluation metrics.

    Supports evaluating:
    - Single function call against expected
    - Multiple function calls (parallel calling)
    - Function call with schema validation
    """

    # The actual function call(s) from the LLM
    response: Union[FunctionCall, List[FunctionCall], Dict[str, Any], str] = Field(
        ...,
        description="The function call(s) from the LLM. Can be FunctionCall object, dict, JSON string, or list of calls."
    )

    # The expected function call(s)
    expected_response: Optional[Union[FunctionCall, List[FunctionCall], Dict[str, Any], str]] = Field(
        default=None,
        description="The expected function call(s). Required for accuracy metrics."
    )

    # Function definitions for schema validation
    function_definitions: Optional[List[FunctionDefinition]] = Field(
        default=None,
        description="Available function definitions for validation."
    )

    # Evaluation options
    strict_type_check: bool = Field(
        default=False,
        description="If True, require exact type matches. If False, allow compatible types (e.g., int/float)."
    )

    ignore_extra_params: bool = Field(
        default=False,
        description="If True, ignore extra parameters not in expected. If False, penalize extra params."
    )

    order_matters: bool = Field(
        default=False,
        description="If True, for parallel calls, order must match. If False, set comparison."
    )


class MultiTurnFunctionCallInput(BaseMetricInput):
    """
    Input for multi-turn function calling evaluation.

    Evaluates a sequence of function calls in a conversation.
    """

    # Sequence of actual function calls
    response_calls: List[FunctionCall] = Field(
        ...,
        description="Sequence of function calls made by the LLM"
    )

    # Expected sequence
    expected_calls: List[FunctionCall] = Field(
        ...,
        description="Expected sequence of function calls"
    )

    # Function definitions
    function_definitions: Optional[List[FunctionDefinition]] = Field(
        default=None,
        description="Available function definitions"
    )

    # Evaluation options
    sequence_strict: bool = Field(
        default=True,
        description="If True, exact sequence must match. If False, allows reordering."
    )
