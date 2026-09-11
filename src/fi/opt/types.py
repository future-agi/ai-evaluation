from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional


class LLMMessage(BaseModel):
    """Every message sent and received by the LLM MUST follow this format."""

    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[str] = None
    tool_call_id: Optional[str] = None


class EvaluationResult(BaseModel):
    """
    A standardized  result from a single evaluation.
    """

    score: float = Field(..., description="The normalized score (0.0 to 1.0).")
    reason: str = Field("", description="The explanation for the score.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Any other metadata from the evaluator."
    )


class IterationHistory(BaseModel):
    """
    A detailed record of a single optimization iteration.
    """

    prompt: str
    average_score: float
    individual_results: List[EvaluationResult]
    candidate_id: Optional[str] = None
    candidate_config: Optional[Dict[str, Any]] = None
    layers: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OptimizationResult(BaseModel):
    """Output Model to hold the results of an optimization run."""

    best_generator: Any
    best_candidate: Any = None
    history: List[IterationHistory]
    final_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Early stopping metadata
    early_stopped: bool = Field(
        default=False,
        description="Whether optimization was terminated early by a stopping criterion"
    )
    stop_reason: Optional[str] = Field(
        default=None,
        description="Explanation for early stopping (if applicable)"
    )
    total_iterations: int = Field(
        default=0,
        description="Total number of iterations completed"
    )
    total_evaluations: int = Field(
        default=0,
        description="Total number of dataset evaluations performed"
    )
