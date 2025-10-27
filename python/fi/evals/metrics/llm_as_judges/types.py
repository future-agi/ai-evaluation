from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from ...types import BaseMetricInput


class BaseLLMJudgeInput(BaseMetricInput):
    pass


class LLMFewShotExample(BaseModel):
    inputs: Dict[str, Any] = Field(
        ..., description="A dictionary representing an input model"
    )
    output: str = Field(
        ...,
        description="The ideal JSON string the judge LLM should produce for these inputs.",
    )


class CustomInput(BaseLLMJudgeInput):
    """A flexible input model for the CustomLLMJudge that allows any field."""

    class Config:
        extra = "allow"


class DefaultJudgeOutput(BaseModel):
    """The default output format for a custom judge."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The normalized evaluation score from 0.0 to 1.0.",
    )
    reason: str = Field(..., description="A brief explanation of the score.")


class LLMMessage(BaseModel):
    role: str
    content: str
    name: Optional[str]
    function_call: Optional[str]
    tool_call_id: Optional[str]


class ConversationInput(BaseLLMJudgeInput):
    messages: List[LLMMessage]
