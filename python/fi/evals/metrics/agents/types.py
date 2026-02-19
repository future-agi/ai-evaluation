"""
Types for Agent Evaluation.

These types support trajectory-based evaluation of AI agent performance,
including multi-step analysis and tool usage tracking.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from ...types import BaseMetricInput


class ToolCall(BaseModel):
    """Represents a tool/function call made by an agent."""

    name: str = Field(..., description="Name of the tool called")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool"
    )
    result: Optional[Any] = Field(
        default=None,
        description="Result returned from the tool"
    )
    success: bool = Field(
        default=True,
        description="Whether the tool call succeeded"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the call failed"
    )


class AgentStep(BaseModel):
    """Represents a single step in an agent's trajectory."""

    step_number: int = Field(..., description="Step number in the trajectory (1-indexed)")
    thought: Optional[str] = Field(
        default=None,
        description="Agent's reasoning/thought for this step"
    )
    action: Optional[str] = Field(
        default=None,
        description="Action taken by the agent"
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="Tool calls made in this step"
    )
    observation: Optional[str] = Field(
        default=None,
        description="Observation from the environment after the action"
    )
    is_final: bool = Field(
        default=False,
        description="Whether this is the final step"
    )
    timestamp_ms: Optional[int] = Field(
        default=None,
        description="Timestamp of the step in milliseconds"
    )


class TaskDefinition(BaseModel):
    """Definition of the task the agent should accomplish."""

    description: str = Field(..., description="Natural language description of the task")
    expected_outcome: Optional[str] = Field(
        default=None,
        description="Expected outcome or result"
    )
    required_tools: Optional[List[str]] = Field(
        default=None,
        description="List of tools required to complete the task"
    )
    max_steps: Optional[int] = Field(
        default=None,
        description="Maximum allowed steps"
    )
    success_criteria: Optional[List[str]] = Field(
        default=None,
        description="Criteria for determining task success"
    )


class ExpectedStep(BaseModel):
    """Expected step in the optimal trajectory."""

    description: str = Field(..., description="Description of the expected action")
    required_tools: Optional[List[str]] = Field(
        default=None,
        description="Tools that should be used in this step"
    )
    key_arguments: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Key arguments that should be present"
    )


class AgentTrajectoryInput(BaseMetricInput):
    """
    Input for agent trajectory evaluation metrics.

    Evaluates multi-step agent execution including:
    - Task completion
    - Step efficiency
    - Tool usage accuracy
    - Trajectory quality
    """

    # The agent's execution trajectory
    trajectory: List[AgentStep] = Field(
        ...,
        description="List of steps taken by the agent"
    )

    # The task definition
    task: TaskDefinition = Field(
        ...,
        description="Definition of the task to accomplish"
    )

    # Optional: expected trajectory for comparison
    expected_trajectory: Optional[List[ExpectedStep]] = Field(
        default=None,
        description="Expected/optimal trajectory for comparison"
    )

    # Optional: final result produced by the agent
    final_result: Optional[Any] = Field(
        default=None,
        description="Final output/result from the agent"
    )

    # Optional: ground truth result for comparison
    expected_result: Optional[Any] = Field(
        default=None,
        description="Expected/ground truth result"
    )

    # Available tools for the task
    available_tools: Optional[List[str]] = Field(
        default=None,
        description="List of available tools the agent can use"
    )


class TrajectoryAnalysis(BaseModel):
    """Detailed analysis of an agent trajectory."""

    total_steps: int = Field(..., description="Total number of steps taken")
    successful_tool_calls: int = Field(..., description="Number of successful tool calls")
    failed_tool_calls: int = Field(..., description="Number of failed tool calls")
    unique_tools_used: List[str] = Field(default_factory=list, description="List of unique tools used")
    unnecessary_steps: int = Field(default=0, description="Number of unnecessary/redundant steps")
    backtracking_count: int = Field(default=0, description="Number of times agent backtracked")
    task_completed: bool = Field(..., description="Whether the task was completed")
    efficiency_score: float = Field(..., description="Efficiency score (0-1)")
