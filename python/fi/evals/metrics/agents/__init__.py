"""
Agent Evaluation Metrics.

Trajectory-based evaluation of AI agent performance.
Provides multi-step analysis beyond single-response evaluation.

Based on:
- AgentBench methodology (ICLR 2024)
- Multi-turn agent evaluation frameworks
"""

from .types import (
    AgentTrajectoryInput,
    AgentStep,
    ToolCall,
    TaskDefinition,
    TrajectoryAnalysis,
)
from .metrics import (
    TaskCompletion,
    StepEfficiency,
    ToolSelectionAccuracy,
    TrajectoryScore,
    GoalProgress,
    ActionSafety,
    ReasoningQuality,
)

__all__ = [
    # Types
    "AgentTrajectoryInput",
    "AgentStep",
    "ToolCall",
    "TaskDefinition",
    "TrajectoryAnalysis",
    # Metrics
    "TaskCompletion",
    "StepEfficiency",
    "ToolSelectionAccuracy",
    "TrajectoryScore",
    "GoalProgress",
    "ActionSafety",
    "ReasoningQuality",
]
