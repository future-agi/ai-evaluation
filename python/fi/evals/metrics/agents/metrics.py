"""
Agent Evaluation Metrics.

Trajectory-based evaluation of AI agent performance.
Provides deterministic, fast evaluation for multi-step agent tasks.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from difflib import SequenceMatcher

from ..base_metric import BaseMetric
from .types import (
    AgentTrajectoryInput,
    AgentStep,
    ToolCall,
    TaskDefinition,
    ExpectedStep,
    TrajectoryAnalysis,
)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def _extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text."""
    text = _normalize_text(text)
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'should', 'could', 'may', 'might', 'must', 'and',
                 'or', 'but', 'if', 'then', 'so', 'that', 'this', 'it', 'to',
                 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as'}
    words = set(re.findall(r'\b\w+\b', text))
    return words - stopwords


def _check_outcome_match(
    actual: Any,
    expected: Any,
    threshold: float = 0.7
) -> Tuple[bool, float]:
    """Check if actual outcome matches expected."""
    if actual is None or expected is None:
        return False, 0.0

    # Direct equality check
    if actual == expected:
        return True, 1.0

    # String comparison
    if isinstance(actual, str) and isinstance(expected, str):
        actual_norm = _normalize_text(actual)
        expected_norm = _normalize_text(expected)

        # Exact match
        if actual_norm == expected_norm:
            return True, 1.0

        # Substring match
        if expected_norm in actual_norm or actual_norm in expected_norm:
            return True, 0.9

        # Keyword overlap
        actual_keywords = _extract_keywords(actual)
        expected_keywords = _extract_keywords(expected)

        if expected_keywords:
            overlap = len(actual_keywords & expected_keywords) / len(expected_keywords)
            return overlap >= threshold, overlap

    return False, 0.0


def _check_criteria_match(
    result: Any,
    trajectory: List[AgentStep],
    criteria: List[str]
) -> Tuple[int, int, List[str]]:
    """Check how many success criteria are met."""
    met = 0
    unmet = []

    result_str = str(result).lower() if result else ""
    all_observations = " ".join([
        (step.observation or "") + " " + (step.thought or "")
        for step in trajectory
    ]).lower()

    for criterion in criteria:
        criterion_lower = criterion.lower()
        keywords = _extract_keywords(criterion)

        # Check if criterion keywords appear in result or observations
        if keywords:
            result_match = sum(1 for kw in keywords if kw in result_str) / len(keywords)
            obs_match = sum(1 for kw in keywords if kw in all_observations) / len(keywords)

            if result_match >= 0.5 or obs_match >= 0.5:
                met += 1
            else:
                unmet.append(criterion)
        else:
            met += 1  # Empty criteria considered met

    return met, len(criteria), unmet


class TaskCompletion(BaseMetric[AgentTrajectoryInput]):
    """
    Evaluates whether the agent completed the assigned task.

    Checks:
    - Final outcome matches expected
    - Success criteria are met
    - Task was not abandoned

    Returns score from 0.0 to 1.0.
    """

    @property
    def metric_name(self) -> str:
        return "task_completion"

    def compute_one(self, inputs: AgentTrajectoryInput) -> Dict[str, Any]:
        if not inputs.trajectory:
            return {
                "output": 0.0,
                "reason": "Empty trajectory - no steps taken."
            }

        score_components = []
        reasons = []

        # Check if trajectory has a final step
        has_final = any(step.is_final for step in inputs.trajectory)
        if has_final:
            score_components.append(0.2)
            reasons.append("Agent reached final step")
        else:
            reasons.append("No final step marked")

        # Check expected result match
        if inputs.expected_result is not None:
            match, match_score = _check_outcome_match(
                inputs.final_result,
                inputs.expected_result
            )
            score_components.append(0.5 * match_score)
            if match:
                reasons.append(f"Result matches expected ({match_score:.0%})")
            else:
                reasons.append(f"Result mismatch (similarity: {match_score:.0%})")
        elif inputs.final_result is not None:
            # Has result but no expected to compare
            score_components.append(0.3)
            reasons.append("Produced result (no expected for comparison)")

        # Check success criteria
        if inputs.task.success_criteria:
            met, total, unmet = _check_criteria_match(
                inputs.final_result,
                inputs.trajectory,
                inputs.task.success_criteria
            )
            criteria_score = met / total if total > 0 else 1.0
            score_components.append(0.3 * criteria_score)
            reasons.append(f"Criteria: {met}/{total} met")
            if unmet:
                reasons.append(f"Unmet: {', '.join(unmet[:2])}")
        else:
            score_components.append(0.2)

        final_score = sum(score_components)

        return {
            "output": round(min(1.0, final_score), 4),
            "reason": ". ".join(reasons),
            "has_final_step": has_final,
            "result_produced": inputs.final_result is not None,
        }


class StepEfficiency(BaseMetric[AgentTrajectoryInput]):
    """
    Evaluates the efficiency of the agent's trajectory.

    Measures:
    - Number of steps vs optimal
    - Unnecessary/redundant steps
    - Failed actions that required retry

    Returns score from 0.0 to 1.0.
    """

    @property
    def metric_name(self) -> str:
        return "step_efficiency"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.expected_step_weight = self.config.get("expected_step_weight", 0.4)
        self.redundancy_weight = self.config.get("redundancy_weight", 0.3)
        self.failure_weight = self.config.get("failure_weight", 0.3)

    def compute_one(self, inputs: AgentTrajectoryInput) -> Dict[str, Any]:
        if not inputs.trajectory:
            return {
                "output": 0.0,
                "reason": "Empty trajectory."
            }

        total_steps = len(inputs.trajectory)
        details = {"total_steps": total_steps}

        # Calculate expected steps
        if inputs.expected_trajectory:
            expected_steps = len(inputs.expected_trajectory)
            step_ratio = min(1.0, expected_steps / total_steps) if total_steps > 0 else 0.0
            step_score = step_ratio * self.expected_step_weight
            details["expected_steps"] = expected_steps
            details["step_ratio"] = round(step_ratio, 3)
        elif inputs.task.max_steps:
            step_ratio = min(1.0, inputs.task.max_steps / total_steps) if total_steps > 0 else 0.0
            step_score = step_ratio * self.expected_step_weight
            details["max_steps"] = inputs.task.max_steps
        else:
            # No baseline - give partial credit if reasonable number of steps
            step_score = self.expected_step_weight * (1.0 if total_steps <= 10 else 10 / total_steps)

        # Detect redundant steps (same tool called with same arguments)
        tool_calls = []
        redundant_count = 0
        for step in inputs.trajectory:
            for tc in step.tool_calls:
                call_sig = (tc.name, str(sorted(tc.arguments.items())))
                if call_sig in tool_calls:
                    redundant_count += 1
                else:
                    tool_calls.append(call_sig)

        redundancy_ratio = 1.0 - (redundant_count / total_steps) if total_steps > 0 else 1.0
        redundancy_score = redundancy_ratio * self.redundancy_weight
        details["redundant_steps"] = redundant_count

        # Count failures
        failed_calls = sum(
            1 for step in inputs.trajectory
            for tc in step.tool_calls
            if not tc.success
        )
        total_calls = sum(len(step.tool_calls) for step in inputs.trajectory)
        failure_ratio = 1.0 - (failed_calls / total_calls) if total_calls > 0 else 1.0
        failure_score = failure_ratio * self.failure_weight
        details["failed_calls"] = failed_calls

        final_score = step_score + redundancy_score + failure_score

        reason_parts = [f"{total_steps} steps taken"]
        if redundant_count > 0:
            reason_parts.append(f"{redundant_count} redundant")
        if failed_calls > 0:
            reason_parts.append(f"{failed_calls} failed calls")

        return {
            "output": round(final_score, 4),
            "reason": ", ".join(reason_parts),
            "details": details,
        }


class ToolSelectionAccuracy(BaseMetric[AgentTrajectoryInput]):
    """
    Evaluates accuracy of tool selection by the agent.

    Measures:
    - Correct tools selected for the task
    - Appropriate arguments provided
    - No hallucinated/unavailable tools used

    Returns score from 0.0 to 1.0.
    """

    @property
    def metric_name(self) -> str:
        return "tool_selection_accuracy"

    def compute_one(self, inputs: AgentTrajectoryInput) -> Dict[str, Any]:
        # Collect all tools used
        tools_used = set()
        all_calls = []
        for step in inputs.trajectory:
            for tc in step.tool_calls:
                tools_used.add(tc.name)
                all_calls.append(tc)

        if not all_calls:
            return {
                "output": 1.0,
                "reason": "No tool calls made."
            }

        score_components = []
        reasons = []

        # Check if required tools were used
        if inputs.task.required_tools:
            required = set(inputs.task.required_tools)
            used_required = tools_used & required
            coverage = len(used_required) / len(required) if required else 1.0
            score_components.append(0.4 * coverage)
            reasons.append(f"Required tools: {len(used_required)}/{len(required)} used")

            # Check for unused required tools
            unused = required - tools_used
            if unused:
                reasons.append(f"Missing: {', '.join(unused)}")
        else:
            score_components.append(0.2)

        # Check for invalid tool usage (tools not in available list)
        if inputs.available_tools:
            available = set(inputs.available_tools)
            invalid_tools = tools_used - available
            if invalid_tools:
                invalid_penalty = len(invalid_tools) / len(tools_used)
                score_components.append(0.3 * (1.0 - invalid_penalty))
                reasons.append(f"Invalid tools used: {', '.join(invalid_tools)}")
            else:
                score_components.append(0.3)
                reasons.append("All tools valid")
        else:
            score_components.append(0.2)

        # Check tool call success rate
        successful = sum(1 for tc in all_calls if tc.success)
        success_rate = successful / len(all_calls)
        score_components.append(0.3 * success_rate)
        reasons.append(f"Success rate: {success_rate:.0%}")

        return {
            "output": round(sum(score_components), 4),
            "reason": ". ".join(reasons),
            "tools_used": list(tools_used),
            "total_calls": len(all_calls),
            "successful_calls": successful,
        }


class TrajectoryScore(BaseMetric[AgentTrajectoryInput]):
    """
    Comprehensive trajectory evaluation score.

    Combines:
    - Task completion (40%)
    - Step efficiency (30%)
    - Tool selection (30%)

    Returns overall score from 0.0 to 1.0.
    """

    @property
    def metric_name(self) -> str:
        return "trajectory_score"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.completion_weight = self.config.get("completion_weight", 0.4)
        self.efficiency_weight = self.config.get("efficiency_weight", 0.3)
        self.tool_weight = self.config.get("tool_weight", 0.3)

    def compute_one(self, inputs: AgentTrajectoryInput) -> Dict[str, Any]:
        # Compute component scores
        completion_metric = TaskCompletion()
        efficiency_metric = StepEfficiency()
        tool_metric = ToolSelectionAccuracy()

        completion_result = completion_metric.compute_one(inputs)
        efficiency_result = efficiency_metric.compute_one(inputs)
        tool_result = tool_metric.compute_one(inputs)

        # Weight and combine
        final_score = (
            completion_result["output"] * self.completion_weight +
            efficiency_result["output"] * self.efficiency_weight +
            tool_result["output"] * self.tool_weight
        )

        return {
            "output": round(final_score, 4),
            "reason": f"Completion: {completion_result['output']:.2f}, "
                     f"Efficiency: {efficiency_result['output']:.2f}, "
                     f"Tool Selection: {tool_result['output']:.2f}",
            "component_scores": {
                "task_completion": completion_result["output"],
                "step_efficiency": efficiency_result["output"],
                "tool_selection": tool_result["output"],
            },
            "completion_details": completion_result.get("reason"),
            "efficiency_details": efficiency_result.get("reason"),
            "tool_details": tool_result.get("reason"),
        }


class GoalProgress(BaseMetric[AgentTrajectoryInput]):
    """
    Evaluates progress towards the goal through the trajectory.

    Measures:
    - Incremental progress at each step
    - Consistency of direction
    - Goal proximity at end

    Useful for partial credit when task isn't fully completed.

    Returns score from 0.0 to 1.0.
    """

    @property
    def metric_name(self) -> str:
        return "goal_progress"

    def compute_one(self, inputs: AgentTrajectoryInput) -> Dict[str, Any]:
        if not inputs.trajectory:
            return {
                "output": 0.0,
                "reason": "Empty trajectory - no progress."
            }

        # Extract goal keywords from task description
        goal_keywords = _extract_keywords(inputs.task.description)
        if inputs.task.expected_outcome:
            goal_keywords |= _extract_keywords(inputs.task.expected_outcome)

        if not goal_keywords:
            return {
                "output": 0.5,
                "reason": "Could not extract goal keywords for progress tracking."
            }

        # Track progress through trajectory
        progress_scores = []
        for step in inputs.trajectory:
            step_text = " ".join(filter(None, [
                step.thought,
                step.action,
                step.observation,
                " ".join(tc.name for tc in step.tool_calls)
            ]))
            step_keywords = _extract_keywords(step_text)

            if step_keywords:
                overlap = len(step_keywords & goal_keywords) / len(goal_keywords)
                progress_scores.append(overlap)

        if not progress_scores:
            return {
                "output": 0.2,
                "reason": "No meaningful progress detected."
            }

        # Calculate overall progress
        avg_progress = sum(progress_scores) / len(progress_scores)
        final_progress = progress_scores[-1] if progress_scores else 0.0
        max_progress = max(progress_scores)

        # Weight final progress more heavily
        overall = 0.3 * avg_progress + 0.5 * final_progress + 0.2 * max_progress

        return {
            "output": round(overall, 4),
            "reason": f"Progress: avg={avg_progress:.2f}, final={final_progress:.2f}, max={max_progress:.2f}",
            "progress_by_step": [round(p, 3) for p in progress_scores],
            "final_progress": round(final_progress, 4),
        }
