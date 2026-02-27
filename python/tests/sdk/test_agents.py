"""
Comprehensive tests for Agent Evaluation metrics.

Tests cover:
- TaskCompletion
- StepEfficiency
- ToolSelectionAccuracy
- TrajectoryScore
- GoalProgress
"""

import pytest
from fi.evals.metrics.agents import (
    AgentTrajectoryInput,
    AgentStep,
    ToolCall,
    TaskDefinition,
    TaskCompletion,
    StepEfficiency,
    ToolSelectionAccuracy,
    TrajectoryScore,
    GoalProgress,
)


def create_simple_trajectory(num_steps: int = 3, with_tools: bool = True, mark_final: bool = True):
    """Helper to create a simple test trajectory."""
    steps = []
    for i in range(num_steps):
        tool_calls = []
        if with_tools:
            tool_calls.append(ToolCall(
                name=f"tool_{i+1}",
                arguments={"arg": f"value_{i+1}"},
                result=f"result_{i+1}",
                success=True
            ))
        steps.append(AgentStep(
            step_number=i + 1,
            thought=f"Thinking about step {i+1}",
            action=f"Action for step {i+1}",
            tool_calls=tool_calls,
            observation=f"Observed result {i+1}",
            is_final=(i == num_steps - 1 and mark_final)
        ))
    return steps


class TestTaskCompletion:
    """Tests for TaskCompletion metric."""

    def test_successful_completion(self):
        """Test trajectory with successful task completion."""
        metric = TaskCompletion()
        input_data = AgentTrajectoryInput(
            trajectory=create_simple_trajectory(3, mark_final=True),
            task=TaskDefinition(description="Complete a simple task"),
            final_result="Task completed successfully",
            expected_result="Task completed successfully"
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.7
        assert result["has_final_step"] is True

    def test_no_final_step(self):
        """Test trajectory without final step marked."""
        metric = TaskCompletion()
        input_data = AgentTrajectoryInput(
            trajectory=create_simple_trajectory(3, mark_final=False),
            task=TaskDefinition(description="Complete a task"),
            final_result="Some result"
        )
        result = metric.compute_one(input_data)
        assert result["has_final_step"] is False
        # Should still get some credit for having steps

    def test_empty_trajectory(self):
        """Test with empty trajectory."""
        metric = TaskCompletion()
        input_data = AgentTrajectoryInput(
            trajectory=[],
            task=TaskDefinition(description="Do something")
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 0.0
        assert "Empty trajectory" in result["reason"]

    def test_with_success_criteria(self):
        """Test evaluation with success criteria."""
        metric = TaskCompletion()
        trajectory = create_simple_trajectory(2)
        trajectory[1].observation = "Found the file and extracted data"

        input_data = AgentTrajectoryInput(
            trajectory=trajectory,
            task=TaskDefinition(
                description="Find and extract data from file",
                success_criteria=["Find file", "Extract data"]
            ),
            final_result="Data extracted: [1, 2, 3]"
        )
        result = metric.compute_one(input_data)
        assert "Criteria" in result["reason"]

    def test_result_mismatch(self):
        """Test with mismatched result."""
        metric = TaskCompletion()
        input_data = AgentTrajectoryInput(
            trajectory=create_simple_trajectory(2, mark_final=True),
            task=TaskDefinition(description="Calculate 2+2"),
            final_result="5",
            expected_result="4"
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 0.8
        assert "mismatch" in result["reason"].lower()


class TestStepEfficiency:
    """Tests for StepEfficiency metric."""

    def test_efficient_trajectory(self):
        """Test efficient trajectory with minimal steps."""
        metric = StepEfficiency()
        input_data = AgentTrajectoryInput(
            trajectory=create_simple_trajectory(3),
            task=TaskDefinition(description="Do task", max_steps=5)
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.7
        assert result["details"]["total_steps"] == 3

    def test_inefficient_trajectory(self):
        """Test inefficient trajectory with too many steps."""
        metric = StepEfficiency()
        input_data = AgentTrajectoryInput(
            trajectory=create_simple_trajectory(10),
            task=TaskDefinition(description="Do task", max_steps=3)
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 0.8

    def test_redundant_steps(self):
        """Test trajectory with redundant tool calls."""
        metric = StepEfficiency()
        steps = []
        for i in range(4):
            steps.append(AgentStep(
                step_number=i + 1,
                thought="Thinking",
                tool_calls=[ToolCall(
                    name="search",  # Same tool
                    arguments={"query": "test"},  # Same arguments
                    result="result",
                    success=True
                )],
                is_final=(i == 3)
            ))

        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(description="Search for something")
        )
        result = metric.compute_one(input_data)
        assert result["details"]["redundant_steps"] >= 3

    def test_failed_tool_calls(self):
        """Test trajectory with failed tool calls."""
        metric = StepEfficiency()
        steps = [
            AgentStep(
                step_number=1,
                thought="Try first",
                tool_calls=[ToolCall(name="tool1", success=False, error="Failed")],
            ),
            AgentStep(
                step_number=2,
                thought="Retry",
                tool_calls=[ToolCall(name="tool1", success=True)],
                is_final=True
            )
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(description="Use tool")
        )
        result = metric.compute_one(input_data)
        assert result["details"]["failed_calls"] == 1

    def test_empty_trajectory(self):
        """Test with empty trajectory."""
        metric = StepEfficiency()
        input_data = AgentTrajectoryInput(
            trajectory=[],
            task=TaskDefinition(description="Task")
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 0.0


class TestToolSelectionAccuracy:
    """Tests for ToolSelectionAccuracy metric."""

    def test_correct_tools(self):
        """Test with correct tool selection."""
        metric = ToolSelectionAccuracy()
        steps = [
            AgentStep(
                step_number=1,
                tool_calls=[ToolCall(name="search", success=True)],
            ),
            AgentStep(
                step_number=2,
                tool_calls=[ToolCall(name="read_file", success=True)],
                is_final=True
            )
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(
                description="Search and read",
                required_tools=["search", "read_file"]
            ),
            available_tools=["search", "read_file", "write_file"]
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.8
        assert "search" in result["tools_used"]
        assert "read_file" in result["tools_used"]

    def test_missing_required_tools(self):
        """Test when required tools are not used."""
        metric = ToolSelectionAccuracy()
        steps = [
            AgentStep(
                step_number=1,
                tool_calls=[ToolCall(name="search", success=True)],
                is_final=True
            )
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(
                description="Search and save",
                required_tools=["search", "write_file"]  # write_file not used
            )
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 1.0
        assert "Missing" in result["reason"]

    def test_invalid_tool_usage(self):
        """Test when unavailable tools are used."""
        metric = ToolSelectionAccuracy()
        steps = [
            AgentStep(
                step_number=1,
                tool_calls=[ToolCall(name="invalid_tool", success=True)],
                is_final=True
            )
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(description="Do something"),
            available_tools=["search", "read"]  # invalid_tool not available
        )
        result = metric.compute_one(input_data)
        assert "Invalid" in result["reason"]

    def test_no_tool_calls(self):
        """Test trajectory with no tool calls."""
        metric = ToolSelectionAccuracy()
        input_data = AgentTrajectoryInput(
            trajectory=create_simple_trajectory(2, with_tools=False),
            task=TaskDefinition(description="Task without tools")
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0
        assert "No tool calls" in result["reason"]


class TestTrajectoryScore:
    """Tests for TrajectoryScore metric."""

    def test_high_quality_trajectory(self):
        """Test high-quality trajectory."""
        metric = TrajectoryScore()
        input_data = AgentTrajectoryInput(
            trajectory=create_simple_trajectory(3),
            task=TaskDefinition(
                description="Complete the task",
                max_steps=5,
                required_tools=["tool_1", "tool_2", "tool_3"]
            ),
            final_result="Task done",
            expected_result="Task done",
            available_tools=["tool_1", "tool_2", "tool_3", "tool_4"]
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.5
        assert "component_scores" in result

    def test_component_scores_returned(self):
        """Test that component scores are returned."""
        metric = TrajectoryScore()
        input_data = AgentTrajectoryInput(
            trajectory=create_simple_trajectory(2),
            task=TaskDefinition(description="Do something")
        )
        result = metric.compute_one(input_data)

        assert "component_scores" in result
        assert "task_completion" in result["component_scores"]
        assert "step_efficiency" in result["component_scores"]
        assert "tool_selection" in result["component_scores"]

    def test_custom_weights(self):
        """Test with custom weight configuration."""
        metric = TrajectoryScore(config={
            "completion_weight": 0.6,
            "efficiency_weight": 0.2,
            "tool_weight": 0.2
        })
        input_data = AgentTrajectoryInput(
            trajectory=create_simple_trajectory(2),
            task=TaskDefinition(description="Task")
        )
        result = metric.compute_one(input_data)
        assert "output" in result


class TestGoalProgress:
    """Tests for GoalProgress metric."""

    def test_good_progress(self):
        """Test trajectory with good goal progress."""
        metric = GoalProgress()
        steps = [
            AgentStep(
                step_number=1,
                thought="Need to search for weather data",
                action="Search weather API",
                tool_calls=[ToolCall(name="search", arguments={"query": "weather"})],
                observation="Found weather API endpoint"
            ),
            AgentStep(
                step_number=2,
                thought="Now get the weather data",
                action="Call weather API",
                tool_calls=[ToolCall(name="api_call", arguments={"endpoint": "weather"})],
                observation="Got weather data: sunny, 72F",
                is_final=True
            )
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(
                description="Get current weather data",
                expected_outcome="Weather information"
            )
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.3
        assert "progress_by_step" in result

    def test_no_progress(self):
        """Test trajectory with no goal progress."""
        metric = GoalProgress()
        steps = [
            AgentStep(
                step_number=1,
                thought="Completely unrelated thought",
                action="Do something else",
                observation="Nothing relevant"
            )
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(
                description="Calculate mathematical formula",
                expected_outcome="Numerical result"
            )
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 0.5

    def test_empty_trajectory(self):
        """Test with empty trajectory."""
        metric = GoalProgress()
        input_data = AgentTrajectoryInput(
            trajectory=[],
            task=TaskDefinition(description="Do something")
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 0.0


class TestRealWorldScenarios:
    """Real-world e2e test scenarios."""

    def test_code_generation_agent(self):
        """Test code generation agent trajectory."""
        metric = TrajectoryScore()
        steps = [
            AgentStep(
                step_number=1,
                thought="User wants a Python function to calculate fibonacci",
                action="Plan the implementation",
                observation="Will create recursive fibonacci function"
            ),
            AgentStep(
                step_number=2,
                thought="Write the function",
                action="Generate code",
                tool_calls=[ToolCall(
                    name="code_writer",
                    arguments={"language": "python", "task": "fibonacci"},
                    result="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
                    success=True
                )],
                observation="Code generated"
            ),
            AgentStep(
                step_number=3,
                thought="Test the function",
                action="Run tests",
                tool_calls=[ToolCall(
                    name="code_runner",
                    arguments={"code": "fib(10)"},
                    result="55",
                    success=True
                )],
                observation="Tests passed",
                is_final=True
            )
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(
                description="Write a fibonacci function in Python",
                expected_outcome="Working fibonacci implementation",
                required_tools=["code_writer", "code_runner"]
            ),
            final_result="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
            available_tools=["code_writer", "code_runner", "file_editor"]
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.5

    def test_research_agent(self):
        """Test research agent trajectory."""
        metric = TrajectoryScore()
        steps = [
            AgentStep(
                step_number=1,
                thought="Need to search for information about AI",
                tool_calls=[ToolCall(
                    name="web_search",
                    arguments={"query": "artificial intelligence history"},
                    result="Found 10 relevant articles",
                    success=True
                )]
            ),
            AgentStep(
                step_number=2,
                thought="Read the first article",
                tool_calls=[ToolCall(
                    name="read_url",
                    arguments={"url": "https://example.com/ai-history"},
                    result="Article content about AI history...",
                    success=True
                )]
            ),
            AgentStep(
                step_number=3,
                thought="Summarize findings",
                tool_calls=[ToolCall(
                    name="summarize",
                    arguments={"text": "Article content..."},
                    result="AI was founded in 1956...",
                    success=True
                )],
                is_final=True
            )
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(
                description="Research the history of artificial intelligence",
                required_tools=["web_search", "read_url", "summarize"],
                max_steps=5
            ),
            final_result="AI was founded in 1956...",
            available_tools=["web_search", "read_url", "summarize", "save_note"]
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.6

    def test_data_analysis_agent(self):
        """Test data analysis agent trajectory."""
        metric = GoalProgress()
        steps = [
            AgentStep(
                step_number=1,
                thought="Load the CSV file for analysis",
                tool_calls=[ToolCall(
                    name="read_file",
                    arguments={"path": "data.csv"},
                    result="Loaded 1000 rows",
                    success=True
                )],
                observation="Data loaded successfully"
            ),
            AgentStep(
                step_number=2,
                thought="Calculate statistics",
                tool_calls=[ToolCall(
                    name="calculate_stats",
                    arguments={"columns": ["price", "quantity"]},
                    result={"mean_price": 50.5, "total_quantity": 5000},
                    success=True
                )],
                observation="Statistics calculated"
            ),
            AgentStep(
                step_number=3,
                thought="Create visualization",
                tool_calls=[ToolCall(
                    name="create_chart",
                    arguments={"type": "bar", "data": "price_by_category"},
                    result="chart_saved.png",
                    success=True
                )],
                observation="Chart created",
                is_final=True
            )
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(
                description="Analyze sales data and create visualization",
                expected_outcome="Statistics and chart of sales data"
            )
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.3

    def test_batch_evaluation(self):
        """Test batch evaluation of multiple trajectories."""
        metric = TrajectoryScore()
        inputs = [
            AgentTrajectoryInput(
                trajectory=create_simple_trajectory(2),
                task=TaskDefinition(description="Task 1"),
                final_result="Done"
            ),
            AgentTrajectoryInput(
                trajectory=create_simple_trajectory(3),
                task=TaskDefinition(description="Task 2"),
                final_result="Completed"
            ),
        ]
        results = metric.evaluate(inputs)
        assert len(results.eval_results) == 2
        assert all(r.output >= 0.0 for r in results.eval_results)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_step_trajectory(self):
        """Test trajectory with single step."""
        metric = TrajectoryScore()
        input_data = AgentTrajectoryInput(
            trajectory=[AgentStep(
                step_number=1,
                thought="One and done",
                action="Complete immediately",
                is_final=True
            )],
            task=TaskDefinition(description="Quick task")
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.0

    def test_trajectory_with_no_thoughts(self):
        """Test trajectory without thought field."""
        metric = GoalProgress()
        steps = [
            AgentStep(step_number=1, action="Do something", observation="Result"),
            AgentStep(step_number=2, action="Do more", is_final=True)
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(description="Complete task")
        )
        result = metric.compute_one(input_data)
        assert "output" in result

    def test_very_long_trajectory(self):
        """Test with very long trajectory."""
        metric = StepEfficiency()
        input_data = AgentTrajectoryInput(
            trajectory=create_simple_trajectory(50),
            task=TaskDefinition(description="Long task", max_steps=10)
        )
        result = metric.compute_one(input_data)
        # Should penalize for being over max_steps
        assert result["output"] < 0.8

    def test_all_failed_tool_calls(self):
        """Test trajectory where all tool calls fail."""
        metric = ToolSelectionAccuracy()
        steps = [
            AgentStep(
                step_number=1,
                tool_calls=[
                    ToolCall(name="tool1", success=False, error="Error 1"),
                    ToolCall(name="tool2", success=False, error="Error 2"),
                ],
                is_final=True
            )
        ]
        input_data = AgentTrajectoryInput(
            trajectory=steps,
            task=TaskDefinition(description="Task")
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 0.5
        assert result["successful_calls"] == 0
