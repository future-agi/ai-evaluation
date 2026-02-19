"""Tests for fi.evals.framework.evals.agentic module."""

import pytest
from fi.evals.framework.evals.agentic import (
    ToolUseCorrectnessEval,
    TrajectoryEfficiencyEval,
    GoalCompletionEval,
    ActionSafetyEval,
    ReasoningQualityEval,
    AgenticEvalResult,
    AgentAction,
)
from fi.evals.framework.protocols import EvalRegistry


class TestAgenticEvalResult:
    """Tests for AgenticEvalResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = AgenticEvalResult(
            score=0.85,
            passed=True,
        )

        assert result.score == 0.85
        assert result.passed is True
        assert result.confidence == 1.0
        assert result.trajectory_length == 0
        assert result.details == {}

    def test_with_details(self):
        """Test result with details."""
        result = AgenticEvalResult(
            score=0.9,
            passed=True,
            trajectory_length=5,
            details={"steps": 5, "loops": 0},
        )

        assert result.trajectory_length == 5
        assert result.details == {"steps": 5, "loops": 0}


class TestAgentAction:
    """Tests for AgentAction dataclass."""

    def test_basic_creation(self):
        """Test basic action creation."""
        action = AgentAction(
            action_type="tool_call",
            name="search",
        )

        assert action.action_type == "tool_call"
        assert action.name == "search"
        assert action.input is None
        assert action.output is None

    def test_with_io(self):
        """Test action with input/output."""
        action = AgentAction(
            action_type="tool_call",
            name="weather_api",
            input={"city": "Paris"},
            output={"temp": 22, "condition": "sunny"},
        )

        assert action.input == {"city": "Paris"}
        assert action.output == {"temp": 22, "condition": "sunny"}


class TestToolUseCorrectnessEval:
    """Tests for ToolUseCorrectnessEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = ToolUseCorrectnessEval()

        assert eval.name == "tool_use_correctness"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = ToolUseCorrectnessEval()
        errors = eval.validate_inputs({})

        assert "trajectory" in str(errors)
        assert "available_tools" in str(errors)

    def test_evaluate_valid_tool_use(self):
        """Test with valid tool usage."""
        eval = ToolUseCorrectnessEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "weather Paris"},
                {"type": "tool_call", "tool": "weather_api", "args": {"city": "Paris"}},
            ],
            "available_tools": ["search", "weather_api", "calculator"],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.score > 0.5
        assert result.details["valid_tool_count"] == 2

    def test_evaluate_invalid_tool(self):
        """Test with invalid tool usage."""
        eval = ToolUseCorrectnessEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "unknown_tool", "args": "test"},
            ],
            "available_tools": ["search", "calculator"],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["valid_tool_count"] == 0
        assert len(result.details["issues"]) > 0

    def test_evaluate_string_trajectory(self):
        """Test with string format trajectory."""
        eval = ToolUseCorrectnessEval()
        result = eval.evaluate({
            "trajectory": """
            Thought: I need to search for information
            Action: search(weather in Paris)
            Observation: It's sunny and 22°C
            """,
            "available_tools": ["search"],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.trajectory_length >= 1

    def test_evaluate_no_tool_calls(self):
        """Test with no tool calls."""
        eval = ToolUseCorrectnessEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "thought", "name": "thinking", "args": "Let me think..."},
            ],
            "available_tools": ["search"],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["reason"] == "no_tool_calls"


class TestTrajectoryEfficiencyEval:
    """Tests for TrajectoryEfficiencyEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = TrajectoryEfficiencyEval()

        assert eval.name == "trajectory_efficiency"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = TrajectoryEfficiencyEval()
        errors = eval.validate_inputs({})

        assert "trajectory" in str(errors)

    def test_evaluate_efficient_trajectory(self):
        """Test with efficient trajectory."""
        eval = TrajectoryEfficiencyEval(max_steps=10)
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "query"},
                {"type": "final_answer", "name": "final_answer", "input": "The answer is 42"},
            ],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.score > 0.5
        assert result.details["has_final_answer"] is True

    def test_evaluate_long_trajectory(self):
        """Test with inefficient long trajectory."""
        eval = TrajectoryEfficiencyEval(max_steps=5)
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": f"query{i}"}
                for i in range(10)
            ],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["num_steps"] == 10
        assert result.details["step_efficiency"] < 0.5

    def test_evaluate_with_optimal_steps(self):
        """Test with known optimal steps."""
        eval = TrajectoryEfficiencyEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "query"},
                {"type": "tool_call", "tool": "process", "args": "data"},
                {"type": "final_answer", "name": "final_answer", "input": "result"},
            ],
            "optimal_steps": 3,
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["step_efficiency"] == 1.0

    def test_evaluate_redundant_actions(self):
        """Test detection of redundant actions."""
        eval = TrajectoryEfficiencyEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "query"},
                {"type": "tool_call", "tool": "search", "args": "query"},
                {"type": "tool_call", "tool": "search", "args": "query"},
            ],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["redundancy"] > 0

    def test_evaluate_empty_trajectory(self):
        """Test with empty trajectory."""
        eval = TrajectoryEfficiencyEval()
        result = eval.evaluate({
            "trajectory": [],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.score == 0.0
        assert result.details["reason"] == "empty_trajectory"


class TestGoalCompletionEval:
    """Tests for GoalCompletionEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = GoalCompletionEval()

        assert eval.name == "goal_completion"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = GoalCompletionEval()
        errors = eval.validate_inputs({})

        assert "trajectory" in str(errors)
        assert "goal" in str(errors)

    def test_evaluate_completed_goal(self):
        """Test with completed goal."""
        eval = GoalCompletionEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "weather_api", "args": {"city": "Paris"}},
                {"type": "final_answer", "name": "final_answer", "input": "The weather in Paris is sunny and 22°C"},
            ],
            "goal": "What is the weather in Paris?",
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["has_final_answer"] is True
        assert result.score > 0.5

    def test_evaluate_incomplete_goal(self):
        """Test with incomplete goal (no final answer)."""
        eval = GoalCompletionEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "weather"},
            ],
            "goal": "What is the weather in Paris?",
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["has_final_answer"] is False
        assert result.score < 0.5

    def test_evaluate_with_expected_answer(self):
        """Test with expected answer comparison."""
        eval = GoalCompletionEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "final_answer", "name": "final_answer", "input": "Sunny, 22°C"},
            ],
            "goal": "What is the weather?",
            "expected_answer": "Sunny, 22°C",
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["answer_relevance"] > 0.5

    def test_evaluate_keyword_coverage(self):
        """Test keyword coverage detection."""
        eval = GoalCompletionEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "Paris weather", "result": "Sunny day in Paris"},
                {"type": "final_answer", "name": "final_answer", "input": "Paris weather is sunny"},
            ],
            "goal": "Find the weather in Paris",
        })

        assert isinstance(result, AgenticEvalResult)
        assert "goal_keywords" in result.details


class TestActionSafetyEval:
    """Tests for ActionSafetyEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = ActionSafetyEval()

        assert eval.name == "action_safety"
        assert eval.version == "1.0.0"

    def test_high_default_threshold(self):
        """Test high default threshold for safety."""
        eval = ActionSafetyEval()
        assert eval.threshold == 0.9

    def test_required_fields(self):
        """Test required field validation."""
        eval = ActionSafetyEval()
        errors = eval.validate_inputs({})

        assert "trajectory" in str(errors)

    def test_evaluate_safe_actions(self):
        """Test with safe actions."""
        eval = ActionSafetyEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "weather Paris"},
                {"type": "tool_call", "tool": "read_file", "args": "data.json"},
            ],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.score == 1.0
        assert result.passed is True

    def test_evaluate_dangerous_patterns(self):
        """Test detection of dangerous patterns."""
        eval = ActionSafetyEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "shell", "args": "rm -rf /"},
            ],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.score < 1.0
        assert len(result.details["dangerous_actions"]) > 0

    def test_evaluate_sql_injection(self):
        """Test detection of SQL injection patterns."""
        eval = ActionSafetyEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "database", "args": "DELETE FROM users"},
            ],
        })

        assert isinstance(result, AgenticEvalResult)
        assert len(result.details["dangerous_actions"]) > 0

    def test_evaluate_custom_forbidden(self):
        """Test with custom forbidden actions."""
        eval = ActionSafetyEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "custom", "args": "forbidden_operation"},
            ],
            "forbidden_actions": ["forbidden_operation"],
        })

        assert isinstance(result, AgenticEvalResult)
        assert len(result.details["dangerous_actions"]) > 0

    def test_evaluate_empty_trajectory(self):
        """Test with empty trajectory."""
        eval = ActionSafetyEval()
        result = eval.evaluate({
            "trajectory": [],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.score == 1.0


class TestReasoningQualityEval:
    """Tests for ReasoningQualityEval."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_name_and_version(self):
        """Test evaluation name and version."""
        eval = ReasoningQualityEval()

        assert eval.name == "reasoning_quality"
        assert eval.version == "1.0.0"

    def test_required_fields(self):
        """Test required field validation."""
        eval = ReasoningQualityEval()
        errors = eval.validate_inputs({})

        assert "trajectory" in str(errors)

    def test_evaluate_good_reasoning(self):
        """Test with good reasoning."""
        eval = ReasoningQualityEval()
        result = eval.evaluate({
            "trajectory": """
            Thought: I need to find the weather because the user asked about Paris weather
            Action: weather_api(Paris)
            Observation: Sunny, 22°C
            Thought: Since I have the weather data, I should provide the answer
            Final Answer: The weather in Paris is sunny with a temperature of 22°C
            """,
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["thought_count"] >= 2
        assert result.details["reasoning_indicators"] > 0

    def test_evaluate_no_explicit_thoughts(self):
        """Test with no explicit thoughts."""
        eval = ReasoningQualityEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "query"},
                {"type": "final_answer", "name": "final_answer", "input": "answer"},
            ],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["reason"] == "no_explicit_thoughts"

    def test_evaluate_implicit_reasoning(self):
        """Test detection of implicit reasoning."""
        eval = ReasoningQualityEval()
        result = eval.evaluate({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "I need to search because the user wants info"},
            ],
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details.get("has_implicit_reasoning", False) is True

    def test_evaluate_reasoning_indicators(self):
        """Test reasoning indicator detection."""
        eval = ReasoningQualityEval()
        result = eval.evaluate({
            "trajectory": """
            Thought: First, I need to understand the problem
            Thought: Therefore, I should search for relevant data
            Thought: However, I also need to consider edge cases
            """,
        })

        assert isinstance(result, AgenticEvalResult)
        assert result.details["reasoning_indicators"] >= 3


class TestIntegrationWithFramework:
    """Tests for integration with the evaluation framework."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_use_with_evaluator(self):
        """Test using agentic evals with Evaluator."""
        from fi.evals.framework import Evaluator, ExecutionMode

        evaluator = Evaluator(
            evaluations=[
                ToolUseCorrectnessEval(),
                TrajectoryEfficiencyEval(),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "query"},
                {"type": "final_answer", "name": "final_answer", "input": "result"},
            ],
            "available_tools": ["search", "calculator"],
        })

        assert len(result.results) == 2
        for r in result.results:
            assert hasattr(r.value, "score")
            assert hasattr(r.value, "trajectory_length")

    def test_use_with_async_evaluator(self):
        """Test using agentic evals with async evaluator."""
        from fi.evals.framework import async_evaluator

        evaluator = async_evaluator(
            GoalCompletionEval(),
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "trajectory": [
                {"type": "final_answer", "name": "final_answer", "input": "Paris is sunny"},
            ],
            "goal": "What is the weather in Paris?",
        })

        batch = result.wait()
        assert len(batch.results) == 1
        assert batch.results[0].value.passed is True

        evaluator.shutdown()

    def test_span_attributes_format(self):
        """Test span attributes are OTEL-compatible."""
        eval = ToolUseCorrectnessEval()
        result = AgenticEvalResult(
            score=0.85,
            passed=True,
            trajectory_length=5,
        )

        attrs = eval.get_span_attributes(result)

        # All values should be OTEL-compatible types
        for key, value in attrs.items():
            assert isinstance(value, (str, int, float, bool))

    def test_multiple_agentic_evals(self):
        """Test running multiple agentic evals together."""
        from fi.evals.framework import Evaluator, ExecutionMode

        evaluator = Evaluator(
            evaluations=[
                ToolUseCorrectnessEval(),
                GoalCompletionEval(),
                ActionSafetyEval(),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "weather Paris"},
                {"type": "final_answer", "name": "final_answer", "input": "Sunny in Paris"},
            ],
            "goal": "Find weather in Paris",
            "available_tools": ["search"],
        })

        assert len(result.results) == 3
        assert result.success_rate == 1.0


class TestTrajectoryParsing:
    """Tests for trajectory parsing functionality."""

    def test_parse_dict_list(self):
        """Test parsing list of dicts."""
        eval = ToolUseCorrectnessEval()
        actions = eval._parse_trajectory([
            {"type": "tool_call", "tool": "search", "args": "query"},
            {"action_type": "thought", "name": "thinking", "input": "hmm"},
        ])

        assert len(actions) == 2
        assert actions[0].name == "search"
        assert actions[1].action_type == "thought"

    def test_parse_tuple_list(self):
        """Test parsing list of tuples."""
        eval = ToolUseCorrectnessEval()
        actions = eval._parse_trajectory([
            ("search(query)", "result 1"),
            ("calculate(2+2)", "4"),
        ])

        assert len(actions) == 2
        assert actions[0].name == "search"
        assert actions[0].output == "result 1"

    def test_parse_string_format(self):
        """Test parsing string format."""
        eval = ToolUseCorrectnessEval()
        actions = eval._parse_trajectory("""
        Thought: I need to search
        Action: search(weather)
        Observation: It's sunny
        Final Answer: The weather is sunny
        """)

        assert len(actions) >= 3
        action_types = [a.action_type for a in actions]
        assert "thought" in action_types
        assert "tool_call" in action_types
        assert "final_answer" in action_types

    def test_parse_agent_action_list(self):
        """Test parsing list of AgentAction objects."""
        eval = ToolUseCorrectnessEval()
        actions = eval._parse_trajectory([
            AgentAction(action_type="tool_call", name="search", input="query"),
            AgentAction(action_type="final_answer", name="final_answer", input="result"),
        ])

        assert len(actions) == 2
        assert actions[0].name == "search"
        assert actions[1].action_type == "final_answer"
