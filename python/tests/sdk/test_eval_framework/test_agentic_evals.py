"""Tests for fi.evals.framework.evals.agentic module."""

import pytest
from fi.evals.framework.evals.agentic import (
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
                ActionSafetyEval(),
                ReasoningQualityEval(),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "trajectory": [
                {"type": "tool_call", "tool": "search", "args": "query"},
                {"type": "final_answer", "name": "final_answer", "input": "result"},
            ],
        })

        assert len(result.results) == 2
        for r in result.results:
            assert hasattr(r.value, "score")
            assert hasattr(r.value, "trajectory_length")

    def test_span_attributes_format(self):
        """Test span attributes are OTEL-compatible."""
        eval = ActionSafetyEval()
        result = AgenticEvalResult(
            score=0.85,
            passed=True,
            trajectory_length=5,
        )

        attrs = eval.get_span_attributes(result)

        # All values should be OTEL-compatible types
        for key, value in attrs.items():
            assert isinstance(value, (str, int, float, bool))


class TestTrajectoryParsing:
    """Tests for trajectory parsing functionality."""

    def test_parse_dict_list(self):
        """Test parsing list of dicts."""
        eval = ActionSafetyEval()
        actions = eval._parse_trajectory([
            {"type": "tool_call", "tool": "search", "args": "query"},
            {"action_type": "thought", "name": "thinking", "input": "hmm"},
        ])

        assert len(actions) == 2
        assert actions[0].name == "search"
        assert actions[1].action_type == "thought"

    def test_parse_tuple_list(self):
        """Test parsing list of tuples."""
        eval = ActionSafetyEval()
        actions = eval._parse_trajectory([
            ("search(query)", "result 1"),
            ("calculate(2+2)", "4"),
        ])

        assert len(actions) == 2
        assert actions[0].name == "search"
        assert actions[0].output == "result 1"

    def test_parse_string_format(self):
        """Test parsing string format."""
        eval = ActionSafetyEval()
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
        eval = ActionSafetyEval()
        actions = eval._parse_trajectory([
            AgentAction(action_type="tool_call", name="search", input="query"),
            AgentAction(action_type="final_answer", name="final_answer", input="result"),
        ])

        assert len(actions) == 2
        assert actions[0].name == "search"
        assert actions[1].action_type == "final_answer"
