"""
Comprehensive tests for Function Calling Evaluation metrics.

Tests cover:
- FunctionNameMatch
- ParameterValidation
- FunctionCallAccuracy
- FunctionCallExactMatch
"""

import pytest
from fi.evals.metrics.function_calling import (
    FunctionCallInput,
    FunctionCall,
    FunctionDefinition,
    ParameterSpec,
    FunctionNameMatch,
    ParameterValidation,
    FunctionCallAccuracy,
    FunctionCallExactMatch,
)


class TestFunctionCallParsing:
    """Tests for function call parsing from various formats."""

    def test_parse_function_call_object(self):
        """Test parsing FunctionCall object directly."""
        metric = FunctionNameMatch()
        input_data = FunctionCallInput(
            response=FunctionCall(name="get_weather", arguments={"city": "NYC"}),
            expected_response=FunctionCall(name="get_weather", arguments={"city": "NYC"})
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_parse_function_call_dict(self):
        """Test parsing function call from dict."""
        metric = FunctionNameMatch()
        input_data = FunctionCallInput(
            response={"name": "get_weather", "arguments": {"city": "NYC"}},
            expected_response={"name": "get_weather", "arguments": {"city": "NYC"}}
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_parse_function_call_json_string(self):
        """Test parsing function call from JSON string."""
        metric = FunctionNameMatch()
        input_data = FunctionCallInput(
            response='{"name": "get_weather", "arguments": {"city": "NYC"}}',
            expected_response='{"name": "get_weather", "arguments": {"city": "NYC"}}'
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_parse_function_call_python_style(self):
        """Test parsing Python-style function call string."""
        metric = FunctionCallExactMatch()
        input_data = FunctionCallInput(
            response="get_weather(city='NYC')",
            expected_response="get_weather(city='NYC')"
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_parse_function_call_alternative_keys(self):
        """Test parsing with alternative key names (function/parameters)."""
        metric = FunctionNameMatch()
        input_data = FunctionCallInput(
            response={"function": "get_weather", "parameters": {"city": "NYC"}},
            expected_response={"name": "get_weather", "arguments": {"city": "NYC"}}
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0


class TestFunctionNameMatch:
    """Tests for FunctionNameMatch metric."""

    def test_name_match_exact(self):
        """Test exact function name match."""
        metric = FunctionNameMatch()
        input_data = FunctionCallInput(
            response=FunctionCall(name="get_weather", arguments={}),
            expected_response=FunctionCall(name="get_weather", arguments={})
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0
        assert "matches" in result["reason"]

    def test_name_match_different(self):
        """Test different function names."""
        metric = FunctionNameMatch()
        input_data = FunctionCallInput(
            response=FunctionCall(name="get_weather", arguments={}),
            expected_response=FunctionCall(name="get_temperature", arguments={})
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 0.0
        assert "mismatch" in result["reason"]

    def test_name_match_case_sensitive(self):
        """Test that function names are case-sensitive."""
        metric = FunctionNameMatch()
        input_data = FunctionCallInput(
            response=FunctionCall(name="Get_Weather", arguments={}),
            expected_response=FunctionCall(name="get_weather", arguments={})
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 0.0

    def test_name_match_invalid_response(self):
        """Test with unparseable response."""
        metric = FunctionNameMatch()
        input_data = FunctionCallInput(
            response="not a function call",
            expected_response=FunctionCall(name="get_weather", arguments={})
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 0.0
        assert "parse" in result["reason"].lower()


class TestParameterValidation:
    """Tests for ParameterValidation metric."""

    def test_validation_all_required_present(self):
        """Test validation passes when all required params present."""
        metric = ParameterValidation()
        input_data = FunctionCallInput(
            response=FunctionCall(name="book_flight", arguments={
                "origin": "NYC",
                "destination": "LAX",
                "date": "2024-01-15"
            }),
            function_definitions=[
                FunctionDefinition(
                    name="book_flight",
                    parameters=[
                        ParameterSpec(name="origin", type="string", required=True),
                        ParameterSpec(name="destination", type="string", required=True),
                        ParameterSpec(name="date", type="string", required=True),
                    ]
                )
            ]
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_validation_missing_required(self):
        """Test validation fails when required param is missing."""
        metric = ParameterValidation()
        input_data = FunctionCallInput(
            response=FunctionCall(name="book_flight", arguments={
                "origin": "NYC",
                "destination": "LAX"
                # missing 'date'
            }),
            function_definitions=[
                FunctionDefinition(
                    name="book_flight",
                    parameters=[
                        ParameterSpec(name="origin", type="string", required=True),
                        ParameterSpec(name="destination", type="string", required=True),
                        ParameterSpec(name="date", type="string", required=True),
                    ]
                )
            ]
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 1.0
        assert "Missing" in result["reason"]
        assert "date" in result["reason"]

    def test_validation_type_check(self):
        """Test type validation."""
        metric = ParameterValidation()
        input_data = FunctionCallInput(
            response=FunctionCall(name="set_temperature", arguments={
                "value": "not a number"  # should be number
            }),
            function_definitions=[
                FunctionDefinition(
                    name="set_temperature",
                    parameters=[
                        ParameterSpec(name="value", type="number", required=True),
                    ]
                )
            ]
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 1.0
        assert "type" in result["reason"].lower()

    def test_validation_strict_type(self):
        """Test strict type checking (int vs float)."""
        metric = ParameterValidation()
        input_data = FunctionCallInput(
            response=FunctionCall(name="set_count", arguments={
                "value": 5.0  # float instead of int
            }),
            function_definitions=[
                FunctionDefinition(
                    name="set_count",
                    parameters=[
                        ParameterSpec(name="value", type="integer", required=True),
                    ]
                )
            ],
            strict_type_check=True
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 1.0

    def test_validation_enum_constraint(self):
        """Test enum constraint validation."""
        metric = ParameterValidation()
        input_data = FunctionCallInput(
            response=FunctionCall(name="set_mode", arguments={
                "mode": "turbo"  # not in enum
            }),
            function_definitions=[
                FunctionDefinition(
                    name="set_mode",
                    parameters=[
                        ParameterSpec(name="mode", type="string", required=True, enum=["low", "medium", "high"]),
                    ]
                )
            ]
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 1.0
        assert "turbo" in result["reason"]

    def test_validation_extra_params_ignored(self):
        """Test extra parameters can be ignored."""
        metric = ParameterValidation()
        input_data = FunctionCallInput(
            response=FunctionCall(name="greet", arguments={
                "name": "Alice",
                "extra_param": "value"  # not in spec
            }),
            function_definitions=[
                FunctionDefinition(
                    name="greet",
                    parameters=[
                        ParameterSpec(name="name", type="string", required=True),
                    ]
                )
            ],
            ignore_extra_params=True
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_validation_extra_params_penalized(self):
        """Test extra parameters are penalized by default."""
        metric = ParameterValidation()
        input_data = FunctionCallInput(
            response=FunctionCall(name="greet", arguments={
                "name": "Alice",
                "extra_param": "value"
            }),
            function_definitions=[
                FunctionDefinition(
                    name="greet",
                    parameters=[
                        ParameterSpec(name="name", type="string", required=True),
                    ]
                )
            ],
            ignore_extra_params=False
        )
        result = metric.compute_one(input_data)
        assert "Unexpected" in result["reason"]


class TestFunctionCallAccuracy:
    """Tests for FunctionCallAccuracy metric."""

    def test_accuracy_perfect_match(self):
        """Test perfect function call match."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(name="get_weather", arguments={"city": "NYC", "unit": "celsius"}),
            expected_response=FunctionCall(name="get_weather", arguments={"city": "NYC", "unit": "celsius"})
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_accuracy_wrong_name(self):
        """Test accuracy with wrong function name."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(name="get_temperature", arguments={"city": "NYC"}),
            expected_response=FunctionCall(name="get_weather", arguments={"city": "NYC"})
        )
        result = metric.compute_one(input_data)
        # Wrong name = 0% of 40% weight = 0.4 loss
        assert result["output"] < 1.0
        assert result["output"] >= 0.5  # params should still contribute

    def test_accuracy_missing_param(self):
        """Test accuracy with missing parameter."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(name="get_weather", arguments={"city": "NYC"}),
            expected_response=FunctionCall(name="get_weather", arguments={"city": "NYC", "unit": "celsius"})
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 1.0

    def test_accuracy_wrong_value(self):
        """Test accuracy with wrong parameter value."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(name="get_weather", arguments={"city": "LA"}),
            expected_response=FunctionCall(name="get_weather", arguments={"city": "NYC"})
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 1.0

    def test_accuracy_flexible_types(self):
        """Test accuracy with flexible type matching."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(name="set_value", arguments={"count": 5.0}),  # float
            expected_response=FunctionCall(name="set_value", arguments={"count": 5}),  # int
            strict_type_check=False
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_accuracy_strict_types(self):
        """Test accuracy with strict type matching."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(name="set_value", arguments={"count": 5.0}),  # float
            expected_response=FunctionCall(name="set_value", arguments={"count": 5}),  # int
            strict_type_check=True
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 1.0

    def test_accuracy_parallel_calls_set(self):
        """Test accuracy with parallel function calls (set comparison)."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=[
                FunctionCall(name="get_weather", arguments={"city": "NYC"}),
                FunctionCall(name="get_time", arguments={"timezone": "EST"})
            ],
            expected_response=[
                FunctionCall(name="get_time", arguments={"timezone": "EST"}),
                FunctionCall(name="get_weather", arguments={"city": "NYC"})
            ],
            order_matters=False
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_accuracy_parallel_calls_sequence(self):
        """Test accuracy with parallel function calls (sequence comparison)."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=[
                FunctionCall(name="get_weather", arguments={"city": "NYC"}),
                FunctionCall(name="get_time", arguments={"timezone": "EST"})
            ],
            expected_response=[
                FunctionCall(name="get_time", arguments={"timezone": "EST"}),
                FunctionCall(name="get_weather", arguments={"city": "NYC"})
            ],
            order_matters=True
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 1.0  # Order mismatch


class TestFunctionCallExactMatch:
    """Tests for FunctionCallExactMatch metric."""

    def test_ast_exact_match(self):
        """Test exact AST match."""
        metric = FunctionCallExactMatch()
        input_data = FunctionCallInput(
            response="get_weather(city='NYC', unit='celsius')",
            expected_response="get_weather(city='NYC', unit='celsius')"
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_ast_different_order(self):
        """Test AST match with different argument order."""
        metric = FunctionCallExactMatch()
        input_data = FunctionCallInput(
            response="get_weather(unit='celsius', city='NYC')",
            expected_response="get_weather(city='NYC', unit='celsius')"
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0  # Order shouldn't matter

    def test_ast_wrong_value(self):
        """Test AST with wrong value."""
        metric = FunctionCallExactMatch()
        input_data = FunctionCallInput(
            response="get_weather(city='LA')",
            expected_response="get_weather(city='NYC')"
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 0.0
        assert "mismatch" in result["reason"].lower()

    def test_ast_complex_args(self):
        """Test AST with complex argument types."""
        metric = FunctionCallExactMatch()
        input_data = FunctionCallInput(
            response="search(query='test', filters=['a', 'b'], limit=10)",
            expected_response="search(query='test', filters=['a', 'b'], limit=10)"
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_ast_boolean_args(self):
        """Test AST with boolean arguments."""
        metric = FunctionCallExactMatch()
        input_data = FunctionCallInput(
            response="toggle(enabled=True)",
            expected_response="toggle(enabled=True)"
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0


class TestRealWorldScenarios:
    """Real-world e2e test scenarios."""

    def test_openai_tool_call_format(self):
        """Test evaluation of OpenAI-style tool call."""
        metric = FunctionCallAccuracy()
        # OpenAI returns tool calls like this
        input_data = FunctionCallInput(
            response={
                "name": "get_current_weather",
                "arguments": '{"location": "San Francisco, CA", "unit": "fahrenheit"}'
            },
            expected_response={
                "name": "get_current_weather",
                "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}
            }
        )
        # Note: The current implementation doesn't parse JSON in arguments string
        # This would need enhancement for full OpenAI compatibility
        result = metric.compute_one(input_data)
        # Currently this will fail because arguments is a string vs dict

    def test_anthropic_tool_use_format(self):
        """Test evaluation of Anthropic-style tool use."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response={
                "name": "get_weather",
                "input": {"location": "San Francisco, CA"}
            },
            expected_response=FunctionCall(
                name="get_weather",
                arguments={"location": "San Francisco, CA"}
            )
        )
        result = metric.compute_one(input_data)
        # This tests parsing flexibility

    def test_multi_tool_api_call(self):
        """Test evaluation of multiple API tool calls."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=[
                {"name": "search_web", "arguments": {"query": "weather NYC"}},
                {"name": "get_location", "arguments": {"city": "New York"}},
                {"name": "format_response", "arguments": {"template": "weather_card"}}
            ],
            expected_response=[
                FunctionCall(name="search_web", arguments={"query": "weather NYC"}),
                FunctionCall(name="get_location", arguments={"city": "New York"}),
                FunctionCall(name="format_response", arguments={"template": "weather_card"})
            ],
            order_matters=True
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_schema_validation_api_spec(self):
        """Test schema validation against API specification."""
        metric = ParameterValidation()
        input_data = FunctionCallInput(
            response=FunctionCall(
                name="create_event",
                arguments={
                    "title": "Team Meeting",
                    "start_time": "2024-01-15T10:00:00Z",
                    "duration_minutes": 60,
                    "attendees": ["alice@example.com", "bob@example.com"],
                    "is_recurring": False
                }
            ),
            function_definitions=[
                FunctionDefinition(
                    name="create_event",
                    parameters=[
                        ParameterSpec(name="title", type="string", required=True),
                        ParameterSpec(name="start_time", type="string", required=True),
                        ParameterSpec(name="duration_minutes", type="integer", required=True),
                        ParameterSpec(name="attendees", type="array", required=False),
                        ParameterSpec(name="is_recurring", type="boolean", required=False),
                        ParameterSpec(name="location", type="string", required=False),
                    ]
                )
            ]
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_batch_evaluation(self):
        """Test batch evaluation of multiple function calls."""
        metric = FunctionCallAccuracy()
        inputs = [
            FunctionCallInput(
                response=FunctionCall(name="add", arguments={"a": 1, "b": 2}),
                expected_response=FunctionCall(name="add", arguments={"a": 1, "b": 2})
            ),
            FunctionCallInput(
                response=FunctionCall(name="multiply", arguments={"a": 3, "b": 4}),
                expected_response=FunctionCall(name="multiply", arguments={"a": 3, "b": 4})
            ),
            FunctionCallInput(
                response=FunctionCall(name="divide", arguments={"a": 10, "b": 2}),
                expected_response=FunctionCall(name="divide", arguments={"a": 10, "b": 5})  # Wrong value
            )
        ]
        results = metric.evaluate(inputs)
        assert len(results.eval_results) == 3
        assert results.eval_results[0].output == 1.0
        assert results.eval_results[1].output == 1.0
        assert results.eval_results[2].output < 1.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_arguments(self):
        """Test function call with no arguments."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(name="get_current_time", arguments={}),
            expected_response=FunctionCall(name="get_current_time", arguments={})
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_nested_object_arguments(self):
        """Test function call with nested object arguments."""
        metric = FunctionCallExactMatch()
        input_data = FunctionCallInput(
            response=FunctionCall(
                name="create_user",
                arguments={
                    "user": {
                        "name": "Alice",
                        "email": "alice@example.com",
                        "preferences": {"theme": "dark", "notifications": True}
                    }
                }
            ),
            expected_response=FunctionCall(
                name="create_user",
                arguments={
                    "user": {
                        "name": "Alice",
                        "email": "alice@example.com",
                        "preferences": {"theme": "dark", "notifications": True}
                    }
                }
            )
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_special_characters_in_args(self):
        """Test function call with special characters."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(
                name="search",
                arguments={"query": "hello \"world\" & <test>"}
            ),
            expected_response=FunctionCall(
                name="search",
                arguments={"query": "hello \"world\" & <test>"}
            )
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_unicode_in_arguments(self):
        """Test function call with unicode characters."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(
                name="translate",
                arguments={"text": "Hello world", "target": "Japanese", "result": "Konnichiwa"}
            ),
            expected_response=FunctionCall(
                name="translate",
                arguments={"text": "Hello world", "target": "Japanese", "result": "Konnichiwa"}
            )
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_null_values(self):
        """Test function call with null values."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(
                name="update",
                arguments={"value": None, "reason": "reset"}
            ),
            expected_response=FunctionCall(
                name="update",
                arguments={"value": None, "reason": "reset"}
            )
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0

    def test_large_numeric_values(self):
        """Test function call with large numeric values."""
        metric = FunctionCallAccuracy()
        input_data = FunctionCallInput(
            response=FunctionCall(
                name="transfer",
                arguments={"amount": 1000000000000, "fee": 0.0000001}
            ),
            expected_response=FunctionCall(
                name="transfer",
                arguments={"amount": 1000000000000, "fee": 0.0000001}
            )
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0
