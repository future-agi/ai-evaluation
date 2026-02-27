"""
Function Calling Evaluation Metrics.

AST-based, deterministic evaluation of LLM function calling.
Provides sub-10ms evaluation latency without LLM-as-judge dependency.
"""

import ast
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..base_metric import BaseMetric
from .types import FunctionCallInput, FunctionCall, FunctionDefinition, ParameterSpec


def _parse_function_call(call: Union[FunctionCall, Dict, str, None]) -> Optional[FunctionCall]:
    """Parse various input formats into a FunctionCall object."""
    if call is None:
        return None
    if isinstance(call, FunctionCall):
        return call
    if isinstance(call, dict):
        # Get arguments, handling both dict and JSON string formats (OpenAI-style)
        arguments = call.get("arguments", call.get("parameters", call.get("input", {})))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        return FunctionCall(
            name=call.get("name", call.get("function", "")),
            arguments=arguments if isinstance(arguments, dict) else {}
        )
    if isinstance(call, str):
        try:
            parsed = json.loads(call)
            return _parse_function_call(parsed)
        except json.JSONDecodeError:
            # Try to parse as Python AST (e.g., "get_weather(city='NYC')")
            return _parse_ast_call(call)
    return None


def _parse_ast_call(call_str: str) -> Optional[FunctionCall]:
    """Parse a function call string using AST."""
    try:
        # Wrap in expression for parsing
        tree = ast.parse(call_str.strip(), mode='eval')
        if isinstance(tree.body, ast.Call):
            call_node = tree.body
            func_name = ""
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id
            elif isinstance(call_node.func, ast.Attribute):
                func_name = call_node.func.attr

            # Extract arguments
            arguments = {}

            # Handle positional arguments
            for i, arg in enumerate(call_node.args):
                arguments[f"__positional_{i}"] = _ast_to_value(arg)

            # Handle keyword arguments
            for kw in call_node.keywords:
                if kw.arg:
                    arguments[kw.arg] = _ast_to_value(kw.value)

            return FunctionCall(name=func_name, arguments=arguments)
    except (SyntaxError, ValueError):
        pass
    return None


def _ast_to_value(node: ast.expr) -> Any:
    """Convert an AST node to a Python value."""
    # ast.Constant handles strings, numbers, booleans, None in Python 3.8+
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_ast_to_value(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _ast_to_value(k): _ast_to_value(v)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }
    if isinstance(node, ast.Name):
        # Handle True, False, None as names (fallback)
        if node.id == "True":
            return True
        elif node.id == "False":
            return False
        elif node.id == "None":
            return None
        return node.id
    if isinstance(node, ast.Tuple):
        return tuple(_ast_to_value(elt) for elt in node.elts)
    if isinstance(node, ast.Set):
        return {_ast_to_value(elt) for elt in node.elts}
    return str(node)


def _parse_function_calls(
    calls: Union[FunctionCall, List[FunctionCall], Dict, str, List[Dict], None]
) -> List[FunctionCall]:
    """Parse various input formats into a list of FunctionCall objects."""
    if calls is None:
        return []
    if isinstance(calls, list):
        return [_parse_function_call(c) for c in calls if _parse_function_call(c)]
    parsed = _parse_function_call(calls)
    return [parsed] if parsed else []


def _types_compatible(actual: Any, expected: Any, strict: bool = False) -> bool:
    """Check if types are compatible."""
    if strict:
        return type(actual) == type(expected)

    # Flexible type checking
    if actual is None or expected is None:
        return actual == expected

    # Bool guard: bool is a subclass of int in Python, so must check before numeric
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool)

    # Numeric compatibility (int/float)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return True

    # String compatibility
    if isinstance(actual, str) and isinstance(expected, str):
        return True

    # List compatibility
    if isinstance(actual, list) and isinstance(expected, list):
        return True

    # Dict compatibility
    if isinstance(actual, dict) and isinstance(expected, dict):
        return True

    return type(actual) == type(expected)


def _values_equal(actual: Any, expected: Any, strict_type: bool = False) -> bool:
    """Check if values are equal with optional type flexibility."""
    if not _types_compatible(actual, expected, strict_type):
        return False

    # For numeric types, allow float/int comparison
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return float(actual) == float(expected)

    # For strings, exact match (case-sensitive)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual == expected

    # For lists, recursive comparison
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        return all(_values_equal(a, e, strict_type) for a, e in zip(actual, expected))

    # For dicts, recursive comparison
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(
            _values_equal(actual[k], expected[k], strict_type)
            for k in expected.keys()
        )

    return actual == expected


class FunctionNameMatch(BaseMetric[FunctionCallInput]):
    """
    Evaluates if the function name matches the expected name.

    Returns 1.0 if names match, 0.0 otherwise.
    Fast, deterministic metric (~1ms).
    """

    @property
    def metric_name(self) -> str:
        return "function_name_match"

    def compute_one(self, inputs: FunctionCallInput) -> Dict[str, Any]:
        actual = _parse_function_call(inputs.response)
        expected = _parse_function_call(inputs.expected_response)

        if actual is None:
            return {
                "output": 0.0,
                "reason": "Could not parse actual function call from response."
            }

        if expected is None:
            return {
                "output": 0.0,
                "reason": "Could not parse expected function call. 'expected_response' is required."
            }

        if actual.name == expected.name:
            return {
                "output": 1.0,
                "reason": f"Function name '{actual.name}' matches expected."
            }

        return {
            "output": 0.0,
            "reason": f"Function name mismatch: got '{actual.name}', expected '{expected.name}'."
        }


class ParameterValidation(BaseMetric[FunctionCallInput]):
    """
    Validates function call parameters against a schema.

    Checks:
    - Required parameters are present
    - Parameter types match specification
    - Enum constraints are satisfied

    Returns score from 0.0 to 1.0 based on validation success.
    """

    @property
    def metric_name(self) -> str:
        return "parameter_validation"

    def compute_one(self, inputs: FunctionCallInput) -> Dict[str, Any]:
        actual = _parse_function_call(inputs.response)

        if actual is None:
            return {
                "output": 0.0,
                "reason": "Could not parse function call from response."
            }

        if not inputs.function_definitions:
            return {
                "output": 0.0,
                "reason": "No function definitions provided for validation."
            }

        # Find the matching function definition
        func_def = None
        for fd in inputs.function_definitions:
            if fd.name == actual.name:
                func_def = fd
                break

        if func_def is None:
            return {
                "output": 0.0,
                "reason": f"Function '{actual.name}' not found in definitions."
            }

        errors = []
        total_checks = 0
        passed_checks = 0

        for param_spec in func_def.parameters:
            total_checks += 1

            # Check required parameters
            if param_spec.required and param_spec.name not in actual.arguments:
                errors.append(f"Missing required parameter: {param_spec.name}")
                continue

            if param_spec.name in actual.arguments:
                value = actual.arguments[param_spec.name]

                # Type checking
                if not self._check_type(value, param_spec.type, inputs.strict_type_check):
                    errors.append(
                        f"Parameter '{param_spec.name}' has wrong type: "
                        f"expected {param_spec.type}, got {type(value).__name__}"
                    )
                    continue

                # Enum checking
                if param_spec.enum and value not in param_spec.enum:
                    errors.append(
                        f"Parameter '{param_spec.name}' value '{value}' not in allowed values: {param_spec.enum}"
                    )
                    continue

                passed_checks += 1
            else:
                # Optional parameter not provided - that's fine
                passed_checks += 1

        # Check for extra parameters
        if not inputs.ignore_extra_params:
            expected_params = {p.name for p in func_def.parameters}
            extra_params = set(actual.arguments.keys()) - expected_params
            if extra_params:
                total_checks += len(extra_params)
                errors.append(f"Unexpected parameters: {', '.join(extra_params)}")

        if total_checks == 0:
            return {"output": 1.0, "reason": "No parameters to validate."}

        score = passed_checks / total_checks if total_checks > 0 else 1.0

        if errors:
            return {
                "output": round(score, 4),
                "reason": "; ".join(errors)
            }

        return {
            "output": 1.0,
            "reason": f"All {total_checks} parameter checks passed."
        }

    def _check_type(self, value: Any, expected_type: str, strict: bool) -> bool:
        """Check if a value matches the expected type."""
        type_map = {
            "string": (str,),
            "integer": (int,) if strict else (int, float),
            "number": (int, float),
            "boolean": (bool,),
            "array": (list,),
            "object": (dict,),
            "null": (type(None),),
        }

        expected_types = type_map.get(expected_type.lower(), (str,))

        # Special case: strict integer check
        if expected_type.lower() == "integer" and strict:
            return isinstance(value, int) and not isinstance(value, bool)

        # Boolean should not match int in Python
        if isinstance(value, bool) and expected_type.lower() != "boolean":
            return False

        return isinstance(value, expected_types)


class FunctionCallAccuracy(BaseMetric[FunctionCallInput]):
    """
    Comprehensive function call accuracy evaluation.

    Evaluates:
    - Function name match (weighted 40%)
    - Parameter presence (weighted 30%)
    - Parameter value accuracy (weighted 30%)

    Returns overall score from 0.0 to 1.0.
    """

    @property
    def metric_name(self) -> str:
        return "function_call_accuracy"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name_weight = self.config.get("name_weight", 0.4)
        self.presence_weight = self.config.get("presence_weight", 0.3)
        self.value_weight = self.config.get("value_weight", 0.3)

    def compute_one(self, inputs: FunctionCallInput) -> Dict[str, Any]:
        actual_calls = _parse_function_calls(inputs.response)
        expected_calls = _parse_function_calls(inputs.expected_response)

        if not actual_calls:
            return {
                "output": 0.0,
                "reason": "Could not parse any function calls from response."
            }

        if not expected_calls:
            return {
                "output": 0.0,
                "reason": "No expected function calls provided."
            }

        # Handle single vs multiple calls
        if len(expected_calls) == 1 and len(actual_calls) == 1:
            return self._evaluate_single(
                actual_calls[0], expected_calls[0], inputs
            )

        # Multiple calls - evaluate as set or sequence
        return self._evaluate_multiple(
            actual_calls, expected_calls, inputs
        )

    def _evaluate_single(
        self,
        actual: FunctionCall,
        expected: FunctionCall,
        inputs: FunctionCallInput
    ) -> Dict[str, Any]:
        """Evaluate a single function call pair."""
        details = []

        # Name match (40%)
        name_score = 1.0 if actual.name == expected.name else 0.0
        details.append(f"name: {name_score:.0%}")

        # Parameter presence (30%)
        expected_params = set(expected.arguments.keys())
        actual_params = set(actual.arguments.keys())

        if expected_params:
            presence_score = len(expected_params & actual_params) / len(expected_params)
        else:
            presence_score = 1.0 if not actual_params or inputs.ignore_extra_params else 0.0

        details.append(f"params: {presence_score:.0%}")

        # Parameter values (30%)
        value_matches = 0
        value_total = len(expected.arguments)

        for param_name, expected_value in expected.arguments.items():
            if param_name in actual.arguments:
                actual_value = actual.arguments[param_name]
                if _values_equal(actual_value, expected_value, inputs.strict_type_check):
                    value_matches += 1

        value_score = value_matches / value_total if value_total > 0 else 1.0
        details.append(f"values: {value_score:.0%}")

        # Calculate weighted score
        total_score = (
            name_score * self.name_weight +
            presence_score * self.presence_weight +
            value_score * self.value_weight
        )

        return {
            "output": round(total_score, 4),
            "reason": f"Function call evaluation: {', '.join(details)}. Overall: {total_score:.1%}"
        }

    def _evaluate_multiple(
        self,
        actual_calls: List[FunctionCall],
        expected_calls: List[FunctionCall],
        inputs: FunctionCallInput
    ) -> Dict[str, Any]:
        """Evaluate multiple function calls (parallel calling)."""
        if inputs.order_matters:
            # Sequence comparison
            if len(actual_calls) != len(expected_calls):
                return {
                    "output": 0.0,
                    "reason": f"Call count mismatch: got {len(actual_calls)}, expected {len(expected_calls)}"
                }

            scores = []
            for actual, expected in zip(actual_calls, expected_calls):
                result = self._evaluate_single(actual, expected, inputs)
                scores.append(result["output"])

            avg_score = sum(scores) / len(scores)
            return {
                "output": round(avg_score, 4),
                "reason": f"Sequence evaluation: {len(scores)} calls, avg score {avg_score:.1%}"
            }

        # Set comparison - find best match for each expected call
        matched_scores = []
        unmatched_expected = []

        for expected in expected_calls:
            best_score = 0.0
            for actual in actual_calls:
                result = self._evaluate_single(actual, expected, inputs)
                best_score = max(best_score, result["output"])

            if best_score > 0:
                matched_scores.append(best_score)
            else:
                unmatched_expected.append(expected.name)

        if not matched_scores:
            return {
                "output": 0.0,
                "reason": f"No expected calls matched. Expected: {[c.name for c in expected_calls]}"
            }

        # Penalize for missing calls
        coverage = len(matched_scores) / len(expected_calls)
        avg_match_score = sum(matched_scores) / len(matched_scores)
        final_score = coverage * avg_match_score

        reason = f"Matched {len(matched_scores)}/{len(expected_calls)} calls, avg accuracy {avg_match_score:.1%}"
        if unmatched_expected:
            reason += f". Missing: {unmatched_expected}"

        return {
            "output": round(final_score, 4),
            "reason": reason
        }


class FunctionCallExactMatch(BaseMetric[FunctionCallInput]):
    """
    AST-based exact match evaluation.

    Parses function calls as AST and compares structure.
    Useful for evaluating code-style function calls.

    Returns 1.0 for exact match, 0.0 otherwise.
    """

    @property
    def metric_name(self) -> str:
        return "function_call_exact_match"

    def compute_one(self, inputs: FunctionCallInput) -> Dict[str, Any]:
        actual = _parse_function_call(inputs.response)
        expected = _parse_function_call(inputs.expected_response)

        if actual is None:
            return {
                "output": 0.0,
                "reason": "Could not parse actual function call."
            }

        if expected is None:
            return {
                "output": 0.0,
                "reason": "Could not parse expected function call."
            }

        # Compare name
        if actual.name != expected.name:
            return {
                "output": 0.0,
                "reason": f"Function name mismatch: '{actual.name}' vs '{expected.name}'"
            }

        # Compare arguments
        if set(actual.arguments.keys()) != set(expected.arguments.keys()):
            missing = set(expected.arguments.keys()) - set(actual.arguments.keys())
            extra = set(actual.arguments.keys()) - set(expected.arguments.keys())
            parts = []
            if missing:
                parts.append(f"missing: {missing}")
            if extra and not inputs.ignore_extra_params:
                parts.append(f"extra: {extra}")
            if parts:
                return {
                    "output": 0.0,
                    "reason": f"Parameter mismatch: {', '.join(parts)}"
                }

        # Compare values
        for key, expected_value in expected.arguments.items():
            if key not in actual.arguments:
                continue
            actual_value = actual.arguments[key]
            if not _values_equal(actual_value, expected_value, inputs.strict_type_check):
                return {
                    "output": 0.0,
                    "reason": f"Value mismatch for '{key}': got {actual_value!r}, expected {expected_value!r}"
                }

        return {
            "output": 1.0,
            "reason": f"Function call matches: {actual.name}({', '.join(f'{k}={v!r}' for k, v in actual.arguments.items())})"
        }
