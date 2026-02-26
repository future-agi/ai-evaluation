"""Example: Phase 1 Metrics — Function Calling, Agents, Hallucination
Run: poetry run python examples/09_phase1_metrics.py
Requires: No API keys (all local)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fi.evals import evaluate

RESULTS = []


def log(name, passed, details=""):
    RESULTS.append({"test": name, "passed": passed})
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# =========================================================================
# 1. Function Calling Metrics
# =========================================================================

section("FUNCTION CALLING METRICS")

# --- function_name_match ---
print("\n-- function_name_match --")
r = evaluate(
    "function_name_match",
    response={"name": "get_weather", "arguments": {"city": "NYC"}},
    expected_response={"name": "get_weather", "arguments": {"city": "NYC"}},
)
log("name match (exact)", r.score == 1.0, f"score={r.score}")

r = evaluate(
    "function_name_match",
    response={"name": "get_temperature", "arguments": {}},
    expected_response={"name": "get_weather", "arguments": {}},
)
log("name match (mismatch)", r.score == 0.0, f"score={r.score}")

# --- parameter_validation ---
print("\n-- parameter_validation --")
r = evaluate(
    "parameter_validation",
    response={"name": "book_flight", "arguments": {
        "origin": "NYC", "destination": "LAX", "date": "2024-01-15"
    }},
    function_definitions=[{
        "name": "book_flight",
        "parameters": [
            {"name": "origin", "type": "string", "required": True},
            {"name": "destination", "type": "string", "required": True},
            {"name": "date", "type": "string", "required": True},
        ],
    }],
)
log("param validation (all present)", r.score == 1.0, f"score={r.score}")

r = evaluate(
    "parameter_validation",
    response={"name": "book_flight", "arguments": {"origin": "NYC"}},
    function_definitions=[{
        "name": "book_flight",
        "parameters": [
            {"name": "origin", "type": "string", "required": True},
            {"name": "destination", "type": "string", "required": True},
        ],
    }],
)
log("param validation (missing required)", r.score < 1.0, f"score={r.score}")

# --- function_call_accuracy ---
print("\n-- function_call_accuracy --")
r = evaluate(
    "function_call_accuracy",
    response={"name": "get_weather", "arguments": {"city": "NYC", "unit": "celsius"}},
    expected_response={"name": "get_weather", "arguments": {"city": "NYC", "unit": "celsius"}},
)
log("accuracy (perfect)", r.score == 1.0, f"score={r.score}")

r = evaluate(
    "function_call_accuracy",
    response={"name": "get_weather", "arguments": {"city": "LA"}},
    expected_response={"name": "get_weather", "arguments": {"city": "NYC", "unit": "celsius"}},
)
log("accuracy (wrong value + missing param)", r.score < 1.0, f"score={r.score}")

# --- function_call_exact_match ---
print("\n-- function_call_exact_match --")
r = evaluate(
    "function_call_exact_match",
    response={"name": "search", "arguments": {"q": "hello", "limit": 10}},
    expected_response={"name": "search", "arguments": {"q": "hello", "limit": 10}},
)
log("exact match (identical)", r.score == 1.0, f"score={r.score}")

r = evaluate(
    "function_call_exact_match",
    response={"name": "search", "arguments": {"q": "hello", "limit": 5}},
    expected_response={"name": "search", "arguments": {"q": "hello", "limit": 10}},
)
log("exact match (value differs)", r.score == 0.0, f"score={r.score}")

# --- bool guard fix demo ---
print("\n-- bool guard fix (True != 1) --")
r = evaluate(
    "function_call_exact_match",
    response={"name": "toggle", "arguments": {"enabled": True}},
    expected_response={"name": "toggle", "arguments": {"enabled": 1}},
)
log("bool vs int (should fail)", r.score == 0.0, f"score={r.score}")


# =========================================================================
# 2. Agent Evaluation Metrics
# =========================================================================

section("AGENT EVALUATION METRICS")

# Build a sample trajectory
SAMPLE_TRAJECTORY = [
    {
        "step_number": 1,
        "thought": "I need to search for weather information in Paris",
        "action": "search",
        "tool_calls": [
            {"name": "web_search", "arguments": {"query": "weather Paris"}, "success": True,
             "result": "Current weather in Paris: 18°C, partly cloudy"}
        ],
        "observation": "Found weather data for Paris",
    },
    {
        "step_number": 2,
        "thought": "Now I should format the response for the user",
        "action": "format",
        "tool_calls": [
            {"name": "format_response", "arguments": {"template": "weather"}, "success": True,
             "result": "Paris: 18°C, partly cloudy"}
        ],
        "observation": "Response formatted",
        "is_final": True,
    },
]

TASK = {
    "description": "Find the current weather in Paris and return a formatted response",
    "expected_outcome": "Weather information for Paris",
    "required_tools": ["web_search", "format_response"],
    "max_steps": 5,
    "success_criteria": ["weather information found", "response formatted"],
}

# --- task_completion ---
print("\n-- task_completion --")
r = evaluate(
    "task_completion",
    trajectory=SAMPLE_TRAJECTORY,
    task=TASK,
    final_result="Paris: 18°C, partly cloudy",
    expected_result="Current weather in Paris",
)
log("task completion", r.score >= 0.5, f"score={r.score}")

# --- step_efficiency ---
print("\n-- step_efficiency --")
r = evaluate(
    "step_efficiency",
    trajectory=SAMPLE_TRAJECTORY,
    task=TASK,
)
log("step efficiency (2 steps, max 5)", r.score >= 0.7, f"score={r.score}")

# --- tool_selection_accuracy ---
print("\n-- tool_selection_accuracy --")
r = evaluate(
    "tool_selection_accuracy",
    trajectory=SAMPLE_TRAJECTORY,
    task=TASK,
    available_tools=["web_search", "format_response", "calculator"],
)
log("tool selection", r.score >= 0.7, f"score={r.score}")

# --- trajectory_score ---
print("\n-- trajectory_score (composite) --")
r = evaluate(
    "trajectory_score",
    trajectory=SAMPLE_TRAJECTORY,
    task=TASK,
    final_result="Paris: 18°C, partly cloudy",
    expected_result="Current weather in Paris",
    available_tools=["web_search", "format_response"],
)
log("trajectory score", r.score >= 0.5, f"score={r.score}")

# --- goal_progress ---
print("\n-- goal_progress --")
r = evaluate(
    "goal_progress",
    trajectory=SAMPLE_TRAJECTORY,
    task=TASK,
)
log("goal progress (cumulative)", r.score > 0.0, f"score={r.score}")

# --- action_safety ---
print("\n-- action_safety --")
r = evaluate(
    "action_safety",
    trajectory=SAMPLE_TRAJECTORY,
    task=TASK,
)
log("action safety (clean trajectory)", r.score == 1.0, f"score={r.score}")

# Test with dangerous action
UNSAFE_TRAJECTORY = [
    {
        "step_number": 1,
        "thought": "Let me clean up",
        "tool_calls": [
            {"name": "shell", "arguments": {"cmd": "rm -rf /"}, "success": True}
        ],
    }
]
r = evaluate(
    "action_safety",
    trajectory=UNSAFE_TRAJECTORY,
    task={"description": "clean up files"},
)
log("action safety (dangerous cmd)", r.score < 1.0, f"score={r.score}")

# --- reasoning_quality ---
print("\n-- reasoning_quality --")
r = evaluate(
    "reasoning_quality",
    trajectory=SAMPLE_TRAJECTORY,
    task=TASK,
)
log("reasoning quality", r.score > 0.0, f"score={r.score}")

NO_THOUGHT_TRAJECTORY = [
    {
        "step_number": 1,
        "tool_calls": [
            {"name": "search", "arguments": {"q": "test"}, "success": True}
        ],
    }
]
r = evaluate(
    "reasoning_quality",
    trajectory=NO_THOUGHT_TRAJECTORY,
    task={"description": "test"},
)
log("reasoning quality (no thoughts)", r.score <= 0.5, f"score={r.score}")


# =========================================================================
# 3. Hallucination Detection Metrics
# =========================================================================

section("HALLUCINATION DETECTION METRICS")

CONTEXT = (
    "Paris is the capital of France. The Eiffel Tower was built in 1889 "
    "by Gustave Eiffel. It is 330 meters tall and located on the Champ de Mars."
)

# --- faithfulness ---
print("\n-- faithfulness --")
r = evaluate(
    "faithfulness",
    output="Paris is the capital of France. The Eiffel Tower was built in 1889.",
    context=CONTEXT,
)
log("faithfulness (faithful)", r.score >= 0.7, f"score={r.score}")

r = evaluate(
    "faithfulness",
    output="Paris is the capital of Spain. The Eiffel Tower was built in 2005.",
    context=CONTEXT,
)
log("faithfulness (hallucinated)", r.score < 0.7, f"score={r.score}")

# --- claim_support ---
print("\n-- claim_support --")
r = evaluate(
    "claim_support",
    output="The Eiffel Tower is in Paris. It was built by Gustave Eiffel.",
    context=CONTEXT,
)
log("claim support (high)", r.score >= 0.3, f"score={r.score}")

# --- factual_consistency ---
print("\n-- factual_consistency --")
r = evaluate(
    "factual_consistency",
    output="The tower is 330 meters tall. It is located on the Champ de Mars.",
    reference=CONTEXT,
)
log("factual consistency", r.score >= 0.5, f"score={r.score}")

# --- contradiction_detection ---
print("\n-- contradiction_detection --")
r = evaluate(
    "contradiction_detection",
    output="The Eiffel Tower is in Paris. It was designed by Gustave Eiffel.",
    context=CONTEXT,
)
log("contradiction (none)", r.score == 1.0, f"score={r.score}")

r = evaluate(
    "contradiction_detection",
    output="The Eiffel Tower was not built by Gustave Eiffel.",
    context=CONTEXT,
)
log("contradiction (negation)", r.score == 0.0, f"score={r.score}")

# --- hallucination_score ---
print("\n-- hallucination_score (composite) --")
r = evaluate(
    "hallucination_score",
    output="Paris is the capital of France. The Eiffel Tower is 330 meters tall.",
    context=CONTEXT,
)
log("hallucination score (low risk)", r.score >= 0.6, f"score={r.score}")

r = evaluate(
    "hallucination_score",
    output="Unicorns live in the Eiffel Tower. Dragons fly over Paris.",
    context=CONTEXT,
)
log("hallucination score (high risk)", r.score < 0.7, f"score={r.score}")


# =========================================================================
# Summary
# =========================================================================

passed = sum(1 for r in RESULTS if r["passed"])
total = len(RESULTS)
section(f"SUMMARY: {passed}/{total} passed")

if passed < total:
    for r in RESULTS:
        if not r["passed"]:
            print(f"  \033[91mFAILED\033[0m: {r['test']}")
    sys.exit(1)
else:
    print("  All tests passed!")
