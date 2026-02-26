"""
Travel Agent Evaluation — a realistic demo of Phase 1 metrics.

Scenario: A customer asks an AI travel agent to find and book a round-trip
flight from San Francisco to Tokyo. The agent reasons through the problem,
calls tools (search_flights, check_availability, book_flight), and produces
a final itinerary. We then evaluate:

  1. Were the tool calls correct?         (function calling metrics)
  2. Was the agent's trajectory efficient? (agent metrics)
  3. Is the final answer grounded?         (hallucination metrics)
  4. Does an LLM judge agree?              (Gemini augmented eval)

Run:
    export GOOGLE_API_KEY=...        # for the LLM-judge section
    poetry run python examples/09_phase1_metrics.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fi.evals import evaluate

HAS_GEMINI = bool(os.environ.get("GOOGLE_API_KEY"))


def heading(text):
    print(f"\n{'─' * 64}")
    print(f"  {text}")
    print(f"{'─' * 64}")


def show(label, result):
    print(f"  {label:40s}  score={result.score:.2f}  | {result.reason[:80]}")


# =========================================================================
# The scenario data
# =========================================================================

# What the agent was asked
CUSTOMER_QUERY = "Find me a round-trip flight from San Francisco to Tokyo for Dec 20-27."

# What tools are available to the agent
AVAILABLE_TOOLS = ["search_flights", "check_availability", "book_flight", "send_email"]

# The tool-call API schema (for parameter validation)
TOOL_SCHEMAS = [
    {
        "name": "search_flights",
        "parameters": [
            {"name": "origin", "type": "string", "required": True},
            {"name": "destination", "type": "string", "required": True},
            {"name": "departure_date", "type": "string", "required": True},
            {"name": "return_date", "type": "string", "required": False},
            {"name": "cabin_class", "type": "string", "required": False,
             "enum": ["economy", "business", "first"]},
        ],
    },
    {
        "name": "book_flight",
        "parameters": [
            {"name": "flight_id", "type": "string", "required": True},
            {"name": "passenger_name", "type": "string", "required": True},
            {"name": "payment_method", "type": "string", "required": True},
        ],
    },
]

# The golden tool call the agent SHOULD have made
EXPECTED_SEARCH_CALL = {
    "name": "search_flights",
    "arguments": {
        "origin": "SFO",
        "destination": "TYO",
        "departure_date": "2024-12-20",
        "return_date": "2024-12-27",
    },
}

# What the agent ACTUALLY produced (slightly wrong: used city name instead of code)
ACTUAL_SEARCH_CALL = {
    "name": "search_flights",
    "arguments": {
        "origin": "San Francisco",
        "destination": "Tokyo",
        "departure_date": "2024-12-20",
        "return_date": "2024-12-27",
    },
}

ACTUAL_BOOK_CALL = {
    "name": "book_flight",
    "arguments": {
        "flight_id": "JAL-5012",
        "passenger_name": "Alice Chen",
        "payment_method": "credit_card",
    },
}

# The full agent trajectory (like a ReAct trace)
AGENT_TRAJECTORY = [
    {
        "step_number": 1,
        "thought": (
            "The customer wants a round-trip SFO → Tokyo for Dec 20–27. "
            "I should search for available flights first."
        ),
        "action": "search_flights",
        "tool_calls": [
            {
                "name": "search_flights",
                "arguments": {
                    "origin": "San Francisco",
                    "destination": "Tokyo",
                    "departure_date": "2024-12-20",
                    "return_date": "2024-12-27",
                },
                "success": True,
                "result": (
                    "Found 3 flights: "
                    "JAL-5012 (SFO→NRT, $1,245 economy, 11h20m), "
                    "UA-837 (SFO→HND, $1,380 economy, 11h05m), "
                    "ANA-107 (SFO→NRT, $1,510 business, 10h45m)"
                ),
            }
        ],
        "observation": "Got 3 flight options with prices and durations.",
    },
    {
        "step_number": 2,
        "thought": (
            "JAL-5012 is the cheapest at $1,245. I should check if it's still "
            "available before booking, since it's the holiday season."
        ),
        "action": "check_availability",
        "tool_calls": [
            {
                "name": "check_availability",
                "arguments": {"flight_id": "JAL-5012", "date": "2024-12-20"},
                "success": True,
                "result": "JAL-5012: 14 seats remaining in economy.",
            }
        ],
        "observation": "Flight is available. 14 seats left.",
    },
    {
        "step_number": 3,
        "thought": (
            "The flight is available. I'll go ahead and book it for the customer. "
            "I need flight_id, passenger name, and payment method."
        ),
        "action": "book_flight",
        "tool_calls": [
            {
                "name": "book_flight",
                "arguments": {
                    "flight_id": "JAL-5012",
                    "passenger_name": "Alice Chen",
                    "payment_method": "credit_card",
                },
                "success": True,
                "result": "Booking confirmed. Confirmation #: JAL-CONF-88421",
            }
        ],
        "observation": "Booking confirmed with confirmation number.",
        "is_final": True,
    },
]

TASK_DEFINITION = {
    "description": (
        "Find and book a round-trip flight from San Francisco to Tokyo "
        "for December 20–27, choosing the best value option."
    ),
    "expected_outcome": "A confirmed flight booking with confirmation number",
    "required_tools": ["search_flights", "book_flight"],
    "max_steps": 5,
    "success_criteria": [
        "Searched for flights matching the customer request",
        "Selected a reasonable flight option",
        "Successfully booked the flight",
        "Returned a confirmation number",
    ],
}

# The ground-truth context (what the tools actually returned)
TOOL_CONTEXT = (
    "JAL-5012 is a Japan Airlines flight from SFO to NRT (Narita). "
    "Departure: December 20, 2024. Return: December 27, 2024. "
    "Price: $1,245 for economy class. Duration: 11 hours 20 minutes. "
    "14 seats remaining. Booking confirmation: JAL-CONF-88421."
)

# The agent's final answer to the customer
AGENT_FINAL_ANSWER = (
    "I've booked your round-trip flight on Japan Airlines flight JAL-5012. "
    "You'll be flying from San Francisco SFO to Tokyo Narita NRT. "
    "The departure is on December 20, 2024, and you return on December 27, 2024. "
    "The total price is $1,245 for economy class, and the flight duration is 11 hours 20 minutes each way. "
    "Your booking confirmation number is JAL-CONF-88421."
)

# A bad agent answer that hallucinates details
HALLUCINATED_ANSWER = (
    "I've booked your flight on Japan Airlines JAL-5012 to Tokyo Haneda airport. "  # wrong: NRT not HND
    "The total cost is $980 for economy class. "                                     # wrong: $1,245
    "The flight takes about 9 hours and 15 minutes. "                                # wrong: 11h 20m
    "You also get complimentary lounge access and priority boarding. "                # fabricated
    "Your confirmation number is JAL-CONF-88421."
)


# =========================================================================
# Part 1: Function Calling — Did the agent call tools correctly?
# =========================================================================

heading("PART 1: FUNCTION CALLING")

print("\n  How well did the agent's search_flights call match the expected one?")
r = evaluate(
    "function_call_accuracy",
    response=ACTUAL_SEARCH_CALL,
    expected_response=EXPECTED_SEARCH_CALL,
)
show("search call accuracy", r)

print("\n  Did the agent call the right function?")
r = evaluate("function_name_match", response=ACTUAL_SEARCH_CALL, expected_response=EXPECTED_SEARCH_CALL)
show("function name", r)

print("\n  Do the book_flight params match the schema?")
r = evaluate(
    "parameter_validation",
    response=ACTUAL_BOOK_CALL,
    function_definitions=TOOL_SCHEMAS,
)
show("book_flight param validation", r)

print("\n  Exact match — search call vs golden (strict)?")
r = evaluate(
    "function_call_exact_match",
    response=ACTUAL_SEARCH_CALL,
    expected_response=EXPECTED_SEARCH_CALL,
)
show("exact match (city names vs codes)", r)


# =========================================================================
# Part 2: Agent Trajectory — Was the agent efficient and safe?
# =========================================================================

heading("PART 2: AGENT TRAJECTORY")

shared_kwargs = dict(
    trajectory=AGENT_TRAJECTORY,
    task=TASK_DEFINITION,
    available_tools=AVAILABLE_TOOLS,
    final_result=AGENT_FINAL_ANSWER,
    expected_result="Confirmed booking with JAL-CONF-88421",
)

print("\n  Did the agent complete the task?")
show("task completion", evaluate("task_completion", **shared_kwargs))

print("\n  How efficient was the trajectory? (3 steps, max 5)")
show("step efficiency", evaluate("step_efficiency", **shared_kwargs))

print("\n  Did it pick the right tools?")
show("tool selection accuracy", evaluate("tool_selection_accuracy", **shared_kwargs))

print("\n  Overall trajectory quality:")
show("trajectory score", evaluate("trajectory_score", **shared_kwargs))

print("\n  Was progress toward the goal monotonic?")
show("goal progress", evaluate("goal_progress", **shared_kwargs))

print("\n  Were the actions safe?")
show("action safety", evaluate("action_safety", **shared_kwargs))

print("\n  How well did the agent reason?")
show("reasoning quality", evaluate("reasoning_quality", **shared_kwargs))


# =========================================================================
# Part 3: Hallucination — Is the answer grounded in what the tools returned?
# =========================================================================

heading("PART 3: HALLUCINATION DETECTION")

print("\n  --- Good answer (grounded in tool output) ---")
show("faithfulness",           evaluate("faithfulness", output=AGENT_FINAL_ANSWER, context=TOOL_CONTEXT))
show("claim support",          evaluate("claim_support", output=AGENT_FINAL_ANSWER, context=TOOL_CONTEXT))
show("contradiction detection", evaluate("contradiction_detection", output=AGENT_FINAL_ANSWER, context=TOOL_CONTEXT))
show("hallucination score",    evaluate("hallucination_score", output=AGENT_FINAL_ANSWER, context=TOOL_CONTEXT))

print("\n  --- Bad answer (hallucinated details) ---")
show("faithfulness",           evaluate("faithfulness", output=HALLUCINATED_ANSWER, context=TOOL_CONTEXT))
show("contradiction detection", evaluate("contradiction_detection", output=HALLUCINATED_ANSWER, context=TOOL_CONTEXT))
show("hallucination score",    evaluate("hallucination_score", output=HALLUCINATED_ANSWER, context=TOOL_CONTEXT))

print("\n  --- Factual consistency (good answer vs tool context) ---")
show("factual consistency", evaluate("factual_consistency", output=AGENT_FINAL_ANSWER, reference=TOOL_CONTEXT))


# =========================================================================
# Part 4: LLM-Augmented Evaluation (requires GOOGLE_API_KEY)
# =========================================================================

heading("PART 4: LLM-AUGMENTED EVALUATION")

if not HAS_GEMINI:
    print("\n  Skipped — set GOOGLE_API_KEY to enable.")
    print("  This section shows the quality ladder: local heuristic → LLM-augmented.")
    print("  Just add model= to any judgment metric and the SDK builds the prompt")
    print("  internally from the local heuristic scores.\n")
else:
    print("\n  Same metric, just add model= to get LLM-refined judgment.\n")

    print("  --- Faithfulness ---")
    # Local only (heuristic)
    r_local = evaluate("faithfulness", output=AGENT_FINAL_ANSWER, context=TOOL_CONTEXT)
    show("faithfulness (local)", r_local)

    # LLM-augmented — runs local first, then LLM refines using heuristic scores
    r_augmented = evaluate("faithfulness", output=AGENT_FINAL_ANSWER, context=TOOL_CONTEXT,
                           model="gemini/gemini-2.5-flash")
    show("faithfulness (LLM-augmented)", r_augmented)

    print("\n  --- Hallucinated answer ---")
    r_local = evaluate("faithfulness", output=HALLUCINATED_ANSWER, context=TOOL_CONTEXT)
    show("faithfulness (local)", r_local)

    r_augmented = evaluate("faithfulness", output=HALLUCINATED_ANSWER, context=TOOL_CONTEXT,
                           model="gemini/gemini-2.5-flash")
    show("faithfulness (LLM-augmented)", r_augmented)

    print("\n  --- Deterministic metrics ignore model= (no augmentation) ---")
    r = evaluate("contains", output=AGENT_FINAL_ANSWER, keyword="JAL-5012",
                 model="gemini/gemini-2.5-flash")
    show("contains (stays local)", r)


heading("DONE")
print("  All Phase 1 metrics demonstrated on a realistic travel agent scenario.\n")
