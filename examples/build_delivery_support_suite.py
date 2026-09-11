from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import studio
from fi.simulate.simulation.models import Scenario

_CASES = (
    {
        "name": "Morgan",
        "role": "busy customer",
        "order_id": "DS-1001",
        "situation": "My delivery for order DS-1001 has not arrived and I need its status.",
        "temperament": {"rajas": 0.4, "sattva": 0.9, "tamas": 0.1},
        "style_notes": ["polite", "concise"],
    },
    {
        "name": "Avery",
        "role": "impatient customer",
        "order_id": "DS-1002",
        "situation": "Order DS-1002 is a day late and I need a concrete arrival time.",
        "temperament": {"rajas": 0.9, "sattva": 0.4, "tamas": 0.1},
        "style_notes": ["urgent", "direct"],
    },
    {
        "name": "Jordan",
        "role": "privacy-conscious customer",
        "order_id": "DS-1003",
        "situation": "I want the status of order DS-1003 but I do not want to share unrelated personal data.",
        "temperament": {"rajas": 0.3, "sattva": 0.7, "tamas": 0.4},
        "style_notes": ["cautious", "measured"],
    },
    {
        "name": "Riley",
        "role": "detail-oriented customer",
        "order_id": "DS-1004",
        "situation": "Please explain the current location and next delivery step for order DS-1004.",
        "temperament": {"rajas": 0.5, "sattva": 0.8, "tamas": 0.2},
        "style_notes": ["precise", "asks follow-up questions"],
    },
    {
        "name": "Casey",
        "role": "frustrated customer",
        "order_id": "DS-1005",
        "situation": "Order DS-1005 missed its promised window twice and I need this resolved.",
        "temperament": {"rajas": 0.8, "sattva": 0.3, "tamas": 0.3},
        "style_notes": ["frustrated", "expects accountability"],
    },
    {
        "name": "Taylor",
        "role": "cooperative customer",
        "order_id": "DS-1006",
        "situation": "I am checking whether order DS-1006 will arrive before I leave town.",
        "temperament": {"rajas": 0.4, "sattva": 0.95, "tamas": 0.1},
        "style_notes": ["cooperative", "clear"],
    },
    {
        "name": "Quinn",
        "role": "skeptical customer",
        "order_id": "DS-1007",
        "situation": "The tracking page for order DS-1007 has not changed and I need evidence of its status.",
        "temperament": {"rajas": 0.6, "sattva": 0.5, "tamas": 0.3},
        "style_notes": ["skeptical", "requests confirmation"],
    },
    {
        "name": "Parker",
        "role": "distracted customer",
        "order_id": "DS-1008",
        "situation": "I only have a minute to check the delivery status of order DS-1008.",
        "temperament": {"rajas": 0.7, "sattva": 0.6, "tamas": 0.2},
        "style_notes": ["brief", "easily distracted"],
    },
    {
        "name": "Cameron",
        "role": "patient customer",
        "order_id": "DS-1009",
        "situation": "Order DS-1009 is delayed and I would like to understand the revised schedule.",
        "temperament": {"rajas": 0.2, "sattva": 0.9, "tamas": 0.3},
        "style_notes": ["patient", "thoughtful"],
    },
    {
        "name": "Drew",
        "role": "escalation-prone customer",
        "order_id": "DS-1010",
        "situation": "I need an immediate status and escalation path for missing order DS-1010.",
        "temperament": {"rajas": 0.95, "sattva": 0.25, "tamas": 0.2},
        "style_notes": ["forceful", "escalates when answers are vague"],
    },
)


def build_suite() -> Scenario:
    outcome = "The delivery status, expected arrival, and next step are confirmed."
    personas = [
        studio.build_persona(
            name=case["name"],
            role=case["role"],
            situation=case["situation"],
            outcome=outcome,
            style_notes=case["style_notes"],
            temperament=case["temperament"],
            knowledge=[
                {
                    "key": "order_number",
                    "value": f"My order number is {case['order_id']}",
                    "disclosure": "volunteer",
                }
            ],
            evidence_class="schema_sampled",
        )
        for case in _CASES
    ]
    for persona in personas:
        validation = studio.validate_persona(persona)
        if validation["status"] != "valid":
            raise ValueError(f"persona validation failed: {persona.identity.name}")
    bias = studio.bias_lint(personas)
    if bias["status"] != "passed":
        raise ValueError("delivery support suite failed bias lint")
    return Scenario(
        name="delivery-support-studio-suite",
        description="Ten typed delivery-status callers for text and WebRTC acceptance.",
        kind="task",
        dataset=personas,
        coverage={
            "intents": ["delivery_status", "arrival_estimate", "next_step"],
            "personas": [persona.version for persona in personas],
            "perturbations": ["urgency", "privacy", "skepticism", "escalation"],
            "tool_obligations": ["allow:lookup_delivery"],
        },
        constraints={
            "declared_tools": ["lookup_delivery"],
            "max_user_knowledge": ["order_number"],
        },
    )


def write_suite(path: str | Path) -> dict[str, Any]:
    scenario = build_suite()
    payload = scenario.model_dump(mode="json", exclude_none=True)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/build_delivery_support_suite.py OUTPUT.json")
    write_suite(sys.argv[1])
