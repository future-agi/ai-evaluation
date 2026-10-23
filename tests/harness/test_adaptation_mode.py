"""Which binding mode an agent gets: canonical only when its tools can be called as-is."""

from __future__ import annotations

import pytest

from fi.alk.harness.adaptation.profile import AgentProfile, select_mode
from fi.alk.harness.provider_import import ToolAdmission
from fi.alk.harness.provider_lifecycle import ProviderType
from fi.alk.harness.releases import BindingMode, ToolDefinition

BOOK = ToolDefinition(
    name="book_appointment",
    description="Book a slot.",
    parameters={
        "type": "object",
        "properties": {
            "patient_id": {"type": "string"},
            "slot_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["patient_id", "slot_id", "reason"],
    },
    mutates_state=True,
)
SEND = ToolDefinition(
    name="send_intake_forms",
    description="Send the forms.",
    parameters={
        "type": "object",
        "properties": {
            "patient_id": {"type": "string"},
            "channel": {"type": "string", "enum": ["sms", "email"]},
        },
        "required": ["patient_id", "channel"],
    },
    mutates_state=True,
)


def _agent(*tools: dict) -> AgentProfile:
    return AgentProfile(
        connector="vapi",
        provider=ProviderType.VAPI,
        target={},
        admission=[
            ToolAdmission(
                name=tool["name"], provider_type="function", action="routed", reason=""
            )
            for tool in tools
        ],
        agent_tools=[{**tool, "admission": "routed"} for tool in tools],
        agent=None,
    )


def _tool(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def test_same_names_and_arguments_call_the_environment_directly():
    agent = _agent(
        _tool(
            "book_appointment",
            {
                "patient_id": {"type": "string"},
                "slot_id": {"type": "string"},
                "reason": {"type": "string", "description": "Why they are coming in"},
            },
            ["patient_id", "slot_id", "reason"],
        ),
        _tool(
            "send_intake_forms",
            {
                "patient_id": {"type": "string"},
                "channel": {"type": "string", "enum": ["email"]},
            },
            ["patient_id", "channel"],
        ),
    )
    assert select_mode(agent, [BOOK, SEND]) is BindingMode.CANONICAL


@pytest.mark.parametrize(
    "properties",
    [
        # The reviewer's case: same tool name, different argument names.
        {
            "patient": {"type": "string"},
            "slot": {"type": "string"},
            "reason": {"type": "string"},
        },
        # Same names, a required environment argument the agent never sends.
        {"patient_id": {"type": "string"}, "slot_id": {"type": "string"}},
        # Same names, an argument of another type.
        {
            "patient_id": {"type": "integer"},
            "slot_id": {"type": "string"},
            "reason": {"type": "string"},
        },
    ],
    ids=["renamed-arguments", "missing-required", "different-type"],
)
def test_same_name_with_incompatible_arguments_needs_a_routed_binding(properties):
    agent = _agent(_tool("book_appointment", properties, list(properties)))
    assert select_mode(agent, [BOOK, SEND]) is BindingMode.TOOL_ROUTED


def test_a_value_the_environment_does_not_accept_needs_a_routed_binding():
    agent = _agent(
        _tool(
            "send_intake_forms",
            {
                "patient_id": {"type": "string"},
                "channel": {"type": "string", "enum": ["sms", "email", "fax"]},
            },
            ["patient_id", "channel"],
        )
    )
    assert select_mode(agent, [BOOK, SEND]) is BindingMode.TOOL_ROUTED
