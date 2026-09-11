from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from fi.simulate.agent.wrapper import AgentInput, AgentResponse, AgentWrapper


class EchoAgentWrapper(AgentWrapper):
    """Simple deterministic agent for smoke tests and cookbook examples."""

    async def call(self, input: AgentInput) -> str:
        content = input.new_message.get("content", "") if input.new_message else ""
        return f"Echo: {content}"


class ScriptedAgentWrapper(AgentWrapper):
    """
    Returns a fixed sequence of responses, then repeats a default response.

    Useful for offline simulation, regression tests, and docs because it behaves
    exactly the same without LLM credentials.
    """

    def __init__(
        self,
        responses: Iterable[str | AgentResponse],
        *,
        default_response: str | AgentResponse = "I can help with that. What else do you need?",
    ) -> None:
        self.responses = list(responses)
        self.default_response = default_response
        self.calls: List[AgentInput] = []

    async def call(self, input: AgentInput) -> str | AgentResponse:
        self.calls.append(input)
        index = len(self.calls) - 1
        if index < len(self.responses):
            return self.responses[index]
        return self.default_response


class RuleBasedAgentWrapper(AgentWrapper):
    """
    Tiny keyword-rule agent for local simulations.

    Rules are checked in insertion order. A rule value can be a string,
    AgentResponse, or callable receiving AgentInput.
    """

    def __init__(
        self,
        rules: Mapping[str, str | AgentResponse | Callable[[AgentInput], str | AgentResponse]],
        *,
        default_response: str | AgentResponse = "I do not have enough information yet.",
    ) -> None:
        self.rules = dict(rules)
        self.default_response = default_response
        self.calls: List[AgentInput] = []

    async def call(self, input: AgentInput) -> str | AgentResponse:
        self.calls.append(input)
        content = input.new_message.get("content", "") if input.new_message else ""
        content_lower = content.lower()
        for keyword, response in self.rules.items():
            if keyword.lower() in content_lower:
                if callable(response):
                    return response(input)
                return response
        return self.default_response


def make_tool_response(
    content: str,
    *,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    result: Any = None,
    call_id: str = "call_mock_1",
) -> AgentResponse:
    """Create a response with a normalized tool call and matching tool result."""

    return AgentResponse(
        content=content,
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments or {},
                },
            }
        ],
        tool_responses=[
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result if isinstance(result, str) else str(result),
            }
        ],
    )
