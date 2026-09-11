from __future__ import annotations

import json
from typing import Any

from fi.simulate.agent.definition import AgentDefinition, LLMConfig
from fi.simulate.simulation.models import Persona


class ScenarioGenerator:
    """Generate scenario personas with the configured simulator LLM."""

    def __init__(
        self,
        agent_definition: AgentDefinition,
        *,
        llm_config: LLMConfig,
    ) -> None:
        # Imported at construction, not module level, so `import fi.simulate`
        # works without the optional 'livekit' extra — building the simulator LLM
        # is where livekit is genuinely first required.
        try:
            from livekit.agents.llm.chat_context import ChatContext

            from fi.simulate.simulation.livekit_models import build_livekit_llm
        except ImportError as exc:
            raise ImportError(
                "LiveKit scenario generation requires the 'livekit' optional dependency"
            ) from exc
        self._chat_context_cls = ChatContext
        self._agent_definition = agent_definition
        self._llm = build_livekit_llm(llm_config)

    async def generate(self, topic: str, num_personas: int) -> list[Persona]:
        prompt = self._create_generation_prompt(topic, num_personas)
        chat_ctx = self._chat_context_cls.empty()
        chat_ctx.add_message(role="user", content=prompt)
        stream = self._llm.chat(chat_ctx=chat_ctx)
        text = ""
        async for chunk in stream.to_str_iterable():
            text += chunk
        try:
            generated_data = _parse_generated_json(text)
            personas = generated_data["personas"]
            if not isinstance(personas, list):
                raise TypeError("personas must be a list")
            return [Persona.model_validate(persona) for persona in personas]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("scenario_generation_invalid_response") from exc

    def _create_generation_prompt(self, topic: str, num_personas: int) -> str:
        agent_context = (
            self._agent_definition.system_prompt
            or self._agent_definition.description
            or ""
        )
        return f"""
        You are a creative test case designer for voice AI agents. Your task is to generate {num_personas} diverse and realistic test case personas for an AI agent with the following description:
        ---
        AGENT DESCRIPTION: {agent_context}
        ---

        The user wants to generate scenarios related to the following topic: "{topic}".

        For each persona, you must generate:
        1. A detailed `persona` object (e.g., {{ "name": "John", "age": 45, "mood": "impatient", "background": "Is a busy executive" }}).
        2. A concise `situation` string describing the reason for their call.
        3. A clear `outcome` string describing the ideal resolution of the conversation from the user's perspective.

        Return your response as a single JSON object with a key "personas", which is a list of the generated persona objects. Do not include any other text or formatting.
        """


def _parse_generated_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
        for part in text.strip().split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate.startswith("{") or not candidate.endswith("}"):
                continue
            payload = json.loads(candidate)
            break
        if payload is None:
            raise
    if not isinstance(payload, dict):
        raise TypeError("scenario response must be an object")
    return payload
