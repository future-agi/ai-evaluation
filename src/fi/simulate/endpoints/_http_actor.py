"""Turn-based HTTP target agent (used by the ``http`` actor source).

Lifted out of ``hosted/targets.py`` so ``endpoints`` can own it without a cycle
(``targets`` now dispatches through the endpoint registry).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from fi.simulate.agent.wrapper import AgentInput, AgentWrapper

_HTTP_TIMEOUT_SECONDS = 60.0


class HttpChatAgent(AgentWrapper):
    """Turn-based target that relays each turn to an HTTP chat endpoint."""

    def __init__(
        self,
        *,
        url: str,
        auth_header: str = "Authorization",
        auth_env: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._url = url
        self._auth_header = auth_header
        self._auth_env = auth_env
        self._extra_headers = extra_headers or {}

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._extra_headers}
        if self._auth_env:
            token = os.environ.get(self._auth_env)
            if token:
                headers[self._auth_header] = token
        return headers

    async def call(self, input: AgentInput) -> str:
        payload = {
            "thread_id": input.thread_id,
            "messages": input.messages,
            "new_message": input.new_message,
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(
                self._url, json=payload, headers=self._headers()
            )
            response.raise_for_status()
            return _extract_reply(response.json())


def _extract_reply(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        for key in ("content", "reply", "message", "response", "output", "text"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


__all__ = ["HttpChatAgent"]
