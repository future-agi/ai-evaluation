"""Create and delete the Vapi resources owned by one ALK world."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE = "https://api.vapi.ai"


def request(
    method: str, path: str, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body).encode()
    req = Request(
        BASE + path,
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['VAPI_API_KEY']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "FutureAGI-ALK/1.0",
        },
    )
    try:
        with urlopen(req, timeout=30) as response:
            raw = response.read()
    except HTTPError as exc:
        if method == "DELETE" and exc.code == 404:
            return {}
        raise
    return json.loads(raw) if raw else {}


def provision() -> None:
    context = json.loads(Path(os.environ["ALK_PROVIDER_CONTEXT"]).read_text())
    created = request(
        "POST",
        "/assistant",
        {
            "name": context["provider_resource_prefix"],
            "firstMessage": "Hello, what preference can I record for you today?",
            "server": {"url": os.environ["ALK_EVENT_URL"]},
            "model": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Have a natural multi-turn conversation. Ask what the caller prefers, "
                            "then call record_preference with that preference before finishing."
                        ),
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "record_preference",
                            "description": "Record the caller's stated preference.",
                            "parameters": {
                                "type": "object",
                                "properties": {"preference": {"type": "string"}},
                                "required": ["preference"],
                            },
                        },
                        "server": {
                            "url": os.environ["ALK_TOOL_BASE_URL"]
                            + "/record_preference"
                        },
                    }
                ],
            },
            "voice": {"provider": "vapi", "voiceId": "Elliot"},
            "metadata": {
                "alk_attempt_id": context["attempt_id"],
                "alk_world_id": context["world_id"],
            },
        },
    )
    target_id = created["id"]
    receipt = {
        "schema_version": "1",
        "provider": "vapi",
        "attempt_id": context["attempt_id"],
        "world_id": context["world_id"],
        "target": {"kind": "assistant", "id": target_id},
        "resources": [{"kind": "assistant", "id": target_id, "owned": True}],
        "cleanup": {
            "receipt_version": "1",
            "idempotency_key": context["idempotency_key"],
        },
        "metadata": {"fixture": "code_upload"},
    }
    Path(os.environ["ALK_PROVIDER_OUTPUT"]).write_text(json.dumps(receipt))


def destroy() -> None:
    receipt = json.loads(Path(os.environ["ALK_PROVIDER_RECEIPT"]).read_text())
    request("DELETE", f"/assistant/{receipt['target']['id']}")


if __name__ == "__main__":
    {"provision": provision, "destroy": destroy}[sys.argv[1]]()
