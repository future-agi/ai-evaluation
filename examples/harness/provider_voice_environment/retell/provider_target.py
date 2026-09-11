"""Create and delete the Retell resources owned by one ALK world."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE = "https://api.retellai.com"


def request(
    method: str, path: str, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body).encode()
    req = Request(
        BASE + path,
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['RETELL_API_KEY']}",
            "Content-Type": "application/json",
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
    created_llm = request(
        "POST",
        "/create-retell-llm",
        {
            "general_prompt": (
                "Have a natural multi-turn conversation. After the opening greeting, wait for an "
                "actual caller response before taking any other action. Ask what the caller prefers "
                "and clarify until they give one final preference. If the caller initially asks to "
                "skip or hang up without giving a preference, politely ask exactly once more; do not "
                "agree to end yet. You MUST call record_preference exactly once with that final "
                "preference. After the tool succeeds, briefly confirm it, then MUST invoke end_call. "
                "Never say goodbye, claim completion, or invoke end_call before record_preference "
                "succeeds."
            ),
            "begin_message": "Hello, what preference can I record for you today?",
            "general_tools": [
                {
                    "type": "custom",
                    "name": "record_preference",
                    "description": "Record the caller's stated preference.",
                    "url": os.environ["ALK_TOOL_BASE_URL"] + "/record_preference",
                    "method": "POST",
                    "parameters": {
                        "type": "object",
                        "properties": {"preference": {"type": "string"}},
                        "required": ["preference"],
                    },
                },
                {
                    "type": "end_call",
                    "name": "end_call",
                    "description": (
                        "Required final action. Invoke immediately after record_preference succeeds "
                        "and you briefly confirm the saved preference."
                    ),
                },
            ],
        },
    )
    llm_id = created_llm["llm_id"]
    try:
        created_agent = request(
            "POST",
            "/create-agent",
            {
                "agent_name": context["provider_resource_prefix"],
                "voice_id": os.environ["RETELL_VOICE_ID"],
                "response_engine": {"type": "retell-llm", "llm_id": llm_id},
                "webhook_url": os.environ["ALK_EVENT_URL"],
            },
        )
    except Exception:
        request("DELETE", f"/delete-retell-llm/{llm_id}")
        raise
    agent_id = created_agent["agent_id"]
    receipt = {
        "schema_version": "1",
        "provider": "retell",
        "attempt_id": context["attempt_id"],
        "world_id": context["world_id"],
        "target": {"kind": "voice_agent", "id": agent_id},
        "resources": [
            {"kind": "retell_llm", "id": llm_id, "owned": True},
            {"kind": "voice_agent", "id": agent_id, "owned": True},
        ],
        "cleanup": {
            "receipt_version": "1",
            "idempotency_key": context["idempotency_key"],
        },
        "metadata": {"fixture": "code_upload"},
    }
    Path(os.environ["ALK_PROVIDER_OUTPUT"]).write_text(json.dumps(receipt))


def destroy() -> None:
    receipt = json.loads(Path(os.environ["ALK_PROVIDER_RECEIPT"]).read_text())
    for resource in reversed(receipt["resources"]):
        if resource["kind"] == "voice_agent":
            request("DELETE", f"/delete-agent/{resource['id']}")
        elif resource["kind"] == "retell_llm":
            request("DELETE", f"/delete-retell-llm/{resource['id']}")


if __name__ == "__main__":
    {"provision": provision, "destroy": destroy}[sys.argv[1]]()
