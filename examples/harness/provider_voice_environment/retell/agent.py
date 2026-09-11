"""Environment backend for the Retell code-upload certification fixture."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request


app = FastAPI()
_lock = threading.Lock()
_trace_path = Path(os.environ.get("PROVIDER_TRACE_PATH", "/tmp/provider-trace.jsonl"))


def _record(kind: str, body: Any) -> None:
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "body": body,
    }
    with _lock:
        _trace_path.parent.mkdir(parents=True, exist_ok=True)
        with _trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/provider/events")
async def provider_events(request: Request) -> dict[str, bool]:
    _record("provider_event", await request.json())
    return {"received": True}


@app.post("/provider/tools/record_preference")
async def record_preference(request: Request) -> dict[str, Any]:
    body = await request.json()
    _record("tool.record_preference", body)
    preference = body.get("preference") if isinstance(body, dict) else None
    if preference is None and isinstance(body, dict):
        arguments = body.get("args") or body.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = None
        if isinstance(arguments, dict):
            preference = arguments.get("preference")
    return {
        "recorded": True,
        "preference": preference,
        "message": "The caller preference was recorded in the isolated test environment.",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
