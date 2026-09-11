"""LiveKit lane worker (3B rung 1) — untrusted subprocess entry (P3-D1).

Boots a REAL ``livekit.agents.AgentSession`` and drives it with the
first-party text-rung helper ``session.run(user_input=...)`` (LiveKit's own
pytest surface) under a virtual-clock turn script with a deterministic
scripted LLM (no transport, no credentials — P3-D3 rung 1).

IPC: reads ONE boot JSON line on stdin; emits one JSON object per line on
stdout: ``{"channel": "user"|"agent"|"tool"|"lane", "type": ..., "payload": ...}``.
The handshake event is ``lane/framework_ready`` carrying
``{framework, framework_version, capability_hash, package_paths}``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import traceback
from typing import Any


def _emit(channel: str, type_: str, payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            {"channel": channel, "type": type_, "payload": payload},
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


def _read_boot() -> dict[str, Any]:
    line = sys.stdin.readline()
    if not line.strip():
        raise RuntimeError("missing boot message on stdin")
    boot = json.loads(line)
    if not isinstance(boot, dict) or boot.get("type") != "boot":
        raise RuntimeError("first stdin line must be a boot message")
    return boot


def _capability_hash(framework: str, version: str) -> str:
    return hashlib.sha256(f"{framework}:{version}".encode("utf-8")).hexdigest()


def _extract_reply(result: Any) -> str:
    """Pull the last assistant text out of a session.run RunResult,
    defensively across livekit-agents 1.x minor versions."""

    texts: list[str] = []
    for event in getattr(result, "events", None) or []:
        item = getattr(event, "item", None)
        if getattr(item, "role", None) != "assistant":
            continue
        text_content = getattr(item, "text_content", None)
        if text_content:
            texts.append(str(text_content))
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, (list, tuple)):
            texts.append(
                " ".join(str(part) for part in content if isinstance(part, str))
            )
    return texts[-1] if texts else ""


async def _run(boot: dict[str, Any]) -> None:
    import importlib.metadata

    import livekit
    from livekit.agents import Agent, AgentSession
    from livekit.agents import llm as lk_llm

    version = importlib.metadata.version("livekit-agents")
    _emit(
        "lane",
        "framework_ready",
        {
            "framework": "livekit-agents",
            "framework_version": version,
            "capability_hash": _capability_hash("livekit-agents", version),
            # livekit is a NAMESPACE package: __file__ is None, roots are on
            # __path__ (same fix as langgraph_worker._package_paths).
            "package_paths": (
                [os.path.dirname(livekit.__file__)]
                if getattr(livekit, "__file__", None)
                else [str(path) for path in getattr(livekit, "__path__", None) or []]
            ),
        },
    )
    rung = int(boot.get("rung") or 1)
    if rung != 1:
        raise RuntimeError(f"livekit worker implements rung 1 only, got {rung}")
    config = boot.get("config") or {}
    responses = [str(r) for r in (config.get("responses") or [])]
    instructions = str(
        config.get("instructions")
        or "You are a concise, helpful voice agent under test."
    )
    expect = config.get("expect") if isinstance(config.get("expect"), dict) else {}
    turns = boot.get("turns") or []

    def _make_chunk(text: str) -> Any:
        try:
            return lk_llm.ChatChunk(
                id="scripted",
                delta=lk_llm.ChoiceDelta(role="assistant", content=text),
            )
        except TypeError:
            return lk_llm.ChatChunk(
                request_id="scripted",
                choices=[
                    lk_llm.Choice(
                        delta=lk_llm.ChoiceDelta(role="assistant", content=text),
                        index=0,
                    )
                ],
            )

    class _ScriptedStream(lk_llm.LLMStream):
        def __init__(self, llm_obj: Any, *, chat_ctx: Any, tools: Any, conn_options: Any, text: str) -> None:
            super().__init__(
                llm_obj, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
            )
            self._text = text

        async def _run(self) -> None:
            self._event_ch.send_nowait(_make_chunk(self._text))

    default_conn_options = getattr(lk_llm, "DEFAULT_API_CONNECT_OPTIONS", None)
    if default_conn_options is None:
        try:
            from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS as default_conn_options
        except ImportError:
            default_conn_options = None

    class _ScriptedLLM(lk_llm.LLM):
        """Deterministic stub LLM node — rung 1 is credential-free (P3-D3)."""

        def __init__(self) -> None:
            super().__init__()
            self._index = 0

        @property
        def model(self) -> str:
            return "scripted-stub"

        def chat(self, *, chat_ctx: Any, tools: Any = None, conn_options: Any = None, **kwargs: Any) -> Any:
            if responses:
                text = responses[self._index % len(responses)]
            else:
                text = "Acknowledged."
            self._index += 1
            return _ScriptedStream(
                self,
                chat_ctx=chat_ctx,
                tools=tools or [],
                conn_options=conn_options or default_conn_options,
                text=text,
            )

    session = AgentSession(llm=_ScriptedLLM())
    await session.start(Agent(instructions=instructions))
    checks: list[bool] = []
    try:
        for index, turn in enumerate(turns):
            text = str((turn or {}).get("user") or "")
            _emit("user", "message", {"turn": index, "text": text})
            result = await session.run(user_input=text)
            reply = _extract_reply(result)
            _emit("agent", "message", {"turn": index, "text": reply})
            ok = bool(reply.strip())
            contains = (turn or {}).get("expect", {}).get("contains") if isinstance((turn or {}).get("expect"), dict) else None
            contains = contains or expect.get("contains")
            if isinstance(contains, str):
                ok = ok and contains.lower() in reply.lower()
            checks.append(ok)
    finally:
        aclose = getattr(session, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass
    passed = bool(checks) and all(checks)
    _emit("lane", "verification", {"passed": passed, "checks": checks})


def main() -> int:
    boot = _read_boot()
    try:
        asyncio.run(_run(boot))
    except Exception:
        _emit("lane", "worker_error", {"traceback": traceback.format_exc()})
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
