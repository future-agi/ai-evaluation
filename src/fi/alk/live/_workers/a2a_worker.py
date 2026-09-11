"""A2A lane worker (3E) — untrusted subprocess entry (P3-D1).

Doubles as the loopback A2A peer entry (peer mode). In client mode it walks
the protocol stages against a peer — card discovery → task lifecycle →
artifact exchange (R§1 #18) — spawning its own peer-mode sibling on
127.0.0.1 when no remote peer URL is given (the shipped loopback default
tier). In peer mode it serves a deterministic echo agent over the REAL A2A
HTTP protocol.

IPC with the harness (client mode): see livekit_worker.py — same
one-boot-line / JSONL contract. The peer subprocess receives its own boot
line (``{"type": "boot", "mode": "peer", "port": N}``) from the client.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
import uuid
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# --- peer mode ----------------------------------------------------------------


def _run_peer(boot: dict[str, Any]) -> None:
    import uvicorn
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill
    from a2a.utils import new_agent_text_message

    port = int(boot.get("port") or 0) or _free_port()

    class _EchoExecutor(AgentExecutor):
        """Deterministic loopback peer behavior: echo the user text."""

        async def execute(self, context: Any, event_queue: Any) -> None:
            text = ""
            try:
                text = context.get_user_input()
            except Exception:
                pass
            await event_queue.enqueue_event(
                new_agent_text_message(f"echo: {text}")
            )

        async def cancel(self, context: Any, event_queue: Any) -> None:
            return None

    skill = AgentSkill(
        id="echo",
        name="Echo",
        description="Echoes the inbound message text (deterministic).",
        tags=["echo", "loopback"],
    )
    card = AgentCard(
        name="agent-learning-loopback-peer",
        description="Credential-free loopback A2A peer shipped with the kit.",
        url=f"http://127.0.0.1:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )
    handler = DefaultRequestHandler(
        agent_executor=_EchoExecutor(), task_store=InMemoryTaskStore()
    )
    application = A2AStarletteApplication(agent_card=card, http_handler=handler)
    uvicorn.run(
        application.build(), host="127.0.0.1", port=port, log_level="error"
    )


# --- client mode ----------------------------------------------------------------


def _extract_texts(value: Any, into: list[str]) -> None:
    """Best-effort recursive text extraction across a2a-sdk event shapes."""

    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            into.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _extract_texts(item, into)
        return
    for attribute in ("text", "parts", "artifacts", "history", "root", "message", "status"):
        if hasattr(value, attribute):
            _extract_texts(getattr(value, attribute), into)


async def _run_client(boot: dict[str, Any]) -> None:
    import importlib.metadata

    import httpx

    import a2a as a2a_pkg
    from a2a.client import A2ACardResolver

    version = importlib.metadata.version("a2a-sdk")
    _emit(
        "lane",
        "framework_ready",
        {
            "framework": "a2a-sdk",
            "framework_version": version,
            "capability_hash": _capability_hash("a2a-sdk", version),
            "package_paths": [os.path.dirname(a2a_pkg.__file__)],
        },
    )
    config = boot.get("config") or {}
    stages = [str(stage) for stage in (config.get("stages") or [])]
    message_text = str(config.get("message") or "ping from the harness")
    peer_url = config.get("peer_url")
    peer_process: subprocess.Popen[str] | None = None
    checks: dict[str, bool] = {}
    try:
        if not peer_url:
            port = _free_port()
            peer_process = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                env=dict(os.environ),
            )
            assert peer_process.stdin is not None
            peer_process.stdin.write(
                json.dumps({"type": "boot", "mode": "peer", "port": port}) + "\n"
            )
            peer_process.stdin.flush()
            peer_url = f"http://127.0.0.1:{port}"

        base_url = str(peer_url).rstrip("/")
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            # Wait for the peer to come up (loopback boot is fast but async).
            card_paths = (
                "/.well-known/agent-card.json",
                "/.well-known/agent.json",
            )
            deadline = time.monotonic() + 30.0
            reachable = False
            while time.monotonic() < deadline and not reachable:
                for path in card_paths:
                    try:
                        response = await http_client.get(base_url + path)
                        if response.status_code == 200:
                            reachable = True
                            break
                    except httpx.HTTPError:
                        pass
                if not reachable:
                    await asyncio.sleep(0.2)
            if not reachable:
                raise RuntimeError(f"A2A peer at {base_url} never became reachable")

            # --- stage: card discovery -----------------------------------
            resolver = A2ACardResolver(http_client, base_url)
            card = await resolver.get_agent_card()
            card_ok = bool(getattr(card, "name", None))
            _emit(
                "agent",
                "protocol_stage",
                {"stage": "card_discovery", "ok": card_ok, "peer": getattr(card, "name", None)},
            )
            if "card_discovery" in stages:
                checks["card_discovery"] = card_ok

            # --- stage: task lifecycle + artifact exchange ----------------
            texts: list[str] = []
            lifecycle_ok = False
            try:
                from a2a.client import ClientConfig, ClientFactory
                from a2a.types import Message, Part, Role, TextPart

                factory = ClientFactory(ClientConfig(httpx_client=http_client))
                client = factory.create(card)
                try:
                    outbound = Message(
                        role=Role.user,
                        parts=[Part(root=TextPart(text=message_text))],
                        message_id=uuid.uuid4().hex,
                    )
                except TypeError:
                    outbound = Message(
                        role=Role.user,
                        parts=[Part(root=TextPart(text=message_text))],
                        messageId=uuid.uuid4().hex,
                    )
                _emit("user", "message", {"text": message_text})
                async for event in client.send_message(outbound):
                    lifecycle_ok = True
                    _extract_texts(event, texts)
            except ImportError:
                # Older SDK line: single-shot A2AClient JSON-RPC surface.
                from a2a.client import A2AClient
                from a2a.types import (
                    Message,
                    MessageSendParams,
                    Part,
                    Role,
                    SendMessageRequest,
                    TextPart,
                )

                client = A2AClient(httpx_client=http_client, agent_card=card)
                request = SendMessageRequest(
                    id=uuid.uuid4().hex,
                    params=MessageSendParams(
                        message=Message(
                            role=Role.user,
                            parts=[Part(root=TextPart(text=message_text))],
                            messageId=uuid.uuid4().hex,
                        )
                    ),
                )
                _emit("user", "message", {"text": message_text})
                response = await client.send_message(request)
                lifecycle_ok = response is not None
                _extract_texts(response, texts)

            reply = next((text for text in texts if "echo" in text.lower()), "")
            artifact_ok = bool(reply) or any(text.strip() for text in texts)
            _emit(
                "agent",
                "protocol_stage",
                {"stage": "task_lifecycle", "ok": lifecycle_ok},
            )
            _emit(
                "agent",
                "protocol_stage",
                {
                    "stage": "artifact_exchange",
                    "ok": artifact_ok,
                    "text": (reply or " ".join(texts))[:500],
                },
            )
            if "task_lifecycle" in stages:
                checks["task_lifecycle"] = lifecycle_ok
            if "artifact_exchange" in stages:
                checks["artifact_exchange"] = artifact_ok

        passed = bool(checks) and all(checks.values())
        _emit("lane", "verification", {"passed": passed, "checks": checks})
    finally:
        if peer_process is not None:
            peer_process.terminate()
            try:
                peer_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                peer_process.kill()


def main() -> int:
    boot = _read_boot()
    mode = str(boot.get("mode") or "client")
    try:
        if mode == "peer":
            _run_peer(boot)
        else:
            asyncio.run(_run_client(boot))
    except Exception:
        _emit("lane", "worker_error", {"traceback": traceback.format_exc()})
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
