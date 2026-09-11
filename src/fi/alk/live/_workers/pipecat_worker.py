"""Pipecat lane worker (3C rung 1) — untrusted subprocess entry (P3-D1).

Builds a REAL Pipecat ``Pipeline`` and injects ``TranscriptionFrame``s
(bypassing STT/TTS — Pipecat's own documented eval technique), collecting
output text frames + TTFB timing into the JSONL stdio stream.

Boot config: ``pipeline_factory`` is an optional dotted ``module:attr``
returning a LIST of frame processors (the user's pipeline core); when
absent, a deterministic scripted responder is used. The worker always
appends its own collector sink to observe output frames.

IPC: see livekit_worker.py — same one-boot-line / JSONL-events contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
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


async def _run(boot: dict[str, Any]) -> None:
    import importlib.metadata

    import pipecat
    from pipecat.frames.frames import EndFrame, TextFrame, TranscriptionFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    version = importlib.metadata.version("pipecat-ai")
    _emit(
        "lane",
        "framework_ready",
        {
            "framework": "pipecat-ai",
            "framework_version": version,
            "capability_hash": _capability_hash("pipecat-ai", version),
            "package_paths": [os.path.dirname(pipecat.__file__)],
        },
    )
    rung = int(boot.get("rung") or 1)
    if rung != 1:
        raise RuntimeError(f"pipecat worker implements rung 1 only, got {rung}")
    config = boot.get("config") or {}
    responses = [str(r) for r in (config.get("responses") or [])]
    turns = boot.get("turns") or []

    response_index = 0

    class _ScriptedResponder(FrameProcessor):
        """Deterministic stand-in for the user's LLM stage (rung 1)."""

        async def process_frame(self, frame: Any, direction: Any) -> None:
            nonlocal response_index
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                if responses:
                    reply = responses[response_index % len(responses)]
                else:
                    reply = f"ack: {frame.text}"
                response_index += 1
                await self.push_frame(TextFrame(reply), FrameDirection.DOWNSTREAM)
            await self.push_frame(frame, direction)

    collected: list[tuple[float, str]] = []

    class _Collector(FrameProcessor):
        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, TextFrame) and not isinstance(
                frame, TranscriptionFrame
            ):
                collected.append((time.monotonic(), str(frame.text)))
            await self.push_frame(frame, direction)

    factory_path = config.get("pipeline_factory")
    if factory_path:
        module_name, _, attr = str(factory_path).partition(":")
        if not module_name or not attr:
            raise RuntimeError(
                f"pipeline_factory must be 'module:attr', got {factory_path!r}"
            )
        factory = getattr(importlib.import_module(module_name), attr)
        processors = factory()
        if not isinstance(processors, (list, tuple)) or not processors:
            raise RuntimeError(
                "pipeline_factory must return a non-empty list of frame "
                "processors (the worker appends its own collector sink)"
            )
        processors = list(processors)
    else:
        processors = [_ScriptedResponder()]
    pipeline = Pipeline([*processors, _Collector()])
    task = PipelineTask(pipeline)
    runner = PipelineRunner(handle_sigint=False)

    checks: list[bool] = []

    async def _drive() -> None:
        for index, turn in enumerate(turns):
            text = str((turn or {}).get("user") or "")
            _emit("user", "message", {"turn": index, "text": text})
            injected_at = time.monotonic()
            seen_before = len(collected)
            frame_kwargs = {
                "text": text,
                "user_id": "user",
                "timestamp": str(injected_at),
            }
            try:
                frame = TranscriptionFrame(**frame_kwargs)
            except TypeError:
                frame = TranscriptionFrame(text, "user", str(injected_at))
            await task.queue_frame(frame)
            # Wait (bounded) for the pipeline to produce this turn's output.
            deadline = time.monotonic() + 10.0
            while len(collected) <= seen_before and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            new_outputs = collected[seen_before:]
            if new_outputs:
                first_at, reply = new_outputs[0]
                ttfb_ms = round((first_at - injected_at) * 1000.0, 3)
                _emit("agent", "message", {"turn": index, "text": reply})
                _emit(
                    "lane",
                    "timing",
                    {"turn": index, "ttfb_ms": ttfb_ms},
                )
                ok = bool(reply.strip())
                expect = (turn or {}).get("expect")
                if isinstance(expect, dict) and isinstance(
                    expect.get("contains"), str
                ):
                    ok = ok and expect["contains"].lower() in reply.lower()
                checks.append(ok)
            else:
                checks.append(False)
        await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(task), _drive())
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
