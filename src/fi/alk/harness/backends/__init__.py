"""Harness backends, selected by name.

``ALK_HARNESS`` picks the backend the way ``ALK_HARNESS_MODEL`` already picks the model. With
nothing set the choice is ``vertex-gemini``, so a machine holding only Google credentials runs
without being told to; ``ALK_HARNESS=claude`` selects the Claude Code loop instead.

Backends load lazily: choosing one never imports the other's SDK, so a deployment installs only
the provider it uses. A new backend is a module implementing ``HarnessBackend`` plus one
``register`` call, from anywhere; nothing else in the harness changes.
"""

from __future__ import annotations

import os
from typing import Callable

from .base import (
    ASK_TOOL,
    FILE_TOOLS,
    KNOWN_BUILTINS,
    Call,
    HarnessBackend,
    HarnessSession,
    ModelReply,
    Say,
    SessionOpened,
    SessionSpec,
    StageDone,
    ToolReturned,
    ToolServer,
    ToolSpec,
    qualified,
    tool,
    tool_server,
)

__all__ = [
    "ASK_TOOL",
    "FILE_TOOLS",
    "KNOWN_BUILTINS",
    "Call",
    "HarnessBackend",
    "HarnessSession",
    "ModelReply",
    "Say",
    "SessionOpened",
    "SessionSpec",
    "StageDone",
    "ToolReturned",
    "ToolServer",
    "ToolSpec",
    "qualified",
    "tool",
    "tool_server",
    "register",
    "resolve",
    "backend_names",
]

DEFAULT_BACKEND = "vertex-gemini"

_LOADERS: dict[str, Callable[[], HarnessBackend]] = {}
_ALIASES = {
    "gemini": "vertex-gemini",
    "vertex_gemini": "vertex-gemini",
    "vertexai-gemini": "vertex-gemini",
    "claude-code": "claude",
}
_LIVE: dict[str, HarnessBackend] = {}


def register(name: str, loader: Callable[[], HarnessBackend]) -> None:
    """Make a backend selectable by name. Loader runs on first use, not at registration."""
    _LOADERS[name] = loader


def _load_claude() -> HarnessBackend:
    from .claude import ClaudeBackend

    return ClaudeBackend()


def _load_vertex_gemini() -> HarnessBackend:
    from .vertex_gemini import VertexGeminiBackend

    return VertexGeminiBackend()


register("claude", _load_claude)
register("vertex-gemini", _load_vertex_gemini)


def backend_names() -> list[str]:
    return sorted(_LOADERS)


def resolve(name: str | None = None) -> HarnessBackend:
    """The backend a run will use: the one named, or ALK_HARNESS, or the default.

    An unknown name is a loud error naming what exists. Falling back silently would run a whole
    suite on the wrong harness, which is only discovered from the bill.
    """
    asked = (name or os.environ.get("ALK_HARNESS") or DEFAULT_BACKEND).strip().lower()
    asked = _ALIASES.get(asked, asked)
    if asked not in _LOADERS:
        raise ValueError(
            f"no harness backend named {asked!r}; installed backends: "
            f"{', '.join(backend_names())}"
        )
    if asked not in _LIVE:
        _LIVE[asked] = _LOADERS[asked]()
    backend = _LIVE[asked]
    # A named model that this backend cannot reach is a configuration mistake, and it is only
    # visible here. Left to run, the provider rejects the model mid-stage and the failure reads
    # as the harness having nothing to say rather than as the wrong pairing.
    wanted = os.environ.get("ALK_HARNESS_MODEL", "").strip()
    if wanted and not backend.can_drive(wanted):
        raise ValueError(
            f"harness backend {backend.name!r} cannot drive model {wanted!r}; "
            f"its default is {backend.default_model!r}"
        )
    return backend
