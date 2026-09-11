# Harness backends

A stage of this harness is a conversation: a system prompt, tools, a turn budget, and a loop
that feeds tool results back until the model stops. A backend is whoever runs that loop.

```
ALK_HARNESS=claude          # default; Claude Code loop, exactly the pre-seam behaviour
ALK_HARNESS=vertex-gemini   # Google's ADK against Vertex (location=global)
ALK_HARNESS_MODEL=...       # optional; unset means the backend's own default
ALK_VERTEX_LOCATION=...     # vertex-gemini only; defaults to global (Gemini 3.x lives there)
```

## The seam

- `base.py` is the whole contract. `SessionSpec` is what a stage asks for; the reply
  vocabulary (`SessionOpened`, `ModelReply`, `ToolReturned`, `StageDone`) is what a session
  emits; `HarnessBackend` / `HarnessSession` are the two protocols a backend implements.
- Tools are declared once, neutrally (`tool`, `tool_server`), and each backend adapts them:
  the Claude backend builds an in-process MCP server, the Gemini backend builds function
  declarations. Gating follows the same split: hook-based on Claude, structural on Gemini
  (an ungranted tool is never declared).
- `Stage` (in `session.py`) drives any backend and renders the replies into events. It never
  names a vendor.

## Adding a backend

Write a module with a class exposing `name`, `default_model`, `can_drive(model)` and
`create(spec) -> HarnessSession`, where the session yields the neutral replies and ends every
exchange with a `StageDone`. Register it in `__init__.py` (or call `register` from anywhere).
Nothing else in the harness changes: every stage, gate, and artifact works as-is. That is the
slot a Bedrock, Azure, or Gemini-CLI backend drops into.

Backends load lazily, so one backend's SDK is never imported because a different one ran.

## What the Gemini backend supplies itself

The ADK owns the loop, tool execution, and session history; the backend only adapts a
``ToolSpec`` through ADK's ``BaseTool`` extension point and translates its event stream into
the neutral replies. Claude Code ships Read/Glob/Grep and an operator-question tool; ADK has
no coding-CLI file tools, so `files.py` implements the read-only file tools once for any
backend that needs them. AskUserQuestion is deliberately not declared on the Gemini backend
yet: unattended runs never call it, and declaring a tool the backend cannot answer would cost
the model a turn finding that out.
