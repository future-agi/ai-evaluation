---
kind: agent-learning.docs-page.v1
track: frameworks
backing: []
artifact_kinds: []
claims: []
---

# Framework-adapter IO profiles

The framework-adapter layer binds each preset to one of eight executable IO
contracts by adapter shape. These five pages document the IO surfaces that did
not already have a dedicated framework page; the other three (`keyword_inputs`,
`side_kwargs`, `provider_response`) are documented by the crewai, pipecat, and
model-client cookbook pages.

- [Streaming](streaming.md) — incremental token/tool deltas (`astream` shape).
- [Typed output](typed_output.md) — a structured/typed object the adapter coerces.
- [Nested method](nested_method.md) — a dotted-path method on a nested client.
- [Message history](message_history.md) — a multi-turn transcript with tool events.
- [Handoff transcript](handoff_transcript.md) — a multi-agent handoff/review/reconciliation transcript.

Each page is backed by its existing IO-contract example and admitted by the gate
that already covers that example (`framework_adapter_io_readiness` for streaming /
typed_output / nested_method; `framework_adapter_probe_readiness` for
message_history / handoff_transcript).
