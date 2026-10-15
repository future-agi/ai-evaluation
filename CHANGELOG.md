# Changelog

All notable release changes for Agent Learning Kit are tracked here.

## Unreleased

Post-rc.1 increments on the release branch. Each one adds gates on top of the
66 proved at rc.1; the rc.1 entry below is historical and unchanged.

- Docs corpus: the full `docs/` tree (quickstarts, per-track guides, framework
  pages, reference material, and the `docs/llms.txt` machine index) with
  machine-checkable page metadata, enforced by the new `docs_executability`
  release gate (67 gates).
- Live lanes: opt-in live execution lanes (LiveKit, Pipecat, LangGraph, MCP,
  A2A) behind per-framework extras, with the engine/public boundary enforced
  by the new `live_lane_boundary` release gate (68 gates).
- Optimizer expansion: optimizer portfolio routing, frozen capability
  profiles, and apply plans, enforced by the two new Phase-4 gates
  `optimizer_profile_matrix_readiness` and `capability_profile_freeze_readiness`
  (70 gates).
- Hosted Vapi and Retell calls now carry the target agent's provider call ID,
  native end reason, tokens, cost, and latency on result receipts instead of
  dropping that evidence; simulator usage remains separate.
- Hosted voice workers preserve stdout/stderr in the parent diagnostics stream,
  with per-call correlation, turn-state transitions, transcription dispatch,
  and simulator generation lifecycle events.
- Voice reports now seal transcripts after bounded STT/native-stream finalization,
  retain conversation evidence on failures, and explicitly report incomplete
  capture instead of silently completing with a partial transcript.
- Retell and LiveKit audio bridges close owned streams and sources, reap media
  tasks on cancellation, and drain queued playback before closing a provider call.
- Voice diagnostics include sanitized SIP error codes, transcript-finalization
  and cleanup results, isolated-worker exit status, and native crash backtraces.
- Outbound SIP calls no longer block on `wait_until_answered`. LiveKit v1.9.11
  retried that request with 3s/5s/7s deadlines, cancelling and redialing a
  ringing call so people and slow lines could never answer; the engine now
  waits for `sip.callStatus == "active"` in the room within the answer
  deadline and reports a rejected call's reason (e.g. `user_unavailable`).
- With `AGENTCC_API_KEY` (and `AGENTCC_BASE_URL`) set, the simulated caller's
  LLM and the `vertex-gemini` harness backend reach Gemini through Agent
  Command Center in the Gemini API's own format, as the claude backend already
  did, so no Vertex credential is needed. `fi.agentcc` is the one place that
  reads the gateway settings.
- Simulator model fallbacks: `SIMULATOR_LLM_FALLBACK_MODELS` (ordered,
  comma-separated) gives the simulated caller models to switch to when an
  attempt fails or times out before it speaks.
- The simulated caller's speech goes through Agent Command Center too: with
  `AGENTCC_API_KEY` set, Deepgram and Cartesia speech-to-text and
  text-to-speech connect to the gateway's copy of each vendor's streaming API
  with the gateway key, so a sandbox needs no `DEEPGRAM_API_KEY` or
  `CARTESIA_API_KEY`. Without a speech provider named, the caller then speaks
  through Cartesia, as it does when a Cartesia key is present.
- The `vertex-gemini` backend no longer fails on its first reply with the locked
  ADK (2.7.1), which does not re-export `GetSessionConfig` from
  `google.adk.sessions`.

## v1.0.0-rc.1 — 2026-06-10

First locally-cut v1 release candidate. Package labels: Python
`agent-learning-kit==0.1.0`, TypeScript `@future-agi/agent-learning-kit==0.2.0`
(decision records D1/D2; the tag, not the semver, names the product milestone).

- One SDK and CLI (`fi.alk` / `agent-learn`) consolidating the
  `simulate`, `evals`, and `opt` engines — three engines, four workflows
  (test, simulate, red-team, optimize).
- 66 executable release gates behind `agent-learn release-check`, proved by
  `agent-learn release-proof` (`agent-learning.release-proof.v1`) on the cut
  commit.
- Distribution hygiene: the sdist now ships only `src/`, `tests/`, `examples/`,
  `docs/`, and the standard release files — `internal-docs/`, `uv.lock`, the
  roadmap, internal guides, the `typescript/` workspace, and build artifacts no
  longer leak; enforced by the new `package_distribution_hygiene` gate.
- `Development Status` classifier moved to `4 - Beta` (D3); `uv.lock` tracked
  in git and excluded from the sdist (D4).
- README claims reconciled with executable proof (LlamaIndex listed,
  `AGENT_LEARNING_API_KEY` named, OpenEnv positioning deduplicated with the
  robustness bar defined, install framing honest pre-publish, probe-promoted
  vs runtime-simulated coverage distinguished).
- Prepared v1 release-candidate documentation, Apache-2.0 licensing artifacts,
  community files, and release-proof handoff notes.
- Added a developer-first README opening with install, quickstart, workflow,
  release proof, repository map, and community links.
- Hardened `agent-learn release-proof` timeout handling and raised the default
  per-command timeout to 7200s for the expanded v1 proof suite (the full pytest
  suite, which executes every release gate inside the milestone test, exceeds
  the previous 2400s budget).
