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
  reads the gateway settings. A caller on `SIMULATOR_LLM_PROVIDER=vertex`, and
  the `vertex-gemini` backend, stay on Vertex: a bare model name such as
  `gemini-3.7-flash` is sent as `vertex_ai/gemini-3.7-flash`, which the gateway
  routes to Vertex rather than to the Gemini API.
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
- Hosted chat transcripts stamp each turn with when it was delivered (`at`,
  on the tool trace's clock), so tool calls sort between the customer's turn
  and the agent's reply instead of after the conversation.
- Hosted voice calls whose agent writes its own tool trace no longer report
  each call twice: the agent's line (real arguments, structured result) takes
  the time from the LiveKit hook's line for the same call.
- Hosted chat simulations are metered again. Since chat calls moved into an
  isolated worker, the worker had no usage reporter, so no chat call was checked
  against the usage limit or recorded as `text_call`. The parent now checks
  before starting the worker and records the simulator tokens the worker
  measured, for repository and Retell chat agents alike.
- Environment releases can attach MCP servers (`servers.json`): each world
  starts them fresh, their tools reach agents as `<id>__<tool>`, and their
  calls are traced like the release's own tools. The first is ALK's read-only
  `sql` server, which reads only the tables a release lists through a reader
  role. Scenarios that use attached tools are proved on PostgreSQL. The clinic
  release attaches it and adds a schedule question only it can answer; for
  agents without that tool, adaptation reports the scenario unscorable.
- The official filesystem MCP server (`@modelcontextprotocol/server-filesystem`
  2026.1.14, MIT) can be attached with a `directory` under the release's
  `attached/`; it is installed in the hosted image from a committed lockfile and
  never downloaded at run time. The clinic attaches its patient guides through
  it (read tools only) and adds a first-visit question only the guide answers.
- Our own agent can run on an environment release: connector `hosted_agent`,
  an `AgentSpec` (instructions and a model) whose digest the binding pins. Its
  bundle serves the world's tools over MCP (`fi.alk.harness.world_mcp`, `/mcp`,
  the release's tools plus attached servers'), and the agent runs in the chat
  worker as a tool loop through Agent Command Center, billed in `text_call`.
  Its evidence is the world's trace, not anything it reports.
- A scenario with no tool calls is graded normally when the harness wrote and
  read the trace itself (our agent, or a request-bound provider run); only an
  unreadable trace is still `evidence_missing`.
- Suite sealing refuses a simulator prompt without `{{ persona }}` and
  `{{ instruction }}` slots. The clinic's had neither, so in chat runs the
  simulated patient never saw the scenario's task; it now has both.
- Environment sealing refuses handler files with no tool, handler modes other
  than `environment` (`customer_source`/`customer_spec` are reserved), attached
  tool names an agent cannot be offered, and an extension that changes its
  parent's `attached/` files. Resealing a sealed directory with a different
  parent or different scenario origins is refused instead of silently turning
  an extension into a root or a derived suite into a base one.
- The PostgreSQL proof also replays every reference call as crashed, as the
  SQLite proof does, so a check that credits an attempted call without its
  effect is refused on both engines. An attached tool counts as changing state
  unless its server declares it read-only; the `sql` server does.
- The clinic's world no longer declares foreign keys, CHECK constraints or a
  partial index, which PostgreSQL seeds dropped anyway; its handlers enforce
  those rules, so both engines grade alike. Its release digests changed.
- Hosted adaptation keeps the scratch PostgreSQL's superuser DSN in
  `.authoring-credentials/` beside the workspace, which the model's file tools
  refuse, instead of in `adaptation/inputs.json`.
- Suite proofs gain a fourth gate on both engines: the untouched world is graded
  against the solution's calls with every successful state-changing call still
  `ok` but without its result (`mutates_state` in `tools.json`). A check that
  holds credits the call rather than its effect, and the scenario is refused.
  The clinic's suite passes unchanged.
- Design workspace (`fi.alk.harness.design`), the first part of the environment
  builder: `open_design` unpacks a root template (environment and suite) into a
  flat workspace, `release_view` materializes the release roots through an
  allow-list, and `seal_design` derives the policy rows, re-saves the world as an
  environment's, refuses schema PostgreSQL seeds would drop or rewrite (foreign
  keys, CHECK, partial and plain indexes, views, triggers, generated columns,
  expression defaults, other types; a template's are listed as inherited), and
  refuses check files that differ from the catalogue, before sealing both
  releases on PostgreSQL.
- Design mode (`python -m fi.alk.harness.design run|hosted`): the builder
  interviews the person one question at a time, and nothing is built until they
  approve `propose_design`. It then builds the world with the world server and
  its own tools (`define_tool`, `change_tool`, `drop_tool`, `add_policy_rule`,
  `change_policy_rule`, `drop_policy_rule`, `detach_server`), writes the suite
  with the scenario tools, and seals with `finish_design`. Each handler is checked
  before it is kept: only allow-listed imports, no builtins that reach outside
  the world, a sample call whose effect matches `mutates_state`, missing required
  arguments refused, and every argument used. A designed suite must grade every
  agent rule and cover exactly the approved use cases. An unanswered question
  parks the design (hosted: the job ends `design_waiting` with the question
  checkpointed), and continuing asks it again and resumes.
- Designs attach MCP servers: `attach_server` validates a `sql` or `filesystem`
  server against the registry, the world's tables and the folder, starts it
  against the design's world and keeps only read-only tools; `write_attached_file`
  writes the documents a filesystem server reads. Every environment's attached
  tools must be read-only (each registered server lists its attachable tools).
  Attached tool names reach the scenario writers (the world summary lists them
  with what they take), the health check and the readiness gates, and a
  scenario whose solution calls one is proved on PostgreSQL when it is
  submitted, off the event loop, against a template database kept per design
  view (`design/proof.py`, sharing the seal's `postgres_verdict`).
- Design documents (`--documents`; hosted under `/work/documents`): each upload
  is parsed in a child process with an allow-listed environment and CPU, memory
  and output limits (PDF with pypdf, XLSX with openpyxl, DOCX with the standard
  library, CSV, TSV, JSON, JSONL, text; oversized zips refused), into text and
  tables the builder reads with `list_documents` and `read_document`.
  `import_rows` inserts a table into the world with each value coerced to its
  column's declared type, and `attach_document` writes a document as markdown
  under `attached/`; both can replace personal details with made-up ones, and
  both are recorded with their source's sha256 in the sealed environment's
  `metadata.provenance`. New `documents` extra, installed in the hosted image.

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
