# Agent Learning Kit — Roadmap

> What is implemented today, what is planned next. Every "implemented" claim
> below is backed by an executable release gate (`agent-learn release-check`)
> and a passed full release proof (`agent-learning.release-proof.v1`) — the
> kit's rule is that no capability claim ships without a gate that proves it.
> The per-gate map is maintained internally; every claim here is enforced by `agent-learn release-check`.

Status date: 2026-06-22. Release candidate: tag `v1.0.0-rc.1`
(Python `agent-learning-kit==0.1.0`, TypeScript `@future-agi/agent-learning-kit==0.2.0`).

---

## Implemented

### Core: one SDK, one CLI, three engines

- Single public surface: `fi.alk` (Python), `agent-learn` (CLI),
  `@future-agi/agent-learning-kit` (TypeScript, evaluation-focused).
- Three engines, four workflows — `simulate`, `evals`, `optimize`, with
  red-teaming riding on simulate + evals.
- One key: `AGENT_LEARNING_API_KEY` (with `FUTURE_AGI_API_KEY` / `FI_API_KEY`
  aliases). Fully offline by default — no credential is required for any
  local workflow, golden path, or release gate.
- A suite of executable release gates behind `agent-learn release-check` (run
  the command for the authoritative count); the heavier `agent-learn
  release-proof` runs gates + full test suites + package builds and emits a
  verifiable proof artifact.

### Evaluation (evaluate any task)

- Eval suites (promptfoo-style JSON manifests), saved-artifact evaluation,
  raw task-evidence evaluation, and deterministic evaluation-config synthesis
  (criteria/tools/weights inferred from evidence — no hosted judge required).
- Localhost-default evaluation hooks for custom judges (non-local endpoints
  are explicit opt-in).
- Judge-reliability tooling: perturbation checks (formatting / verbosity /
  paraphrase) over scripted judges, taught as a first-class cookbook.
- JSON / JUnit / SARIF / Markdown outputs on every eval surface.

### Simulation (simulate any framework)

- Local worlds, task simulations, world hooks, stateful tool worlds, memory
  layers, multi-agent rooms, orchestration stacks, realtime/voice fixtures,
  browser/CUA traces, multimodal image runs.
- Framework adapters — probe → discover → optimize → promote — for LangChain,
  LangGraph, LlamaIndex, AutoGen, CrewAI, LiveKit, Pipecat, Browser Use, MCP,
  A2A, and custom orchestration objects (probe-promoted coverage); PydanticAI
  and OpenAI Agents via runtime simulation (runtime-simulated coverage).
- OpenEnv/Gymnasium shapes consumed as compatibility inputs (wire format
  only); environment replay is the owned surface, enforced by gate.
- Regression lifecycle: baseline → compare → report → promote-to-regression →
  replay → shrink, all CLI-first and CI-ready.

### Live framework lanes (opt-in, never release prerequisites)

- Real framework processes under one harness contract, behind per-lane env
  flags and extras: LiveKit `AgentSession`, Pipecat `Pipeline`, LangChain/
  LangGraph compiled graphs with real checkpoint stores (including
  cross-session stored-prompt-injection probes), loopback MCP server
  processes, A2A HTTP peers. All five lanes proven against the real
  frameworks.
- Evidence classes (`local_gate` / `live_lane` / `live_stressed` /
  `captured_fixture`) keep live results out of release claims; the
  `live_lane_boundary` gate enforces the boundary statically.
- Untrusted-subprocess isolation with scrubbed env (harness credentials never
  reach lane processes); layer-attributed failures (lane infra never scores
  the agent); n-repeat variance statistics (ICC, divergence step) with
  pass / fail / unstable verdicts; replayable transcripts demotable into
  credential-free regression fixtures with reviewed provenance.

### Red-teaming

- Canonical research-backed corpus and campaign execution, adaptive loops,
  attack evolution with counterexample shrinking, persistent-state /
  cross-session stored-injection scenarios, long-horizon campaigns, causal
  attribution, society-driven scenario optimization, readiness certification,
  and promotion of findings into replayable regression packs.

### Optimization (optimize the whole agent, not just the prompt)

- Path-exact `optimize_target()` family proven across surfaces: world
  transitions, framework adapter method, memory operations, multi-agent
  roster, orchestration spans, workflow traces, adapter matrices.
- Whole-agent search: `base_agent` + `search_space` over model, voice, first
  message, instructions, tools, memory policy, and topology paths — staged
  conditioning (component text → structure/config → global re-polish),
  diagnosis-scoped search locality with harness-layer attribution, declared
  eval budgets with opt-in Elo tournament selection, external-verification-
  only ranking, and an apply-plan artifact for provider application with
  read-back verification (execution of the apply stays platform-side).
- Optimizer profile matrix: 33 declared (framework × target-kind × backend)
  cells, per-cell winners only — the gate rejects any "globally best backend"
  aggregate by construction.
- Capability-profile regression freezing: promoted profiles become frozen,
  content-addressed evidence rows; an optimization win that breaks any frozen
  row is vetoed; security rows are non-tradable.
- Trajectory-profiled backend routing: every backend run emits a fitness
  profile (improvement frequency, locality, dedupe, regressions); routing
  recommendations cite that evidence, with cold-start fallback and explicit
  override (`--backend`).
- Society-of-agents governance: deterministic role graph with asymmetric
  authority, two-chamber (samiti/sabhā) rounds, guṇa temperament parameters
  on roles, structured pañca-avayava proposal justifications, fallacy-class
  (hetvābhāsa) rejection records, pooled diagnosis ledger, full audit trail.

### Benchmark harness (run a benchmark against any agent, any modality)

- One harness surface — `fi.alk.bench.run_bench(...)` / `agent-learn
  bench` — over a fixed **Task↔Verifier** contract with a pluggable Environment
  and Agent-adapter, emitting a unified `Result` (`scalar` / `components` /
  `pass_fail` / `explanation`) every modality projects into.
- Three control modes, all live: **push** (the harness drives the agent through a
  world — text / tool), **artifact-in** (score a submitted artifact against a
  held-out oracle, no live agent — coding + voice), and **pull** (the agent drives
  a simulated environment via reset/step — RL/Gym/OpenEnv shape).
- **Coding** — two tiers. A convenience `check_*` tier (trusted/accidental-gaming
  only), and a hardened **command/artifact-graded** tier: the candidate produces
  files/output, a HELD-OUT grader runs *after* (candidate processes killed) and
  emits the verdict via its exit code + a grader-owned reward file — never parsed
  from candidate stdout. This structurally defeats verdict-forgery and
  oracle-reads, and is **multi-language** (Python, bash, … — the candidate and
  grader are arbitrary commands). Two sandboxes: credential-free **subprocess**
  (the gate runs it on trusted reference code, no Docker) and an opt-in hardened
  **Docker** lane for untrusted output (`--network none`, read-only rootfs,
  `--cap-drop ALL`, no-new-privileges, nosuid tmpfs, non-root, CPU/mem/PID caps,
  per-task ephemeral containers, hard timeout).
- **Pull / RL** — a deterministic, credential-free environment registry
  (navigation, search) with a reference policy; the agent (a policy callable or
  spec) steps it to a reward. A live external env server (HTTP step/reset) is the
  same contract with a transport — deferred to owner infra.
- **Voice** — a deterministic voice-episode verifier scoring a transcript on
  latency, turn-taking, barge-in handling, and content (pass only if every
  dimension meets its floor). Live audio/SIP/WebRTC + real WER capture is deferred
  to owner infra and plugs in by producing the same transcript shape.
- Trustworthy by construction: the oracle is held out; a fake-success no-op (or a
  forged-stdout candidate) fails; results are deterministic; Docker runs are
  stamped `evidence_class=live_lane` (never mislabeled a fixture); infra failures
  (no Docker daemon) are honest `void`, never a 0% score. Enforced by the
  `bench_contract_readiness` gate, whose every failure mode is itself tested.

### Run telemetry & dashboard (Weights & Biases / promptfoo–style)

- Every workflow — `run_benchmark`, `optimize_against_dataset`,
  `improve_agent_code` — records a run summary locally by default: a
  deterministic ledger plus a one-line stderr digest, with **zero credentials
  required**.
- When an API key is present and sync is enabled (`AGENT_LEARNING_SYNC=auto`,
  the default), the same run additionally emits a trace to the Future AGI
  dashboard and prints a deep-link URL to that run; with no key — or
  `AGENT_LEARNING_SYNC=local` — it stays fully local. No workflow ever blocks
  on the network.
- Honest sync reporting: a run is reported `synced` only on an *observed*
  successful export; auth/transport failures surface as `export_failed`
  instead of a false success.
- The `telemetry_boundary` gate enforces the contract on every release check:
  local-default, payload redaction, evidence-class honesty, and no unsolicited
  network emission or stdout side-channel during gated runs.

### Developer experience

- Born-executable docs pages across every track: each page opens with a
  YAML-frontmatter "manifest twin" backed by a CI-executed example; the
  `docs_executability` gate re-verifies backing, claims, and the generated
  `docs/llms.txt` machine index on every release check — docs cannot rot.
- `agent-learn init` golden paths: all five presets (`run`, `redteam`, `ci`,
  `optimize`, `all`) reach a first replayable artifact in ≤3 commands,
  offline, with machine-checkable postconditions and doctor mappings in every
  scaffold README.
- Bring-your-own framework examples (`examples/frameworks/`): runnable
  end-to-end loops (evaluate → simulate → optimize → code-level self-improve)
  for 16 agent frameworks plus a synthetic third-party agent, with a
  bring-your-own guide. These run live against your own provider keys and are
  examples — not release prerequisites; the certified credential-free adapter
  surface is the gated one above.
- Packaging hygiene: the sdist ships only source, tests, examples, docs, and
  standard release files — enforced by gate; `pip install -e .` from source
  today, PyPI/npm at launch.

---

## Planned

### Near term (current program)

- **Benchmark harness extensions** — push / artifact-in / pull modes, the
  hardened multi-language coding lane (subprocess + Docker), the pull/RL env
  registry, and the voice transcript verifier are all implemented + gated. What
  remains: a born-executable docs/cookbook page for the harness; and the
  owner-infra live tiers (a live external env server for pull, and live
  audio/SIP/WebRTC + WER capture for voice) — both already contract-compatible.
- **Voice lane rungs 2–3** — loopback real-transport audio (WebRTC/WS over
  localhost) and real telephony/SIP for the LiveKit/Pipecat lanes, with
  dual-channel barge-in/overlap evidence. Rung 1 (virtual-clock simulated
  user) is implemented; higher rungs currently raise `NotImplementedError`
  by design rather than pretending.
- **Credentialed lane runs** — owner-keyed runs for LiveKit Cloud/SIP and
  provider-applied whole-agent optimization (ElevenLabs-style apply with
  read-back), producing the first reviewed captured fixtures.
- **Live red-team targets** — pointing the persona/corpus generators at live
  lane targets, including repo-conditioned test generation.
- **Platform artifact surface** — Future AGI UI rendering and acting on kit
  artifacts (report cards, action cards, run/red-team/optimization pages),
  with the platform consuming apply-plan artifacts.
- **TypeScript parity** — simulate/optimize/red-team surfaces in the TS
  package (currently evaluation-focused), plus npm publish readiness.

### Release cut (owner actions)

- Security-contact address in `SECURITY.md`; push, tag publication, and
  PyPI/npm publishing from the proved commit.

### Post-v1 queue

- Split the release-gate registry (`trinity.py`) into a `trinity/gates/`
  package (internal refactor; no behavior change).
- Additional framework/provider adapter promotions as the ecosystem moves;
  more per-framework optimizer profile matrix cells beyond the declared 33.
- Generated notebook views of cookbook pages (docs remain script-backed; the
  executability gate stays the source of truth).
- Meta-optimization of society parameters (guṇa mix as an optimizable
  meta-parameter) and live-lane-evidence-informed routing once captured
  fixtures accumulate.

---

## How to verify any claim on this page

```bash
uv run agent-learn release-check --project-root .   # every release gate
uv run agent-learn release-proof --project-root . \
  --output /tmp/proof.json --quiet                  # full proof artifact
```

If a claim here ever drifts from what those commands prove, the commands win.
