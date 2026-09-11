<p align="center">
  <img src="docs/assets/futureagi-mark-email.png" alt="Future AGI" width="72" />
</p>

<h1 align="center">Agent Learning Kit</h1>

<p align="center">
  Local-first testing, simulation, red teaming, and optimization for AI agents.
</p>

<p align="center">
  <a href="LICENSE">Apache-2.0</a>
  ·
  <a href="docs/index.md">Docs</a>
  ·
  <a href="CONTRIBUTING.md">Contributing</a>
  ·
  <a href="SECURITY.md">Security</a>
  ·
  <a href="ROADMAP.md">V1 roadmap</a>
  ·
  <a href="LIBRARIES.md">Library inventory</a>
</p>

![Agent Learning lifecycle blueprint](docs/assets/hero-agent-blueprint.jpg)

Agent Learning Kit is the local-first SDK and CLI for testing, simulating,
red-teaming, and optimizing AI agents.

It brings the three core Future AGI engines into one public developer surface —
three engines, four workflows: red-teaming rides on the `simulate` and `evals`
engines rather than being a fourth engine:

- `simulate`: run local worlds, tasks, framework-shaped adapters, replays, and
  regression artifacts.
- `evals`: evaluate prompts, task outputs, runtime contracts, traces, memory,
  retrieval, safety, and robustness evidence.
- `optimize`: search over prompts, agents, framework adapters, worlds,
  multi-agent interactions, memory layers, workflows, and red-team scenarios.

Use it when you want one reproducible loop:

1. Simulate an agent or framework workflow.
2. Evaluate the behavior and runtime evidence.
3. Optimize the weak layer.
4. Promote the result into a replayable artifact.
5. Prove release readiness with local gates.

### The harness: point it at an agent and talk to it

`src/fi/alk/harness/` builds all of the above **for** an agent instead of asking you to write it.
Point it at an agent's source and it reads what that agent verifiably is, builds a real world its
tools act on, and writes test scenarios that are each proved before they are kept. It is driven
as a conversation, in a terminal or on a web page.

- **[Start here](src/fi/alk/harness/README.md)**: setup from nothing, then how to use it
- **[The web page](harness-ui/README.md)**: the same harness as a chat, on `localhost:8777`
- **[How it works](src/fi/alk/harness/HOW-IT-WORKS.md)** and
  **[why it is shaped this way](src/fi/alk/harness/DESIGN.md)**

OpenEnv/Gymnasium shapes are compatibility inputs, not the product center.
Agent Learning Kit is the primary runtime and release contract, and the bar is
the executable `environment_10x_robustness` release gate.
OpenEnv/Gymnasium-shaped traces remain compatibility evidence inside that bar.

## Install

PyPI and npm publishing land at the v1 launch. Today, install from source:

```bash
git clone https://github.com/future-agi/agent-learning-kit
cd agent-learning-kit
pip install -e .
```

(or `uv sync` for contributors)

At launch:

```bash
pip install agent-learning-kit
```

Optional Python extras:

```bash
pip install "agent-learning-kit[livekit]"
pip install "agent-learning-kit[nli]"
pip install "agent-learning-kit[all]"
```

TypeScript evaluation package (npm at launch; today build from
[`typescript/agent-learning-kit`](typescript/agent-learning-kit)):

```bash
pnpm add @future-agi/agent-learning-kit
```

## Quickstart

Everything below runs fully offline — no API key, no network. Start with the
local doctor:

```bash
agent-learn doctor
```

Then run the golden path against the bundled example manifests. The
`AGENT_LEARNING_*_EXAMPLE_KEY` prefixes satisfy each manifest's
`required_env` list — that list is CI wiring metadata, not a provider
credential, so any placeholder value works.

> Prefer the SDK spine over the CLI?
> [Spec + Runner](docs/simulate/spec-and-runner.md) runs the same simulation as
> one `SimulationSpec` fed to one `SimulationRunner` — the plug-and-play surface
> behind every simulation.

Evaluate a suite:

```bash
agent-learn eval examples/eval_suite.json \
  --output artifacts/eval.json
```

Simulate a run manifest:

```bash
AGENT_LEARNING_RUN_EXAMPLE_KEY=offline-demo-key \
  agent-learn run examples/run_manifest.json \
  --no-eval \
  --output artifacts/run.json
```

Optimize an agent workflow:

```bash
AGENT_LEARNING_OPTIMIZE_EXAMPLE_KEY=offline-demo-key \
  agent-learn optimize examples/optimization_manifest.json \
  --output artifacts/optimization.json
```

Run a red-team campaign:

```bash
AGENT_LEARNING_REDTEAM_EXAMPLE_KEY=offline-demo-key \
  agent-learn redteam examples/redteam_manifest.json \
  --output artifacts/redteam.json
```

Each command prints a `wrote <path>` line; relative `--output` paths resolve
against your current working directory.

Optional platform mode: to use Future AGI platform-backed evaluation, set
`AGENT_LEARNING_API_KEY` (it takes precedence over the `FUTURE_AGI_API_KEY`
and `FI_API_KEY` aliases), or call `configure(api_key="...")` from
`fi.alk`. See
[docs/reference/configure.md](docs/reference/configure.md).

Cut local release proof:

```bash
agent-learn release-check --project-root .
agent-learn release-proof \
  --project-root . \
  --output /tmp/agent-learning-release-proof.json \
  --quiet
```

## TypeScript

```typescript
import { Evaluator } from "@future-agi/agent-learning-kit";
import { LocalEvaluator } from "@future-agi/agent-learning-kit/evals/local";
```

## What You Can Build

- Prompt and response evaluations.
- Local task and world simulations.
- Framework adapter probes (probe-promoted coverage) for LangChain, LangGraph,
  LlamaIndex, AutoGen, CrewAI, LiveKit, Pipecat, Browser Use, MCP, A2A, and
  custom orchestration objects.
- Runtime-simulated coverage for PydanticAI (multi-framework runtime
  simulation) and OpenAI Agents (handoff-transcript promotion).
- Runtime-contract and trace-quality checks.
- Multi-agent coordination and handoff tests.
- Retrieval and memory quality checks.
- Voice, realtime, browser/CUA, workflow, lifecycle, and protocol traces.
- Red-team corpus, campaign, adaptive-loop, and persistent-state checks.
- Optimizer governance, candidate lineage, rollback, and release proof.

## Why It Exists

Most agent stacks split testing, simulation, optimization, and safety review
across separate tools. Agent Learning Kit keeps those steps in one artifact
model so a developer can inspect what happened, score it, improve it, and replay
it in CI.

The public SDK is `agent-learning-kit`, the Python namespace is
`fi.alk`, the CLI is `agent-learn`, and the TypeScript package is
`@future-agi/agent-learning-kit`.

The active `ai-evaluation` code is included here under `src/fi/evals`, with its
TypeScript SDK source under `typescript/agent-learning-kit/src`. The
`simulate-sdk` and `agent-opt` engine code is included under `src/fi/simulate`
and `src/fi/opt`. See [LIBRARIES.md](LIBRARIES.md) for the complete source map.
The ai-evaluation source inventory used by `agent-learn release-check` lives at
the ai-evaluation source inventory (maintained in the internal-docs repo).

## Repository Map

- [`examples/`](examples): runnable cookbooks and manifests.
- [`src/fi/alk`](src/fi/alk): public Python SDK facade and CLI.
- [`src/fi/evals`](src/fi/evals): active `ai-evaluation` engine code.
- [`src/fi/simulate`](src/fi/simulate): migrated `simulate-sdk` engine code.
- [`src/fi/opt`](src/fi/opt): migrated `agent-opt` engine code.
- [`typescript/agent-learning-kit`](typescript/agent-learning-kit): public
  TypeScript package, including the active evaluation SDK source.
- [`docs/index.md`](docs/index.md): full documentation index.
- [`ROADMAP.md`](ROADMAP.md): public v1 roadmap and post-v1 extensions.
- [`LIBRARIES.md`](LIBRARIES.md): source map for the consolidated engines.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): local development and PR workflow.
- [`SECURITY.md`](SECURITY.md): vulnerability reporting policy.
- [`LICENSE`](LICENSE): Apache-2.0 license.
- [`NOTICE`](NOTICE): Apache notice metadata.

## Development

New public SDK development belongs here. See [DEVELOPMENT.md](DEVELOPMENT.md)
for the boundary between this package and the backing engine repos.

```bash
uv sync
uv run ruff check .
uv run pytest -q
uv run python -m build
pnpm --dir typescript --filter @future-agi/agent-learning-kit build
pnpm --dir typescript --filter @future-agi/agent-learning-kit test -- --runInBand
```

For the heavier release cut, run `agent-learn release-proof --project-root .`.
It emits `agent-learning.release-proof.v1` with command evidence for the full
local proof stack.

Before a release:

```bash
uv run python -m fi.alk.cli release-proof \
  --project-root . \
  --output /tmp/agent-learning-release-proof.json \
  --quiet
```

`release-proof` includes release-check, full-repo ruff, pytest, Python package
build, TypeScript package build/test, and `git diff --check`. Use
`--only <check>` for partial proof during development or `--dry-run` to emit the
exact command plan without executing commands.

## Project Status

The v1 release gate is local-first and executable. It covers SDK consolidation,
promptfoo-style CLI usage, native optimizer evidence, docs/examples, schema
kinds, packaging metadata, red-team corpus/campaign coverage, Future AGI
UI/action/report artifacts, framework/provider compatibility, environment
robustness, regression replay, and release proof.

All v1 gates are green on the proved release commit (see the release-proof
artifact). Roadmap milestones marked "mostly complete" or "in progress" are
extend-only: the v1 contract those gates assert is frozen and proved; the named
extensions land post-v1 without weakening any gate.

## Community

- Contributions: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Security reports: [SECURITY.md](SECURITY.md)
- License: [Apache-2.0](LICENSE)

## Deep Dive

The full documentation set — quickstarts, per-track guides, framework pages,
and reference material — starts at [docs/index.md](docs/index.md).
