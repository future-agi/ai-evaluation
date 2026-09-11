---
kind: agent-learning.docs-page.v1
track: reference
backing: []
artifact_kinds: []
commands:
  - agent-learn doctor
postcondition: python -c "import fi.alk; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
opt_in_lane: false
---

# Cookbook Index

> Pick a row (what stage you're at) or a column (what you care about). Every
> cell links real pages; empty cells render as `—`, never invented pages. A
> page may appear in at most two cells.

## The matrix (stage × objective)

| | Behavior | Capability | Reliability | Safety |
| --- | --- | --- | --- | --- |
| **simulate** | [golden-path-run](../quickstart/golden-path-run.md) · [spec-and-runner](../simulate/spec-and-runner.md) · [first-run](../simulate/first-run.md) · [worlds-and-hooks](../simulate/worlds-and-hooks.md) | [simulate-any-framework](../simulate/simulate-any-framework.md) · [memory](../simulate/memory.md) · [multimodal-image](../simulate/multimodal-image.md) · [voice-realtime](../simulate/voice-realtime.md) · [langchain](../frameworks/langchain.md) · [langgraph](../frameworks/langgraph.md) · [llamaindex](../frameworks/llamaindex.md) · [autogen](../frameworks/autogen.md) · [crewai](../frameworks/crewai.md) · [openai-agents](../frameworks/openai-agents.md) · [pydantic-ai](../frameworks/pydantic-ai.md) · [livekit](../frameworks/livekit.md) · [pipecat](../frameworks/pipecat.md) · [browser-use](../frameworks/browser-use.md) · [mcp](../frameworks/mcp.md) · [a2a](../frameworks/a2a.md) · [custom](../frameworks/custom.md) · [openenv](../frameworks/openenv.md) · [persona-scenario-studio](../simulate/persona-scenario-studio.md) | [multi-agent](../simulate/multi-agent.md) · [orchestration](../simulate/orchestration.md) | [golden-path-redteam](../quickstart/golden-path-redteam.md) · [red-team-anything](../redteam/red-team-anything.md) · [corpus](../redteam/corpus.md) · [first-campaign](../redteam/first-campaign.md) · [stored-prompt-injection](../redteam/stored-prompt-injection.md) · [long-horizon](../redteam/long-horizon.md) · [autonomous-task-world](../redteam/autonomous-task-world.md) · [persona-conditioned-attacks](../redteam/persona-conditioned-attacks.md) |
| **evaluate** | [evaluate-any-task](../eval/evaluate-any-task.md) · [eval-suites](../eval/eval-suites.md) · [artifact-evals](../eval/artifact-evals.md) | [task-evidence](../eval/task-evidence.md) · [eval-hooks](../eval/eval-hooks.md) · [benchmark-overview](../eval/benchmark-overview.md) · [benchmark-coding](../eval/benchmark-coding.md) · [benchmark-voice](../eval/benchmark-voice.md) · [benchmark-pull-rl](../eval/benchmark-pull-rl.md) · [benchmark-write-a-suite](../eval/benchmark-write-a-suite.md) | [judge-reliability](../eval/judge-reliability.md) · [persona-fidelity](../simulate/persona-fidelity.md) · [benchmark-command-graded](../eval/benchmark-command-graded.md) | [causal-attribution](../redteam/causal-attribution.md) · [benchmark-sandboxes](../eval/benchmark-sandboxes.md) |
| **optimize** | [golden-path-optimize](../quickstart/golden-path-optimize.md) · [optimize-any-agent](../optimize/optimize-any-agent.md) · [optimization-lifecycle](../optimize/optimization-lifecycle.md) · [behavior-and-collaboration](../optimize/behavior-and-collaboration.md) | [world-model](../optimize/world-model.md) · [memory-targets](../optimize/memory-targets.md) · [workflow-profile-matrix](../optimize/workflow-profile-matrix.md) · [optimizer-profile-matrix](../optimize/optimizer-profile-matrix.md) · [agent-control-plane](../optimize/agent-control-plane.md) | [governance](../optimize/governance.md) · [optimizer-portfolio](../optimize/optimizer-portfolio.md) · [backend-routing](../optimize/backend-routing.md) · [multi-agent-targets](../optimize/multi-agent-targets.md) · [eval-suite-optimization](../optimize/eval-suite-optimization.md) | [society-of-agents](../optimize/society-of-agents.md) · [campaign-optimization](../redteam/campaign-optimization.md) · [adaptive-loop](../redteam/adaptive-loop.md) |
| **promote** | [regression-lifecycle](../simulate/regression-lifecycle.md) | [promote-to-regression](../redteam/promote-to-regression.md) | [golden-path-ci](../quickstart/golden-path-ci.md) · [attack-evolution-shrink](../redteam/attack-evolution-shrink.md) · [capability-profile-freezing](../optimize/capability-profile-freezing.md) | [promote-to-regression](../redteam/promote-to-regression.md) |
| **prove** | [trinity-suite](../prove/trinity-suite.md) | [actions](../prove/actions.md) · [capabilities](../prove/capabilities.md) | [release-check-in-your-ci](../prove/release-check-in-your-ci.md) · [observability](../prove/observability.md) · [benchmark-in-ci](../prove/benchmark-in-ci.md) | [trust-certificates](../prove/trust-certificates.md) |

Notes on placement: red-team pages live in the simulate and evaluate rows
because red-teaming rides the `simulate` and `evals` engines; the promote row
holds the pages whose output is a regression baseline you keep
(`promote-to-regression` appears under both capability and safety — same page,
two reasons to need it).

## By layer (literature vocabulary → kit surface)

| Layer | Pages |
| --- | --- |
| memory | [optimize/memory-targets](../optimize/memory-targets.md) · [simulate/memory](../simulate/memory.md) |
| skills / tools | [prove/actions](../prove/actions.md) · [redteam/first-campaign](../redteam/first-campaign.md) |
| protocol | [frameworks/mcp](../frameworks/mcp.md) · [frameworks/a2a](../frameworks/a2a.md) · [frameworks/openenv](../frameworks/openenv.md) |
| harness | [prove/release-check-in-your-ci](../prove/release-check-in-your-ci.md) · [quickstart/golden-path-ci](../quickstart/golden-path-ci.md) |

## By framework

[LangChain](../frameworks/langchain.md) ·
[LangGraph](../frameworks/langgraph.md) ·
[LlamaIndex](../frameworks/llamaindex.md) ·
[AutoGen](../frameworks/autogen.md) ·
[CrewAI](../frameworks/crewai.md) ·
[OpenAI Agents](../frameworks/openai-agents.md) ·
[PydanticAI](../frameworks/pydantic-ai.md) ·
[LiveKit](../frameworks/livekit.md) ·
[Pipecat](../frameworks/pipecat.md) ·
[Browser Use](../frameworks/browser-use.md) ·
[MCP](../frameworks/mcp.md) ·
[A2A](../frameworks/a2a.md) ·
[custom](../frameworks/custom.md) ·
[OpenEnv](../frameworks/openenv.md) (compatibility input)

## Indexes and reference

These pages carry `backing: []` — they catalog the corpus rather than run it,
so they sit outside the matrix:

- [Landing page](../index.md) — the spine, golden paths, and the compact mirror of this matrix.
- This page — the full 2-axis index.
- [reference/artifacts.md](../reference/artifacts.md) — the closed artifact-kind catalog.
- [reference/cli.md](../reference/cli.md) — the `agent-learn` command surface.
- [reference/configure.md](../reference/configure.md) — API key semantics, offline vs platform.

## How to read a page

Every cookbook page follows one skeleton: frontmatter (the manifest twin),
what you are testing, run it (CLI and SDK for the same operation), what you
built (the machine-checkable postcondition), when it fails (symptom → layer →
doctor check), and the next spine step. The frontmatter declares the backing
example under `examples/` and the artifact kind the page emits; the
`docs_executability` release gate re-verifies both on every release, so a page
whose runnable twin breaks cannot ship. Pages flagged `opt_in_lane: true`
(voice and live-provider sessions) are not on the golden path and link the
follow-up phase that covers their live infrastructure.
