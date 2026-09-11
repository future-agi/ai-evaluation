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
  - public_boundary_passed
opt_in_lane: false
---

# Agent Learning Kit Documentation

> Local-first testing, simulation, red teaming, and optimization for AI agents.
> Every cookbook below is a thin narrative over a runnable object in
> [`examples/`](../examples) and is admitted by the `docs_executability` release
> gate — the YAML frontmatter at the top of each page is its manifest twin.

## The spine

One loop, taught once:

```
simulate ──▶ evaluate ──▶ optimize ──▶ promote ──▶ prove
```

Each stage opens with a task guide; red-teaming is the fourth workflow, not a
fourth engine — it rides on the `simulate` and `evals` engines, so its pages
sit in the simulate and evaluate rows of the index below.

| Stage | Start here |
| --- | --- |
| Simulate | [Simulate any framework](simulate/simulate-any-framework.md) |
| Evaluate | [Evaluate any task](eval/evaluate-any-task.md) |
| Optimize | [Optimize any agent](optimize/optimize-any-agent.md) |
| Red-team | [Red-team anything](redteam/red-team-anything.md) |
| Promote · Prove | [Regression lifecycle](simulate/regression-lifecycle.md) · [Trinity suite](prove/trinity-suite.md) |

## Pick a cookbook (stage × objective)

Rows are the spine; columns are what you care about. This is the compact
mirror — the full 70-page index (every Markdown page under `docs/`, the
same set `docs/llms.txt` lists) is [cookbooks/index.md](cookbooks/index.md).

| | Behavior | Capability | Reliability | Safety |
| --- | --- | --- | --- | --- |
| **simulate** | [spec-and-runner](simulate/spec-and-runner.md) · [first-run](simulate/first-run.md) · [worlds-and-hooks](simulate/worlds-and-hooks.md) | [simulate-any-framework](simulate/simulate-any-framework.md) · 14 [framework pages](cookbooks/index.md#by-framework) | [multi-agent](simulate/multi-agent.md) · [orchestration](simulate/orchestration.md) | [first-campaign](redteam/first-campaign.md) · [stored-prompt-injection](redteam/stored-prompt-injection.md) |
| **evaluate** | [eval-suites](eval/eval-suites.md) · [artifact-evals](eval/artifact-evals.md) | [task-evidence](eval/task-evidence.md) | [judge-reliability](eval/judge-reliability.md) | [causal-attribution](redteam/causal-attribution.md) |
| **optimize** | [optimization-lifecycle](optimize/optimization-lifecycle.md) | [workflow-profile-matrix](optimize/workflow-profile-matrix.md) | [governance](optimize/governance.md) · [optimizer-portfolio](optimize/optimizer-portfolio.md) | [society-of-agents](optimize/society-of-agents.md) |
| **promote** | [regression-lifecycle](simulate/regression-lifecycle.md) | [promote-to-regression](redteam/promote-to-regression.md) | [attack-evolution-shrink](redteam/attack-evolution-shrink.md) | [promote-to-regression](redteam/promote-to-regression.md) |
| **prove** | [trinity-suite](prove/trinity-suite.md) | [capabilities](prove/capabilities.md) | [release-check-in-your-ci](prove/release-check-in-your-ci.md) | [trust-certificates](prove/trust-certificates.md) |

## Golden paths

First replayable artifact in three commands or fewer, offline — no API keys,
no provider accounts:

```bash
agent-learn init . --preset run --quiet
agent-learn run manifests/run.json --output artifacts/run.json
python -c "import json; assert json.load(open('artifacts/run.json'))['kind']=='agent-learning.run.v1'; print('ok')"
```

- [Golden path: run](quickstart/golden-path-run.md) — scaffold, simulate, verify the artifact.
- [Golden path: red-team](quickstart/golden-path-redteam.md) — scaffold, campaign, verify the findings.
- [Golden path: optimize](quickstart/golden-path-optimize.md) — scaffold, search, verify the lineage.
- [Golden path: CI](quickstart/golden-path-ci.md) — run → redteam → replay, the loop your pipeline keeps.

## Layer vocabulary

If you arrive from the agent-infrastructure literature (memory, skills/tools,
protocols, harness engineering), this maps that vocabulary onto kit surfaces:

| Layer | Kit pages |
| --- | --- |
| memory | [optimize/memory-targets](optimize/memory-targets.md) · [simulate/memory](simulate/memory.md) |
| skills / tools | [prove/actions](prove/actions.md) · [redteam/first-campaign](redteam/first-campaign.md) |
| protocol | [frameworks/mcp](frameworks/mcp.md) · [frameworks/a2a](frameworks/a2a.md) · [frameworks/openenv](frameworks/openenv.md) |
| harness | [prove/release-check-in-your-ci](prove/release-check-in-your-ci.md) · [quickstart pages](quickstart/golden-path-ci.md) |

## How these docs stay honest

Every page's frontmatter names its backing example, the artifact kind it
emits, and the postcondition that checks the result. `agent-learn
release-check` re-verifies all of it on every release: a page whose backing
object stops running cannot ship. Superlative phrasing is linted the same way —
a claim appears in prose only when a named release gate licenses it in the
same run. The artifact contract is cataloged in
[reference/artifacts.md](reference/artifacts.md); the release proof objects are
described in the README's [Project Status](../README.md#project-status)
section.

Framework coverage wording, copied verbatim from the [README](../README.md):

> - Framework adapter probes (probe-promoted coverage) for LangChain, LangGraph,
>   LlamaIndex, AutoGen, CrewAI, LiveKit, Pipecat, Browser Use, MCP, A2A, and
>   custom orchestration objects.
> - Runtime-simulated coverage for PydanticAI (multi-framework runtime
>   simulation) and OpenAI Agents (handoff-transcript promotion).
