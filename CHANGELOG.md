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
