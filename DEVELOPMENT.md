# Agent Learning Kit Development Boundary

`agent-learning-kit` is the public SDK and code home for agent simulation,
evaluation, red teaming, and optimization.

All new public SDK work should land here first:

- Public Python imports belong under `fi.alk.*`.
- Public TypeScript package work belongs under `typescript/agent-learning-kit`
  and publishes as `@future-agi/agent-learning-kit`.
- Public CLI commands belong under `agent-learn`.
- Public examples and cookbooks should use `agent-learning-kit` install commands.
- Runtime implementation should live under this repo, either in
  `fi.alk.*` for public APIs or vendored `fi.*` engine packages while
  migration is in progress.
- Shared configuration and keys should flow through `fi.alk.configure()`
  and `AGENT_LEARNING_*` environment variables. Vendored engine aliases
  (`FI_API_KEY`, `FI_SECRET_KEY`, and Future AGI variants) are synced from that
  public config for compatibility only; new public code should not introduce a
  separate key model.

`ai-evaluation` is an active engine for this release, not legacy history. Its
Python runtime must be present under `src/fi/evals`, and its TypeScript SDK
source must be present under `typescript/agent-learning-kit/src`.
`agent-learn release-check` compares those source trees with
the ai-evaluation source inventory (maintained in the internal-docs repo) so missing ai-evaluation
files fail the v1 release gate.

The older `simulate-sdk` and `agent-opt` repositories are source/history during
the migration. New runtime code should be moved into `agent-learning-kit`, not
merely wrapped here. If a fix must first land in an old repo to stabilize an
engine, copy the verified implementation into this repo before treating the
public SDK work as done.

For the current source map, see [LIBRARIES.md](LIBRARIES.md). In short:

- `ai-evaluation` lives under `src/fi/evals`.
- `ai-evaluation` TypeScript source lives under `typescript/agent-learning-kit/src`.
- `simulate-sdk` lives under `src/fi/simulate`.
- `agent-opt` lives under `src/fi/opt`.
- Public Python APIs live under `src/fi/alk`.

When moving an existing surface:

1. Move or add the implementation code under this repository.
2. Add or update the `fi.alk.*` API/CLI.
3. For TypeScript surfaces, add/update the package under
   `typescript/agent-learning-kit` and verify `pnpm --dir typescript --filter
   @future-agi/agent-learning-kit build` plus the package test command.
4. Verify it against real local artifacts and relevant engine tests using this
   repository as the source path.
5. Update public docs/examples to use `agent-learning-kit`.
6. Only then simplify or hide the older engine-level surface.
