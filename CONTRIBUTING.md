# Contributing

Thanks for helping improve Agent Learning Kit.

This repository contains the public Python SDK, TypeScript SDK, local
simulation/evaluation/optimization engines, examples, and release gates. Keep
changes local-first, deterministic, and backed by executable evidence.

## Development Setup

```bash
uv sync
pnpm --dir typescript install
```

Useful checks:

```bash
uv run ruff check .
uv run pytest -q
uv run python -m build
pnpm --dir typescript --filter @future-agi/agent-learning-kit build
pnpm --dir typescript --filter @future-agi/agent-learning-kit test -- --runInBand
```

For release-candidate changes:

```bash
uv run python -m fi.alk.cli release-proof \
  --project-root . \
  --output /tmp/agent-learning-release-proof.json \
  --quiet
```

## Contribution Guidelines

- Use short branch names that describe the changed surface, for example
  `docs/release-readiness`, `fix/release-proof-timeout`, or
  `feat/framework-adapter-probe`.
- Use imperative commit messages, for example
  `Harden release proof timeout handling`.
- Prefer small, deterministic changes with focused tests.
- Keep public APIs under `fi.alk.*`, `agent-learn`, and
  `@future-agi/agent-learning-kit`.
- Do not add hosted-service requirements to release-gated examples.
- Keep OpenEnv/Gymnasium as compatibility input shapes, not the product center.
- Do not add runtime dependencies unless the workflow cannot be implemented with
  the standard library or an existing dependency.
- Put optional integration dependencies behind extras or dev tooling when
  possible.
- Add or update cookbooks when adding a new user-facing workflow.
- Add release-check or release-proof coverage when a claim becomes part of the
  v1 contract.
- Avoid broad refactors unless they are required for the change.

## Pull Request Checklist

- Explain the user-facing behavior change.
- Link the relevant issue, roadmap item, or release gate when available.
- Include focused tests for the changed behavior.
- Run `uv run ruff check .`.
- Run the smallest relevant pytest target.
- Run full `uv run pytest -q` for shared runtime, CLI, SDK, or release-gate
  changes.
- Run TypeScript build/test for TypeScript package changes.
- Run full `release-proof` for release-candidate changes.
- Update README, examples, or internal docs when the developer workflow changes.

## Licensing

By contributing to this repository, you agree that your contribution is
licensed under the Apache License, Version 2.0.

Do not contribute code or assets unless you have the right to submit them under
Apache-2.0-compatible terms.

This repository does not currently require a separate CLA or DCO sign-off. If
that policy changes, maintainers should update this file before requiring it on
pull requests.
