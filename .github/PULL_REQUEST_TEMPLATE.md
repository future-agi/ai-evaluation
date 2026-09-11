## Summary

Describe the user-facing change and why it is needed.

## Verification

- [ ] `uv run ruff check .`
- [ ] Focused pytest target:
- [ ] Full `uv run pytest -q` when touching shared runtime, CLI, SDK, or release gates
- [ ] TypeScript build/test when touching `typescript/`
- [ ] `agent-learn release-proof` when touching release-candidate behavior

## Checklist

- [ ] Public API names stay under `agent_learning.*`, `agent-learn`, or `@future-agi/agent-learning-kit`
- [ ] New user-facing workflow has docs or an example
- [ ] New release claim has executable gate coverage
- [ ] No hosted-service dependency was added to release-gated examples
- [ ] Security-sensitive output is redacted or justified
