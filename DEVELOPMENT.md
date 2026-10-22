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
# Release pipeline

One GitHub Actions workflow, `Release` (`.github/workflows/publish-pypi.yml`), runs
on every push to `main`, including merges. What a push does depends on whether the
`version` in `pyproject.toml` is already on PyPI:

- Already there: the push is ordinary code, so the workflow opens the next
  version-bump pull request to `main` as `futureagi-release-bot`, from the branch
  `chore/bump-<next version>`, or leaves the one that is already open alone. The bump
  is a patch bump: `uv version --bump patch --frozen` changes the one line in
  `pyproject.toml`, and the workflow changes the matching line in `uv.lock`.
- Not there yet: the push is the release. The workflow publishes the SDK to PyPI,
  builds and certifies the hosted E2B template, and opens a pull request in
  `future-agi/deployment` that pins the new template.

To release, merge the open `chore(release): bump ...` pull request, then approve the
`pypi` and `e2b-template` deployments. The SDK upload and the template build each wait
for their own approval and run side by side; neither waits for the other. Pull
requests that change the release inputs run credential-free validation only.

Keep the file name `publish-pypi.yml`. PyPI accepts uploads only from the trusted
publisher registered for this repository, that workflow file name, and the `pypi`
environment; change the publisher on pypi.org before renaming the file.

One-time setup:

- `E2B_API_KEY`, a repository Actions secret holding the E2B team's API key.
- `RELEASE_BOT_APP_ID` and `RELEASE_BOT_PRIVATE_KEY`, the organization Actions
  secrets of the release bot GitHub App, shared with this repository. The app must
  be installed on this repository and on `future-agi/deployment`, with write access to
  contents and pull requests on both.
- The PyPI trusted publisher above.
- Two environments, each with required reviewers and deployments limited to `main`:
  `pypi` for the SDK upload and `e2b-template` for the template build. An
  environment without reviewers approves itself, so keep reviewers on both. Do not
  reuse `pypi` for other jobs: PyPI trusts uploads from that environment.
- GHCR: the workflow uses `GITHUB_TOKEN` with `packages: write` to push to
  `ghcr.io/future-agi/agent-learning-kit/alk-hosted-runtime`, and supplies that
  short-lived credential to the E2B builder for the image import. No Docker Hub
  credential is needed. Organization policy must permit the repository to create
  GHCR packages; if the package already exists, grant this repository Actions access
  in its package settings. See GitHub's
  [Container registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

## SDK

The SDK is uploaded only when the `version` in `pyproject.toml` is not on PyPI yet.
To release: merge the open bump pull request, then approve the `pypi` environment
deployment. The upload runs on pushes only, never on a manual run.

For a minor, major or pre-release version, close the bump pull request, delete its
branch, and open one of your own to `main` that runs `uv version --bump minor` (or
`--bump major`). Its version is not on PyPI either, so merging it releases the same
way.

## E2B template

The template is built only for a release push or a manual run on `main`; an ordinary
push builds nothing and asks for no approval. The build starts only after a reviewer
approves the `e2b-template` environment deployment for that run; the one approval
also covers the deployment pull request that follows. A run nobody approves builds
nothing.

The workflow runs `scripts/e2b-template.py` against the checked-out commit. It builds
`Dockerfile.hosted` for Linux amd64, pushes a uniquely tagged image, imports its
immutable digest into `alk-hosted-<12-character commit SHA>`, and certifies the
capabilities in `hosted-snapshot/catalog.json` in a temporary sandbox. The existing
publisher defaults apply: 4 CPUs, 8192 MB RAM, and at least 10 GB verified disk.
The publisher cleans up its certification sandbox when the check finishes.

A successful run exposes the immutable `template-name:build-id` reference in the
job summary and saves `e2b-template-release.json` as a 90-day workflow artifact.
That file includes the source commit, image digest, resource sizes, and certification
results. A failed certification fails the job and produces no certified release
artifact; its image/template may already exist, so use only successful releases.

Each commit has its own concurrency group, so a newer merge does not cancel or
replace an older commit's publication. Reruns of the same commit are serialized.
Reruns create a new image tag and E2B build; always use the immutable reference
from the desired run. To build a template without a merge, use
**Actions → Release → Run workflow**, select `main`, and approve the deployment.

## Deployment pull request

After a template is certified, the workflow opens a pull request, with no further
approval, against `main` of
`future-agi/deployment` as `futureagi-release-bot`, on the branch
`chore/harness-pin-<template name>`. `scripts/bump_deployment_values.py` edits
`us/gcp/deployment/values.yaml` and `eu/gcp/deployment/values.yaml`: every
`ALK_E2B_TEMPLATE_REFERENCE`, `ALK_E2B_TEMPLATE_BUILD_ID`,
`ALK_E2B_TEMPLATE_CPU_UNITS`, `ALK_E2B_TEMPLATE_MEMORY_MB` and
`ALK_E2B_TEMPLATE_DISK_GB` is set from the certified release, and where
`HARNESS_PARALLEL_SNAPSHOT_DIGESTS` lists the previous build ID it is swapped for the
new one. Where `ALK_HOSTED_OBSERVABILITY_EXPORT_SNAPSHOTS` lists the previous template
reference it is swapped for the new one; the setting is never added where it is absent.
No other line changes.

The workflow never merges the pull request and never moves a production alias; a
person reviews and merges it. If a setting is missing from a values file, appears a
different number of times than the others, or is written in a form the script cannot
rewrite safely, the job fails and neither file is changed. No certified template
means no pull request. A rerun for the same commit updates the same pull request.
Pull requests opened for older commits are not closed automatically.

## Known behaviours

- An open bump pull request is left as it is on later pushes to `main`. It is never
  refreshed or force-pushed, and it still merges into the latest `main`. A stale one
  is not closed automatically.
- If the version check (`sdk-check`) fails on a release push, the template is not
  built either. Use "re-run failed jobs" to recover both.
- A full re-run of a release run after the SDK is on PyPI skips the template and opens
  the next bump pull request early. Use "re-run failed jobs" or a manual run instead.
- If the bump branch exists without an open pull request (the pull request was
  closed, or opening it failed), the job fails and says so. Delete the branch, or
  reopen the pull request, and re-run the job.
- A push that lands after the bump is merged but before the SDK upload has finished
  also counts as a release, because its version is not on PyPI yet: it asks for both
  approvals again and opens no bump pull request. The next push to `main` opens it.
- The bump lands on `main` only. A branch that also changes the version conflicts
  with it on that line; merge `main` into that branch first.
