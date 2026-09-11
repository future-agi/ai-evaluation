# Library Inventory

Agent Learning Kit is the public release home for the Future AGI agent learning
engines. The code is consolidated here under one Python package and one CLI.
`ai-evaluation` remains the active evaluation engine; its Python runtime lives
under `fi.evals`, and its TypeScript SDK source lives under the consolidated
TypeScript package.

## Public Surface

| Surface | Path | Purpose |
| --- | --- | --- |
| Python SDK | [`src/fi/alk`](src/fi/alk) | Public facade for configuration, evaluation, simulation, optimization, red teaming, suites, and release gates. |
| CLI | [`src/fi/alk/cli.py`](src/fi/alk/cli.py) | `agent-learn` command surface for doctor, eval, simulate/run, redteam, optimize, report, release-check, and release-proof. |
| TypeScript SDK | [`typescript/agent-learning-kit`](typescript/agent-learning-kit) | Public TypeScript package published as `@future-agi/agent-learning-kit`, including the migrated `ai-evaluation` TypeScript source. |
| Examples | [`examples`](examples) | Runnable cookbooks and manifests that use the consolidated public package. |

## Engine Code

| Library or engine | Active source path | Runtime namespace | What lives there |
| --- | --- | --- | --- |
| `ai-evaluation` Python runtime | [`src/fi/evals`](src/fi/evals) | `fi.evals` | Active evaluation framework, local evaluators, metrics, guardrails, RAG and structured-output checks, OpenTelemetry evaluation processors, streaming evaluators, and agent report scoring. |
| `ai-evaluation` TypeScript SDK | [`typescript/agent-learning-kit/src`](typescript/agent-learning-kit/src) | `@future-agi/agent-learning-kit` | TypeScript evaluator, local metrics, templates, execution, manager, protect, scanner, streaming, and RAG/heuristic evaluation source. |
| `simulate-sdk` | [`src/fi/simulate`](src/fi/simulate) | `fi.simulate` | Simulation manifests, local/cloud simulation engines, framework adapter probes, LiveKit/local text engines, environment replay, report rendering, recording, suites, and CLI implementation. |
| `agent-opt` | [`src/fi/opt`](src/fi/opt) | `fi.opt` | Optimizer base classes, agent optimizers, mutation/evidence models, simulation integrations, deployment and observability helpers, and optimizer utilities. |

The package build includes both the public facade and the active engine
namespace:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/fi/alk", "src/fi"]
```

That means a built wheel contains the public `fi.alk.*` API and the
engine implementations under `fi.evals`, `fi.simulate`, and `fi.opt`.

## Import Direction

New public code should use the consolidated package:

```python
from fi.alk import configure
from fi.alk import evals, optimize, simulate

configure(api_key="...")
```

Internal compatibility code may still import the engine namespaces directly:

```python
from fi import evals
from fi import opt
from fi import simulate
```

Do not add new release-facing examples that require cloning `ai-evaluation`,
`simulate-sdk`, or `agent-opt` separately. `ai-evaluation` changes required for
v1 should be present in this repository before the Agent Learning Kit release is
called complete. If a fix starts in a separate engine repo, copy the verified
implementation into this repository before treating the public SDK work as done.
The file
the ai-evaluation source inventory (maintained in the internal-docs repo)
records the ai-evaluation source snapshots that were consolidated here, and
`agent-learn release-check` fails if those mapped file paths are missing.

## Reviewer Checklist

When checking whether the migrated engines are present, inspect these paths in
the release branch:

1. [`src/fi/evals`](src/fi/evals) for evaluation and scoring code.
2. [`typescript/agent-learning-kit/src`](typescript/agent-learning-kit/src) for the TypeScript evaluation SDK source.
3. [`src/fi/simulate`](src/fi/simulate) for simulation, framework adapters, and reports.
4. [`src/fi/opt`](src/fi/opt) for optimizer primitives and agent optimizers.
5. [`src/fi/alk`](src/fi/alk) for the public SDK facade that ties them together.
6. [`examples`](examples) for runnable trinity cookbooks.
7. [`pyproject.toml`](pyproject.toml) for package inclusion and the `agent-learn` CLI entry point.

## Migration Rule

`agent-learning-kit` is the source of truth for v1. `ai-evaluation` is not a
legacy dependency for this release; it is the active evaluation engine embedded
in this package. The older `simulate-sdk` and `agent-opt` repositories remain
history and compatibility references during migration, not separate release
requirements for the public v1 package.
