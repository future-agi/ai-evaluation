---
kind: agent-learning.docs-page.v1
track: eval
objective: capability
stage: evaluate
backing:
  - examples/bench_pull_rl.py
artifact_kinds: []
commands:
  - python examples/bench_pull_rl.py artifacts/bench-pull-rl.json
  - agent-learn bench examples/bench_suites/pull_starter.json --mode pull --agent '{"type":"reference"}'
postcondition: python -c "import json; p=json.load(open('artifacts/bench-pull-rl.json')); a=p['aggregate']; assert a['pass_rate']==1.0, p; assert a['scored']==2, p; assert p['modalities']==['rl'], p; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
opt_in_lane: false
---

# Pull / RL benchmark: the agent drives a live environment via reset/step

> **Twin:** [`examples/bench_pull_rl.py`](../../examples/bench_pull_rl.py)
> · `pull` control mode · offline, no credentials, no network.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The `push` lane has the harness drive the agent through a world; `artifact_in`
scores a submitted artifact with no live agent. **Pull** inverts control: the
agent is a *policy* — a callable `obs -> action` — that steps an environment via
`reset` / `step` until the episode is done, and the score is the environment's
**cumulative reward**. This is the Gym / OpenEnv environment *shape*, run live
(reset/step), not replayed.

The environments here are deterministic, in-process, credential-free simulators,
so the lane is fully reproducible. Each implements the same contract:

* `reset(spec) -> (state, obs)`
* `step(state, action) -> (state, obs, reward, done, info)`
* `optimal_action(obs) -> action` — a reference policy that proves solvability

A *live external* env server (an HTTP reset/step endpoint) is the same contract
with a network transport. It plugs in as another `Environment` without changing
the driver or the unified `Result`, and is deferred to owner infra.

The shipped `pull_starter` suite carries two tasks: `reach_target` (1-D
navigation — move from `start` to `target` inside a step budget) and
`guess_number` (binary-search the secret with higher/lower hints). The registry
of built-in envs is `reach_target` and `guess_number`. The failure classes this
page targets: a policy that raises mid-episode (the episode fails, the lane does
not), an unknown env kind (recorded `void`, never a silent `fail`), and a runner
that reports a never-run lane as `0% passed`.

## 2. Run it

Drive both `pull_starter` envs with each env's own reference policy
(`{"type": "reference"}`, the optimal `obs -> action`), so the run is
deterministic and credential-free, and write the artifact:

```bash
python examples/bench_pull_rl.py artifacts/bench-pull-rl.json
```

The same run from the CLI — the agent spec is the policy:

```bash
agent-learn bench examples/bench_suites/pull_starter.json --mode pull --agent '{"type":"reference"}'
```

To drive **your own** policy, pass a callable `obs -> action` as `agent`. The
observation keys differ per env (`reach_target` exposes `pos` / `target` /
`remaining`; `guess_number` exposes `low` / `high` / `last` / `hint` /
`remaining`), so a portable policy branches on what it sees:

```python
from fi.alk import bench


def policy(obs: dict) -> str:
    if "target" in obs:  # reach_target: step toward the target
        if obs["pos"] < obs["target"]:
            return "right"
        if obs["pos"] > obs["target"]:
            return "left"
        return "stay"
    # guess_number: bisect the remaining range
    return str((int(obs["low"]) + int(obs["high"])) // 2)


result = bench.run_bench(
    "examples/bench_suites/pull_starter.json",
    agent=policy,
    control_mode="pull",
)
print(result["aggregate"]["pass_rate"], result["aggregate"]["scored"])
```

Two other policy specs ship for quick baselines: `{"type": "reference"}` (the
env's optimal policy) and `{"type": "noop"}` (always the first action — a
deliberately weak floor). An unknown env kind or an unresolvable policy is
recorded `void` (the lane never ran honestly), never a silent `fail`;
`pass_rate` is computed over *scored* tasks only.

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/bench-pull-rl.json')); a=p['aggregate']; assert a['pass_rate']==1.0, p; assert a['scored']==2, p; assert p['modalities']==['rl'], p; print('ok')"
```

The artifact records the suite name and version, the `modalities` (here `["rl"]`),
the aggregate (`count`, `scored`, `void`, `passed`, `pass_rate`, `mean_score`,
plus the `by_modality` / `by_world_kind` / `by_execution_class` rollups and the
`honesty` block), and one row per task carrying the unified `result` — whose
`scalar` is the cumulative reward, `components` hold `reward` + `steps`, and
`pass_fail.goal_reached` records the terminal outcome — alongside the honesty
fields (`execution_class`, `evidence_class`, `overclaim`).

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| every row shows `verdict: void` with `unknown env kind` | the task's `env.kind` is not in the registry (`reach_target` / `guess_number`) — fix the suite or register the env | `missing_public_modules` |
| `verdict: void` with `unknown pull policy` | the `agent` spec `type` is not `reference` / `noop` and is not a callable | `missing_public_modules` |
| a task `fail`s with `policy raised: ...` | your policy callable threw mid-episode — the episode fails, the lane stays honest; fix the policy, not the harness | `missing_public_modules` |
| `pass_rate` lower than expected | the policy did not reach the goal inside the step budget — read `result.components.steps` and the env's `spec.max_steps` | `missing_public_modules` |
| `BenchError: pull bench suites run under control_mode='pull'` | you passed a non-`pull` mode for a `control: pull` suite | `missing_public_modules` |

## 5. Prove it / keep it

Pull is one of three control modes the unified harness exposes; the harness, the
modes, and the cross-modality `Result` are covered in
[benchmark-overview](./benchmark-overview.md). For the submit-and-score lane
(score candidate code against a held-out oracle), see
[benchmark-coding](./benchmark-coding.md). To fit a policy by running it
repeatedly against the same envs and keeping what improves, feed the same suite
through the optimize track and gate on `pass_rate` as the metric to beat.
