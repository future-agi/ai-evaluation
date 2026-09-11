---
kind: agent-learning.docs-page.v1
track: eval
objective: safety
stage: evaluate
backing:
  - examples/bench_sandboxes.py
artifact_kinds: []
commands:
  - python examples/bench_sandboxes.py artifacts/bench-sandboxes.json
  - agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference --sandbox subprocess
  - agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference --sandbox docker --evidence-class live_lane
postcondition: python -c "import json; p=json.load(open('artifacts/bench-sandboxes.json')); s=p['subprocess_lane']; assert s['aggregate']['pass_rate']==1.0, s; assert s['aggregate']['scored']==3, s; d=p['docker_lane']; assert d['launched_container'] is False, d; assert d['all_hardening_present'] is True, d; assert d['isolation_flags']['--network none'] is True, d; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
opt_in_lane: false
---

# Bench sandboxes: where candidate code actually runs

> **Twin:** [`examples/bench_sandboxes.py`](../../examples/bench_sandboxes.py)
> · two code-exec lanes · offline by default, no credentials, no container launched.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The unified bench harness scores a coding suite by *actually executing* candidate
code against a held-out oracle. The one dimension this page covers is **where that
execution happens** — the `sandbox` argument to `run_bench(...)`:

* **`subprocess`** (default) — a fresh interpreter in a throwaway tempdir, with a
  scrubbed environment (no harness secrets cross in) and a hard wall-clock
  timeout. It is credential-free, runs anywhere, and is the lane the release gate
  uses on trusted, shipped reference code. It is **not** a security boundary
  against deliberately hostile code: there is no real filesystem or network
  isolation, so a candidate could read the host or reach the network.

* **`docker`** (opt-in) — per-task, ephemeral, OS-level isolation for
  **untrusted agent output**. The container drops everything the subprocess lane
  cannot: `--network none`, `--cap-drop ALL`, `--security-opt no-new-privileges`,
  a `--read-only` rootfs, a nosuid size-capped tmpfs as the only writable surface,
  a non-root user (`65534`), and PID / memory / CPU caps. It is `--rm`, so the
  container is killed and removed after the run.

The failure classes this page targets: an untrusted candidate that escapes a weak
sandbox, a runner that *silently downgrades* to a weaker sandbox when Docker is
absent (mislabelling isolation), and a missing daemon read as an agent failure
instead of an honest `void`.

**Honesty.** A real Docker run of untrusted code is a genuine live event, so the
harness stamps those rows `evidence_class=live_lane` (never `captured_fixture`)
and never downgrades the label. The subprocess lane on trusted shipped code is a
`local_gate` event. If the daemon is unreachable, the lane is recorded `void` —
it never ran, so it is neither a pass nor a fail.

## 2. Run it

Score the shipped `coding_starter` suite in the `subprocess` lane (deterministic,
credential-free), report whether Docker is reachable, and capture the exact
hardened `docker run` argv **without launching a container** — then write the
artifact:

```bash
python examples/bench_sandboxes.py artifacts/bench-sandboxes.json
```

The same convenience-lane scoring from the CLI, against the suite's own
references:

```bash
agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference --sandbox subprocess
```

To run **untrusted output** under OS-level isolation, opt into the Docker lane on
a host with a daemon (it stamps the rows `live_lane`):

```bash
agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference --sandbox docker --evidence-class live_lane
```

In Python, the sandbox is one argument; the example inspects the hardened argv via
the **pure** builder, so it never contacts a daemon:

```python
from fi.alk import bench
from fi.alk.bench import _docker

# Convenience lane: trusted shipped code, scored in a scrubbed subprocess.
suite = bench.load_coding_suite("examples/bench_suites/coding_starter.json")
result = bench.run_bench(
    "examples/bench_suites/coding_starter.json",
    control_mode="artifact_in",
    submission=bench.reference_submission(suite),
    sandbox="subprocess",
    evidence_class="local_gate",
)
print(result["aggregate"]["pass_rate"], result["aggregate"]["scored"])

# Report the hardened lane without starting anything: probe the daemon, then
# build the EXACT argv a Docker run would launch (no container is created).
print("docker reachable:", _docker.docker_available())
argv = _docker._build_docker_argv(
    name="demo", image=_docker.DEFAULT_IMAGE,
    memory=_docker._DEFAULT_MEMORY, cpus=_docker._DEFAULT_CPUS, bootstrap="...",
)
print("--network" in argv and argv[argv.index("--network") + 1] == "none")
```

A daemon that is missing or unreachable makes every Docker row `void` (never
silently passed, never re-run in the weaker lane). `pass_rate` is computed over
*scored* tasks only, so a host with no Docker does not read as "0% passed".

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/bench-sandboxes.json')); s=p['subprocess_lane']; assert s['aggregate']['pass_rate']==1.0, s; assert s['aggregate']['scored']==3, s; d=p['docker_lane']; assert d['launched_container'] is False, d; assert d['all_hardening_present'] is True, d; assert d['isolation_flags']['--network none'] is True, d; print('ok')"
```

The artifact records two lanes. `subprocess_lane` carries the real run — the
aggregate (`count`, `scored`, `void`, `passed`, `pass_rate`, `mean_score`, the
`by_*` rollups, and the `honesty` block stamped `local_gate`) plus one row per
task. `docker_lane` carries the *audit of the hardened lane without running it*:
`docker_available`, `launched_container: false`, the chosen `image` / `memory` /
`cpus` / `pids_limit`, the exact `hardened_argv` from the pure builder, an
`isolation_flags` map auditing each control flag, and `all_hardening_present`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| Docker rows show `verdict: void` with `infra: docker unavailable` | no reachable daemon — the lane never ran, so it is `void`, not `fail` | `missing_public_modules` |
| `docker run failed (exit ...)` recorded as `void` | the image is not pulled or the daemon erred — infra, not the agent | `missing_public_modules` |
| `all_hardening_present` is `false` in the artifact | an isolation flag was dropped from the argv — file a bug; never weaken the untrusted lane | `missing_public_modules` |
| an untrusted candidate "passes" the Docker lane | structural hold-out only defends accidental gaming, not a candidate that attacks the harness protocol — do not treat as authoritative | `missing_public_modules` |
| `BenchError: unknown sandbox '...'` | `sandbox` must be `subprocess` or `docker` | `missing_public_modules` |

## 5. Prove it / keep it

The subprocess lane and the held-out-oracle scoring model are covered in
[benchmark-coding](./benchmark-coding.md). For the forge- and oracle-read-resistant
grading model — a held-out grader that runs *after* the candidate is killed,
which the Docker lane composes with for untrusted output — see
[benchmark-command-graded](./benchmark-command-graded.md). The harness, its three
control modes, and the unified `Result` are covered in
[benchmark-overview](./benchmark-overview.md).
