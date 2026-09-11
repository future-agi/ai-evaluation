---
kind: agent-learning.docs-page.v1
track: eval
objective: capability
stage: evaluate
backing:
  - examples/bench_voice.py
artifact_kinds: []
commands:
  - python examples/bench_voice.py artifacts/bench-voice.json
  - python -c "import json; p=json.load(open('artifacts/bench-voice.json')); print(p['per_task'][0]['result']['components'])"
postcondition: python -c "import json; p=json.load(open('artifacts/bench-voice.json')); a=p['aggregate']; assert a['pass_rate']==1.0, p; assert a['scored']==1, p; assert p['modalities']==['voice'], p; c=p['per_task'][0]['result']['components']; assert set(c)=={'latency','turn_taking','barge_in','content'}, c; assert all(v>=0.75 for v in c.values()), c; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
opt_in_lane: false
---

# Voice benchmark: score a voice episode on a temporal contract

> **Twin:** [`examples/bench_voice.py`](../../examples/bench_voice.py)
> · `artifact_in` control mode · offline, no credentials, no network.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Voice is the modality that stress-tests the harness: the environment is an active
caller and the verifier is *temporal*, not an exit code. The task is a phone-style
episode, and the artifact you score is a **transcript** — an interleaved list of
caller and agent turns with millisecond timing. `score_voice_episode` grades that
transcript on four sub-scores, each a fraction in `[0, 1]`:

* **latency** — for every agent reply, the gap after the caller stops talking must
  be within the task's `max_latency_ms` budget. Score = fraction of agent replies
  inside budget. The agent has to answer promptly, not just correctly.
* **turn_taking** — the agent must not talk over the caller. An agent turn that
  overlaps a caller turn, with no caller interrupt to excuse it, is *harmful
  overlap*. Score = `1 - harmful_overlaps / agent_turns`.
* **barge_in** — when a caller turn is flagged `interrupt: true` and lands while
  the agent is mid-sentence, the agent must yield: its turn has to end within the
  yield window (600 ms) of the interrupt's start. Score = fraction of barge-ins
  handled. (With no interrupts in the episode, this is a vacuous `1.0`.)
* **content** — the agent's words must cover the task's `required_content`
  keywords (case-insensitive substring). Score = fraction of keywords hit. (With
  no required content, a vacuous `1.0`.)

The scalar is the mean of the four sub-scores. The verdict is **all-or-nothing**:
a task is a pass only when *every* sub-score meets the pass floor (`0.75`). A
single bad dimension — the agent talks over the caller, or ignores a barge-in —
fails the whole episode even if the other three are perfect.

A voice bench suite (`agent-learning.bench-suite.v1` with `control: voice`)
carries, per task, an `instruction`, a `budgets` block (`max_latency_ms`), a
`required_content` list, and a gold `reference_dialogue`. You score it through one
call — `run_bench(suite, control_mode="artifact_in", submission={task_id:
dialogue})` — and get one unified `Result` per task plus an honest aggregate.

This is the **simulated / deep-contract** tier: the transcript comes from a
deterministic simulated caller, so the lane is credential-free and reproducible.
The *same* verifier consumes a transcript captured from a live audio / SIP /
WebRTC call plus ASR — that live capture (and real word-error rate) is deferred to
owner infra and plugs in unchanged by producing the same turn shape.

## 2. Run it

Score the shipped `voice_starter` suite against its own gold `reference_dialogue`
(so the run is deterministic and credential-free), and write the artifact:

```bash
python examples/bench_voice.py artifacts/bench-voice.json
```

A voice suite runs `artifact_in` with `submission={task_id: dialogue}`, where a
dialogue is a list of turns. To grade **your** captured transcript instead of the
gold reference, build the submission map yourself and call the harness:

```python
from fi.alk import bench

# A transcript is a list of turns. Caller turns may carry interrupt=True.
dialogue = [
    {"speaker": "caller", "start_ms": 0, "end_ms": 1500, "text": "I want a refund please"},
    {"speaker": "agent", "start_ms": 1700, "end_ms": 3500,
     "text": "Sure - our refund policy gives you a 30 day window."},
    {"speaker": "caller", "start_ms": 3100, "end_ms": 3300, "text": "wait", "interrupt": True},
    {"speaker": "agent", "start_ms": 3650, "end_ms": 4200, "text": "Yes, go ahead."},
]
result = bench.run_bench(
    "examples/bench_suites/voice_starter.json",
    control_mode="artifact_in",
    submission={"refund-call": dialogue},
)
row = result["per_task"][0]
print(row["verdict"], row["result"]["components"])
```

`budgets` and `required_content` come from each task in the suite — the latency
budget and content keywords are part of the contract, not the submission. A task
with no transcript in the submission is recorded `void` (never silently passed)
and does not count against `pass_rate`, which is computed over *scored* tasks only.
You can also score a single transcript directly with
`bench._voice.score_voice_episode(dialogue, budgets=..., required_content=...)`
when you want the raw sub-scores without the suite wrapper.

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/bench-voice.json')); a=p['aggregate']; assert a['pass_rate']==1.0, p; assert a['scored']==1, p; assert p['modalities']==['voice'], p; c=p['per_task'][0]['result']['components']; assert set(c)=={'latency','turn_taking','barge_in','content'}, c; assert all(v>=0.75 for v in c.values()), c; print('ok')"
```

The artifact records the suite name and version, the `modalities` list
(`["voice"]` here), the aggregate (`count`, `scored`, `void`, `passed`,
`pass_rate`, `mean_score`, plus the `by_modality` / `by_world_kind` /
`by_execution_class` rollups and the `honesty` block), and one row per task
carrying the unified `result` — `scalar`, the four `components`
(`latency` / `turn_taking` / `barge_in` / `content`), and a `pass_fail` map with
the overall `voice` verdict plus a per-dimension `<name>_floor` boolean.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| a task you submitted shows `verdict: void` | no transcript for that `task_id` in the submission map — `void` is excluded from `pass_rate`, not counted as a fail | `missing_public_modules` |
| `latency_floor: false` on a correct answer | an agent reply lands more than `max_latency_ms` after the caller stopped — raise the task's `budgets.max_latency_ms` or tighten the agent's response time | `missing_public_modules` |
| `turn_taking_floor: false` | an agent turn overlaps a caller turn with no `interrupt` to excuse it — the agent is talking over the caller | `missing_public_modules` |
| `barge_in_floor: false` | a caller `interrupt: true` turn was not yielded to inside the 600 ms window — the agent kept talking through the interruption | `missing_public_modules` |
| `content_floor: false` | the agent's words miss a `required_content` keyword (case-insensitive substring) — check the keyword list against what the agent actually said | `missing_public_modules` |
| `BenchError: voice artifact_in requires submission={task_id: dialogue}` | you called `artifact_in` with no `submission` — voice has no live agent, so a transcript map is mandatory | `missing_public_modules` |

## 5. Prove it / keep it

The harness, its three control modes, and the unified `Result` are covered in
[benchmark-overview](./benchmark-overview.md). For the `artifact_in` lane applied
to candidate code against a held-out oracle, see
[benchmark-coding](./benchmark-coding.md); for the agent-driven reset/step lane,
see [benchmark-pull-rl](./benchmark-pull-rl.md). To assemble your own voice (or
any-modality) suite — the `control` discriminator, the Task↔Verifier coupling, and
the Goodhart guards — see
[benchmark-write-a-suite](./benchmark-write-a-suite.md).
