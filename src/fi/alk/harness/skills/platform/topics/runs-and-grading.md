---
name: runs-and-grading
description: How runs, repeats and trials work, how a call is decided (held, failed, not decided), what the judge sees, platform evaluations on top of sub-goals, receipts, and run limits.
---

# Runs and grading

## Runs

A run is a separate job that replays the sealed environment for the scenarios chosen, each played a
number of times (repeats, 1 to 20); each play is one call, also called a trial. Scenarios times repeats
is the call count, at most 200 per run by default. Runs need a completed environment. Calls run in
parallel up to the environment's admitted parallelism (1 to 8). A build never places calls; only runs do.

Why repeats: no two calls are the same. The caller phrases things differently and is seeded per call,
the agent's model varies, and on voice a misheard word changes the path. One pass shows the agent can;
several show it does every time.

## How a call is decided

After the call ends, in the sandbox, with the world still available, each sub-goal the scenario names
is settled:

- a **check** runs on the world and the tool calls and returns held or not, with a reason;
- a **judged** sub-goal goes to the judge, which reads its criteria, the agent's own instructions (never
  holding it to more), the customer's side as context only, the actions the agent took, the transcript
  and the records (it reads the transcript text, never the audio), and decides. Content the channel adds itself (a recording notice) is not the agent's.
  If it still cannot decide after one retry, the sub-goal is recorded as not decided (`held: null`).

The call passes when every sub-goal holds and fails when any one does not. A judged sub-goal left not
decided leaves the call not decided rather than passed. Reasons shown to the owner are one or two plain
sentences about what the agent did.

## What each grader sees

Sub-goals are the only grading that sees the agent's tool calls: checks receive them, and the judge
lists them. Platform evaluations bound to the environment (up to eight, chosen at build or added by
hand) run on the platform on every finished call, from the transcript or the recording, and do not see
tool calls. Write a criterion that depends on what the agent did as a sub-goal, not an evaluation.

## Receipts

Each call sends one receipt: status, each sub-goal with held, judged and reason, the call record
(turns, duration, how it ended), and artifacts (transcript, recording, tool trace). `read_runs` shows
them per scenario number.

## Known limits on results

- Judged sub-goals read the transcript; a broken transcript (dropped turns, wrong language) can give a
  wrong verdict on a good call.
- Checks confirm a step was taken, not always that it was right in every detail, and there is no
  separate overall-objective check: a call passes when every named sub-goal passes.
- The target's own limits (for example a rate limit on its account) count as call failures; re-run them.
