---
name: environment-model
description: What an environment is made of - contract, world, tools, sub-goal catalogue, scenarios, the proof gates, the runtime certificate, and the sealed bundle runs replay.
---

# The environment model

An environment is what a completed build leaves behind: the contract, the world, the sub-goal
catalogue and the proved suite, sealed together in one authoring archive. Runs replay that archive;
they never re-author.

## The parts

| Part | What it is |
|---|---|
| Contract | What the agent verifiably is, read from its source or prompt: tools with exact argument names and permitted values, hard rules, data, use cases, modality (voice or chat), and up to six platform evals it should be graded by. Corrections are recorded as amendments on the contract, never blended into what was read |
| World | The data and services the agent's tools act on: a schema, seed data, and the agent's tools bound to it so every call resolves against something real. A refusal from the world is the world working; a crash is a defect in what was built. An agent reached only by a phone number or a hosted platform has no world (connect-only) |
| Simulator prompt | The template the simulated caller runs on, filled per scenario |
| Sub-goal catalogue | Named checks defined once for the agent and shared by every scenario that names them (`sub_goals.json`). Sharing is what makes results add up: one sub-goal failing in seven of twelve scenarios is one finding |
| Scenarios | One folder each: `scenario.json` (persona, situation, ideal outcome, sub-goal names, reference solution, coverage), `setup.py`, `ready.py`, and `checks/<sub_goal>.py` for each deterministic sub-goal. An index (`scenarios.json`) and a coverage report (`coverage.json`) sit beside them |
| Environment bundle | The sealed copy runs execute (`environment-bundle/`): a manifest listing every file with its hash, the scenarios, the catalogue, the contract, the seed for the world database, the processes to start, readiness probes. Its digest covers every listed file, so any change is detectable |

## Sub-goals: checked or judged

A sub-goal has either a **check** or **judged criteria**.

- A **check** is code, `check(world, calls)`, that reads the world after the call and every tool call
  with its arguments. It returns nothing when the sub-goal held, or a sentence saying what was wrong.
  It settles facts exactly: an action taken with the wrong value is a failure and needs no judgement.
- **Judged criteria** are for what leaves no trace: words and manner (whether a refusal was explained,
  whether something was invented, tone). They are four labelled lines: `Applies when`, `Pass when`,
  `Fail when` (agent behaviour only, never the caller's), and `If it does not arise` (Pass or Fail and
  why: fail when the agent's own behaviour kept the situation away, pass when only the caller's path did).
  Criteria may not cite tool calls or world state, because code settles those.
- Checks come first; judging is only where nothing is observable. A catalogue more than half judged
  is flagged, except on a connect-only agent where judging everything is correct.
- A scenario names at least one sub-goal that fails when the agent gets its own difficulty wrong. A
  scenario carrying an attack names the sub-goal for that attack as well as one for the ordinary task,
  or an agent that refuses everything would pass.

## How a scenario is proved

Every scenario passes three gates before it is kept. They are pure code, no model asked:

| Gate | Question | Why it matters |
|---|---|---|
| Ready | Reset the world, run `setup.py`, then `ready.py`. Does the world hold what the scenario presumes? | Otherwise the agent fails for something the test got wrong |
| Solvable | Run the reference solution, then the checks. Do they pass? | Otherwise the scenario cannot be passed, or a check is wrong |
| Not vacuous | Reset, set up, do nothing, run the checks. Do they fail? | A check that passes when nothing happens grades nothing |

Before the gates, validation refuses scenarios that break structural rules (off the declared grid, a
level the persona cannot deliver, a duplicate cell asserting the same sub-goals, a reused caller name,
an instruction that tells the caller what the agent will do, an attack asserting nothing, more
scenarios than asked for). The same refusal three times tells the writer to write a different one.

**Hosted (connect-only) agents are judged-only.** When the world holds nothing a check could read (no
store, no tool endpoint, because the agent and its data live on another platform), the conversation is
the only evidence. The scenario still runs setup and ready, but solvable and not-vacuous cannot mean
anything without a check, so the proof records it as judged-only and every sub-goal is judged from the
call. Its reference solution is cleared, because nothing local could replay it. Conversely, on a world
with readable state, a scenario whose sub-goals are all judged is refused: settle it by reading the state.

## Runtime validation and the certificate

After the suite is saved, the build validates it as a whole against the real runtime: it builds the
bundle, starts it, checks the count, and runs every scenario's setup against the running environment
(in parallel lanes). For an agent built from code, the result is a certificate (`runtime-validation.json`)
bound to the bundle digest and the source digest: static checks, schema and seed, processes, source
invariants, scenario setup ready N/N, tool contract, reset equivalence, world isolation, plus stated
limitations. A run refuses to place a call unless the certificate matches the bundle it is about to run.

## Sealing and chat edits

The bundle is sealed once at build time. When this conversation changes scenarios, `save_changes`
carries the scenario folders, catalogue and contract into the bundle, recomputes every file hash and
the digest, and reissues the certificate for the new set with a recorded limitation: the changed
scenarios were proved by the gates against the authoring world, not re-validated on the runtime. The
runtime itself (processes, seed) is unchanged. The environment keeps one archive; a save overwrites it.
There are no scenario versions yet: an edit replaces the earlier scenario.
