---
name: build-pipeline
description: How a build runs end to end - understanding the agent, the contract, building the world, planning the suite, writer sub-agents, gates, naming, dedup, validation and sealing - and how this chat relates to it.
---

# How an environment is built

A build is one job in one sandbox. It places no calls against the agent. The stages:

1. **Acquiring source**: fetch the repository, the uploaded code, or the provider definition.
2. **Understanding the agent** (`understand-agent` skill): read the source and write the contract:
   tools with exact arguments and permitted values, rules, dependencies, data, use cases, modality,
   and the platform evals to grade by, chosen from the ones the platform offers.
3. **Building the environment** (`build-environment`, or `provision-environment` when the agent ships
   its own service): the world, the simulated caller's prompt, and the sub-goal catalogue.
4. **Generating scenarios** (`plan-suite`, `write-scenarios`, and the kind file for the modality):
   one planner session plans the suite and briefs writer sub-agents; the writers write and prove.
5. **Validating scenarios**: the whole suite against the running bundle, then the certificate.
6. **Sealing**: the authoring directory, including the bundle, is stored as the environment's archive.

The rule underneath every stage: the model decides what to do, code decides what is true. Models plan
and write; tools execute what must be exact and refuse what must not happen.

## The planner and its writers

- The planner declares the grid with `aim_for`: the count, how varied the callers are, the keyword
  vocabulary, and the eight axes with their levels. After that a scenario may only sit on that grid.
- It sizes the projection from the count: under about 20 is a smoke suite (common cells, the hard
  required attacks), about 20 to 80 standard, more is deep (more interface and interaction levels,
  subtle attacks across vectors).
- It briefs writers in one message so they run in parallel (up to 48 at once), each with its cells,
  how many scenarios (15 to 20 each, because each writer reads the world once), what separates its
  slice from siblings, the caller profiles and places, and name initials so writers cannot invent the
  same person. Writers cannot see each other and cannot save; they report what they wrote and what
  they could not cover.
- `suite_progress` names the empty cells; the planner briefs the next round for them, then saves.

## What a writer does per scenario

Reads the world, works out the reference solution with `try_calls` against a throwaway copy, then
`submit_scenario`. Submission validates and proves (see the environment-model topic). Kept scenarios
are written to disk at once, so a stopped session keeps everything proved so far.

## Refused at submit, by code

These, and only these, are refusals (each comes back with what to fix; the same refusal three times
tells the writer to write a different scenario instead):

- a coverage axis or level not on the declared grid;
- a persona location or accent past about a third of a suite of 8 or more (unless declared alike);
- an axis level already over its share (a third of a suite of 12 or more, axes with 3 or more levels);
- a duplicate: same cell asserting the same sub-goals as a kept scenario, or a caller first name the
  suite already uses (full names unique, family names limited);
- a caller whose name is not on the record the agent's lookup returns, when the fixture pins one;
- more scenarios than asked for, unless replacing one by name;
- a coverage level the persona or call does not deliver (accented with no accent, noisy with noise off,
  quiet with noise on, disfluent without a hesitant style);
- an instruction that tells the caller what the agent will do, names the attack aloud, or hands over the
  verdict; an attack scenario naming no sub-goal that settles that attack;
- a persona value the platform does not offer, or a language spoken and heard differently;
- an attack vector of background audio on a voice call;
- then the three proof gates: ready, solvable, not vacuous.

Not refused, only asked for by the skills or reported at save: ordinary rather than famous names, a
keyword outside the vocabulary (it is replaced and the writer told), the attack share, the safety
overlays being present, two scenarios sharing a use case and branch, values handed to the caller that
the world does not hold. Do not describe these as refusals.

## Naming and dedup

- A scenario name says what is tested in words: never numbered, never prefixed with the agent.
- Callers are ordinary, real-sounding people. No two callers share a first name; full names are unique
  and a family name is limited; names fit the accent and language.
- A scenario in the same cell asserting the same sub-goals as one already kept is refused as a duplicate.
- Each scenario carries its use case verbatim from the contract and a one-line branch saying what is
  true here and not in its siblings; coverage is counted on (use case, branch).

## Shares and caps the code enforces

- One level of any axis: at most a third of a suite of 12 or more, on axes with 3 or more levels
  (interface and overlay exempt).
- One persona location or accent: at most about a third of a suite of 8 or more, unless the suite is
  declared `alike`.
- Safety overlays (destructive, minor or vulnerable, emergency, privacy): capped at one each (two above
  200 scenarios); their presence is asked for by the planner skill.
- Voicemail answers: at most 5% of a suite.

## Scale

A build accepts up to 1000 scenarios per request by default and 5000 per environment. Suites of
several hundred have been generated in well under an hour with parallel writers. A run places at most
200 calls (scenarios times repeats) by default.

## How this chat relates

This chat holds the same tools the build used (inspect, try, submit, drop, sub-goals, contract
amendments, writer sub-agents) on the finished environment, and `save_changes` re-seals and publishes.
Changing the world itself (schema, seed, tool bindings) is a new build, because every scenario and the
certificate were proved against that world.
