---
name: platform
description: What the platform supports around an environment - how it is built, what a scenario holds, how the Scenarios tab, coverage, runs, repeats and grading work, and what cannot be done. Use to answer any question about the platform or a limit, before promising anything.
---

# The platform around an environment

Facts about this particular environment still come from tools. This skill is how the platform
works, so you can tell the person what is possible and what a result means.

## How an environment is built

The person points the builder at an agent: a code repository, uploaded code, or an agent hosted on
a voice or chat platform (or reached by a phone number). The build then:

1. **Understands the agent**: reads its source or definition and writes the contract: tools with
   exact arguments and permitted values, rules, data, use cases.
2. **Builds the world**: a test copy of the data the agent's tools act on, the simulated caller's
   prompt, and the shared catalogue of sub-goals.
3. **Writes the scenarios**: plans a suite over eight coverage axes, writes each scenario, and
   proves each one before keeping it.
4. **Seals** everything into a bundle that every run replays exactly.

An agent on a hosted platform keeps its data on that platform. Its world is empty, so its
scenarios are proved to set up correctly and every sub-goal is judged from the conversation. An
agent from code acts on the world, so most sub-goals are settled by code reading what the call
left behind.

## What a scenario holds

| Field | Meaning |
|---|---|
| name | Unique within the environment; the `#` number is the tab's stable number for it |
| use case | Which of the agent's use cases it tests |
| branch | The path the conversation is meant to take |
| situation (instruction) | What the caller wants and is living through, written for the simulated caller |
| persona | Name, gender, age group, accent, languages, location, personality, communication style, occupation, and the facts they hold |
| starting state | The change made to the world before the call (setup), checked by its ready check |
| sub-goals | Named checks from the shared catalogue; all must hold for a pass |
| ideal outcome (tests) | One line: what a pass looks like |
| coverage | Its value on each of the eight axes |
| background noise | The place heard behind the caller on a voice call, or a quiet line |
| keywords | Tags the person can filter on |

Each scenario carries a reference solution and checks. Three gates prove it before it is kept:
**ready** (the world holds what it presumes), **solvable** (the solution passes its own checks),
**not vacuous** (doing nothing fails them).

## The eight coverage axes

Task (one value per use case), who is calling (counterparty), their situation (disposition), line
and speech or how they write (interface), how the call goes (interaction), attack (overlay),
attack method (overlay vector), attack intensity. The Coverage card on the Scenarios tab shows how
many axes the suite varies, the share of the pairwise grid it reaches for two axes, and how many of
five forced situations it includes: destructive or irreversible requests, minors or vulnerable
callers, emergencies, privacy and other people's data, and prompt injection. Coverage maps what the
suite tests; a suite of 40 reaches at most 40 cells of any grid.

## The Scenarios tab

A table of every scenario with its number, grouped by use case by default (also by sub-goal,
accent, age, attack, task, or none). Search matches every word in a scenario's name, situation, use
case and branch. Filters pick values per field (use case, caller name, accent, language, age,
gender, location, personality, style, background, attack, attack intensity, task, sub-goal,
keywords; exact match on scenario name and situation). The person can tick rows, select all
matching, delete, and run a selection with repeats.

Its edit panel changes only surface fields without asking you: the ideal outcome, accent,
communication style, personality, occupation, location, languages, keywords and background noise.
Name, use case, branch, situation, sub-goals, and the caller's name, age group and gender are locked
there because the scenario was proved with them; changing those is your work, with a fresh proof.

## Runs and results

A run plays the selected scenarios against the agent. **Repeats** sets how many calls each scenario
gets, from 1 to 20; scenarios times repeats is the number of calls, and one run is capped at the
platform's limit on calls per run. Repeats matter because no two calls are the same: the caller
phrases things differently, the agent's model varies, and on a voice call a misheard word changes
the path. One pass shows the agent can; several show it does every time.

A call passes when every sub-goal holds and fails when any one does not. A judged sub-goal that
cannot be settled leaves the call not decided rather than passed. Evaluations the person added
grade every call on top of its sub-goals. Each result names every sub-goal, whether it held,
whether it was judged, and the reason.

## What cannot be done from the chat

The chat changes the test environment, never the agent. It cannot change the agent's prompt, code,
or provider configuration, the phone number or connection a run dials, credentials, or the voice
and model the agent uses. It cannot offer a caller value the platform does not list
(`offered_values` is the list). Rebuilding the world itself from scratch is a new build of the
environment, started from the Build environment tab.
