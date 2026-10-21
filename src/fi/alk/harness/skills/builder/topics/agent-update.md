---
name: agent-update
description: How to bring the environment or its suite up to a new version of the agent in a rebuild turn, and how to end it with report_update.
---

# Updating for a new agent version

A rebuild turn is not a conversation. The person asked for it from a card and is not watching, so
do not ask questions: decide from the facts in the briefing and the agent's source, and say what you
decided in the notes.

## Method

- Read the facts in the briefing first: which tools, rules, prompts and values changed between the
  two agent versions. Change only what those facts touch; leave everything else as it is.
- For every tool or rule you add or remove, cite the file in the agent's source that shows it, and
  quote the line. A change you cannot point at in the source is a guess; leave it out and say so.
- Before changing anything, run `impact_of` on it, and repair what depends on it.
- Never rename. A sub-goal that no longer fits is retired with `retire_sub_goal` and, if still
  needed, added again under its new meaning with `add_sub_goal` (environment stage only). A scenario keeps its name.

## Environment stage

- Change the contract, the world and the sub-goal catalogue. Never edit scenarios: the suite is
  rebuilt in its own turn after this one lands.
- There is no running world and no credentials here. Do not try to run the agent's tools; the
  checks are static.
- Change the world's data with `seed` or `change_data`, and keep it with `save_world`.

## Suite stage

- The environment is fixed. Repair scenarios that broke, drop the ones the new agent makes
  meaningless, add scenarios for what it newly does, and never rename one.
- Before adding, raise the target with `aim_for` to the size the suite will have.
- Every sub-goal the environment added must be named by at least one scenario; one scenario may
  cover several.
- Drop a sub-goal from scenarios with `retire_sub_goal` only after the environment retired it.
- Prove everything with `prove_scenarios`. Scenarios the person is still editing are listed at the
  end of the message; leave them as they are, even when they fail.

## Ending the turn

End with `report_update(changed, notes)`. It runs this stage's checks and records the result.
If it reports problems, fix them and call it again; the last call is the one that counts.
`changed` is false only when nothing needed to change. Keep the notes short: what changed and
why, what you left out, and anything you could not check. Never put a credential, token or other
secret value in the notes.
