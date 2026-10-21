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
- For every tool the agent gained, add a sub-goal whose check names the tool, so the suite stage
  adds a scenario for what it lets the caller do. `report_update` fails a new tool no sub-goal
  checks.
- Change the world's data with `seed` or `change_data`, and keep it with `save_world`.
- When the agent's seed changes, reproduce its new rows and values; never drop one to keep a world
  check passing. Rewrite a world check that pins the old seed with `add_world_check` under its
  own name, citing the seed file and quoting the new row.
- The world holds every row of the agent's current seed, including in-flight ones such as an
  active booking or an open order, even when the facts do not list that row as changed. A world
  check that keeps such a table empty pins the old seed: rewrite it the same way.

## Suite stage

- The environment is fixed. Repair scenarios that broke, drop the ones the new agent makes
  meaningless, add scenarios for what it newly does, and never rename one.
- Before adding, raise the target with `aim_for` to the size the suite will have.
- Every sub-goal the environment added must be named by at least one scenario; one scenario may
  cover several.
- Drop a sub-goal from scenarios with `retire_sub_goal` only after the environment retired it.
- Prove everything with `prove_scenarios`. Scenarios the person is still editing are listed at the
  end of the message. Only this message's list counts: a scenario an earlier message listed is
  no longer being edited and is repaired like any other. In a listed scenario keep what the
  person wrote and only carry over a mechanical change, such as a renamed tool or argument in its
  solution; leave it as it is otherwise, even when it fails.

## Ending the turn

End with `report_update(changed, notes)`. It runs this stage's checks and records the result.
If it reports problems, fix them and call it again; the last call is the one that counts.
`changed` is false only when nothing needed to change. Keep the notes short: what changed and
why, what you left out, and anything you could not check. Never put a credential, token or other
secret value in the notes.
