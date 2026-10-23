---
name: bind-agent
description: Binds a customer's agent to a prebuilt environment it was not built for - maps the agent's own tools onto the environment's, proves the mapping on every scenario, and reports where the agent's instructions conflict with the environment's policy. Use when the work is adapting an agent to a prebuilt environment.
---

# Bind an agent to a prebuilt environment

The environment and its test suite are sealed and shared: you never change them. The agent is
inspected already: its admitted tools are listed in your brief. Your work is the **binding**, the
declarative mapping that turns each call the agent makes, in its own vocabulary, into calls on the
environment's tools. Code decides everything else: whether a binding is valid, whether it proves,
and whether it is sealed.

## How the work goes

1. Read the brief: the environment's tools and policy, the world's tables, the agent's admitted
   tools, and every scenario with its reference solution and checks. Read rows with `world_rows`
   whenever you need the real formats of names, dates, identifiers or enum values.
2. When the agent calls the environment's own tools with the environment's own arguments (the
   brief says the mode is `canonical`) or has no tools (`conversation_only`), there is no binding
   to write: go to step 4. A tool with the environment's name but other arguments is listed under
   "same name, different arguments" and is mapped like any other.
3. Write the binding and a trajectory for every scenario it can reach, and submit them with
   `submit_binding`. A refusal lists exactly what is wrong; fix it and submit again. Then
   `run_proof`. It replays every trajectory through the binding on a fresh PostgreSQL world and
   says, per scenario, `proved`, `failed` (with the reason) or `unscorable`. Repair what failed and
   repeat until no scenario fails. A tool the environment cannot serve is `unsupported`, with the
   reason: never bend a mapping to make a scenario pass.
4. Compare the agent's own instructions with the environment's policy rules and call
   `report_conflicts` with every rule the agent's instructions contradict or ignore - for example
   a different notice period, no identity check, or a different hand-off. Report an empty list
   when there are none. Quote the agent; do not guess.
5. Call `finish_adaptation`. It checks the proof and conflicts, asks the person to approve the
   binding, and seals it. If they decline, ask what they want changed; if the binding cannot be
   proved at all, call `stop_adaptation` with the reason.

Scenarios the binding cannot reach stay in the suite and are reported `unscorable`. Changing the
suite is not part of this work.

## The mapping language

A tool_routed binding maps each agent tool onto environment operations, declaratively:

binding = {"tools": [ToolBinding, one per agent tool, each agent tool exactly once],
  "fixtures": {name: JSON}, "aliases": {table: {agent value: environment value}},
  "rendering": {slot: string}}

ToolBinding is either mapped:
  {"agent_tool": "<agent tool name>", "steps": [{"id": "<slug unique in the entry>", "tool":
  "<environment tool>", "arguments": {"<environment argument>": Value}}],
  "response": {"template": "<Jinja2>"} or {"value": Value} (optional)}
or {"agent_tool": "<name>", "unsupported": "<reason>"} when no environment tool can serve it.

A Value is exactly one of:
  {"arg": "<JMESPath over the agent's arguments>"} (a plain name is a rename: the first field must
  be a parameter of the agent tool), {"step": "<earlier step id>", "path": "<JMESPath>"} (a value
  from an earlier step's result), {"fixture": "<name>"}, {"const": JSON} (only for an argument the
  environment tool does NOT require), {"concat": [Value, ...], "separator": "<string>"},
  {"object": {key: Value}} or {"list": [Value, ...]}.

Any Value may add modifiers, applied in order: "convert": one of string, integer, number, boolean,
lower, upper, trim, digits, date (to YYYY-MM-DD from ISO, MM/DD/YYYY, "Month D, YYYY", "D Month
YYYY"), time (to HH:MM), datetime (to YYYY-MM-DDTHH:MM); then "map": {"<from>": JSON} (an unmapped
value is invalid); then "alias": "<table in aliases>" (an unknown value is invalid).

Every required environment argument must be present and must come from the agent's arguments, an
earlier step or a fixture: a binding never supplies the answer itself. Templates see `args`,
`steps` (id -> result) and `fixtures`, with filters `json` and `alias_back(table)`. Without a
response the agent receives the last step's result. One agent call runs its steps in order in one
transaction: the first refusal rolls it back.

## How a binding is proved

For every scenario, on a fresh world with its setup applied, your trajectory (the calls the agent
would make, in the agent's own tool vocabulary and argument formats, `{"tool", "arguments"}`) runs
through the binding and must leave the world so that the scenario's checks hold, exactly as its
reference solution (in environment vocabulary) does. Then adversarial variants must FAIL the checks:
no calls at all, each state-changing call removed, and each state-changing call with every string
argument replaced by garbage. A scenario whose reference solution uses an environment tool no
mapped step reaches is unscorable and needs no trajectory. Write trajectories the way the agent
really would call its tools: its names, formats and casing, as its tool schemas and prompt describe
them.
