---
name: surfaces
description: Exactly where things are in the product and what the person sees - My Environments, Build environment and statuses, the workspace tabs, the Scenarios tab (columns, #, search, filters, group by, Coverage card, Edit panel, delete, selection), Runs and results, Evaluations, Settings, and this chat drawer. Use to tell someone where to click and what they will see after a change.
---

# Product surfaces

Use the labels exactly as written here, in bold, when you tell someone where to click.

## Finding the environment

- **Simulate** in the sidebar, then **Environment**. It opens on **Build environment**; **My
  Environments** beside it lists every environment with its status, agent type, and counts of tools,
  scenarios and runs. Click one to open its workspace.
- Build sources under **Or connect your own agent**: **Source repository** (GitHub `owner/repo`, branch,
  optional entry path, Public or Private with a GitHub App installation ID), **Code upload** (choose a
  folder, name the entry file), **Hosted platform** (Agent type Voice or Chat; Vapi with Assistant ID and
  key, Retell AI with Agent ID and key, or Others with a phone number and the agent's system prompt; Web
  or Phone calls; **Agent speaks first**). **Environment values (optional)** are fixed at build time.
  **Scenarios to generate** defaults to 10, up to 1000. **Parallel worlds** 1 to 4. **Run preflight**,
  then **Build environment**.
- Statuses: **Building**, **Finalizing**, **Ready**, **Failed**, **Cancelling**, **Cancelled**. Build
  steps shown: Understanding agent, Generating environment, Building environment, Validating
  environment, Generating data, Generating scenarios, Validating scenarios.

## The workspace

Tabs: **Overview** (Channel, Connector, Test subject; Capabilities: Tools and Rules read from source),
**Contract** (Capability graph with Flows and Guardrails, World state tables and row counts, Tools, Run
end conditions), **Scenarios**, **Runs**, **Evaluations**, **Settings** (Environment info to rename,
Environment variables with secrets never shown). This chat is the drawer on the left: the box **Reply
to the builder…** under **Ask, correct, or steer what it builds**; it reads **Working on your last
message…** while a turn runs, and the turn can be stopped. Each step shows under the reply with a green
dot (succeeded) or a red dot (a check turned it away and it is being fixed).

## The Scenarios tab

- One row per scenario. Columns: **#**, **Scenario**, **Persona**, **Situation**, **Sub-goals**,
  **Ideal outcome**. The **#** is the scenario's stable number and stays with it through every search,
  filter, addition and removal. Switch to **List** and click a name to read one in full (its What this
  tests section shows the ideal outcome). Rows page 25 at a time.
- **Coverage** card above the table, collapsed to three figures: **Axes** (how many of the eight are
  varied, like `8/8`), **Pairs** (share of the Pairwise grid reached), **Forced** (how many of five
  forced situations, out of `5`). Expanded: the axes table (**Levels**, **Covered levels**), **Forced
  overlays** (Destructive / irreversible, Minor / vulnerable, Emergency / crisis, PII / privacy,
  Prompt-injection; a filled dot means present), and the **Pairwise grid** with **Rows** and **Columns**
  pickers, task against attack by default. A search or filter updates the card too.
- **Search** matches every word, in any order, against the name, the situation, the use case and the
  branch.
- **Filter**: **Basic** tab builds rows from menus, **Query** tab types them as field, operator, value.
  Fields **Use case**, **Name** (the caller's name), **Accent**, **Language**, **Age**, **Gender**,
  **Location**, **Personality**, **Style**, **Background**, **Attack**, **Attack intensity**, **Task**,
  **Sub-goal**, **Keywords** take **Is** / **Is not** with one or more values; **Scenario** and
  **Situation** take **Equals** / **Not equals** on the whole text. Several values in one row match any;
  separate rows must all hold. Accent and Background only appear for voice agents. **Clear** resets
  search and filters.
- **Group by**: **Use case** (default), **Sub-goal**, **Accent**, **Age**, **Attack**, **Task**, **No
  grouping**. Scenarios with no value land under **Ungrouped**.
- **Edit scenario**: the pencil icon on a row. Locked, marked proved: the name, use case, branch, what
  the caller wants (situation), sub-goals, and the caller's name, age group and gender. Editable:
  **Passes when** (the ideal outcome), **Accent** (voice), **Communication style**, **Personality**,
  **Occupation**, **Location**, **Languages**, **Keywords**, and **Background noise** (voice; **Off**,
  **Street**, **In a car**, **Station / transit**, **Airport**, **Shop / mall**, **Office**, **Home**,
  **Hospital**, **Outdoors**, **Crowded room**). **Save scenario** turns on once something changed. So:
  to change a caller's accent yourself, open **Scenarios**, click the pencil on that row, change
  **Accent** under the persona section, and click **Save scenario**.
- **Delete**: the trash icon on a row, or tick rows and click **Delete** in the selection bar; confirm in
  **Delete scenarios?**. A removed scenario cannot be brought back. Keep at least one.
- **Selection**: the header checkbox ticks the page; then **Select all N matching** takes in every
  scenario matching the search and filters. A new search, filter or grouping clears the selection.
  Selected rows are what "these" means when the person messages this chat.

## Running

- Whole suite: **Repeats** at the top right (**1×**, **3×**, **5×**, **8×**, **Custom…** up to 20; it
  shows the total, like `30 scenarios × 1 repeat = 30 runs`), then **Run simulation** and confirm.
- Selection: tick rows on **Scenarios**; the bar shows the count, a trials picker (1 to 20), an
  estimate, and **Run simulation (N)**.
- **Runs** tab: **Simulations summary** chart of evaluation averages across runs; the table has
  Scenarios, Trials, Simulations, Pass, Duration, Tokens. Run statuses: Queued, Provisioning, Running,
  Grading, Completed, Failed, Cancelled. A run's name carries the agent version.
- A run's page: **Debug failures**, **Add evals**, **Export**, **Run again**, **Stop simulation** while
  running; tabs **Test runs** (one row per call, grouped by **Goal** or **Status**; chips **Failing**,
  **Errored**, **Inconclusive**, **Passing**; a column per evaluation and per sub-goal) and **Analytics**.
  Click a call for its drawer: **Transcript** (with tool calls and their arguments) and **Recording** on
  the left; **Call Analytics**, **Evals**, **Scenario**, **Messages**, **Logs**, **Attributes**,
  **Annotations** on the right.

## Evaluations tab

**Added evaluations** (each lists what it **Reads**), **Add evaluations**, a **Suggested** list,
**Enable tool call evaluation**. Adding one grades calls from then on; **Add evals** on a run, then
**Grade this run**, grades a finished run.
