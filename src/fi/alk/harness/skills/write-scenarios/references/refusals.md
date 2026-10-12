# Every refusal, its cause and its fix

Validation runs before the three gates when you submit a scenario. Every problem is reported at
once, so fix them together and submit again. The second table covers the refusals writers hit most
often; each is cheaper to avoid while writing than to fix after a refusal.

| What you are told | Why | Fix |
|---|---|---|
| `no name` / `no instruction` | Empty required field. | Supply it. |
| `persona has no details` | A persona was given with every field blank. | Fill it, or leave `persona` out entirely. |
| `persona is incomplete: ...` | A persona needs `name`, `personality`, `communication_style`, `initial_message`, `accent`, at least one language and at least one keyword. | Fill the named fields. |
| `persona <field> ... is not one the platform knows` | `personality`, `communication_style`, `accent` and `languages` must come from the offered values. A word of your own renders fine and then selects no behaviour. | Use an offered value. Anything else about the person goes in `metadata`. |
| `no sub_goals` | Nothing would grade the scenario. | Name the catalogue entries this scenario exercises. |
| `sub_goals not in the catalogue: ...` | A name that does not exist. The message lists what does. | Use an existing name, or `add_sub_goal` first. |
| `no solution` | Without the actions a correct agent would take, nothing can show the scenario is passable. | Work it out with `try_calls`. |
| `no fixture manifest` | The world has data and the scenario declared none. | Add `fixture` with `origin` and the facts the person relies on. |
| `fixture.origin must be seed, generated, or mixed` | Any other value. | Use one of the three. |
| `fixture.origin is 'generated' ... but setup_code is empty` | The fixture claims the scenario creates data while creating none. | Seed everything the fixture names, or declare `origin: seed` and use only records that already exist. |
| `setup_code only adjusts records that were already there ...` | The setup changes or drops rows it did not create, so the scenario shares its data with every other scenario touching those rows. | Create what the outcome turns on with `world.put`, or by driving the agent's own tool, then adjust that. An empty setup stays legal for the no-seam case. |
| `the reference solution is a single call ...` | Nothing had to be established before the outcome, so an agent that fires that call on arrival passes. | Show how the outcome is reached: the lookups the decision depends on, named as sub-goals too. |
| `the name contains the person's own name` | The name says who was on the other end rather than what broke. | Name it for the behaviour: `cancel_active_booking_with_fee`, not `dana_cancels_her_booking`. |
| `fixture uses predictable verification code(s)` | Sequential or repeated digits. | Generate an unremarkable value of the right shape. |
| `fixture uses placeholder payment-card ending(s)` | `4242`, `1234`, `0000` and similar in a card-ending field of the fixture. | Use an unremarkable ending. |
| `setup_code must define setup(world)` / `ready_code must define ready(world)` | Wrong entry point. | Define the function with that exact name. |
| `the prompt asks for ..., which this scenario does not supply` | The prompt has a slot nothing fills, and an unfilled slot reaches the person verbatim. | Add it to `variables`. |
| `<tool> requires <value> from this call, but the reference solution does not create it first` | A hard rule says a value must come from this conversation, and the solution supplies it from setup or `environment_arguments` instead. | Put the step that produces it earlier in the solution. |


## The refusals writers hit most, and how to avoid them first time

| What you are told | Why | Write it right first time |
|---|---|---|
| `... already occupies this cell and asserts the same sub-goals` | Same coordinate, same checks: the same test twice. | Deal it a different difficulty, a different cell, or a check that only this scenario can fail. |
| `coverage puts <axis> at ..., which is not a level the plan deals` | The level was invented rather than copied from the brief. | Copy every coverage value from your brief, spelled as dealt. |
| `... already has a caller named ...` | Two people in the suite share a full name, so their results cannot be told apart. | Choose another full name from the same background; a common given name is fine with a different family name. |
| `<axis> is already at ... which is its whole share of this suite` | One level of an axis already holds a third of the suite while other levels are thin. | Write one of the thinner levels it names; behaviour, interaction, task and person all have room. |
| `... may not put more than ... on one` | One location, accent or language already holds its share of the suite. | Choose a different one, with a name and language that fit it; the situation can stay. |
| `the coordinate claims a condition the call does not carry` | A level the persona fields do not deliver. | Set the persona field the kind file names for that level, or choose a level the scenario really carries. |

## What no check catches, and a reviewer will

Nothing refuses these; the hard requirements and "Before you submit" are how you avoid them.

| Mistake | Write it right first time |
|---|---|
| The instruction states what the agent will say, offer or decide. | Give the person a stance and condition on what they experience, never on the agent's words. |
| The instruction names the attack or its category. | Write the attack as the person would say it; the category lives only in the coordinate. |
| The person is told to take the unsafe path if it is offered. | Write only the pressing; whether the agent gives way is what the check measures. |
| The words describe somebody the persona is not (age, name, language). | Choose the person first and let every field and every line follow from them. |
| The person is told to say a value that nothing seeds. | Seed it in `setup_code`, or give the person a value the world already holds. |
| The situation needs something the channel cannot carry. | Put the difficulty in what the person says; the kind file lists what the channel carries. |
| Placeholder people, places or references. | Use ordinary, real-sounding values that fit the person and the world. |
