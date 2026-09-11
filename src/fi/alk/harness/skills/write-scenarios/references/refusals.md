# Every refusal, its cause and its fix

Validation runs before the three gates when you submit a scenario. Every problem is reported at
once, so fix them together and submit again.

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
| `the instruction gives the person ... to say back, and neither setup_code nor the world holds it` | The instruction hands over a code, reference or identifier that exists nowhere, so the conversation cannot succeed however well the agent behaves. | Seed that exact value in `setup_code`, or tell the person the value that is seeded. Naming it in `fixture` only declares it. |
| `setup_code only adjusts records that were already there ...` | The setup changes or drops rows it did not create, so the scenario shares its data with every other scenario touching those rows. | Create what the outcome turns on with `world.put`, or by driving the agent's own tool, then adjust that. An empty setup stays legal for the no-seam case. |
| `the reference solution is a single call ...` | Nothing had to be established before the outcome, so an agent that fires that call on arrival passes. | Show how the outcome is reached: the lookups the decision depends on, named as sub-goals too. |
| `the name contains the person's own name` | The name says who was on the other end rather than what broke. | Name it for the behaviour: `cancel_active_booking_with_fee`, not `dana_cancels_her_booking`. |
| `fixture uses predictable verification code(s)` | Sequential or repeated digits. | Generate an unremarkable value of the right shape. |
| `fixture contains placeholder demo data` | `test user`, `john doe`, `123 main street` and similar. | Use plausible real-world values. |
| `fixture uses placeholder payment-card ending(s)` | `4242`, `1234`, `0000` and similar, in the fixture or spoken in the instruction. | Use an unremarkable ending. |
| `fixture uses placeholder transaction identifier(s)` | Identifiers ending in a bare `1`, or obvious stand-ins. | Use values shaped like the agent's real ones. |
| `setup_code must define setup(world)` / `ready_code must define ready(world)` | Wrong entry point. | Define the function with that exact name. |
| `the prompt asks for ..., which this scenario does not supply` | The prompt has a slot nothing fills, and an unfilled slot reaches the person verbatim. | Add it to `variables`. |
| `<tool> requires <value> from this call, but the reference solution does not create it first` | A hard rule says a value must come from this conversation, and the solution supplies it from setup or `environment_arguments` instead. | Put the step that produces it earlier in the solution. |
