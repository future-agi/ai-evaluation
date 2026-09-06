---
name: write-scenarios
description: Write the scenarios an agent is tested with, each proved against the real world before it is kept. Use whenever scenarios, test cases or a suite are wanted for an agent whose contract and world have already been built.
---

# Write the scenarios

You are writing tests for an AI agent. The environment already exists: a world its tools really act
on, a prompt for the person it talks to, and a shared catalogue of the named things this agent can be
checked on. Your job is to write the individual tests, prove each one, and keep it.

Everything you need about the agent is in front of you. The contract above lists its tools with their
arguments, its hard rules, its data, its real use cases and how its tools report a refusal. A summary
of the world follows it. Do not restate those; read them.

## Which job you have

These instructions are loaded by more than one kind of session. Work out which you are from what you
were asked, then follow only that part.

**You were asked for a number of scenarios.** You are planning a suite. Decide what it covers, then
hand it to `generate_suite`, which runs one writer per part of your plan. Separate planning
instructions follow this file when a suite is what was asked for. You do not write the scenarios
yourself.

**You were given one brief.** You are a writer. Somebody has already read the agent, decided which
pairings of thing-acted-on and thing-wanted are worth testing, and how many scenarios each earns.
Your brief is one of those. Write inside it, and:

- Do not widen the brief to take in something interesting you noticed. Say so when you finish and
  let the plan decide.
- Do not write a second scenario because the person could be somebody else. The same test with a
  different person is one test written twice.

**You were asked for one particular scenario**, or to replace one that came back wrong. Write that
one and nothing else.

If the agent's modality has its own instructions, they follow below. They add requirements; they do
not replace any of these.

## What a scenario is

One test. It changes the world a little, gives a person a task, and names what must be true
afterwards. It is a whole conversation from first contact to a settled outcome, not a single step.

| Field | What it is | Required |
|---|---|---|
| `name` | Short identifier, lower case with hyphens or underscores. Becomes the scenario's folder name, so it is how a result is read later. | yes |
| `instruction` | What the person is trying to achieve, written to them, plus everything they hold. | yes |
| `sub_goals` | Names from the shared catalogue that must hold. Nothing else grades this scenario. | yes |
| `solution` | What a correct agent would do, as steps. Never run against the agent under test. | yes |
| `fixture` | A readable manifest of the data this scenario relies on. Its `origin` must be `seed`, `generated` or `mixed`. A fixture changes nothing by itself. | yes, when the world has data |
| `persona` | Who the person is, as structured fields. | yes, when the prompt asks for one |
| `setup_code` | Python defining `setup(world)`. This is what actually changes the world. | when the instruction presumes anything |
| `ready_code` | Python defining `ready(world)`. Returns `None` when the world is ready, or a sentence naming what is missing. | recommended |
| `use_case` | Which of the agent's use cases this belongs to, copied from the contract word for word. Results are grouped by matching this string exactly, so a rewording becomes a group of its own. | no |
| `branch` | What is true here that is not true of its siblings in the same use case. | no |
| `tests` | One line: the condition this scenario passes on. Shown to people as "passes when", so write it to complete that phrase. | no |
| `variables` | Extra values the prompt asks for, by slot name. Each is substituted into the prompt where its name appears. | when the prompt asks |
| `max_turns` | How many turns the conversation may take. Defaults to 10. | no |

`branch` and `tests` are different and easy to confuse. `branch` is the **condition**: what is
different about this scenario's world or request. `tests` is the **question**: what the run will find
out. For a scenario about an expired payment method, `branch` is "the method on file has expired" and
`tests` is "the agent notices before charging and offers another".

Write `branch` and `tests` about the agent's behaviour, never about how the scenario was built.
"Synthetic", "seeded", "setup_code" and "fixture" name your machinery, not anything the agent did,
and they are noise in a report.

**A solution step** is a tool name plus the arguments the agent would supply:

```
{"tool": "get_account", "arguments": {"account_id": "..."}}
```

Some agents insert trusted values between what the model chooses and what the underlying service
receives: resolved identifiers, prices, routes. Those must never appear as arguments the model
supposedly chose. Put them in `environment_arguments` on the same step, which the proof passes to the
service and the agent never sees.

## How grading works

A **sub-goal** is one named thing the agent can be checked on, defined once for the agent and shared
by every scenario that names it. That sharing is what makes results add up: the same sub-goal failing
in seven of twelve scenarios is one sentence somebody can act on, rather than seven separate notes.

Every sub-goal is graded one of two ways, and **you choose which by whether you give it a check**:

- **Deterministic.** The sub-goal carries a `check`: Python receiving the world as the run left it and
  every call the agent made with its arguments, returning nothing if it held or a sentence saying
  what was wrong. This is what you want almost always.
- **Judged.** The sub-goal carries no check, so a model reads the transcript and decides. Reserve
  this for things nothing observable can settle: whether a refusal was explained kindly, whether a
  number was invented.

Prefer a check. You have the world afterwards and every call with its arguments, so most things worth
checking are visible in one of them. A judged sub-goal is reported as judged, and a suite of them
tells you less than it appears to.

**A check that only asks whether a tool was called is not a check.** It proves the plumbing worked, not
that the agent behaved: any agent that reaches the tool at all passes it, and no agent that behaves
correctly by another route can. Assert the arguments it was given, or the state the world was left in.
"The row now holds the value the caller gave" is a check. "the tool appears in the calls" is not.

**And do not make the mechanics of ending a call a sub-goal.** Whether a particular closing tool was
invoked is plumbing. What is worth checking is what the agent did before it stopped: that it left a
message naming who was calling and why, that it stopped asking questions once there was nobody to
answer, that it did not press on after being told to stop. A measured run failed a scenario because a
named closing tool was not called, while the agent had already closed the call through the tool that
honours a removal request, which is correct behaviour scored as a failure. Where one tool's documented effect
already covers another's, requiring both is asking for a redundant call.

Name entries that already exist. Do not restate one in your own words and do not invent a second name
for something already covered. If something genuinely needs checking and no entry covers it, add one
with `add_sub_goal`.

## The tools, and the order to use them

| Tool | What it does |
|---|---|
| `inspect_world` | Lists the world's collections and their sizes; with a collection, returns records from it. `matching` is plain text, not SQL. |
| `inspect_scenario` | Reads one already-kept scenario in full. Use before replacing one, rather than reconstructing it from memory. |
| `try_calls` | Runs calls against a **throwaway copy** of the world and shows the state they leave. This is how you work out a solution and what its checks should assert. Nothing you do here is visible to anybody else. |
| `add_sub_goal` | Adds a named thing this agent can be checked on, with its check in code. |
| `submit_scenario` | Keeps one scenario, after validation and the three gates. |
| `drop_scenario` | Removes one by name, or all of them with `*`. |
| `aim_for` | Sets how many scenarios are wanted. Needed when reopening an existing suite to add more, because the target starts at what is already there. Not for saving a suite nobody asked for. |
| `save_scenarios` | Finishes the suite. Reports what is off about it as a whole. |
| `amend_contract`, `add_rule`, `drop_rule`, `fix_tool` | Correct the contract when it is wrong. See the last section. |

The order for one scenario:

1. `inspect_world` with no collection, then look at the ones that matter. Read the sub-goals that
   already exist.
2. Read the agent's hard rules. Each one is a branch waiting to be written.
3. Work out the solution with `try_calls`, passing your `setup_code` so you see the world the agent
   will actually face. Confirm the sub-goals you intend to name respond to it.
4. `submit_scenario`. Read what comes back: a refusal names exactly what is wrong.
5. `save_scenarios` once you have what was asked for.

A scenario is written to disk the moment it is kept, so proved work survives a stopped turn. Submit
as you go rather than composing a whole suite before the first call.

## The three gates

Every scenario is put through these when you submit it. Failing any one means it is not kept, and you
are told which.

**1. Ready.** The world is restored, your `setup_code` runs, then your `ready_code`. The world must
end up holding what your scenario presumes.

This is the gate people skip and the one that saves you. A scenario about the last five items in
stock is only a test of the agent if there really are five. If there are none, the agent fails for
something you got wrong and it reads as the agent's fault. `ready_code` makes that impossible.

**2. Solvable.** Your reference solution is played through that world, and the checks of every
sub-goal you named must pass. If they do not, either the scenario cannot be passed at all or a check
is wrong.

**3. Not vacuous.** The same checks run again with nothing done, and must fail. A check that passes
while the agent does nothing grades nothing while reporting a result.

Gate 3 has a common trap. If your scenario is about something that must **not** happen, checking the
world alone cannot show it: an untouched world looks exactly like one where the agent correctly
refused. Check the calls instead: that the agent tried, and that the attempt was refused rather than
succeeding.

## What will be refused, and what to do

Validation runs before the gates, and every problem is reported at once, so fix them together.
**`references/refusals.md` lists every refusal, its cause and its fix.** Read it before your
first submission rather than after a refusal: most of what it names is cheaper to avoid than to
correct, and several entries are mistakes that look correct on the page.

The ones worth knowing before you write anything:

- A value the instruction tells the person to say back must exist in `setup_code` or the world.
  Naming it in `fixture` only declares it.
- A reference solution of one call is refused, because nothing had to be established first.
- A scenario name may not contain the person's own name.
- A `fixture` whose `origin` is `generated` or `mixed` must actually create data.
- `personality`, `communication_style`, `accent` and `languages` must use offered values.

`save_scenarios` additionally reports what is off about the suite as a whole: too few distinct people,
opening lines repeated word for word, too few locations, verification codes reused between scenarios,
identical setup data, and for suites where the agent started the conversation, one awareness value
used for more than about two thirds of them. These are reported rather than refused. Read them and
fix what they name.

## The bar every scenario has to clear

Four of these are enforced by validation. Two are your judgement, and no check can make them for you.

- **A competent agent could plausibly fail it.** *(judgement)* If any correct implementation passes
  for free, it teaches nothing. Do not write it.
- **A real person could plausibly bring this situation.** *(judgement)* Nothing contrived.
- **Every concrete value is real**, taken from the contract or the world. *(enforced: values handed to
  the person must exist)* An invented identifier makes the test worthless whatever else it does.
- **Check the path, not only the outcome.** *(enforced: a one-step solution is refused)* Where the
  right answer depends on something the agent must find out first, the sub-goals cover that too.
- **The scenario seeds what it needs.** *(enforced: a fixture claiming data must create it)* Every
  record whose state decides the outcome is created by this scenario's `setup_code`.
- **The name says what is tested.** *(enforced: the person's name may not appear in it)*

**What is not a scenario.** A person asks for the ordinary thing, the agent does it, both are polite,
it ends. Nothing was withheld, nothing contradicted, no rule was pressed, no state had to carry, and
any working agent passes. That is a demonstration. It costs a real run and real money and returns no
information about the agent. One scenario covers the ordinary path for a whole suite; everything else
has to earn its place by being able to fail.

```
BAD    solution   [transfer_to_human(reason="Account suspended")]
       sub_goals  [transferred_to_human]
       (an agent that hands off every request on arrival passes this. Whether it
        looked the account up, and found the suspension, is never measured)

GOOD   solution   [find_account(identifier=...), get_account(account_id=...),
                   transfer_to_human(reason="Account suspended")]
       sub_goals  [account_identified, account_state_checked, transferred_to_human]
       (the handoff now has to be reached by discovering the reason for it)
```

## Three parts that must never leak into each other

Getting this wrong is the most common way to write a scenario that looks fine and measures nothing.

| | What it is | What it must never contain |
|---|---|---|
| **instruction** | what the person is living through | the answer, the checks, facts they could not know, or anything the agent is expected to do |
| **setup** | the world's condition | anything the person is supposed to say |
| **checks** | the hidden pass or fail rules | anything the agent was told |

## Writing the instruction

**The instruction is a circumstance, not a script.** Write it in the second person, as what this
person is living through: who they are, what is happening to them, and what they want. Never a list
of lines to say, and never the agent's turns.

```
BAD    Ask for <thing A>. Then change your mind and ask for <thing B> instead.
       Confirm the total at the end.
       (a stage direction. The person recites it, and the run measures whether the
        agent can follow dictation. Nothing about the change of mind is tested,
        because it arrives exactly when the script says so)

GOOD   You want <thing A>, and you are not particular about <the detail the agent
       has to settle>. Partway through, you realise <thing B> is what you actually
       need, and you would rather swap than end up with both.
       (a situation. What they say is theirs to work out, and the agent has to cope
        with a change of mind arriving mid-conversation rather than on cue)
```

Those placeholders are deliberate. Fill them from **this** agent's own data, never from a worked
example of another agent.

**Write the objective, not the history.** A person told what happened narrates it; a person told what
they want pursues it. Open with the goal in their own words, "get <the thing> put right", not with the
history that led to it. Then give them the facts they hold, the values they can be asked for, and
what they will only say when asked.

**What they know but will not volunteer goes in its own paragraph**, marked as such: *"You know the
reference for it, but you will only give it if asked."* Whether the agent asks is the whole point of
many scenarios. Put it in the instruction and the agent gets it for free; leave it out entirely and
the scenario cannot be completed.

**Knowing a value and volunteering it are separate choices.** The person must possess every value the
agent could legitimately ask for. Whether they offer it unprompted is the scenario's decision. Those
are two different sentences and only the second is optional.

**Never tell the person what the agent will do.** This is the single most common way a scenario stops
measuring anything. The agent's moves are what is being tested, so a person told to expect them plays
along whether or not they happen, and the check passes on a conversation that never earned it. Write
only what this person knows before the conversation begins.

```
BAD    The agent will tell you about <the condition>. Accept it and say yes when
       asked to confirm.
       (the scenario is testing whether the agent discloses <the condition>. A person
        primed to accept it agrees even when the agent never says it, so the run
        reports a pass for behaviour that did not occur)

GOOD   You want <the outcome>. You will accept <the condition> if there is one, but
       you want to know <the detail> before you agree to anything.
       (the person's own position. If the agent discloses, they accept; if it does
        not, they ask, and the transcript records which happened)
```

The rule covers every phrasing: "the agent will send you <a value>", "they will offer you <an
option>", "they should hand you over". Give the person the value, the preference or the problem they
arrived with. What the agent does about it is the measurement, so it cannot also be part of the brief.

**The test that catches all of it: could this person say the sentence out loud?** A parenthetical
explaining where the agent should find a value is not a smaller version of the mistake, it is the same
mistake more quietly.

```
BAD    Your <destination>: <value> (the agent should find this from your <record>)
       (the person has no idea the agent has records, let alone which one. The note is
        written for whoever reads the scenario, and it names the mechanism being tested)

GOOD   Your <destination> is the same one you used last time. You do not remember the
       exact address and would rather not look it up.
       (now the person has a reason to expect the agent to know, which is what makes
        the lookup worth testing, without being told the lookup exists)
```

Pre-agreeing to something the agent has not done yet is the most damaging form. "You have already
<completed the step> that the agent will <send>" hands the agent a pass. Write what the person has
done, never what they have done in response to an action the agent has not taken.

**Steps that happen outside the conversation need a state, not a response.** Some flows depend on the
person doing something the simulation cannot perform: following a link, checking another device,
reading a message. The temptation is to write their answer in advance, which is the pass-handing form
again, because the answer arrives whether or not the agent ever asked.

```
BAD    The agent will send you <the out-of-band thing>. Tell them you have
       completed it when asked.
       (the scenario is testing whether the agent sends it. This person confirms
        completing it even in a run where nothing was ever sent)

GOOD   You have your <device> with you and you are willing to follow anything you
       are sent. You have not been sent anything yet.
       (a state. If the agent sends it, this person can act on it and say so
        truthfully. If the agent never does, they have nothing to confirm, and the
        transcript shows the difference)
```

The closing sentence matters: saying what has **not** happened yet is what stops the person assuming
it has. And only write such a step where the agent can observe it completing, because the person
saying they did it changes nothing the agent reads. If the agent confirms progress by checking state,
the world has to move when the person acts, or the agent polls something that never changes and the
scenario measures the world's gap instead of the agent.

The same applies to anything the agent can only offer. A check that passes only once the person
accepts an optional courtesy needs that willingness written in, because the agent can raise the offer
but cannot make them take it. Either give them a reason to accept, or check that the offer was made
rather than what followed it.

### What this person is known by

Many agents establish who they are dealing with before they will act. Give that its own short section
at the end of the instruction, and **read every value out of the world with `inspect_world` first**.
Never invented, never carried from another scenario: the record has to be the one the agent's own
lookup will find.

Four rules, and each has cost a whole run:

**Cover every route, not the one you expect.** Where an agent can establish something more than one
way, which way it takes is not yours to choose. An instruction carrying values for one route is
complete until that route fails, and then the person cannot answer a question they plainly should be
able to answer.

**Say what each value is for.** Where two values share a shape but not a role, the current one and
the replacement, the account's and the order's, give both and name each role. Handed one, the person
offers it for the other purpose because it is the only such value they have. It is real, it is in the
instruction, and it still fails, which is harder to diagnose than a missing value.

**Take them all from one record.** Fields from two records describe somebody who does not exist, and
no lookup will find them.

### The person, and why they are hard

`persona` is the structured profile of the person making the request: `name`, `gender`, `age_group`,
`occupation`, `location`, `personality`, `communication_style`, `keywords`, `languages`, `accent`,
`multilingual`, and free-form `metadata`. `personality`, `communication_style`, `accent` and
`languages` must use offered values, because each selects real behaviour downstream; a word of your
own renders fine and selects nothing.

Use the fields that change the risk being tested, and **make the person the reason the scenario is
hard**. If swapping in a calm, fully informed person would not change the outcome, the persona is
doing no work.

**A different name is not a different person.** Personas drift toward one temperament: cooperative,
articulate, patient, answering exactly what was asked. A suite of those tests a person the agent will
rarely encounter, and passes every scenario for the same reason. Vary `personality` and
`communication_style`, not just identity: somebody terse to the point of unhelpfulness, somebody who
volunteers three things at once, somebody distracted who has to be asked twice, somebody impatient who
pushes back early. Let the situation pick the temperament rather than attaching one at random.

Keep the person and the world's condition apart: the persona is who is asking, `setup_code` is what is
true of the world. A name that says one person in the persona and another in the instruction
misreports every result anybody reads.

## When the agent started the conversation

Read `CALL DIRECTION` on the contract before writing a single instruction. The two directions need the
person written differently, and getting it wrong tests the wrong half of the exchange.

**Inbound: the person approached the agent.** Everything above assumes this. They have an errand, they
know why they are there, and they open by saying what they want.

**Outbound: the agent approached the person.** This inverts almost everything:

- They have **no errand of their own.** They were doing something else.
- They do **not know who this is** until the agent says so, and must not act as if they do.
- Their first turn is a bare greeting and nothing more. It answers the agent's opening, which still
  comes first.
- They may be **suspicious.** An unexpected approach about their account is what a scam looks like,
  so asking the agent to prove itself is correct behaviour, not obstruction.
- They may be **busy or unwilling.** Declining to talk now is a legitimate outcome worth testing.
- What this is about is the **agent's** purpose. The instruction says how the person reacts to it,
  not what they wanted.

```
BAD    Book the premium tier from your home to your office.
       (they never approached anyone; nothing prompts them to ask for this)

GOOD   You are at home getting ready for work. If someone contacts you about the
       booking you have on file, you would take it, leaving from home and going to
       the office. You will not raise any of that yourself.
```

An outbound instruction that opens with a request has been written as inbound, and the scenario then
tests an errand the agent never raised.

### How much the person already knows

`caller_awareness` changes the whole exchange, so choose it deliberately. Each value needs
**different data** in the instruction:

| They are | What the instruction must carry |
|---|---|
| `expecting` | They know what it is about and roughly what they agreed, so they can be asked to confirm a detail. They must hold that detail, and their version may differ from the world's. |
| `partial` | They know something happened but not the detail: not the date, not the amount, not which of two things. Say what they do recall and what they have lost. |
| `unaware` | No context at all. The agent has to establish who they are and why it is contacting them before anything else. Give them the facts they hold about themselves and nothing about the reason. |

**Do not write every outbound scenario as `expecting`.** It is the easiest and least informative: a
person who was told to expect this can reasonably ask for what they want, so the scenario stops
testing how the agent opens something it started. At least one outbound scenario per use case must be
`unaware`, and where a use case gets only one or two, prefer `unaware`.

**But an `unaware` person still needs facts.** Somebody with no context and nothing to offer produces
a short, empty exchange, and that is a badly written scenario rather than a finding about the agent.
They hold their own details and a reaction to being contacted unexpectedly; what they must not hold is
the reason.

## Making the world match the instruction

**Whatever the instruction presumes, setup has to make true.** This is where scenarios most often go
wrong: the instruction says the person is returning an order that has already shipped, setup leaves
every order pending, so the agent refuses correctly and the scenario fails it for being right.

Read your own instruction back, list every condition it assumes, and make sure `setup_code`
establishes each one and `ready_code` proves it. If the instruction hands the person a value to say
back, `setup_code` is what puts that exact value where the agent will look for it.

### setup_code

Python defining `setup(world)`.

**Create the records this scenario turns on.** Every record whose state decides the outcome is made
here, with values belonging to this scenario: its own person, its own order, its own booking, its own
code. Shared reference data the whole world sits on, a product catalogue or a list of regions, can be
read as it is and used as a model for what a realistic new record looks like. What you must not do is
build the test on rows that were already there: another scenario may change them, two scenarios then
quietly test the same row, and neither describes a world it controls.

**Write every setup against the base world, never against another scenario.** At run time each
scenario restores its own copy of the frozen base and applies only its own setup, so nothing another
scenario did is there. Writers run at the same time and in any order, so there is no "before" to
depend on: if a scenario needs an order delivered, its own setup delivers it. The calls you make while
rehearsing with `try_calls` run on a throwaway copy and change nothing anybody else sees.

You have two ways to change things, and **neither names what the world is kept in**. A scenario that
wrote SQL would only work against a world that happened to use that engine, and the store varies more
between agents than anything else.

**Prefer the agent's own tools.** They go through the same path the agent will, so anything the world
would refuse you would have refused the agent too.

```python
def setup(world):
    world.call("add_to_stock", {"item_id": "widget", "quantity": 5})
```

**Otherwise change the world directly.** Three calls cover it, and none of them names what the world
is kept in: `world.put(collection, record)` adds one, `world.change(collection, key, changes, by=...)`
alters one, `world.drop(collection, key, by=...)` removes one. Use the direct route only for states no
tool can produce: a record already in a condition the agent could never create itself.

**Collections are not all lists.** A collection held in a store gives a list of records; one the agent's own code
keeps is often a mapping, and iterating it yields keys rather than records. Look with `inspect_world`
before writing against one.

**Fill every field an existing record has.** Read one back with `inspect_world` and give your new record
the same fields, timestamps included. A column the store requires and you leave out, or set to nothing,
fails the insert and the scenario dies in its own setup:

```
NotNullViolation: null value in column "issued_at" of relation "otp_codes" violates not-null constraint
```

Measured on a real run: four of five calls lost that way, to the "issued_at" column on a code row and
"taken_at" on a past trip. If a field is a time, give it a plausible one rather than nothing.

**Write a value of the type the column actually holds.** A true or false field takes `True` or `False`,
never `1` or `0`. Some stores accept either and some reject the number outright, and the scenario then dies
in its own setup before the conversation starts: measured on a real run, three of five calls failed with
`column "phone_verified" is of type boolean but expression is of type smallint`, because the setup wrote
`1`. Records you read back may display as `1` and `0`; that is how they are shown, not what the column is.

`references/world-api.md` has the exact signatures, the `key=` caveat, how to handle either shape, and
a worked `ready_code`.

One exception. Where the contract says the target's store is hardcoded and process-local, with no
configuration seam, `setup_code` cannot alter target records, because the world and the live target
are separate copies. Use only records already in the frozen base, keep setup empty for them, and
settle outcomes from the captured calls. If coverage needs state the base lacks, report that the
target needs a seed or reset seam rather than writing a scenario that cannot run.

## The solution is not optional

Every scenario carries what a correct agent would do. It is never run against the agent under test.
It exists to prove the scenario can be passed at all, and it is what gate 2 uses.

Work it out with `try_calls` before you submit: run the calls, pass your `setup_code` so you see the
world the agent will face, and confirm the sub-goals you name respond to the state they leave.

**A one-call solution is almost always wrong, and is refused.** The agent does not begin knowing who
it is dealing with or what is true of their account, so before the step that resolves the scenario it
has to find out: identify the person, read the record, check the state that decides the answer. Those
lookups belong in the solution, and the sub-goals have to name them.

Refusals and handoffs are where this goes wrong most often, because the terminal call looks so
obviously like the point. It is not. **Deciding** to refuse is the point, and a decision never reached
from evidence was never tested.

## Realistic values

Placeholder data makes a paid run look like a demo, and several kinds are refused outright.

Recognisable stand-ins are refused outright, and `references/refusals.md` lists which. Two rules go
beyond what any check can see:

- **Keep every fact internally consistent.** The persona, the fixture, the records the setup creates
  and the instruction must all describe the same person. A detail in the persona that does not match
  the record the agent will find is a scenario that fails for its own reasons.
- **Vary the outcome as well as the wording.** Success, refusal, correction, ambiguity, retry, stale
  state, unavailable dependency and recovery should not all share one happy-path fixture.

## One coherent terminal outcome

Do not combine branches whose correct outcomes stop one another. A scenario that asks the agent to
hand off an out-of-scope request must not also require a transaction to finish afterwards. A scenario
that correctly refuses, escalates, cancels or ends the exchange must not carry a sub-goal for work
that only happens when it continues.

**An outcome the harness cannot observe is not a terminal outcome.** Where the agent can start
something whose completion happens elsewhere, test that it started it, and test the work that would
follow in a separate scenario.

Before keeping a scenario, read its instruction, solution and every named sub-goal as a single path.
If satisfying one sub-goal can correctly prevent another from being reached, split them. Never add an
unrelated sub-goal merely to make every scenario exercise a tool.

## Two scenarios differ only if the right answer differs

Not if the wording differs. "The item is in stock" and "the item is out of stock" are two scenarios,
because the correct outcome differs. Two polite requests for the same thing are one scenario written
twice.

Changing who is asking, where they are going, or which option they pick does **not** make a second
scenario. The agent does the same things in the same order and the same checks decide the result; all
that changed is the noun. Ten of those look like coverage in a list and are one test.

Vary the person **within** a scenario you were already going to write, never to produce another one.
A suite where everybody is calm and cooperative tests one kind of person, so let temperament and
communication style differ across the suite. That is diversity inside the tests you have, not a source
of extra tests.

**A count you were given is a ceiling, not a quota.** If the agent's real branches run out at twelve,
submit twelve and say why. Padding buys rows that can never fail independently, and hides the branches
nobody wrote behind a suite that looks thorough. An even spread across every use case is a warning
sign, not a goal: real agents have use cases worth five scenarios and use cases worth one.

## If the contract is wrong

You will sometimes find the contract does not match what the world does: a tool that accepts a value
it was not recorded as accepting, a rule that is not really a rule. Correct it with `amend_contract`,
`add_rule`, `drop_rule` or `fix_tool`, and say why. Every amendment is recorded on the contract.

Never work around a contract you believe is wrong. A scenario written to dodge a bad contract hides
the problem, and everything built afterwards inherits it.

## Finishing

Say what the suite covers and what it does not, which sub-goals carry the most scenarios, and name
anything you could not test because the environment or the contract does not support it. Report the
honest number: a smaller suite that is entirely real is worth more than a padded one.
