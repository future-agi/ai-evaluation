"""A scenario: a delta on the base environment, and what must hold afterwards.

The base is built once — the world, the simulator's prompt, the catalogue of sub-goals. A
scenario changes a few values in that world, fills the prompt's slots, and names which sub-goals
must hold. It is not a template with values slotted into it; the harness writes each one.

It also carries a **solution**: what a correct agent would do. That is not decoration. It is what
proves, before the scenario is ever used, that the scenario can be passed at all and that its
checks are not vacuous — the two gates in ``prove.py``. Terminal-bench keeps its tasks honest the
same way, and it needs no model to do it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections import Counter
from math import ceil
from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

from .catalogue import Catalogue
from .simulator import variables_in


# What a fixture's `origin` may say, and which of those claim the scenario creates data itself.
FIXTURE_ORIGINS = ("seed", "generated", "mixed")
ORIGINS_THAT_CREATE = ("generated", "mixed")

# For an outbound call, how much the person already knows about why they are being rung. The order
# is the axis: told to expect it, half remembers, no idea at all. Named once so the schema a writer
# is offered, the suite's spread rule and the caller's own briefing cannot drift apart.
CALLER_AWARENESS = ("expecting", "partial", "unaware")
LEAST_AWARE = "unaware"

# Who picked up an outbound call. A person is the ordinary case and needs no saying, so empty means
# a person; "voicemail" means a mailbox answered and there is nobody on the line at all.
ANSWERED_BY = ("person", "voicemail")
VOICEMAIL = "voicemail"

# Which kind of mailbox answered. The style settles two things and nothing else: what the greeting
# says, and whether a tone follows it. Empty means a person's own greeting, which is the ordinary
# case and the one that carries the persona's name.
VOICEMAIL_STYLES = ("personal", "carrier", "operator", "full")
DEFAULT_VOICEMAIL_STYLE = "personal"

# The share of a suite a rare call condition may occupy: a mailbox answering, or a second voice in
# the room. One number, in one place, because the right value is a product judgement rather than a
# technical one and it will be argued about.
#
# A ceiling and nothing else. There is deliberately no floor: a suite with no mailbox at all is a
# legitimate suite, and requiring one would put a narrow test into every run whether it earned its
# place or not. The previous value was a sixth, which is 17 percent and not rare by any reading.
RARE_CONDITION_SHARE = 0.05

# The switch that removes mailboxes from a run altogether, for when they are not wanted at all
# rather than merely kept rare.
VOICEMAIL_SWITCH = "ALK_VOICEMAIL_SCENARIOS"
_OFF = ("0", "off", "false", "no")


def voicemail_enabled() -> bool:
    """Whether this run may write or place a call a mailbox answers.

    **On unless ``ALK_VOICEMAIL_SCENARIOS`` turns it off**, since a mailbox is a real thing that
    happens to a real outbound agent and a suite that never meets one has not tested that path.

    Off has to mean off at every layer, not just at the call: the writer is not told mailboxes exist,
    the fields that would ask for one are not offered, a scenario that names one anyway is refused,
    and nothing is exported to the call runtime. Anything less leaves a run that was asked not to
    generate them generating them and failing quietly instead.
    """
    return os.environ.get(VOICEMAIL_SWITCH, "1").strip().lower() not in _OFF


# The longest a second person's interjection may be. One line is the whole point: somebody in the
# room says something across the call, and the test is whether the agent notices and who it answers.
# A paragraph is a second caller, which is a different feature and not this one.
LONGEST_BYSTANDER = 200


class Step(BaseModel):
    """One action in a reference solution."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Source-backed agents often add trusted session state between the model-facing function
    # and the dependency API: internal identifiers, resolved lookups, priced results, and similar
    # must never be exposed as arguments the model supposedly chose.  A reference proof still
    # has to drive the real dependency so its database effects can be checked, so it may carry
    # that dependency payload separately.  Agent runs never read this field.
    environment_arguments: dict[str, Any] = Field(default_factory=dict)


class Persona(BaseModel):
    """The simulated caller, in the same shape used by existing voice scenarios.

    A persona controls how the caller pursues a scenario's task. The task itself remains on
    ``Scenario.instruction`` so the harness can vary either one without conflating them.
    """

    name: str = ""
    gender: str = ""
    age_group: str = ""
    occupation: str = ""
    location: str = ""
    personality: str = ""
    communication_style: str = ""
    # The first thing this person actually says. Voice agents often greet immediately; leaving
    # this to the simulator model produced generic "Hello?" turns and avoidable silence races.
    initial_message: str = ""
    keywords: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    accent: str = ""
    multilingual: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Optional deterministic voice policy for transactional scenarios. It keeps
    # caller facts realistic and varied while avoiding LLM role drift during a
    # long tool-heavy phone flow.
    scripted_caller: dict[str, Any] | None = None

    def described(self) -> bool:
        return bool(
            self.name
            or self.gender
            or self.age_group
            or self.occupation
            or self.location
            or self.personality
            or self.communication_style
            or self.keywords
            or self.languages
            or self.accent
            or self.metadata
        )

    def missing_profile_fields(self) -> list[str]:
        """The minimum needed for a scenario to exercise caller variation intentionally."""
        missing = [
            name
            for name, value in (
                ("name", self.name),
                ("personality", self.personality),
                ("communication_style", self.communication_style),
                ("initial_message", self.initial_message),
                ("accent", self.accent),
            )
            if not value.strip()
        ]
        if not self.languages:
            missing.append("languages")
        if not self.keywords:
            missing.append("keywords")
        return missing

    def format_persona(self) -> str:
        """A stable, human-readable profile the simulator can consistently embody."""
        parts = []
        identity = []
        for label, value in (
            ("Name", self.name),
            ("Gender", self.gender),
            ("Age Group", self.age_group),
            ("Occupation", self.occupation),
            ("Location", self.location),
        ):
            if value:
                identity.append(f"- {label}: {value}")
        if identity:
            parts.append("# YOUR IDENTITY\n\n" + "\n".join(identity))

        behavior = []
        if self.personality:
            behavior.append(f"- Personality: {self.personality}")
        if self.communication_style:
            behavior.append(f"- Communication Style: {self.communication_style}")
        if self.keywords:
            behavior.append("- Key Traits: " + ", ".join(self.keywords))
        if behavior:
            parts.append("# YOUR PERSONALITY & COMMUNICATION\n\n" + "\n".join(behavior))

        speech = []
        if self.languages:
            speech.append("- Language(s): " + ", ".join(self.languages))
        if self.accent:
            speech.append(f"- Accent: {self.accent}")
        if self.multilingual:
            speech.append(
                "- Switch languages naturally when the conversation calls for it."
            )
        if speech:
            parts.append("# LANGUAGE & SPEECH PATTERNS\n\n" + "\n".join(speech))

        if self.metadata:
            characteristics = [
                f"- {key.replace('_', ' ').title()}: {value}"
                for key, value in self.metadata.items()
            ]
            parts.append(
                "# ADDITIONAL CHARACTERISTICS\n\n" + "\n".join(characteristics)
            )
        return "\n".join(parts)


def _slug(name: str) -> str:
    """An ASCII key for ``name``, safe to send as a header value.

    Falls back to a digest rather than an empty string: an empty key would collapse every
    scenario in a job onto one idempotency key on the receiving side.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return cleaned or "scenario-" + hashlib.sha256(name.encode()).hexdigest()[:12]


def _decided_by(name: str) -> bool:
    """Whether this scenario is noisy, decided by its name so a rerun decides the same."""
    return hashlib.sha256((name or "").encode()).digest()[0] % 2 == 0


class Scenario(BaseModel):
    """One test: what changes, what is asked, what a correct agent does, what must hold."""

    name: str
    # How this scenario is identified on the wire. Derived from ``name``, which is already unique
    # across a suite and already a slug because it is the folder name. It ships as a header, so
    # anything outside ASCII is dropped and an empty result falls back to a digest.
    scenario_key: str = ""
    # Assigned by the platform when the scenario is pre-allocated. Never written here.
    scenario_id: str = ""
    use_case: str = ""
    # What makes this row different from its siblings in the same use case. Coverage is counted
    # on the pair, so a use case can carry many scenarios without any reading as a duplicate.
    branch: str = ""
    tests: str = ""

    # What this scenario changes about the world after it is reset, as code: a file defining
    # ``setup(world)``. Rows in a table were enough while every world was a database, and they
    # are not enough now — a scenario may need a service to start returning errors, a file to be
    # missing, a queue to be backed up. Code can express all of that; a table of rows cannot.
    setup_code: str = ""

    # Whether the world is actually ready for this scenario, as code: a file defining
    # ``ready(world)`` that answers with nothing when the world holds what this scenario
    # presumes, or a sentence saying what is missing.
    #
    # This is the precondition, and it is the difference between a real finding and a wasted
    # run: a scenario about the last five chocolates is only a test of the agent if there really
    # are five. Otherwise the agent fails for something we got wrong, and it looks like the
    # agent's fault.
    ready_code: str = ""

    # The task. For a conversational agent it fills the simulator prompt's instruction slot; for
    # a browser or coding agent it goes to the agent directly.
    instruction: str = ""
    # Who is making the request. This is deliberately separate from the task so a caller's
    # communication needs do not get buried in an unstructured instruction.
    persona: Persona | None = None
    # Anything else that prompt asks for, by slot name.
    variables: dict[str, str] = Field(default_factory=dict)
    # A readable declaration of which data makes this scenario real. ``setup_code`` remains the
    # executable delta; this is the index a person and the UI can inspect without reverse-
    # engineering Python. Typical keys are origin (seed/generated/mixed), identity, credentials,
    # location and account_state. It is intentionally open-ended across agent domains.
    fixture: dict[str, Any] = Field(default_factory=dict)

    # What a correct agent would do. Run by the gates, never by the agent under test.
    solution: list[Step] = Field(default_factory=list)

    # Which entries of the shared catalogue must hold. Named, not restated, so results roll up
    # across the suite: the same sub-goal failing in seven of twelve scenarios is one sentence.
    sub_goals: list[str] = Field(default_factory=list)

    max_turns: int = 10

    # Where this call is made from. A string names the place ("street", "vehicle", "retail"), and
    # True asks for noise while leaving the place to the fixture. Left unset it is decided from
    # the name, so a suite still covers both conditions but the same suite decides the same way
    # twice; a coin flip here made a seeded run unreproducible.
    background_noise: bool | str = ""

    # Whether the agent placed this call or answered it. Voice only: a chat is always started by
    # the person, so it stays inbound. An outbound caller has no opening request to make, which is
    # a different test of the agent rather than the same one with a reworded greeting.
    # Empty means defer to the run and then to the contract, which is where the agent's own
    # direction was identified. Defaulting it to "inbound" here would be written into the saved
    # document and silently outrank both of them.
    call_direction: str = ""
    # For an outbound call, how much this person already knows about why they are being rung:
    # "expecting", "partial" or "unaware". Unset means unaware, the case the agent must work
    # hardest for.
    caller_awareness: str = ""
    # Who answered. Empty or "person" is somebody picking up; "voicemail" is a mailbox answering,
    # which tests whether the agent notices it is talking to a machine and leaves a usable message
    # rather than running its interactive script at a recording. Outbound only: a mailbox cannot
    # answer a call the person placed themselves.
    answered_by: str = ""
    # Which kind of mailbox, when one answered: "personal" is the person's own recorded greeting,
    # "carrier" the network default that names nobody, "operator" a formal automated announcement,
    # and "full" a mailbox that cannot record at all. Only read where answered_by is "voicemail";
    # empty means personal.
    voicemail_style: str = ""
    # One line somebody else in the room says across this call, a child from the back seat, a
    # colleague at the next desk. Spoken over the caller's own audio partway through, so what is
    # tested is whether the agent notices a second voice, keeps answering the person it is talking
    # to, and does not treat the interruption as its caller's turn. Empty means nobody else speaks.
    bystander: str = ""

    # Slots the caller filled by the run rather than by the scenario. Listed so a template that
    # uses one is not rejected as unfillable at write time.
    RUNTIME_SLOTS: ClassVar[tuple[str, ...]] = ("channel", "situation")

    @model_validator(mode="after")
    def _identify(self) -> "Scenario":
        if not self.scenario_key:
            self.scenario_key = _slug(self.name)
        if self.background_noise == "":
            self.background_noise = _decided_by(self.name)
        return self

    def slots(self) -> dict[str, str]:
        """Every value this scenario offers the simulator prompt."""
        persona = {"persona": self.persona.format_persona()} if self.persona else {}
        runtime = {name: "" for name in self.RUNTIME_SLOTS}
        return {
            "instruction": self.instruction,
            **runtime,
            **self.variables,
            **persona,
        }


def validate_scenario(
    scenario: Scenario,
    catalogue: Catalogue,
    world_state: dict[str, list[dict[str, Any]]],
    simulator_prompt: str = "",
) -> list[str]:
    """Problems that make a scenario unusable, found without running anything.

    Whether it can actually be passed is a different question, and no amount of reading settles
    it. That is what the gates are for.
    """
    problems: list[str] = []
    if not scenario.name.strip():
        problems.append("no name")
    if not scenario.instruction.strip():
        problems.append("no instruction: there is nothing for the run to be about")
    if scenario.persona is not None and not scenario.persona.described():
        problems.append("persona has no details")
    elif scenario.persona is not None and (
        missing := scenario.persona.missing_profile_fields()
    ):
        problems.append("persona is incomplete: " + ", ".join(missing))
    elif scenario.persona is not None:
        # A persona in words of its own renders fine and then does nothing: no behaviour guidance
        # attaches, and the accent it names selects no voice.
        from .persona_guides import unrecognised

        problems.extend(unrecognised(scenario.persona.model_dump()))
    if not scenario.sub_goals:
        problems.append(
            "no sub_goals: nothing would be graded. Name the entries of the catalogue this "
            "scenario is meant to exercise"
        )
    if world_state and not scenario.fixture:
        problems.append(
            "no fixture manifest: declare the seed/generated/mixed data this scenario relies on"
        )
    elif scenario.fixture and str(
        scenario.fixture.get("origin") or ""
    ).lower() not in set(FIXTURE_ORIGINS):
        problems.append(
            "fixture.origin must be "
            + ", ".join(FIXTURE_ORIGINS[:-1])
            + f", or {FIXTURE_ORIGINS[-1]}"
        )
    elif (
        scenario.fixture
        and str(scenario.fixture.get("origin") or "").lower()
        in set(ORIGINS_THAT_CREATE)
        and not (scenario.setup_code or "").strip()
    ):
        # A fixture claiming data it never creates is the whole class of scenario that names a value
        # the scenario reads as self-sufficient, the world has none of it, and the agent has nothing
        # to answer with. Caught here because it is provable from the document alone.
        problems.append(
            f"fixture.origin is {scenario.fixture.get('origin')!r}, which claims this scenario "
            "creates data, but setup_code is empty. Either seed everything the fixture names, or "
            "declare origin 'seed' and use only records that already exist"
        )

    unknown = sorted(set(scenario.sub_goals) - catalogue.names())
    if unknown:
        problems.append(
            f"sub_goals not in the catalogue: {', '.join(unknown)}. Use the shared names, or add "
            f"them to the catalogue first. It has: {', '.join(sorted(catalogue.names())) or 'none'}"
        )

    # setup_code and ready_code are not read here. Whether they work is not a question reading
    # them can answer, and running them is exactly what the first gate does.
    if scenario.setup_code.strip() and "def setup(" not in scenario.setup_code:
        problems.append("setup_code must define setup(world)")
    if scenario.ready_code.strip() and "def ready(" not in scenario.ready_code:
        problems.append("ready_code must define ready(world)")

    if simulator_prompt:
        unfilled = sorted(variables_in(simulator_prompt) - set(scenario.slots()))
        if unfilled:
            problems.append(
                f"the simulator prompt asks for {', '.join(unfilled)}, which this scenario does "
                "not supply. An unfilled slot reaches the caller verbatim"
            )

    if not scenario.solution:
        problems.append(
            "no solution: without the actions a correct agent would take, there is no way to "
            "show this scenario can be passed at all"
        )
    problems.extend(fixture_problems(scenario))
    problems.extend(answered_by_problems(scenario))
    problems.extend(voicemail_style_problems(scenario))
    problems.extend(voicemail_sub_goal_problems(scenario, catalogue))
    problems.extend(bystander_problems(scenario))
    problems.extend(_world_credential_problems(scenario, world_state))
    problems.extend(self_sufficiency_problems(scenario))
    problems.extend(alignment_problems(scenario, world_state))
    problems.extend(hollow_scenario_problems(scenario))
    problems.extend(naming_problems(scenario))
    return problems


def answered_by_problems(scenario: Scenario) -> list[str]:
    """Whether what answered this call is a thing that could have answered it.

    A mailbox only exists on a call the agent placed, and the direction has to be stated on the
    scenario rather than left to the run: `call_direction` empty means defer to the contract, so a
    voicemail scenario that stays quiet about direction is one the run may legally make inbound,
    and then a mailbox is answering a call the person dialled.
    """
    chosen = str(scenario.answered_by or "").strip().lower()
    if not chosen:
        return []
    if chosen not in set(ANSWERED_BY):
        return [
            "answered_by must be "
            + ", ".join(ANSWERED_BY)
            + f", not {scenario.answered_by!r}"
        ]
    if chosen != VOICEMAIL:
        return []
    if not voicemail_enabled():
        return [
            f"answered_by {VOICEMAIL!r} is turned off for this run, so write a scenario somebody "
            "answers instead"
        ]
    if str(scenario.call_direction or "").strip().lower() != "outbound":
        return [
            "answered_by is 'voicemail', which only happens on a call the agent placed, so this "
            "scenario must state call_direction 'outbound' itself. Left unset, the direction is "
            "taken from the contract and a mailbox would be answering a call the person dialled"
        ]
    return []


def voicemail_sub_goal_problems(scenario: Scenario, catalogue: Catalogue) -> list[str]:
    """Whether this mailbox scenario asks for something a mailbox call can produce.

    A sub-goal that needs a tool call cannot hold when nobody answers: the agent reaches most of its
    tools only once the person it called has said something, and on a mailbox nobody ever does. Six
    measured mailbox calls failed on exactly this, all four styles, every one of them reporting a tool
    "was not called" while the agent had behaved correctly. The scenario was wrong, not the agent, and
    a suite that records it as a failure is measuring nothing.

    Read from the check rather than from the name, because the name is the writer's word for it and
    the check is what decides. Only a check that fails when a call is *absent* is caught; one that
    fails when a call is present is a mailbox sub-goal worth having, since talking to a machine is
    where an agent should stop calling things.
    """
    if str(scenario.answered_by or "").strip().lower() != VOICEMAIL:
        return []
    problems: list[str] = []
    for name in scenario.sub_goals:
        sub_goal = catalogue.named(name)
        if sub_goal is None:
            continue
        check = " ".join(str(sub_goal.check or "").split())
        if not check or "calls" not in check:
            continue
        # The shape writers actually produce, taken from a real run rather than imagined: filter the
        # calls into a local, then fail when that local is empty.
        #
        #     checks = [c for c in calls if c.name == "get_booking_status" and c.ok]
        #     if not checks:
        #         return "Booking status was not retrieved"
        #
        # An earlier version of this looked for `not any(` and `if not calls`, matched neither, and let
        # two mailbox scenarios through on a live run. What all these forms share is a negative test
        # that fails when nothing was called, so that is what to look for. A check that fails when a
        # call WAS made has no negation and stays welcome, since a mailbox is where an agent should
        # stop calling things.
        needs_a_call = (
            "not any(" in check
            or "not called" in check
            or "calls == []" in check
            or re.search(r"if not [A-Za-z_][A-Za-z0-9_]*\s*:", check) is not None
            or re.search(r"len\([A-Za-z_][A-Za-z0-9_]*\) *== *0", check) is not None
        )
        if needs_a_call:
            problems.append(
                f"sub_goal {name!r} fails when a tool was not called, and on this scenario a mailbox "
                "answers, so the agent never gets the turn that leads it to call anything. Ask for "
                "what the agent can do with nobody on the line: that it recognised a machine, that "
                "the message it left says who is calling and why, that it stopped instead of asking "
                "questions. A sub-goal needing an answer marks a correctly handled mailbox as failed"
            )
    return problems


def voicemail_style_problems(scenario: Scenario) -> list[str]:
    """Whether the kind of mailbox named exists, and whether a mailbox answered at all.

    A style with nobody to play it is a scenario that reads as though it varies the mailbox and does
    not, which is worse than leaving it out, because the suite rule then counts a variation that
    never reaches the call.
    """
    chosen = str(scenario.voicemail_style or "").strip().lower()
    if not chosen:
        return []
    if chosen not in set(VOICEMAIL_STYLES):
        return [
            "voicemail_style must be "
            + ", ".join(VOICEMAIL_STYLES)
            + f", not {scenario.voicemail_style!r}"
        ]
    if str(scenario.answered_by or "").strip().lower() != VOICEMAIL:
        return [
            f"voicemail_style is {chosen!r} but answered_by is not 'voicemail', so no mailbox "
            "answers and nothing plays it. State answered_by 'voicemail', or leave the style out"
        ]
    return []


def bystander_problems(scenario: Scenario) -> list[str]:
    """Whether a second voice in the room is one that could be there, and is one line.

    Nobody speaks across a mailbox: `answered_by` voicemail means the other end is a recording, so
    there is no room and no caller for a bystander to talk over.
    """
    said = str(scenario.bystander or "").strip()
    if not said:
        return []
    problems: list[str] = []
    if str(scenario.answered_by or "").strip().lower() == VOICEMAIL:
        problems.append(
            "bystander is set on a scenario a mailbox answers, and a recording has no room for "
            "somebody to speak across. Drop one of the two"
        )
    if len(said) > LONGEST_BYSTANDER:
        problems.append(
            f"bystander is {len(said)} characters, over {LONGEST_BYSTANDER}. It is one thing said "
            "across the call, not a second conversation"
        )
    return problems


def _world_credential_problems(
    scenario: Scenario, world_state: dict[str, list[dict[str, Any]]]
) -> list[str]:
    """Reject caller credentials paired with the wrong world identity.

    Hosted source authoring cannot execute a repository's runtime-owned tools until the sealed
    bundle is provisioned.  Static scenario validation must therefore catch identity-bound test
    credentials that a deferred reference rehearsal cannot.  The matching is deliberately
    schema-shaped rather than application-shaped: any collection containing a phone-like
    identity and an OTP/verification code participates.
    """
    credentials: dict[str, set[str]] = {}
    for collection, rows in world_state.items():
        if not any(token in collection.lower() for token in ("otp", "verification")):
            continue
        for row in rows:
            phone = next(
                (
                    str(value)
                    for key, value in row.items()
                    if "phone" in str(key).lower() and value not in (None, "")
                ),
                "",
            )
            code = next(
                (
                    str(value).replace(" ", "")
                    for key, value in row.items()
                    if ("code" in str(key).lower() or "otp" in str(key).lower())
                    and re.fullmatch(r"\d{4,10}", str(value).replace(" ", ""))
                ),
                "",
            )
            if phone and code:
                credentials.setdefault(phone, set()).add(code)

    fixture = scenario.fixture or {}
    phones: set[str] = set()
    codes: set[str] = set()

    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child, item in value.items():
                collect(item, str(child))
        elif isinstance(value, list):
            for item in value:
                collect(item, key)
        elif "phone" in key.lower() and value not in (None, ""):
            phones.add(str(value))
        elif "otp" in key.lower() or key.lower() in {"code", "verification_code"}:
            candidate = str(value).replace(" ", "")
            if re.fullmatch(r"\d{4,10}", candidate):
                codes.add(candidate)

    collect(fixture)
    if scenario.persona is not None:
        collect(scenario.persona.metadata)
        collect(scenario.persona.scripted_caller or {})
    for step in scenario.solution:
        if "verify" in step.tool.lower() or "otp" in step.tool.lower():
            collect(step.arguments)

    problems: list[str] = []
    for phone in sorted(phones & credentials.keys()):
        wrong = codes - credentials[phone]
        if codes and wrong and not (codes & credentials[phone]):
            problems.append(
                "verification credential does not belong to the scenario caller "
                f"{phone}; inspect the world and use that identity's code"
            )
    return problems


def contract_sequence_problems(
    scenario: Scenario, hard_constraints: list[str]
) -> list[str]:
    """Catch reference solutions that hide required same-call state in a fixture.

    A dependency can accept a pre-seeded identifier even when the public agent API cannot. For
    a rule such as ``cancel_ride requires a booking_ref from this call``, require a producer
    (``book_ride``) earlier in the same reference solution instead of allowing setup code or
    environment-only arguments to make an impossible scenario look solvable.
    """
    problems: list[str] = []
    names = [step.tool for step in scenario.solution]
    pattern = re.compile(
        r"\b(?P<consumer>[a-z][a-z0-9_]*)\b\s+requires\b.*?\b"
        r"(?P<resource>[a-z][a-z0-9_]*(?:_id|_ref))\s+from this call\b",
        re.IGNORECASE,
    )
    for constraint in hard_constraints:
        found = pattern.search(constraint)
        if found is None:
            continue
        consumer = found.group("consumer").lower()
        lowered = [name.lower() for name in names]
        if consumer not in lowered:
            continue
        resource = re.sub(r"_(?:id|ref)$", "", found.group("resource").lower())
        stems = {resource, resource.removesuffix("ing")}
        before = lowered[: lowered.index(consumer)]
        produced = any(
            any(stem and stem in tool for stem in stems)
            and not tool.startswith(("get_", "list_", "find_", "cancel_"))
            for tool in before
        )
        if not produced:
            problems.append(
                f"{consumer} requires {found.group('resource')} from this call, but the "
                "reference solution does not create it first; do not hide it in setup or "
                "environment_arguments"
            )
    return problems


_WEAK_CODES = {
    "000000",
    "111111",
    "222222",
    "333333",
    "444444",
    "555555",
    "666666",
    "777777",
    "888888",
    "999999",
    "012345",
    "123456",
    "234567",
    "345678",
    "456789",
    "987654",
    "876543",
    "765432",
    "654321",
}


def _six_digit_values(scenario: Scenario) -> list[str]:
    """Likely one-time codes declared by a scenario, without treating phone digits as OTPs."""
    found: list[str] = []

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child, item in value.items():
                walk(item, str(child))
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif "otp" in key.lower() or key.lower() in {"code", "verification_code"}:
            found.extend(re.findall(r"(?<!\d)\d{6}(?!\d)", str(value)))

    walk(scenario.fixture)
    if scenario.persona:
        walk(scenario.persona.metadata)
        walk(scenario.persona.scripted_caller or {})
    for step in scenario.solution:
        walk(step.arguments)
        walk(step.environment_arguments)
    # Setup is code, so key-aware traversal is unavailable. Restrict matches to a nearby field
    # name instead of collecting six digits from a phone number or an unrelated identifier.
    found.extend(
        match.group(1)
        for match in re.finditer(
            r"(?:otp|verification[_ ]?code|['\"]code['\"])[^\n]{0,80}?(?<!\d)(\d{6})(?!\d)",
            scenario.setup_code,
            flags=re.IGNORECASE,
        )
    )
    return found


# A value the instruction hands the caller so they can say it back: a code, a reference, an account
# number, an id. Deliberately not named after any one domain, because the failure is the same
# whatever the agent does: the caller reads out something the agent then cannot find.
_QUOTED_VALUE = re.compile(
    r"(?<![\w-])(?=[A-Za-z-]*\d)[A-Za-z0-9][A-Za-z0-9-]{3,}(?![\w-])"
)

# Values that look quotable but are never records the agent looks up.
_NOT_A_RECORD = re.compile(
    r"^(?:\d{1,2}[:.]\d{2}|\d{1,4}(?:st|nd|rd|th)|20\d{2}|1?\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)$",
    re.IGNORECASE,
)


def _quotable_values(text: str) -> set[str]:
    """Tokens in a piece of text that read as a value somebody would be asked to repeat."""
    return {
        token
        for token in _QUOTED_VALUE.findall(text or "")
        if not _NOT_A_RECORD.match(token)
    }


# A value only has to be reachable if the caller is going to be asked for it. An address they are
# travelling to, or a price they are quoted, is the agent's to produce; a value they are told to say
# back is one the agent will check. Domain-neutral: the cue is the verb, not the kind of value.
_HANDED_OVER = re.compile(
    r"(?:say|give|read|quote|provide|confirm|tell|repeat|use|enter|supply)\b[^.\n]{0,70}?"
    r"(?<![\w-])((?=[A-Za-z-]*\d)[A-Za-z0-9][A-Za-z0-9-]{3,})(?![\w-])",
    re.IGNORECASE,
)


def _handed_to_caller(text: str) -> set[str]:
    """Values the instruction tells the caller to say back, which the agent will then check."""
    return {
        match.group(1)
        for match in _HANDED_OVER.finditer(text or "")
        if not _NOT_A_RECORD.match(match.group(1))
    }


def naming_problems(scenario: Scenario) -> list[str]:
    """Whether the name says what is tested, or only who the agent was dealing with.

    The folder name is how a failure is read weeks later. A caller's name in it says the caller was
    carrying the difference the test should have been carrying, which is the same mistake as planning
    a second scenario because the person could be somebody else. Measured on an earlier suite: twelve
    of thirty one were still named for the caller after the skill asked them not to be, which is why
    this is checked rather than requested.
    """
    caller = str(getattr(scenario.persona, "name", "") or "").strip().lower()
    if not caller:
        return []
    # Each part of the name, not the whole string: "marcus vance" is never a token of
    # `refuse_expired_card_marcus`, so matching the full name lets every first-name suffix through.
    parts = {part for part in caller.split() if len(part) > 2}
    written = set(scenario.name.lower().replace("-", " ").replace("_", " ").split())
    named_in = sorted(parts & written)
    if not named_in:
        return []
    return [
        f"the name contains the person's own name ({', '.join(named_in)}). Name it for the behaviour "
        "under test, so a red result says which rule broke rather than who the agent was dealing "
        "with, and so the suite sorts by what it covers rather than by who appeared in it"
    ]


def hollow_scenario_problems(scenario: Scenario) -> list[str]:
    """Whether the scenario tests reaching an outcome, or only the outcome itself.

    A reference solution of one call, graded by one sub-goal naming that same call, is passed by an
    agent that makes that call the moment it answers, having established nothing. Measured on a
    suite of a hundred, eleven scenarios were a single `transfer_to_human` step graded by a single
    `transferred_to_human` sub-goal, differing from each other only in the pretext, and every one of
    them was passed by an agent that transfers every caller on arrival.

    The bar is in the write skill and was not enough on its own, so it is checked here.
    """
    if len(scenario.solution) > 1:
        return []
    if not scenario.solution:
        return []
    return [
        "the reference solution is a single call and there is nothing the agent has to establish "
        "first, so an agent that makes that call on arrival passes without doing any of the work. "
        "Either the solution shows how the outcome is reached, gathering what the decision depends "
        "on before making it, or this is not a scenario"
    ]


def alignment_problems(
    scenario: Scenario, world_state: dict[str, list[dict[str, Any]]] | None = None
) -> list[str]:
    """Whether the values the caller is told are values the world actually holds.

    The failure this exists for, seen across a whole suite: an instruction telling the caller a
    verification code, a reference or an account number that the scenario never seeds and the world
    never had. The call cannot succeed however well the agent behaves, and the result is reported as
    a finding about the agent when it is a finding about the scenario.

    Deliberately domain-neutral. A code, a booking reference, a policy number and an order id all
    fail the same way, so the rule is about values rather than about any one kind of value: anything
    the instruction hands the caller has to be somewhere the agent can reach, which means this
    scenario's `setup_code` or the world it starts from. A fixture entry is not enough, because a
    fixture describes what a scenario relies on and only `setup_code` changes what is there.
    """
    told = _handed_to_caller(scenario.instruction)
    if not told:
        return []
    reachable = _quotable_values(scenario.setup_code or "")
    for step in scenario.solution:
        reachable |= _quotable_values(json.dumps(step.arguments, default=str))
        reachable |= _quotable_values(
            json.dumps(step.environment_arguments, default=str)
        )
    if world_state:
        reachable |= _quotable_values(json.dumps(world_state, default=str)[:200000])
    missing = sorted(told - reachable)
    if not missing:
        return []
    return [
        "the instruction gives the caller "
        + ", ".join(missing)
        + " to say back, and neither setup_code nor the world holds "
        + ("them" if len(missing) > 1 else "it")
        + ". Seed what the caller is told, or tell them what is seeded. Naming a value in fixture "
        "only declares it: setup_code is what the world ends up holding"
    ]


# What a setup does to the world, told apart by which call it makes. `put` adds a record and
# `call` drives a tool that produces one; `change` and `drop` only touch what was already there.
_CREATES_A_RECORD = re.compile(r"world\.(?:put|call)\s*\(")
_ONLY_TOUCHES_EXISTING = re.compile(r"world\.(?:change|drop)\s*\(")


def self_sufficiency_problems(scenario: Scenario) -> list[str]:
    """Whether this scenario owns the records its outcome turns on, or borrows them.

    A setup that only adjusts rows it did not create is building the test on state it does not
    control: the row belongs to the frozen base, so a second scenario adjusting the same row is
    testing the same record from two directions and neither describes a world it owns. Measured on
    a fan-out suite of 86, sixty seven were one or two `world.change` calls against base rows, four
    scenarios deep on the same rider, and the reused verification codes were the visible symptom of
    it.

    An empty setup stays legal. That is the documented case where the target's store is
    process-local with no seam, so the scenario cannot alter it and says so by touching nothing.
    """
    body = (scenario.setup_code or "").strip()
    if not body:
        return []
    if _CREATES_A_RECORD.search(body):
        return []
    if not _ONLY_TOUCHES_EXISTING.search(body):
        return []
    return [
        "setup_code only adjusts records that were already there and creates none of its own, so "
        "this scenario shares its data with every other scenario that touches the same records. "
        "Create what the outcome turns on: its own person, its own record, its own code, with "
        "world.put or by driving the agent's own tool. Shared reference data a whole world sits on "
        "can be read as it is, but the thing being tested has to belong to this scenario"
    ]


def _predictable(code: str) -> bool:
    """Whether a one-time code is one nobody would be issued.

    The hand-kept list of obvious ones caught `111111` and `123456` and let `000111` through, which then
    turned up twice in a 200-scenario suite. Tested as a property instead: a code built from one or two
    digits, or one that simply counts up or down, is a placeholder however it is arranged.
    """
    if not code.isdigit() or len(code) < 4:
        return code in _WEAK_CODES
    if len(set(code)) <= 2:
        return True
    steps = {ord(later) - ord(earlier) for earlier, later in zip(code, code[1:])}
    if steps in ({1}, {-1}):
        return True
    return code in _WEAK_CODES


def fixture_problems(scenario: Scenario) -> list[str]:
    """Reject demo-shaped data before a paid run makes it look like production traffic."""
    problems: list[str] = []
    codes = _six_digit_values(scenario)
    weak = sorted({code for code in codes if _predictable(code)})
    if weak:
        problems.append(
            "fixture uses predictable verification code(s): "
            + ", ".join(weak)
            + ". Generate a different non-sequential six-digit value for this scenario"
        )
    written = json.dumps(
        {
            "instruction": scenario.instruction,
            "persona": scenario.persona.model_dump() if scenario.persona else {},
            "fixture": scenario.fixture,
            "setup": scenario.setup_code,
        },
        default=str,
    ).lower()
    clichés = [
        value
        for value in ("test user", "john doe", "jane doe", "123 main street")
        if value in written
    ]
    if clichés:
        problems.append("fixture contains placeholder demo data: " + ", ".join(clichés))
    card_endings = sorted(
        set(
            re.findall(
                r"(?:last4|card_last4|payment_last4)[^\n]{0,30}?[\"']?(0000|1111|1234|4242|4444)[\"']?",
                written,
            )
        )
    )
    if card_endings:
        problems.append(
            "fixture uses placeholder payment-card ending(s): "
            + ", ".join(card_endings)
        )
    spoken_card_endings = sorted(
        set(
            re.findall(
                r"(?:ending(?:\s+in)?|last\s+four(?:\s+digits)?(?:\s+are)?)\D{0,12}"
                r"(0000|1111|1234|4242|4444)",
                written,
            )
        )
    )
    if spoken_card_endings:
        problems.append(
            "fixture/instruction uses placeholder payment-card ending(s): "
            + ", ".join(spoken_card_endings)
        )
    demo_ids = sorted(
        value
        for value in ("ub12345678", "booking123", "booking_123", "test123")
        if value in written
    )
    demo_ids.extend(
        re.findall(r"\b(?:ub_[a-z]+_0*1|pay_[a-z]+(?:_[a-z]+)*0*1)\b", written)
    )
    demo_ids = sorted(set(demo_ids))
    if demo_ids:
        problems.append(
            "fixture uses placeholder transaction identifier(s): " + ", ".join(demo_ids)
        )
    return problems


def rare_event_ceiling(suite_size: int) -> int:
    """The most scenarios in a suite of this size that may carry one rare call condition.

    Rounded up, so a small suite is allowed one rather than none: a suite of ten would otherwise be
    permitted half a mailbox, and refusing the only interesting call in a short suite is worse than
    allowing one in ten. It grows from there, so 50 allows 3 and 200 allows 10.
    """
    return max(1, ceil(suite_size * RARE_CONDITION_SHARE))


def suite_diversity_problems(scenarios: list[Scenario]) -> list[str]:
    """Whether a conversational suite represents meaningfully different people and data."""
    if len(scenarios) < 4:
        return []
    problems: list[str] = []
    personas = [one.persona for one in scenarios if one.persona]
    names = [one.name.strip().lower() for one in personas if one and one.name.strip()]
    unique_names = len(set(names))
    required_names = min(len(scenarios), max(3, ceil(len(scenarios) * 0.9)))
    if unique_names < required_names:
        repeated = [name for name, count in Counter(names).items() if count > 2]
        problems.append(
            f"only {unique_names} distinct caller names across {len(scenarios)} scenarios; "
            f"need at least {required_names}"
            + (f". Overused: {', '.join(repeated)}" if repeated else "")
        )
    openings = [
        one.initial_message.strip().lower()
        for one in personas
        if one and one.initial_message.strip()
    ]
    if len(set(openings)) != len(openings):
        problems.append("caller opening messages repeat verbatim across scenarios")
    locations = {
        one.location.strip().lower() for one in personas if one and one.location.strip()
    }
    if len(scenarios) >= 8 and len(locations) < 3:
        problems.append(
            f"only {len(locations)} persona locations across {len(scenarios)} scenarios; need 3"
        )
    # An outbound suite that is all one awareness tests one opening repeatedly. Enforced rather than
    # asked for: told to prefer `unaware`, writers made it the default and produced seven of eight,
    # and told to cover more than one they had settled on `expecting` instead. Both leave two thirds
    # of the opening untested.
    outbound = [one for one in scenarios if one.call_direction == "outbound"]
    if len(outbound) >= 3:
        spread = Counter(one.caller_awareness or LEAST_AWARE for one in outbound)
        if len(spread) < 2:
            problems.append(
                f"all {len(outbound)} outbound scenarios are caller_awareness "
                f"{next(iter(spread))!r}; cover at least two of "
                + ", ".join(CALLER_AWARENESS)
            )
        elif max(spread.values()) > ceil(len(outbound) * 0.7):
            worst, count = spread.most_common(1)[0]
            problems.append(
                f"{count} of {len(outbound)} outbound scenarios are caller_awareness {worst!r}; "
                "keep any one of "
                + ", ".join(CALLER_AWARENESS)
                + " under 70 percent of them"
            )
        if not spread.get(LEAST_AWARE):
            problems.append(
                f"no outbound scenario has caller_awareness {LEAST_AWARE!r}, the one that tests whether "
                "the agent says who it is and why it called before asking for anything"
            )
    # A mailbox tests one narrow thing: that the agent notices nobody is listening. It is worth a
    # few scenarios and never a theme, because a suite of mailboxes learns nothing about the agent
    # talking to people.
    mailboxes = [one for one in scenarios if one.answered_by == VOICEMAIL]
    allowed = rare_event_ceiling(len(scenarios))
    if len(mailboxes) > allowed:
        problems.append(
            f"{len(mailboxes)} of {len(scenarios)} scenarios are answered_by {VOICEMAIL!r}; keep "
            f"them to at most {allowed} here"
        )
    # A second voice in the room is a rare event, and a suite where it keeps happening is testing
    # that instead of testing the agent. Same ceiling as mailboxes, for the same reason.
    bystanders = [one for one in scenarios if str(one.bystander or "").strip()]
    if len(bystanders) > allowed:
        problems.append(
            f"{len(bystanders)} of {len(scenarios)} scenarios have a bystander speaking; keep them "
            f"to at most {allowed} here"
        )
    # And where there is more than one, they have to be different mailboxes. Distinct greetings is
    # what can be measured; the shapes worth covering are named in the voice skill.
    #
    # Two, not three. Three was written when the ceiling was a sixth of the suite and it fitted in
    # eighteen scenarios; at a twentieth it needs forty one, which put the rule beyond every suite we
    # run. Two mailboxes wearing the same style is the same waste the rule was written for.
    if len(mailboxes) >= 2:
        greetings = {
            " ".join(
                (one.persona.initial_message if one.persona else "").lower().split()
            )
            for one in mailboxes
        }
        if len(greetings - {""}) < 2:
            problems.append(
                f"all {len(mailboxes)} voicemail scenarios use the same greeting; vary it, since a "
                "named personal mailbox, a carrier mailbox with no name, a full mailbox and a long "
                "greeting are four different tests of the agent"
            )
        # The style is the stronger axis, because it also decides whether a tone follows the
        # greeting, and a suite of one style never tests the agent against a mailbox it cannot
        # leave a message on.
        styles = {
            str(one.voicemail_style or DEFAULT_VOICEMAIL_STYLE).strip().lower()
            for one in mailboxes
        }
        if len(styles) < 2:
            problems.append(
                f"all {len(mailboxes)} voicemail scenarios are voicemail_style "
                f"{next(iter(styles))!r}; cover at least two of "
                + ", ".join(VOICEMAIL_STYLES)
            )
    # A code naturally appears several times inside one scenario (fixture, caller script,
    # reference verify call). Diversity is about reuse *between* callers, not repeated mention
    # of the same fact inside one test.
    codes = [
        code for scenario in scenarios for code in set(_six_digit_values(scenario))
    ]
    duplicated_codes = sorted(
        code for code, count in Counter(codes).items() if count > 1
    )
    if duplicated_codes:
        problems.append(
            "verification codes are reused across scenarios: "
            + ", ".join(duplicated_codes)
        )
    setups = [signature for one in scenarios if (signature := _setup_signature(one))]
    if len(set(setups)) != len(setups):
        problems.append("identical scenario setup data is reused more than once")
    return problems


def _setup_signature(scenario: Scenario) -> str:
    """Comparable setup code, excluding the generated no-op function/documentation."""
    source = scenario.setup_code.strip()
    if not source:
        return ""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return " ".join(source.split())
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)), None
    )
    if function is None:
        return " ".join(source.split())
    meaningful = [
        node
        for node in function.body
        if not isinstance(node, ast.Pass)
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    return (
        "" if not meaningful else ast.dump(ast.Module(body=meaningful, type_ignores=[]))
    )
