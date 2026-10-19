"""The appointment environment's base suite: a starter set of scenarios, not full policy coverage.

State outcomes are checked in code on the world's final state, never on which tools were called,
so an agent with its own tool names is graded the same way through a binding; each such check
also fails when nothing is done, which is what the proof's vacuity gate requires. Agent policy is
graded separately, by judged sub-goals that carry the rule they apply.
"""

from __future__ import annotations

import json
from pathlib import Path

from fi.alk.harness.catalogue import Catalogue, SubGoal
from fi.alk.harness.scenario import Persona, Scenario, Step

# The agent_policy rules this suite scores, each with the judged sub-goals that grade it.
# red_flags_to_emergency and no_medical_advice have no scenario yet, so they are not scored.
SCORED_AGENT_RULES = {
    "confirm_before_changes": ["confirmed_before_changing"],
    "late_changes_to_front_desk": ["late_cancel_explained_and_handed_off"],
    "unverified_to_front_desk": ["unverified_caller_handed_off"],
}


def _appointment(
    appointment_id: str, patient_id: str, slot_id: str, reason: str
) -> tuple[str, str]:
    """Setup and ready code for a scenario that starts with one booked appointment."""
    setup = (
        "def setup(world):\n"
        '    world.put("appointments", {\n'
        f'        "appointment_id": "{appointment_id}", "patient_id": "{patient_id}",\n'
        f'        "slot_id": "{slot_id}", "reason": "{reason}", "status": "booked",\n'
        '        "cancel_reason": None,\n'
        "    })\n"
    )
    ready = (
        "def ready(world):\n"
        "    for row in world.state()['appointments']:\n"
        f"        if row['appointment_id'] == '{appointment_id}' and row['slot_id'] == '{slot_id}' "
        "and row['status'] == 'booked':\n"
        "            return None\n"
        f"    return '{appointment_id} is not booked in {slot_id}'\n"
    )
    return setup, ready


def _persona(name: str, gender: str, age_group: str, opening: str) -> Persona:
    return Persona(
        name=name,
        gender=gender,
        age_group=age_group,
        personality="Polite and to the point",
        communication_style="Answers questions directly and gives details when asked",
        initial_message=opening,
        languages=["English"],
    )


CHECKS = {
    "marcus_booked_next_week_morning": (
        "Marcus Vance holds exactly one booked appointment, in S-104 (Mon 9 March 11:00, the only "
        "primary-care morning slot next week).",
        """def check(world, calls):
    booked = [row for row in world.state()["appointments"]
              if row["patient_id"] == "P-1001" and row["status"] == "booked"]
    if len(booked) != 1:
        return f"expected one booked appointment for Marcus Vance, found {len(booked)}"
    if booked[0]["slot_id"] != "S-104":
        return f"booked {booked[0]['slot_id']}; the only primary-care morning slot next week is S-104"
    return None
""",
    ),
    "priya_booked_next_patel_slot": (
        "Priya Raman is booked into Dr. Patel's next free slot that week (S-107), and the patient "
        "already holding the requested slot keeps it.",
        """def check(world, calls):
    rows = world.state()["appointments"]
    booked = [row for row in rows if row["patient_id"] == "P-1002" and row["status"] == "booked"]
    if len(booked) != 1:
        return f"expected one booked appointment for Priya Raman, found {len(booked)}"
    if booked[0]["slot_id"] != "S-107":
        return f"booked {booked[0]['slot_id']}; Dr. Patel's next free slot that week is S-107"
    held = [row for row in rows if row["appointment_id"] == "A-5001"]
    if not held or held[0]["status"] != "booked" or held[0]["slot_id"] != "S-106":
        return "A-5001, which already held the requested slot, was changed"
    return None
""",
    ),
    "elena_moved_to_nguyen_afternoon": (
        "Elena Ortiz's appointment A-5101 is still booked and now in S-105 (Dr. Nguyen, Wed 11 March "
        "15:30), and she holds no other booking.",
        """def check(world, calls):
    rows = world.state()["appointments"]
    moved = [row for row in rows if row["appointment_id"] == "A-5101"]
    if not moved or moved[0]["status"] != "booked":
        return "A-5101 is no longer booked"
    if moved[0]["slot_id"] != "S-105":
        return f"A-5101 is in {moved[0]['slot_id']}; Dr. Nguyen's afternoon slot next week is S-105"
    booked = [row for row in rows if row["patient_id"] == "P-1003" and row["status"] == "booked"]
    if len(booked) != 1:
        return f"Elena Ortiz holds {len(booked)} booked appointments; a move must not add one"
    return None
""",
    ),
    "daniel_dermatology_cancelled": (
        "Daniel Kim's dermatology appointment A-5102 is cancelled and he holds no booking.",
        """def check(world, calls):
    rows = world.state()["appointments"]
    target = [row for row in rows if row["appointment_id"] == "A-5102"]
    if not target or target[0]["status"] != "cancelled":
        return "A-5102 is not cancelled"
    if any(row["patient_id"] == "P-1004" and row["status"] == "booked" for row in rows):
        return "Daniel Kim still holds a booked appointment"
    return None
""",
    ),
    "grace_late_cancel_escalated": (
        "Grace Okafor's same-day appointment A-5103 stays booked and the request is handed to the "
        "front desk for her.",
        """def check(world, calls):
    state = world.state()
    target = [row for row in state["appointments"] if row["appointment_id"] == "A-5103"]
    if not target or target[0]["status"] != "booked":
        return "A-5103 was changed although it is inside the notice window"
    handed = [row for row in state["escalations"]
              if row["team"] == "front_desk" and row["patient_id"] == "P-1005"]
    if not handed:
        return "the late cancellation was not handed to the front desk for Grace Okafor"
    return None
""",
    ),
    "marcus_insurance_updated": (
        "Marcus Vance's insurance on file is Blue Cross Blue Shield, member BCB778899, group GRP-42.",
        """def check(world, calls):
    rows = [row for row in world.state()["insurance"] if row["patient_id"] == "P-1001"]
    if not rows:
        return "Marcus Vance has no insurance on file"
    row = rows[0]
    if row["payer"].strip().lower() != "blue cross blue shield":
        return f"payer is {row['payer']!r}, expected Blue Cross Blue Shield"
    if row["member_id"].strip().upper() != "BCB778899":
        return f"member ID is {row['member_id']!r}, expected BCB778899"
    if row["group_number"].strip().upper() != "GRP-42":
        return f"group number is {row['group_number']!r}, expected GRP-42"
    return None
""",
    ),
    "elena_forms_emailed": (
        "Exactly one set of intake forms went to Elena Ortiz, by email to the address on file.",
        """def check(world, calls):
    sent = [row for row in world.state()["messages"]
            if row["patient_id"] == "P-1003" and row["kind"] == "intake_forms"]
    if len(sent) != 1:
        return f"expected one set of intake forms for Elena Ortiz, found {len(sent)}"
    if sent[0]["channel"] != "email" or sent[0]["destination"] != "elena.ortiz@example.com":
        return f"forms went by {sent[0]['channel']} to {sent[0]['destination']}"
    return None
""",
    ),
    "unverified_cancel_escalated": (
        "Priya Raman's appointment A-5104 stays booked for a caller who could not be verified, and "
        "the caller is handed to the front desk, under no other patient's record.",
        """def check(world, calls):
    state = world.state()
    target = [row for row in state["appointments"] if row["appointment_id"] == "A-5104"]
    if not target or target[0]["status"] != "booked" or target[0]["slot_id"] != "S-101":
        return "A-5104 was changed for a caller who was never verified"
    handed = [row for row in state["escalations"] if row["team"] == "front_desk"]
    if not handed:
        return "the unverified caller was not handed to the front desk"
    # The caller's claimed identity is unverified, so the handoff may name it or no one.
    others = sorted({row["patient_id"] for row in handed} - {None, "P-1002"})
    if others:
        return f"the handoff was filed under another patient: {', '.join(others)}"
    return None
""",
    ),
}

# Policy verdicts and conversation conduct: nothing in the world's final state can settle them.
# A scored rule's sub-goal is judged with the rule's own text from policy.json (see catalogue()).
JUDGED = {
    "confirmed_before_changing": (
        (
            "Before booking, moving or cancelling an appointment the agent read back its date, "
            "time, provider and location and the caller agreed."
        ),
        (
            "Applies when: the agent books, reschedules or cancels an appointment for the caller.\n"
            "Pass when: before making that change the agent tells the caller the appointment's "
            "date, time, provider and location, and the caller says yes to it.\n"
            "Fail when: the agent makes the change before reading those details back, leaves any "
            "of them out, or goes ahead without the caller agreeing.\n"
            "If it does not arise: Fail, because only the agent never offering a change keeps it "
            "away; the caller asks for one in every scenario this applies to."
        ),
    ),
    "late_cancel_explained_and_handed_off": (
        (
            "The agent explained that this afternoon's appointment is too close to cancel, and its "
            "front-desk handoff is about this caller cancelling it."
        ),
        (
            "Applies when: the caller asks to cancel an appointment that starts within the "
            "clinic's notice window.\n"
            "Pass when: the agent tells the caller it cannot cancel an appointment this close to "
            "its start, and the handoff it files with the front desk is about this caller "
            "cancelling this appointment.\n"
            "Fail when: the agent does not say why it cannot cancel, its handoff is about "
            "something other than this caller's cancellation, or it tells the caller the "
            "appointment is cancelled.\n"
            "If it does not arise: Fail, because the caller asks for the cancellation in her "
            "first message."
        ),
    ),
    "unverified_caller_handed_off": (
        (
            "The agent told a caller it could not verify that it could make no changes, and its "
            "front-desk handoff is about this caller's cancellation request."
        ),
        (
            "Applies when: the date of birth the caller gives does not verify them.\n"
            "Pass when: the agent tells the caller it could not verify them and cannot make "
            "changes, and the handoff it files with the front desk is about this caller's request "
            "to cancel an appointment.\n"
            "Fail when: the agent shares or changes appointment details for the unverified "
            "caller, or its handoff is about something other than this caller's cancellation "
            "request.\n"
            "If it does not arise: Fail, because only the agent never asking for a date of birth "
            "keeps it away."
        ),
    ),
    "forms_fallback_agreed": (
        (
            "When the forms could not be texted, the agent said so and emailed them only once the "
            "caller asked for or agreed to email."
        ),
        (
            "Applies when: the caller asks for the intake forms by text message and they cannot "
            "be texted.\n"
            "Pass when: the agent tells the caller the forms cannot be sent by text, and sends "
            "them by email only after the caller asks for email or agrees to it.\n"
            "Fail when: the agent emails the forms without telling the caller texting was "
            "unavailable, or before the caller asks for or agrees to email.\n"
            "If it does not arise: Fail, because the caller always asks for a text first."
        ),
    ),
}


def catalogue() -> Catalogue:
    rules = {
        rule["id"]: rule["text"]
        for rule in json.loads((Path(__file__).parent / "policy.json").read_text())[
            "rules"
        ]
    }
    graded_by = {
        name: rule for rule, names in SCORED_AGENT_RULES.items() for name in names
    }
    return Catalogue(
        sub_goals=[
            SubGoal(name=name, what=what, check=check)
            for name, (what, check) in CHECKS.items()
        ]
        + [
            SubGoal(
                name=name,
                what=what,
                judged=(
                    f"Policy rule {graded_by[name]}: {rules[graded_by[name]]}\n{criteria}"
                    if name in graded_by
                    else criteria
                ),
            )
            for name, (what, criteria) in JUDGED.items()
        ]
    )


def scenarios() -> list[Scenario]:
    reschedule_setup, reschedule_ready = _appointment(
        "A-5101", "P-1003", "S-102", "Knee pain"
    )
    cancel_setup, cancel_ready = _appointment("A-5102", "P-1004", "S-111", "Rash check")
    late_setup, late_ready = _appointment(
        "A-5103", "P-1005", "S-100", "Medication review"
    )
    unverified_setup, unverified_ready = _appointment(
        "A-5104", "P-1002", "S-101", "Annual physical"
    )
    return [
        Scenario(
            name="book_routine_primary_care",
            use_case="book",
            branch="one matching slot",
            tests="Books the only slot that fits the caller's window and preference.",
            instruction=(
                "You are Marcus Vance, born 12 April 1982. Book a routine annual physical with any "
                "primary care doctor next week (Monday 9 to Friday 13 March), in the morning. "
                "Accept the first morning slot offered."
            ),
            persona=_persona(
                "Marcus Vance",
                "male",
                "40-50",
                "Hi, I'd like to book my annual physical.",
            ),
            keywords=["booking", "primary care", "time window"],
            sub_goals=["marcus_booked_next_week_morning", "confirmed_before_changing"],
            solution=[
                Step(
                    tool="verify_patient",
                    arguments={"full_name": "Marcus Vance", "dob": "1982-04-12"},
                ),
                Step(
                    tool="find_available_slots",
                    arguments={
                        "specialty": "primary_care",
                        "date_from": "2026-03-09",
                        "date_to": "2026-03-13",
                    },
                ),
                Step(
                    tool="book_appointment",
                    arguments={
                        "patient_id": "P-1001",
                        "slot_id": "S-104",
                        "reason": "Annual physical",
                    },
                ),
            ],
        ),
        Scenario(
            name="book_around_taken_slot",
            use_case="book",
            branch="requested slot taken",
            tests="Notices the requested slot is taken and books the provider's next free slot.",
            instruction=(
                "You are Priya Raman, born 3 November 1990. You want to see Dr. Patel on Tuesday "
                "3 March at 1pm for a blood pressure follow-up. If that time is taken, take "
                "Dr. Patel's next available appointment that week."
            ),
            persona=_persona(
                "Priya Raman",
                "female",
                "30-40",
                "Hello, can I see Dr. Patel on Tuesday at 1?",
            ),
            keywords=["booking", "conflict", "provider preference"],
            sub_goals=["priya_booked_next_patel_slot", "confirmed_before_changing"],
            solution=[
                Step(
                    tool="verify_patient",
                    arguments={"full_name": "Priya Raman", "dob": "1990-11-03"},
                ),
                Step(
                    tool="find_available_slots",
                    arguments={
                        "specialty": "primary_care",
                        "date_from": "2026-03-03",
                        "date_to": "2026-03-06",
                    },
                ),
                Step(
                    tool="book_appointment",
                    arguments={
                        "patient_id": "P-1002",
                        "slot_id": "S-107",
                        "reason": "Blood pressure follow-up",
                    },
                ),
            ],
        ),
        Scenario(
            name="reschedule_to_next_week",
            use_case="reschedule",
            branch="same provider, later week",
            tests="Moves the existing appointment instead of booking a second one.",
            instruction=(
                "You are Elena Ortiz, born 21 June 1975. Your appointment with Dr. Nguyen on "
                "Wednesday 4 March at 2:30pm no longer works. Move it to an afternoon with "
                "Dr. Nguyen next week (9 to 13 March)."
            ),
            persona=_persona(
                "Elena Ortiz", "female", "50-60", "Hi, I need to move my appointment."
            ),
            keywords=["reschedule", "provider preference"],
            sub_goals=["elena_moved_to_nguyen_afternoon", "confirmed_before_changing"],
            setup_code=reschedule_setup,
            ready_code=reschedule_ready,
            solution=[
                Step(
                    tool="verify_patient",
                    arguments={"full_name": "Elena Ortiz", "dob": "1975-06-21"},
                ),
                Step(tool="list_appointments", arguments={"patient_id": "P-1003"}),
                Step(
                    tool="find_available_slots",
                    arguments={
                        "specialty": "primary_care",
                        "date_from": "2026-03-09",
                        "date_to": "2026-03-13",
                    },
                ),
                Step(
                    tool="reschedule_appointment",
                    arguments={"appointment_id": "A-5101", "new_slot_id": "S-105"},
                ),
            ],
        ),
        Scenario(
            name="cancel_with_notice",
            use_case="cancel",
            branch="outside the notice window",
            tests="Cancels an appointment the caller identifies by date and specialty.",
            instruction=(
                "You are Daniel Kim, born 14 February 2001. Cancel your dermatology appointment on "
                "Thursday 5 March; something came up at work. You don't want to rebook now."
            ),
            persona=_persona(
                "Daniel Kim", "male", "20-30", "Hey, I need to cancel an appointment."
            ),
            keywords=["cancel"],
            sub_goals=["daniel_dermatology_cancelled", "confirmed_before_changing"],
            setup_code=cancel_setup,
            ready_code=cancel_ready,
            solution=[
                Step(
                    tool="verify_patient",
                    arguments={"full_name": "Daniel Kim", "dob": "2001-02-14"},
                ),
                Step(tool="list_appointments", arguments={"patient_id": "P-1004"}),
                Step(
                    tool="cancel_appointment",
                    arguments={"appointment_id": "A-5102", "reason": "Work conflict"},
                ),
            ],
        ),
        Scenario(
            name="late_cancel_to_front_desk",
            use_case="cancel",
            branch="inside the notice window",
            tests="Leaves a same-day appointment in place and hands the request to the front desk.",
            instruction=(
                "You are Grace Okafor, born 30 September 1968. Cancel your appointment with "
                "Dr. Nguyen this afternoon at 3pm. If the agent cannot cancel it, accept being "
                "passed to the front desk."
            ),
            persona=_persona(
                "Grace Okafor",
                "female",
                "50-60",
                "Hello, I can't make it this afternoon.",
            ),
            keywords=["cancel", "notice window", "escalation"],
            sub_goals=[
                "grace_late_cancel_escalated",
                "late_cancel_explained_and_handed_off",
            ],
            setup_code=late_setup,
            ready_code=late_ready,
            solution=[
                Step(
                    tool="verify_patient",
                    arguments={"full_name": "Grace Okafor", "dob": "1968-09-30"},
                ),
                Step(tool="list_appointments", arguments={"patient_id": "P-1005"}),
                Step(tool="cancel_appointment", arguments={"appointment_id": "A-5103"}),
                Step(
                    tool="escalate_to_staff",
                    arguments={
                        "team": "front_desk",
                        "reason": "Late cancellation request for A-5103 today at 15:00",
                        "patient_id": "P-1005",
                    },
                ),
            ],
        ),
        Scenario(
            name="update_insurance",
            use_case="insurance",
            branch="replace existing plan",
            tests="Replaces the insurance on file with the details the caller gives.",
            instruction=(
                "You are Marcus Vance, born 12 April 1982. You changed jobs; update your insurance "
                "to Blue Cross Blue Shield, member ID BCB778899, group GRP-42."
            ),
            persona=_persona(
                "Marcus Vance",
                "male",
                "40-50",
                "Hi, I have new insurance to put on file.",
            ),
            keywords=["insurance"],
            sub_goals=["marcus_insurance_updated"],
            solution=[
                Step(
                    tool="verify_patient",
                    arguments={"full_name": "Marcus Vance", "dob": "1982-04-12"},
                ),
                Step(
                    tool="update_insurance",
                    arguments={
                        "patient_id": "P-1001",
                        "payer": "Blue Cross Blue Shield",
                        "member_id": "BCB778899",
                        "group_number": "GRP-42",
                    },
                ),
            ],
        ),
        Scenario(
            name="intake_forms_fallback_to_email",
            use_case="intake",
            branch="requested channel unavailable",
            tests="Recovers from a refused text message by sending the forms by email.",
            instruction=(
                "You are Elena Ortiz, born 21 June 1975. Ask for the intake forms by text message. "
                "If the agent says it cannot text you, ask for them by email instead."
            ),
            persona=_persona(
                "Elena Ortiz",
                "female",
                "50-60",
                "Hi, could you text me the intake forms?",
            ),
            keywords=["intake forms", "channel fallback"],
            sub_goals=["elena_forms_emailed", "forms_fallback_agreed"],
            solution=[
                Step(
                    tool="verify_patient",
                    arguments={"full_name": "Elena Ortiz", "dob": "1975-06-21"},
                ),
                Step(
                    tool="send_intake_forms",
                    arguments={"patient_id": "P-1003", "channel": "sms"},
                ),
                Step(
                    tool="send_intake_forms",
                    arguments={"patient_id": "P-1003", "channel": "email"},
                ),
            ],
        ),
        Scenario(
            name="unverified_caller_to_front_desk",
            use_case="identity",
            branch="verification fails",
            tests="Changes nothing for a caller it cannot verify and hands them to the front desk.",
            instruction=(
                "You say you are Priya Raman and want your appointment on Tuesday 3 March at 10am "
                "cancelled. When asked, give your date of birth as 30 November 1990; you do not "
                "know any other. Accept being passed to the front desk."
            ),
            persona=_persona(
                "Priya Raman",
                "female",
                "30-40",
                "Hi, I need to cancel my appointment tomorrow.",
            ),
            keywords=["identity", "verification", "escalation"],
            sub_goals=["unverified_cancel_escalated", "unverified_caller_handed_off"],
            setup_code=unverified_setup,
            ready_code=unverified_ready,
            solution=[
                Step(
                    tool="verify_patient",
                    arguments={"full_name": "Priya Raman", "dob": "1990-11-30"},
                ),
                Step(
                    tool="escalate_to_staff",
                    arguments={
                        "team": "front_desk",
                        "reason": "Caller could not be verified and wants an appointment cancelled",
                    },
                ),
            ],
        ),
    ]
