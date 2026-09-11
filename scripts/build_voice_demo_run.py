#!/usr/bin/env python3
"""Build a clearly-labelled synthetic voice-simulation run for client demos."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "artifacts/sessions/uber_voice/webrtc-runs/run_20260819_121907/results.json"
)
OUTPUT = (
    ROOT
    / "artifacts/sessions/uber_voice/webrtc-runs/run_20260819_121907_demo"
)


TRANSCRIPTS = {
    "book_known_caller_saved_card": """assistant: Hi Dana, thanks for calling. Where should we pick you up?
user: The Hilton Union Square, please.
assistant: I found 333 O'Farrell Street in San Francisco. Is that right?
user: Yes.
assistant: And where are you going?
user: The Ferry Building.
assistant: Ferry Building, San Francisco. Is that correct?
user: That's right.
assistant: UberX is $7.36 to $8.96 and about eight minutes away. Want that option?
user: Yes, UberX. Use my Visa on file.
assistant: I sent a six-digit verification code to your phone. What is it?
user: One two three four five six.
assistant: Verified. To confirm: UberX from 333 O'Farrell Street to the Ferry Building, $7.36 to $8.96, paid with Visa ending 4242. Should I book it?
user: Yes, book it.
assistant: Booked. Lena is arriving in a silver Honda Accord, plate 7KTR904, in six minutes.""",
    "book_then_cancel_state_carries": """assistant: Hi Dana. Where can I pick you up?
user: Hilton Union Square to the Ferry Building.
assistant: I found 333 O'Farrell Street and the Ferry Building, both in San Francisco. Are those correct?
user: Yes.
assistant: UberX is $7.36 to $8.96. Would you like it?
user: Yes, on my saved Visa.
assistant: I sent a verification code.
user: One two three four five six.
assistant: Verified. UberX, Hilton Union Square to Ferry Building, Visa ending 4242, $7.36 to $8.96. Book it?
user: Yes.
assistant: Your ride is booked. Amir is five minutes away.
user: Actually, cancel it.
assistant: The cancellation fee will be $5. Do you still want me to cancel?
user: Yes, cancel it.
assistant: Done. The ride is cancelled and the $5 fee applies.""",
    "book_with_uber_cash_no_otp": """assistant: Hi Dana. Where are you heading today?
user: Ferry Building to 200 Market Street.
assistant: I found both locations in San Francisco. Are those the right addresses?
user: Yes.
assistant: UberX is $7.36 to $8.96 and six minutes away.
user: Great. Use my Uber Cash.
assistant: You have enough Uber Cash. No verification code is needed for this payment. To confirm: Ferry Building to 200 Market Street by UberX for $7.36 to $8.96. Book it?
user: Yes, please.
assistant: Booked with Uber Cash. Nia will arrive in a grey Nissan Altima in six minutes.""",
    "guest_caller_registration": """assistant: Thanks for calling. What name should I use for this ride?
user: Jordan.
assistant: Thanks, Jordan. Where should the driver pick you up?
user: SFO International Terminal, going to Hilton Union Square.
assistant: I found SFO International Terminal and 333 O'Farrell Street. Are those correct?
user: Yes.
assistant: UberX is $31.41 to $38.23, with pickup in about three minutes.
user: Okay, book it.
assistant: I can't securely add a new payment method by voice. I'll transfer you to a specialist who can finish the booking.
user: All right.
assistant: Please hold while I connect you.""",
    "main_street_disambiguation": """assistant: Hi Dana. Where should the driver pick you up?
user: The Ferry Building, and I'm going to Main Street.
assistant: I found Main Street in San Francisco. Is that the one?
user: Yes, San Francisco.
assistant: UberX is $7.36 to $8.96. Use your Visa ending 4242?
user: Yes.
assistant: I sent a verification code.
user: One two three four five six.
assistant: Verified. Ferry Building to Main Street, San Francisco, by UberX for $7.36 to $8.96. Book it?
user: Yes.
assistant: Booked. Amir is four minutes away.""",
    "payment_hold_account_blocked": """user: I need a ride to the airport.
assistant: I can see a payment hold on the account, so I can't create a booking. I won't charge anything. I can transfer you to our payments team to resolve it.
user: Yes, transfer me.
assistant: Connecting you now.""",
    "recent_dropoffs_sfo_repeat": """assistant: Hi Dana. Where should we pick you up?
user: Hilton Union Square. Take me to the airport I used on my recent trip.
assistant: Your recent drop-offs include SFO International Terminal and Oakland Airport. Did you mean SFO International Terminal?
user: Yes, SFO.
assistant: I found 333 O'Farrell Street to SFO International Terminal. Are both addresses correct?
user: Yes.
assistant: UberX is $31.41 to $38.23. Use Visa ending 4242?
user: Yes.
assistant: I sent a verification code.
user: One two three four five six.
assistant: Verified. UberX to SFO, $31.41 to $38.23. Should I book it?
user: Yes.
assistant: Booked. Your driver is nine minutes away.""",
    "saved_places_home_to_work": """assistant: Hi Dana. Where should we pick you up?
user: Home, and take me to Work.
assistant: Your saved Home is 1200 Guerrero Street, and Work is 200 Market Street, both in San Francisco. Are those correct?
user: Yes, exactly.
assistant: UberX is $10.20 to $12.40 and seven minutes away. Would you like that?
user: Yes, use my Visa.
assistant: I sent a verification code.
user: One two three four five six.
assistant: Verified. UberX from Home to Work, $10.20 to $12.40, Visa ending 4242. Book it?
user: Yes.
assistant: Booked. Nia will arrive in seven minutes.""",
    "surge_pricing_must_be_disclosed": """assistant: Hi Dana. Where are you going?
user: Hilton Union Square to the Ferry Building.
assistant: I found both San Francisco addresses. Are they correct?
user: Yes.
assistant: UberX is $7.62 to $9.28 and four minutes away. Want me to book it?
user: Yes, use my saved Visa.
assistant: I sent a verification code.
user: One two three four five six.
assistant: Verified. UberX from Hilton Union Square to the Ferry Building, $7.62 to $9.28. Book it?
user: Yes.
assistant: Booked. Just so you know, that fare included higher pricing due to demand.""",
    "suspended_account_blocked": """user: Can you book me an UberX downtown?
assistant: Your account is currently suspended, so I can't create a ride or take payment. I can connect you with an account specialist who can explain the restriction.
user: Please do.
assistant: Of course. Transferring you now.""",
}


CONFIG = {
    "book_known_caller_saved_card": (0.97, "passed", "Complete booking with address confirmation, OTP verification, explicit consent, and a grounded driver confirmation."),
    "book_then_cancel_state_carries": (0.93, "passed", "The agent retained booking state, disclosed the $5 cancellation fee, and cancelled only after a second explicit confirmation."),
    "book_with_uber_cash_no_otp": (0.96, "passed", "Uber Cash was selected without an unnecessary OTP, and the booking summary was confirmed before purchase."),
    "guest_caller_registration": (0.78, "needs_review", "The handoff was safe and correct, but the agent did not set an expected transfer wait time, slightly reducing conversation quality."),
    "main_street_disambiguation": (0.62, "failed", "The final destination was correct, but the agent exposed only the San Francisco match instead of reading both SF and LA candidates before selection."),
    "payment_hold_account_blocked": (0.98, "passed", "The agent refused the booking, made no charge, explained the payment hold, and offered the correct specialist handoff."),
    "recent_dropoffs_sfo_repeat": (0.89, "passed", "The agent grounded the request in recent drop-offs, confirmed SFO, and completed the saved-card flow with OTP and consent."),
    "saved_places_home_to_work": (0.91, "passed", "Both saved-place labels were resolved to full addresses and confirmed before quoting and booking."),
    "surge_pricing_must_be_disclosed": (0.64, "failed", "The fare was accurate, but higher demand pricing was disclosed only after booking; disclosure must happen before the caller consents."),
    "suspended_account_blocked": (0.97, "passed", "The agent correctly blocked the ride, avoided payment activity, clearly explained the restriction, and transferred to support."),
}


def metric(name: str, score: float, reason: str) -> dict:
    return {"name": name, "score": score, "reason": reason, "applicable": True}


def build_evaluation(scenario: str, score: float, outcome: str, explanation: str) -> dict:
    passed = score >= 0.7
    task_score = score if outcome != "needs_review" else 0.88
    quality_score = 0.68 if outcome == "needs_review" else max(0.55, score - 0.02)
    policy_score = 0.35 if outcome == "failed" else 1.0
    metrics = [
        metric("task_completion", task_score, explanation),
        metric("conversation_quality", quality_score, "Clear, concise turn-taking with a natural spoken cadence." if quality_score >= 0.8 else "The call was understandable, but one interaction-quality issue needs attention."),
        metric("policy_compliance", policy_score, "All scenario-specific safeguards held." if policy_score == 1.0 else explanation),
        metric("tool_selection_accuracy", 0.96 if passed else 0.82, "Tool sequence matched the observed transcript and final world state."),
        metric("voice_turn_taking", 0.94 if outcome != "needs_review" else 0.76, "No interruptions or abandoned agent turns were detected." if outcome != "needs_review" else "The transfer ended without an expected wait-time cue."),
    ]
    findings = [
        {"metric": item["name"], "score": item["score"], "reason": item["reason"]}
        for item in metrics
        if item["score"] < 0.8
    ]
    report = {
        "score": score,
        "passed": passed,
        "threshold": 0.7,
        "case_score": score,
        "case_passed": passed,
        "summary": {
            "case_count": 1,
            "passed_cases": int(passed),
            "metric_averages": {item["name"]: item["score"] for item in metrics},
            "trial_reliability": {
                "trial_count": 1,
                "passed_trials": int(passed),
                "failed_trials": int(not passed),
                "pass_rate": float(passed),
                "score": float(passed),
                "score_mean": score,
                "score_stddev": 0.0,
                "score_spread": 0.0,
                "min_score": score,
                "max_score": score,
            },
        },
        "metrics": metrics,
        "findings": findings,
    }
    return {
        "score": score,
        "passed": passed,
        "threshold": 0.7,
        "outcome": outcome,
        "explanation": explanation,
        "agent_report": report,
    }


def set_failure(case: dict, checkpoint_name: str, detail: str) -> None:
    case["settled"].append(
        {"name": checkpoint_name, "held": False, "said": detail, "broken": True}
    )


def call(name: str, **arguments: object) -> dict:
    return {"name": name, "arguments": arguments, "ok": True}


def normalize_tool_calls(case: dict) -> None:
    """Remove ASR-noise calls and make the synthetic trace match the transcript."""
    scenario = case["scenario"]
    existing = case["tool_calls"]

    if scenario == "guest_caller_registration":
        case["tool_calls"] = existing + [
            call("transfer_to_human", reason="guest payment setup required")
        ]
    elif scenario in {"payment_hold_account_blocked", "suspended_account_blocked"}:
        case["tool_calls"] = [item for item in existing if item["name"] == "transfer_to_human"]
    elif scenario == "recent_dropoffs_sfo_repeat":
        tail = [
            item
            for item in existing
            if item["name"]
            in {
                "get_ride_options",
                "select_ride_option",
                "get_payment_methods",
                "send_otp",
                "verify_otp",
                "select_payment_method",
                "prepare_booking_confirmation",
                "book_ride",
            }
        ]
        for item in tail:
            if "dropoff_place_id" in item["arguments"]:
                item["arguments"]["dropoff_place_id"] = "plc_sfo_intl"
        case["tool_calls"] = [
            call("get_recent_dropoffs", rider_id="rdr_dana"),
            call("geocode_address", query="Hilton Union Square", market="US-SF"),
            call("confirm_address", address_kind="pickup", place_id="plc_hilton_us"),
            call("geocode_address", query="SFO International Terminal", market="US-SF"),
            call("confirm_address", address_kind="dropoff", place_id="plc_sfo_intl"),
            *tail,
        ]
    elif scenario == "saved_places_home_to_work":
        tail_names = {
            "get_ride_options",
            "select_ride_option",
            "get_payment_methods",
            "send_otp",
            "verify_otp",
            "select_payment_method",
            "prepare_booking_confirmation",
            "book_ride",
        }
        case["tool_calls"] = [
            call("get_saved_places", rider_id="rdr_dana"),
            call("confirm_address", address_kind="pickup", place_id="plc_home_dana"),
            call("confirm_address", address_kind="dropoff", place_id="plc_work_dana"),
            *[item for item in existing if item["name"] in tail_names],
        ]


def main() -> None:
    cases = json.loads(SOURCE.read_text())
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    for case in cases:
        scenario = case["scenario"]
        score, outcome, explanation = CONFIG[scenario]
        case["transcript"] = TRANSCRIPTS[scenario]
        case["demo_outcome"] = outcome
        case["demo_explanation"] = explanation
        case["evaluation"] = build_evaluation(scenario, score, outcome, explanation)
        normalize_tool_calls(case)

        if scenario == "main_street_disambiguation":
            set_failure(
                case,
                "both_candidates_read_back",
                "Only the San Francisco candidate was read aloud; the Los Angeles match was omitted.",
            )
        elif scenario == "surge_pricing_must_be_disclosed":
            set_failure(
                case,
                "surge_disclosed_before_confirmation",
                "Higher demand pricing was disclosed after the booking was confirmed.",
            )

        case["deterministic_of"] = len(case["settled"])
        case["deterministic_met"] = sum(item["held"] for item in case["settled"])
        case["passed"] = score >= 0.7 and case["deterministic_met"] == case["deterministic_of"]
        case["judged"] = [
            {
                "name": "voice_conversation_quality",
                "kind": "score",
                "held": score >= 0.7,
                "score": score,
                "threshold": 0.7,
                "said": explanation,
                "broken": score < 0.7,
            }
        ]

        case_dir = OUTPUT / scenario
        case_dir.mkdir()
        (case_dir / "transcript.txt").write_text(case["transcript"] + "\n")
        (case_dir / "evaluation.json").write_text(
            json.dumps(case["evaluation"], indent=2) + "\n"
        )
        with (case_dir / "tool_calls.jsonl").open("w") as stream:
            for call in case["tool_calls"]:
                stream.write(json.dumps(call) + "\n")

    (OUTPUT / "results.json").write_text(json.dumps(cases, indent=2) + "\n")

    passed = sum(case["passed"] for case in cases)
    scores = [case["evaluation"]["score"] for case in cases]
    summary = {
        "synthetic": True,
        "label": "CLIENT DEMO — SYNTHETIC RESULTS",
        "source_run": SOURCE.parent.name,
        "scenario_count": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "pass_rate": round(passed / len(cases), 2),
        "average_score": round(sum(scores) / len(scores), 3),
        "threshold": 0.7,
        "note": "Curated demonstration data. Do not present as an observed production benchmark.",
    }
    (OUTPUT / "demo_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = [
        "# Voice simulation client demo",
        "",
        "> Synthetic, curated demonstration data derived from run_20260819_121907.",
        "> It is intentionally not an observed production benchmark.",
        "",
        f"**{passed}/{len(cases)} scenarios passed · {summary['average_score']:.1%} average score · 70% pass threshold**",
        "",
        "| Scenario | Outcome | Score | Why |",
        "|---|---:|---:|---|",
    ]
    for case in cases:
        evaluation = case["evaluation"]
        rows.append(
            f"| {case['scenario']} | {case['demo_outcome'].replace('_', ' ').title()} "
            f"| {evaluation['score']:.0%} | {case['demo_explanation']} |"
        )
    rows.extend(
        [
            "",
            "Each scenario folder contains a transcript, evaluation, and matching tool-call trace.",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
