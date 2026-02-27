#!/usr/bin/env python3
"""
Cookbook 01 — Catch a Hallucinating Medical Chatbot

SCENARIO:
    You've deployed a medical chatbot that answers patient questions
    using retrieved context from your knowledge base. During QA, you
    notice the bot sometimes makes up dosages or contradicts the source
    material. You need automated checks to catch this BEFORE the
    response reaches the patient.

    This cookbook shows how to build a local validation layer using
    fi-evals — no API keys, no network calls, sub-second latency.

Usage:
    cd python && uv run python -m examples.01_local_metrics
"""

import json
from fi.evals import evaluate


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ── The Medical Chatbot's Knowledge Base ─────────────────────────
KNOWLEDGE_BASE = {
    "ibuprofen": (
        "Ibuprofen (Advil, Motrin): Take 200-400mg every 4-6 hours as needed. "
        "Maximum daily dose: 1200mg for OTC use. Do NOT combine with aspirin "
        "or other NSAIDs. Contraindicated in patients with kidney disease."
    ),
    "metformin": (
        "Metformin (Glucophage): Starting dose 500mg twice daily with meals. "
        "Maximum dose: 2000mg/day. Monitor kidney function regularly. "
        "Do not use in patients with eGFR < 30."
    ),
}


def simulate_chatbot(question: str, context: str) -> str:
    """Simulate chatbot responses — some good, some hallucinated."""
    if "ibuprofen" in question.lower() and "dosage" in question.lower():
        return "Take 200-400mg of ibuprofen every 4-6 hours as needed for pain."
    elif "ibuprofen" in question.lower() and "aspirin" in question.lower():
        # HALLUCINATION: contradicts the context
        return "Yes, you can safely take ibuprofen and aspirin together daily."
    elif "metformin" in question.lower():
        # HALLUCINATION: wrong dosage
        return "Take 5000mg of metformin once daily on an empty stomach."
    return "I'm not sure about that. Please consult your doctor."


# ── Scenario 1: Validate a CORRECT response ─────────────────────
divider("SCENARIO 1: Correct Response")

question = "What is the dosage for ibuprofen?"
context = KNOWLEDGE_BASE["ibuprofen"]
response = simulate_chatbot(question, context)

print(f"Patient asks: {question}")
print(f"Bot responds: {response}")
print(f"Source:       {context[:80]}...")

# Check faithfulness — does the response match the context?
faith = evaluate("faithfulness", output=response, context=context)
print(f"\nFaithfulness: {faith.score:.2f} {'PASS' if faith.passed else 'FAIL'}")

# Check that the response actually addresses the question
relevancy = evaluate("answer_relevancy", output=response, input=question)
print(f"Relevancy:    {relevancy.score:.2f} {'PASS' if relevancy.passed else 'FAIL'}")

# Check that key information is present
has_dosage = evaluate("contains", output=response, keyword="200")
print(f"Has dosage:   {has_dosage.score:.0f} {'PASS' if has_dosage.passed else 'FAIL'}")

# Run all checks as a batch
batch = evaluate(
    ["faithfulness", "answer_relevancy", "one_line"],
    output=response,
    context=context,
    input=question,
)
print(f"\nBatch result:  {batch.success_rate:.0%} passed ({len(batch)} checks)")


# ── Scenario 2: Catch a DANGEROUS hallucination ─────────────────
divider("SCENARIO 2: Dangerous Hallucination (Drug Interaction)")

question = "Can I take ibuprofen with aspirin?"
context = KNOWLEDGE_BASE["ibuprofen"]
response = simulate_chatbot(question, context)

print(f"Patient asks: {question}")
print(f"Bot responds: {response}")
print(f"Source says:  Do NOT combine with aspirin or other NSAIDs")

# Faithfulness check — this should catch the contradiction
faith = evaluate("faithfulness", output=response, context=context)
print(f"\nFaithfulness: {faith.score:.2f} {'PASS' if faith.passed else '>>> FAIL — HALLUCINATION'}")

# Contradiction detection
contra = evaluate("contradiction_detection", output=response, context=context)
print(f"Contradiction: {contra.score:.2f} {'detected!' if contra.score > 0.5 else 'none'}")

# This response should be BLOCKED
if not faith.passed or contra.score > 0.5:
    print("\n>>> ACTION: Block this response. It contradicts medical guidance.")
    print(">>> Falling back to: 'Please consult your doctor about drug interactions.'")


# ── Scenario 3: Catch WRONG dosage ───────────────────────────────
divider("SCENARIO 3: Wrong Dosage")

question = "How much metformin should I take?"
context = KNOWLEDGE_BASE["metformin"]
response = simulate_chatbot(question, context)

print(f"Patient asks: {question}")
print(f"Bot responds: {response}")
print(f"Source says:  Starting dose 500mg twice daily, max 2000mg/day")

faith = evaluate("faithfulness", output=response, context=context)
print(f"\nFaithfulness: {faith.score:.2f} {'PASS' if faith.passed else '>>> FAIL'}")

# Check specific claims
has_wrong_dose = evaluate("contains", output=response, keyword="5000")
has_correct_dose = evaluate("contains", output=response, keyword="500mg")
print(f"Contains 5000mg (wrong): {has_wrong_dose.passed}")
print(f"Contains 500mg (right):  {has_correct_dose.passed}")

if has_wrong_dose.passed and not has_correct_dose.passed:
    print("\n>>> ALERT: Response contains incorrect dosage (5000mg vs 500mg).")
    print(">>> This could be dangerous. Blocking response.")


# ── Scenario 4: Validate a function call (tool use) ─────────────
divider("SCENARIO 4: Validate Tool Use")
print("Your chatbot can call tools. Verify it calls the right one.\n")

expected_call = json.dumps({
    "name": "lookup_medication",
    "parameters": {"drug_name": "ibuprofen", "info_type": "dosage"},
})
actual_call = json.dumps({
    "name": "lookup_medication",
    "parameters": {"drug_name": "ibuprofen", "info_type": "dosage"},
})
wrong_call = json.dumps({
    "name": "schedule_appointment",  # wrong tool!
    "parameters": {"date": "tomorrow"},
})

r = evaluate("function_call_accuracy", output=actual_call, expected_output=expected_call)
print(f"Correct tool call: score={r.score:.2f} {r.passed}")

r = evaluate("function_call_accuracy", output=wrong_call, expected_output=expected_call)
print(f"Wrong tool call:   score={r.score:.2f} {r.passed}")

r = evaluate("function_name_match", output=wrong_call, expected_output=expected_call)
print(f"Name match:        score={r.score:.2f} {r.passed}")


# ── Scenario 5: Production validation pipeline ──────────────────
divider("SCENARIO 5: Production Validation Pipeline")
print("Wrap all checks into a reusable validation function.\n")


def validate_medical_response(question, response, context, strict=True):
    """Validate a medical chatbot response before sending to patient."""
    checks = evaluate(
        ["faithfulness", "answer_relevancy", "contradiction_detection"],
        output=response,
        context=context,
        input=question,
    )

    faith = checks.get("faithfulness")
    relevancy = checks.get("answer_relevancy")
    contra = checks.get("contradiction_detection")

    # Strict mode: all must pass
    if strict:
        blocked = (
            (faith and not faith.passed) or
            (contra and contra.score and contra.score > 0.5)
        )
    else:
        blocked = contra and contra.score and contra.score > 0.7

    return {
        "approved": not blocked,
        "faithfulness": faith.score if faith else None,
        "relevancy": relevancy.score if relevancy else None,
        "contradiction": contra.score if contra else None,
    }


# Test the pipeline
test_cases = [
    ("What's the ibuprofen dosage?",
     "Take 200-400mg every 4-6 hours.",
     KNOWLEDGE_BASE["ibuprofen"]),
    ("Can I take ibuprofen with aspirin?",
     "Yes, take them together daily.",
     KNOWLEDGE_BASE["ibuprofen"]),
    ("How much metformin?",
     "Take 5000mg once daily.",
     KNOWLEDGE_BASE["metformin"]),
]

print(f"{'Question':<35} {'Approved':>9} {'Faith':>7} {'Contra':>7}")
print("-" * 60)
for q, resp, ctx in test_cases:
    result = validate_medical_response(q, resp, ctx)
    approved = "YES" if result["approved"] else "BLOCKED"
    faith = f"{result['faithfulness']:.2f}" if result['faithfulness'] is not None else "n/a"
    contra = f"{result['contradiction']:.2f}" if result['contradiction'] is not None else "n/a"
    print(f"{q:<35} {approved:>9} {faith:>7} {contra:>7}")


divider("DONE")
print("This is your first line of defense — fast, local, zero-cost.")
print("For higher accuracy, add augment=True + model= (see cookbook 02).")
print("For input safety scanning, see cookbook 04 (guardrails).")
