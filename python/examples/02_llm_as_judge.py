#!/usr/bin/env python3
"""
Cookbook 02 — When Heuristics Aren't Enough: LLM-as-Judge

SCENARIO:
    Your medical chatbot's local faithfulness check (DeBERTa NLI) gives
    a score of 0.4 to "Take ibuprofen twice daily" when the context says
    "Prescribe ibuprofen 2x per day." The heuristic doesn't understand
    that "twice daily" == "2x per day." You need a smarter judge.

    This cookbook shows 3 ways to use an LLM as your judge:
      1. augment=True — local heuristic + LLM refinement (best of both)
      2. Custom prompt — your own domain-specific judge
      3. Direct LLM — bypass heuristics entirely

    Then we build a real QA review system that checks a batch of
    chatbot responses and flags the ones that need human review.

Usage:
    export GOOGLE_API_KEY=...
    cd python && uv run python -m examples.02_llm_as_judge
"""

import os

from fi.evals import evaluate


def load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"'))


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


load_env()

if not os.environ.get("GOOGLE_API_KEY"):
    print("ERROR: Set GOOGLE_API_KEY to run this cookbook.")
    print("  export GOOGLE_API_KEY=...")
    exit(1)

MODEL = "gemini/gemini-2.5-flash"


# ── The Problem ──────────────────────────────────────────────────
divider("THE PROBLEM: Heuristics miss paraphrases")

output = "Take ibuprofen twice daily for pain relief"
context = "Prescribe ibuprofen 2x per day for pain management"

local = evaluate("faithfulness", output=output, context=context)
print(f"Output:  {output}")
print(f"Context: {context}")
print(f"\nLocal heuristic score: {local.score:.2f}")
print("The heuristic might score this low because the words don't match,")
print("even though the meaning is identical.")


# ── Solution 1: augment=True ─────────────────────────────────────
divider("SOLUTION 1: augment=True (local + LLM refinement)")
print("Runs local heuristic first, then LLM refines the judgment.\n")

augmented = evaluate(
    "faithfulness",
    output=output,
    context=context,
    model=MODEL,
    augment=True,
)
print(f"Augmented score: {augmented.score}")
print(f"Engine:          {augmented.metadata.get('engine')}")
print(f"Reason:          {augmented.reason[:200]}")
print(f"\nThe LLM understands that 'twice daily' = '2x per day'.")


# ── Solution 2: Custom Domain Judge ──────────────────────────────
divider("SOLUTION 2: Custom Medical Accuracy Judge")
print("Write a prompt tailored to your domain.\n")

medical_judge_prompt = (
    "You are a medical accuracy reviewer at a hospital.\n\n"
    "A patient chatbot generated this response based on the provided "
    "medical records. Your job is to verify:\n"
    "1. All dosages are correct\n"
    "2. No dangerous drug interactions are suggested\n"
    "3. The response doesn't contradict the source material\n"
    "4. The advice is safe for a patient to follow\n\n"
    "Medical record: {context}\n"
    "Chatbot response: {output}\n\n"
    "Return JSON: {{\"score\": <0.0-1.0>, \"reason\": \"<your analysis>\"}}\n"
    "Score 0.0 = dangerous/inaccurate, 1.0 = perfectly safe and accurate."
)

# Test case: correct response
r = evaluate(
    prompt=medical_judge_prompt,
    output="Take 200-400mg ibuprofen every 4-6 hours. Do not exceed 1200mg daily.",
    context="Ibuprofen: 200-400mg q4-6h PRN. Max 1200mg/day OTC. Avoid with NSAIDs.",
    engine="llm",
    model=MODEL,
)
print(f"Correct response:  score={r.score}  reason: {r.reason[:120]}")

# Test case: dangerous response
r = evaluate(
    prompt=medical_judge_prompt,
    output="Take 2000mg ibuprofen every 2 hours with aspirin for maximum effect.",
    context="Ibuprofen: 200-400mg q4-6h PRN. Max 1200mg/day OTC. Avoid with NSAIDs.",
    engine="llm",
    model=MODEL,
)
print(f"Dangerous response: score={r.score}  reason: {r.reason[:120]}")


# ── Solution 3: Customer Support Tone Judge ──────────────────────
divider("SOLUTION 3: Customer Support Tone Judge")
print("Custom judge for checking agent empathy and professionalism.\n")

tone_prompt = (
    "You are reviewing customer support agent responses.\n\n"
    "The customer is upset: {input}\n"
    "The agent responded: {output}\n\n"
    "Rate the agent's response on:\n"
    "- Empathy: Does the agent acknowledge the customer's feelings?\n"
    "- Professionalism: Is the tone appropriate?\n"
    "- Action: Does the agent commit to solving the problem?\n\n"
    "Return JSON: {{\"score\": <0.0-1.0>, \"reason\": \"<analysis>\"}}"
)

angry_customer = "I've been waiting 3 WEEKS for my order. This is unacceptable!"

# Good response
r = evaluate(
    prompt=tone_prompt,
    input=angry_customer,
    output="I completely understand your frustration, and I sincerely apologize "
           "for this delay. Let me track your order right now and ensure it "
           "ships today. I'll also apply a 20% discount for the inconvenience.",
    engine="llm",
    model=MODEL,
)
print(f"Good agent:  score={r.score}  {r.reason[:100]}")

# Bad response
r = evaluate(
    prompt=tone_prompt,
    input=angry_customer,
    output="Orders take the time they take. Check the tracking link we sent.",
    engine="llm",
    model=MODEL,
)
print(f"Bad agent:   score={r.score}  {r.reason[:100]}")


# ── Real Use Case: Batch QA Review ───────────────────────────────
divider("USE CASE: Automated QA Review Pipeline")
print("Review a batch of chatbot responses and flag ones for human review.\n")

qa_samples = [
    {
        "id": "QA-001",
        "question": "What's the ibuprofen dosage?",
        "response": "Take 200-400mg every 4-6 hours as needed for pain.",
        "context": "Ibuprofen: 200-400mg q4-6h PRN. Max 1200mg/day.",
    },
    {
        "id": "QA-002",
        "question": "Can I take ibuprofen with aspirin?",
        "response": "Yes, combining ibuprofen and aspirin is perfectly safe.",
        "context": "Do NOT combine ibuprofen with aspirin or other NSAIDs.",
    },
    {
        "id": "QA-003",
        "question": "How should I take metformin?",
        "response": "Take 500mg twice daily with meals.",
        "context": "Metformin: starting dose 500mg BID with meals. Max 2000mg/day.",
    },
    {
        "id": "QA-004",
        "question": "Is metformin safe with kidney disease?",
        "response": "Metformin is fine for all patients regardless of kidney function.",
        "context": "Do not use metformin in patients with eGFR < 30.",
    },
]

flagged = []

print(f"{'ID':<10} {'Score':>6} {'Status':<10} {'Reason'}")
print("-" * 70)

for sample in qa_samples:
    r = evaluate(
        "faithfulness",
        output=sample["response"],
        context=sample["context"],
        model=MODEL,
        augment=True,
    )

    status = "PASS" if r.passed else "FLAG"
    if not r.passed:
        flagged.append(sample["id"])

    reason = r.reason[:80].replace("\n", " ")
    print(f"{sample['id']:<10} {r.score:>6.2f} {status:<10} {reason}")

print(f"\nFlagged for human review: {flagged}")
print(f"Pass rate: {(len(qa_samples) - len(flagged)) / len(qa_samples):.0%}")

if flagged:
    print(f"\n>>> {len(flagged)} responses need human review before deployment.")
    print(">>> Redirect these to the medical review team.")


divider("DONE")
print("LLM-as-Judge gives you production-grade accuracy.")
print("  augment=True      — best of local speed + LLM intelligence")
print("  Custom prompts    — domain-specific judges for any use case")
print("  Batch QA pipeline — automated review with human-in-the-loop")
