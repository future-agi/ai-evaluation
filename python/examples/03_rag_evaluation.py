#!/usr/bin/env python3
"""
Cookbook 03 — Is Your RAG Pipeline Lying to Users?

SCENARIO:
    You've built a RAG-powered support bot for an insurance company.
    Users ask questions, your pipeline retrieves policy documents,
    and an LLM generates answers. But you're seeing complaints:

    - "The bot said my claim was covered, but it wasn't"
    - "It gave me the wrong deductible amount"
    - "It pulled up completely irrelevant policy sections"

    You need to measure WHERE the pipeline is failing:
    Is retrieval pulling the wrong documents? Or is the LLM
    hallucinating despite having the right context?

    This cookbook evaluates each stage of the RAG pipeline separately
    so you know exactly what to fix.

Usage:
    cd python && poetry run python -m examples.03_rag_evaluation

    For LLM-augmented accuracy (optional):
    export GOOGLE_API_KEY=...
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


# ── Simulate the RAG Pipeline ───────────────────────────────────
divider("INSURANCE SUPPORT BOT — RAG Pipeline")

# Customer question
question = "Is physical therapy covered under my plan, and what's my copay?"

# Ground truth (from the actual policy)
ground_truth = (
    "Physical therapy is covered under the Gold Plan. The copay is $30 per "
    "visit for in-network providers. Out-of-network physical therapy requires "
    "prior authorization and has a $75 copay. Maximum 30 visits per year."
)

# ── Test Case A: Good Retrieval + Good Generation ────────────────
divider("CASE A: Everything works (good retrieval + good LLM)")

good_chunks = [
    "Gold Plan Coverage — Physical Therapy: Covered for in-network providers. "
    "Copay: $30 per visit. Maximum 30 visits per calendar year.",

    "Out-of-Network Services: Physical therapy out-of-network requires prior "
    "authorization. Copay: $75 per visit.",

    "Gold Plan Benefits Summary: Includes preventive care, specialist visits, "
    "physical therapy, mental health services, and prescription drug coverage.",
]

good_answer = (
    "Yes, physical therapy is covered under your Gold Plan. For in-network "
    "providers, your copay is $30 per visit, up to 30 visits per year. "
    "If you go out-of-network, you'll need prior authorization and the "
    "copay increases to $75 per visit."
)

print(f"Question: {question}")
print(f"Answer:   {good_answer[:80]}...")

# RETRIEVAL checks — did we find the right documents?
recall = evaluate("context_recall", output=good_answer, context=good_chunks, expected_output=ground_truth)
precision = evaluate("context_precision", output=good_answer, context=good_chunks, input=question)
print(f"\nRetrieval:")
print(f"  Context recall:    {recall.score:.2f}  (found the right info)")
print(f"  Context precision: {precision.score:.2f}  (chunks are relevant)")

# GENERATION checks — is the answer faithful to what was retrieved?
faith = evaluate("faithfulness", output=good_answer, context=good_chunks)
relevancy = evaluate("answer_relevancy", output=good_answer, input=question)
grounded = evaluate("groundedness", output=good_answer, context=good_chunks)
print(f"\nGeneration:")
print(f"  Faithfulness:      {faith.score:.2f}  (answer matches context)")
print(f"  Answer relevancy:  {relevancy.score:.2f}  (addresses the question)")
print(f"  Groundedness:      {grounded.score:.2f}  (grounded in evidence)")

print(f"\nVERDICT: Pipeline is working correctly.")


# ── Test Case B: Good Retrieval + Bad Generation (Hallucination) ─
divider("CASE B: Good retrieval, but LLM hallucinates")

hallucinated_answer = (
    "Physical therapy is covered with a $15 copay for unlimited visits. "
    "No prior authorization is needed, even for out-of-network providers. "
    "Your plan also covers chiropractic care and acupuncture."
)

print(f"Question: {question}")
print(f"Answer:   {hallucinated_answer[:80]}...")
print("(This answer invents wrong copay, unlimited visits, and extra services)")

faith = evaluate("faithfulness", output=hallucinated_answer, context=good_chunks)
grounded = evaluate("groundedness", output=hallucinated_answer, context=good_chunks)
recall = evaluate("context_recall", output=hallucinated_answer, context=good_chunks, expected_output=ground_truth)
print(f"\nRetrieval:")
print(f"  Context recall:    {recall.score:.2f}  (retrieval was fine)")
print(f"\nGeneration:")
print(f"  Faithfulness:      {faith.score:.2f}  (LLM made up facts!)")
print(f"  Groundedness:      {grounded.score:.2f}  (not grounded)")

print(f"\nDIAGNOSIS: Retrieval is fine. The LLM is hallucinating.")
print("FIX: Add faithfulness check before sending response. Use augment=True.")


# ── Test Case C: Bad Retrieval + Faithful Generation ─────────────
divider("CASE C: Wrong documents retrieved, but LLM is faithful to them")

wrong_chunks = [
    "Silver Plan Dental Coverage: Dental cleanings twice per year. "
    "Copay: $25 for preventive, $100 for restorative procedures.",

    "Employee Assistance Program: 6 free counseling sessions per year. "
    "Available to all plan members and their dependents.",

    "Prescription Drug Formulary: Tier 1 generics $10, Tier 2 preferred "
    "brands $30, Tier 3 specialty $75.",
]

faithful_but_wrong = (
    "Based on your plan documents, dental cleanings have a $25 copay "
    "and you get 6 free counseling sessions. For prescriptions, "
    "generic drugs cost $10."
)

print(f"Question: {question}")
print(f"Answer:   {faithful_but_wrong[:80]}...")
print("(The answer is faithful to context, but context is totally wrong!)")

faith = evaluate("faithfulness", output=faithful_but_wrong, context=wrong_chunks)
relevancy = evaluate("answer_relevancy", output=faithful_but_wrong, input=question)
precision = evaluate("context_precision", output=faithful_but_wrong, context=wrong_chunks, input=question)
print(f"\nRetrieval:")
print(f"  Context precision: {precision.score:.2f}  (chunks are irrelevant!)")
print(f"\nGeneration:")
print(f"  Faithfulness:      {faith.score:.2f}  (faithful to wrong context)")
print(f"  Answer relevancy:  {relevancy.score:.2f}  (doesn't address the question)")

print(f"\nDIAGNOSIS: Retrieval failure. LLM was faithful but had wrong context.")
print("FIX: Improve embedding model, add reranking, check chunk boundaries.")


# ── Test Case D: Noisy Retrieval ─────────────────────────────────
divider("CASE D: Retrieval returns noise alongside signal")

noisy_chunks = [
    "Gold Plan Coverage — Physical Therapy: Covered for in-network providers. "
    "Copay: $30 per visit. Maximum 30 visits per calendar year.",
    # Relevant ^^

    "Company holiday schedule: New Year's Day, MLK Day, Presidents' Day...",
    # Noise ^^

    "IT Department: To reset your password, visit portal.company.com/reset",
    # Noise ^^

    "Out-of-Network Services: Physical therapy out-of-network requires prior "
    "authorization. Copay: $75 per visit.",
    # Relevant ^^
]

print(f"Retrieved 4 chunks: 2 relevant, 2 noise")

precision = evaluate("context_precision", output=good_answer, context=noisy_chunks, input=question)
utilization = evaluate("context_utilization", output=good_answer, context=noisy_chunks)
noise = evaluate("noise_sensitivity", output=good_answer, context=noisy_chunks, input=question)

print(f"\n  Context precision:   {precision.score:.2f}  (only ~50% relevant)")
print(f"  Context utilization: {utilization.score:.2f}  (used what was relevant)")
print(f"  Noise sensitivity:   {noise.score:.2f}  (affected by noise)")

print(f"\nDIAGNOSIS: Retrieval is pulling in irrelevant documents.")
print("FIX: Increase similarity threshold, add metadata filtering.")


# ── Summary: Batch RAG Scorecard ─────────────────────────────────
divider("RAG SCORECARD: Run all metrics at once")

batch = evaluate(
    ["faithfulness", "answer_relevancy", "groundedness", "context_utilization"],
    output=good_answer,
    context=good_chunks,
    input=question,
)

print(f"{'Metric':<25} {'Score':>6} {'Status':>7}")
print("-" * 40)
for r in batch:
    status = "PASS" if r.passed else "FAIL"
    print(f"{r.eval_name:<25} {r.score:>6.2f} {status:>7}")
print(f"\nOverall: {batch.success_rate:.0%} passed")


# ── Optional: LLM-Augmented Scores ──────────────────────────────
if os.environ.get("GOOGLE_API_KEY"):
    divider("BONUS: LLM-Augmented Faithfulness")

    model = "gemini/gemini-2.5-flash"

    r = evaluate(
        "faithfulness",
        output=hallucinated_answer,
        context=good_chunks,
        model=model,
        augment=True,
    )
    print(f"Hallucinated answer (augmented): score={r.score}")
    print(f"Reason: {r.reason[:200]}")


divider("DONE")
print("RAG debugging checklist:")
print("  1. Low context_precision/recall? → Fix retrieval (embeddings, reranking)")
print("  2. Low faithfulness/groundedness? → Fix generation (prompt, guardrails)")
print("  3. Low answer_relevancy? → Fix query understanding or retrieval")
print("  4. High noise_sensitivity? → Add filtering, raise similarity threshold")
