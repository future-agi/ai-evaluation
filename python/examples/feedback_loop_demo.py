#!/usr/bin/env python3
"""
Feedback Loop — End-to-End Demo

Demonstrates the full feedback loop with a REAL LLM judge:
1. Run faithfulness metric locally → heuristic gives wrong score
2. Developer submits feedback corrections → stored in ChromaDB
3. Run the SAME metric again with feedback_store → ChromaDB retrieves
   similar past corrections as few-shot examples → injected into
   LLM judge prompt → Gemini produces a calibrated result
4. Compare: without feedback vs with feedback

Usage:
    export GOOGLE_API_KEY=...
    cd python && uv run python -m examples.feedback_loop_demo
"""

import json
import os
import shutil
import tempfile

from fi.evals import evaluate
from fi.evals.core.result import EvalResult
from fi.evals.feedback import (
    FeedbackCollector,
    ChromaFeedbackStore,
    InMemoryFeedbackStore,
    FeedbackRetriever,
    configure_feedback,
)


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_real_llm_judge():
    """The real deal: LLM judge with feedback-driven few-shot examples."""
    divider("REAL E2E: LLM Judge + Feedback Loop")

    model = "gemini/gemini-2.5-flash"
    print(f"Using model: {model}")

    # --- Step 1: Run faithfulness WITHOUT feedback ---
    divider("STEP 1: Run faithfulness WITHOUT feedback")

    test_output = "The patient should take ibuprofen twice daily for pain relief"
    test_context = "Prescribe ibuprofen 2x per day for pain management"

    print(f"Output:  {test_output}")
    print(f"Context: {test_context}")
    print("(These are semantically equivalent — score should be HIGH)")

    result_no_feedback = evaluate(
        "faithfulness",
        output=test_output,
        context=test_context,
        model=model,
        augment=True,
    )
    print(f"\nResult WITHOUT feedback:")
    print(f"  Score:  {result_no_feedback.score}")
    print(f"  Reason: {result_no_feedback.reason[:200]}")
    print(f"  Engine: {result_no_feedback.metadata.get('engine', 'unknown')}")

    # --- Step 2: Build a feedback store with corrections ---
    divider("STEP 2: Submit feedback corrections to ChromaDB")

    tmpdir = tempfile.mkdtemp(prefix="fi_feedback_e2e_")
    store = ChromaFeedbackStore(persist_directory=tmpdir)
    collector = FeedbackCollector(store)

    # Submit corrections: teach the judge that paraphrases are faithful
    corrections = [
        {
            "output": "Apply the cream twice daily",
            "context": "Use topical cream 2x per day",
            "original_score": 0.3,
            "correct_score": 0.95,
            "reason": "Semantically equivalent — 'twice daily' == '2x per day'",
        },
        {
            "output": "Take 500mg of ibuprofen for pain",
            "context": "Prescribe 500mg ibuprofen for pain management",
            "original_score": 0.4,
            "correct_score": 0.9,
            "reason": "Faithful — correctly states the prescription",
        },
        {
            "output": "Take this medication forever",
            "context": "Take for 7 days only",
            "original_score": 0.7,
            "correct_score": 0.1,
            "reason": "UNFAITHFUL — hallucinated 'forever', context says 7 days",
        },
        {
            "output": "Avoid all physical activity",
            "context": "Light exercise is recommended during recovery",
            "original_score": 0.5,
            "correct_score": 0.05,
            "reason": "UNFAITHFUL — directly contradicts context recommendation",
        },
        {
            "output": "The dosage is 200mg per day",
            "context": "Recommended daily dose: 200 milligrams",
            "original_score": 0.35,
            "correct_score": 0.95,
            "reason": "Faithful — exact same dosage, just different wording",
        },
    ]

    print(f"Submitting {len(corrections)} feedback corrections...")
    for c in corrections:
        fake_result = EvalResult(
            eval_name="faithfulness",
            score=c["original_score"],
            reason=f"Heuristic score: {c['original_score']}",
        )
        entry = collector.submit(
            fake_result,
            inputs={"output": c["output"], "context": c["context"]},
            correct_score=c["correct_score"],
            correct_reason=c["reason"],
        )
        print(f"  {c['original_score']:.1f} -> {c['correct_score']:.2f} | {c['reason'][:55]}")

    print(f"\nChromaDB entries: {store.count('faithfulness')}")

    # --- Step 3: Show what gets retrieved ---
    divider("STEP 3: Semantic retrieval for the test input")

    retriever = FeedbackRetriever(store=store, max_examples=3)
    examples = retriever.retrieve_few_shot_examples(
        "faithfulness",
        {"output": test_output, "context": test_context},
    )
    print(f"Retrieved {len(examples)} similar feedback entries:")
    for i, ex in enumerate(examples):
        parsed = json.loads(ex["output"])
        print(f"  {i+1}. score={parsed['score']:.2f} | {parsed['reason'][:60]}")

    # --- Step 4: Run faithfulness WITH feedback ---
    divider("STEP 4: Run faithfulness WITH feedback (few-shot injected)")

    print(f"Output:  {test_output}")
    print(f"Context: {test_context}")

    result_with_feedback = evaluate(
        "faithfulness",
        output=test_output,
        context=test_context,
        model=model,
        augment=True,
        feedback_store=store,
    )
    print(f"\nResult WITH feedback:")
    print(f"  Score:  {result_with_feedback.score}")
    print(f"  Reason: {result_with_feedback.reason[:200]}")
    print(f"  Engine: {result_with_feedback.metadata.get('engine', 'unknown')}")
    examples_used = result_with_feedback.metadata.get("feedback_examples_used", 0)
    print(f"  Feedback examples injected: {examples_used}")

    # --- Step 5: Compare ---
    divider("COMPARISON")

    print(f"WITHOUT feedback: score={result_no_feedback.score}")
    print(f"WITH feedback:    score={result_with_feedback.score}")
    print(f"Feedback examples used: {examples_used}")

    if examples_used > 0:
        print("\nThe LLM judge received few-shot examples from your past")
        print("corrections, teaching it how to handle paraphrases in")
        print("medical contexts. This is the feedback loop in action.")
    else:
        print("\nNote: No feedback examples were injected. This can happen")
        print("if the retriever found no sufficiently similar entries.")

    # --- Step 6: Test a clearly unfaithful case ---
    divider("BONUS: Test an unfaithful case WITH feedback")

    bad_output = "Stop all medications immediately"
    bad_context = "Continue current medication regimen as prescribed"

    print(f"Output:  {bad_output}")
    print(f"Context: {bad_context}")
    print("(These CONTRADICT each other — score should be LOW)")

    result_bad = evaluate(
        "faithfulness",
        output=bad_output,
        context=bad_context,
        model=model,
        augment=True,
        feedback_store=store,
    )
    print(f"\nResult:")
    print(f"  Score:  {result_bad.score}")
    print(f"  Reason: {result_bad.reason[:200]}")
    bad_examples = result_bad.metadata.get("feedback_examples_used", 0)
    print(f"  Feedback examples injected: {bad_examples}")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    # --- Step 7: Calibration ---
    divider("CALIBRATION")
    # Use InMemory store for calibration demo (ChromaDB was cleaned up)
    mem_store = InMemoryFeedbackStore()
    cal_collector = FeedbackCollector(mem_store)
    for c in corrections:
        fake_result = EvalResult(
            eval_name="faithfulness",
            score=c["original_score"],
            reason="",
        )
        cal_collector.submit(
            fake_result,
            inputs={"output": c["output"], "context": c["context"]},
            correct_score=c["correct_score"],
            correct_reason=c["reason"],
        )

    profile = cal_collector.calibrate("faithfulness")
    print(f"Optimal threshold: {profile.optimal_threshold}")
    print(f"Accuracy: {profile.accuracy_at_threshold:.0%}")
    print(f"Sample size: {profile.sample_size}")
    print(f"TP={profile.true_positives} FP={profile.false_positives} "
          f"TN={profile.true_negatives} FN={profile.false_negatives}")


if __name__ == "__main__":
    # Load env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().strip('"')
                    os.environ.setdefault(key.strip(), val)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set. Export it or add to .env")
        exit(1)

    print("=" * 60)
    print("  FI-EVALS FEEDBACK LOOP — REAL LLM JUDGE E2E")
    print("=" * 60)

    demo_real_llm_judge()

    divider("DONE")
    print("The feedback loop works end-to-end:")
    print("  1. Feedback stored in ChromaDB with semantic embeddings")
    print("  2. Similar past corrections retrieved via vector search")
    print("  3. Injected as few-shot examples into LLM judge prompt")
    print("  4. Gemini produces calibrated scores informed by your feedback")
    print("  5. Thresholds optimized statistically from feedback data")
