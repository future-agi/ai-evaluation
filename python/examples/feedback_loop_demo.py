#!/usr/bin/env python3
"""
Feedback Loop — End-to-End Demo

Demonstrates the full feedback loop:
1. Run a metric locally (faithfulness heuristic)
2. Developer disagrees with the result
3. Submit feedback → stored in ChromaDB with semantic embeddings
4. Run the metric again on similar input → feedback retrieved as few-shot
5. Calibrate thresholds from accumulated feedback
6. Show stats

Usage:
    cd python && poetry run python examples/feedback_loop_demo.py
"""

import json
import shutil
import tempfile

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


def demo_inmemory():
    """Quick demo with InMemoryFeedbackStore (no ChromaDB needed)."""
    divider("DEMO 1: InMemory Feedback Store")

    store = InMemoryFeedbackStore()
    collector = FeedbackCollector(store)

    # Simulate eval results that a developer disagrees with
    bad_results = [
        (
            EvalResult(eval_name="faithfulness", score=0.2, reason="Output contradicts context"),
            {"output": "The patient should take 500mg of ibuprofen", "context": "Prescribe 500mg ibuprofen for pain"},
            0.9, "Output IS faithful - it correctly states the prescription",
        ),
        (
            EvalResult(eval_name="faithfulness", score=0.3, reason="Low semantic overlap"),
            {"output": "Apply the cream twice daily", "context": "Use topical cream 2x per day"},
            0.95, "Semantically equivalent - 'twice daily' == '2x per day'",
        ),
        (
            EvalResult(eval_name="faithfulness", score=0.8, reason="High faithfulness"),
            {"output": "Take this medication forever", "context": "Take for 7 days only"},
            0.1, "Hallucinated 'forever' - context says 7 days only",
        ),
    ]

    print("Submitting 3 feedback entries...")
    for result, inputs, correct_score, reason in bad_results:
        entry = collector.submit(
            result,
            inputs=inputs,
            correct_score=correct_score,
            correct_reason=reason,
        )
        print(f"  {result.score:.1f} -> {correct_score:.1f} | {reason[:60]}")

    # Also confirm some correct results
    good_result = EvalResult(eval_name="faithfulness", score=0.85, reason="Faithful output")
    collector.confirm(
        good_result,
        inputs={"output": "Take 200mg aspirin", "context": "Prescribe 200mg aspirin"},
    )
    print("  Confirmed 1 correct result")

    # Retrieve few-shot examples for a new input
    divider("RETRIEVAL: Few-shot examples for new input")
    retriever = collector.get_retriever(max_examples=2)
    examples = retriever.retrieve_few_shot_examples(
        "faithfulness",
        {"output": "Use the medication three times daily", "context": "Apply 3x per day"},
    )
    print(f"Retrieved {len(examples)} few-shot examples:")
    for i, ex in enumerate(examples):
        output = json.loads(ex["output"])
        print(f"  Example {i+1}: score={output['score']}, reason={output['reason'][:60]}")

    # Show stats
    divider("STATS")
    stats = collector.stats("faithfulness")
    print(f"Total entries: {stats.total_entries}")
    print(f"Agreement rate: {stats.agreement_rate:.0%}")
    print(f"Avg score delta: {stats.avg_score_delta:+.2f}")
    print(f"Score distribution: {stats.score_distribution}")

    # Calibrate (need >= 5 entries with corrections)
    # Add 2 more to reach 5
    collector.submit(
        EvalResult(eval_name="faithfulness", score=0.5, reason="Medium"),
        inputs={"output": "Take as needed", "context": "Take when experiencing pain"},
        correct_score=0.7,
        correct_reason="Reasonable paraphrase",
    )
    collector.submit(
        EvalResult(eval_name="faithfulness", score=0.6, reason="Moderate"),
        inputs={"output": "Avoid alcohol", "context": "Do not consume alcohol with this medication"},
        correct_score=0.85,
        correct_reason="Correct advice, just shortened",
    )

    divider("CALIBRATION")
    profile = collector.calibrate("faithfulness")
    print(f"Optimal threshold: {profile.optimal_threshold}")
    print(f"Accuracy at threshold: {profile.accuracy_at_threshold:.0%}")
    print(f"Sample size: {profile.sample_size}")
    print(f"Score mean: {profile.score_mean:.3f}")
    print(f"Score std: {profile.score_std:.3f}")
    print(f"Confusion matrix: TP={profile.true_positives} FP={profile.false_positives} "
          f"TN={profile.true_negatives} FN={profile.false_negatives}")


def demo_chromadb():
    """Full demo with ChromaDB (local persistent mode)."""
    divider("DEMO 2: ChromaDB Feedback Store (Local Persistent)")

    # Use a temp directory so we don't pollute ~/.fi
    tmpdir = tempfile.mkdtemp(prefix="fi_feedback_demo_")
    print(f"ChromaDB path: {tmpdir}")

    try:
        store = ChromaFeedbackStore(persist_directory=tmpdir)
        collector = FeedbackCollector(store)

        # Submit diverse feedback for RAG faithfulness
        scenarios = [
            # (original_score, correct_score, output, context, reason)
            (0.2, 0.9, "Paris is the capital of France", "France's capital city is Paris", "Semantically equivalent"),
            (0.3, 0.85, "The API returns JSON data", "REST endpoint responds with JSON format", "Same meaning, different words"),
            (0.7, 0.1, "Python is the fastest language", "Python is known for readability not speed", "Contradicts context"),
            (0.1, 0.8, "Use TLS 1.3 for encryption", "Encrypt connections with TLS version 1.3", "Faithful technical advice"),
            (0.4, 0.7, "The model has 7B parameters", "LLM contains 7 billion parameters", "Same fact, abbreviated"),
            (0.9, 0.2, "Always use root access", "Use least-privilege access principle", "Dangerous contradiction"),
        ]

        print(f"Submitting {len(scenarios)} feedback entries...")
        for orig, correct, output, context, reason in scenarios:
            result = EvalResult(eval_name="faithfulness", score=orig, reason=f"Score: {orig}")
            collector.submit(
                result,
                inputs={"output": output, "context": context},
                correct_score=correct,
                correct_reason=reason,
            )
            print(f"  {orig:.1f} -> {correct:.1f} | {reason}")

        # Verify ChromaDB persistence
        count = store.count("faithfulness")
        print(f"\nChromaDB count: {count} entries stored")

        # Semantic retrieval — find similar feedback
        divider("SEMANTIC RETRIEVAL")
        retriever = FeedbackRetriever(store=store, max_examples=3)

        # Query about similar technical content
        query_inputs = {
            "output": "Enable HTTPS for all connections",
            "context": "All web traffic must use HTTPS encryption",
        }
        examples = retriever.retrieve_few_shot_examples("faithfulness", query_inputs)
        print(f"Query: '{query_inputs['output']}'")
        print(f"Retrieved {len(examples)} semantically similar examples:")
        for i, ex in enumerate(examples):
            output = json.loads(ex["output"])
            print(f"  {i+1}. score={output['score']:.1f} | {output['reason'][:70]}")

        # Config injection (what the LLM judge would receive)
        divider("CONFIG INJECTION (what LLM judge sees)")
        config = retriever.inject_into_config("faithfulness", query_inputs)
        if "few_shot_examples" in config:
            print(f"Injected {len(config['few_shot_examples'])} few-shot examples into judge config")
            print("First example preview:")
            first = config["few_shot_examples"][0]
            print(f"  inputs: {json.dumps(first['inputs'])[:100]}...")
            print(f"  output: {first['output']}")
        else:
            print("No examples injected (store might be empty)")

        # Calibrate
        divider("CALIBRATION (ChromaDB-backed)")
        profile = collector.calibrate("faithfulness")
        print(f"Optimal threshold: {profile.optimal_threshold}")
        print(f"Accuracy: {profile.accuracy_at_threshold:.0%}")
        print(f"Sample size: {profile.sample_size}")

        # Global configuration
        divider("GLOBAL FEEDBACK CONFIGURATION")
        configure_feedback(store, max_examples=3)
        print("Feedback configured globally.")
        print("All augmented metric runs will now auto-retrieve feedback.")

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"\nCleaned up: {tmpdir}")


if __name__ == "__main__":
    print("=" * 60)
    print("  FI-EVALS FEEDBACK LOOP — END-TO-END DEMO")
    print("=" * 60)

    demo_inmemory()
    print("\n" + "~" * 60 + "\n")

    try:
        demo_chromadb()
    except ImportError as e:
        print(f"\nSkipping ChromaDB demo: {e}")
        print("Install with: pip install ai-evaluation[feedback]")

    divider("DEMO COMPLETE")
    print("The feedback loop is fully functional!")
    print("Key features demonstrated:")
    print("  - Submit feedback on eval results")
    print("  - Semantic retrieval of similar past feedback")
    print("  - Few-shot injection into LLM judge config")
    print("  - Statistical threshold calibration")
    print("  - ChromaDB persistent storage")
    print("  - Global feedback configuration")
