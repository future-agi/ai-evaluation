#!/usr/bin/env python3
"""
Cookbook 05 — Stop Toxic LLM Output Mid-Stream

SCENARIO:
    You're serving LLM responses via streaming (SSE/WebSocket).
    The LLM starts generating a helpful response... then suddenly
    veers into toxic, harmful, or off-topic territory.

    You can't wait for the full response — by then, the user has
    already read the toxic content. You need to monitor the stream
    token-by-token and CUT IT OFF the moment things go wrong.

    This cookbook shows how to build a real-time stream monitor that:
    - Tracks toxicity, coherence, and topic coverage as tokens arrive
    - Stops generation immediately when safety thresholds are breached
    - Fires callbacks for logging and alerting
    - Reports exactly WHERE the stream went bad

Usage:
    cd python && poetry run python -m examples.05_streaming
"""

import time
from fi.evals import (
    StreamingEvaluator,
    StreamingConfig,
    EarlyStopPolicy,
)


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ── Simulate streaming from an LLM ──────────────────────────────
def simulate_llm_stream(text: str, words_per_chunk: int = 3):
    """Simulate token-by-token streaming from an LLM."""
    words = text.split()
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i + words_per_chunk])
        yield chunk + " "
        time.sleep(0.01)


# ── Scoring functions (plug in your own) ─────────────────────────
def detect_toxicity(chunk: str, full_text: str) -> float:
    """Keyword-based toxicity. In production, use a real model."""
    toxic_words = {"kill", "hate", "die", "stupid", "idiot", "destroy", "attack"}
    words = full_text.lower().split()
    toxic_count = sum(1 for w in words if w.strip(".,!?") in toxic_words)
    return min(toxic_count / max(len(words), 1) * 10, 1.0)


def check_coherence(chunk: str, full_text: str) -> float:
    """Measure vocabulary diversity (proxy for coherence)."""
    words = full_text.lower().split()
    if len(words) < 3:
        return 1.0
    return len(set(words)) / len(words)


def track_topic(chunk: str, full_text: str) -> float:
    """Check if the response stays on topic (cooking keywords)."""
    cooking_words = {"recipe", "cook", "ingredient", "heat", "stir", "bake",
                     "mix", "chop", "serve", "pan", "oven", "minutes", "food"}
    words = set(full_text.lower().split())
    found = words & cooking_words
    return min(len(found) / 3, 1.0)  # need at least 3 keywords


# ── Scenario 1: Normal response — stream completes ──────────────
divider("SCENARIO 1: Normal response (stream completes)")

monitor = StreamingEvaluator.with_defaults()
monitor.add_eval("toxicity", detect_toxicity, threshold=0.2, pass_above=False, weight=2.0)
monitor.add_eval("coherence", check_coherence, threshold=0.4, pass_above=True, weight=1.0)

normal_response = (
    "To make a classic pasta carbonara, start by cooking the spaghetti "
    "in salted boiling water. While the pasta cooks, mix egg yolks with "
    "grated pecorino cheese. Pan fry the guanciale until crispy. Combine "
    "the hot pasta with the egg mixture and toss with the crispy guanciale."
)

print("Streaming a cooking recipe...")
print("User asked: 'How do I make pasta carbonara?'\n")

for token in simulate_llm_stream(normal_response):
    result = monitor.process_token(token)
    if result:
        tox = result.scores.get("toxicity", 0)
        coh = result.scores.get("coherence", 0)
        print(f"  chunk {result.chunk_index}: tox={tox:.2f} coh={coh:.2f} ", end="")
        print(f"{'OK' if result.all_passed else 'ALERT'}")

final = monitor.finalize()
print(f"\nStream completed normally.")
print(f"  Passed: {final.passed}  |  Chunks: {final.total_chunks}  |  "
      f"Latency: {final.total_latency_ms:.0f}ms")


# ── Scenario 2: Toxic turn — stream gets cut ────────────────────
divider("SCENARIO 2: Response turns toxic (stream gets CUT)")

monitor = StreamingEvaluator.for_safety(toxicity_threshold=0.3)
monitor.add_eval("toxicity", detect_toxicity, threshold=0.15, pass_above=False, weight=2.0)
monitor.set_policy(EarlyStopPolicy.strict())

toxic_response = (
    "Here's a recipe for chocolate cake. First, preheat your oven "
    "to 350 degrees. Then I hate to say this but you should destroy "
    "all the stupid ingredients. Kill the recipe and die. "
    "Anyway, mix the flour with sugar."
)

print("Streaming a response that turns toxic mid-way...")
print("User asked: 'Give me a chocolate cake recipe'\n")

for token in simulate_llm_stream(toxic_response):
    result = monitor.process_token(token)
    if result:
        tox = result.scores.get("toxicity", 0)
        status = "OK" if result.all_passed else "!!! TOXIC"
        print(f"  chunk {result.chunk_index}: tox={tox:.2f}  {status}")
        if result.should_stop:
            print(f"\n  >>> STREAM CUT at chunk {result.chunk_index}")
            print(f"  >>> Reason: {result.stop_reason}")
            break

final = monitor.finalize()
print(f"\n  Early stopped: {final.early_stopped}")
print(f"  Text before cut: '{final.final_text[:80]}...'")
print(f"  Chunks processed: {final.total_chunks}")


# ── Scenario 3: Quality degradation over time ───────────────────
divider("SCENARIO 3: Track quality degradation")

monitor = StreamingEvaluator.for_quality(min_chunk_size=10)
monitor.add_eval("on_topic", track_topic, threshold=0.3, pass_above=True)
monitor.add_eval("coherence", check_coherence, threshold=0.4, pass_above=True)

drifting_response = (
    "To bake bread, you need flour, water, yeast, and salt. "
    "Mix the ingredients and knead the dough for ten minutes. "
    "Speaking of minutes, time is a fascinating concept in physics. "
    "Einstein showed that time is relative. The speed of light "
    "is approximately 300 million meters per second. Quantum mechanics "
    "suggests that particles exist in superposition until observed."
)

print("Streaming a response that drifts off-topic...")
print("User asked: 'How do I bake bread?'\n")

for token in simulate_llm_stream(drifting_response, words_per_chunk=5):
    result = monitor.process_token(token)
    if result:
        topic = result.scores.get("on_topic", 0)
        bar = "#" * int(topic * 10)
        print(f"  chunk {result.chunk_index}: topic={topic:.2f} |{bar:<10}| "
              f"{'on-topic' if topic >= 0.3 else 'DRIFTING'}")

final = monitor.finalize()
print(f"\nTopic coverage over time:")
for i, score in enumerate(final.score_by_eval.get("on_topic", [])):
    trend = ">>>" if score >= 0.3 else "..."
    print(f"  chunk {i}: {score:.2f} {trend}")
print(f"\nFinal passed: {final.passed}")


# ── Scenario 4: Real-time alerting with callbacks ────────────────
divider("SCENARIO 4: Alert on safety violations")

incidents = []


def on_chunk_alert(chunk_result):
    """Called after every chunk. Log violations."""
    if not chunk_result.all_passed:
        incidents.append({
            "chunk": chunk_result.chunk_index,
            "text": chunk_result.cumulative_text[-50:],
            "scores": dict(chunk_result.scores),
        })


def on_emergency_stop(reason, text):
    """Called when stream is killed."""
    incidents.append({
        "type": "EMERGENCY_STOP",
        "reason": str(reason),
        "text_length": len(text),
    })


monitor = StreamingEvaluator(
    config=StreamingConfig(
        min_chunk_size=5,
        on_chunk_callback=on_chunk_alert,
        on_stop_callback=on_emergency_stop,
        enable_early_stop=True,
    ),
)
monitor.add_eval("toxicity", detect_toxicity, threshold=0.1, pass_above=False, weight=1.0)

adversarial = (
    "I'd be happy to help! However, I hate people who are stupid "
    "and they should all die. Just kidding! Let me actually help you."
)

print("Processing adversarial content with alerting enabled...\n")
for token in simulate_llm_stream(adversarial):
    result = monitor.process_token(token)
    if result and result.should_stop:
        break
monitor.finalize()

print(f"Incidents logged: {len(incidents)}")
for inc in incidents:
    if inc.get("type") == "EMERGENCY_STOP":
        print(f"  STOP: {inc['reason']}")
    else:
        print(f"  Violation at chunk {inc['chunk']}: "
              f"toxicity={inc['scores'].get('toxicity', 0):.2f}")


# ── Scenario 5: One-shot processing ─────────────────────────────
divider("SCENARIO 5: Quick one-shot stream check")

monitor = StreamingEvaluator.with_defaults()
monitor.add_eval("coherence", check_coherence, threshold=0.5, pass_above=True, weight=1.0)
monitor.add_eval("on_topic", track_topic, threshold=0.3, pass_above=True, weight=1.0)

stream = simulate_llm_stream(
    "Heat the oven to 375 degrees. Mix flour and butter. "
    "Bake for 25 minutes until golden brown. Serve warm."
)
final = monitor.evaluate_stream(stream)

print(f"Quick check: passed={final.passed}")
print(f"Scores: {final.final_scores}")
print(f"\n{final.summary()}")


divider("DONE")
print("Real-time stream monitoring protects users from:")
print("  - Toxic content that appears mid-response")
print("  - Off-topic drift away from the user's question")
print("  - Quality degradation in long responses")
print("  - Any custom safety signal you define")
