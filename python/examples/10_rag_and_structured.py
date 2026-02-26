"""
Insurance FAQ Bot Evaluation — RAG retrieval, generation, and structured output.

Scenario: A customer asks an AI support bot about their health insurance plan.
The bot retrieves context chunks from the knowledge base, generates a natural
language answer, AND returns a structured JSON ticket for the CRM system.

We evaluate four axes:

  1. Retrieval quality    — did we fetch the right chunks?
  2. Generation quality   — is the answer faithful, relevant, grounded?
  3. Faithfulness deep-dive — compare three NLI-based faithfulness metrics
  4. Structured output    — does the JSON ticket match the schema?

Run:
    poetry run python examples/10_rag_and_structured.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fi.evals import evaluate


def heading(text):
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")


def show(label, result):
    reason = result.reason or ""
    # LLM augmented results return JSON — extract just the reason text
    if reason.startswith("{"):
        try:
            import json as _json
            reason = _json.loads(reason).get("reason", reason)
        except Exception:
            pass
    reason = reason[:90]
    print(f"  {label:42s}  score={result.score:.2f}  | {reason}")


# =========================================================================
# Knowledge base — insurance policy chunks
# =========================================================================

KNOWLEDGE_BASE = [
    # Chunk 0 — deductible info (RELEVANT)
    "The Standard Health Plan has an annual deductible of $1,500 for "
    "individuals and $3,000 for families. Once the deductible is met, "
    "the plan covers 80% of in-network costs and 60% of out-of-network costs.",

    # Chunk 1 — prescription coverage (RELEVANT)
    "Prescription drugs are covered under the Standard Health Plan after "
    "a $10 copay for generic medications and a $35 copay for brand-name "
    "drugs. Specialty medications require prior authorization and have "
    "a $75 copay per fill.",

    # Chunk 2 — dental info (IRRELEVANT to the query)
    "Dental coverage is available as an add-on to the Standard Health Plan. "
    "Preventive care including cleanings and X-rays is covered at 100%. "
    "Major procedures like crowns and bridges are covered at 50%.",

    # Chunk 3 — emergency care (SOMEWHAT RELEVANT)
    "Emergency room visits are covered with a $250 copay, which is waived "
    "if the patient is admitted to the hospital. Urgent care visits have "
    "a $50 copay. Ambulance services are covered at 80% after the deductible.",
]

# What the customer asked
CUSTOMER_QUERY = "What's my deductible and how much do I pay for prescriptions?"

# Ground truth answer (for recall/precision evaluation)
REFERENCE_ANSWER = (
    "The annual deductible is $1,500 for individuals and $3,000 for families. "
    "Generic prescriptions have a $10 copay and brand-name drugs cost $35. "
    "Specialty medications require prior authorization with a $75 copay."
)

# Simulated retriever returned chunks 0, 1, 2 — note: chunk 2 is noise
RETRIEVED_CHUNKS = [KNOWLEDGE_BASE[0], KNOWLEDGE_BASE[1], KNOWLEDGE_BASE[2]]


# =========================================================================
# Bot responses — GOOD vs BAD
# =========================================================================

GOOD_ANSWER = (
    "Your annual deductible is $1,500 for individuals or $3,000 for families. "
    "For prescriptions, you'll pay a $10 copay for generics, $35 for brand-name "
    "drugs, and $75 for specialty medications (which need prior authorization). "
    "After meeting your deductible, the plan covers 80% of in-network costs."
)

BAD_ANSWER = (
    "Your deductible is $2,000 per person. Prescriptions are fully covered "
    "with no copay after your deductible is met. The plan also includes "
    "free dental cleanings and vision coverage."
)

# Structured CRM tickets
GOOD_TICKET = json.dumps({
    "ticket_type": "inquiry",
    "category": "benefits",
    "subcategory": "deductible_and_rx",
    "customer_query": CUSTOMER_QUERY,
    "resolved": True,
    "confidence": 0.92,
    "escalate": False,
})

BAD_TICKET = json.dumps({
    "ticket_type": "inquiry",
    "resolved": True,
    # missing: category, subcategory, customer_query, confidence, escalate
})

TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "ticket_type": {"type": "string", "enum": ["inquiry", "complaint", "claim"]},
        "category": {"type": "string"},
        "subcategory": {"type": "string"},
        "customer_query": {"type": "string"},
        "resolved": {"type": "boolean"},
        "confidence": {"type": "number"},
        "escalate": {"type": "boolean"},
    },
    "required": ["ticket_type", "category", "subcategory", "customer_query",
                  "resolved", "confidence", "escalate"],
}


# =========================================================================
# Part 1 — Retrieval Quality
# =========================================================================

heading("Part 1: Retrieval Quality")
print("  Did the retriever fetch the right chunks?")
print(f"  Retrieved: chunks 0 (deductible), 1 (rx), 2 (dental — noise)\n")

show("Context Recall", evaluate(
    "context_recall",
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
    reference=REFERENCE_ANSWER,
))

show("Context Precision", evaluate(
    "context_precision",
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
    reference=REFERENCE_ANSWER,
))

show("Context Entity Recall", evaluate(
    "context_entity_recall",
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
    reference=REFERENCE_ANSWER,
))


# =========================================================================
# Part 2 — Generation Quality (good vs bad)
# =========================================================================

heading("Part 2: Generation Quality — Good Answer")
print(f"  \"{GOOD_ANSWER[:75]}...\"\n")

show("Answer Relevancy", evaluate(
    "answer_relevancy",
    response=GOOD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
))

show("Context Utilization", evaluate(
    "context_utilization",
    response=GOOD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
))

show("RAG Faithfulness", evaluate(
    "rag_faithfulness",
    response=GOOD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
))

show("Groundedness", evaluate(
    "groundedness",
    response=GOOD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
))

show("RAG Score (composite)", evaluate(
    "rag_score",
    response=GOOD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
    reference=REFERENCE_ANSWER,
))


heading("Part 2b: Generation Quality — Bad Answer (hallucinated)")
print(f"  \"{BAD_ANSWER[:75]}...\"\n")

show("Answer Relevancy", evaluate(
    "answer_relevancy",
    response=BAD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
))

show("Context Utilization", evaluate(
    "context_utilization",
    response=BAD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
))

show("RAG Faithfulness", evaluate(
    "rag_faithfulness",
    response=BAD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
))

show("Groundedness", evaluate(
    "groundedness",
    response=BAD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
))

show("RAG Score (composite)", evaluate(
    "rag_score",
    response=BAD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
    reference=REFERENCE_ANSWER,
))


# =========================================================================
# Part 3 — Faithfulness Deep-Dive
# =========================================================================

heading("Part 3: Faithfulness Deep-Dive — Three NLI Metrics Compared")
print("  All three use the same NLI model (DeBERTa) but score differently:")
print("  - rag_faithfulness: supported + 0.4*neutral partial credit")
print("  - groundedness:     supported - 2x contradiction penalty")
print("  - faithfulness:     hallucination module, same NLI, Claim objects\n")

# Run all three on the BAD answer
context_str = " ".join(RETRIEVED_CHUNKS)

r_rag = evaluate("rag_faithfulness", response=BAD_ANSWER,
                 query=CUSTOMER_QUERY, contexts=RETRIEVED_CHUNKS)
r_ground = evaluate("groundedness", response=BAD_ANSWER,
                    query=CUSTOMER_QUERY, contexts=RETRIEVED_CHUNKS)
r_faith = evaluate("faithfulness", output=BAD_ANSWER, context=context_str)
r_contra = evaluate("contradiction_detection", output=BAD_ANSWER, context=context_str)
r_halluc = evaluate("hallucination_score", output=BAD_ANSWER, context=context_str)

show("RAG Faithfulness", r_rag)
show("Groundedness", r_ground)
show("Faithfulness (hallucination)", r_faith)
show("Contradiction Detection", r_contra)
show("Hallucination Score (composite)", r_halluc)


# =========================================================================
# Part 4 — Structured Output (CRM ticket)
# =========================================================================

heading("Part 4: Structured Output — CRM Ticket")

print("  Good ticket (all 7 required fields):")
show("  JSON Validation", evaluate(
    "json_validation", output=GOOD_TICKET, schema=TICKET_SCHEMA,
))
show("  Schema Compliance", evaluate(
    "schema_compliance", output=GOOD_TICKET, schema=TICKET_SCHEMA,
))
show("  Field Completeness", evaluate(
    "field_completeness", output=GOOD_TICKET, schema=TICKET_SCHEMA,
))
show("  Type Compliance", evaluate(
    "type_compliance", output=GOOD_TICKET, schema=TICKET_SCHEMA,
))
show("  Structured Score (composite)", evaluate(
    "structured_output_score", output=GOOD_TICKET, schema=TICKET_SCHEMA,
))

print()
print("  Bad ticket (missing 5 of 7 required fields):")
show("  JSON Validation", evaluate(
    "json_validation", output=BAD_TICKET, schema=TICKET_SCHEMA,
))
show("  Schema Compliance", evaluate(
    "schema_compliance", output=BAD_TICKET, schema=TICKET_SCHEMA,
))
show("  Field Completeness", evaluate(
    "field_completeness", output=BAD_TICKET, schema=TICKET_SCHEMA,
))
show("  Type Compliance", evaluate(
    "type_compliance", output=BAD_TICKET, schema=TICKET_SCHEMA,
))
show("  Structured Score (composite)", evaluate(
    "structured_output_score", output=BAD_TICKET, schema=TICKET_SCHEMA,
))

print()
print("  No schema provided (syntax only):")
show("  Structured Score (no schema)", evaluate(
    "structured_output_score", output=GOOD_TICKET,
))


# =========================================================================
# Part 5 — Side-by-side summary
# =========================================================================

heading("Part 5: Side-by-Side Summary")

good_rag = evaluate("rag_score", response=GOOD_ANSWER, query=CUSTOMER_QUERY,
                     contexts=RETRIEVED_CHUNKS, reference=REFERENCE_ANSWER)
bad_rag = evaluate("rag_score", response=BAD_ANSWER, query=CUSTOMER_QUERY,
                    contexts=RETRIEVED_CHUNKS, reference=REFERENCE_ANSWER)
good_faith = evaluate("rag_faithfulness", response=GOOD_ANSWER,
                       query=CUSTOMER_QUERY, contexts=RETRIEVED_CHUNKS)
bad_faith = evaluate("rag_faithfulness", response=BAD_ANSWER,
                      query=CUSTOMER_QUERY, contexts=RETRIEVED_CHUNKS)
good_ground = evaluate("groundedness", response=GOOD_ANSWER,
                        query=CUSTOMER_QUERY, contexts=RETRIEVED_CHUNKS)
bad_ground = evaluate("groundedness", response=BAD_ANSWER,
                       query=CUSTOMER_QUERY, contexts=RETRIEVED_CHUNKS)
good_struct = evaluate("structured_output_score", output=GOOD_TICKET, schema=TICKET_SCHEMA)
bad_struct = evaluate("structured_output_score", output=BAD_TICKET, schema=TICKET_SCHEMA)

print(f"  {'Metric':<30s}  {'Good':>6s}  {'Bad':>6s}")
print(f"  {'─' * 48}")
print(f"  {'RAG Score (composite)':<30s}  {good_rag.score:6.2f}  {bad_rag.score:6.2f}")
print(f"  {'RAG Faithfulness':<30s}  {good_faith.score:6.2f}  {bad_faith.score:6.2f}")
print(f"  {'Groundedness':<30s}  {good_ground.score:6.2f}  {bad_ground.score:6.2f}")
print(f"  {'Structured Output Score':<30s}  {good_struct.score:6.2f}  {bad_struct.score:6.2f}")
print()


# =========================================================================
# Part 6 — LLM-Augmented Evaluation (local heuristic + LLM refinement)
# =========================================================================

heading("Part 6: LLM-Augmented Evaluation")
print("  Same metrics, but augment=True sends local NLI scores to an LLM")
print("  for refinement. The LLM sees the heuristic analysis + raw data.\n")

# Check if a model is available
LLM_MODEL = os.environ.get("GOOGLE_API_KEY") and "gemini/gemini-2.5-flash"

if not LLM_MODEL:
    print("  [SKIPPED] Set GOOGLE_API_KEY to run LLM-augmented evaluation.")
    print("  Example: export GOOGLE_API_KEY=your_key_here\n")
else:
    print(f"  Model: {LLM_MODEL}\n")
    context_str = " ".join(RETRIEVED_CHUNKS)

    # Local-only vs LLM-augmented on the BAD answer
    local_faith = evaluate("faithfulness", output=BAD_ANSWER, context=context_str)
    augmented_faith = evaluate(
        "faithfulness", output=BAD_ANSWER, context=context_str,
        model=LLM_MODEL, augment=True,
    )
    show("Faithfulness (local NLI)", local_faith)
    show("Faithfulness (LLM-augmented)", augmented_faith)

    print()

    local_ground = evaluate(
        "groundedness", response=BAD_ANSWER,
        query=CUSTOMER_QUERY, contexts=RETRIEVED_CHUNKS,
    )
    augmented_ground = evaluate(
        "groundedness", response=BAD_ANSWER,
        query=CUSTOMER_QUERY, contexts=RETRIEVED_CHUNKS,
        model=LLM_MODEL, augment=True,
    )
    show("Groundedness (local NLI)", local_ground)
    show("Groundedness (LLM-augmented)", augmented_ground)

    print()

    local_rag_faith = evaluate(
        "rag_faithfulness", response=BAD_ANSWER,
        query=CUSTOMER_QUERY, contexts=RETRIEVED_CHUNKS,
    )
    augmented_rag_faith = evaluate(
        "rag_faithfulness", response=BAD_ANSWER,
        query=CUSTOMER_QUERY, contexts=RETRIEVED_CHUNKS,
        model=LLM_MODEL, augment=True,
    )
    show("RAG Faithfulness (local NLI)", local_rag_faith)
    show("RAG Faithfulness (LLM-augmented)", augmented_rag_faith)

    print()
    print("  Engine metadata:")
    print(f"    local:     engine={local_faith.metadata.get('engine', 'local')}")
    print(f"    augmented: engine={augmented_faith.metadata.get('engine', '?')}")

print("\n  Done.")
