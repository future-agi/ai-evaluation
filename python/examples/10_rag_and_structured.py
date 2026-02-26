"""
Insurance FAQ Bot Evaluation — RAG retrieval, generation, and structured output.

Scenario: A customer asks an AI support bot about their health insurance plan.
The bot retrieves context chunks from the knowledge base, generates a natural
language answer, AND returns a structured JSON ticket for the CRM system.

We evaluate three axes:

  1. Retrieval quality  — did we fetch the right chunks?
  2. Generation quality — is the answer relevant, faithful, grounded?
  3. Structured output  — does the JSON ticket match the schema?

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
    reason = (result.reason or "")[:90]
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

# Simulated retriever returned these chunks (indices 0, 1, 2 — note: chunk 2 is noise)
RETRIEVED_CHUNKS = [KNOWLEDGE_BASE[0], KNOWLEDGE_BASE[1], KNOWLEDGE_BASE[2]]


# =========================================================================
# Bot responses — we'll evaluate a GOOD one and a BAD one
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

# The bot also outputs a structured CRM ticket
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

# CRM ticket schema
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
print("  Did the retriever fetch the right knowledge base chunks?\n")

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
# Part 2 — Generation Quality (good answer)
# =========================================================================

heading("Part 2: Generation Quality — Good Answer")
print(f"  Answer: \"{GOOD_ANSWER[:80]}...\"\n")

show("Answer Relevancy", evaluate(
    "answer_relevancy",
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

show("RAG Faithfulness", evaluate(
    "rag_faithfulness",
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


# =========================================================================
# Part 3 — Generation Quality (bad answer — hallucinated)
# =========================================================================

heading("Part 3: Generation Quality — Bad Answer (hallucinated)")
print(f"  Answer: \"{BAD_ANSWER[:80]}...\"\n")

show("Answer Relevancy", evaluate(
    "answer_relevancy",
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

show("RAG Faithfulness", evaluate(
    "rag_faithfulness",
    response=BAD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
))

show("Faithfulness (NLI)", evaluate(
    "faithfulness",
    output=BAD_ANSWER,
    context=" ".join(RETRIEVED_CHUNKS),
))

show("RAG Score (composite)", evaluate(
    "rag_score",
    response=BAD_ANSWER,
    query=CUSTOMER_QUERY,
    contexts=RETRIEVED_CHUNKS,
    reference=REFERENCE_ANSWER,
))


# =========================================================================
# Part 4 — Structured Output (CRM ticket)
# =========================================================================

heading("Part 4: Structured Output — CRM Ticket")

print("  Good ticket (all fields):")
show("  JSON Validation", evaluate(
    "json_validation", output=GOOD_TICKET,
))
show("  Schema Compliance", evaluate(
    "schema_compliance", output=GOOD_TICKET, schema=TICKET_SCHEMA,
))
show("  Field Completeness", evaluate(
    "field_completeness", output=GOOD_TICKET, schema=TICKET_SCHEMA,
))
show("  Structured Score", evaluate(
    "structured_output_score", output=GOOD_TICKET, schema=TICKET_SCHEMA,
))

print()
print("  Bad ticket (missing fields):")
show("  JSON Validation", evaluate(
    "json_validation", output=BAD_TICKET,
))
show("  Schema Compliance", evaluate(
    "schema_compliance", output=BAD_TICKET, schema=TICKET_SCHEMA,
))
show("  Field Completeness", evaluate(
    "field_completeness", output=BAD_TICKET, schema=TICKET_SCHEMA,
))
show("  Structured Score", evaluate(
    "structured_output_score", output=BAD_TICKET, schema=TICKET_SCHEMA,
))


# =========================================================================
# Part 5 — Side-by-side summary
# =========================================================================

heading("Part 5: Side-by-Side Summary")

good_rag = evaluate("rag_score", response=GOOD_ANSWER, query=CUSTOMER_QUERY,
                     contexts=RETRIEVED_CHUNKS, reference=REFERENCE_ANSWER)
bad_rag = evaluate("rag_score", response=BAD_ANSWER, query=CUSTOMER_QUERY,
                    contexts=RETRIEVED_CHUNKS, reference=REFERENCE_ANSWER)
good_struct = evaluate("structured_output_score", output=GOOD_TICKET, schema=TICKET_SCHEMA)
bad_struct = evaluate("structured_output_score", output=BAD_TICKET, schema=TICKET_SCHEMA)

print(f"  {'Metric':<30s}  {'Good':>6s}  {'Bad':>6s}")
print(f"  {'─' * 48}")
print(f"  {'RAG Score (composite)':<30s}  {good_rag.score:6.2f}  {bad_rag.score:6.2f}")
print(f"  {'Structured Output Score':<30s}  {good_struct.score:6.2f}  {bad_struct.score:6.2f}")
print()
print("  Done.")
