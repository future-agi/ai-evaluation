# Evaluation Framework

A scalable evaluation infrastructure for AI systems with support for blocking, non-blocking, and distributed execution modes.

## Quick Start

```python
from fi.evals.framework import Evaluator, ExecutionMode
from fi.evals.framework.evals import (
    SemanticSimilarityEval,
    CoherenceEval,
    FactualConsistencyEval,
)

# Create an evaluator with multiple evaluations
evaluator = Evaluator(
    evaluations=[
        SemanticSimilarityEval(),
        CoherenceEval(),
        FactualConsistencyEval(),
    ],
    mode=ExecutionMode.BLOCKING,
)

# Run evaluations
result = evaluator.run({
    "response": "Paris is the capital of France.",
    "reference": "The capital city of France is Paris.",
    "context": "France is a country in Western Europe.",
})

# Check results
for r in result.results:
    print(f"{r.eval_name}: score={r.value.score:.2f}, passed={r.value.passed}")
```

## Real-World Examples

### 1. Customer Support Chatbot Evaluation

Evaluate chatbot responses for quality, safety, and relevance:

```python
from fi.evals.framework import Evaluator, ExecutionMode
from fi.evals.framework.evals import (
    SemanticSimilarityEval,
    CoherenceEval,
    FactualConsistencyEval,
    ActionSafetyEval,
    custom_eval,
    pattern_match_eval,
)

# Custom evaluation for response politeness
@custom_eval("politeness", required_fields=["response"])
def check_politeness(inputs):
    response = inputs["response"].lower()
    polite_phrases = ["thank you", "please", "happy to help", "let me know", "sorry"]
    score = sum(1 for phrase in polite_phrases if phrase in response) / len(polite_phrases)
    return {"score": min(1.0, score * 2), "passed": score >= 0.2}

# Evaluation to ensure no personal data exposure
no_pii_eval = pattern_match_eval(
    "no_pii_exposure",
    patterns=[
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{16}\b",  # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    ],
    mode="none",  # Pass if NONE of these patterns match
    field="response",
)

# Build the evaluator
evaluator = Evaluator(
    evaluations=[
        SemanticSimilarityEval(threshold=0.6),  # Response matches expected
        CoherenceEval(),  # Response is coherent
        check_politeness,  # Response is polite
        no_pii_eval,  # No PII exposure
    ],
    mode=ExecutionMode.BLOCKING,
)

# Evaluate a chatbot response
result = evaluator.run({
    "response": "Thank you for contacting us! I'd be happy to help you with your order. Your package is scheduled to arrive tomorrow. Please let me know if you have any other questions.",
    "reference": "The customer's order will be delivered tomorrow.",
})

print(f"Overall success rate: {result.success_rate:.0%}")
for r in result.results:
    print(f"  {r.eval_name}: {'✓' if r.value.passed else '✗'} (score: {r.value.score:.2f})")
```

### 2. RAG (Retrieval-Augmented Generation) Evaluation

Evaluate RAG system responses for factual accuracy and relevance:

```python
from fi.evals.framework import Evaluator, ExecutionMode
from fi.evals.framework.evals import (
    FactualConsistencyEval,
    EntailmentEval,
    SemanticSimilarityEval,
    simple_eval,
)

# Custom evaluation for citation coverage
citation_eval = simple_eval(
    "citation_coverage",
    scorer=lambda inputs: (
        len([c for c in inputs.get("citations", []) if c in inputs["response"]]) /
        max(len(inputs.get("citations", [])), 1)
    ),
    threshold=0.5,
    required_fields=["response"],
)

# Build RAG evaluator
rag_evaluator = Evaluator(
    evaluations=[
        FactualConsistencyEval(threshold=0.7),  # Response matches context
        EntailmentEval(threshold=0.6),  # Response is entailed by context
        SemanticSimilarityEval(threshold=0.5),  # Response relevant to query
        citation_eval,  # Citations are included
    ],
    mode=ExecutionMode.BLOCKING,
)

# Evaluate a RAG response
result = rag_evaluator.run({
    "context": """
    The Eiffel Tower was built in 1889 for the World's Fair. It stands 330 meters tall
    and was designed by Gustave Eiffel. Initially criticized by artists and intellectuals,
    it has become a global cultural icon of France.
    """,
    "response": "The Eiffel Tower, designed by Gustave Eiffel, was constructed in 1889 for the World's Fair. Standing at 330 meters, it is now an iconic symbol of France.",
    "reference": "Information about the Eiffel Tower's history and significance.",
    "citations": ["Gustave Eiffel", "1889", "World's Fair"],
})

print(f"RAG Quality Score: {result.success_rate:.0%}")
```

### 3. AI Agent Trajectory Evaluation

Evaluate an AI agent's decision-making and tool usage:

```python
from fi.evals.framework import Evaluator, ExecutionMode
from fi.evals.framework.evals import (
    ToolUseCorrectnessEval,
    TrajectoryEfficiencyEval,
    GoalCompletionEval,
    ActionSafetyEval,
    ReasoningQualityEval,
)

# Build agent evaluator
agent_evaluator = Evaluator(
    evaluations=[
        ToolUseCorrectnessEval(),
        TrajectoryEfficiencyEval(max_steps=10),
        GoalCompletionEval(),
        ActionSafetyEval(),
        ReasoningQualityEval(),
    ],
    mode=ExecutionMode.BLOCKING,
)

# Example agent trajectory
trajectory = [
    {"type": "thought", "name": "thinking", "input": "I need to find the weather in Tokyo to answer the user's question."},
    {"type": "tool_call", "tool": "weather_api", "args": {"city": "Tokyo"}, "result": "Sunny, 24°C"},
    {"type": "thought", "name": "thinking", "input": "I have the weather data. Now I should format a helpful response."},
    {"type": "final_answer", "name": "final_answer", "input": "The weather in Tokyo is sunny with a temperature of 24°C."},
]

result = agent_evaluator.run({
    "trajectory": trajectory,
    "goal": "What is the weather in Tokyo?",
    "available_tools": ["weather_api", "search", "calculator"],
    "expected_answer": "Sunny, 24°C",
})

print(f"Agent Performance: {result.success_rate:.0%}")
for r in result.results:
    print(f"  {r.eval_name}: {r.value.score:.2f}")
```

### 4. Content Moderation Pipeline

Evaluate content for safety and policy compliance:

```python
from fi.evals.framework import Evaluator, ExecutionMode
from fi.evals.framework.evals import (
    ActionSafetyEval,
    pattern_match_eval,
    threshold_eval,
    custom_eval,
)

# Profanity check
profanity_eval = pattern_match_eval(
    "no_profanity",
    patterns=[r"\b(bad_word1|bad_word2)\b"],  # Add actual patterns
    mode="none",
    field="content",
)

# Length check
length_eval = threshold_eval(
    "content_length",
    metric_fn=lambda inputs: len(inputs["content"]),
    min_threshold=10,
    max_threshold=5000,
    required_fields=["content"],
)

# Custom spam detection
@custom_eval("spam_detection", required_fields=["content"])
def detect_spam(inputs):
    content = inputs["content"].lower()
    spam_indicators = [
        "buy now", "click here", "free money", "act now",
        "limited time", "winner", "congratulations"
    ]
    spam_count = sum(1 for indicator in spam_indicators if indicator in content)
    score = 1.0 - min(1.0, spam_count / 3)
    return {"score": score, "passed": spam_count < 2, "spam_indicators_found": spam_count}

# Build moderation pipeline
moderator = Evaluator(
    evaluations=[
        profanity_eval,
        length_eval,
        detect_spam,
    ],
    mode=ExecutionMode.BLOCKING,
)

# Moderate content
result = moderator.run({
    "content": "Check out our new product! It's designed to help you work more efficiently. Contact us for more information.",
})

all_passed = all(r.value.passed for r in result.results)
print(f"Content {'APPROVED' if all_passed else 'REJECTED'}")
```

### 5. Multi-Modal Content Evaluation

Evaluate image captions and visual descriptions:

```python
from fi.evals.framework import Evaluator, ExecutionMode
from fi.evals.framework.evals import (
    ImageTextConsistencyEval,
    CaptionQualityEval,
    VisualQAEval,
)

# Build multi-modal evaluator
mm_evaluator = Evaluator(
    evaluations=[
        ImageTextConsistencyEval(threshold=0.6),
        CaptionQualityEval(threshold=0.7),
        VisualQAEval(threshold=0.6),
    ],
    mode=ExecutionMode.BLOCKING,
)

# Evaluate image caption
result = mm_evaluator.run({
    "image_description": "A golden retriever dog playing fetch in a sunny park with green grass",
    "text": "A happy dog enjoying outdoor activities in the park",
    "caption": "A golden retriever plays fetch on a beautiful sunny day in the park.",
    "question": "What is the dog doing?",
    "answer": "The dog is playing fetch in the park.",
})

print(f"Multi-modal Quality: {result.success_rate:.0%}")
```

### 6. Zero-Latency Production Evaluation

Run evaluations asynchronously without blocking your main application:

```python
from fi.evals.framework import async_evaluator
from fi.evals.framework.evals import (
    SemanticSimilarityEval,
    CoherenceEval,
    FactualConsistencyEval,
)

# Create async evaluator (evaluations run in background)
evaluator = async_evaluator(
    SemanticSimilarityEval(),
    CoherenceEval(),
    FactualConsistencyEval(),
    max_workers=4,
)

def handle_llm_response(response, context, reference):
    """Handle LLM response with zero-latency evaluation."""

    # Start evaluation (returns immediately)
    eval_future = evaluator.run({
        "response": response,
        "context": context,
        "reference": reference,
    })

    # Return response to user immediately
    # Evaluation continues in background
    return response

# Later, check results if needed
def check_evaluation_results(eval_future):
    if eval_future.is_ready:
        batch = eval_future.wait()
        return {r.eval_name: r.value.score for r in batch.results}
    return None

# Don't forget to shutdown when done
# evaluator.shutdown()
```

### 7. Custom Evaluation with Builder Pattern

Create custom evaluations without writing full classes:

```python
from fi.evals.framework import Evaluator, ExecutionMode
from fi.evals.framework.evals import EvalBuilder, comparison_eval

# Using builder pattern
sentiment_eval = (
    EvalBuilder("sentiment_analysis")
    .version("1.0.0")
    .required_fields(["text"])
    .threshold(0.5)
    .description("Analyzes text sentiment")
    .evaluator(lambda inputs: {
        "score": 0.8 if any(w in inputs["text"].lower() for w in ["good", "great", "excellent"]) else 0.3,
        "passed": True,
    })
    .build()
)

# Using comparison factory
json_structure_eval = comparison_eval(
    "json_structure_match",
    comparator=lambda response, expected: (
        1.0 if set(response.keys()) == set(expected.keys()) else
        len(set(response.keys()) & set(expected.keys())) / len(set(expected.keys()))
    ) if isinstance(response, dict) and isinstance(expected, dict) else 0.0,
    source_field="response_json",
    target_field="expected_json",
    threshold=0.8,
)

# Use in evaluator
evaluator = Evaluator(
    evaluations=[sentiment_eval, json_structure_eval],
    mode=ExecutionMode.BLOCKING,
)
```

## Execution Modes

| Mode | Use Case | Latency Impact |
|------|----------|----------------|
| `BLOCKING` | Development, testing, sync workflows | Full evaluation time |
| `NON_BLOCKING` | Production, real-time applications | Zero (async) |
| `DISTRIBUTED` | Batch processing, high throughput | Zero (remote) |

## Available Evaluations

### Semantic Evaluations
- `SemanticSimilarityEval` - Compare text similarity
- `CoherenceEval` - Check text coherence
- `EntailmentEval` - Verify logical entailment
- `ContradictionEval` - Detect contradictions
- `FactualConsistencyEval` - Verify factual accuracy

### Multi-Modal Evaluations
- `ImageTextConsistencyEval` - Image-text alignment
- `CaptionQualityEval` - Caption quality assessment
- `VisualQAEval` - Visual Q&A evaluation
- `ImageSafetyEval` - Image safety checks
- `CrossModalConsistencyEval` - Cross-modal consistency

### Agentic Evaluations
- `ToolUseCorrectnessEval` - Tool usage validation
- `TrajectoryEfficiencyEval` - Agent efficiency
- `GoalCompletionEval` - Task completion
- `ActionSafetyEval` - Safety scanning
- `ReasoningQualityEval` - Reasoning quality

### Custom Evaluation Builders
- `EvalBuilder` - Fluent builder pattern
- `@custom_eval` - Decorator for functions
- `simple_eval()` - Score-based evaluation
- `comparison_eval()` - Compare two fields
- `threshold_eval()` - Min/max thresholds
- `pattern_match_eval()` - Regex patterns

## OpenTelemetry Integration

All evaluations automatically generate span attributes compatible with OpenTelemetry:

```python
from fi.evals.framework import register_current_span, async_evaluator

# Register span for cross-thread enrichment
with tracer.start_as_current_span("llm_call") as span:
    register_current_span()

    response = llm.complete(prompt)
    evaluator.run({"response": response})  # Enriches span automatically

    return response

# Span attributes include:
#   eval.semantic_similarity.score
#   eval.semantic_similarity.passed
#   eval.coherence.score
#   etc.
```
