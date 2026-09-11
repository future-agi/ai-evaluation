# Evaluation Framework

A scalable evaluation infrastructure for AI systems with support for blocking, non-blocking, and distributed execution modes.

## Quick Start

```python
from fi.evals import FrameworkEvaluator, ExecutionMode
from fi.evals.framework.evals import CoherenceEval, ActionSafetyEval

# Create an evaluator with multiple evaluations
evaluator = FrameworkEvaluator(
    evaluations=[
        CoherenceEval(),
        ActionSafetyEval(),
    ],
    mode=ExecutionMode.BLOCKING,
)

# Run evaluations
result = evaluator.run({
    "response": "Paris is the capital of France. It is in Western Europe.",
    "trajectory": [
        {"type": "tool_call", "tool": "search", "args": "Paris facts"},
    ],
})

# Check results
for r in result.results:
    print(f"{r.eval_name}: score={r.value.score:.2f}, passed={r.value.passed}")
```

## Execution Modes

| Mode | Use Case | Latency Impact |
|------|----------|----------------|
| `BLOCKING` | Development, testing, sync workflows | Full evaluation time |
| `NON_BLOCKING` | Production, real-time applications | Zero (async) |
| `DISTRIBUTED` | Batch processing, high throughput | Zero (remote) |

## Available Evaluations

### Semantic Evaluations
- `CoherenceEval` - Check text coherence

### Agentic Evaluations
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
#   eval.coherence.score
#   eval.coherence.passed
#   eval.action_safety.score
#   etc.
```
