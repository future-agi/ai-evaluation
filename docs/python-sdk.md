# Python SDK Reference

> Complete reference for the AI Evaluation Python SDK

---

## Table of Contents

- [Installation](#installation)
- [Evaluator Class](#evaluator-class)
- [Methods](#methods)
- [Types](#types)
- [Templates](#templates)
- [Advanced Usage](#advanced-usage)
- [Error Handling](#error-handling)

---

## Installation

```bash
pip install ai-evaluation
```

**Requirements:**
- Python 3.10+
- pip or uv

---

## Evaluator Class

The main class for running evaluations.

### Constructor

```python
from fi.evals import Evaluator

evaluator = Evaluator(
    fi_api_key: str = None,           # API key (or use FI_API_KEY env var)
    fi_secret_key: str = None,        # Secret key (or use FI_SECRET_KEY env var)
    fi_base_url: str = None,          # Base URL (default: https://api.futureagi.com)
    timeout: int = 200,               # Default timeout in seconds
    max_workers: int = 8,             # Max parallel workers
    max_queue_bound: int = 5000,      # Max queue size
    langfuse_secret_key: str = None,  # Langfuse integration
    langfuse_public_key: str = None,  # Langfuse integration
    langfuse_host: str = None,        # Langfuse integration
)
```

### Example

```python
from fi.evals import Evaluator

# Using environment variables (recommended)
evaluator = Evaluator()

# Or with explicit credentials
evaluator = Evaluator(
    fi_api_key="your_api_key",
    fi_secret_key="your_secret_key"
)
```

---

## Methods

### evaluate()

Run a single evaluation.

```python
result = evaluator.evaluate(
    eval_templates: str | EvalTemplate,  # Template name or class
    inputs: Dict[str, Any],              # Input data for evaluation
    timeout: int = None,                 # Optional timeout override
    model_name: str = None,              # Model to use (e.g., "turing_flash")
    custom_eval_name: str = None,        # Custom name for tracing
    trace_eval: bool = False,            # Enable OpenTelemetry tracing
    platform: str = None,                # Platform for configuration
    is_async: bool = False,              # Async evaluation
    error_localizer: bool = False,       # Enable error localization
) -> BatchRunResult
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `eval_templates` | `str` or `EvalTemplate` | Yes | Template name (e.g., "groundedness") or template class |
| `inputs` | `Dict[str, Any]` | Yes | Input data matching template requirements |
| `model_name` | `str` | No | Model to use: `turing_flash`, `turing_pro`, `protect_flash`, `protect_pro` |
| `timeout` | `int` | No | Timeout in seconds (default: 200) |
| `custom_eval_name` | `str` | No | Custom name for tracing/tracking |
| `trace_eval` | `bool` | No | Enable OpenTelemetry tracing |
| `is_async` | `bool` | No | Run evaluation asynchronously |

**Example:**

```python
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={
        "context": "The sky is blue due to Rayleigh scattering.",
        "output": "The sky appears blue because of how light scatters in the atmosphere."
    },
    model_name="turing_flash"
)

print(result.eval_results[0].output)   # e.g., "GROUNDED"
print(result.eval_results[0].reason)   # Explanation
print(result.eval_results[0].runtime)  # Execution time
```

---

### list_evaluations()

List all available evaluation templates.

```python
templates = evaluator.list_evaluations() -> List[Dict]
```

**Returns:** List of template dictionaries with:
- `name`: Template name
- `description`: Template description
- `eval_tags`: Category tags
- `config`: Required inputs and outputs

**Example:**

```python
templates = evaluator.list_evaluations()

for template in templates:
    print(f"{template['name']}: {template['description']}")
```

---

### get_eval_result()

Retrieve results for an async evaluation by ID.

```python
result = evaluator.get_eval_result(
    eval_id: str  # Evaluation ID from async evaluation
) -> Dict
```

**Example:**

```python
# Start async evaluation
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={"context": "...", "output": "..."},
    is_async=True
)

# Get result later
eval_id = result.eval_results[0].eval_id
final_result = evaluator.get_eval_result(eval_id)
```

---

### evaluate_pipeline()

Evaluate a pipeline version with multiple test cases.

```python
result = evaluator.evaluate_pipeline(
    project_name: str,              # Project name
    version: str,                   # Pipeline version
    eval_data: List[Dict[str, Any]] # List of evaluation data
) -> Dict
```

**Example:**

```python
result = evaluator.evaluate_pipeline(
    project_name="my-chatbot",
    version="v1.0.0",
    eval_data=[
        {
            "eval_name": "groundedness",
            "inputs": {"context": "...", "output": "..."}
        },
        {
            "eval_name": "is_helpful",
            "inputs": {"input": "...", "output": "..."}
        }
    ]
)
```

---

### get_pipeline_results()

Get results for a pipeline evaluation.

```python
results = evaluator.get_pipeline_results(
    project_name: str,      # Project name
    versions: List[str]     # List of versions to retrieve
) -> Dict
```

---

## Types

### BatchRunResult

Container for evaluation results.

```python
from fi.evals.types import BatchRunResult

class BatchRunResult:
    eval_results: List[EvalResult]  # List of individual results
```

### EvalResult

Individual evaluation result.

```python
from fi.evals.types import EvalResult

class EvalResult:
    name: str           # Evaluation template name
    output: Any         # Evaluation output (varies by template)
    reason: str         # Explanation of the evaluation
    runtime: float      # Execution time in seconds
    output_type: str    # Type of output
    eval_id: str        # Unique evaluation ID
```

**Example:**

```python
result = evaluator.evaluate(...)

for eval_result in result.eval_results:
    print(f"Template: {eval_result.name}")
    print(f"Output: {eval_result.output}")
    print(f"Reason: {eval_result.reason}")
    print(f"Runtime: {eval_result.runtime}s")
```

---

## Templates

### Using Template Names

```python
# Use template name as string
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={"context": "...", "output": "..."}
)
```

### Common Templates

| Template | Category | Required Inputs |
|----------|----------|-----------------|
| `groundedness` | RAG | `context`, `output` |
| `context_adherence` | RAG | `context`, `output` |
| `factual_accuracy` | Quality | `input`, `output`, `context` |
| `is_helpful` | Tone | `input`, `output` |
| `is_polite` | Tone | `input` |
| `tone` | Tone | `input` |
| `content_moderation` | Safety | `text` |
| `toxicity` | Safety | `text` |
| `pii` | Safety | `text` |
| `prompt_injection` | Safety | `input` |
| `is_json` | Format | `text` |
| `is_good_summary` | Quality | `input`, `output` |

See [Templates Reference](./templates-reference.md) for all 60+ templates.

---

## Advanced Usage

### Batch Evaluation

Process multiple test cases efficiently:

```python
from fi.evals import Evaluator

evaluator = Evaluator()

test_cases = [
    {"context": "Context 1", "output": "Response 1"},
    {"context": "Context 2", "output": "Response 2"},
    {"context": "Context 3", "output": "Response 3"},
]

results = []
for case in test_cases:
    result = evaluator.evaluate(
        eval_templates="groundedness",
        inputs=case,
        model_name="turing_flash"
    )
    results.append(result.eval_results[0])

# Process results
for i, r in enumerate(results):
    print(f"Case {i+1}: {r.output}")
```

### Multiple Templates

Run multiple evaluations on the same data:

```python
templates = ["groundedness", "completeness", "is_helpful"]

for template in templates:
    result = evaluator.evaluate(
        eval_templates=template,
        inputs={
            "context": "The Eiffel Tower is 324 meters tall.",
            "input": "How tall is the Eiffel Tower?",
            "output": "The Eiffel Tower is 324 meters tall."
        },
        model_name="turing_flash"
    )
    print(f"{template}: {result.eval_results[0].output}")
```

### OpenTelemetry Tracing

Enable tracing for observability:

```python
from opentelemetry import trace

# Requires fi-instrumentation-otel package
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={"context": "...", "output": "..."},
    trace_eval=True,
    custom_eval_name="my_groundedness_check"
)
```

### Langfuse Integration

Connect to Langfuse for tracing:

```python
evaluator = Evaluator(
    langfuse_secret_key="sk-lf-...",
    langfuse_public_key="pk-lf-...",
    langfuse_host="https://cloud.langfuse.com"
)

result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={"context": "...", "output": "..."},
    platform="langfuse",
    custom_eval_name="my_eval"
)
```

### Custom Timeout and Workers

```python
evaluator = Evaluator(
    timeout=300,        # 5 minute timeout
    max_workers=16      # More parallel workers
)
```

---

## Error Handling

### Common Exceptions

```python
from fi.utils.errors import InvalidAuthError

try:
    result = evaluator.evaluate(...)
except InvalidAuthError:
    print("Invalid API credentials")
except Exception as e:
    print(f"Evaluation failed: {e}")
```

### Validation Errors

```python
# Missing required input
try:
    result = evaluator.evaluate(
        eval_templates="groundedness",
        inputs={"output": "..."}  # Missing 'context'
    )
except Exception as e:
    print(f"Validation error: {e}")
```

### Timeout Handling

```python
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={"context": "...", "output": "..."},
    timeout=60  # 60 second timeout
)

# Check for timeout failures in logs
# Failed evaluations are logged automatically
```

---

## Convenience Functions

### Quick Evaluation

```python
from fi.evals import evaluate

# One-liner evaluation
result = evaluate(
    eval_templates="tone",
    inputs={"input": "Hello, how are you?"}
)
```

### List Templates

```python
from fi.evals import list_evaluations

templates = list_evaluations()
for t in templates:
    print(t['name'])
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FI_API_KEY` | API key | Required |
| `FI_SECRET_KEY` | Secret key | Required |
| `FI_BASE_URL` | API base URL | `https://api.futureagi.com` |
| `LANGFUSE_SECRET_KEY` | Langfuse secret | Optional |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | Optional |
| `LANGFUSE_HOST` | Langfuse host | Optional |

---

## Heuristic Metrics

For fast, local evaluations without API calls, use heuristic metrics:

```python
from fi.evals.metrics.heuristics import (
    IsJson, JsonSchema, Contains, ContainsAll,
    Regex, BLEUScore, ROUGEScore, LevenshteinSimilarity
)
from fi.evals.types import TextMetricInput

# JSON validation
json_metric = IsJson()
result = json_metric.run(TextMetricInput(text='{"key": "value"}'))
print(result.success)  # True

# String matching
contains = Contains(substring="error")
result = contains.run(TextMetricInput(text="No errors found"))
print(result.success)  # False

# BLEU score
bleu = BLEUScore(threshold=0.5)
result = bleu.run(TextMetricInput(
    text="The cat sat on the mat",
    expected_text="The cat is sitting on the mat"
))
print(f"BLEU Score: {result.score}")
```

See [Heuristic Metrics Reference](./heuristics-reference.md) for all available metrics.

---

## See Also

- [Getting Started](./getting-started.md)
- [Templates Reference](./templates-reference.md)
- [Heuristic Metrics](./heuristics-reference.md)
- [CLI Guide](./cli-guide.md)
- [TypeScript SDK](./typescript-sdk.md)
