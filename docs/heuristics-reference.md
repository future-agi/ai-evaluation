# Heuristic Metrics Reference

> Local, fast evaluation metrics that run without API calls

---

## Overview

Heuristic metrics provide fast, deterministic evaluations that run locally without API calls. They are ideal for:

- **CI/CD pipelines** - Fast feedback without network latency
- **Cost optimization** - No API usage costs
- **Batch processing** - Process thousands of items quickly
- **Format validation** - Validate outputs before LLM evaluation

---

## Installation

Heuristic metrics are included in the Python SDK:

```bash
pip install ai-evaluation
```

---

## String Metrics

Fast string-based validation and matching.

### Regex

Check if text matches a regular expression pattern.

```python
from fi.evals.metrics.heuristics import Regex

metric = Regex(pattern=r"^\d{3}-\d{2}-\d{4}$")  # SSN pattern
result = metric.run(TextMetricInput(text="123-45-6789"))
# result.success = True
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pattern` | `str` | Yes | Regular expression pattern |
| `flags` | `int` | No | Regex flags (e.g., `re.IGNORECASE`) |

---

### Contains

Check if text contains a substring.

```python
from fi.evals.metrics.heuristics import Contains

metric = Contains(substring="hello")
result = metric.run(TextMetricInput(text="Hello world"))
# result.success = True (case-insensitive by default)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `substring` | `str` | Yes | String to find |
| `case_sensitive` | `bool` | No | Case sensitivity (default: `False`) |

---

### ContainsAll

Check if text contains all specified substrings.

```python
from fi.evals.metrics.heuristics import ContainsAll

metric = ContainsAll(substrings=["hello", "world"])
result = metric.run(TextMetricInput(text="Hello beautiful world"))
# result.success = True
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `substrings` | `List[str]` | Yes | List of required substrings |
| `case_sensitive` | `bool` | No | Case sensitivity (default: `False`) |

---

### ContainsAny

Check if text contains at least one of the specified substrings.

```python
from fi.evals.metrics.heuristics import ContainsAny

metric = ContainsAny(substrings=["error", "warning", "failure"])
result = metric.run(TextMetricInput(text="Operation completed with warning"))
# result.success = True
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `substrings` | `List[str]` | Yes | List of substrings (any must match) |
| `case_sensitive` | `bool` | No | Case sensitivity (default: `False`) |

---

### ContainsNone

Check that text contains none of the specified substrings.

```python
from fi.evals.metrics.heuristics import ContainsNone

metric = ContainsNone(substrings=["error", "exception", "failed"])
result = metric.run(TextMetricInput(text="Operation completed successfully"))
# result.success = True
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `substrings` | `List[str]` | Yes | List of forbidden substrings |
| `case_sensitive` | `bool` | No | Case sensitivity (default: `False`) |

---

### OneLine

Check if text is a single line (no newlines).

```python
from fi.evals.metrics.heuristics import OneLine

metric = OneLine()
result = metric.run(TextMetricInput(text="This is a single line"))
# result.success = True
```

---

### IsEmail

Validate email format.

```python
from fi.evals.metrics.heuristics import IsEmail

metric = IsEmail()
result = metric.run(TextMetricInput(text="user@example.com"))
# result.success = True
```

---

### Equals

Check exact string equality.

```python
from fi.evals.metrics.heuristics import Equals

metric = Equals(expected="success")
result = metric.run(TextMetricInput(text="SUCCESS"))
# result.success = True (case-insensitive by default)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `expected` | `str` | Yes | Expected string value |
| `case_sensitive` | `bool` | No | Case sensitivity (default: `False`) |

---

### StartsWith

Check if text starts with a prefix.

```python
from fi.evals.metrics.heuristics import StartsWith

metric = StartsWith(prefix="Dear")
result = metric.run(TextMetricInput(text="Dear Customer, we are pleased..."))
# result.success = True
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prefix` | `str` | Yes | Required prefix |
| `case_sensitive` | `bool` | No | Case sensitivity (default: `False`) |

---

### EndsWith

Check if text ends with a suffix.

```python
from fi.evals.metrics.heuristics import EndsWith

metric = EndsWith(suffix="Regards")
result = metric.run(TextMetricInput(text="Thank you. Best Regards"))
# result.success = True
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `suffix` | `str` | Yes | Required suffix |
| `case_sensitive` | `bool` | No | Case sensitivity (default: `False`) |

---

### LengthLessThan

Check if text length is less than a threshold.

```python
from fi.evals.metrics.heuristics import LengthLessThan

metric = LengthLessThan(max_length=100)
result = metric.run(TextMetricInput(text="Short message"))
# result.success = True
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `max_length` | `int` | Yes | Maximum allowed length |

---

### LengthGreaterThan

Check if text length is greater than a threshold.

```python
from fi.evals.metrics.heuristics import LengthGreaterThan

metric = LengthGreaterThan(min_length=10)
result = metric.run(TextMetricInput(text="This is a longer message"))
# result.success = True
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `min_length` | `int` | Yes | Minimum required length |

---

### LengthBetween

Check if text length is within a range.

```python
from fi.evals.metrics.heuristics import LengthBetween

metric = LengthBetween(min_length=10, max_length=100)
result = metric.run(TextMetricInput(text="This message is just right"))
# result.success = True
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `min_length` | `int` | Yes | Minimum length |
| `max_length` | `int` | Yes | Maximum length |

---

## JSON Metrics

Validate and check JSON structures.

### IsJson

Validate if text is valid JSON.

```python
from fi.evals.metrics.heuristics import IsJson

metric = IsJson()
result = metric.run(TextMetricInput(text='{"name": "John", "age": 30}'))
# result.success = True
```

---

### ContainsJson

Check if text contains valid JSON (even with surrounding text).

```python
from fi.evals.metrics.heuristics import ContainsJson

metric = ContainsJson()
result = metric.run(TextMetricInput(text='Response: {"status": "ok"} End.'))
# result.success = True
```

---

### JsonSchema

Validate JSON against a JSON Schema.

```python
from fi.evals.metrics.heuristics import JsonSchema

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
    },
    "required": ["name", "age"]
}

metric = JsonSchema(schema=schema)
result = metric.run(JsonMetricInput(
    generated='{"name": "John", "age": 30}'
))
# result.success = True
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `schema` | `dict` | Yes | JSON Schema definition |

---

## Similarity Metrics

Measure text similarity using various algorithms.

### LevenshteinSimilarity

Calculate normalized Levenshtein (edit distance) similarity.

```python
from fi.evals.metrics.heuristics import LevenshteinSimilarity

metric = LevenshteinSimilarity(threshold=0.8)
result = metric.run(TextMetricInput(
    text="hello world",
    expected_text="hello worlb"
))
# result.score = 0.91 (normalized similarity)
# result.success = True (score >= threshold)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `threshold` | `float` | No | Minimum similarity (0-1) for success |

---

### BLEUScore

Calculate BLEU score for translation/generation quality.

```python
from fi.evals.metrics.heuristics import BLEUScore

metric = BLEUScore(threshold=0.5)
result = metric.run(TextMetricInput(
    text="The cat sat on the mat",
    expected_text="The cat is sitting on the mat"
))
# result.score = BLEU score
# result.success = True if score >= threshold
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `threshold` | `float` | No | Minimum BLEU score for success |

---

### ROUGEScore

Calculate ROUGE score for summarization quality.

```python
from fi.evals.metrics.heuristics import ROUGEScore

metric = ROUGEScore(
    rouge_type="rouge-l",
    threshold=0.6
)
result = metric.run(TextMetricInput(
    text="AI is transforming industries",
    expected_text="Artificial intelligence is changing many industries"
))
# result.score = ROUGE score
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `rouge_type` | `str` | No | Type: `rouge-1`, `rouge-2`, `rouge-l` (default) |
| `threshold` | `float` | No | Minimum score for success |

---

### NumericSimilarity

Check if numeric values are within a percentage tolerance.

```python
from fi.evals.metrics.heuristics import NumericSimilarity

metric = NumericSimilarity(tolerance=0.1)  # 10% tolerance
result = metric.run(TextMetricInput(
    text="105",
    expected_text="100"
))
# result.success = True (5% difference is within 10% tolerance)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tolerance` | `float` | No | Allowed percentage difference (0-1) |

---

### RecallScore

Calculate recall (how much of expected content is in generated).

```python
from fi.evals.metrics.heuristics import RecallScore

metric = RecallScore(threshold=0.7)
result = metric.run(TextMetricInput(
    text="Paris is the capital of France",
    expected_text="Paris is the beautiful capital city of France in Europe"
))
# result.score = recall value
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `threshold` | `float` | No | Minimum recall for success |

---

### EmbeddingSimilarity

Calculate semantic similarity using embeddings.

```python
from fi.evals.metrics.heuristics import EmbeddingSimilarity

metric = EmbeddingSimilarity(
    model="sentence-transformers/all-MiniLM-L6-v2",
    threshold=0.7
)
result = metric.run(TextMetricInput(
    text="The weather is nice today",
    expected_text="It's a beautiful day outside"
))
# result.score = cosine similarity
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | No | Embedding model name |
| `threshold` | `float` | No | Minimum similarity for success |

---

## Input Types

### TextMetricInput

Used for most string-based metrics.

```python
from fi.evals.types import TextMetricInput

input = TextMetricInput(
    text="Generated text",
    expected_text="Expected text"  # For comparison metrics
)
```

### JsonMetricInput

Used for JSON-based metrics.

```python
from fi.evals.types import JsonMetricInput

input = JsonMetricInput(
    generated='{"key": "value"}',
    expected='{"key": "expected_value"}'  # For comparison
)
```

---

## Combining with LLM Evaluations

Use heuristics for fast checks, then LLM for deeper analysis:

```python
from fi.evals import Evaluator
from fi.evals.metrics.heuristics import IsJson, ContainsAll

evaluator = Evaluator()

def evaluate_response(response: str, context: str):
    # Fast heuristic checks first
    json_check = IsJson().run(TextMetricInput(text=response))
    if not json_check.success:
        return {"status": "fail", "reason": "Invalid JSON"}

    keywords = ContainsAll(substrings=["status", "result"])
    keyword_check = keywords.run(TextMetricInput(text=response))
    if not keyword_check.success:
        return {"status": "fail", "reason": "Missing required fields"}

    # LLM evaluation for deeper analysis
    result = evaluator.evaluate(
        eval_templates="groundedness",
        inputs={"context": context, "output": response},
        model_name="turing_flash"
    )

    return {
        "status": "pass" if result.eval_results[0].output == "GROUNDED" else "fail",
        "details": result.eval_results[0].reason
    }
```

---

## Use Cases

### CI/CD Pipeline Validation

```python
from fi.evals.metrics.heuristics import IsJson, JsonSchema, LengthBetween

def validate_api_response(response: str) -> bool:
    # Check JSON validity
    if not IsJson().run(TextMetricInput(text=response)).success:
        return False

    # Check schema
    schema = {"type": "object", "required": ["id", "status"]}
    if not JsonSchema(schema=schema).run(JsonMetricInput(generated=response)).success:
        return False

    # Check length
    if not LengthBetween(min_length=10, max_length=10000).run(TextMetricInput(text=response)).success:
        return False

    return True
```

### Translation Quality Check

```python
from fi.evals.metrics.heuristics import BLEUScore, LevenshteinSimilarity

def check_translation_quality(translation: str, reference: str) -> dict:
    bleu = BLEUScore(threshold=0.5).run(TextMetricInput(
        text=translation,
        expected_text=reference
    ))

    leven = LevenshteinSimilarity(threshold=0.6).run(TextMetricInput(
        text=translation,
        expected_text=reference
    ))

    return {
        "bleu_score": bleu.score,
        "edit_similarity": leven.score,
        "quality": "good" if bleu.success and leven.success else "needs_review"
    }
```

### Content Filtering

```python
from fi.evals.metrics.heuristics import ContainsNone, Regex

def filter_content(text: str) -> dict:
    # Check for forbidden content
    forbidden_check = ContainsNone(substrings=[
        "password", "api_key", "secret", "token"
    ]).run(TextMetricInput(text=text))

    # Check for valid format
    format_check = Regex(pattern=r"^[a-zA-Z0-9\s.,!?]+$").run(TextMetricInput(text=text))

    return {
        "safe": forbidden_check.success,
        "valid_format": format_check.success
    }
```

---

## Performance

Heuristic metrics are optimized for speed:

| Metric | Typical Speed | Use Case |
|--------|---------------|----------|
| `Contains`, `Regex` | < 1ms | Basic validation |
| `IsJson`, `JsonSchema` | 1-5ms | JSON validation |
| `LevenshteinSimilarity` | 1-10ms | String comparison |
| `BLEUScore`, `ROUGEScore` | 5-50ms | NLP metrics |
| `EmbeddingSimilarity` | 50-500ms | Semantic comparison |

---

## Phase 1 Metrics (New)

Advanced evaluation metrics for agents, hallucination detection, and function calling.

### Function Calling Metrics

Evaluate LLM function/tool calling accuracy.

```python
from fi.evals.metrics.function_calling import (
    FunctionCallInput, FunctionCall, FunctionCallAccuracy
)

metric = FunctionCallAccuracy()
result = metric.compute_one(FunctionCallInput(
    response=FunctionCall(name="get_weather", arguments={"city": "NYC"}),
    expected_response=FunctionCall(name="get_weather", arguments={"city": "NYC"})
))
# result["output"] = 1.0
```

| Metric | Speed | Description |
|--------|-------|-------------|
| `FunctionNameMatch` | < 1ms | Check function name |
| `ParameterValidation` | < 5ms | Validate against schema |
| `FunctionCallAccuracy` | < 10ms | Comprehensive scoring |
| `FunctionCallAST` | < 5ms | AST-based exact match |

### Hallucination Detection Metrics

Detect unsupported claims in LLM responses.

```python
from fi.evals.metrics.hallucination import (
    HallucinationInput, Faithfulness, HallucinationScore
)

metric = Faithfulness()
result = metric.compute_one(HallucinationInput(
    response="Paris is the capital of France.",
    context="Paris is the capital city of France."
))
# result["output"] = 1.0 (fully faithful)
```

| Metric | Speed | Description |
|--------|-------|-------------|
| `Faithfulness` | 10-50ms | Claim support by context |
| `ClaimSupport` | 10-50ms | Granular claim analysis |
| `ContradictionDetection` | 10-50ms | Detect contradictions |
| `HallucinationScore` | 20-100ms | Comprehensive scoring |

### Agent Evaluation Metrics

Evaluate multi-step agent trajectories.

```python
from fi.evals.metrics.agents import (
    AgentTrajectoryInput, AgentStep, ToolCall, TaskDefinition, TrajectoryScore
)

metric = TrajectoryScore()
result = metric.compute_one(AgentTrajectoryInput(
    trajectory=[
        AgentStep(step_number=1, tool_calls=[ToolCall(name="search", success=True)]),
        AgentStep(step_number=2, is_final=True)
    ],
    task=TaskDefinition(description="Complete the task")
))
```

| Metric | Speed | Description |
|--------|-------|-------------|
| `TaskCompletion` | < 10ms | Task completion check |
| `StepEfficiency` | < 10ms | Trajectory efficiency |
| `ToolSelectionAccuracy` | < 10ms | Tool usage validation |
| `TrajectoryScore` | < 20ms | Composite scoring |
| `GoalProgress` | < 10ms | Incremental progress |

---

## See Also

- [Python SDK](./python-sdk.md)
- [Templates Reference](./templates-reference.md)
- [Getting Started](./getting-started.md)
- [Phase 1 Implementation](./implementations/phase-1/overview.md)
