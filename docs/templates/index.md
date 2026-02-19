# Evaluation Templates

> 60+ pre-built templates for evaluating LLM outputs

---

## Template Categories

| Category | Count | Description |
|----------|-------|-------------|
| [RAG & Context](./categories/rag.md) | 10+ | Groundedness, context adherence, hallucination |
| [Safety](./categories/safety.md) | 8+ | Content moderation, toxicity, PII |
| [Quality](./categories/quality.md) | 6+ | Factual accuracy, helpfulness, completeness |
| [Bias](./categories/bias.md) | 5+ | Gender, racial, age bias detection |
| [Tone](./categories/tone.md) | 5+ | Politeness, formality, sentiment |
| [Format](./categories/format.md) | 8+ | JSON, CSV, code validation |
| [Conversation](./categories/conversation.md) | 2+ | Coherence, resolution |
| [Translation](./categories/translation.md) | 2+ | Accuracy, cultural sensitivity |
| [Function Calling](./categories/function-calling.md) | 2+ | Tool use validation |
| [Audio](./categories/audio.md) | 2+ | Transcription, quality |

---

## Most Popular Templates

### RAG Evaluation

| Template | Input Fields | Output |
|----------|--------------|--------|
| `groundedness` | context, output | GROUNDED / NOT_GROUNDED |
| `context_adherence` | context, output | Score 0-1 |
| `completeness` | input, output | Score 0-1 |
| `chunk_utilization` | context[], output | Score 0-1 |

### Safety

| Template | Input Fields | Output |
|----------|--------------|--------|
| `content_moderation` | text | SAFE / UNSAFE |
| `toxicity` | text | SAFE / TOXIC |
| `pii` | text | DETECTED / NOT_DETECTED |
| `prompt_injection` | input | DETECTED / NOT_DETECTED |

### Quality

| Template | Input Fields | Output |
|----------|--------------|--------|
| `factual_accuracy` | input, output, context | Score 0-1 |
| `is_helpful` | input, output | YES / NO |
| `is_good_summary` | input, output | YES / NO |
| `task_completion` | input, output | YES / NO |

---

## Using Templates

### Python

```python
from fi.evals import Evaluator

evaluator = Evaluator()

# Single template
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={
        "context": "The Eiffel Tower is 324 meters tall.",
        "output": "The Eiffel Tower stands at 324 meters."
    },
    model_name="turing_flash"
)
```

### TypeScript

```typescript
import { Evaluator } from "@future-agi/ai-evaluation";

const evaluator = new Evaluator();

const result = await evaluator.evaluate(
  "groundedness",
  {
    context: "The Eiffel Tower is 324 meters tall.",
    output: "The Eiffel Tower stands at 324 meters."
  },
  { modelName: "turing_flash" }
);
```

### CLI

```yaml
# fi-evaluation.yaml
evaluations:
  - name: "rag_eval"
    template: "groundedness"
    data: "./data/tests.json"
```

---

## Template Input Reference

### Common Input Fields

| Field | Description | Used By |
|-------|-------------|---------|
| `input` | User query/prompt | Quality, tone templates |
| `output` | LLM response | Most templates |
| `context` | Reference context | RAG templates |
| `text` | Text to analyze | Safety, format templates |
| `expected` | Expected output | Comparison templates |

### Input Types

```python
# String input
inputs = {"text": "Hello world"}

# Array input (for chunks)
inputs = {"context": ["Chunk 1", "Chunk 2", "Chunk 3"]}

# Multiple fields
inputs = {
    "input": "User question",
    "output": "LLM response",
    "context": "Reference text"
}
```

---

## Template Output Reference

### Output Types

| Type | Example | Templates |
|------|---------|-----------|
| Binary | `GROUNDED` / `NOT_GROUNDED` | groundedness, toxicity |
| Boolean | `true` / `false` | is_json, is_code |
| Score | `0.85` | factual_accuracy, completeness |
| Category | `FORMAL` / `INFORMAL` | tone |

### Result Object

```python
result.eval_results[0].output     # The evaluation result
result.eval_results[0].reason     # Explanation
result.eval_results[0].runtime    # Execution time (seconds)
result.eval_results[0].name       # Template name
result.eval_results[0].eval_id    # Unique ID
```

---

## Choosing the Right Template

### By Use Case

| Use Case | Recommended Templates |
|----------|----------------------|
| RAG quality | groundedness, context_adherence, chunk_utilization |
| Chatbot safety | content_moderation, toxicity, pii |
| Response quality | is_helpful, completeness, factual_accuracy |
| Output format | is_json, is_code, is_csv |
| Bias detection | no_gender_bias, no_racial_bias, no_age_bias |

### By Model

| Model | Best For |
|-------|----------|
| `turing_flash` | General evaluations, speed-critical |
| `turing_pro` | Complex evaluations, accuracy-critical |
| `protect_flash` | Safety evaluations, speed-critical |
| `protect_pro` | Detailed safety analysis |

---

## Custom Templates

Create your own evaluation templates:

```python
# Coming soon
result = evaluator.evaluate(
    eval_templates="custom",
    inputs={...},
    custom_criteria="Check if response mentions pricing",
    model_name="turing_flash"
)
```

See [Custom Templates Guide](./custom-templates.md) for more details.

---

## See Also

- [Getting Started](../getting-started/index.md)
- [SDK Reference](../sdks/index.md)
- [Tutorials](../tutorials/index.md)
