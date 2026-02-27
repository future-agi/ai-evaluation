# Core Concepts

> Understand the fundamentals of AI Evaluation

---

## What is LLM Evaluation?

LLM Evaluation is the process of measuring the quality, safety, and reliability of Large Language Model outputs. Instead of manual review, AI Evaluation uses automated metrics and LLM-as-judge approaches to assess responses at scale.

---

## Key Concepts

### [Evaluations](./evaluations.md)

An **evaluation** is a single assessment of an LLM output against specific criteria.

```python
result = evaluator.evaluate(
    eval_templates="groundedness",  # What to evaluate
    inputs={...},                   # Data to evaluate
    model_name="turing_flash"       # Judge model
)
```

### [Templates](./templates.md)

**Templates** are pre-built evaluation criteria. AI Evaluation provides 60+ templates across categories:

| Category | Examples | Use Case |
|----------|----------|----------|
| RAG | groundedness, context_adherence | Check if responses use provided context |
| Safety | toxicity, pii, prompt_injection | Detect harmful content |
| Quality | factual_accuracy, is_helpful | Measure response quality |
| Bias | no_gender_bias, no_racial_bias | Detect biased outputs |
| Format | is_json, is_code | Validate output structure |

### [Models](./models.md)

**Models** are the LLM judges that evaluate outputs:

| Model | Speed | Use Case |
|-------|-------|----------|
| `turing_flash` | Fast | General evaluations |
| `turing_pro` | Moderate | Complex evaluations |
| `protect_flash` | Fast | Safety evaluations |
| `protect_pro` | Moderate | Detailed safety analysis |

### [Inputs & Outputs](./inputs-outputs.md)

Each template requires specific **inputs** and produces structured **outputs**:

```python
# Input
inputs = {
    "context": "Paris is the capital of France.",
    "output": "The capital of France is Paris."
}

# Output
result.eval_results[0].output   # "GROUNDED"
result.eval_results[0].reason   # "The response accurately..."
result.eval_results[0].runtime  # 1.234
```

---

## Evaluation Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Inputs    │ ──► │   Template   │ ──► │   Result    │
│             │     │   + Model    │     │             │
│ - context   │     │              │     │ - output    │
│ - output    │     │ groundedness │     │ - reason    │
│ - query     │     │ turing_flash │     │ - runtime   │
└─────────────┘     └──────────────┘     └─────────────┘
```

---

## Common Patterns

### Single Evaluation

Evaluate one response:

```python
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={"context": "...", "output": "..."},
    model_name="turing_flash"
)
```

### Batch Evaluation

Evaluate multiple responses:

```python
for test_case in test_cases:
    result = evaluator.evaluate(
        eval_templates="groundedness",
        inputs=test_case,
        model_name="turing_flash"
    )
```

### Multi-Template Evaluation

Apply multiple templates to same data:

```python
templates = ["groundedness", "completeness", "is_helpful"]
for template in templates:
    result = evaluator.evaluate(
        eval_templates=template,
        inputs=inputs,
        model_name="turing_flash"
    )
```

### CI/CD Evaluation

Run evaluations in your pipeline:

```yaml
# fi-evaluation.yaml
evaluations:
  - name: "quality_gate"
    templates: ["groundedness", "factual_accuracy"]
    data: "./data/tests.json"

assertions:
  - template: "groundedness"
    condition: "score >= 0.8"
    on_fail: "error"
```

---

## Best Practices

### 1. Choose the Right Template

| If you want to check... | Use template |
|------------------------|--------------|
| Response uses context | `groundedness` |
| Response is safe | `content_moderation` |
| Response is helpful | `is_helpful` |
| Output is valid JSON | `is_json` |
| No PII in output | `pii` |

### 2. Use Appropriate Models

- **Safety checks**: Use `protect_flash` or `protect_pro`
- **Quality checks**: Use `turing_flash` or `turing_pro`
- **Speed-critical**: Use `*_flash` variants
- **Accuracy-critical**: Use `*_pro` variants

### 3. Handle Errors

```python
try:
    result = evaluator.evaluate(...)
except Exception as e:
    print(f"Evaluation failed: {e}")
```

### 4. Monitor in Production

- Log evaluation results
- Set up alerts for low scores
- Track evaluation latency
- Review failed evaluations

---

## Next Steps

- [Evaluations Deep Dive](./evaluations.md)
- [Templates Guide](./templates.md)
- [Models Reference](./models.md)
- [Best Practices](./best-practices.md)
