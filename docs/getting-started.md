# Getting Started with AI Evaluation

> Get up and running with AI Evaluation in under 5 minutes

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Authentication](#authentication)
- [Your First Evaluation](#your-first-evaluation)
- [CLI Quick Start](#cli-quick-start)
- [Next Steps](#next-steps)

---

## Prerequisites

### Python
- Python 3.10 or higher
- pip or poetry package manager

### TypeScript/JavaScript
- Node.js 18.0.0 or higher
- npm, yarn, or pnpm

### API Keys
- Future AGI API Key
- Future AGI Secret Key

Get your keys from the [Future AGI Platform](https://app.futureagi.com) under **Settings > Keys**.

---

## Installation

### Python

```bash
pip install ai-evaluation
```

Or with poetry:

```bash
poetry add ai-evaluation
```

### TypeScript/JavaScript

```bash
npm install @future-agi/ai-evaluation
```

Or with yarn/pnpm:

```bash
yarn add @future-agi/ai-evaluation
pnpm add @future-agi/ai-evaluation
```

### Verify Installation

**Python:**
```bash
python -c "from fi.evals import Evaluator; print('Success!')"
```

**TypeScript:**
```bash
npx ts-node -e "import { Evaluator } from '@future-agi/ai-evaluation'; console.log('Success!');"
```

---

## Authentication

### Option 1: Environment Variables (Recommended)

Set environment variables in your shell:

```bash
export FI_API_KEY="your_api_key_here"
export FI_SECRET_KEY="your_secret_key_here"
```

Or create a `.env` file:

```env
FI_API_KEY=your_api_key_here
FI_SECRET_KEY=your_secret_key_here
```

### Option 2: Direct Initialization

**Python:**
```python
from fi.evals import Evaluator

evaluator = Evaluator(
    fi_api_key="your_api_key",
    fi_secret_key="your_secret_key"
)
```

**TypeScript:**
```typescript
import { Evaluator } from "@future-agi/ai-evaluation";

const evaluator = new Evaluator({
  apiKey: "your_api_key",
  secretKey: "your_secret_key"
});
```

---

## Your First Evaluation

### Python Example

```python
from fi.evals import Evaluator

# Initialize (uses environment variables)
evaluator = Evaluator()

# Run a simple tone evaluation
result = evaluator.evaluate(
    eval_templates="tone",
    inputs={
        "input": "Dear Sir, I hope this email finds you well."
    },
    model_name="turing_flash"
)

# Print results
print(f"Output: {result.eval_results[0].output}")
print(f"Reason: {result.eval_results[0].reason}")
```

**Expected Output:**
```
Output: FORMAL
Reason: The text uses formal salutation "Dear Sir" and polite phrasing...
```

### TypeScript Example

```typescript
import { Evaluator } from "@future-agi/ai-evaluation";

async function main() {
  // Initialize (uses environment variables)
  const evaluator = new Evaluator();

  // Run a factual accuracy evaluation
  const result = await evaluator.evaluate(
    "factual_accuracy",
    {
      input: "What is the capital of France?",
      output: "The capital of France is Paris.",
      context: "France is a country in Europe with Paris as its capital."
    },
    { modelName: "turing_flash" }
  );

  console.log(result);
}

main();
```

---

## Common Evaluation Examples

### RAG/Groundedness Check

```python
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={
        "context": "The Eiffel Tower was built in 1889 and is 324 meters tall.",
        "output": "The Eiffel Tower, built in 1889, stands at 324 meters."
    },
    model_name="turing_flash"
)
```

### Content Safety

```python
result = evaluator.evaluate(
    eval_templates="content_moderation",
    inputs={
        "text": "User message to check for harmful content..."
    },
    model_name="protect_flash"
)
```

### JSON Validation

```python
result = evaluator.evaluate(
    eval_templates="is_json",
    inputs={
        "text": '{"name": "Alice", "age": 30}'
    },
    model_name="turing_flash"
)
```

### Helpfulness Check

```python
result = evaluator.evaluate(
    eval_templates="is_helpful",
    inputs={
        "input": "How do I reset my password?",
        "output": "Click on 'Forgot Password' on the login page and follow the instructions."
    },
    model_name="turing_flash"
)
```

---

## CLI Quick Start

The CLI is included with the Python package.

### Initialize a Project

```bash
fi init my-evals
cd my-evals
```

This creates:
```
my-evals/
├── fi-evaluation.yaml    # Configuration
├── data/
│   └── test_cases.json   # Sample data
└── results/              # Output directory
```

### Run Evaluations

```bash
# Set API keys
export FI_API_KEY="your_api_key"
export FI_SECRET_KEY="your_secret_key"

# Run evaluations
fi run
```

### List Available Templates

```bash
fi list templates
fi list templates --category rag
fi list categories
```

### Validate Configuration

```bash
fi validate
fi validate --strict
```

See the full [CLI Guide](./cli-guide.md) for more details.

---

## Batch Evaluations

Run multiple evaluations efficiently:

### Python

```python
from fi.evals import Evaluator

evaluator = Evaluator()

# Multiple test cases
test_cases = [
    {"input": "Formal email content...", "expected": "FORMAL"},
    {"input": "Hey buddy, what's up?", "expected": "INFORMAL"},
    {"input": "Dear Customer, we regret...", "expected": "FORMAL"},
]

# Run batch evaluation
results = evaluator.batch_evaluate(
    eval_templates="tone",
    inputs_list=[{"input": tc["input"]} for tc in test_cases],
    model_name="turing_flash"
)

for i, result in enumerate(results):
    print(f"Case {i+1}: {result.eval_results[0].output}")
```

---

## Available Models

| Model | Use Case | Speed |
|-------|----------|-------|
| `turing_flash` | General evaluations | Fast |
| `turing_pro` | Complex evaluations | Moderate |
| `protect_flash` | Safety evaluations | Fast |
| `protect_pro` | Detailed safety analysis | Moderate |

---

## Next Steps

1. **Explore Templates** - See all [60+ templates](./templates-reference.md)
2. **Python SDK** - Full [Python reference](./python-sdk.md)
3. **TypeScript SDK** - Full [TypeScript reference](./typescript-sdk.md)
4. **CLI Usage** - Complete [CLI guide](./cli-guide.md)
5. **Configuration** - [YAML configuration](./configuration.md) for CI/CD

---

## Troubleshooting

### API Keys Not Found

```
Error: API keys not configured
```

**Solution:** Set environment variables or pass keys directly to the Evaluator.

### Template Not Found

```
Error: Unknown template 'my_template'
```

**Solution:** Run `fi list templates` to see available templates.

### Rate Limiting

```
Error: Rate limit exceeded
```

**Solution:** Add delays between requests or contact support for higher limits.

---

## Support

- **Documentation**: [docs.futureagi.com](https://docs.futureagi.com)
- **GitHub Issues**: [Report bugs](https://github.com/future-agi/ai-evaluation/issues)
- **Email**: support@futureagi.com
