# Getting Started

> Get up and running with AI Evaluation in under 5 minutes

---

## Choose Your Language

| Language | Install Command | Guide |
|----------|-----------------|-------|
| **Python** | `pip install ai-evaluation` | [Python Guide](./python.md) |
| **TypeScript** | `npm install @future-agi/ai-evaluation` | [TypeScript Guide](./typescript.md) |
| **CLI** | `pip install ai-evaluation` | [CLI Guide](./cli.md) |
| **Go** | `go get github.com/future-agi/ai-evaluation-go` | [Go Guide](./go.md) 🚧 |
| **Java** | Coming soon | [Java Guide](./java.md) 📋 |

---

## Prerequisites

### API Keys

Get your API keys from [Future AGI Platform](https://app.futureagi.com):

1. Log in to your account
2. Navigate to **Settings > Keys**
3. Copy your **API Key** and **Secret Key**

### Set Environment Variables

```bash
export FI_API_KEY="your_api_key"
export FI_SECRET_KEY="your_secret_key"
```

---

## Quick Comparison

### Python

```python
from fi.evals import Evaluator

evaluator = Evaluator()
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={"context": "...", "output": "..."},
    model_name="turing_flash"
)
```

### TypeScript

```typescript
import { Evaluator } from "@future-agi/ai-evaluation";

const evaluator = new Evaluator();
const result = await evaluator.evaluate(
  "groundedness",
  { context: "...", output: "..." },
  { modelName: "turing_flash" }
);
```

### CLI

```yaml
# fi-evaluation.yaml
evaluations:
  - name: "rag_check"
    template: "groundedness"
    data: "./data/tests.json"
```

```bash
fi run
```

---

## Next Steps

After completing a getting started guide:

1. **[Core Concepts](../concepts/index.md)** - Understand how evaluations work
2. **[Templates](../templates/index.md)** - Explore 60+ evaluation templates
3. **[Tutorials](../tutorials/index.md)** - Follow step-by-step guides
4. **[SDK Reference](../sdks/index.md)** - Deep dive into the API
