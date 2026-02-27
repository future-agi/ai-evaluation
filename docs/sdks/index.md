# SDK Reference

> Detailed API documentation for all language SDKs

---

## Available SDKs

| Language | Package | Status | Docs |
|----------|---------|--------|------|
| **Python** | `ai-evaluation` | ✅ Stable | [Python SDK](./python/index.md) |
| **TypeScript** | `@future-agi/ai-evaluation` | ✅ Stable | [TypeScript SDK](./typescript/index.md) |
| **Go** | `github.com/future-agi/ai-evaluation-go` | 🚧 Alpha | [Go SDK](./go/index.md) |
| **Java** | `com.futureagi:ai-evaluation` | 📋 Planned | [Java SDK](./java/index.md) |
| **Ruby** | `ai-evaluation` | 📋 Planned | [Ruby SDK](./ruby/index.md) |
| **Rust** | `ai-evaluation` | 📋 Planned | [Rust SDK](./rust/index.md) |

---

## Feature Matrix

| Feature | Python | TypeScript | Go | Java |
|---------|:------:|:----------:|:--:|:----:|
| **Core** |
| Single evaluation | ✅ | ✅ | 🚧 | 📋 |
| Batch evaluation | ✅ | ✅ | 🚧 | 📋 |
| Async evaluation | ✅ | ✅ | 🚧 | 📋 |
| List templates | ✅ | ✅ | 🚧 | 📋 |
| Get result by ID | ✅ | 🚧 | 📋 | 📋 |
| **Pipeline** |
| Pipeline evaluation | ✅ | 🚧 | 📋 | 📋 |
| Pipeline results | ✅ | 🚧 | 📋 | 📋 |
| **Integrations** |
| OpenTelemetry | ✅ | ✅ | 🚧 | 📋 |
| Langfuse | ✅ | 🚧 | ❌ | ❌ |
| LangChain | ✅ | 🚧 | ❌ | ❌ |
| **Advanced** |
| Custom templates | ✅ | 🚧 | 📋 | 📋 |
| Error localization | ✅ | 🚧 | 📋 | 📋 |
| Retry logic | ✅ | ✅ | 🚧 | 📋 |

✅ Available | 🚧 In Progress | 📋 Planned | ❌ Not Planned

---

## Quick Installation

### Python

```bash
pip install ai-evaluation

# With optional dependencies
pip install ai-evaluation[langfuse]
pip install ai-evaluation[opentelemetry]
```

### TypeScript

```bash
npm install @future-agi/ai-evaluation
# or
yarn add @future-agi/ai-evaluation
# or
pnpm add @future-agi/ai-evaluation
```

### Go

```bash
go get github.com/future-agi/ai-evaluation-go
```

---

## Quick Example

All SDKs follow the same pattern:

### 1. Initialize

```python
# Python
from fi.evals import Evaluator
evaluator = Evaluator()
```

```typescript
// TypeScript
import { Evaluator } from "@future-agi/ai-evaluation";
const evaluator = new Evaluator();
```

```go
// Go
import "github.com/future-agi/ai-evaluation-go"
evaluator := aievaluation.NewEvaluator()
```

### 2. Evaluate

```python
# Python
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={"context": "...", "output": "..."},
    model_name="turing_flash"
)
```

```typescript
// TypeScript
const result = await evaluator.evaluate(
  "groundedness",
  { context: "...", output: "..." },
  { modelName: "turing_flash" }
);
```

```go
// Go
result, err := evaluator.Evaluate(
    "groundedness",
    map[string]string{"context": "...", "output": "..."},
    &EvalOptions{ModelName: "turing_flash"},
)
```

### 3. Get Results

```python
# Python
print(result.eval_results[0].output)
print(result.eval_results[0].reason)
```

```typescript
// TypeScript
console.log(result.eval_results[0].data);
console.log(result.eval_results[0].reason);
```

```go
// Go
fmt.Println(result.EvalResults[0].Output)
fmt.Println(result.EvalResults[0].Reason)
```

---

## SDK Documentation

### [Python SDK](./python/index.md)

- [Installation](./python/installation.md)
- [Quickstart](./python/quickstart.md)
- [Evaluator Class](./python/evaluator.md)
- [Templates](./python/templates.md)
- [Async Patterns](./python/async.md)
- [Error Handling](./python/error-handling.md)
- [Testing](./python/testing.md)
- [Migration Guide](./python/migration.md)
- [Changelog](./python/changelog.md)

### [TypeScript SDK](./typescript/index.md)

- [Installation](./typescript/installation.md)
- [Quickstart](./typescript/quickstart.md)
- [Evaluator Class](./typescript/evaluator.md)
- [Templates](./typescript/templates.md)
- [Async Patterns](./typescript/async.md)
- [Error Handling](./typescript/error-handling.md)
- [Testing](./typescript/testing.md)
- [Migration Guide](./typescript/migration.md)
- [Changelog](./typescript/changelog.md)

### [Go SDK](./go/index.md) 🚧

- [Installation](./go/installation.md)
- [Quickstart](./go/quickstart.md)
- [Evaluator](./go/evaluator.md)
- [Error Handling](./go/error-handling.md)

---

## Environment Variables

All SDKs use the same environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `FI_API_KEY` | API key | Yes |
| `FI_SECRET_KEY` | Secret key | Yes |
| `FI_BASE_URL` | API base URL | No |

---

## See Also

- [Getting Started](../getting-started/index.md)
- [Core Concepts](../concepts/index.md)
- [API Reference](../api/index.md)
