# TypeScript SDK Reference

> Complete reference for the AI Evaluation TypeScript/JavaScript SDK

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
npm install @future-agi/ai-evaluation
```

Or with yarn/pnpm:

```bash
yarn add @future-agi/ai-evaluation
pnpm add @future-agi/ai-evaluation
```

**Requirements:**
- Node.js 18.0.0+
- npm, yarn, or pnpm

---

## Evaluator Class

The main class for running evaluations.

### Constructor

```typescript
import { Evaluator } from "@future-agi/ai-evaluation";

const evaluator = new Evaluator({
  fiApiKey?: string,      // API key (or use FI_API_KEY env var)
  fiSecretKey?: string,   // Secret key (or use FI_SECRET_KEY env var)
  fiBaseUrl?: string,     // Base URL (default: https://api.futureagi.com)
  timeout?: number,       // Default timeout in seconds
  maxQueue?: number,      // Max queue size
  maxWorkers?: number,    // Max parallel workers (default: 8)
});
```

### Example

```typescript
import { Evaluator } from "@future-agi/ai-evaluation";

// Using environment variables (recommended)
const evaluator = new Evaluator();

// Or with explicit credentials
const evaluator = new Evaluator({
  fiApiKey: "your_api_key",
  fiSecretKey: "your_secret_key"
});
```

---

## Methods

### evaluate()

Run a single evaluation.

```typescript
const result = await evaluator.evaluate(
  evalTemplates: string | EvalTemplate | (string | EvalTemplate)[],
  inputs: Record<string, string | string[]>,
  options: {
    modelName: string,           // Required: Model to use
    timeout?: number,            // Optional timeout override
    customEvalName?: string,     // Custom name for tracing
    traceEval?: boolean,         // Enable OpenTelemetry tracing
  }
): Promise<BatchRunResult>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `evalTemplates` | `string` or `EvalTemplate` | Yes | Template name (e.g., "groundedness") or template object |
| `inputs` | `Record<string, string \| string[]>` | Yes | Input data matching template requirements |
| `options.modelName` | `string` | Yes | Model to use: `turing_flash`, `turing_pro`, `protect_flash`, `protect_pro` |
| `options.timeout` | `number` | No | Timeout in seconds (default: 200) |
| `options.customEvalName` | `string` | No | Custom name for tracing/tracking |
| `options.traceEval` | `boolean` | No | Enable OpenTelemetry tracing |

**Example:**

```typescript
const result = await evaluator.evaluate(
  "groundedness",
  {
    context: "The sky is blue due to Rayleigh scattering.",
    output: "The sky appears blue because of how light scatters."
  },
  { modelName: "turing_flash" }
);

console.log(result.eval_results[0].data);     // e.g., "GROUNDED"
console.log(result.eval_results[0].reason);   // Explanation
console.log(result.eval_results[0].runtime);  // Execution time
```

---

### list_evaluations()

List all available evaluation templates.

```typescript
const templates = await evaluator.list_evaluations(): Promise<Record<string, any>[]>
```

**Returns:** Array of template objects with:
- `name`: Template name
- `description`: Template description
- `eval_tags`: Category tags
- `config`: Required inputs and outputs

**Example:**

```typescript
const templates = await evaluator.list_evaluations();

for (const template of templates) {
  console.log(`${template.name}: ${template.description}`);
}
```

---

## Types

### BatchRunResult

Container for evaluation results.

```typescript
interface BatchRunResult {
  eval_results: (EvalResult | null)[];
}
```

### EvalResult

Individual evaluation result.

```typescript
interface EvalResult {
  data?: any;                    // Evaluation output
  failure?: boolean;             // Whether evaluation failed
  reason: string;                // Explanation of the evaluation
  runtime: number;               // Execution time in seconds
  metadata: Record<string, any>; // Additional metadata
  metrics: EvalResultMetric[];   // Detailed metrics
}

interface EvalResultMetric {
  id: string;
  value: any;
}
```

**Example:**

```typescript
const result = await evaluator.evaluate(...);

for (const evalResult of result.eval_results) {
  if (evalResult) {
    console.log(`Output: ${evalResult.data}`);
    console.log(`Reason: ${evalResult.reason}`);
    console.log(`Runtime: ${evalResult.runtime}s`);
  }
}
```

---

## Templates

### Using Template Names

```typescript
// Use template name as string
const result = await evaluator.evaluate(
  "groundedness",
  { context: "...", output: "..." },
  { modelName: "turing_flash" }
);
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

Process multiple test cases:

```typescript
import { Evaluator } from "@future-agi/ai-evaluation";

const evaluator = new Evaluator();

const testCases = [
  { context: "Context 1", output: "Response 1" },
  { context: "Context 2", output: "Response 2" },
  { context: "Context 3", output: "Response 3" },
];

const results = await Promise.all(
  testCases.map(testCase =>
    evaluator.evaluate(
      "groundedness",
      testCase,
      { modelName: "turing_flash" }
    )
  )
);

results.forEach((result, i) => {
  console.log(`Case ${i + 1}: ${result.eval_results[0]?.data}`);
});
```

### Multiple Templates

Run multiple evaluations on the same data:

```typescript
const templates = ["groundedness", "completeness", "is_helpful"];
const inputs = {
  context: "The Eiffel Tower is 324 meters tall.",
  input: "How tall is the Eiffel Tower?",
  output: "The Eiffel Tower is 324 meters tall."
};

for (const template of templates) {
  const result = await evaluator.evaluate(
    template,
    inputs,
    { modelName: "turing_flash" }
  );
  console.log(`${template}: ${result.eval_results[0]?.data}`);
}
```

### Array Inputs

Pass multiple values for a single field:

```typescript
const result = await evaluator.evaluate(
  "chunk_utilization",
  {
    context: ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"],
    output: "Response using some chunks"
  },
  { modelName: "turing_flash" }
);
```

### OpenTelemetry Tracing

Enable tracing for observability:

```typescript
// Requires @opentelemetry/api and @traceai/fi-core packages
const result = await evaluator.evaluate(
  "groundedness",
  { context: "...", output: "..." },
  {
    modelName: "turing_flash",
    traceEval: true,
    customEvalName: "my_groundedness_check"
  }
);
```

### Custom Timeout

```typescript
const evaluator = new Evaluator({
  timeout: 300  // 5 minute default timeout
});

// Or per-evaluation
const result = await evaluator.evaluate(
  "groundedness",
  { context: "...", output: "..." },
  {
    modelName: "turing_flash",
    timeout: 60  // 60 second timeout for this call
  }
);
```

---

## Convenience Functions

### Quick Evaluation

```typescript
import { evaluate, list_evaluations } from "@future-agi/ai-evaluation";

// One-liner evaluation
const result = await evaluate(
  "tone",
  { input: "Hello, how are you?" },
  { modelName: "turing_flash" }
);

// List templates
const templates = await list_evaluations();
```

---

## Error Handling

### Common Exceptions

```typescript
import { Evaluator } from "@future-agi/ai-evaluation";

const evaluator = new Evaluator();

try {
  const result = await evaluator.evaluate(
    "groundedness",
    { context: "...", output: "..." },
    { modelName: "turing_flash" }
  );
} catch (error) {
  if (error.message.includes("403")) {
    console.error("Invalid API credentials");
  } else if (error.message.includes("400")) {
    console.error("Invalid request:", error.message);
  } else {
    console.error("Evaluation failed:", error);
  }
}
```

### Validation Errors

```typescript
// Missing required modelName
try {
  const result = await evaluator.evaluate(
    "groundedness",
    { context: "...", output: "..." },
    { modelName: "" }  // Empty modelName
  );
} catch (error) {
  // TypeError: 'modelName' is a required option
}

// Invalid input type
try {
  const result = await evaluator.evaluate(
    "groundedness",
    { context: 123 as any, output: "..." },  // Invalid type
    { modelName: "turing_flash" }
  );
} catch (error) {
  // TypeError: Invalid input type
}
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FI_API_KEY` | API key | Required |
| `FI_SECRET_KEY` | Secret key | Required |
| `FI_BASE_URL` | API base URL | `https://api.futureagi.com` |

---

## ESM and CommonJS

The package supports both ESM and CommonJS:

**ESM:**
```typescript
import { Evaluator } from "@future-agi/ai-evaluation";
```

**CommonJS:**
```javascript
const { Evaluator } = require("@future-agi/ai-evaluation");
```

---

## TypeScript Configuration

Ensure your `tsconfig.json` includes:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "esModuleInterop": true
  }
}
```

---

## See Also

- [Getting Started](./getting-started.md)
- [Templates Reference](./templates-reference.md)
- [CLI Guide](./cli-guide.md)
- [Python SDK](./python-sdk.md)
