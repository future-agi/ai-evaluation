# AI Evaluation

This is a typescript project that contains the code for the AI Evaluation.

## Getting Started

1. Install dependencies

```bash
npm install @future-agi/ai-evaluation
```

to use the library, you can import it like this:

```typescript
import { Evaluator } from '@future-agi/ai-evaluation';
```

## Usage

```typescript
import { Evaluator } from "@future-agi/ai-evaluation";

const evaluator = new Evaluator();

const result = await evaluator.evaluate(
  "factual_accuracy",
  {
    input: "The capital of France is Paris.",
    output: "The capital of France is Paris.",
    context: "The capital of France is Paris.",
  },
  {
    modelName: "turing_flash",
  }
);

console.log(result);
```

