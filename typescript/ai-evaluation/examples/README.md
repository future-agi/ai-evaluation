# Examples

This directory contains working examples for the `@future-agi/ai-evaluation` SDK.

## Prerequisites

```bash
# Install the package
npm install @future-agi/ai-evaluation

# For local LLM examples, install and run Ollama
# https://ollama.ai/
ollama pull llama3.2
ollama serve
```

## Running Examples

These examples are designed to work with the published package. To run them:

```bash
# 1. Copy an example to your project
cp examples/02-local-heuristic-metrics.ts my-test.ts

# 2. Update imports to use the package name
# Change: import { ... } from '../src/local'
# To:     import { ... } from '@future-agi/ai-evaluation/local'

# 3. Run with ts-node or tsx
npx tsx my-test.ts
```

**Development Note:** These examples import from `../src` for documentation purposes. When using the published package, import from `@future-agi/ai-evaluation` instead.

## Example Overview

### 01 - Basic Cloud Evaluation
Demonstrates cloud-based evaluation using the Future AGI API.
- Convenience function usage
- Evaluator class with configuration
- Multiple evaluations at once
- Async evaluations with polling

**Requires:** `FI_API_KEY` and `FI_SECRET_KEY` environment variables

### 02 - Local Heuristic Metrics
Run evaluations locally without any API calls.
- String metrics (contains, regex, length)
- JSON metrics (validation, schema)
- Similarity metrics (BLEU, ROUGE, Levenshtein)
- Batch evaluation
- Performance benchmarks

**Requires:** Nothing (fully offline)

### 03 - Hybrid Evaluation
Automatically route between local and cloud execution.
- Understanding routing logic
- Forcing local or cloud execution
- Offline mode
- Evaluation partitioning

**Requires:** Optionally `FI_API_KEY`/`FI_SECRET_KEY` and/or Ollama

### 04 - Ollama LLM-as-Judge
Use local Ollama for LLM-based evaluations.
- Text generation
- Chat completion
- Single judgment evaluations
- Custom evaluation criteria
- Batch judgments

**Requires:** Ollama running with a model (e.g., llama3.2)

### 05 - Batch Evaluation
Efficiently evaluate multiple inputs and metrics.
- Local batch operations
- Multi-metric evaluation
- JSON validation batches
- Similarity matrices
- Performance optimization

**Requires:** Nothing for local, API keys for cloud examples

## Environment Variables

Create a `.env` file or export these variables:

```bash
# Required for cloud evaluations
export FI_API_KEY=your-api-key
export FI_SECRET_KEY=your-secret-key

# Optional: Custom API URL
export FI_BASE_URL=https://api.futureagi.com

# Optional: Langfuse integration
export LANGFUSE_SECRET_KEY=your-langfuse-secret
export LANGFUSE_PUBLIC_KEY=your-langfuse-public
export LANGFUSE_HOST=https://cloud.langfuse.com
```

## Quick Start

```typescript
// Simplest possible example - local evaluation
import { LocalEvaluator } from '@future-agi/ai-evaluation/local';

const evaluator = new LocalEvaluator();
const result = evaluator.evaluate(
  'contains',
  [{ response: 'Hello World' }],
  { keyword: 'World' }
);

console.log(result.results.eval_results[0]?.output); // 1.0
```
