# @future-agi/ai-evaluation

TypeScript SDK for Future AGI's AI evaluation platform. Evaluate LLM outputs with 50+ metrics including factual accuracy, groundedness, relevance, and more.

## Installation

```bash
npm install @future-agi/ai-evaluation
# or
pnpm add @future-agi/ai-evaluation
# or
yarn add @future-agi/ai-evaluation
```

## Quick Start

```typescript
import { evaluate } from '@future-agi/ai-evaluation';

// Set your API key
process.env.FI_API_KEY = 'your-api-key';
process.env.FI_SECRET_KEY = 'your-secret-key';

// Run an evaluation
const result = await evaluate(
  'Factual Accuracy',
  {
    response: ['The capital of France is Paris.'],
    context: ['Paris is the capital and largest city of France.']
  }
);

console.log(result.eval_results[0].output); // Score: 0-1
console.log(result.eval_results[0].reason); // Explanation
```

## Features

- **Cloud Evaluations**: 50+ evaluation metrics via Future AGI API
- **Local Evaluations**: Run heuristic metrics offline without API calls
- **Hybrid Mode**: Automatically route between local and cloud execution
- **Local LLM Support**: Use Ollama for LLM-as-judge evaluations locally
- **Platform Integration**: Langfuse integration for observability
- **Pipeline Evaluation**: Evaluate entire ML pipelines

## Usage

### Cloud Evaluation (Default)

```typescript
import { Evaluator } from '@future-agi/ai-evaluation';

const evaluator = new Evaluator({
  fiApiKey: 'your-api-key',
  fiSecretKey: 'your-secret-key'
});

// Single evaluation
const result = await evaluator.evaluate(
  'Groundedness',
  {
    query: ['What is machine learning?'],
    response: ['Machine learning is a subset of AI...'],
    context: ['Machine learning (ML) is a field of AI...']
  },
  { modelName: 'gpt-4o' }
);

// Async evaluation (returns immediately, poll for results)
const asyncResult = await evaluator.evaluate(
  'Factual Accuracy',
  { response: ['...'], context: ['...'] },
  { isAsync: true }
);

// Get async result later
const finalResult = await evaluator.getEvalResult(asyncResult.eval_id);
```

### Local Evaluation (Offline)

Run evaluations locally without API calls using heuristic metrics:

```typescript
import { LocalEvaluator } from '@future-agi/ai-evaluation/local';

const evaluator = new LocalEvaluator();

// String metrics
const containsResult = evaluator.evaluate(
  'contains',
  [{ response: 'Hello world' }],
  { keyword: 'world' }
);
// Score: 1.0 (contains the keyword)

// JSON validation
const jsonResult = evaluator.evaluate(
  'json_schema',
  [{ response: '{"name": "John", "age": 30}' }],
  {
    schema: {
      type: 'object',
      properties: { name: { type: 'string' }, age: { type: 'number' } },
      required: ['name']
    }
  }
);

// Similarity metrics
const bleuResult = evaluator.evaluate(
  'bleu_score',
  [{ response: 'The cat sat on the mat' }],
  { reference: 'The cat is on the mat' }
);
```

### Available Local Metrics

| Category | Metrics |
|----------|---------|
| **String** | `regex`, `contains`, `contains_all`, `contains_any`, `contains_none`, `one_line`, `equals`, `starts_with`, `ends_with`, `length_less_than`, `length_greater_than`, `length_between` |
| **JSON** | `contains_json`, `is_json`, `json_schema` |
| **Similarity** | `bleu_score`, `rouge_score`, `recall_score`, `levenshtein_similarity`, `numeric_similarity`, `semantic_list_contains` |

### Hybrid Evaluation

Automatically route between local and cloud execution:

```typescript
import { HybridEvaluator, OllamaLLM } from '@future-agi/ai-evaluation/local';
import { Evaluator } from '@future-agi/ai-evaluation';

// Setup hybrid evaluator with local LLM
const localLLM = new OllamaLLM({ model: 'llama3.2' });
const cloudEvaluator = new Evaluator();

const hybrid = new HybridEvaluator({
  localLLM,
  cloudEvaluator,
  preferLocal: true,      // Prefer local when possible
  fallbackToCloud: true,  // Fall back to cloud if local fails
  offlineMode: false      // Set true to disable cloud entirely
});

// Heuristic metrics run locally
const localResult = await hybrid.evaluate(
  'contains',
  [{ response: 'Hello world' }],
  { keyword: 'world' }
);

// LLM-based metrics use local Ollama if available
const llmResult = await hybrid.evaluate(
  'groundedness',
  [{
    query: 'What is AI?',
    response: 'AI is artificial intelligence.',
    context: 'Artificial intelligence (AI) is...'
  }]
);
```

### Local LLM with Ollama

Use Ollama for local LLM-as-judge evaluations:

```typescript
import { OllamaLLM } from '@future-agi/ai-evaluation/local';

// Ensure Ollama is running: ollama serve
const llm = new OllamaLLM({
  model: 'llama3.2',           // Model name
  baseUrl: 'http://localhost:11434',  // Ollama URL
  temperature: 0.0,            // Deterministic output
  maxTokens: 1024,
  timeout: 120                 // Seconds
});

// Check availability
const isAvailable = await llm.isAvailable();

// Direct generation
const response = await llm.generate('Explain quantum computing');

// Chat completion
const chatResponse = await llm.chat([
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is machine learning?' }
]);

// LLM-as-judge
const judgment = await llm.judge(
  'What is the capital of France?',      // Query
  'The capital of France is Paris.',     // Response to evaluate
  'Evaluate factual accuracy. Score 0-1.', // Criteria
  'Paris is the capital of France.'      // Optional context
);
// Returns: { score: 1.0, passed: true, reason: '...' }
```

### Pipeline Evaluation

Evaluate entire ML pipelines:

```typescript
const evaluator = new Evaluator();

// Submit pipeline evaluation
await evaluator.evaluatePipeline(
  'my-project',
  'v1.0.0',
  [
    { input: 'query1', output: 'response1', context: 'ctx1' },
    { input: 'query2', output: 'response2', context: 'ctx2' }
  ]
);

// Get results for multiple versions
const results = await evaluator.getPipelineResults(
  'my-project',
  ['v1.0.0', 'v1.1.0', 'v2.0.0']
);
```

### Langfuse Integration

Enable observability with Langfuse:

```typescript
const evaluator = new Evaluator({
  fiApiKey: process.env.FI_API_KEY,
  fiSecretKey: process.env.FI_SECRET_KEY,
  langfuseSecretKey: process.env.LANGFUSE_SECRET_KEY,
  langfusePublicKey: process.env.LANGFUSE_PUBLIC_KEY,
  langfuseHost: process.env.LANGFUSE_HOST
});

// Evaluations will be logged to Langfuse
const result = await evaluator.evaluate(
  'Groundedness',
  { response: ['...'], context: ['...'] },
  { platform: 'langfuse', customEvalName: 'my-eval' }
);
```

## API Reference

### Main Exports (`@future-agi/ai-evaluation`)

| Export | Description |
|--------|-------------|
| `Evaluator` | Main class for cloud evaluations |
| `evaluate()` | Convenience function for single evaluation |
| `list_evaluations()` | List available evaluation templates |
| `get_eval_result()` | Get async evaluation result |
| `evaluate_pipeline()` | Evaluate a pipeline |
| `get_pipeline_results()` | Get pipeline results |

### Local Exports (`@future-agi/ai-evaluation/local`)

| Export | Description |
|--------|-------------|
| `LocalEvaluator` | Run heuristic metrics locally |
| `HybridEvaluator` | Route between local and cloud |
| `OllamaLLM` | Local LLM client via Ollama |
| `LocalLLMFactory` | Factory for creating LLM instances |
| `canRunLocally()` | Check if metric runs locally |
| `requiresLLM()` | Check if metric needs LLM |
| Individual metrics | `contains`, `regex`, `bleuScore`, etc. |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FI_API_KEY` | Future AGI API key |
| `FI_SECRET_KEY` | Future AGI secret key |
| `FI_BASE_URL` | API base URL (optional) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key (optional) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (optional) |
| `LANGFUSE_HOST` | Langfuse host URL (optional) |

## Requirements

- Node.js >= 18.0.0
- For local LLM: [Ollama](https://ollama.ai/) installed and running

## License

MIT

## Links

- [Documentation](https://docs.futureagi.com)
- [GitHub](https://github.com/futureagi/ai-evaluation)
- [Issues](https://github.com/futureagi/ai-evaluation/issues)
