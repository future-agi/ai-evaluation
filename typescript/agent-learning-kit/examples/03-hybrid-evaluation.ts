/**
 * Example 3: Hybrid Evaluation
 *
 * This example demonstrates how to use the HybridEvaluator to automatically
 * route between local and cloud execution based on metric type.
 *
 * Benefits:
 * - Heuristic metrics run instantly (local)
 * - LLM-based metrics use cloud or local Ollama
 * - Automatic fallback to cloud if local fails
 * - Offline mode for fully local operation
 *
 * Prerequisites:
 * - For cloud fallback: FI_API_KEY and FI_SECRET_KEY
 * - For local LLM: Ollama installed and running
 *
 * Run: npx ts-node examples/03-hybrid-evaluation.ts
 */

import { Evaluator } from '../src';
import { HybridEvaluator, OllamaLLM, canRunLocally, requiresLLM, selectExecutionMode } from '../src/local';

async function main() {
  console.log('=== Hybrid Evaluation ===\n');

  // ============================================================
  // Setup: Create evaluators
  // ============================================================
  console.log('--- Setting up evaluators ---\n');

  // Create a local LLM client (optional - for LLM-based local evals)
  const localLLM = new OllamaLLM({
    model: 'llama3.2',
    baseUrl: 'http://localhost:11434',
    temperature: 0.0,
    maxTokens: 1024,
    timeout: 120
  });

  // Check if local LLM is available
  const llmAvailable = await localLLM.isAvailable();
  console.log('Local LLM (Ollama) available:', llmAvailable);

  if (llmAvailable) {
    const models = await localLLM.listModels();
    console.log('Available models:', models.slice(0, 5).join(', '));
  }

  // Create cloud evaluator (for fallback)
  const cloudEvaluator = new Evaluator({
    fiApiKey: process.env.FI_API_KEY,
    fiSecretKey: process.env.FI_SECRET_KEY
  });

  // Create hybrid evaluator
  const hybrid = new HybridEvaluator({
    localLLM: llmAvailable ? localLLM : undefined,
    cloudEvaluator,
    preferLocal: true,       // Prefer local execution when possible
    fallbackToCloud: true,   // Fall back to cloud if local fails
    offlineMode: false       // Set to true to disable cloud entirely
  });

  // ============================================================
  // Understanding Routing
  // ============================================================
  console.log('\n--- Understanding Metric Routing ---\n');

  const metricsToCheck = [
    'contains',
    'is_json',
    'bleu_score',
    'groundedness',
    'factual_accuracy',
    'relevance'
  ];

  console.log('Metric routing analysis:');
  for (const metric of metricsToCheck) {
    const local = canRunLocally(metric);
    const needsLLM = requiresLLM(metric);
    const mode = selectExecutionMode(metric, llmAvailable, true);
    console.log(`  ${metric}:`);
    console.log(`    - Can run locally (heuristic): ${local}`);
    console.log(`    - Requires LLM: ${needsLLM}`);
    console.log(`    - Selected mode: ${mode}`);
  }

  // ============================================================
  // Running Hybrid Evaluations
  // ============================================================
  console.log('\n--- Running Hybrid Evaluations ---\n');

  // Test 1: Heuristic metric (always runs locally)
  console.log('Test 1: Heuristic metric (contains)');
  const containsResult = await hybrid.evaluate(
    'contains',
    [{ response: 'The answer is 42' }],
    { keyword: '42' }
  );
  console.log('  Result:', containsResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');
  console.log('  Executed locally:', containsResult.executedLocally);
  console.log('  Executed cloud:', containsResult.executedCloud);

  // Test 2: JSON validation (runs locally)
  console.log('\nTest 2: JSON schema validation');
  const jsonResult = await hybrid.evaluate(
    'json_schema',
    [{ response: '{"name": "John", "age": 30}' }],
    {
      schema: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          age: { type: 'number' }
        },
        required: ['name']
      }
    }
  );
  console.log('  Result:', jsonResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');
  console.log('  Executed locally:', jsonResult.executedLocally);

  // Test 3: Similarity metric (runs locally)
  console.log('\nTest 3: BLEU score comparison');
  const bleuResult = await hybrid.evaluate(
    'bleu_score',
    [{ response: 'The quick brown fox jumps over the lazy dog' }],
    { reference: 'A quick brown fox jumped over a lazy dog' }
  );
  console.log('  Score:', (bleuResult.results.eval_results[0]?.output as number)?.toFixed(3));
  console.log('  Executed locally:', bleuResult.executedLocally);

  // Test 4: LLM-based metric (uses local Ollama or falls back to cloud)
  if (llmAvailable || process.env.FI_API_KEY) {
    console.log('\nTest 4: LLM-based evaluation (groundedness)');
    console.log('  This will use:', llmAvailable ? 'Local Ollama' : 'Cloud API');

    try {
      const groundednessResult = await hybrid.evaluate(
        'groundedness',
        [{
          query: 'What is the capital of France?',
          response: 'The capital of France is Paris.',
          context: 'Paris is the capital and most populous city of France.'
        }]
      );
      console.log('  Score:', groundednessResult.results.eval_results[0]?.output);
      console.log('  Executed locally:', groundednessResult.executedLocally);
      console.log('  Executed cloud:', groundednessResult.executedCloud);
    } catch (error) {
      console.log('  Error:', (error as Error).message);
      console.log('  (This is expected if neither Ollama nor cloud credentials are available)');
    }
  }

  // ============================================================
  // Routing Methods
  // ============================================================
  console.log('\n--- Routing Methods ---\n');

  // Check how a metric would be routed
  const groundednessRoute = hybrid.routeEvaluation('groundedness');
  console.log('Groundedness would route to:', groundednessRoute);

  const containsRoute = hybrid.routeEvaluation('contains');
  console.log('Contains would route to:', containsRoute);

  // Force local (even for LLM metrics if Ollama available)
  const forcedLocal = hybrid.routeEvaluation('groundedness', true, false);
  console.log('Groundedness forced local:', forcedLocal);

  // Force cloud
  const forcedCloud = hybrid.routeEvaluation('contains', false, true);
  console.log('Contains forced cloud:', forcedCloud);

  // ============================================================
  // Partitioning Evaluations
  // ============================================================
  console.log('\n--- Partitioning Evaluations ---\n');

  const evaluations = [
    { metricName: 'contains', inputs: [{ response: 'test' }], config: { keyword: 'test' } },
    { metricName: 'is_json', inputs: [{ response: '{}' }] },
    { metricName: 'groundedness', inputs: [{ response: 'test', context: 'context' }] },
    { metricName: 'factual_accuracy', inputs: [{ response: 'test', context: 'context' }] },
    { metricName: 'bleu_score', inputs: [{ response: 'test' }], config: { reference: 'test' } }
  ];

  const partitioned = hybrid.partitionEvaluations(evaluations);
  console.log('Partitioned evaluations:');
  console.log('  Local metrics:', partitioned.local?.map(e => e.metricName) || []);
  console.log('  Cloud metrics:', partitioned.cloud?.map(e => e.metricName) || []);
  console.log('  Hybrid metrics:', partitioned.hybrid?.map(e => e.metricName) || []);

  // ============================================================
  // Offline Mode
  // ============================================================
  console.log('\n--- Offline Mode ---\n');

  const offlineHybrid = new HybridEvaluator({
    localLLM: llmAvailable ? localLLM : undefined,
    preferLocal: true,
    fallbackToCloud: false, // No cloud fallback
    offlineMode: true       // Strictly offline
  });

  // This works (local metric)
  const offlineContains = await offlineHybrid.evaluate(
    'contains',
    [{ response: 'Hello World' }],
    { keyword: 'World' }
  );
  console.log('Offline contains result:', offlineContains.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');

  // This would fail without local LLM (LLM metric in offline mode)
  console.log('Attempting LLM metric in offline mode...');
  try {
    const offlineGroundedness = await offlineHybrid.evaluate(
      'groundedness',
      [{ response: 'test', context: 'context' }]
    );
    console.log('  Result available (local LLM used)');
    console.log('  Executed locally:', offlineGroundedness.executedLocally);
  } catch (error) {
    console.log('  Error (expected without local LLM):', (error as Error).message);
  }

  // ============================================================
  // Check Local LLM Capabilities
  // ============================================================
  console.log('\n--- Local LLM Capabilities ---\n');

  const llmMetrics = ['groundedness', 'factual_accuracy', 'relevance', 'coherence'];
  console.log('Can handle with local LLM:');
  for (const metric of llmMetrics) {
    const canHandle = hybrid.canUseLocalLLM(metric);
    console.log(`  ${metric}: ${canHandle}`);
  }
}

// Run the example
main().catch(console.error);
