/**
 * Example 5: Batch Evaluation
 *
 * This example demonstrates how to efficiently evaluate multiple inputs
 * and metrics using batch operations - both locally and in the cloud.
 *
 * Batch evaluation is essential for:
 * - Evaluating test datasets
 * - Comparing multiple model outputs
 * - Running comprehensive quality checks
 *
 * Run: npx ts-node examples/05-batch-evaluation.ts
 */

import { Evaluator } from '../src';
import { LocalEvaluator } from '../src/local';

// Sample test dataset
const testDataset = [
  {
    query: 'What is the capital of France?',
    response: 'The capital of France is Paris.',
    context: 'Paris is the capital and most populous city of France.',
    expectedKeywords: ['Paris', 'capital']
  },
  {
    query: 'Explain photosynthesis',
    response: 'Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.',
    context: 'Photosynthesis is a biological process where plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.',
    expectedKeywords: ['plants', 'sunlight', 'energy']
  },
  {
    query: 'What is machine learning?',
    response: 'Machine learning is a subset of AI that enables computers to learn from data without explicit programming.',
    context: 'Machine learning is a field of artificial intelligence that uses algorithms to learn patterns from data.',
    expectedKeywords: ['AI', 'data', 'learn']
  },
  {
    query: 'How does gravity work?',
    response: 'Gravity is a force that attracts objects with mass toward each other.',
    context: 'Gravity is one of the four fundamental forces of nature, attracting objects with mass.',
    expectedKeywords: ['force', 'mass', 'attract']
  },
  {
    query: 'What is the speed of light?',
    response: 'The speed of light is approximately 300,000 kilometers per second.',
    context: 'The speed of light in vacuum is exactly 299,792,458 meters per second.',
    expectedKeywords: ['300,000', 'kilometers', 'second']
  }
];

async function main() {
  console.log('=== Batch Evaluation Examples ===\n');
  console.log(`Test dataset: ${testDataset.length} samples\n`);

  // ============================================================
  // Local Batch Evaluation
  // ============================================================
  console.log('--- Local Batch Evaluation ---\n');

  const localEvaluator = new LocalEvaluator();

  // Prepare batch evaluations
  const localBatchEvals = testDataset.flatMap((sample, index) => [
    {
      metricName: 'contains_all',
      inputs: [{ response: sample.response }],
      config: { keywords: sample.expectedKeywords }
    },
    {
      metricName: 'length_between',
      inputs: [{ response: sample.response }],
      config: { minLength: 20, maxLength: 500 }
    },
    {
      metricName: 'one_line',
      inputs: [{ response: sample.response }]
    }
  ]);

  console.log(`Running ${localBatchEvals.length} local evaluations...`);
  const localStart = Date.now();

  const localResults = localEvaluator.evaluateBatch(localBatchEvals);

  const localElapsed = Date.now() - localStart;
  console.log(`Completed in ${localElapsed}ms`);
  console.log(`Executed locally: ${localResults.executedLocally.length} metrics`);

  // Analyze results
  const passedCount = localResults.results.eval_results.filter(r => r?.output === 1).length;
  const totalCount = localResults.results.eval_results.length;
  console.log(`Results: ${passedCount}/${totalCount} passed (${(passedCount/totalCount*100).toFixed(1)}%)\n`);

  // Show detailed results per sample
  console.log('Per-sample results:');
  for (let i = 0; i < testDataset.length; i++) {
    const baseIndex = i * 3;
    const containsAll = localResults.results.eval_results[baseIndex]?.output === 1;
    const lengthOk = localResults.results.eval_results[baseIndex + 1]?.output === 1;
    const oneLine = localResults.results.eval_results[baseIndex + 2]?.output === 1;

    console.log(`  Sample ${i + 1}: Contains=${containsAll ? 'PASS' : 'FAIL'}, Length=${lengthOk ? 'PASS' : 'FAIL'}, OneLine=${oneLine ? 'PASS' : 'FAIL'}`);
  }

  // ============================================================
  // Multi-Metric Local Evaluation
  // ============================================================
  console.log('\n--- Multi-Metric Evaluation per Sample ---\n');

  // Evaluate each sample with multiple metrics
  for (let i = 0; i < 2; i++) { // Show first 2 samples
    const sample = testDataset[i];
    console.log(`Sample ${i + 1}: "${sample.query}"`);

    const multiResult = localEvaluator.evaluateBatch([
      {
        metricName: 'contains_all',
        inputs: [{ response: sample.response }],
        config: { keywords: sample.expectedKeywords }
      },
      {
        metricName: 'length_greater_than',
        inputs: [{ response: sample.response }],
        config: { minLength: 10 }
      },
      {
        metricName: 'regex',
        inputs: [{ response: sample.response }],
        config: { pattern: '[A-Z][a-z]+' } // Proper nouns
      },
      {
        metricName: 'contains',
        inputs: [{ response: sample.response }],
        config: { keyword: 'is', caseSensitive: false }
      }
    ]);

    multiResult.results.eval_results.forEach((result, idx) => {
      const metricName = multiResult.executedLocally[idx] || 'unknown';
      const status = result?.output === 1 ? 'PASS' : 'FAIL';
      console.log(`  ${metricName}: ${status} - ${result?.reason?.substring(0, 50)}...`);
    });
    console.log();
  }

  // ============================================================
  // JSON Validation Batch
  // ============================================================
  console.log('--- JSON Response Validation ---\n');

  const jsonResponses = [
    '{"name": "John", "age": 30}',
    '{"status": "success", "data": [1, 2, 3]}',
    'Invalid JSON here',
    '{"error": null, "result": {"value": 42}}',
    '[{"id": 1}, {"id": 2}]'
  ];

  const jsonBatch = jsonResponses.map((response, idx) => ({
    metricName: 'is_json',
    inputs: [{ response }]
  }));

  const jsonResults = localEvaluator.evaluateBatch(jsonBatch);

  console.log('JSON validation results:');
  jsonResponses.forEach((resp, idx) => {
    const isValid = jsonResults.results.eval_results[idx]?.output === 1;
    const truncated = resp.length > 30 ? resp.substring(0, 30) + '...' : resp;
    console.log(`  "${truncated}": ${isValid ? 'VALID' : 'INVALID'}`);
  });

  // ============================================================
  // Similarity Comparison Batch
  // ============================================================
  console.log('\n--- Similarity Score Matrix ---\n');

  const reference = 'Machine learning is a type of artificial intelligence.';
  const variations = [
    'Machine learning is a type of artificial intelligence.',  // Exact
    'ML is a form of AI.',                                      // Short
    'Artificial intelligence includes machine learning.',       // Reordered
    'Deep learning is a subset of machine learning.',           // Different
    'The weather is nice today.'                                // Unrelated
  ];

  console.log(`Reference: "${reference}"\n`);

  const similarityBatch = variations.map(response => ({
    metricName: 'bleu_score',
    inputs: [{ response }],
    config: { reference }
  }));

  const similarityResults = localEvaluator.evaluateBatch(similarityBatch);

  console.log('BLEU scores:');
  variations.forEach((variation, idx) => {
    const score = similarityResults.results.eval_results[idx]?.output as number;
    const bar = '█'.repeat(Math.round(score * 20)) + '░'.repeat(20 - Math.round(score * 20));
    const truncated = variation.length > 50 ? variation.substring(0, 50) + '...' : variation;
    console.log(`  ${bar} ${score?.toFixed(3)} - "${truncated}"`);
  });

  // ============================================================
  // Cloud Batch Evaluation (if credentials available)
  // ============================================================
  if (process.env.FI_API_KEY && process.env.FI_SECRET_KEY) {
    console.log('\n--- Cloud Batch Evaluation ---\n');

    const cloudEvaluator = new Evaluator({
      fiApiKey: process.env.FI_API_KEY,
      fiSecretKey: process.env.FI_SECRET_KEY,
      maxWorkers: 8 // Parallel execution
    });

    // Evaluate multiple samples with cloud metrics
    console.log('Running cloud evaluations on test dataset...');

    try {
      const cloudResult = await cloudEvaluator.evaluate(
        ['Factual Accuracy', 'Relevance'],
        {
          query: testDataset.map(s => s.query),
          response: testDataset.map(s => s.response),
          context: testDataset.map(s => s.context)
        }
      );

      console.log(`\nCloud results (${cloudResult.eval_results.length} evaluations):`);
      cloudResult.eval_results.forEach((result, idx) => {
        if (result) {
          console.log(`  ${result.name}: ${result.output} (${result.runtime?.toFixed(2)}s)`);
        }
      });
    } catch (error) {
      console.log('Cloud evaluation error:', (error as Error).message);
    }
  } else {
    console.log('\n--- Cloud Batch Evaluation ---\n');
    console.log('Skipped: Set FI_API_KEY and FI_SECRET_KEY to run cloud evaluations.');
  }

  // ============================================================
  // Performance Summary
  // ============================================================
  console.log('\n--- Performance Summary ---\n');

  // Run a performance test
  const perfIterations = 100;
  const perfBatch = Array(perfIterations).fill(null).map(() => ({
    metricName: 'contains',
    inputs: [{ response: 'Test response with keyword' }],
    config: { keyword: 'keyword' }
  }));

  const perfStart = Date.now();
  localEvaluator.evaluateBatch(perfBatch);
  const perfElapsed = Date.now() - perfStart;

  console.log(`Batch performance test:`);
  console.log(`  Evaluations: ${perfIterations}`);
  console.log(`  Total time: ${perfElapsed}ms`);
  console.log(`  Per evaluation: ${(perfElapsed / perfIterations).toFixed(2)}ms`);
  console.log(`  Throughput: ${(perfIterations / perfElapsed * 1000).toFixed(0)} evals/sec`);

  console.log('\nTips for efficient batch evaluation:');
  console.log('  1. Group evaluations by metric type');
  console.log('  2. Use local metrics when possible (zero latency)');
  console.log('  3. Set appropriate maxWorkers for cloud evaluations');
  console.log('  4. Consider async mode for large cloud batches');
}

// Run the example
main().catch(console.error);
