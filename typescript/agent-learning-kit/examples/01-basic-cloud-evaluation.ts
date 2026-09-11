/**
 * Example 1: Basic Cloud Evaluation
 *
 * This example demonstrates how to run evaluations using the Future AGI
 * cloud API. You'll need API credentials to run this example.
 *
 * Prerequisites:
 * - Set FI_API_KEY and FI_SECRET_KEY environment variables
 * - Or pass credentials directly to the Evaluator
 *
 * Run: npx ts-node examples/01-basic-cloud-evaluation.ts
 */

import { Evaluator, evaluate } from '../src';

async function main() {
  // ============================================================
  // Option 1: Using the convenience function (simplest)
  // ============================================================
  console.log('=== Option 1: Convenience Function ===\n');

  // The evaluate() function creates an Evaluator internally
  // Credentials are read from environment variables
  const simpleResult = await evaluate(
    'Factual Accuracy',
    {
      response: ['The capital of France is Paris.'],
      context: ['Paris is the capital and largest city of France.']
    }
  );

  console.log('Evaluation:', simpleResult.eval_results[0]?.name);
  console.log('Score:', simpleResult.eval_results[0]?.output);
  console.log('Reason:', simpleResult.eval_results[0]?.reason);
  console.log('Runtime:', simpleResult.eval_results[0]?.runtime, 'seconds\n');

  // ============================================================
  // Option 2: Using the Evaluator class (more control)
  // ============================================================
  console.log('=== Option 2: Evaluator Class ===\n');

  const evaluator = new Evaluator({
    fiApiKey: process.env.FI_API_KEY,
    fiSecretKey: process.env.FI_SECRET_KEY,
    timeout: 300, // 5 minute timeout
    maxWorkers: 8 // Parallel workers for batch operations
  });

  // Single evaluation
  const groundednessResult = await evaluator.evaluate(
    'Groundedness',
    {
      query: ['What is machine learning?'],
      response: ['Machine learning is a subset of AI that enables computers to learn from data.'],
      context: ['Machine learning (ML) is a field of artificial intelligence that uses statistical techniques to give computers the ability to learn from data.']
    },
    {
      modelName: 'gpt-4o' // Specify which model to use for evaluation
    }
  );

  console.log('Groundedness Score:', groundednessResult.eval_results[0]?.output);
  console.log('Reason:', groundednessResult.eval_results[0]?.reason);

  // ============================================================
  // Option 3: Multiple evaluations at once
  // ============================================================
  console.log('\n=== Option 3: Multiple Evaluations ===\n');

  const multiResult = await evaluator.evaluate(
    ['Factual Accuracy', 'Relevance', 'Completeness'],
    {
      query: ['Explain photosynthesis'],
      response: ['Photosynthesis is the process by which plants convert sunlight into energy. They use chlorophyll to capture light and transform carbon dioxide and water into glucose and oxygen.'],
      context: ['Photosynthesis is a biological process used by plants, algae, and some bacteria to convert light energy into chemical energy stored in glucose.']
    }
  );

  console.log('Multiple evaluation results:');
  for (const result of multiResult.eval_results) {
    if (result) {
      console.log(`  ${result.name}: ${result.output} (${result.reason?.substring(0, 50)}...)`);
    }
  }

  // ============================================================
  // Option 4: Async evaluation (for long-running tasks)
  // ============================================================
  console.log('\n=== Option 4: Async Evaluation ===\n');

  const asyncResult = await evaluator.evaluate(
    'Factual Accuracy',
    {
      response: ['The speed of light is approximately 300,000 km/s.'],
      context: ['The speed of light in vacuum is exactly 299,792,458 metres per second.']
    },
    {
      isAsync: true // Returns immediately with eval_id
    }
  );

  console.log('Async eval started, ID:', asyncResult.eval_results[0]?.eval_id);

  // Poll for results
  if (asyncResult.eval_results[0]?.eval_id) {
    const evalId = asyncResult.eval_results[0].eval_id;
    console.log('Polling for result...');

    // Wait a bit then get result
    await new Promise(resolve => setTimeout(resolve, 2000));

    const finalResult = await evaluator.getEvalResult(evalId);
    console.log('Final result:', finalResult);
  }

  // ============================================================
  // List available evaluations
  // ============================================================
  console.log('\n=== Available Evaluations ===\n');

  const availableEvals = await evaluator.list_evaluations();
  console.log(`Found ${availableEvals.length} evaluation templates:`);
  availableEvals.slice(0, 10).forEach(evalTemplate => {
    console.log(`  - ${evalTemplate.name}`);
  });
  if (availableEvals.length > 10) {
    console.log(`  ... and ${availableEvals.length - 10} more`);
  }
}

// Run the example
main().catch(console.error);
