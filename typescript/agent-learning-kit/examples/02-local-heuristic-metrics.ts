/**
 * Example 2: Local Heuristic Metrics
 *
 * This example demonstrates how to run evaluations locally without
 * any API calls. Perfect for offline use, fast testing, and cost savings.
 *
 * No API keys required!
 *
 * Run: npx ts-node examples/02-local-heuristic-metrics.ts
 */

import { LocalEvaluator, canRunLocally, getAvailableMetrics } from '../src/local/index';

function main() {
  const evaluator = new LocalEvaluator();

  console.log('=== Local Heuristic Metrics ===\n');

  // ============================================================
  // String Metrics
  // ============================================================
  console.log('--- String Metrics ---\n');

  // Contains keyword
  const containsResult = evaluator.evaluate(
    'contains',
    [{ response: 'The answer to life is 42.' }],
    { keyword: '42' }
  );
  console.log('Contains "42":', containsResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');
  console.log('  Reason:', containsResult.results.eval_results[0]?.reason);

  // Contains all keywords
  const containsAllResult = evaluator.evaluate(
    'contains_all',
    [{ response: 'Hello World! Welcome to JavaScript.' }],
    { keywords: ['hello', 'world', 'javascript'] }
  );
  console.log('\nContains all keywords:', containsAllResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');
  console.log('  Reason:', containsAllResult.results.eval_results[0]?.reason);

  // Regex pattern matching
  const regexResult = evaluator.evaluate(
    'regex',
    [{ response: 'Contact us at support@example.com for help.' }],
    { pattern: '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}' }
  );
  console.log('\nEmail pattern found:', regexResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');
  console.log('  Reason:', regexResult.results.eval_results[0]?.reason);

  // Length validation
  const lengthResult = evaluator.evaluate(
    'length_between',
    [{ response: 'This is a medium-length response with adequate detail.' }],
    { minLength: 10, maxLength: 100 }
  );
  console.log('\nLength in range [10, 100]:', lengthResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');
  console.log('  Reason:', lengthResult.results.eval_results[0]?.reason);

  // One-line check
  const oneLineResult = evaluator.evaluate(
    'one_line',
    [{ response: 'This is a single line response.' }]
  );
  console.log('\nIs single line:', oneLineResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');

  // Starts with / Ends with
  const startsWithResult = evaluator.evaluate(
    'starts_with',
    [{ response: 'ERROR: Something went wrong' }],
    { prefix: 'ERROR:' }
  );
  console.log('\nStarts with "ERROR:":', startsWithResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');

  // ============================================================
  // JSON Metrics
  // ============================================================
  console.log('\n--- JSON Metrics ---\n');

  // Check if response contains JSON
  const containsJsonResult = evaluator.evaluate(
    'contains_json',
    [{ response: 'Here is the data: {"name": "John", "age": 30}' }]
  );
  console.log('Contains JSON:', containsJsonResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');
  console.log('  Reason:', containsJsonResult.results.eval_results[0]?.reason);

  // Check if entire response is valid JSON
  const isJsonResult = evaluator.evaluate(
    'is_json',
    [{ response: '{"status": "success", "data": [1, 2, 3]}' }]
  );
  console.log('\nIs valid JSON:', isJsonResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');

  // JSON Schema validation
  const schemaResult = evaluator.evaluate(
    'json_schema',
    [{ response: '{"name": "Alice", "email": "alice@example.com", "age": 25}' }],
    {
      schema: {
        type: 'object',
        properties: {
          name: { type: 'string', minLength: 1 },
          email: { type: 'string', pattern: '^[^@]+@[^@]+\\.[^@]+$' },
          age: { type: 'number', minimum: 0 }
        },
        required: ['name', 'email']
      }
    }
  );
  console.log('\nMatches JSON schema:', schemaResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');
  console.log('  Reason:', schemaResult.results.eval_results[0]?.reason);

  // Invalid JSON example
  const invalidJsonResult = evaluator.evaluate(
    'is_json',
    [{ response: 'This is not JSON at all' }]
  );
  console.log('\nInvalid JSON test:', invalidJsonResult.results.eval_results[0]?.output === 0 ? 'Correctly detected as invalid' : 'ERROR');

  // ============================================================
  // Similarity Metrics
  // ============================================================
  console.log('\n--- Similarity Metrics ---\n');

  // BLEU Score
  const bleuResult = evaluator.evaluate(
    'bleu_score',
    [{ response: 'The cat sat on the mat' }],
    { reference: 'The cat is sitting on the mat' }
  );
  console.log('BLEU Score:', (bleuResult.results.eval_results[0]?.output as number)?.toFixed(3));
  console.log('  Reason:', bleuResult.results.eval_results[0]?.reason);

  // ROUGE Score
  const rougeResult = evaluator.evaluate(
    'rouge_score',
    [{ response: 'Machine learning is a type of AI that learns from data.' }],
    { reference: 'Machine learning is artificial intelligence that learns from examples.', variant: 'rouge-l' }
  );
  console.log('\nROUGE-L Score:', (rougeResult.results.eval_results[0]?.output as number)?.toFixed(3));

  // Levenshtein Similarity
  const levenshteinResult = evaluator.evaluate(
    'levenshtein_similarity',
    [{ response: 'hello world' }],
    { reference: 'hello word' }
  );
  console.log('\nLevenshtein Similarity:', (levenshteinResult.results.eval_results[0]?.output as number)?.toFixed(3));

  // Numeric Similarity
  const numericResult = evaluator.evaluate(
    'numeric_similarity',
    [{ response: '3.14159' }],
    { reference: Math.PI, tolerance: 0.001 }
  );
  console.log('\nNumeric Similarity (pi):', (numericResult.results.eval_results[0]?.output as number)?.toFixed(3));
  console.log('  Reason:', numericResult.results.eval_results[0]?.reason);

  // Semantic List Contains
  const listResult = evaluator.evaluate(
    'semantic_list_contains',
    [{ response: 'I bought apples, oranges, and milk from the store.' }],
    { items: ['apples', 'bananas', 'milk', 'bread'], threshold: 0.5 }
  );
  console.log('\nList item match (50% threshold):', listResult.results.eval_results[0]?.output === 1 ? 'PASS' : 'FAIL');
  console.log('  Reason:', listResult.results.eval_results[0]?.reason);

  // ============================================================
  // Utility Functions
  // ============================================================
  console.log('\n--- Utility Functions ---\n');

  // Check which metrics can run locally
  console.log('Can run locally:');
  console.log('  contains:', canRunLocally('contains'));
  console.log('  bleu_score:', canRunLocally('bleu_score'));
  console.log('  groundedness:', canRunLocally('groundedness'), '(requires LLM)');

  // List all available local metrics
  const metrics = getAvailableMetrics();
  console.log('\nAvailable local metrics:', metrics.length);
  console.log('  ' + metrics.slice(0, 5).join(', ') + '...');

  // ============================================================
  // Batch Evaluation
  // ============================================================
  console.log('\n--- Batch Evaluation ---\n');

  const batchResults = evaluator.evaluateBatch([
    { metricName: 'contains', inputs: [{ response: 'Hello World' }], config: { keyword: 'World' } },
    { metricName: 'is_json', inputs: [{ response: '{"valid": true}' }] },
    { metricName: 'one_line', inputs: [{ response: 'Single line' }] },
    { metricName: 'length_less_than', inputs: [{ response: 'Short' }], config: { maxLength: 20 } }
  ]);

  console.log('Batch results:');
  console.log('  Executed locally:', batchResults.executedLocally.join(', '));
  console.log('  Total evaluations:', batchResults.results.eval_results.length);
  console.log('  All passed:', batchResults.results.eval_results.every(r => r?.output === 1));

  // ============================================================
  // Performance Demonstration
  // ============================================================
  console.log('\n--- Performance ---\n');

  const iterations = 1000;
  const start = Date.now();

  for (let i = 0; i < iterations; i++) {
    evaluator.evaluate(
      'contains',
      [{ response: `Test response ${i}` }],
      { keyword: 'response' }
    );
  }

  const elapsed = Date.now() - start;
  console.log(`Ran ${iterations} evaluations in ${elapsed}ms`);
  console.log(`Average: ${(elapsed / iterations).toFixed(2)}ms per evaluation`);
}

// Run the example
main();
