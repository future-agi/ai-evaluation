/**
 * Example 4: Ollama LLM-as-Judge
 *
 * This example demonstrates how to use a local Ollama LLM to evaluate
 * responses using LLM-as-judge techniques. Run evaluations completely
 * offline with no API costs.
 *
 * Prerequisites:
 * 1. Install Ollama: https://ollama.ai/
 * 2. Pull a model: ollama pull llama3.2
 * 3. Start Ollama: ollama serve
 *
 * Run: npx ts-node examples/04-ollama-llm-judge.ts
 */

import { OllamaLLM, LocalLLMFactory } from '../src/local';

async function main() {
  console.log('=== Ollama LLM-as-Judge ===\n');

  // ============================================================
  // Setup: Create OllamaLLM client
  // ============================================================
  console.log('--- Setting up Ollama LLM ---\n');

  // Option 1: Direct instantiation
  const llm = new OllamaLLM({
    model: 'llama3.2',              // Model name (must be pulled first)
    baseUrl: 'http://localhost:11434', // Default Ollama URL
    temperature: 0.0,               // 0 = deterministic outputs
    maxTokens: 1024,                // Max response length
    timeout: 120                    // Timeout in seconds
  });

  // Option 2: Using the factory
  // const llm = LocalLLMFactory.fromString('ollama/llama3.2');
  // const llm = LocalLLMFactory.createDefault(); // Uses llama3.2

  // Check if Ollama is available
  const isAvailable = await llm.isAvailable();
  console.log('Ollama available:', isAvailable);

  if (!isAvailable) {
    console.log('\nOllama is not running. Please:');
    console.log('1. Install Ollama from https://ollama.ai/');
    console.log('2. Run: ollama pull llama3.2');
    console.log('3. Run: ollama serve');
    console.log('\nExiting example.');
    return;
  }

  // List available models
  const models = await llm.listModels();
  console.log('Available models:', models.join(', '));

  // ============================================================
  // Basic Text Generation
  // ============================================================
  console.log('\n--- Basic Text Generation ---\n');

  // Simple prompt
  const simpleResponse = await llm.generate('What is 2 + 2? Answer in one word.');
  console.log('Simple generation:', simpleResponse.trim());

  // With system prompt
  const systemResponse = await llm.generate(
    'Explain quantum computing',
    {
      system: 'You are a science teacher. Explain concepts simply in 2 sentences.',
      temperature: 0.3
    }
  );
  console.log('\nWith system prompt:', systemResponse.trim());

  // ============================================================
  // Chat Completion
  // ============================================================
  console.log('\n--- Chat Completion ---\n');

  const chatResponse = await llm.chat([
    { role: 'system', content: 'You are a helpful coding assistant. Be concise.' },
    { role: 'user', content: 'Write a Python hello world' },
    { role: 'assistant', content: 'print("Hello, World!")' },
    { role: 'user', content: 'Now in JavaScript' }
  ]);
  console.log('Chat response:', chatResponse.trim());

  // ============================================================
  // LLM-as-Judge: Single Evaluation
  // ============================================================
  console.log('\n--- LLM-as-Judge: Single Evaluation ---\n');

  // Evaluate factual accuracy
  const factualResult = await llm.judge(
    'What is the capital of France?',                    // Query
    'The capital of France is Paris, known for the Eiffel Tower.', // Response
    'Evaluate factual accuracy. Score 0-1 based on correctness.', // Criteria
    'Paris is the capital city of France.'               // Context (optional)
  );

  console.log('Factual accuracy evaluation:');
  console.log('  Score:', factualResult.score.toFixed(2));
  console.log('  Passed:', factualResult.passed);
  console.log('  Reason:', factualResult.reason);

  // Evaluate relevance
  const relevanceResult = await llm.judge(
    'How do I make coffee?',
    'Coffee is a beverage made from roasted coffee beans. The beans are ground and brewed with hot water.',
    'Evaluate if the response directly answers the question. Score 0-1.'
  );

  console.log('\nRelevance evaluation:');
  console.log('  Score:', relevanceResult.score.toFixed(2));
  console.log('  Passed:', relevanceResult.passed);
  console.log('  Reason:', relevanceResult.reason);

  // Evaluate completeness
  const completenessResult = await llm.judge(
    'What are the three primary colors?',
    'Red and blue are primary colors.',
    'Evaluate completeness. Score based on whether all information is provided. Penalize missing items.'
  );

  console.log('\nCompleteness evaluation:');
  console.log('  Score:', completenessResult.score.toFixed(2));
  console.log('  Passed:', completenessResult.passed);
  console.log('  Reason:', completenessResult.reason);

  // ============================================================
  // LLM-as-Judge: Custom Criteria
  // ============================================================
  console.log('\n--- Custom Evaluation Criteria ---\n');

  // Code quality evaluation
  const codeResult = await llm.judge(
    'Write a function to add two numbers',
    `function add(a, b) {
  return a + b;
}`,
    `Evaluate the code quality based on:
1. Correctness (does it work?)
2. Readability (is it clear?)
3. Best practices (proper naming, etc.)
Score 0-1 where 1 is excellent.`
  );

  console.log('Code quality evaluation:');
  console.log('  Score:', codeResult.score.toFixed(2));
  console.log('  Reason:', codeResult.reason);

  // Tone evaluation
  const toneResult = await llm.judge(
    'Customer complaint about late delivery',
    'We sincerely apologize for the delay in your order. We understand this is frustrating and have escalated your case to our shipping team. You should receive an update within 24 hours.',
    `Evaluate the response tone for customer service. Consider:
- Empathy and acknowledgment
- Professionalism
- Action-oriented response
Score 0-1.`
  );

  console.log('\nCustomer service tone:');
  console.log('  Score:', toneResult.score.toFixed(2));
  console.log('  Reason:', toneResult.reason);

  // ============================================================
  // Batch Evaluation
  // ============================================================
  console.log('\n--- Batch Evaluation ---\n');

  const batchEvaluations = [
    {
      query: 'What is 2+2?',
      response: 'The answer is 4.',
      criteria: 'Evaluate correctness. Score 0-1.'
    },
    {
      query: 'Explain gravity',
      response: 'Gravity is a force that attracts objects with mass toward each other.',
      criteria: 'Evaluate scientific accuracy. Score 0-1.'
    },
    {
      query: 'What is Python?',
      response: 'Python is a high-level programming language known for its readability.',
      criteria: 'Evaluate accuracy and helpfulness. Score 0-1.',
      context: 'Python was created by Guido van Rossum in 1991.'
    }
  ];

  console.log(`Running ${batchEvaluations.length} evaluations...`);
  const startTime = Date.now();

  const batchResults = await llm.batchJudge(batchEvaluations);

  const elapsed = (Date.now() - startTime) / 1000;
  console.log(`Completed in ${elapsed.toFixed(2)}s\n`);

  console.log('Batch results:');
  batchResults.forEach((result, index) => {
    console.log(`  ${index + 1}. Score: ${result.score.toFixed(2)}, Passed: ${result.passed}`);
    console.log(`     Reason: ${result.reason.substring(0, 60)}...`);
  });

  // ============================================================
  // Error Handling
  // ============================================================
  console.log('\n--- Error Handling ---\n');

  // Test with edge case
  try {
    const edgeResult = await llm.judge(
      '',  // Empty query
      'This is a response',
      'Evaluate the response'
    );
    console.log('Empty query handled:', edgeResult.score.toFixed(2));
  } catch (error) {
    console.log('Empty query error:', (error as Error).message);
  }

  // ============================================================
  // Performance Tips
  // ============================================================
  console.log('\n--- Performance Tips ---\n');

  console.log('1. Use temperature=0 for consistent judgments');
  console.log('2. Keep criteria clear and specific');
  console.log('3. Provide context when available');
  console.log('4. Use smaller models (phi3, llama3.2) for faster evals');
  console.log('5. Batch evaluations when possible');

  // Show available model options
  console.log('\nRecommended models for evaluation:');
  console.log('  - llama3.2 (default) - Good balance of speed and quality');
  console.log('  - phi3 - Very fast, good for simple evaluations');
  console.log('  - mistral - Good reasoning capabilities');
  console.log('  - qwen2.5 - Strong instruction following');
}

// Run the example
main().catch(console.error);
