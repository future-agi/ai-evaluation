/**
 * Integration tests for the local evaluation module.
 * These tests verify the complete flow of local evaluation without mocking.
 */

import {
  LocalEvaluator,
  HybridEvaluator,
  canRunLocally,
  requiresLLM,
  selectExecutionMode,
  getAvailableMetrics,
  ExecutionMode,
} from '../../local';

describe('Local Evaluation Integration Tests', () => {
  let evaluator: LocalEvaluator;

  beforeAll(() => {
    evaluator = new LocalEvaluator();
  });

  describe('LocalEvaluator End-to-End', () => {
    describe('String Metrics Flow', () => {
      it('should evaluate contains metric with various inputs', () => {
        const testCases = [
          { response: 'Hello World', keyword: 'World', caseSensitive: true, expected: 1 },
          { response: 'Hello World', keyword: 'world', caseSensitive: true, expected: 0 }, // case-sensitive
          { response: 'Hello World', keyword: 'world', caseSensitive: false, expected: 1 }, // case-insensitive
          { response: 'Hello World', keyword: 'xyz', caseSensitive: false, expected: 0 },
          { response: '', keyword: 'test', caseSensitive: false, expected: 0 },
        ];

        for (const tc of testCases) {
          const result = evaluator.evaluate(
            'contains',
            [{ response: tc.response }],
            { keyword: tc.keyword, caseSensitive: tc.caseSensitive }
          );
          expect(result.results.eval_results[0]?.output).toBe(tc.expected);
          expect(result.executedLocally).toContain('contains');
          expect(result.executedCloud).toHaveLength(0);
        }
      });

      it('should evaluate regex metric with complex patterns', () => {
        const testCases = [
          { response: 'Email: test@example.com', pattern: '[\\w.+-]+@[\\w-]+\\.[\\w.-]+', expected: 1 },
          { response: 'Phone: 123-456-7890', pattern: '\\d{3}-\\d{3}-\\d{4}', expected: 1 },
          { response: 'No match here', pattern: '\\d{10}', expected: 0 },
          { response: 'URL: https://example.com', pattern: 'https?://[\\w.-]+', expected: 1 },
        ];

        for (const tc of testCases) {
          const result = evaluator.evaluate(
            'regex',
            [{ response: tc.response }],
            { pattern: tc.pattern }
          );
          expect(result.results.eval_results[0]?.output).toBe(tc.expected);
        }
      });

      it('should evaluate contains_all with keyword arrays', () => {
        const result = evaluator.evaluate(
          'contains_all',
          [{ response: 'The quick brown fox jumps over the lazy dog' }],
          { keywords: ['quick', 'fox', 'dog'] }
        );
        expect(result.results.eval_results[0]?.output).toBe(1);

        // contains_all returns a ratio of matched keywords
        const resultPartial = evaluator.evaluate(
          'contains_all',
          [{ response: 'The quick brown fox' }],
          { keywords: ['quick', 'fox', 'elephant'] }
        );
        // 2 out of 3 keywords found = 0.666...
        const score = resultPartial.results.eval_results[0]?.output as number;
        expect(score).toBeGreaterThan(0.6);
        expect(score).toBeLessThan(0.7);
      });

      it('should evaluate length constraints', () => {
        const shortText = 'Hi';
        const mediumText = 'This is a medium length text for testing.';
        const longText = 'A'.repeat(500);

        // length_less_than
        expect(
          evaluator.evaluate('length_less_than', [{ response: shortText }], { maxLength: 10 })
            .results.eval_results[0]?.output
        ).toBe(1);
        expect(
          evaluator.evaluate('length_less_than', [{ response: longText }], { maxLength: 100 })
            .results.eval_results[0]?.output
        ).toBe(0);

        // length_greater_than
        expect(
          evaluator.evaluate('length_greater_than', [{ response: longText }], { minLength: 100 })
            .results.eval_results[0]?.output
        ).toBe(1);
        expect(
          evaluator.evaluate('length_greater_than', [{ response: shortText }], { minLength: 10 })
            .results.eval_results[0]?.output
        ).toBe(0);

        // length_between
        expect(
          evaluator.evaluate('length_between', [{ response: mediumText }], { minLength: 10, maxLength: 100 })
            .results.eval_results[0]?.output
        ).toBe(1);
      });
    });

    describe('JSON Metrics Flow', () => {
      it('should validate JSON correctly', () => {
        const validJson = '{"name": "test", "value": 123}';
        const invalidJson = '{name: test}';
        const jsonInText = 'Here is the data: {"key": "value"}';

        // is_json
        expect(
          evaluator.evaluate('is_json', [{ response: validJson }])
            .results.eval_results[0]?.output
        ).toBe(1);
        expect(
          evaluator.evaluate('is_json', [{ response: invalidJson }])
            .results.eval_results[0]?.output
        ).toBe(0);

        // contains_json
        expect(
          evaluator.evaluate('contains_json', [{ response: jsonInText }])
            .results.eval_results[0]?.output
        ).toBe(1);
      });

      it('should validate against JSON schema', () => {
        const schema = {
          type: 'object',
          properties: {
            name: { type: 'string' },
            age: { type: 'number', minimum: 0 },
            email: { type: 'string', pattern: '^[^@]+@[^@]+$' },
          },
          required: ['name', 'age'],
        };

        const validData = '{"name": "John", "age": 30, "email": "john@example.com"}';
        const missingRequired = '{"name": "John"}';
        const wrongType = '{"name": "John", "age": "thirty"}';
        const invalidPattern = '{"name": "John", "age": 30, "email": "invalid"}';

        expect(
          evaluator.evaluate('json_schema', [{ response: validData }], { schema })
            .results.eval_results[0]?.output
        ).toBe(1);

        expect(
          evaluator.evaluate('json_schema', [{ response: missingRequired }], { schema })
            .results.eval_results[0]?.output
        ).toBe(0);

        expect(
          evaluator.evaluate('json_schema', [{ response: wrongType }], { schema })
            .results.eval_results[0]?.output
        ).toBe(0);

        expect(
          evaluator.evaluate('json_schema', [{ response: invalidPattern }], { schema })
            .results.eval_results[0]?.output
        ).toBe(0);
      });
    });

    describe('Similarity Metrics Flow', () => {
      it('should calculate BLEU scores', () => {
        // Exact match should be high
        const exactResult = evaluator.evaluate(
          'bleu_score',
          [{ response: 'The cat sat on the mat' }],
          { reference: 'The cat sat on the mat' }
        );
        expect(exactResult.results.eval_results[0]?.output).toBeGreaterThan(0.9);

        // Similar should be moderate
        const similarResult = evaluator.evaluate(
          'bleu_score',
          [{ response: 'The cat is sitting on the mat' }],
          { reference: 'The cat sat on the mat' }
        );
        expect(similarResult.results.eval_results[0]?.output).toBeGreaterThan(0.3);

        // Completely different should be low
        const differentResult = evaluator.evaluate(
          'bleu_score',
          [{ response: 'Hello world' }],
          { reference: 'Goodbye moon' }
        );
        expect(differentResult.results.eval_results[0]?.output).toBeLessThan(0.3);
      });

      it('should calculate ROUGE scores', () => {
        const result = evaluator.evaluate(
          'rouge_score',
          [{ response: 'Machine learning is a subset of AI' }],
          { reference: 'Machine learning is artificial intelligence', variant: 'rouge-l' }
        );
        expect(result.results.eval_results[0]?.output).toBeGreaterThan(0);
        expect(result.results.eval_results[0]?.output).toBeLessThanOrEqual(1);
      });

      it('should calculate Levenshtein similarity', () => {
        // Very similar strings
        const similar = evaluator.evaluate(
          'levenshtein_similarity',
          [{ response: 'hello world' }],
          { reference: 'hello word' }
        );
        expect(similar.results.eval_results[0]?.output).toBeGreaterThan(0.8);

        // Completely different
        const different = evaluator.evaluate(
          'levenshtein_similarity',
          [{ response: 'abc' }],
          { reference: 'xyz' }
        );
        expect(different.results.eval_results[0]?.output).toBeLessThan(0.5);
      });

      it('should calculate numeric similarity', () => {
        // Exact match
        const exact = evaluator.evaluate(
          'numeric_similarity',
          [{ response: '3.14159' }],
          { reference: 3.14159, tolerance: 0.001 }
        );
        expect(exact.results.eval_results[0]?.output).toBe(1);

        // Within tolerance
        const within = evaluator.evaluate(
          'numeric_similarity',
          [{ response: '3.14' }],
          { reference: 3.14159, tolerance: 0.01 }
        );
        expect(within.results.eval_results[0]?.output).toBe(1);

        // numeric_similarity returns a continuous similarity score
        // Values further from reference have lower scores
        const different = evaluator.evaluate(
          'numeric_similarity',
          [{ response: '3.0' }],
          { reference: 3.14159, tolerance: 0.01 }
        );
        const score = different.results.eval_results[0]?.output as number;
        expect(score).toBeLessThan(1);
        expect(score).toBeGreaterThan(0.9); // Still relatively close
      });
    });

    describe('Batch Evaluation Flow', () => {
      it('should process multiple evaluations in batch', () => {
        const results = evaluator.evaluateBatch([
          { metricName: 'contains', inputs: [{ response: 'Hello World' }], config: { keyword: 'World' } },
          { metricName: 'is_json', inputs: [{ response: '{"valid": true}' }] },
          { metricName: 'one_line', inputs: [{ response: 'Single line' }] },
          { metricName: 'length_less_than', inputs: [{ response: 'Short' }], config: { maxLength: 20 } },
          { metricName: 'regex', inputs: [{ response: 'test@email.com' }], config: { pattern: '@' } },
        ]);

        expect(results.results.eval_results).toHaveLength(5);
        expect(results.executedLocally).toHaveLength(5);
        expect(results.executedCloud).toHaveLength(0);
        expect(results.errors).toHaveLength(0);

        // All should pass
        results.results.eval_results.forEach((result) => {
          expect(result?.output).toBe(1);
        });
      });

      it('should handle mixed pass/fail in batch', () => {
        const results = evaluator.evaluateBatch([
          { metricName: 'contains', inputs: [{ response: 'Hello' }], config: { keyword: 'World' } }, // fail
          { metricName: 'is_json', inputs: [{ response: 'not json' }] }, // fail
          { metricName: 'one_line', inputs: [{ response: 'Line 1\nLine 2' }] }, // fail
          { metricName: 'contains', inputs: [{ response: 'Hello World' }], config: { keyword: 'World' } }, // pass
        ]);

        expect(results.results.eval_results).toHaveLength(4);
        expect(results.results.eval_results[0]?.output).toBe(0);
        expect(results.results.eval_results[1]?.output).toBe(0);
        expect(results.results.eval_results[2]?.output).toBe(0);
        expect(results.results.eval_results[3]?.output).toBe(1);
      });

      it('should handle errors gracefully in batch', () => {
        const nonStrictEvaluator = new LocalEvaluator({ strictMode: false });
        const results = nonStrictEvaluator.evaluateBatch([
          { metricName: 'contains', inputs: [{ response: 'test' }], config: { keyword: 'test' } },
          { metricName: 'unknown_metric', inputs: [{ response: 'test' }] }, // unknown metric
          { metricName: 'is_json', inputs: [{ response: '{}' }] },
        ]);

        // Unknown metrics are excluded from results but recorded in errors
        expect(results.results.eval_results.length).toBeGreaterThanOrEqual(2);
        expect(results.errors.length).toBeGreaterThan(0);
        expect(results.errors.some((e) => e.metric === 'unknown_metric')).toBe(true);
      });
    });
  });

  describe('Execution Mode Utilities', () => {
    it('should correctly identify local-capable metrics', () => {
      // Local metrics include heuristic implementations
      const localMetrics = ['contains', 'regex', 'is_json', 'bleu_score', 'one_line', 'groundedness', 'hallucination_detection'];
      // These metrics have no local implementation and require cloud
      const cloudMetrics = ['coherence', 'factual_accuracy', 'toxicity'];

      localMetrics.forEach((metric) => {
        expect(canRunLocally(metric)).toBe(true);
      });

      cloudMetrics.forEach((metric) => {
        expect(canRunLocally(metric)).toBe(false);
      });
    });

    it('should correctly identify LLM-required metrics', () => {
      const heuristicMetrics = ['contains', 'regex', 'is_json', 'bleu_score'];
      const llmMetrics = ['groundedness', 'factual_accuracy', 'coherence'];

      heuristicMetrics.forEach((metric) => {
        expect(requiresLLM(metric)).toBe(false);
      });

      llmMetrics.forEach((metric) => {
        expect(requiresLLM(metric)).toBe(true);
      });
    });

    it('should select correct execution mode', () => {
      // Heuristic metrics always run locally (including groundedness which now has local implementation)
      expect(selectExecutionMode('contains', false, true)).toBe(ExecutionMode.LOCAL);
      expect(selectExecutionMode('is_json', true, true)).toBe(ExecutionMode.LOCAL);
      expect(selectExecutionMode('groundedness', false, true)).toBe(ExecutionMode.LOCAL);

      // LLM-only metrics depend on availability
      expect(selectExecutionMode('coherence', false, true)).toBe(ExecutionMode.CLOUD);
      expect(selectExecutionMode('coherence', true, true)).toBe(ExecutionMode.LOCAL);
      expect(selectExecutionMode('coherence', true, false)).toBe(ExecutionMode.CLOUD);
    });

    it('should list all available metrics', () => {
      const metrics = getAvailableMetrics();
      expect(metrics.length).toBeGreaterThan(0);
      expect(metrics).toContain('contains');
      expect(metrics).toContain('is_json');
      expect(metrics).toContain('bleu_score');
    });
  });

  describe('HybridEvaluator Integration', () => {
    it('should route heuristic metrics locally without LLM', async () => {
      const hybrid = new HybridEvaluator({
        preferLocal: true,
        fallbackToCloud: false,
        offlineMode: true,
      });

      const result = await hybrid.evaluate(
        'contains',
        [{ response: 'Hello World' }],
        { keyword: 'World' }
      );

      expect(result.results.eval_results[0]?.output).toBe(1);
      expect(result.executedLocally).toContain('contains');
      expect(result.executedCloud).toHaveLength(0);
    });

    it('should partition evaluations correctly', () => {
      // Create hybrid with cloud evaluator to test cloud routing
      const hybridWithCloud = new HybridEvaluator({
        preferLocal: true,
        fallbackToCloud: true,
        // Note: Without actual cloud evaluator, LLM metrics fall back to local
      });

      const evaluations = [
        { metricName: 'contains', inputs: [{ response: 'test' }], config: { keyword: 'test' } },
        { metricName: 'is_json', inputs: [{ response: '{}' }] },
        { metricName: 'groundedness', inputs: [{ response: 'test', context: 'ctx' }] },
        { metricName: 'bleu_score', inputs: [{ response: 'test' }], config: { reference: 'test' } },
      ];

      const partitioned = hybridWithCloud.partitionEvaluations(evaluations);

      // Heuristic metrics should be local
      const localMetrics = partitioned.local?.map((e) => e.metricName) || [];
      expect(localMetrics).toContain('contains');
      expect(localMetrics).toContain('is_json');
      expect(localMetrics).toContain('bleu_score');

      // Without cloudEvaluator instance, LLM metrics also fall back to local
      // This is expected behavior - the routing logic returns LOCAL when no cloud is available
      expect(localMetrics).toContain('groundedness');
    });

    it('should route evaluation correctly', () => {
      const hybrid = new HybridEvaluator({ preferLocal: true });

      // Heuristic metrics always route to local
      expect(hybrid.routeEvaluation('contains')).toBe(ExecutionMode.LOCAL);
      expect(hybrid.routeEvaluation('is_json')).toBe(ExecutionMode.LOCAL);

      // LLM metrics without cloud evaluator fall back to local
      // (the implementation returns LOCAL when no cloud is available)
      expect(hybrid.routeEvaluation('groundedness')).toBe(ExecutionMode.LOCAL);

      // Force local
      expect(hybrid.routeEvaluation('groundedness', true, false)).toBe(ExecutionMode.LOCAL);

      // Force cloud works when not in offline mode
      const hybridNotOffline = new HybridEvaluator({ preferLocal: true, offlineMode: false });
      expect(hybridNotOffline.routeEvaluation('contains', false, true)).toBe(ExecutionMode.CLOUD);
    });

    it('should check local LLM capability', () => {
      const hybridNoLLM = new HybridEvaluator({ preferLocal: true });
      expect(hybridNoLLM.canUseLocalLLM('groundedness')).toBe(false);
    });
  });

  describe('Error Handling', () => {
    it('should throw in strict mode for non-local metrics', () => {
      const strictEvaluator = new LocalEvaluator({ strictMode: true });

      expect(() => {
        strictEvaluator.evaluate('coherence', [{ response: 'test' }]);  // Changed from 'groundedness'
      }).toThrow();
    });

    it('should return error in non-strict mode for non-local metrics', () => {
      const nonStrictEvaluator = new LocalEvaluator({ strictMode: false });

      const result = nonStrictEvaluator.evaluate('coherence', [{ response: 'test' }]);  // Changed from 'groundedness'
      expect(result.errors.length).toBeGreaterThan(0);
      expect(result.errors[0].metric).toBe('coherence');
    });

    it('should handle empty input gracefully', () => {
      const result = evaluator.evaluate('contains', [{ response: '' }], { keyword: 'test' });
      expect(result.results.eval_results[0]?.output).toBe(0);
    });

    it('should handle special characters in input', () => {
      const specialChars = 'Test with special chars: <>&"\'`\n\t\r';
      const result = evaluator.evaluate('contains', [{ response: specialChars }], { keyword: '<>' });
      expect(result.results.eval_results[0]?.output).toBe(1);
    });
  });

  describe('Performance', () => {
    it('should handle large batch efficiently', () => {
      const batchSize = 100;
      const batch = Array(batchSize).fill(null).map((_, i) => ({
        metricName: 'contains',
        inputs: [{ response: `Test response ${i}` }],
        config: { keyword: 'response' },
      }));

      const start = Date.now();
      const results = evaluator.evaluateBatch(batch);
      const elapsed = Date.now() - start;

      expect(results.results.eval_results).toHaveLength(batchSize);
      expect(elapsed).toBeLessThan(5000); // Should complete in under 5 seconds
    });

    it('should handle long strings', () => {
      const longString = 'A'.repeat(10000) + 'target' + 'B'.repeat(10000);

      const result = evaluator.evaluate('contains', [{ response: longString }], { keyword: 'target' });
      expect(result.results.eval_results[0]?.output).toBe(1);
    });
  });
});
