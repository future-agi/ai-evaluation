/**
 * Tests for LocalEvaluator and HybridEvaluator
 */

import { LocalEvaluator, HybridEvaluator } from '../evaluator';
import { ExecutionMode } from '../execution-mode';

describe('LocalEvaluator', () => {
    let evaluator: LocalEvaluator;

    beforeEach(() => {
        evaluator = new LocalEvaluator();
    });

    describe('canRunLocally', () => {
        it('should return true for heuristic metrics', () => {
            expect(evaluator.canRunLocally('contains')).toBe(true);
            expect(evaluator.canRunLocally('regex')).toBe(true);
            expect(evaluator.canRunLocally('is_json')).toBe(true);
            expect(evaluator.canRunLocally('bleu_score')).toBe(true);
        });

        it('should return false for LLM-only metrics', () => {
            // These metrics require LLM and have no local heuristic implementation
            expect(evaluator.canRunLocally('coherence')).toBe(false);
            expect(evaluator.canRunLocally('fluency')).toBe(false);
            expect(evaluator.canRunLocally('toxicity')).toBe(false);
        });

        it('should return true for metrics with local heuristic implementation', () => {
            // These metrics have local heuristic implementations now
            expect(evaluator.canRunLocally('groundedness')).toBe(true);
            expect(evaluator.canRunLocally('hallucination_detection')).toBe(true);
        });
    });

    describe('evaluate', () => {
        it('should evaluate contains metric', () => {
            const result = evaluator.evaluate(
                'contains',
                [{ response: 'Hello world' }],
                { keyword: 'world' }
            );

            expect(result.executedLocally).toContain('contains');
            expect(result.executedCloud).toHaveLength(0);
            expect(result.errors).toHaveLength(0);
            expect(result.results.eval_results).toHaveLength(1);
            expect(result.results.eval_results[0]?.output).toBe(1.0);
        });

        it('should evaluate regex metric', () => {
            const result = evaluator.evaluate(
                'regex',
                [{ response: 'User ID: 12345' }],
                { pattern: '\\d+' }
            );

            expect(result.results.eval_results[0]?.output).toBe(1.0);
        });

        it('should evaluate is_json metric', () => {
            const result = evaluator.evaluate(
                'is_json',
                [{ response: '{"key": "value"}' }]
            );

            expect(result.results.eval_results[0]?.output).toBe(1.0);
        });

        it('should handle multiple inputs', () => {
            const result = evaluator.evaluate(
                'contains',
                [
                    { response: 'Hello world' },
                    { response: 'Goodbye world' },
                    { response: 'Hello there' }
                ],
                { keyword: 'world' }
            );

            expect(result.results.eval_results).toHaveLength(3);
            expect(result.results.eval_results[0]?.output).toBe(1.0);
            expect(result.results.eval_results[1]?.output).toBe(1.0);
            expect(result.results.eval_results[2]?.output).toBe(0.0);
        });

        it('should return error for non-local metrics', () => {
            const result = evaluator.evaluate(
                'coherence',  // Changed from 'groundedness' which now has local implementation
                [{ response: 'test' }]
            );

            expect(result.errors).toHaveLength(1);
            expect(result.errors[0]?.metric).toBe('coherence');
            expect(result.executedLocally).toHaveLength(0);
        });

        it('should throw in strict mode for non-local metrics', () => {
            const strictEvaluator = new LocalEvaluator({ strictMode: true });

            expect(() => {
                strictEvaluator.evaluate('coherence', [{ response: 'test' }]);
            }).toThrow("Metric 'coherence' cannot run locally");
        });

        it('should handle evaluation errors gracefully', () => {
            const result = evaluator.evaluate(
                'regex',
                [{ response: 'test' }],
                { pattern: '[invalid' } // Invalid regex
            );

            expect(result.errors).toHaveLength(1);
            expect(result.results.eval_results[0]?.output).toBeNull();
        });

        it('should extract response from different input keys', () => {
            const result1 = evaluator.evaluate(
                'contains',
                [{ output: 'Hello world' }],
                { keyword: 'world' }
            );
            expect(result1.results.eval_results[0]?.output).toBe(1.0);

            const result2 = evaluator.evaluate(
                'contains',
                [{ text: 'Hello world' }],
                { keyword: 'world' }
            );
            expect(result2.results.eval_results[0]?.output).toBe(1.0);
        });
    });

    describe('evaluateBatch', () => {
        it('should evaluate multiple metrics in batch', () => {
            const result = evaluator.evaluateBatch([
                {
                    metricName: 'contains',
                    inputs: [{ response: 'Hello world' }],
                    config: { keyword: 'world' }
                },
                {
                    metricName: 'is_json',
                    inputs: [{ response: '{"valid": true}' }]
                }
            ]);

            expect(result.results.eval_results).toHaveLength(2);
            expect(result.executedLocally).toContain('contains');
            expect(result.executedLocally).toContain('is_json');
        });

        it('should deduplicate executed metrics', () => {
            const result = evaluator.evaluateBatch([
                {
                    metricName: 'contains',
                    inputs: [{ response: 'test1' }],
                    config: { keyword: 'test1' }
                },
                {
                    metricName: 'contains',
                    inputs: [{ response: 'test2' }],
                    config: { keyword: 'test2' }
                }
            ]);

            expect(result.executedLocally).toEqual(['contains']);
        });
    });
});

describe('HybridEvaluator', () => {
    describe('routeEvaluation', () => {
        it('should route heuristic metrics to local', () => {
            const hybrid = new HybridEvaluator();
            expect(hybrid.routeEvaluation('contains')).toBe(ExecutionMode.LOCAL);
            expect(hybrid.routeEvaluation('is_json')).toBe(ExecutionMode.LOCAL);
        });

        it('should route LLM metrics to local when no cloud available', () => {
            const hybrid = new HybridEvaluator({ offlineMode: true });
            expect(hybrid.routeEvaluation('groundedness')).toBe(ExecutionMode.LOCAL);
        });

        it('should respect forceLocal option', () => {
            const hybrid = new HybridEvaluator();
            expect(hybrid.routeEvaluation('groundedness', true)).toBe(ExecutionMode.LOCAL);
        });

        it('should respect forceCloud option when not in offline mode', () => {
            const hybrid = new HybridEvaluator();
            expect(hybrid.routeEvaluation('contains', false, true)).toBe(ExecutionMode.CLOUD);
        });

        it('should ignore forceCloud in offline mode', () => {
            const hybrid = new HybridEvaluator({ offlineMode: true });
            expect(hybrid.routeEvaluation('contains', false, true)).toBe(ExecutionMode.LOCAL);
        });
    });

    describe('canUseLocalLLM', () => {
        it('should return false when no local LLM configured', () => {
            const hybrid = new HybridEvaluator();
            expect(hybrid.canUseLocalLLM('groundedness')).toBe(false);
        });

        it('should return false for non-LLM metrics', () => {
            const hybrid = new HybridEvaluator();
            expect(hybrid.canUseLocalLLM('contains')).toBe(false);
        });
    });

    describe('partitionEvaluations', () => {
        it('should partition evaluations by execution mode', () => {
            const hybrid = new HybridEvaluator();
            const partitions = hybrid.partitionEvaluations([
                { metricName: 'contains', inputs: [{ response: 'test' }], config: { keyword: 'test' } },
                { metricName: 'is_json', inputs: [{ response: '{}' }] },
                { metricName: 'groundedness', inputs: [{ response: 'test' }] }
            ]);

            // Without a cloud evaluator, all metrics fall through to LOCAL
            // Heuristic metrics go to LOCAL directly, LLM metrics fall back to LOCAL
            expect(partitions[ExecutionMode.LOCAL]).toHaveLength(3);
            expect(partitions[ExecutionMode.LOCAL]?.map(e => e.metricName)).toContain('contains');
            expect(partitions[ExecutionMode.LOCAL]?.map(e => e.metricName)).toContain('is_json');
            expect(partitions[ExecutionMode.LOCAL]?.map(e => e.metricName)).toContain('groundedness');
        });
    });

    describe('evaluateLocalPartition', () => {
        it('should evaluate local partition using LocalEvaluator', () => {
            const hybrid = new HybridEvaluator();
            const result = hybrid.evaluateLocalPartition([
                { metricName: 'contains', inputs: [{ response: 'Hello world' }], config: { keyword: 'world' } }
            ]);

            expect(result.results.eval_results).toHaveLength(1);
            expect(result.results.eval_results[0]?.output).toBe(1.0);
        });
    });

    describe('evaluate', () => {
        it('should evaluate heuristic metrics locally', async () => {
            const hybrid = new HybridEvaluator();
            const result = await hybrid.evaluate(
                'contains',
                [{ response: 'Hello world' }],
                { keyword: 'world' }
            );

            expect(result.executedLocally).toContain('contains');
            expect(result.results.eval_results[0]?.output).toBe(1.0);
        });

        it('should return error for LLM-only metrics without local LLM', async () => {
            const hybrid = new HybridEvaluator({ offlineMode: true });
            const result = await hybrid.evaluate(
                'coherence',  // Changed from 'groundedness' which now has local heuristic
                [{ response: 'test' }]
            );

            expect(result.errors).toHaveLength(1);
        });
    });
});
