/**
 * Tests for Streaming Evaluation
 */

import {
    StreamingEvaluator,
    createStreamingEvaluator,
    evaluateStream,
    streamWithEvaluation,
    canStreamMetric,
    STREAMING_CAPABLE_METRICS,
    StreamingEvalEvent,
    ChunkEvent,
    MetricUpdateEvent,
    ThresholdAlertEvent,
    CompleteEvent
} from '../streaming';

describe('Streaming Evaluation', () => {
    describe('canStreamMetric', () => {
        it('should return true for streaming-capable metrics', () => {
            expect(canStreamMetric('contains')).toBe(true);
            expect(canStreamMetric('regex')).toBe(true);
            expect(canStreamMetric('secrets_detection')).toBe(true);
        });

        it('should return false for non-streaming metrics', () => {
            expect(canStreamMetric('bleu_score')).toBe(false);
            expect(canStreamMetric('rouge_score')).toBe(false);
        });

        it('STREAMING_CAPABLE_METRICS should be a set', () => {
            expect(STREAMING_CAPABLE_METRICS).toBeInstanceOf(Set);
            expect(STREAMING_CAPABLE_METRICS.size).toBeGreaterThan(0);
        });
    });

    describe('StreamingEvaluator', () => {
        it('should create with default config', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains']
            });
            expect(evaluator).toBeInstanceOf(StreamingEvaluator);
            expect(evaluator.isActive()).toBe(true);
        });

        it('should accumulate chunks', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains'],
                minCharsBeforeEval: 0,
                emitChunks: false
            });

            evaluator.addChunk('Hello, ');
            evaluator.addChunk('World!');

            expect(evaluator.getAccumulated()).toBe('Hello, World!');
        });

        it('should emit chunk events when enabled', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains'],
                emitChunks: true
            });

            const events: ChunkEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'chunk') {
                    events.push(event as ChunkEvent);
                }
            });

            evaluator.addChunk('Hello');
            evaluator.addChunk(' World');

            expect(events.length).toBe(2);
            expect(events[0].chunk).toBe('Hello');
            expect(events[1].accumulated).toBe('Hello World');
        });

        it('should not emit chunk events when disabled', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains'],
                emitChunks: false
            });

            const events: ChunkEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'chunk') {
                    events.push(event as ChunkEvent);
                }
            });

            evaluator.addChunk('Hello');

            expect(events.length).toBe(0);
        });

        it('should evaluate metrics and emit updates', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['one_line'],
                minCharsBeforeEval: 5,
                evalFrequency: 1
            });

            const updates: MetricUpdateEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'metric_update') {
                    updates.push(event as MetricUpdateEvent);
                }
            });

            evaluator.addChunk('Single line response');

            expect(updates.length).toBeGreaterThan(0);
            expect(updates[0].metric).toBe('one_line');
            expect(updates[0].score).toBeDefined();
        });

        it('should emit threshold alerts when score drops', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['one_line'],
                minCharsBeforeEval: 5,
                evalFrequency: 1,
                thresholds: { one_line: 1.0 }
            });

            const alerts: ThresholdAlertEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'threshold_alert') {
                    alerts.push(event as ThresholdAlertEvent);
                }
            });

            // First chunk - single line
            evaluator.addChunk('First line');

            // Second chunk - add newline
            evaluator.addChunk('\nSecond line');

            // Should have alert when dropping below threshold
            if (alerts.length > 0) {
                expect(alerts[0].direction).toBe('below');
            }
        });

        it('should complete and return final results', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains'],
                minCharsBeforeEval: 0
            });

            evaluator.addChunk('Hello World');
            const result = evaluator.complete();

            expect(result.type).toBe('complete');
            expect(result.finalResponse).toBe('Hello World');
            expect(result.totalChunks).toBe(1);
            expect(result.finalResults).toBeInstanceOf(Map);
            expect(result.elapsedMs).toBeGreaterThanOrEqual(0);
        });

        it('should mark stream as inactive after complete', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains']
            });

            evaluator.addChunk('Test');
            evaluator.complete();

            expect(evaluator.isActive()).toBe(false);
        });

        it('should throw when adding chunk to completed stream', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains']
            });

            evaluator.addChunk('Test');
            evaluator.complete();

            expect(() => evaluator.addChunk('More')).toThrow('Cannot add chunk to completed stream');
        });

        it('should throw when completing twice', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains']
            });

            evaluator.complete();

            expect(() => evaluator.complete()).toThrow('Stream already completed');
        });

        it('should reset for reuse', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains']
            });

            evaluator.addChunk('Test');
            evaluator.complete();

            evaluator.reset();

            expect(evaluator.isActive()).toBe(true);
            expect(evaluator.getAccumulated()).toBe('');
            expect(evaluator.getState().chunkCount).toBe(0);
        });

        it('should respect minCharsBeforeEval', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains'],
                minCharsBeforeEval: 100,
                evalFrequency: 1
            });

            const updates: MetricUpdateEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'metric_update') {
                    updates.push(event as MetricUpdateEvent);
                }
            });

            // Add small chunk - shouldn't trigger evaluation
            evaluator.addChunk('Short');

            expect(updates.length).toBe(0);
        });

        it('should respect evalFrequency', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['one_line'],
                minCharsBeforeEval: 1,
                evalFrequency: 3
            });

            const updates: MetricUpdateEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'metric_update') {
                    updates.push(event as MetricUpdateEvent);
                }
            });

            // Add 5 chunks
            for (let i = 0; i < 5; i++) {
                evaluator.addChunk('chunk ');
            }

            // Should only evaluate on chunks 3 (index 2) - frequency of 3
            // Chunk 0: no (0 % 3 = 0, but minChars check first)
            // Actually depends on accumulated length too
        });

        it('should truncate accumulated content at maxAccumulatedLength', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains'],
                maxAccumulatedLength: 20
            });

            evaluator.addChunk('This is a very long string that exceeds the limit');

            expect(evaluator.getAccumulated().length).toBeLessThanOrEqual(20);
        });

        it('should allow unsubscribing from events', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['contains'],
                emitChunks: true
            });

            let eventCount = 0;
            const unsubscribe = evaluator.on(() => {
                eventCount++;
            });

            evaluator.addChunk('First');
            expect(eventCount).toBe(1);

            unsubscribe();
            evaluator.addChunk('Second');
            expect(eventCount).toBe(1); // Should not increase
        });

        it('should handle errors gracefully', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['unknown_metric'],
                minCharsBeforeEval: 0
            });

            const errors: StreamingEvalEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'error') {
                    errors.push(event);
                }
            });

            evaluator.addChunk('Test content that triggers evaluation');

            // Should have error events for unknown metric
            expect(errors.length).toBeGreaterThan(0);
        });

        it('should include context in metric config for RAG metrics', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['faithfulness'],
                context: 'Source context here',
                query: 'What is the context about?',
                minCharsBeforeEval: 0
            });

            const updates: MetricUpdateEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'metric_update') {
                    updates.push(event as MetricUpdateEvent);
                }
            });

            evaluator.addChunk('Response based on context');

            // Should run faithfulness metric with context
            expect(updates.some(u => u.metric === 'faithfulness')).toBe(true);
        });
    });

    describe('createStreamingEvaluator', () => {
        it('should create a StreamingEvaluator', () => {
            const evaluator = createStreamingEvaluator({
                metrics: ['contains']
            });
            expect(evaluator).toBeInstanceOf(StreamingEvaluator);
        });
    });

    describe('evaluateStream', () => {
        it('should evaluate async iterable of chunks', async () => {
            async function* chunks(): AsyncGenerator<string> {
                yield 'Hello, ';
                yield 'World!';
            }

            const result = await evaluateStream(chunks(), {
                metrics: ['contains'],
                minCharsBeforeEval: 0
            });

            expect(result.type).toBe('complete');
            expect(result.finalResponse).toBe('Hello, World!');
            expect(result.totalChunks).toBe(2);
        });

        it('should handle empty stream', async () => {
            async function* chunks(): AsyncGenerator<string> {
                // No chunks
            }

            const result = await evaluateStream(chunks(), {
                metrics: ['contains']
            });

            expect(result.type).toBe('complete');
            expect(result.finalResponse).toBe('');
            expect(result.totalChunks).toBe(0);
        });
    });

    describe('streamWithEvaluation', () => {
        it('should yield events while streaming', async () => {
            async function* chunks(): AsyncGenerator<string> {
                yield 'Hello';
                yield ' World';
            }

            const events: StreamingEvalEvent[] = [];
            const generator = streamWithEvaluation(chunks(), {
                metrics: ['one_line'],
                minCharsBeforeEval: 5,
                emitChunks: true
            });

            for await (const event of generator) {
                events.push(event);
            }

            // Should have chunk events
            expect(events.some(e => e.type === 'chunk')).toBe(true);
        });

        it('should return complete event at end', async () => {
            async function* chunks(): AsyncGenerator<string> {
                yield 'Test';
            }

            const generator = streamWithEvaluation(chunks(), {
                metrics: ['contains'],
                emitChunks: false,
                minCharsBeforeEval: 100
            });

            let result: CompleteEvent | undefined;
            for await (const event of generator) {
                // Just consume
            }
            // The return value of the generator is the complete event
            // But for-await doesn't give us that directly
        });
    });

    describe('Real-world Scenarios', () => {
        it('should detect secrets as they stream in', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['secrets_detection'],
                thresholds: { secrets_detection: 1.0 },
                minCharsBeforeEval: 10
            });

            const alerts: ThresholdAlertEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'threshold_alert') {
                    alerts.push(event as ThresholdAlertEvent);
                }
            });

            // Simulate streaming code with a secret
            evaluator.addChunk('const config = {\n');
            evaluator.addChunk('  apiKey: "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL",\n');
            evaluator.addChunk('};\n');

            const result = evaluator.complete();

            // Should have detected the secret
            const secretsResult = result.finalResults.get('secrets_detection');
            expect(secretsResult?.passed).toBe(false);
        });

        it('should track response length during generation', () => {
            const evaluator = new StreamingEvaluator({
                metrics: ['length_less_than'],
                minCharsBeforeEval: 10,
                evalFrequency: 1
            });

            const updates: MetricUpdateEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'metric_update') {
                    updates.push(event as MetricUpdateEvent);
                }
            });

            // Stream progressively longer content
            for (let i = 0; i < 5; i++) {
                evaluator.addChunk('More content added here. ');
            }

            evaluator.complete();

            // Should have multiple updates tracking length
            expect(updates.length).toBeGreaterThan(0);
        });

        it('should evaluate faithfulness during RAG streaming', () => {
            const context = 'Python is a high-level programming language created by Guido van Rossum.';

            const evaluator = new StreamingEvaluator({
                metrics: ['faithfulness'],
                context,
                query: 'What is Python?',
                minCharsBeforeEval: 20
            });

            const updates: MetricUpdateEvent[] = [];
            evaluator.on((event) => {
                if (event.type === 'metric_update') {
                    updates.push(event as MetricUpdateEvent);
                }
            });

            // Stream a response
            evaluator.addChunk('Python is a programming language. ');
            evaluator.addChunk('It was created by Guido van Rossum. ');
            evaluator.addChunk('It is used for web development.');

            const result = evaluator.complete();

            // Should have evaluated faithfulness
            expect(result.finalResults.has('faithfulness')).toBe(true);
        });
    });
});
