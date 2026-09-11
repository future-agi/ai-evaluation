/**
 * Streaming Evaluator.
 * Evaluates responses in real-time as chunks stream in.
 *
 * @module local/streaming/evaluator
 */

import {
    StreamingEvalConfig,
    StreamingState,
    StreamingEventHandler,
    StreamingEvalEvent,
    ChunkEvent,
    MetricUpdateEvent,
    ThresholdAlertEvent,
    CompleteEvent,
    ErrorEvent,
    canStreamMetric
} from './types';
import { runMetric, hasMetric, MetricResult } from '../metrics';

/**
 * Streaming Evaluator - evaluates responses as they stream in.
 *
 * @example
 * ```typescript
 * const evaluator = new StreamingEvaluator({
 *     metrics: ['contains', 'secrets_detection'],
 *     thresholds: { secrets_detection: 1.0 },
 *     query: 'Write a greeting function'
 * });
 *
 * // Subscribe to events
 * evaluator.on((event) => {
 *     if (event.type === 'threshold_alert') {
 *         console.log('Alert:', event.message);
 *     }
 * });
 *
 * // Feed chunks as they arrive
 * for await (const chunk of streamingResponse) {
 *     evaluator.addChunk(chunk);
 * }
 *
 * // Complete and get final results
 * const results = evaluator.complete();
 * ```
 */
export class StreamingEvaluator {
    private config: Required<StreamingEvalConfig>;
    private state: StreamingState;
    private handlers: StreamingEventHandler[] = [];
    private previousScores: Map<string, number> = new Map();

    constructor(config: StreamingEvalConfig) {
        this.config = {
            metrics: config.metrics,
            minCharsBeforeEval: config.minCharsBeforeEval ?? 50,
            evalFrequency: config.evalFrequency ?? 1,
            thresholds: config.thresholds ?? {},
            emitChunks: config.emitChunks ?? true,
            maxAccumulatedLength: config.maxAccumulatedLength ?? 100000,
            context: config.context ?? '',
            query: config.query ?? '',
            reference: config.reference ?? ''
        };

        this.state = {
            accumulated: '',
            chunkCount: 0,
            startTime: Date.now(),
            lastEvalTime: 0,
            scores: new Map(),
            active: true
        };

        // Validate metrics
        this.validateMetrics();
    }

    /**
     * Validate that all configured metrics exist and can stream
     */
    private validateMetrics(): void {
        for (const metric of this.config.metrics) {
            if (!hasMetric(metric)) {
                console.warn(`Metric '${metric}' not found, will be skipped`);
            }
            if (!canStreamMetric(metric)) {
                console.warn(`Metric '${metric}' may not evaluate well during streaming`);
            }
        }
    }

    /**
     * Register an event handler
     */
    on(handler: StreamingEventHandler): () => void {
        this.handlers.push(handler);

        // Return unsubscribe function
        return () => {
            const index = this.handlers.indexOf(handler);
            if (index > -1) {
                this.handlers.splice(index, 1);
            }
        };
    }

    /**
     * Emit an event to all handlers
     */
    private emit(event: StreamingEvalEvent): void {
        for (const handler of this.handlers) {
            try {
                handler(event);
            } catch (error) {
                console.error('Error in event handler:', error);
            }
        }
    }

    /**
     * Add a chunk of content
     */
    addChunk(chunk: string): void {
        if (!this.state.active) {
            throw new Error('Cannot add chunk to completed stream');
        }

        // Append chunk
        this.state.accumulated += chunk;
        this.state.chunkCount++;

        // Truncate if needed
        if (this.state.accumulated.length > this.config.maxAccumulatedLength) {
            this.state.accumulated = this.state.accumulated.slice(-this.config.maxAccumulatedLength);
        }

        // Emit chunk event
        if (this.config.emitChunks) {
            const chunkEvent: ChunkEvent = {
                type: 'chunk',
                timestamp: Date.now(),
                chunk,
                accumulated: this.state.accumulated,
                chunkCount: this.state.chunkCount
            };
            this.emit(chunkEvent);
        }

        // Check if we should evaluate
        if (this.shouldEvaluate()) {
            this.evaluate();
        }
    }

    /**
     * Determine if we should run evaluation
     */
    private shouldEvaluate(): boolean {
        // Check minimum characters
        if (this.state.accumulated.length < this.config.minCharsBeforeEval) {
            return false;
        }

        // Check frequency
        if (this.state.chunkCount % this.config.evalFrequency !== 0) {
            return false;
        }

        return true;
    }

    /**
     * Run evaluation on current accumulated content
     */
    private evaluate(): void {
        this.state.lastEvalTime = Date.now();

        for (const metricName of this.config.metrics) {
            try {
                // Build config for the metric
                const metricConfig = this.buildMetricConfig(metricName);

                // Run metric
                const result = runMetric(metricName, this.state.accumulated, metricConfig);

                // Get previous score
                const previousScore = this.state.scores.get(metricName);

                // Update score
                this.state.scores.set(metricName, result.score);

                // Emit metric update
                const updateEvent: MetricUpdateEvent = {
                    type: 'metric_update',
                    timestamp: Date.now(),
                    metric: metricName,
                    score: result.score,
                    previousScore,
                    delta: previousScore !== undefined ? result.score - previousScore : undefined,
                    passing: result.passed
                };
                this.emit(updateEvent);

                // Check thresholds
                this.checkThreshold(metricName, result.score, previousScore);

            } catch (error) {
                const errorEvent: ErrorEvent = {
                    type: 'error',
                    timestamp: Date.now(),
                    message: `Error evaluating ${metricName}: ${error instanceof Error ? error.message : 'Unknown error'}`,
                    error: error instanceof Error ? error : undefined
                };
                this.emit(errorEvent);
            }
        }
    }

    /**
     * Build metric configuration from streaming config
     */
    private buildMetricConfig(metricName: string): Record<string, unknown> {
        const config: Record<string, unknown> = {};

        // Add context fields for RAG metrics
        if (this.config.context) {
            config.context = this.config.context;
        }
        if (this.config.query) {
            config.query = this.config.query;
        }
        if (this.config.reference) {
            config.reference = this.config.reference;
        }

        // Response is the accumulated content
        config.response = this.state.accumulated;

        // For code security metrics
        config.code = this.state.accumulated;

        return config;
    }

    /**
     * Check if a threshold was crossed
     */
    private checkThreshold(metricName: string, currentScore: number, previousScore?: number): void {
        const threshold = this.config.thresholds[metricName];
        if (threshold === undefined) {
            return;
        }

        // Check if we crossed the threshold
        const wasAbove = previousScore === undefined || previousScore >= threshold;
        const isAbove = currentScore >= threshold;

        if (wasAbove !== isAbove) {
            const direction = isAbove ? 'above' : 'below';
            const alertEvent: ThresholdAlertEvent = {
                type: 'threshold_alert',
                timestamp: Date.now(),
                metric: metricName,
                score: currentScore,
                threshold,
                direction,
                message: `${metricName} score dropped ${direction} threshold: ${currentScore.toFixed(3)} vs ${threshold}`
            };
            this.emit(alertEvent);
        }
    }

    /**
     * Complete the streaming session and get final results
     */
    complete(): CompleteEvent {
        if (!this.state.active) {
            throw new Error('Stream already completed');
        }

        this.state.active = false;

        // Run final evaluation
        this.evaluate();

        // Build final results
        const finalResults = new Map<string, MetricResult>();
        for (const metricName of this.config.metrics) {
            try {
                const metricConfig = this.buildMetricConfig(metricName);
                const result = runMetric(metricName, this.state.accumulated, metricConfig);
                finalResults.set(metricName, result);
            } catch {
                // Skip failed metrics
            }
        }

        const completeEvent: CompleteEvent = {
            type: 'complete',
            timestamp: Date.now(),
            finalResponse: this.state.accumulated,
            totalChunks: this.state.chunkCount,
            finalResults,
            elapsedMs: Date.now() - this.state.startTime
        };

        this.emit(completeEvent);
        return completeEvent;
    }

    /**
     * Get the current accumulated response
     */
    getAccumulated(): string {
        return this.state.accumulated;
    }

    /**
     * Get current scores
     */
    getScores(): Map<string, number> {
        return new Map(this.state.scores);
    }

    /**
     * Get current state
     */
    getState(): Readonly<StreamingState> {
        return { ...this.state };
    }

    /**
     * Check if the stream is still active
     */
    isActive(): boolean {
        return this.state.active;
    }

    /**
     * Reset the evaluator for reuse
     */
    reset(): void {
        this.state = {
            accumulated: '',
            chunkCount: 0,
            startTime: Date.now(),
            lastEvalTime: 0,
            scores: new Map(),
            active: true
        };
        this.previousScores.clear();
    }
}

/**
 * Create a streaming evaluator with common defaults
 */
export function createStreamingEvaluator(config: StreamingEvalConfig): StreamingEvaluator {
    return new StreamingEvaluator(config);
}

/**
 * Evaluate a stream of chunks (async generator)
 */
export async function evaluateStream(
    chunks: AsyncIterable<string>,
    config: StreamingEvalConfig
): Promise<CompleteEvent> {
    const evaluator = new StreamingEvaluator(config);

    for await (const chunk of chunks) {
        evaluator.addChunk(chunk);
    }

    return evaluator.complete();
}

/**
 * Create an async generator that evaluates chunks and yields events
 */
export async function* streamWithEvaluation(
    chunks: AsyncIterable<string>,
    config: StreamingEvalConfig
): AsyncGenerator<StreamingEvalEvent, CompleteEvent> {
    const evaluator = new StreamingEvaluator({
        ...config,
        emitChunks: true
    });

    const events: StreamingEvalEvent[] = [];
    evaluator.on((event) => events.push(event));

    for await (const chunk of chunks) {
        evaluator.addChunk(chunk);

        // Yield any accumulated events
        while (events.length > 0) {
            yield events.shift()!;
        }
    }

    const completeEvent = evaluator.complete();

    // Yield any remaining events
    while (events.length > 0) {
        yield events.shift()!;
    }

    return completeEvent;
}
