/**
 * Streaming Evaluation Module
 *
 * Evaluate AI responses in real-time as chunks stream in.
 * Provides immediate feedback during generation.
 *
 * @module local/streaming
 *
 * @example
 * ```typescript
 * import { StreamingEvaluator, evaluateStream } from '@future-agi/ai-evaluation/local/streaming';
 *
 * // Option 1: Event-based evaluation
 * const evaluator = new StreamingEvaluator({
 *     metrics: ['contains', 'secrets_detection'],
 *     thresholds: { secrets_detection: 1.0 }
 * });
 *
 * evaluator.on((event) => {
 *     if (event.type === 'threshold_alert') {
 *         console.warn('Security alert:', event.message);
 *     }
 * });
 *
 * // Feed chunks
 * evaluator.addChunk('Hello, ');
 * evaluator.addChunk('World!');
 * const result = evaluator.complete();
 *
 * // Option 2: Evaluate async stream
 * async function evaluateGeneration() {
 *     const response = await fetch('/api/generate');
 *     const reader = response.body.getReader();
 *
 *     const chunks = async function*() {
 *         while (true) {
 *             const { done, value } = await reader.read();
 *             if (done) break;
 *             yield new TextDecoder().decode(value);
 *         }
 *     };
 *
 *     const result = await evaluateStream(chunks(), {
 *         metrics: ['faithfulness'],
 *         context: 'Source material here...'
 *     });
 * }
 * ```
 */

// Types
export {
    StreamingEventType,
    StreamingEvent,
    ChunkEvent,
    MetricUpdateEvent,
    ThresholdAlertEvent,
    CompleteEvent,
    ErrorEvent,
    StreamingEvalEvent,
    StreamingEventHandler,
    StreamingEvalConfig,
    StreamingState,
    STREAMING_CAPABLE_METRICS,
    canStreamMetric
} from './types';

// Evaluator
export {
    StreamingEvaluator,
    createStreamingEvaluator,
    evaluateStream,
    streamWithEvaluation
} from './evaluator';
