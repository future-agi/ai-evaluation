/**
 * Type definitions for streaming evaluation.
 * @module local/streaming/types
 */

import { MetricResult } from '../metrics/types';

/**
 * Event types emitted during streaming evaluation
 */
export type StreamingEventType =
    | 'chunk'           // New chunk received
    | 'metric_update'   // Metric score updated
    | 'threshold_alert' // Threshold crossed
    | 'complete'        // Stream completed
    | 'error';          // Error occurred

/**
 * Base streaming event
 */
export interface StreamingEvent {
    type: StreamingEventType;
    timestamp: number;
}

/**
 * Chunk received event
 */
export interface ChunkEvent extends StreamingEvent {
    type: 'chunk';
    /** The chunk content */
    chunk: string;
    /** Accumulated response so far */
    accumulated: string;
    /** Number of chunks received */
    chunkCount: number;
}

/**
 * Metric update event
 */
export interface MetricUpdateEvent extends StreamingEvent {
    type: 'metric_update';
    /** Metric name */
    metric: string;
    /** Current score */
    score: number;
    /** Previous score (if any) */
    previousScore?: number;
    /** Score change since last update */
    delta?: number;
    /** Whether the metric is currently passing */
    passing: boolean;
}

/**
 * Threshold alert event
 */
export interface ThresholdAlertEvent extends StreamingEvent {
    type: 'threshold_alert';
    /** Metric name */
    metric: string;
    /** Current score */
    score: number;
    /** The threshold that was crossed */
    threshold: number;
    /** Direction of crossing */
    direction: 'above' | 'below';
    /** Alert message */
    message: string;
}

/**
 * Stream complete event
 */
export interface CompleteEvent extends StreamingEvent {
    type: 'complete';
    /** Final accumulated response */
    finalResponse: string;
    /** Total chunks received */
    totalChunks: number;
    /** Final metric results */
    finalResults: Map<string, MetricResult>;
    /** Total time elapsed (ms) */
    elapsedMs: number;
}

/**
 * Error event
 */
export interface ErrorEvent extends StreamingEvent {
    type: 'error';
    /** Error message */
    message: string;
    /** Original error */
    error?: Error;
}

/**
 * All streaming events
 */
export type StreamingEvalEvent =
    | ChunkEvent
    | MetricUpdateEvent
    | ThresholdAlertEvent
    | CompleteEvent
    | ErrorEvent;

/**
 * Event handler type
 */
export type StreamingEventHandler = (event: StreamingEvalEvent) => void;

/**
 * Configuration for streaming evaluation
 */
export interface StreamingEvalConfig {
    /** Metrics to evaluate during streaming */
    metrics: string[];
    /** Minimum characters before first evaluation */
    minCharsBeforeEval?: number;
    /** How often to re-evaluate (in chunks) */
    evalFrequency?: number;
    /** Thresholds for alerts */
    thresholds?: Record<string, number>;
    /** Whether to emit chunk events */
    emitChunks?: boolean;
    /** Maximum accumulated length before truncation */
    maxAccumulatedLength?: number;
    /** Context for RAG metrics */
    context?: string | string[];
    /** Query for Q&A metrics */
    query?: string;
    /** Reference for comparison metrics */
    reference?: string;
}

/**
 * State of a streaming evaluation session
 */
export interface StreamingState {
    /** Accumulated response so far */
    accumulated: string;
    /** Number of chunks received */
    chunkCount: number;
    /** Start timestamp */
    startTime: number;
    /** Last evaluation timestamp */
    lastEvalTime: number;
    /** Current metric scores */
    scores: Map<string, number>;
    /** Is the stream still active */
    active: boolean;
}

/**
 * Metrics that can be evaluated incrementally during streaming
 */
export const STREAMING_CAPABLE_METRICS = new Set([
    // String metrics - can evaluate partial content
    'contains',
    'contains_all',
    'contains_any',
    'contains_none',
    'regex',
    'starts_with',
    'length_less_than',
    'length_greater_than',
    'one_line',

    // JSON metrics - can detect partial JSON
    'contains_json',

    // Code security - can scan partial code
    'secrets_detection',

    // RAG metrics - need enough content but can evaluate early
    'faithfulness',
    'answer_relevance',
]);

/**
 * Check if a metric can be evaluated during streaming
 */
export function canStreamMetric(metricName: string): boolean {
    return STREAMING_CAPABLE_METRICS.has(metricName.toLowerCase());
}
