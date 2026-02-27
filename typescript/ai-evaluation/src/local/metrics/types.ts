/**
 * Types for local metric evaluation.
 */

/**
 * Result from a metric evaluation
 */
export interface MetricResult {
    score: number;
    passed: boolean;
    reason: string;
}

/**
 * Base configuration for metrics
 */
export interface MetricConfig {
    [key: string]: any;
}

/**
 * Input for metric evaluation
 */
export interface MetricInput {
    response: string;
    expectedResponse?: string;
    query?: string;
    context?: string;
    [key: string]: any;
}

/**
 * Metric function signature
 */
export type MetricFunction = (
    response: string,
    config?: MetricConfig
) => MetricResult;

/**
 * Registry entry for a metric
 */
export interface MetricRegistryEntry {
    name: string;
    fn: MetricFunction;
    description: string;
    requiredConfig?: string[];
}
