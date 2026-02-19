/**
 * Type definitions for AutoEval pipeline.
 * @module local/autoeval/types
 */

import { MetricResult } from '../metrics/types';

/**
 * Input types that AutoEval can detect and handle
 */
export type InputType =
    | 'text'           // Plain text response
    | 'json'           // JSON response
    | 'code'           // Code/programming response
    | 'rag'            // RAG response with context
    | 'qa'             // Question-answer pair
    | 'conversation'   // Multi-turn conversation
    | 'structured'     // Structured data response
    | 'unknown';       // Cannot determine type

/**
 * Content characteristics detected in the input
 */
export interface ContentCharacteristics {
    /** Detected input type */
    inputType: InputType;
    /** Whether the response contains JSON */
    hasJson: boolean;
    /** Whether the response contains code */
    hasCode: boolean;
    /** Detected programming language (if code) */
    codeLanguage?: string;
    /** Whether context is provided */
    hasContext: boolean;
    /** Whether a reference/expected answer is provided */
    hasReference: boolean;
    /** Whether it's a question-answer format */
    isQA: boolean;
    /** Estimated response length category */
    lengthCategory: 'short' | 'medium' | 'long';
    /** Detected content domains */
    domains: string[];
}

/**
 * Input for AutoEval
 */
export interface AutoEvalInput {
    /** The query/prompt (optional) */
    query?: string;
    /** The response to evaluate */
    response: string;
    /** Context for RAG evaluations (optional) */
    context?: string | string[];
    /** Expected/reference answer (optional) */
    reference?: string;
    /** Code to evaluate (optional, for security checks) */
    code?: string;
    /** Additional metadata */
    metadata?: Record<string, unknown>;
}

/**
 * Configuration for AutoEval
 */
export interface AutoEvalConfig {
    /** Minimum confidence threshold for metric selection */
    minConfidence?: number;
    /** Maximum number of metrics to run */
    maxMetrics?: number;
    /** Specific metric categories to include */
    includeCategories?: MetricCategory[];
    /** Specific metric categories to exclude */
    excludeCategories?: MetricCategory[];
    /** Whether to run security checks on code */
    enableSecurityChecks?: boolean;
    /** Whether to run hallucination detection */
    enableHallucinationCheck?: boolean;
    /** Custom metric weights */
    metricWeights?: Record<string, number>;
    /** Threshold for passing overall evaluation */
    passThreshold?: number;
}

/**
 * Categories of metrics
 */
export type MetricCategory =
    | 'string'        // String/text metrics
    | 'json'          // JSON validation metrics
    | 'similarity'    // Similarity/comparison metrics
    | 'rag'           // RAG-specific metrics
    | 'security'      // Code security metrics
    | 'hallucination' // Hallucination detection
    | 'quality';      // General quality metrics

/**
 * A selected metric with its relevance score
 */
export interface SelectedMetric {
    /** Metric name */
    name: string;
    /** Category of the metric */
    category: MetricCategory;
    /** Confidence/relevance score (0-1) */
    confidence: number;
    /** Reason for selection */
    reason: string;
    /** Required config for this metric */
    requiredConfig?: string[];
}

/**
 * Result from a single metric execution
 */
export interface MetricExecutionResult {
    /** Metric name */
    metric: string;
    /** Category of the metric */
    category: MetricCategory;
    /** The metric result */
    result: MetricResult;
    /** Time taken to execute (ms) */
    executionTimeMs: number;
    /** Any error that occurred */
    error?: string;
}

/**
 * Overall result from AutoEval
 */
export interface AutoEvalResult {
    /** Overall score (weighted average) */
    overallScore: number;
    /** Whether the evaluation passed */
    passed: boolean;
    /** Detected content characteristics */
    characteristics: ContentCharacteristics;
    /** Metrics that were selected */
    selectedMetrics: SelectedMetric[];
    /** Results from each metric */
    metricResults: MetricExecutionResult[];
    /** Summary of the evaluation */
    summary: string;
    /** Recommendations for improvement */
    recommendations: string[];
    /** Total execution time (ms) */
    totalExecutionTimeMs: number;
}

/**
 * Code language detection patterns
 */
export const CODE_LANGUAGE_PATTERNS: Record<string, RegExp[]> = {
    javascript: [
        /\bconst\s+\w+\s*=/,
        /\blet\s+\w+\s*=/,
        /\bfunction\s+\w+\s*\(/,
        /=>\s*\{/,
        /\bconsole\.log\(/,
        /\brequire\s*\(/,
        /\bmodule\.exports/,
    ],
    typescript: [
        /:\s*(?:string|number|boolean|any)\b/,
        /\binterface\s+\w+/,
        /\btype\s+\w+\s*=/,
        /<\w+(?:,\s*\w+)*>/,
    ],
    python: [
        /\bdef\s+\w+\s*\(/,
        /\bclass\s+\w+.*:/,
        /\bimport\s+\w+/,
        /\bfrom\s+\w+\s+import/,
        /\bif\s+__name__\s*==\s*['"]__main__['"]/,
        /:\s*$/m,
    ],
    java: [
        /\bpublic\s+class\s+\w+/,
        /\bprivate\s+\w+\s+\w+/,
        /\bSystem\.out\.println/,
        /\bvoid\s+main\s*\(/,
    ],
    sql: [
        /\bSELECT\s+.+\s+FROM\b/i,
        /\bINSERT\s+INTO\b/i,
        /\bUPDATE\s+.+\s+SET\b/i,
        /\bDELETE\s+FROM\b/i,
        /\bCREATE\s+TABLE\b/i,
    ],
    html: [
        /<html\b/i,
        /<div\b/i,
        /<span\b/i,
        /<script\b/i,
        /<style\b/i,
    ],
    css: [
        /\{[^}]*:\s*[^}]+;[^}]*\}/,
        /\.\w+\s*\{/,
        /#\w+\s*\{/,
        /@media\s+/,
    ],
    shell: [
        /^#!/,
        /\becho\s+/,
        /\bgrep\s+/,
        /\bawk\s+/,
        /\|\s*\w+/,
    ],
};
