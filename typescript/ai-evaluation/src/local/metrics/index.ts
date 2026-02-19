/**
 * Local metrics registry and exports.
 * Aligned with Python SDK's fi.evals.local.metrics module.
 */

export * from './types';
export * from './string';
export * from './json';
export * from './similarity';
export * from './rag';
export * from './code';
// Hallucination - selective exports to avoid conflicts with RAG module
export {
    HallucinationInput,
    HallucinationConfig,
    HallucinationResult,
    Claim,
    hallucinationTokenize,
    hallucinationGetNgrams,
    hallucinationExtractSentences,
    isFactualClaim,
    calculateOverlap,
    hallucinationDetection,
    detectHallucination,
    noHallucination
} from './hallucination';

import { MetricResult, MetricConfig, MetricRegistryEntry } from './types';
import * as stringMetrics from './string';
import * as jsonMetrics from './json';
import * as similarityMetrics from './similarity';
import * as ragMetrics from './rag';
import * as codeMetrics from './code';
import * as hallucinationMetrics from './hallucination';

/**
 * Registry of all available local metrics
 */
export const METRIC_REGISTRY: Map<string, MetricRegistryEntry> = new Map([
    // String metrics
    ['regex', {
        name: 'regex',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.regex(response, config as any),
        description: 'Check if text matches a regex pattern',
        requiredConfig: ['pattern']
    }],
    ['contains', {
        name: 'contains',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.contains(response, config as any),
        description: 'Check if text contains a keyword',
        requiredConfig: ['keyword']
    }],
    ['contains_all', {
        name: 'contains_all',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.containsAll(response, config as any),
        description: 'Check if text contains all specified keywords',
        requiredConfig: ['keywords']
    }],
    ['contains_any', {
        name: 'contains_any',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.containsAny(response, config as any),
        description: 'Check if text contains any of the specified keywords',
        requiredConfig: ['keywords']
    }],
    ['contains_none', {
        name: 'contains_none',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.containsNone(response, config as any),
        description: 'Check if text contains none of the specified keywords',
        requiredConfig: ['keywords']
    }],
    ['one_line', {
        name: 'one_line',
        fn: (response: string) => stringMetrics.oneLine(response),
        description: 'Check if text is a single line'
    }],
    ['equals', {
        name: 'equals',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.equals(response, config as any),
        description: 'Check if text equals expected value',
        requiredConfig: ['expected']
    }],
    ['starts_with', {
        name: 'starts_with',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.startsWith(response, config as any),
        description: 'Check if text starts with a prefix',
        requiredConfig: ['prefix']
    }],
    ['ends_with', {
        name: 'ends_with',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.endsWith(response, config as any),
        description: 'Check if text ends with a suffix',
        requiredConfig: ['suffix']
    }],
    ['length_less_than', {
        name: 'length_less_than',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.lengthLessThan(response, config as any),
        description: 'Check if text length is less than a threshold',
        requiredConfig: ['maxLength']
    }],
    ['length_greater_than', {
        name: 'length_greater_than',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.lengthGreaterThan(response, config as any),
        description: 'Check if text length is greater than a threshold',
        requiredConfig: ['minLength']
    }],
    ['length_between', {
        name: 'length_between',
        fn: (response: string, config?: MetricConfig) =>
            stringMetrics.lengthBetween(response, config as any),
        description: 'Check if text length is between min and max',
        requiredConfig: ['minLength', 'maxLength']
    }],

    // JSON metrics
    ['contains_json', {
        name: 'contains_json',
        fn: (response: string) => jsonMetrics.containsJson(response),
        description: 'Check if the response contains valid JSON'
    }],
    ['is_json', {
        name: 'is_json',
        fn: (response: string) => jsonMetrics.isJson(response),
        description: 'Check if the response is valid JSON'
    }],
    ['json_schema', {
        name: 'json_schema',
        fn: (response: string, config?: MetricConfig) =>
            jsonMetrics.jsonSchema(response, config as any),
        description: 'Validate JSON against a schema',
        requiredConfig: ['schema']
    }],

    // Similarity metrics
    ['bleu_score', {
        name: 'bleu_score',
        fn: (response: string, config?: MetricConfig) =>
            similarityMetrics.bleuScore(response, config as any),
        description: 'Calculate BLEU score',
        requiredConfig: ['reference']
    }],
    ['rouge_score', {
        name: 'rouge_score',
        fn: (response: string, config?: MetricConfig) =>
            similarityMetrics.rougeScore(response, config as any),
        description: 'Calculate ROUGE score',
        requiredConfig: ['reference']
    }],
    ['recall_score', {
        name: 'recall_score',
        fn: (response: string, config?: MetricConfig) =>
            similarityMetrics.recallScore(response, config as any),
        description: 'Calculate recall score',
        requiredConfig: ['reference']
    }],
    ['levenshtein_similarity', {
        name: 'levenshtein_similarity',
        fn: (response: string, config?: MetricConfig) =>
            similarityMetrics.levenshteinSimilarity(response, config as any),
        description: 'Calculate Levenshtein similarity',
        requiredConfig: ['reference']
    }],
    ['numeric_similarity', {
        name: 'numeric_similarity',
        fn: (response: string, config?: MetricConfig) =>
            similarityMetrics.numericSimilarity(response, config as any),
        description: 'Calculate numeric similarity',
        requiredConfig: ['reference']
    }],
    ['semantic_list_contains', {
        name: 'semantic_list_contains',
        fn: (response: string, config?: MetricConfig) =>
            similarityMetrics.semanticListContains(response, config as any),
        description: 'Check if response contains items from a list',
        requiredConfig: ['items']
    }],

    // RAG metrics
    ['context_precision', {
        name: 'context_precision',
        fn: (_response: string, config?: MetricConfig) =>
            ragMetrics.contextPrecision(config as any, config),
        description: 'Measure precision of retrieved contexts',
        requiredConfig: ['query', 'context']
    }],
    ['context_recall', {
        name: 'context_recall',
        fn: (_response: string, config?: MetricConfig) =>
            ragMetrics.contextRecall(config as any, config),
        description: 'Measure recall of retrieved contexts',
        requiredConfig: ['query', 'context']
    }],
    ['faithfulness', {
        name: 'faithfulness',
        fn: (_response: string, config?: MetricConfig) =>
            ragMetrics.faithfulness(config as any, config),
        description: 'Measure if response is faithful to context',
        requiredConfig: ['query', 'response', 'context']
    }],
    ['groundedness', {
        name: 'groundedness',
        fn: (_response: string, config?: MetricConfig) =>
            ragMetrics.groundedness(config as any, config),
        description: 'Alias for faithfulness - measure if response is grounded in context',
        requiredConfig: ['query', 'response', 'context']
    }],
    ['answer_relevance', {
        name: 'answer_relevance',
        fn: (_response: string, config?: MetricConfig) =>
            ragMetrics.answerRelevance(config as any, config),
        description: 'Measure if response is relevant to query',
        requiredConfig: ['query', 'response', 'context']
    }],
    ['context_relevance', {
        name: 'context_relevance',
        fn: (_response: string, config?: MetricConfig) =>
            ragMetrics.contextRelevance(config as any, config),
        description: 'Measure if context is relevant to query',
        requiredConfig: ['query', 'context']
    }],
    ['context_utilization', {
        name: 'context_utilization',
        fn: (_response: string, config?: MetricConfig) =>
            ragMetrics.contextUtilization(config as any, config),
        description: 'Measure how much of context is used in response',
        requiredConfig: ['query', 'response', 'context']
    }],

    // Code Security metrics
    ['sql_injection', {
        name: 'sql_injection',
        fn: (_response: string, config?: MetricConfig) =>
            codeMetrics.sqlInjection(config as any, config),
        description: 'Detect SQL injection vulnerabilities in code',
        requiredConfig: ['code']
    }],
    ['xss_detection', {
        name: 'xss_detection',
        fn: (_response: string, config?: MetricConfig) =>
            codeMetrics.xssDetection(config as any, config),
        description: 'Detect XSS vulnerabilities in code',
        requiredConfig: ['code']
    }],
    ['secrets_detection', {
        name: 'secrets_detection',
        fn: (_response: string, config?: MetricConfig) =>
            codeMetrics.secretsDetection(config as any, config),
        description: 'Detect hardcoded secrets in code',
        requiredConfig: ['code']
    }],
    ['code_security_scan', {
        name: 'code_security_scan',
        fn: (_response: string, config?: MetricConfig) =>
            codeMetrics.allSecurityChecks(config as any, config),
        description: 'Run all code security checks',
        requiredConfig: ['code']
    }],

    // Hallucination detection metrics
    ['hallucination_detection', {
        name: 'hallucination_detection',
        fn: (_response: string, config?: MetricConfig) =>
            hallucinationMetrics.hallucinationDetection(config as any, config),
        description: 'Detect hallucinations (unsupported claims) in response',
        requiredConfig: ['query', 'response', 'context']
    }],
    ['detect_hallucination', {
        name: 'detect_hallucination',
        fn: (_response: string, config?: MetricConfig) =>
            hallucinationMetrics.detectHallucination(config as any, config),
        description: 'Alias for hallucination_detection',
        requiredConfig: ['query', 'response', 'context']
    }],
    ['no_hallucination', {
        name: 'no_hallucination',
        fn: (_response: string, config?: MetricConfig) =>
            hallucinationMetrics.noHallucination(config as any, config),
        description: 'Alias for hallucination_detection',
        requiredConfig: ['query', 'response', 'context']
    }],
]);

/**
 * Get a metric by name
 */
export function getMetric(name: string): MetricRegistryEntry | undefined {
    return METRIC_REGISTRY.get(name.toLowerCase());
}

/**
 * Check if a metric exists
 */
export function hasMetric(name: string): boolean {
    return METRIC_REGISTRY.has(name.toLowerCase());
}

/**
 * Run a metric by name
 */
export function runMetric(
    metricName: string,
    response: string,
    config?: MetricConfig
): MetricResult {
    const entry = getMetric(metricName);
    if (!entry) {
        throw new Error(`Unknown metric: ${metricName}`);
    }
    return entry.fn(response, config);
}

/**
 * Get all available metric names
 */
export function getAvailableMetrics(): string[] {
    return Array.from(METRIC_REGISTRY.keys());
}
