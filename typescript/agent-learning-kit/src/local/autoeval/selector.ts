/**
 * Metric selector for AutoEval.
 * Selects appropriate metrics based on content characteristics.
 *
 * @module local/autoeval/selector
 */

import {
    ContentCharacteristics,
    AutoEvalConfig,
    SelectedMetric,
    MetricCategory,
    AutoEvalInput
} from './types';

/**
 * Metric definition for selection
 */
interface MetricDefinition {
    name: string;
    category: MetricCategory;
    requiredConfig?: string[];
    /** Function to determine relevance score */
    relevance: (chars: ContentCharacteristics, input: AutoEvalInput) => number;
    /** Reason template for selection */
    reasonTemplate: string;
}

/**
 * All available metrics with their selection criteria
 */
const METRIC_DEFINITIONS: MetricDefinition[] = [
    // String metrics
    {
        name: 'one_line',
        category: 'string',
        relevance: (chars) => chars.lengthCategory === 'short' ? 0.6 : 0.2,
        reasonTemplate: 'Checking if response is a single line'
    },

    // JSON metrics
    {
        name: 'is_json',
        category: 'json',
        relevance: (chars) => chars.hasJson ? 0.95 : (chars.inputType === 'json' ? 0.8 : 0.1),
        reasonTemplate: 'Validating JSON structure'
    },
    {
        name: 'contains_json',
        category: 'json',
        relevance: (chars) => chars.hasJson ? 0.9 : 0.2,
        reasonTemplate: 'Checking for embedded JSON'
    },

    // Similarity metrics (require reference)
    {
        name: 'bleu_score',
        category: 'similarity',
        requiredConfig: ['reference'],
        relevance: (chars) => chars.hasReference ? 0.85 : 0,
        reasonTemplate: 'Computing BLEU score against reference'
    },
    {
        name: 'rouge_score',
        category: 'similarity',
        requiredConfig: ['reference'],
        relevance: (chars) => chars.hasReference ? 0.85 : 0,
        reasonTemplate: 'Computing ROUGE score against reference'
    },
    {
        name: 'levenshtein_similarity',
        category: 'similarity',
        requiredConfig: ['reference'],
        relevance: (chars) => chars.hasReference && chars.lengthCategory === 'short' ? 0.7 : 0,
        reasonTemplate: 'Computing edit distance similarity'
    },

    // RAG metrics (require context)
    {
        name: 'context_precision',
        category: 'rag',
        requiredConfig: ['query', 'context'],
        relevance: (chars) => chars.hasContext && chars.isQA ? 0.9 : 0,
        reasonTemplate: 'Evaluating precision of retrieved contexts'
    },
    {
        name: 'context_recall',
        category: 'rag',
        requiredConfig: ['query', 'context'],
        relevance: (chars) => chars.hasContext && chars.isQA ? 0.85 : 0,
        reasonTemplate: 'Evaluating recall of context retrieval'
    },
    {
        name: 'faithfulness',
        category: 'rag',
        requiredConfig: ['query', 'response', 'context'],
        relevance: (chars) => chars.hasContext ? 0.95 : 0,
        reasonTemplate: 'Checking if response is faithful to context'
    },
    {
        name: 'answer_relevance',
        category: 'rag',
        requiredConfig: ['query', 'response', 'context'],
        relevance: (chars) => chars.hasContext && chars.isQA ? 0.9 : 0,
        reasonTemplate: 'Evaluating answer relevance to query'
    },
    {
        name: 'context_utilization',
        category: 'rag',
        requiredConfig: ['query', 'response', 'context'],
        relevance: (chars) => chars.hasContext ? 0.8 : 0,
        reasonTemplate: 'Measuring how well context is utilized'
    },

    // Code security metrics
    {
        name: 'sql_injection',
        category: 'security',
        requiredConfig: ['code'],
        relevance: (chars) => {
            if (!chars.hasCode) return 0;
            const sqlLangs = ['javascript', 'typescript', 'python', 'java', 'php', 'sql'];
            if (chars.codeLanguage && sqlLangs.includes(chars.codeLanguage)) return 0.9;
            return 0.6;
        },
        reasonTemplate: 'Scanning for SQL injection vulnerabilities'
    },
    {
        name: 'xss_detection',
        category: 'security',
        requiredConfig: ['code'],
        relevance: (chars) => {
            if (!chars.hasCode) return 0;
            const webLangs = ['javascript', 'typescript', 'html', 'php'];
            if (chars.codeLanguage && webLangs.includes(chars.codeLanguage)) return 0.9;
            return 0.5;
        },
        reasonTemplate: 'Scanning for XSS vulnerabilities'
    },
    {
        name: 'secrets_detection',
        category: 'security',
        requiredConfig: ['code'],
        relevance: (chars) => chars.hasCode ? 0.95 : 0,
        reasonTemplate: 'Scanning for hardcoded secrets'
    },

    // Hallucination detection
    {
        name: 'hallucination_detection',
        category: 'hallucination',
        requiredConfig: ['query', 'response', 'context'],
        relevance: (chars) => chars.hasContext ? 0.9 : 0,
        reasonTemplate: 'Detecting potential hallucinations'
    },

    // Quality metrics for general text
    {
        name: 'length_between',
        category: 'quality',
        requiredConfig: ['minLength', 'maxLength'],
        relevance: (chars) => chars.inputType === 'text' ? 0.5 : 0.3,
        reasonTemplate: 'Checking response length is appropriate'
    },
];

/**
 * Select appropriate metrics based on content characteristics.
 *
 * @param chars - Detected content characteristics
 * @param input - The original input
 * @param config - AutoEval configuration
 * @returns List of selected metrics with relevance scores
 */
export function selectMetrics(
    chars: ContentCharacteristics,
    input: AutoEvalInput,
    config: AutoEvalConfig = {}
): SelectedMetric[] {
    const {
        minConfidence = 0.5,
        maxMetrics = 10,
        includeCategories,
        excludeCategories,
        enableSecurityChecks = true,
        enableHallucinationCheck = true
    } = config;

    const selected: SelectedMetric[] = [];

    for (const metric of METRIC_DEFINITIONS) {
        // Check category filters
        if (includeCategories && !includeCategories.includes(metric.category)) {
            continue;
        }
        if (excludeCategories && excludeCategories.includes(metric.category)) {
            continue;
        }

        // Check feature flags
        if (metric.category === 'security' && !enableSecurityChecks) {
            continue;
        }
        if (metric.category === 'hallucination' && !enableHallucinationCheck) {
            continue;
        }

        // Calculate relevance
        const confidence = metric.relevance(chars, input);

        // Skip low confidence metrics
        if (confidence < minConfidence) {
            continue;
        }

        // Check if required config can be satisfied
        if (metric.requiredConfig) {
            const canSatisfy = canSatisfyConfig(metric.requiredConfig, input);
            if (!canSatisfy) {
                continue;
            }
        }

        selected.push({
            name: metric.name,
            category: metric.category,
            confidence,
            reason: metric.reasonTemplate,
            requiredConfig: metric.requiredConfig
        });
    }

    // Sort by confidence descending
    selected.sort((a, b) => b.confidence - a.confidence);

    // Apply max metrics limit
    return selected.slice(0, maxMetrics);
}

/**
 * Check if the required config can be satisfied from the input
 */
function canSatisfyConfig(required: string[], input: AutoEvalInput): boolean {
    for (const field of required) {
        switch (field) {
            case 'query':
                if (!input.query?.trim()) return false;
                break;
            case 'response':
                if (!input.response?.trim()) return false;
                break;
            case 'context':
                if (!input.context) return false;
                if (Array.isArray(input.context) && input.context.length === 0) return false;
                if (typeof input.context === 'string' && !input.context.trim()) return false;
                break;
            case 'reference':
                if (!input.reference?.trim()) return false;
                break;
            case 'code':
                // Code can come from explicit code field or be detected in response
                if (!input.code?.trim() && !input.response?.trim()) return false;
                break;
            // Config fields like minLength, maxLength are optional - we'll use defaults
            case 'minLength':
            case 'maxLength':
                break;
            default:
                // Unknown required config - assume it can be provided
                break;
        }
    }
    return true;
}

/**
 * Get recommended metrics for a specific input type
 */
export function getRecommendedMetrics(inputType: string): string[] {
    const recommendations: Record<string, string[]> = {
        'text': ['one_line', 'length_between'],
        'json': ['is_json', 'contains_json'],
        'code': ['sql_injection', 'xss_detection', 'secrets_detection'],
        'rag': ['faithfulness', 'context_precision', 'context_recall', 'answer_relevance', 'hallucination_detection'],
        'qa': ['answer_relevance', 'bleu_score', 'rouge_score'],
        'structured': ['is_json'],
        'conversation': ['faithfulness'],
        'unknown': ['one_line']
    };

    return recommendations[inputType] || recommendations['unknown'];
}

/**
 * Build configuration for a selected metric from the input
 */
export function buildMetricConfig(
    metric: SelectedMetric,
    input: AutoEvalInput
): Record<string, unknown> {
    const config: Record<string, unknown> = {};

    // Map input fields to config
    if (input.query) config.query = input.query;
    if (input.response) config.response = input.response;
    if (input.context) config.context = input.context;
    if (input.reference) config.reference = input.reference;
    if (input.code) {
        config.code = input.code;
    } else if (metric.category === 'security' && input.response) {
        // Use response as code for security checks
        config.code = input.response;
    }

    // Add defaults for quality metrics
    if (metric.name === 'length_between') {
        config.minLength = config.minLength ?? 10;
        config.maxLength = config.maxLength ?? 5000;
    }

    return config;
}
