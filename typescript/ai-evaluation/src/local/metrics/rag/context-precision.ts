/**
 * Context Precision metric for RAG evaluation.
 * Measures what proportion of the retrieved contexts are relevant.
 *
 * @module local/metrics/rag/context-precision
 */

import {
    RAGInput,
    RAGConfig,
    RAGResult,
    tokenize,
    jaccardSimilarity,
    normalizeContext
} from './types';

/**
 * Calculate context precision - proportion of retrieved contexts that are relevant.
 *
 * Precision = (Number of relevant retrieved contexts) / (Total retrieved contexts)
 *
 * If ground truth contexts are provided, compares against them.
 * Otherwise, uses query-context relevance scoring.
 *
 * @param input - RAG evaluation input
 * @param config - Metric configuration
 * @returns Precision score and details
 *
 * @example
 * ```typescript
 * const result = contextPrecision({
 *     query: 'What is machine learning?',
 *     response: 'Machine learning is a subset of AI...',
 *     context: ['ML is a type of AI...', 'Unrelated context about cooking...'],
 *     ground_truth_contexts: ['ML is a type of AI...']
 * });
 * console.log(result.score); // 0.5 (1 relevant out of 2)
 * ```
 */
export function contextPrecision(
    input: RAGInput,
    config: RAGConfig = {}
): RAGResult {
    const threshold = config.threshold ?? 0.5;
    const similarityThreshold = config.similarityThreshold ?? 0.3;

    const contexts = normalizeContext(input.context);
    const queryTokens = tokenize(input.query);

    if (contexts.length === 0) {
        return {
            score: 0,
            passed: false,
            reason: 'No context provided'
        };
    }

    // Calculate relevance for each context chunk
    const relevanceScores: number[] = [];
    const relevantIndices: number[] = [];

    if (input.ground_truth_contexts && input.ground_truth_contexts.length > 0) {
        // Compare against ground truth contexts
        for (let i = 0; i < contexts.length; i++) {
            const contextTokens = tokenize(contexts[i]);
            let maxSimilarity = 0;

            for (const gtContext of input.ground_truth_contexts) {
                const gtTokens = tokenize(gtContext);
                const similarity = jaccardSimilarity(contextTokens, gtTokens);
                maxSimilarity = Math.max(maxSimilarity, similarity);
            }

            relevanceScores.push(maxSimilarity);
            if (maxSimilarity >= similarityThreshold) {
                relevantIndices.push(i);
            }
        }
    } else {
        // Use query-context relevance as proxy
        for (let i = 0; i < contexts.length; i++) {
            const contextTokens = tokenize(contexts[i]);
            const similarity = jaccardSimilarity(queryTokens, contextTokens);
            relevanceScores.push(similarity);

            if (similarity >= similarityThreshold) {
                relevantIndices.push(i);
            }
        }
    }

    const precision = relevantIndices.length / contexts.length;
    const passed = precision >= threshold;

    return {
        score: precision,
        passed,
        reason: passed
            ? `${relevantIndices.length}/${contexts.length} contexts are relevant (precision: ${(precision * 100).toFixed(1)}%)`
            : `Only ${relevantIndices.length}/${contexts.length} contexts are relevant (precision: ${(precision * 100).toFixed(1)}%, threshold: ${(threshold * 100).toFixed(1)}%)`,
        precision,
        chunkScores: relevanceScores,
        relevantIndices
    };
}
