/**
 * Context Relevance metric for RAG evaluation.
 * Measures how relevant the retrieved context is to the query.
 *
 * @module local/metrics/rag/context-relevance
 */

import {
    RAGInput,
    RAGConfig,
    RAGResult,
    tokenize,
    jaccardSimilarity,
    normalizeContext,
    ngramOverlap
} from './types';

/**
 * Calculate context relevance - how well the retrieved context matches the query.
 *
 * Different from precision in that it measures relevance quality, not just
 * whether contexts match ground truth.
 *
 * @param input - RAG evaluation input
 * @param config - Metric configuration
 * @returns Relevance score and details
 *
 * @example
 * ```typescript
 * const result = contextRelevance({
 *     query: 'What is machine learning?',
 *     response: 'Machine learning is...',
 *     context: [
 *         'Machine learning is a branch of AI that enables computers to learn.',
 *         'Recipe for chocolate cake...'  // Irrelevant
 *     ]
 * });
 * // Score based on how relevant each context chunk is to the query
 * ```
 */
export function contextRelevance(
    input: RAGInput,
    config: RAGConfig = {}
): RAGResult {
    const threshold = config.threshold ?? 0.5;
    const ngramSize = config.ngramSize ?? 2;

    const contexts = normalizeContext(input.context);
    const queryTokens = tokenize(input.query);

    if (contexts.length === 0) {
        return {
            score: 0,
            passed: false,
            reason: 'No context provided'
        };
    }

    if (!input.query || input.query.trim().length === 0) {
        return {
            score: 1.0,
            passed: true,
            reason: 'No query provided to check relevance'
        };
    }

    // Calculate relevance for each context chunk
    const chunkScores: number[] = [];
    const relevantIndices: number[] = [];

    for (let i = 0; i < contexts.length; i++) {
        const contextTokens = tokenize(contexts[i]);

        // Multiple relevance signals
        const tokenSim = jaccardSimilarity(queryTokens, contextTokens);
        const ngramSim = ngramOverlap(input.query, contexts[i], ngramSize);

        // Check for query key terms
        const keyTerms = queryTokens.filter(t => t.length > 3);
        let keyTermsFound = 0;
        for (const term of keyTerms) {
            if (contextTokens.some(ct => ct.includes(term) || term.includes(ct))) {
                keyTermsFound++;
            }
        }
        const keyTermScore = keyTerms.length > 0 ? keyTermsFound / keyTerms.length : 0.5;

        // Weighted combination
        const relevanceScore = (tokenSim * 0.3) + (ngramSim * 0.3) + (keyTermScore * 0.4);
        chunkScores.push(relevanceScore);

        if (relevanceScore >= threshold * 0.5) {
            relevantIndices.push(i);
        }
    }

    // Overall score is average of chunk scores
    const avgScore = chunkScores.reduce((a, b) => a + b, 0) / chunkScores.length;
    const passed = avgScore >= threshold;

    return {
        score: avgScore,
        passed,
        reason: passed
            ? `Context is relevant to query (avg score: ${(avgScore * 100).toFixed(1)}%)`
            : `Context may not be sufficiently relevant (avg score: ${(avgScore * 100).toFixed(1)}%, threshold: ${(threshold * 100).toFixed(1)}%)`,
        chunkScores,
        relevantIndices
    };
}

/**
 * Calculate context utilization - how much of the context is used in the response.
 *
 * @param input - RAG evaluation input
 * @param config - Metric configuration
 * @returns Utilization score and details
 */
export function contextUtilization(
    input: RAGInput,
    config: RAGConfig = {}
): RAGResult {
    const threshold = config.threshold ?? 0.5;

    const contexts = normalizeContext(input.context);

    if (contexts.length === 0) {
        return {
            score: 0,
            passed: false,
            reason: 'No context provided'
        };
    }

    if (!input.response || input.response.trim().length === 0) {
        return {
            score: 0,
            passed: false,
            reason: 'No response provided'
        };
    }

    const responseTokens = tokenize(input.response);

    // Check how much of each context is reflected in response
    const utilizationScores: number[] = [];
    const usedIndices: number[] = [];

    for (let i = 0; i < contexts.length; i++) {
        const contextTokens = tokenize(contexts[i]);

        // Count tokens from context that appear in response
        let usedTokens = 0;
        for (const token of contextTokens) {
            if (responseTokens.includes(token)) {
                usedTokens++;
            }
        }

        const utilization = contextTokens.length > 0 ? usedTokens / contextTokens.length : 0;
        utilizationScores.push(utilization);

        if (utilization > 0.1) {
            usedIndices.push(i);
        }
    }

    // Overall utilization: proportion of contexts that contributed
    const contextContribution = usedIndices.length / contexts.length;

    // Average utilization across used contexts
    const avgUtilization = utilizationScores.length > 0
        ? utilizationScores.reduce((a, b) => a + b, 0) / utilizationScores.length
        : 0;

    // Combined score
    const score = (contextContribution * 0.5) + (avgUtilization * 0.5);
    const passed = score >= threshold;

    return {
        score,
        passed,
        reason: passed
            ? `${usedIndices.length}/${contexts.length} contexts contributed to response (utilization: ${(score * 100).toFixed(1)}%)`
            : `Low context utilization: ${usedIndices.length}/${contexts.length} contexts used (score: ${(score * 100).toFixed(1)}%, threshold: ${(threshold * 100).toFixed(1)}%)`,
        chunkScores: utilizationScores,
        relevantIndices: usedIndices
    };
}
