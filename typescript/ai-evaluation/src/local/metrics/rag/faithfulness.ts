/**
 * Faithfulness metric for RAG evaluation.
 * Measures whether the response is faithful to the provided context.
 *
 * @module local/metrics/rag/faithfulness
 */

import {
    RAGInput,
    RAGConfig,
    RAGResult,
    tokenize,
    jaccardSimilarity,
    normalizeContext,
    extractSentences,
    ngramOverlap
} from './types';

/**
 * Calculate faithfulness - whether response is grounded in the context.
 *
 * Extracts claims/statements from the response and checks if they
 * can be supported by the provided context.
 *
 * Faithfulness = (Supported claims) / (Total claims)
 *
 * @param input - RAG evaluation input
 * @param config - Metric configuration
 * @returns Faithfulness score and details
 *
 * @example
 * ```typescript
 * const result = faithfulness({
 *     query: 'What is the capital of France?',
 *     response: 'Paris is the capital of France. It is also known as the City of Light.',
 *     context: ['Paris is the capital of France and its largest city.']
 * });
 * // First claim supported, second partially supported
 * ```
 */
export function faithfulness(
    input: RAGInput,
    config: RAGConfig = {}
): RAGResult {
    const threshold = config.threshold ?? 0.5;
    const supportThreshold = config.similarityThreshold ?? 0.25;
    const ngramSize = config.ngramSize ?? 2;

    const contexts = normalizeContext(input.context);
    const combinedContext = contexts.join(' ');

    if (!input.response || input.response.trim().length === 0) {
        return {
            score: 0,
            passed: false,
            reason: 'No response provided'
        };
    }

    if (contexts.length === 0 || combinedContext.trim().length === 0) {
        return {
            score: 0,
            passed: false,
            reason: 'No context provided to verify faithfulness'
        };
    }

    // Extract claims (sentences) from response
    const claims = extractSentences(input.response);

    if (claims.length === 0) {
        return {
            score: 1.0,
            passed: true,
            reason: 'No claims found in response',
            claims: []
        };
    }

    // Check each claim against context
    let supportedCount = 0;
    const claimScores: number[] = [];

    for (const claim of claims) {
        // Calculate support from context using multiple methods
        const tokenSimilarity = jaccardSimilarity(
            tokenize(claim),
            tokenize(combinedContext)
        );

        // Also check n-gram overlap for phrase matching
        const ngramSim = ngramOverlap(claim, combinedContext, ngramSize);

        // Use the higher of the two
        const supportScore = Math.max(tokenSimilarity, ngramSim);
        claimScores.push(supportScore);

        if (supportScore >= supportThreshold) {
            supportedCount++;
        }
    }

    const faithfulnessScore = supportedCount / claims.length;
    const passed = faithfulnessScore >= threshold;

    return {
        score: faithfulnessScore,
        passed,
        reason: passed
            ? `${supportedCount}/${claims.length} claims are supported by context (faithfulness: ${(faithfulnessScore * 100).toFixed(1)}%)`
            : `Only ${supportedCount}/${claims.length} claims are supported (faithfulness: ${(faithfulnessScore * 100).toFixed(1)}%, threshold: ${(threshold * 100).toFixed(1)}%)`,
        claims,
        chunkScores: claimScores
    };
}

/**
 * Alias for faithfulness - groundedness metric
 */
export const groundedness = faithfulness;
