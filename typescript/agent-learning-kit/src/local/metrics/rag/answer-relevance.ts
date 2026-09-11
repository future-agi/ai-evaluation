/**
 * Answer Relevance metric for RAG evaluation.
 * Measures how relevant the response is to the original query.
 *
 * @module local/metrics/rag/answer-relevance
 */

import {
    RAGInput,
    RAGConfig,
    RAGResult,
    tokenize,
    jaccardSimilarity,
    ngramOverlap
} from './types';

/**
 * Calculate answer relevance - how well the response addresses the query.
 *
 * Uses multiple signals:
 * 1. Token overlap between query and response
 * 2. N-gram overlap for phrase matching
 * 3. Key term coverage
 *
 * @param input - RAG evaluation input
 * @param config - Metric configuration
 * @returns Relevance score and details
 *
 * @example
 * ```typescript
 * const result = answerRelevance({
 *     query: 'What is the capital of France?',
 *     response: 'The capital of France is Paris.',
 *     context: ['Paris is the capital of France.']
 * });
 * console.log(result.score); // High score due to query terms in response
 * ```
 */
export function answerRelevance(
    input: RAGInput,
    config: RAGConfig = {}
): RAGResult {
    const threshold = config.threshold ?? 0.5;
    const ngramSize = config.ngramSize ?? 2;

    if (!input.response || input.response.trim().length === 0) {
        return {
            score: 0,
            passed: false,
            reason: 'No response provided'
        };
    }

    if (!input.query || input.query.trim().length === 0) {
        return {
            score: 1.0,
            passed: true,
            reason: 'No query provided to check relevance'
        };
    }

    const queryTokens = tokenize(input.query);
    const responseTokens = tokenize(input.response);

    // Calculate multiple relevance signals
    const tokenOverlap = jaccardSimilarity(queryTokens, responseTokens);
    const ngramSim = ngramOverlap(input.query, input.response, ngramSize);

    // Check for key terms (nouns/verbs in query that should appear in response)
    // Simple heuristic: longer words are more likely to be content words
    const keyTerms = queryTokens.filter(t => t.length > 3);
    let keyTermsCovered = 0;

    for (const term of keyTerms) {
        if (responseTokens.some(rt => rt.includes(term) || term.includes(rt))) {
            keyTermsCovered++;
        }
    }

    const keyTermCoverage = keyTerms.length > 0 ? keyTermsCovered / keyTerms.length : 1.0;

    // Weighted combination of signals
    const relevanceScore = (tokenOverlap * 0.3) + (ngramSim * 0.3) + (keyTermCoverage * 0.4);
    const passed = relevanceScore >= threshold;

    return {
        score: relevanceScore,
        passed,
        reason: passed
            ? `Response is relevant to query (score: ${(relevanceScore * 100).toFixed(1)}%)`
            : `Response may not fully address the query (relevance: ${(relevanceScore * 100).toFixed(1)}%, threshold: ${(threshold * 100).toFixed(1)}%)`,
        chunkScores: [tokenOverlap, ngramSim, keyTermCoverage]
    };
}
