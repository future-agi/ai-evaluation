/**
 * Context Recall metric for RAG evaluation.
 * Measures what proportion of relevant information was retrieved.
 *
 * @module local/metrics/rag/context-recall
 */

import {
    RAGInput,
    RAGConfig,
    RAGResult,
    tokenize,
    jaccardSimilarity,
    normalizeContext,
    extractSentences
} from './types';

/**
 * Calculate context recall - proportion of relevant information that was retrieved.
 *
 * Recall = (Number of ground truth contexts found) / (Total ground truth contexts)
 *
 * If no ground truth contexts provided, uses expected_response as reference.
 *
 * @param input - RAG evaluation input
 * @param config - Metric configuration
 * @returns Recall score and details
 *
 * @example
 * ```typescript
 * const result = contextRecall({
 *     query: 'What is machine learning?',
 *     response: 'Machine learning is...',
 *     context: ['ML is a type of AI...'],
 *     ground_truth_contexts: ['ML is a type of AI...', 'ML uses data to learn...']
 * });
 * console.log(result.score); // 0.5 (found 1 of 2 ground truth)
 * ```
 */
export function contextRecall(
    input: RAGInput,
    config: RAGConfig = {}
): RAGResult {
    const threshold = config.threshold ?? 0.5;
    const similarityThreshold = config.similarityThreshold ?? 0.3;

    const contexts = normalizeContext(input.context);
    const combinedContext = contexts.join(' ');
    const contextTokens = tokenize(combinedContext);

    // Determine what to measure recall against
    let references: string[];

    if (input.ground_truth_contexts && input.ground_truth_contexts.length > 0) {
        references = input.ground_truth_contexts;
    } else if (input.expected_response) {
        // Use sentences from expected response as reference
        references = extractSentences(input.expected_response);
    } else {
        return {
            score: 1.0,
            passed: true,
            reason: 'No ground truth provided, assuming full recall',
            recall: 1.0
        };
    }

    if (references.length === 0) {
        return {
            score: 1.0,
            passed: true,
            reason: 'No references to check recall against',
            recall: 1.0
        };
    }

    // Check which references are covered by retrieved contexts
    let coveredCount = 0;
    const coverageScores: number[] = [];

    for (const reference of references) {
        const refTokens = tokenize(reference);

        // Check against each context chunk
        let maxSimilarity = 0;
        for (const context of contexts) {
            const similarity = jaccardSimilarity(refTokens, tokenize(context));
            maxSimilarity = Math.max(maxSimilarity, similarity);
        }

        coverageScores.push(maxSimilarity);
        if (maxSimilarity >= similarityThreshold) {
            coveredCount++;
        }
    }

    const recall = coveredCount / references.length;
    const passed = recall >= threshold;

    return {
        score: recall,
        passed,
        reason: passed
            ? `Retrieved ${coveredCount}/${references.length} relevant references (recall: ${(recall * 100).toFixed(1)}%)`
            : `Only retrieved ${coveredCount}/${references.length} relevant references (recall: ${(recall * 100).toFixed(1)}%, threshold: ${(threshold * 100).toFixed(1)}%)`,
        recall,
        chunkScores: coverageScores
    };
}
