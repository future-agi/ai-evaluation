/**
 * RAG (Retrieval-Augmented Generation) evaluation metrics.
 *
 * These metrics help evaluate the quality of RAG systems by measuring:
 * - Context Precision: Are the retrieved contexts relevant?
 * - Context Recall: Did we retrieve all relevant information?
 * - Faithfulness: Is the response grounded in the context?
 * - Answer Relevance: Does the response address the query?
 * - Context Relevance: How relevant is the context to the query?
 * - Context Utilization: How much of the context is used?
 *
 * @module local/metrics/rag
 *
 * @example
 * ```typescript
 * import {
 *     contextPrecision,
 *     contextRecall,
 *     faithfulness,
 *     answerRelevance
 * } from '@future-agi/ai-evaluation/local';
 *
 * const input = {
 *     query: 'What is the capital of France?',
 *     response: 'The capital of France is Paris.',
 *     context: ['Paris is the capital of France.', 'France is in Europe.'],
 *     ground_truth_contexts: ['Paris is the capital of France.']
 * };
 *
 * // Check if retrieved contexts are precise
 * const precision = contextPrecision(input);
 *
 * // Check if all relevant contexts were retrieved
 * const recall = contextRecall(input);
 *
 * // Check if response is faithful to context
 * const faith = faithfulness(input);
 *
 * // Check if response addresses the query
 * const relevance = answerRelevance(input);
 * ```
 */

// Types
export {
    RAGInput,
    RAGConfig,
    RAGResult,
    tokenize,
    jaccardSimilarity,
    ngramOverlap,
    normalizeContext,
    extractSentences
} from './types';

// Metrics
export { contextPrecision } from './context-precision';
export { contextRecall } from './context-recall';
export { faithfulness, groundedness } from './faithfulness';
export { answerRelevance } from './answer-relevance';
export { contextRelevance, contextUtilization } from './context-relevance';

// Re-export as named object for registry
import { contextPrecision } from './context-precision';
import { contextRecall } from './context-recall';
import { faithfulness, groundedness } from './faithfulness';
import { answerRelevance } from './answer-relevance';
import { contextRelevance, contextUtilization } from './context-relevance';

/**
 * All RAG metrics as a named object
 */
export const RAG_METRICS = {
    context_precision: contextPrecision,
    context_recall: contextRecall,
    faithfulness,
    groundedness,
    answer_relevance: answerRelevance,
    context_relevance: contextRelevance,
    context_utilization: contextUtilization
};

/**
 * List of available RAG metric names
 */
export const RAG_METRIC_NAMES = Object.keys(RAG_METRICS);
