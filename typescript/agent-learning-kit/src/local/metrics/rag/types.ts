/**
 * Type definitions for RAG metrics.
 * @module local/metrics/rag/types
 */

import { MetricResult, MetricConfig } from '../types';

/**
 * Input for RAG evaluation metrics.
 * Note: Does not extend MetricInput because RAG metrics have different requirements.
 */
export interface RAGInput {
    /** The user query */
    query: string;
    /** The generated response */
    response: string;
    /** The context/chunks retrieved for the query */
    context: string | string[];
    /** Expected/reference response (optional, for some metrics) */
    expected_response?: string;
    /** Ground truth relevant passages (optional, for precision/recall) */
    ground_truth_contexts?: string[];
}

/**
 * Configuration for RAG metrics
 */
export interface RAGConfig extends MetricConfig {
    /** Threshold for passing (default: 0.5) */
    threshold?: number;
    /** Whether to use exact matching (vs. fuzzy) */
    exactMatch?: boolean;
    /** Minimum similarity score for fuzzy matching */
    similarityThreshold?: number;
    /** N-gram size for overlap calculations */
    ngramSize?: number;
}

/**
 * Extended result for RAG metrics with additional metadata
 */
export interface RAGResult extends MetricResult {
    /** List of claims/facts extracted (for faithfulness) */
    claims?: string[];
    /** Breakdown of scores per context chunk */
    chunkScores?: number[];
    /** Relevant context indices */
    relevantIndices?: number[];
    /** Precision of retrieval */
    precision?: number;
    /** Recall of retrieval */
    recall?: number;
    /** F1 score */
    f1?: number;
}

/**
 * Tokenize text into words
 */
export function tokenize(text: string): string[] {
    return text.toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(token => token.length > 0);
}

/**
 * Calculate Jaccard similarity between two sets of tokens
 */
export function jaccardSimilarity(tokens1: string[], tokens2: string[]): number {
    const set1 = new Set(tokens1);
    const set2 = new Set(tokens2);

    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);

    if (union.size === 0) return 0;
    return intersection.size / union.size;
}

/**
 * Calculate n-gram overlap between two texts
 */
export function ngramOverlap(text1: string, text2: string, n: number = 2): number {
    const getNgrams = (tokens: string[], n: number): Set<string> => {
        const ngrams = new Set<string>();
        for (let i = 0; i <= tokens.length - n; i++) {
            ngrams.add(tokens.slice(i, i + n).join(' '));
        }
        return ngrams;
    };

    const tokens1 = tokenize(text1);
    const tokens2 = tokenize(text2);

    if (tokens1.length < n || tokens2.length < n) {
        return jaccardSimilarity(tokens1, tokens2);
    }

    const ngrams1 = getNgrams(tokens1, n);
    const ngrams2 = getNgrams(tokens2, n);

    const intersection = new Set([...ngrams1].filter(x => ngrams2.has(x)));
    const union = new Set([...ngrams1, ...ngrams2]);

    if (union.size === 0) return 0;
    return intersection.size / union.size;
}

/**
 * Normalize context to array of strings
 */
export function normalizeContext(context: string | string[]): string[] {
    if (Array.isArray(context)) {
        return context;
    }
    // Split by double newline or treat as single chunk
    const chunks = context.split(/\n\n+/).filter(c => c.trim().length > 0);
    return chunks.length > 0 ? chunks : [context];
}

/**
 * Extract sentences from text
 */
export function extractSentences(text: string): string[] {
    return text
        .split(/[.!?]+/)
        .map(s => s.trim())
        .filter(s => s.length > 0);
}
