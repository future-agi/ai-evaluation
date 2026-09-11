/**
 * Type definitions for hallucination detection metrics.
 * @module local/metrics/hallucination/types
 */

import { MetricResult, MetricConfig } from '../types';

/**
 * Input for hallucination detection
 */
export interface HallucinationInput {
    /** The query/question that was asked */
    query: string;
    /** The response/answer to evaluate */
    response: string;
    /** The context/source material the response should be based on */
    context: string | string[];
    /** Additional reference information (optional) */
    reference?: string;
}

/**
 * Configuration for hallucination detection
 */
export interface HallucinationConfig extends MetricConfig {
    /** Threshold for passing (default: 0.8 for low hallucination) */
    threshold?: number;
    /** Whether to include detailed claim analysis */
    includeClaimAnalysis?: boolean;
    /** Method to use for detection */
    method?: 'heuristic' | 'llm' | 'hybrid';
}

/**
 * A detected claim in the response
 */
export interface Claim {
    /** The text of the claim */
    text: string;
    /** Whether the claim is supported by context */
    supported: boolean;
    /** Confidence score (0-1) */
    confidence: number;
    /** Supporting evidence from context (if any) */
    evidence?: string;
}

/**
 * Extended result for hallucination detection
 */
export interface HallucinationResult extends MetricResult {
    /** List of extracted claims from the response */
    claims?: Claim[];
    /** Number of supported claims */
    supportedClaims: number;
    /** Number of unsupported claims */
    unsupportedClaims: number;
    /** Hallucination rate (0 = no hallucination, 1 = complete hallucination) */
    hallucinationRate: number;
}

/**
 * Simple tokenizer for text processing (internal to hallucination module)
 */
export function hallucinationTokenize(text: string): string[] {
    return text.toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(t => t.length > 0);
}

/**
 * Get n-grams from tokens (internal to hallucination module)
 */
export function hallucinationGetNgrams(tokens: string[], n: number): Set<string> {
    const ngrams = new Set<string>();
    for (let i = 0; i <= tokens.length - n; i++) {
        ngrams.add(tokens.slice(i, i + n).join(' '));
    }
    return ngrams;
}

/**
 * Extract sentences from text (internal to hallucination module)
 */
export function hallucinationExtractSentences(text: string): string[] {
    return text
        .split(/[.!?]+/)
        .map(s => s.trim())
        .filter(s => s.length > 10);
}

/**
 * Check if a sentence is likely a factual claim
 */
export function isFactualClaim(sentence: string): boolean {
    const lower = sentence.toLowerCase();

    // Skip opinion/uncertainty markers
    const opinionMarkers = [
        'i think', 'i believe', 'in my opinion', 'probably', 'maybe',
        'might be', 'could be', 'seems like', 'appears to', 'i feel'
    ];
    if (opinionMarkers.some(m => lower.includes(m))) {
        return false;
    }

    // Skip questions
    if (sentence.endsWith('?')) {
        return false;
    }

    // Factual indicators
    const factualIndicators = [
        'is', 'are', 'was', 'were', 'has', 'have', 'had',
        'contains', 'includes', 'consists', 'provides', 'shows',
        'according to', 'based on', 'research shows', 'studies show'
    ];

    return factualIndicators.some(i => lower.includes(i));
}

/**
 * Calculate word overlap between two texts
 */
export function calculateOverlap(text1: string, text2: string): number {
    const tokens1 = new Set(hallucinationTokenize(text1));
    const tokens2 = new Set(hallucinationTokenize(text2));

    if (tokens1.size === 0) return 0;

    let overlap = 0;
    for (const token of tokens1) {
        if (tokens2.has(token)) {
            overlap++;
        }
    }

    return overlap / tokens1.size;
}
