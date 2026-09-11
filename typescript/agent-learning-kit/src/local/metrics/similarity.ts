/**
 * Similarity-based heuristic metrics for local evaluation.
 * Aligned with Python SDK's fi.evals.local.metrics.similarity module.
 */

import { MetricResult, MetricConfig } from './types';

/**
 * Calculate BLEU score between response and reference
 * Simplified implementation for local evaluation
 */
export function bleuScore(
    response: string,
    config: MetricConfig & { reference: string; ngramWeights?: number[] }
): MetricResult {
    const weights = config.ngramWeights || [0.25, 0.25, 0.25, 0.25];
    const maxN = weights.length;

    const responseTokens = tokenize(response);
    const referenceTokens = tokenize(config.reference);

    if (responseTokens.length === 0) {
        return {
            score: 0.0,
            passed: false,
            reason: 'Empty response'
        };
    }

    // Calculate n-gram precisions
    const precisions: number[] = [];
    for (let n = 1; n <= maxN; n++) {
        const responseNgrams = getNgrams(responseTokens, n);
        const referenceNgrams = getNgrams(referenceTokens, n);

        if (responseNgrams.length === 0) {
            precisions.push(0);
            continue;
        }

        let matches = 0;
        const referenceCounts = new Map<string, number>();
        for (const ngram of referenceNgrams) {
            const key = ngram.join(' ');
            referenceCounts.set(key, (referenceCounts.get(key) || 0) + 1);
        }

        const usedCounts = new Map<string, number>();
        for (const ngram of responseNgrams) {
            const key = ngram.join(' ');
            const refCount = referenceCounts.get(key) || 0;
            const usedCount = usedCounts.get(key) || 0;
            if (usedCount < refCount) {
                matches++;
                usedCounts.set(key, usedCount + 1);
            }
        }

        precisions.push(matches / responseNgrams.length);
    }

    // Calculate brevity penalty
    const bp = responseTokens.length >= referenceTokens.length
        ? 1.0
        : Math.exp(1 - referenceTokens.length / responseTokens.length);

    // Calculate weighted geometric mean
    let logSum = 0;
    let weightSum = 0;
    for (let i = 0; i < precisions.length; i++) {
        if (precisions[i] > 0) {
            logSum += weights[i] * Math.log(precisions[i]);
            weightSum += weights[i];
        }
    }

    const score = weightSum > 0 ? bp * Math.exp(logSum / weightSum) : 0;

    return {
        score,
        passed: score >= 0.5,
        reason: `BLEU score: ${score.toFixed(4)}`
    };
}

/**
 * Calculate ROUGE-L score (Longest Common Subsequence based)
 */
export function rougeScore(
    response: string,
    config: MetricConfig & { reference: string; variant?: 'rouge-l' | 'rouge-1' | 'rouge-2' }
): MetricResult {
    const variant = config.variant || 'rouge-l';
    const responseTokens = tokenize(response);
    const referenceTokens = tokenize(config.reference);

    if (responseTokens.length === 0 || referenceTokens.length === 0) {
        return {
            score: 0.0,
            passed: false,
            reason: 'Empty response or reference'
        };
    }

    let score: number;

    if (variant === 'rouge-l') {
        // LCS-based ROUGE-L
        const lcsLength = longestCommonSubsequence(responseTokens, referenceTokens);
        const precision = lcsLength / responseTokens.length;
        const recall = lcsLength / referenceTokens.length;
        score = precision + recall > 0
            ? (2 * precision * recall) / (precision + recall)
            : 0;
    } else {
        // N-gram based ROUGE
        const n = variant === 'rouge-1' ? 1 : 2;
        const responseNgrams = getNgrams(responseTokens, n);
        const referenceNgrams = getNgrams(referenceTokens, n);

        const responseSet = new Set(responseNgrams.map(ng => ng.join(' ')));
        const referenceSet = new Set(referenceNgrams.map(ng => ng.join(' ')));

        let overlap = 0;
        for (const ng of responseSet) {
            if (referenceSet.has(ng)) overlap++;
        }

        const precision = responseSet.size > 0 ? overlap / responseSet.size : 0;
        const recall = referenceSet.size > 0 ? overlap / referenceSet.size : 0;
        score = precision + recall > 0
            ? (2 * precision * recall) / (precision + recall)
            : 0;
    }

    return {
        score,
        passed: score >= 0.5,
        reason: `${variant.toUpperCase()} score: ${score.toFixed(4)}`
    };
}

/**
 * Calculate recall score (what fraction of reference tokens appear in response)
 */
export function recallScore(
    response: string,
    config: MetricConfig & { reference: string }
): MetricResult {
    const responseTokens = new Set(tokenize(response.toLowerCase()));
    const referenceTokens = tokenize(config.reference.toLowerCase());

    if (referenceTokens.length === 0) {
        return {
            score: 1.0,
            passed: true,
            reason: 'Empty reference (trivially satisfied)'
        };
    }

    let matches = 0;
    for (const token of referenceTokens) {
        if (responseTokens.has(token)) matches++;
    }

    const score = matches / referenceTokens.length;

    return {
        score,
        passed: score >= 0.5,
        reason: `Recall: ${matches}/${referenceTokens.length} tokens (${(score * 100).toFixed(1)}%)`
    };
}

/**
 * Calculate Levenshtein similarity
 */
export function levenshteinSimilarity(
    response: string,
    config: MetricConfig & { reference: string }
): MetricResult {
    const distance = levenshteinDistance(response, config.reference);
    const maxLen = Math.max(response.length, config.reference.length);
    const score = maxLen > 0 ? 1 - distance / maxLen : 1;

    return {
        score,
        passed: score >= 0.5,
        reason: `Levenshtein similarity: ${(score * 100).toFixed(1)}% (distance: ${distance})`
    };
}

/**
 * Calculate numeric similarity (for numeric responses)
 */
export function numericSimilarity(
    response: string,
    config: MetricConfig & { reference: number; tolerance?: number }
): MetricResult {
    const parsed = parseFloat(response.trim());

    if (isNaN(parsed)) {
        return {
            score: 0.0,
            passed: false,
            reason: `Cannot parse response as number: "${response}"`
        };
    }

    const tolerance = config.tolerance || 0;
    const diff = Math.abs(parsed - config.reference);
    const isWithinTolerance = diff <= tolerance;

    // Calculate score based on relative difference
    const maxDiff = Math.max(Math.abs(config.reference), 1);
    const score = Math.max(0, 1 - diff / maxDiff);

    return {
        score: isWithinTolerance ? 1.0 : score,
        passed: isWithinTolerance,
        reason: isWithinTolerance
            ? `Value ${parsed} is within tolerance ${tolerance} of ${config.reference}`
            : `Value ${parsed} differs from ${config.reference} by ${diff}`
    };
}

/**
 * Check if response semantically contains items from a list
 * Simple word-based matching (for true semantic matching, use cloud)
 */
export function semanticListContains(
    response: string,
    config: MetricConfig & { items: string[]; threshold?: number }
): MetricResult {
    const threshold = config.threshold || 0.5;
    const responseWords = new Set(tokenize(response.toLowerCase()));

    let matches = 0;
    const matchedItems: string[] = [];
    const missingItems: string[] = [];

    for (const item of config.items) {
        const itemWords = tokenize(item.toLowerCase());
        const itemMatched = itemWords.some(word => responseWords.has(word));

        if (itemMatched) {
            matches++;
            matchedItems.push(item);
        } else {
            missingItems.push(item);
        }
    }

    const score = matches / config.items.length;
    const passed = score >= threshold;

    return {
        score,
        passed,
        reason: passed
            ? `Found ${matches}/${config.items.length} items: ${matchedItems.join(', ')}`
            : `Missing items: ${missingItems.join(', ')}`
    };
}

// Helper functions

function tokenize(text: string): string[] {
    return text
        .toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(token => token.length > 0);
}

function getNgrams(tokens: string[], n: number): string[][] {
    const ngrams: string[][] = [];
    for (let i = 0; i <= tokens.length - n; i++) {
        ngrams.push(tokens.slice(i, i + n));
    }
    return ngrams;
}

function longestCommonSubsequence(a: string[], b: string[]): number {
    const m = a.length;
    const n = b.length;
    const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (a[i - 1] === b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp[m][n];
}

function levenshteinDistance(a: string, b: string): number {
    const m = a.length;
    const n = b.length;
    const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (a[i - 1] === b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + Math.min(
                    dp[i - 1][j],     // deletion
                    dp[i][j - 1],     // insertion
                    dp[i - 1][j - 1]  // substitution
                );
            }
        }
    }

    return dp[m][n];
}
