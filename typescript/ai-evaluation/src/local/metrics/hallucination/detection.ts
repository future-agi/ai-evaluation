/**
 * Hallucination detection metric.
 * Detects when a response contains claims not supported by the provided context.
 *
 * @module local/metrics/hallucination/detection
 */

import {
    HallucinationInput,
    HallucinationConfig,
    HallucinationResult,
    Claim,
    hallucinationExtractSentences as extractSentences,
    isFactualClaim,
    calculateOverlap,
    hallucinationTokenize as tokenize,
    hallucinationGetNgrams as getNgrams
} from './types';

/**
 * Detect hallucinations in a response by checking if claims are supported by context.
 *
 * Uses a heuristic approach that:
 * 1. Extracts factual claims from the response
 * 2. Checks each claim against the context for support
 * 3. Calculates a hallucination rate based on unsupported claims
 *
 * @param input - The query, response, and context to evaluate
 * @param config - Configuration options
 * @returns Hallucination detection result
 *
 * @example
 * ```typescript
 * const result = hallucinationDetection({
 *     query: "What is the capital of France?",
 *     response: "Paris is the capital of France and has a population of 2 million.",
 *     context: "Paris is the capital and largest city of France."
 * });
 * console.log(result.hallucinationRate); // ~0.5 (population claim not in context)
 * ```
 */
export function hallucinationDetection(
    input: HallucinationInput,
    config: HallucinationConfig = {}
): HallucinationResult {
    const { query, response, context, reference } = input;
    const threshold = config.threshold ?? 0.8;
    const includeClaimAnalysis = config.includeClaimAnalysis ?? true;

    // Handle empty response
    if (!response || response.trim().length === 0) {
        return {
            score: 1.0,
            passed: true,
            reason: 'Empty response - no hallucination possible',
            supportedClaims: 0,
            unsupportedClaims: 0,
            hallucinationRate: 0,
            claims: []
        };
    }

    // Normalize context to array
    const contexts = Array.isArray(context) ? context : [context];
    const combinedContext = contexts.join(' ');

    // Also include reference if provided
    const allContext = reference
        ? combinedContext + ' ' + reference
        : combinedContext;

    // Handle empty context
    if (!allContext || allContext.trim().length === 0) {
        return {
            score: 0.0,
            passed: false,
            reason: 'No context provided - cannot verify claims',
            supportedClaims: 0,
            unsupportedClaims: 0,
            hallucinationRate: 1.0,
            claims: []
        };
    }

    // Extract sentences from response
    const sentences = extractSentences(response);

    if (sentences.length === 0) {
        return {
            score: 1.0,
            passed: true,
            reason: 'No factual claims detected in response',
            supportedClaims: 0,
            unsupportedClaims: 0,
            hallucinationRate: 0,
            claims: []
        };
    }

    // Analyze each sentence for support in context
    const claims: Claim[] = [];
    let supportedCount = 0;
    let unsupportedCount = 0;

    // Prepare context tokens and n-grams for efficient lookup
    const contextTokens = new Set(tokenize(allContext));
    const contextBigrams = getNgrams(tokenize(allContext), 2);
    const contextTrigrams = getNgrams(tokenize(allContext), 3);

    for (const sentence of sentences) {
        // Check if this looks like a factual claim
        if (!isFactualClaim(sentence)) {
            continue;
        }

        // Calculate support metrics
        const wordOverlap = calculateOverlap(sentence, allContext);
        const sentenceTokens = tokenize(sentence);

        // Check n-gram overlap
        const sentenceBigrams = getNgrams(sentenceTokens, 2);
        const sentenceTrigrams = getNgrams(sentenceTokens, 3);

        let bigramOverlap = 0;
        let trigramOverlap = 0;

        for (const bigram of sentenceBigrams) {
            if (contextBigrams.has(bigram)) {
                bigramOverlap++;
            }
        }

        for (const trigram of sentenceTrigrams) {
            if (contextTrigrams.has(trigram)) {
                trigramOverlap++;
            }
        }

        const normalizedBigramOverlap = sentenceBigrams.size > 0
            ? bigramOverlap / sentenceBigrams.size
            : 0;
        const normalizedTrigramOverlap = sentenceTrigrams.size > 0
            ? trigramOverlap / sentenceTrigrams.size
            : 0;

        // Combined support score
        // Weight: word overlap (30%), bigram overlap (35%), trigram overlap (35%)
        const supportScore = (
            wordOverlap * 0.30 +
            normalizedBigramOverlap * 0.35 +
            normalizedTrigramOverlap * 0.35
        );

        // Threshold for considering a claim supported
        const supportThreshold = 0.3;
        const isSupported = supportScore >= supportThreshold;

        if (isSupported) {
            supportedCount++;
        } else {
            unsupportedCount++;
        }

        // Find evidence (substring match in context)
        let evidence: string | undefined;
        if (isSupported) {
            // Find the most relevant part of context
            const contextSentences = extractSentences(allContext);
            let maxOverlap = 0;
            for (const ctxSentence of contextSentences) {
                const overlap = calculateOverlap(sentence, ctxSentence);
                if (overlap > maxOverlap) {
                    maxOverlap = overlap;
                    evidence = ctxSentence;
                }
            }
        }

        if (includeClaimAnalysis) {
            claims.push({
                text: sentence,
                supported: isSupported,
                confidence: supportScore,
                evidence
            });
        }
    }

    const totalClaims = supportedCount + unsupportedCount;

    // Calculate hallucination rate
    const hallucinationRate = totalClaims > 0
        ? unsupportedCount / totalClaims
        : 0;

    // Score is inverse of hallucination rate (high score = low hallucination)
    const score = 1 - hallucinationRate;
    const passed = score >= threshold;

    let reason: string;
    if (totalClaims === 0) {
        reason = 'No factual claims detected in response';
    } else if (unsupportedCount === 0) {
        reason = `All ${supportedCount} claim(s) are supported by the context`;
    } else if (supportedCount === 0) {
        reason = `All ${unsupportedCount} claim(s) appear unsupported by the context`;
    } else {
        reason = `${supportedCount} of ${totalClaims} claims supported; ${unsupportedCount} potentially hallucinated`;
    }

    return {
        score,
        passed,
        reason,
        supportedClaims: supportedCount,
        unsupportedClaims: unsupportedCount,
        hallucinationRate,
        claims: includeClaimAnalysis ? claims : undefined
    };
}

/**
 * Alias for hallucinationDetection
 */
export const detectHallucination = hallucinationDetection;
export const noHallucination = hallucinationDetection;
