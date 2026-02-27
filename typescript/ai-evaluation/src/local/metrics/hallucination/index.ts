/**
 * Hallucination Detection Metrics
 *
 * Metrics for detecting hallucinations (claims not supported by context) in AI responses.
 * Uses heuristic approaches that work locally without requiring an external LLM.
 *
 * @module local/metrics/hallucination
 */

// Types
export {
    HallucinationInput,
    HallucinationConfig,
    HallucinationResult,
    Claim,
    hallucinationTokenize,
    hallucinationGetNgrams,
    hallucinationExtractSentences,
    isFactualClaim,
    calculateOverlap
} from './types';

// Re-export with simpler names for direct hallucination module usage
export {
    hallucinationTokenize as tokenize,
    hallucinationGetNgrams as getNgrams,
    hallucinationExtractSentences as extractSentences
} from './types';

// Main detection function
export {
    hallucinationDetection,
    detectHallucination,
    noHallucination
} from './detection';
