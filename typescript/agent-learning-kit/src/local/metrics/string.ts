/**
 * String-based heuristic metrics for local evaluation.
 * Aligned with Python SDK's fi.evals.local.metrics.string module.
 */

import { MetricResult, MetricConfig } from './types';

/**
 * Check if text matches a regex pattern
 */
export function regex(
    response: string,
    config: MetricConfig & { pattern: string }
): MetricResult {
    if (!config.pattern) {
        throw new Error('Regex pattern is required');
    }
    const pattern = new RegExp(config.pattern, config.flags || '');
    const matches = pattern.test(response);
    return {
        score: matches ? 1.0 : 0.0,
        passed: matches,
        reason: matches
            ? `Response matches pattern: ${config.pattern}`
            : `Response does not match pattern: ${config.pattern}`
    };
}

/**
 * Check if text contains a keyword
 */
export function contains(
    response: string,
    config: MetricConfig & { keyword: string; caseSensitive?: boolean }
): MetricResult {
    const text = config.caseSensitive ? response : response.toLowerCase();
    const keyword = config.caseSensitive ? config.keyword : config.keyword.toLowerCase();
    const found = text.includes(keyword);

    return {
        score: found ? 1.0 : 0.0,
        passed: found,
        reason: found
            ? `Response contains keyword: "${config.keyword}"`
            : `Response does not contain keyword: "${config.keyword}"`
    };
}

/**
 * Check if text contains all specified keywords
 */
export function containsAll(
    response: string,
    config: MetricConfig & { keywords: string[]; caseSensitive?: boolean }
): MetricResult {
    const text = config.caseSensitive ? response : response.toLowerCase();
    const keywords = config.keywords.map(k =>
        config.caseSensitive ? k : k.toLowerCase()
    );

    const found = keywords.filter(k => text.includes(k));
    const missing = keywords.filter(k => !text.includes(k));
    const allFound = missing.length === 0;

    return {
        score: found.length / keywords.length,
        passed: allFound,
        reason: allFound
            ? `Response contains all keywords`
            : `Response missing keywords: ${missing.join(', ')}`
    };
}

/**
 * Check if text contains any of the specified keywords
 */
export function containsAny(
    response: string,
    config: MetricConfig & { keywords: string[]; caseSensitive?: boolean }
): MetricResult {
    const text = config.caseSensitive ? response : response.toLowerCase();
    const keywords = config.keywords.map(k =>
        config.caseSensitive ? k : k.toLowerCase()
    );

    const found = keywords.filter(k => text.includes(k));
    const anyFound = found.length > 0;

    return {
        score: anyFound ? 1.0 : 0.0,
        passed: anyFound,
        reason: anyFound
            ? `Response contains keywords: ${found.join(', ')}`
            : `Response does not contain any of the specified keywords`
    };
}

/**
 * Check if text contains none of the specified keywords
 */
export function containsNone(
    response: string,
    config: MetricConfig & { keywords: string[]; caseSensitive?: boolean }
): MetricResult {
    const text = config.caseSensitive ? response : response.toLowerCase();
    const keywords = config.keywords.map(k =>
        config.caseSensitive ? k : k.toLowerCase()
    );

    const found = keywords.filter(k => text.includes(k));
    const noneFound = found.length === 0;

    return {
        score: noneFound ? 1.0 : 0.0,
        passed: noneFound,
        reason: noneFound
            ? `Response contains none of the specified keywords`
            : `Response contains forbidden keywords: ${found.join(', ')}`
    };
}

/**
 * Check if text is a single line (no newlines)
 */
export function oneLine(response: string): MetricResult {
    const isSingleLine = !response.includes('\n');
    return {
        score: isSingleLine ? 1.0 : 0.0,
        passed: isSingleLine,
        reason: isSingleLine
            ? 'Response is a single line'
            : 'Response contains multiple lines'
    };
}

/**
 * Check if text equals expected value
 */
export function equals(
    response: string,
    config: MetricConfig & { expected: string; caseSensitive?: boolean; trim?: boolean }
): MetricResult {
    let actual = config.trim !== false ? response.trim() : response;
    let expected = config.trim !== false ? config.expected.trim() : config.expected;

    if (!config.caseSensitive) {
        actual = actual.toLowerCase();
        expected = expected.toLowerCase();
    }

    const isEqual = actual === expected;
    return {
        score: isEqual ? 1.0 : 0.0,
        passed: isEqual,
        reason: isEqual
            ? 'Response equals expected value'
            : 'Response does not equal expected value'
    };
}

/**
 * Check if text starts with a prefix
 */
export function startsWith(
    response: string,
    config: MetricConfig & { prefix: string; caseSensitive?: boolean }
): MetricResult {
    const text = config.caseSensitive ? response : response.toLowerCase();
    const prefix = config.caseSensitive ? config.prefix : config.prefix.toLowerCase();
    const matches = text.startsWith(prefix);

    return {
        score: matches ? 1.0 : 0.0,
        passed: matches,
        reason: matches
            ? `Response starts with: "${config.prefix}"`
            : `Response does not start with: "${config.prefix}"`
    };
}

/**
 * Check if text ends with a suffix
 */
export function endsWith(
    response: string,
    config: MetricConfig & { suffix: string; caseSensitive?: boolean }
): MetricResult {
    const text = config.caseSensitive ? response : response.toLowerCase();
    const suffix = config.caseSensitive ? config.suffix : config.suffix.toLowerCase();
    const matches = text.endsWith(suffix);

    return {
        score: matches ? 1.0 : 0.0,
        passed: matches,
        reason: matches
            ? `Response ends with: "${config.suffix}"`
            : `Response does not end with: "${config.suffix}"`
    };
}

/**
 * Check if text length is less than a threshold
 */
export function lengthLessThan(
    response: string,
    config: MetricConfig & { maxLength: number }
): MetricResult {
    const length = response.length;
    const isValid = length < config.maxLength;

    return {
        score: isValid ? 1.0 : 0.0,
        passed: isValid,
        reason: isValid
            ? `Response length (${length}) is less than ${config.maxLength}`
            : `Response length (${length}) is not less than ${config.maxLength}`
    };
}

/**
 * Check if text length is greater than a threshold
 */
export function lengthGreaterThan(
    response: string,
    config: MetricConfig & { minLength: number }
): MetricResult {
    const length = response.length;
    const isValid = length > config.minLength;

    return {
        score: isValid ? 1.0 : 0.0,
        passed: isValid,
        reason: isValid
            ? `Response length (${length}) is greater than ${config.minLength}`
            : `Response length (${length}) is not greater than ${config.minLength}`
    };
}

/**
 * Check if text length is between min and max
 */
export function lengthBetween(
    response: string,
    config: MetricConfig & { minLength: number; maxLength: number }
): MetricResult {
    const length = response.length;
    const isValid = length >= config.minLength && length <= config.maxLength;

    return {
        score: isValid ? 1.0 : 0.0,
        passed: isValid,
        reason: isValid
            ? `Response length (${length}) is between ${config.minLength} and ${config.maxLength}`
            : `Response length (${length}) is not between ${config.minLength} and ${config.maxLength}`
    };
}
