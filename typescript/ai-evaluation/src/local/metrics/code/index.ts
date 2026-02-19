/**
 * Code Security Metrics
 *
 * Static analysis metrics for detecting security vulnerabilities in code.
 * These run entirely locally without requiring an LLM.
 *
 * @module local/metrics/code
 */

// Types
export {
    CodeInput,
    CodeSecurityConfig,
    CodeSecurityResult,
    SecurityIssue,
    SEVERITY_LEVELS,
    meetsMinSeverity,
    createSecurityResult
} from './types';

// SQL Injection Detection
export { sqlInjection, noSqlInjection } from './sql-injection';

// XSS Detection
export { xssDetection, noXss } from './xss-detection';

// Secrets Detection
export { secretsDetection, noSecrets, noHardcodedSecrets } from './secrets-detection';

/**
 * Run all security checks on code
 *
 * @param input - Code to analyze
 * @param config - Configuration options
 * @returns Combined security analysis result
 *
 * @example
 * ```typescript
 * const result = allSecurityChecks({
 *     code: 'const apiKey = "sk-1234"; db.query("SELECT * FROM users WHERE id=" + id);'
 * });
 * console.log(result.passed); // false
 * console.log(result.issues.length); // 2+
 * ```
 */
import { CodeInput, CodeSecurityConfig, CodeSecurityResult, SecurityIssue, createSecurityResult } from './types';
import { sqlInjection } from './sql-injection';
import { xssDetection } from './xss-detection';
import { secretsDetection } from './secrets-detection';

export function allSecurityChecks(
    input: CodeInput,
    config: CodeSecurityConfig = {}
): CodeSecurityResult {
    const allIssues: SecurityIssue[] = [];

    // Run all detection methods
    const sqlResult = sqlInjection(input, config);
    const xssResult = xssDetection(input, config);
    const secretsResult = secretsDetection(input, config);

    // Combine all issues
    allIssues.push(...sqlResult.issues);
    allIssues.push(...xssResult.issues);
    allIssues.push(...secretsResult.issues);

    // Create combined result
    const result = createSecurityResult(allIssues, config);
    result.language = input.language;
    return result;
}

/**
 * Convenience alias for allSecurityChecks
 */
export const codeSecurityScan = allSecurityChecks;
