/**
 * Scanner Pipeline
 *
 * Comprehensive content scanning for security, quality, and compliance issues.
 * Combines multiple detection capabilities into a unified scanning interface.
 *
 * @module local/scanner
 *
 * @example
 * ```typescript
 * import { Scanner, quickScan, securityScan } from '@future-agi/ai-evaluation/local/scanner';
 *
 * // Quick scan
 * const result = quickScan(`
 *     const apiKey = "sk-secret123";
 *     const query = "SELECT * FROM users WHERE id = " + userId;
 * `);
 *
 * console.log(result.passed); // false
 * console.log(result.summary); // "Failed: Found 2 issue(s) - 2 critical."
 *
 * // Full scanner with configuration
 * const scanner = new Scanner({
 *     minSeverity: 'medium',
 *     failOnSeverity: 'high'
 * });
 *
 * // Add custom rule
 * scanner.addRule({
 *     id: 'custom/my-rule',
 *     name: 'My Custom Rule',
 *     category: 'custom',
 *     severity: 'medium',
 *     description: 'Checks for specific patterns',
 *     enabledByDefault: true,
 *     check: (content) => {
 *         // Return findings
 *         return [];
 *     }
 * });
 *
 * // Scan code
 * const codeResult = scanner.scan(codeContent, {
 *     context: { language: 'typescript' }
 * });
 *
 * // Scan RAG response for hallucinations
 * const ragResult = scanner.scanRAG(
 *     response,
 *     'Source context here...',
 *     'What is the question?'
 * );
 *
 * // Security-only scan
 * const secResult = scanner.scanSecurity(codeContent);
 *
 * // Privacy scan for PII
 * const piiResult = scanner.scanPrivacy(textContent);
 * ```
 */

// Types
export {
    Severity,
    FindingCategory,
    Finding,
    FindingLocation,
    ScanRule,
    ScanContext,
    ScanConfig,
    ScanResult,
    SEVERITY_VALUES,
    compareSeverity,
    meetsSeverityThreshold,
    generateFindingId
} from './types';

// Built-in Rules
export {
    sqlInjectionRule,
    xssRule,
    secretsRule,
    hallucinationRule,
    piiRule,
    todoRule,
    unsafeEvalRule,
    consoleLogRule,
    profanityRule,
    BUILTIN_RULES,
    getRulesByTag,
    getRulesByCategory,
    getDefaultRules,
    getRuleById
} from './rules';

// Scanner
export {
    Scanner,
    createScanner,
    quickScan,
    securityScan,
    privacyScan
} from './scanner';
