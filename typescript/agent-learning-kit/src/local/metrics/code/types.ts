/**
 * Type definitions for code security metrics.
 * @module local/metrics/code/types
 */

import { MetricResult, MetricConfig } from '../types';

/**
 * Input for code security evaluation
 */
export interface CodeInput {
    /** The code to evaluate */
    code: string;
    /** Programming language (optional, for language-specific checks) */
    language?: string;
}

/**
 * Configuration for code security metrics
 */
export interface CodeSecurityConfig extends MetricConfig {
    /** Threshold for passing (default: 1.0 for security - must be completely safe) */
    threshold?: number;
    /** Custom patterns to check (regex strings) */
    customPatterns?: string[];
    /** Patterns to ignore/whitelist */
    ignorePatterns?: string[];
    /** Whether to check comments (default: true) */
    checkComments?: boolean;
    /** Severity level to flag ('low' | 'medium' | 'high' | 'critical') */
    minSeverity?: 'low' | 'medium' | 'high' | 'critical';
}

/**
 * A detected security issue
 */
export interface SecurityIssue {
    /** Type of vulnerability */
    type: string;
    /** Severity level */
    severity: 'low' | 'medium' | 'high' | 'critical';
    /** Line number where issue was found (if applicable) */
    line?: number;
    /** The matched pattern or text */
    match: string;
    /** Description of the issue */
    description: string;
    /** Suggested fix */
    suggestion?: string;
}

/**
 * Extended result for code security metrics
 */
export interface CodeSecurityResult extends MetricResult {
    /** List of detected security issues */
    issues: SecurityIssue[];
    /** Count of issues by severity */
    severityCounts: {
        low: number;
        medium: number;
        high: number;
        critical: number;
    };
    /** The detected or specified language */
    language?: string;
}

/**
 * Severity levels as numeric values for comparison
 */
export const SEVERITY_LEVELS: Record<string, number> = {
    low: 1,
    medium: 2,
    high: 3,
    critical: 4
};

/**
 * Check if a severity meets the minimum threshold
 */
export function meetsMinSeverity(
    severity: string,
    minSeverity: string = 'low'
): boolean {
    return SEVERITY_LEVELS[severity] >= SEVERITY_LEVELS[minSeverity];
}

/**
 * Create a security result from issues
 */
export function createSecurityResult(
    issues: SecurityIssue[],
    config: CodeSecurityConfig = {}
): CodeSecurityResult {
    const threshold = config.threshold ?? 1.0;
    const minSeverity = config.minSeverity ?? 'low';

    // Filter issues by minimum severity
    const relevantIssues = issues.filter(i => meetsMinSeverity(i.severity, minSeverity));

    // Count by severity
    const severityCounts = {
        low: relevantIssues.filter(i => i.severity === 'low').length,
        medium: relevantIssues.filter(i => i.severity === 'medium').length,
        high: relevantIssues.filter(i => i.severity === 'high').length,
        critical: relevantIssues.filter(i => i.severity === 'critical').length
    };

    // Score: 1.0 if no issues, decreases with issues
    // Critical issues have more impact
    const weightedIssueCount =
        severityCounts.low * 0.1 +
        severityCounts.medium * 0.3 +
        severityCounts.high * 0.6 +
        severityCounts.critical * 1.0;

    const score = Math.max(0, 1 - weightedIssueCount);
    const passed = score >= threshold && severityCounts.critical === 0 && severityCounts.high === 0;

    let reason: string;
    if (relevantIssues.length === 0) {
        reason = 'No security issues detected';
    } else {
        const issueTypes = [...new Set(relevantIssues.map(i => i.type))];
        reason = `Found ${relevantIssues.length} security issue(s): ${issueTypes.join(', ')}`;
    }

    return {
        score,
        passed,
        reason,
        issues: relevantIssues,
        severityCounts
    };
}
