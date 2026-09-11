/**
 * Type definitions for Scanner Pipeline.
 * @module local/scanner/types
 */

/**
 * Severity levels for findings
 */
export type Severity = 'info' | 'low' | 'medium' | 'high' | 'critical';

/**
 * Categories of scannable issues
 */
export type FindingCategory =
    | 'security'        // Security vulnerabilities
    | 'quality'         // Code/response quality issues
    | 'compliance'      // Policy/compliance violations
    | 'performance'     // Performance concerns
    | 'style'           // Style/formatting issues
    | 'hallucination'   // Factual accuracy issues
    | 'privacy'         // Privacy/PII concerns
    | 'custom';         // User-defined rules

/**
 * A single finding from a scan
 */
export interface Finding {
    /** Unique identifier for this finding */
    id: string;
    /** Rule that triggered this finding */
    ruleId: string;
    /** Category of the finding */
    category: FindingCategory;
    /** Severity level */
    severity: Severity;
    /** Human-readable title */
    title: string;
    /** Detailed description */
    description: string;
    /** Location in the content (if applicable) */
    location?: FindingLocation;
    /** The matched/offending content */
    match?: string;
    /** Suggested fix or remediation */
    suggestion?: string;
    /** Additional metadata */
    metadata?: Record<string, unknown>;
    /** Confidence score (0-1) */
    confidence: number;
}

/**
 * Location of a finding
 */
export interface FindingLocation {
    /** Line number (1-indexed) */
    line?: number;
    /** Column number (1-indexed) */
    column?: number;
    /** Start character offset */
    startOffset?: number;
    /** End character offset */
    endOffset?: number;
    /** File path (if scanning files) */
    filePath?: string;
}

/**
 * A rule for scanning
 */
export interface ScanRule {
    /** Unique rule identifier */
    id: string;
    /** Rule name */
    name: string;
    /** Category */
    category: FindingCategory;
    /** Default severity */
    severity: Severity;
    /** Description of what this rule checks */
    description: string;
    /** Whether the rule is enabled by default */
    enabledByDefault: boolean;
    /** Tags for filtering */
    tags?: string[];
    /** The check function */
    check: (content: string, context?: ScanContext) => Finding[];
}

/**
 * Context provided to scan rules
 */
export interface ScanContext {
    /** Type of content being scanned */
    contentType?: 'code' | 'text' | 'json' | 'markdown' | 'html';
    /** Programming language (for code) */
    language?: string;
    /** File path (if scanning files) */
    filePath?: string;
    /** Additional context for RAG checks */
    ragContext?: string | string[];
    /** Query for Q&A checks */
    query?: string;
    /** Reference for comparison */
    reference?: string;
    /** Custom context data */
    custom?: Record<string, unknown>;
}

/**
 * Configuration for a scan
 */
export interface ScanConfig {
    /** Rules to enable (by ID or tag) */
    enableRules?: string[];
    /** Rules to disable (by ID or tag) */
    disableRules?: string[];
    /** Minimum severity to report */
    minSeverity?: Severity;
    /** Maximum findings to return */
    maxFindings?: number;
    /** Categories to include */
    includeCategories?: FindingCategory[];
    /** Categories to exclude */
    excludeCategories?: FindingCategory[];
    /** Custom rules to add */
    customRules?: ScanRule[];
    /** Context for the scan */
    context?: ScanContext;
    /** Whether to include suggestions */
    includeSuggestions?: boolean;
    /** Fail threshold - severity that causes scan to fail */
    failOnSeverity?: Severity;
}

/**
 * Result of a scan
 */
export interface ScanResult {
    /** Whether the scan passed (no findings above fail threshold) */
    passed: boolean;
    /** Total findings count */
    totalFindings: number;
    /** Findings by severity */
    findingsBySeverity: Record<Severity, number>;
    /** Findings by category */
    findingsByCategory: Record<FindingCategory, number>;
    /** All findings */
    findings: Finding[];
    /** Rules that were run */
    rulesExecuted: string[];
    /** Scan duration in ms */
    scanDurationMs: number;
    /** Summary message */
    summary: string;
}

/**
 * Severity level values for comparison
 */
export const SEVERITY_VALUES: Record<Severity, number> = {
    info: 0,
    low: 1,
    medium: 2,
    high: 3,
    critical: 4
};

/**
 * Compare two severities
 */
export function compareSeverity(a: Severity, b: Severity): number {
    return SEVERITY_VALUES[a] - SEVERITY_VALUES[b];
}

/**
 * Check if severity meets minimum threshold
 */
export function meetsSeverityThreshold(severity: Severity, minSeverity: Severity): boolean {
    return SEVERITY_VALUES[severity] >= SEVERITY_VALUES[minSeverity];
}

/**
 * Generate a unique finding ID
 */
export function generateFindingId(ruleId: string, location?: FindingLocation): string {
    const locationPart = location?.line ? `-L${location.line}` : '';
    const randomPart = Math.random().toString(36).substring(2, 8);
    return `${ruleId}${locationPart}-${randomPart}`;
}
