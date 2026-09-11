/**
 * Scanner Pipeline.
 * Scans content for security, quality, and compliance issues.
 *
 * @module local/scanner/scanner
 */

import {
    ScanConfig,
    ScanResult,
    ScanRule,
    ScanContext,
    Finding,
    FindingCategory,
    Severity,
    SEVERITY_VALUES,
    meetsSeverityThreshold,
    compareSeverity
} from './types';
import { BUILTIN_RULES, getDefaultRules, getRuleById, getRulesByTag } from './rules';

/**
 * Scanner Pipeline - comprehensive content scanning.
 *
 * @example
 * ```typescript
 * const scanner = new Scanner();
 *
 * // Scan code for security issues
 * const result = scanner.scan(`
 *     const apiKey = "sk-secret123";
 *     const query = "SELECT * FROM users WHERE id = " + userId;
 * `);
 *
 * console.log(result.passed); // false
 * console.log(result.findings); // Security findings
 *
 * // Scan with context for RAG checks
 * const ragResult = scanner.scan(response, {
 *     context: {
 *         ragContext: 'Source material here...',
 *         query: 'What is the answer?'
 *     }
 * });
 * ```
 */
export class Scanner {
    private rules: Map<string, ScanRule> = new Map();
    private defaultConfig: ScanConfig;

    constructor(config: ScanConfig = {}) {
        this.defaultConfig = {
            minSeverity: 'low',
            maxFindings: 1000,
            includeSuggestions: true,
            failOnSeverity: 'high',
            ...config
        };

        // Load default rules
        this.loadDefaultRules();

        // Add custom rules if provided
        if (config.customRules) {
            for (const rule of config.customRules) {
                this.addRule(rule);
            }
        }
    }

    /**
     * Load built-in default rules
     */
    private loadDefaultRules(): void {
        for (const rule of BUILTIN_RULES) {
            this.rules.set(rule.id, rule);
        }
    }

    /**
     * Add a custom rule
     */
    addRule(rule: ScanRule): void {
        this.rules.set(rule.id, rule);
    }

    /**
     * Remove a rule
     */
    removeRule(ruleId: string): boolean {
        return this.rules.delete(ruleId);
    }

    /**
     * Get a rule by ID
     */
    getRule(ruleId: string): ScanRule | undefined {
        return this.rules.get(ruleId);
    }

    /**
     * Get all rules
     */
    getAllRules(): ScanRule[] {
        return Array.from(this.rules.values());
    }

    /**
     * Scan content for issues
     */
    scan(content: string, config?: ScanConfig): ScanResult {
        const startTime = Date.now();
        const mergedConfig = { ...this.defaultConfig, ...config };

        // Determine which rules to run
        const rulesToRun = this.selectRules(mergedConfig);

        // Execute rules
        const allFindings: Finding[] = [];
        const rulesExecuted: string[] = [];

        for (const rule of rulesToRun) {
            try {
                const findings = rule.check(content, mergedConfig.context);
                allFindings.push(...findings);
                rulesExecuted.push(rule.id);
            } catch (error) {
                console.error(`Error executing rule ${rule.id}:`, error);
            }
        }

        // Filter findings
        let filteredFindings = this.filterFindings(allFindings, mergedConfig);

        // Sort by severity (critical first)
        filteredFindings.sort((a, b) => compareSeverity(b.severity, a.severity));

        // Apply max findings limit
        if (mergedConfig.maxFindings && filteredFindings.length > mergedConfig.maxFindings) {
            filteredFindings = filteredFindings.slice(0, mergedConfig.maxFindings);
        }

        // Remove suggestions if not requested
        if (!mergedConfig.includeSuggestions) {
            filteredFindings = filteredFindings.map(f => {
                const { suggestion, ...rest } = f;
                return rest as Finding;
            });
        }

        // Calculate statistics
        const findingsBySeverity = this.countBySeverity(filteredFindings);
        const findingsByCategory = this.countByCategory(filteredFindings);

        // Determine pass/fail
        const failThreshold = mergedConfig.failOnSeverity || 'high';
        const passed = !filteredFindings.some(f =>
            meetsSeverityThreshold(f.severity, failThreshold)
        );

        // Generate summary
        const summary = this.generateSummary(filteredFindings, passed);

        return {
            passed,
            totalFindings: filteredFindings.length,
            findingsBySeverity,
            findingsByCategory,
            findings: filteredFindings,
            rulesExecuted,
            scanDurationMs: Date.now() - startTime,
            summary
        };
    }

    /**
     * Select rules to run based on config
     */
    private selectRules(config: ScanConfig): ScanRule[] {
        let selectedRules: ScanRule[] = [];

        // Start with default enabled rules or all rules
        if (config.enableRules && config.enableRules.length > 0) {
            // Enable specific rules
            for (const ruleSpec of config.enableRules) {
                // Check if it's a tag
                if (ruleSpec.startsWith('tag:')) {
                    const tag = ruleSpec.substring(4);
                    const tagRules = Array.from(this.rules.values())
                        .filter(r => r.tags?.includes(tag));
                    selectedRules.push(...tagRules);
                } else {
                    // It's a rule ID
                    const rule = this.rules.get(ruleSpec);
                    if (rule) {
                        selectedRules.push(rule);
                    }
                }
            }
        } else {
            // Use default enabled rules
            selectedRules = Array.from(this.rules.values())
                .filter(r => r.enabledByDefault);
        }

        // Remove disabled rules
        if (config.disableRules && config.disableRules.length > 0) {
            const disabledIds = new Set<string>();

            for (const ruleSpec of config.disableRules) {
                if (ruleSpec.startsWith('tag:')) {
                    const tag = ruleSpec.substring(4);
                    Array.from(this.rules.values())
                        .filter(r => r.tags?.includes(tag))
                        .forEach(r => disabledIds.add(r.id));
                } else {
                    disabledIds.add(ruleSpec);
                }
            }

            selectedRules = selectedRules.filter(r => !disabledIds.has(r.id));
        }

        // Filter by category
        if (config.includeCategories && config.includeCategories.length > 0) {
            selectedRules = selectedRules.filter(r =>
                config.includeCategories!.includes(r.category)
            );
        }

        if (config.excludeCategories && config.excludeCategories.length > 0) {
            selectedRules = selectedRules.filter(r =>
                !config.excludeCategories!.includes(r.category)
            );
        }

        // Deduplicate
        const uniqueRules = new Map<string, ScanRule>();
        for (const rule of selectedRules) {
            uniqueRules.set(rule.id, rule);
        }

        return Array.from(uniqueRules.values());
    }

    /**
     * Filter findings based on config
     */
    private filterFindings(findings: Finding[], config: ScanConfig): Finding[] {
        let filtered = findings;

        // Filter by minimum severity
        if (config.minSeverity) {
            filtered = filtered.filter(f =>
                meetsSeverityThreshold(f.severity, config.minSeverity!)
            );
        }

        // Filter by category
        if (config.includeCategories && config.includeCategories.length > 0) {
            filtered = filtered.filter(f =>
                config.includeCategories!.includes(f.category)
            );
        }

        if (config.excludeCategories && config.excludeCategories.length > 0) {
            filtered = filtered.filter(f =>
                !config.excludeCategories!.includes(f.category)
            );
        }

        return filtered;
    }

    /**
     * Count findings by severity
     */
    private countBySeverity(findings: Finding[]): Record<Severity, number> {
        const counts: Record<Severity, number> = {
            info: 0,
            low: 0,
            medium: 0,
            high: 0,
            critical: 0
        };

        for (const finding of findings) {
            counts[finding.severity]++;
        }

        return counts;
    }

    /**
     * Count findings by category
     */
    private countByCategory(findings: Finding[]): Record<FindingCategory, number> {
        const counts: Record<string, number> = {};

        for (const finding of findings) {
            counts[finding.category] = (counts[finding.category] || 0) + 1;
        }

        return counts as Record<FindingCategory, number>;
    }

    /**
     * Generate a summary message
     */
    private generateSummary(findings: Finding[], passed: boolean): string {
        if (findings.length === 0) {
            return 'No issues found.';
        }

        const severityCounts = this.countBySeverity(findings);
        const parts: string[] = [];

        if (severityCounts.critical > 0) {
            parts.push(`${severityCounts.critical} critical`);
        }
        if (severityCounts.high > 0) {
            parts.push(`${severityCounts.high} high`);
        }
        if (severityCounts.medium > 0) {
            parts.push(`${severityCounts.medium} medium`);
        }
        if (severityCounts.low > 0) {
            parts.push(`${severityCounts.low} low`);
        }
        if (severityCounts.info > 0) {
            parts.push(`${severityCounts.info} info`);
        }

        const status = passed ? 'Passed' : 'Failed';
        return `${status}: Found ${findings.length} issue(s) - ${parts.join(', ')}.`;
    }

    /**
     * Quick scan with only security rules
     */
    scanSecurity(content: string, context?: ScanContext): ScanResult {
        return this.scan(content, {
            includeCategories: ['security'],
            context
        });
    }

    /**
     * Quick scan with only quality rules
     */
    scanQuality(content: string, context?: ScanContext): ScanResult {
        return this.scan(content, {
            includeCategories: ['quality'],
            enableRules: [
                'quality/todo-comments',
                'quality/console-log',
                'quality/hallucination'
            ],
            context
        });
    }

    /**
     * Quick scan for privacy/PII issues
     */
    scanPrivacy(content: string): ScanResult {
        return this.scan(content, {
            includeCategories: ['privacy']
        });
    }

    /**
     * Scan with RAG context for hallucination detection
     */
    scanRAG(response: string, ragContext: string | string[], query?: string): ScanResult {
        return this.scan(response, {
            enableRules: ['quality/hallucination'],
            context: {
                contentType: 'text',
                ragContext,
                query
            }
        });
    }
}

/**
 * Create a scanner with default configuration
 */
export function createScanner(config?: ScanConfig): Scanner {
    return new Scanner(config);
}

/**
 * Quick scan function without creating a scanner instance
 */
export function quickScan(content: string, config?: ScanConfig): ScanResult {
    const scanner = new Scanner();
    return scanner.scan(content, config);
}

/**
 * Quick security scan
 */
export function securityScan(content: string, context?: ScanContext): ScanResult {
    const scanner = new Scanner();
    return scanner.scanSecurity(content, context);
}

/**
 * Quick privacy scan
 */
export function privacyScan(content: string): ScanResult {
    const scanner = new Scanner();
    return scanner.scanPrivacy(content);
}
