/**
 * Built-in scanning rules.
 * @module local/scanner/rules
 */

import {
    ScanRule,
    Finding,
    ScanContext,
    generateFindingId
} from './types';
import { sqlInjection } from '../metrics/code/sql-injection';
import { xssDetection } from '../metrics/code/xss-detection';
import { secretsDetection } from '../metrics/code/secrets-detection';
import { hallucinationDetection } from '../metrics/hallucination';

/**
 * Helper to find line number from character offset
 */
function findLineNumber(content: string, offset: number): number {
    const lines = content.substring(0, offset).split('\n');
    return lines.length;
}

/**
 * SQL Injection detection rule
 */
export const sqlInjectionRule: ScanRule = {
    id: 'security/sql-injection',
    name: 'SQL Injection',
    category: 'security',
    severity: 'critical',
    description: 'Detects potential SQL injection vulnerabilities',
    enabledByDefault: true,
    tags: ['security', 'owasp', 'injection'],
    check: (content: string, context?: ScanContext): Finding[] => {
        const result = sqlInjection({ code: content, language: context?.language });
        return result.issues.map(issue => ({
            id: generateFindingId('security/sql-injection', { line: issue.line }),
            ruleId: 'security/sql-injection',
            category: 'security',
            severity: issue.severity,
            title: 'SQL Injection Vulnerability',
            description: issue.description,
            location: issue.line ? { line: issue.line } : undefined,
            match: issue.match,
            suggestion: issue.suggestion,
            confidence: 0.9
        }));
    }
};

/**
 * XSS detection rule
 */
export const xssRule: ScanRule = {
    id: 'security/xss',
    name: 'Cross-Site Scripting (XSS)',
    category: 'security',
    severity: 'high',
    description: 'Detects potential XSS vulnerabilities',
    enabledByDefault: true,
    tags: ['security', 'owasp', 'xss', 'web'],
    check: (content: string, context?: ScanContext): Finding[] => {
        const result = xssDetection({ code: content, language: context?.language });
        return result.issues.map(issue => ({
            id: generateFindingId('security/xss', { line: issue.line }),
            ruleId: 'security/xss',
            category: 'security',
            severity: issue.severity,
            title: 'XSS Vulnerability',
            description: issue.description,
            location: issue.line ? { line: issue.line } : undefined,
            match: issue.match,
            suggestion: issue.suggestion,
            confidence: 0.85
        }));
    }
};

/**
 * Secrets detection rule
 */
export const secretsRule: ScanRule = {
    id: 'security/hardcoded-secrets',
    name: 'Hardcoded Secrets',
    category: 'security',
    severity: 'critical',
    description: 'Detects hardcoded API keys, passwords, and other secrets',
    enabledByDefault: true,
    tags: ['security', 'secrets', 'credentials'],
    check: (content: string, context?: ScanContext): Finding[] => {
        const result = secretsDetection({ code: content, language: context?.language });
        return result.issues.map(issue => ({
            id: generateFindingId('security/hardcoded-secrets', { line: issue.line }),
            ruleId: 'security/hardcoded-secrets',
            category: 'security',
            severity: issue.severity,
            title: `Hardcoded ${issue.type.replace(/-/g, ' ')}`,
            description: issue.description,
            location: issue.line ? { line: issue.line } : undefined,
            match: issue.match,
            suggestion: issue.suggestion,
            confidence: 0.95
        }));
    }
};

/**
 * Hallucination detection rule
 */
export const hallucinationRule: ScanRule = {
    id: 'quality/hallucination',
    name: 'Hallucination Detection',
    category: 'hallucination',
    severity: 'medium',
    description: 'Detects claims not supported by the provided context',
    enabledByDefault: true,
    tags: ['quality', 'factual', 'rag'],
    check: (content: string, context?: ScanContext): Finding[] => {
        if (!context?.ragContext) {
            return []; // Need context for hallucination detection
        }

        const result = hallucinationDetection({
            query: context.query || '',
            response: content,
            context: context.ragContext
        }, { includeClaimAnalysis: true });

        const findings: Finding[] = [];

        if (result.claims) {
            for (const claim of result.claims) {
                if (!claim.supported) {
                    findings.push({
                        id: generateFindingId('quality/hallucination'),
                        ruleId: 'quality/hallucination',
                        category: 'hallucination',
                        severity: result.hallucinationRate > 0.5 ? 'high' : 'medium',
                        title: 'Potentially Unsupported Claim',
                        description: `The following claim may not be supported by the context: "${claim.text}"`,
                        match: claim.text,
                        suggestion: 'Verify this claim against the source material or rephrase to be more accurate',
                        confidence: 1 - claim.confidence
                    });
                }
            }
        }

        return findings;
    }
};

/**
 * PII detection patterns
 */
const PII_PATTERNS: Array<{
    pattern: RegExp;
    type: string;
    severity: 'low' | 'medium' | 'high';
}> = [
    {
        pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
        type: 'email',
        severity: 'medium'
    },
    {
        pattern: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g,
        type: 'phone number',
        severity: 'medium'
    },
    {
        pattern: /\b\d{3}[-]?\d{2}[-]?\d{4}\b/g,
        type: 'SSN',
        severity: 'high'
    },
    {
        pattern: /\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b/g,
        type: 'credit card',
        severity: 'high'
    },
    {
        pattern: /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g,
        type: 'IP address',
        severity: 'low'
    }
];

/**
 * PII detection rule
 */
export const piiRule: ScanRule = {
    id: 'privacy/pii',
    name: 'PII Detection',
    category: 'privacy',
    severity: 'medium',
    description: 'Detects personally identifiable information',
    enabledByDefault: true,
    tags: ['privacy', 'pii', 'compliance', 'gdpr'],
    check: (content: string): Finding[] => {
        const findings: Finding[] = [];
        const lines = content.split('\n');

        for (const { pattern, type, severity } of PII_PATTERNS) {
            pattern.lastIndex = 0;
            let match;

            while ((match = pattern.exec(content)) !== null) {
                const line = findLineNumber(content, match.index);
                const maskedMatch = match[0].length > 8
                    ? match[0].substring(0, 4) + '***' + match[0].substring(match[0].length - 2)
                    : '***';

                findings.push({
                    id: generateFindingId('privacy/pii', { line }),
                    ruleId: 'privacy/pii',
                    category: 'privacy',
                    severity,
                    title: `Potential ${type} detected`,
                    description: `Found what appears to be a ${type} in the content`,
                    location: { line },
                    match: maskedMatch,
                    suggestion: `Consider removing or masking this ${type}`,
                    confidence: 0.8
                });
            }
        }

        return findings;
    }
};

/**
 * TODO/FIXME detection rule
 */
export const todoRule: ScanRule = {
    id: 'quality/todo-comments',
    name: 'TODO Comments',
    category: 'quality',
    severity: 'info',
    description: 'Detects TODO, FIXME, HACK, and similar comments',
    enabledByDefault: false,
    tags: ['quality', 'code-review'],
    check: (content: string): Finding[] => {
        const findings: Finding[] = [];
        const pattern = /\b(TODO|FIXME|HACK|XXX|BUG|OPTIMIZE)\b[:\s]*(.{0,100})/gi;
        const lines = content.split('\n');

        let match;
        while ((match = pattern.exec(content)) !== null) {
            const line = findLineNumber(content, match.index);
            const type = match[1].toUpperCase();
            const message = match[2]?.trim() || '';

            findings.push({
                id: generateFindingId('quality/todo-comments', { line }),
                ruleId: 'quality/todo-comments',
                category: 'quality',
                severity: type === 'FIXME' || type === 'BUG' ? 'low' : 'info',
                title: `${type} comment found`,
                description: message || `${type} marker without description`,
                location: { line },
                match: match[0].substring(0, 50),
                confidence: 1.0
            });
        }

        return findings;
    }
};

/**
 * Unsafe eval/exec detection rule
 */
export const unsafeEvalRule: ScanRule = {
    id: 'security/unsafe-eval',
    name: 'Unsafe Eval/Exec',
    category: 'security',
    severity: 'high',
    description: 'Detects use of eval, exec, and similar dangerous functions',
    enabledByDefault: true,
    tags: ['security', 'injection'],
    check: (content: string, context?: ScanContext): Finding[] => {
        const findings: Finding[] = [];
        const patterns: Array<{ pattern: RegExp; name: string; languages?: string[] }> = [
            { pattern: /\beval\s*\(/g, name: 'eval()', languages: ['javascript', 'typescript', 'python'] },
            { pattern: /\bexec\s*\(/g, name: 'exec()', languages: ['python'] },
            { pattern: /\bFunction\s*\(/g, name: 'Function constructor', languages: ['javascript', 'typescript'] },
            { pattern: /\bsetTimeout\s*\(\s*["'`]/g, name: 'setTimeout with string', languages: ['javascript', 'typescript'] },
            { pattern: /\bsetInterval\s*\(\s*["'`]/g, name: 'setInterval with string', languages: ['javascript', 'typescript'] },
            { pattern: /\b__import__\s*\(/g, name: '__import__()', languages: ['python'] },
            { pattern: /\bcompile\s*\(/g, name: 'compile()', languages: ['python'] },
        ];

        for (const { pattern, name, languages } of patterns) {
            // Skip if language specified and doesn't match
            if (languages && context?.language && !languages.includes(context.language)) {
                continue;
            }

            pattern.lastIndex = 0;
            let match;

            while ((match = pattern.exec(content)) !== null) {
                const line = findLineNumber(content, match.index);

                // Check if in a comment
                const lineContent = content.split('\n')[line - 1] || '';
                if (lineContent.trim().startsWith('//') || lineContent.trim().startsWith('#')) {
                    continue;
                }

                findings.push({
                    id: generateFindingId('security/unsafe-eval', { line }),
                    ruleId: 'security/unsafe-eval',
                    category: 'security',
                    severity: 'high',
                    title: `Unsafe ${name} usage`,
                    description: `Use of ${name} can lead to code injection vulnerabilities`,
                    location: { line },
                    match: match[0],
                    suggestion: 'Avoid using eval/exec. Use safer alternatives like JSON.parse for data or function references for callbacks.',
                    confidence: 0.9
                });
            }
        }

        return findings;
    }
};

/**
 * Console log detection rule
 */
export const consoleLogRule: ScanRule = {
    id: 'quality/console-log',
    name: 'Console Logs',
    category: 'quality',
    severity: 'info',
    description: 'Detects console.log and similar debug statements',
    enabledByDefault: false,
    tags: ['quality', 'debug', 'cleanup'],
    check: (content: string): Finding[] => {
        const findings: Finding[] = [];
        const pattern = /console\.(log|debug|info|warn|error|trace)\s*\(/g;

        let match;
        while ((match = pattern.exec(content)) !== null) {
            const line = findLineNumber(content, match.index);

            findings.push({
                id: generateFindingId('quality/console-log', { line }),
                ruleId: 'quality/console-log',
                category: 'quality',
                severity: 'info',
                title: `console.${match[1]} statement`,
                description: 'Debug logging statement found',
                location: { line },
                match: match[0],
                suggestion: 'Remove debug logging before production',
                confidence: 1.0
            });
        }

        return findings;
    }
};

/**
 * Profanity/inappropriate content detection
 */
export const profanityRule: ScanRule = {
    id: 'compliance/profanity',
    name: 'Profanity Detection',
    category: 'compliance',
    severity: 'medium',
    description: 'Detects potentially inappropriate language',
    enabledByDefault: false,
    tags: ['compliance', 'content', 'moderation'],
    check: (content: string): Finding[] => {
        // Basic word list (would be more comprehensive in production)
        const profanityPatterns = [
            /\b(damn|hell|crap)\b/gi, // Mild
        ];

        const findings: Finding[] = [];

        for (const pattern of profanityPatterns) {
            let match;
            while ((match = pattern.exec(content)) !== null) {
                const line = findLineNumber(content, match.index);

                findings.push({
                    id: generateFindingId('compliance/profanity', { line }),
                    ruleId: 'compliance/profanity',
                    category: 'compliance',
                    severity: 'low',
                    title: 'Potentially inappropriate language',
                    description: 'Content may contain inappropriate language',
                    location: { line },
                    match: '***',
                    suggestion: 'Consider using more professional language',
                    confidence: 0.7
                });
            }
        }

        return findings;
    }
};

/**
 * All built-in rules
 */
export const BUILTIN_RULES: ScanRule[] = [
    sqlInjectionRule,
    xssRule,
    secretsRule,
    hallucinationRule,
    piiRule,
    todoRule,
    unsafeEvalRule,
    consoleLogRule,
    profanityRule
];

/**
 * Get rules by tag
 */
export function getRulesByTag(tag: string): ScanRule[] {
    return BUILTIN_RULES.filter(rule => rule.tags?.includes(tag));
}

/**
 * Get rules by category
 */
export function getRulesByCategory(category: string): ScanRule[] {
    return BUILTIN_RULES.filter(rule => rule.category === category);
}

/**
 * Get enabled-by-default rules
 */
export function getDefaultRules(): ScanRule[] {
    return BUILTIN_RULES.filter(rule => rule.enabledByDefault);
}

/**
 * Get a rule by ID
 */
export function getRuleById(id: string): ScanRule | undefined {
    return BUILTIN_RULES.find(rule => rule.id === id);
}
