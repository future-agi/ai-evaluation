/**
 * XSS (Cross-Site Scripting) detection metric.
 * Detects potential XSS vulnerabilities in code.
 *
 * @module local/metrics/code/xss-detection
 */

import {
    CodeInput,
    CodeSecurityConfig,
    CodeSecurityResult,
    SecurityIssue,
    createSecurityResult
} from './types';

/**
 * Patterns that indicate XSS vulnerabilities
 */
const XSS_PATTERNS: Array<{
    pattern: RegExp;
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    suggestion: string;
}> = [
    // innerHTML with user input
    {
        pattern: /\.innerHTML\s*=\s*(?:req\.|request\.|params\.|query\.|body\.|user|input|\$\{|`)/gi,
        severity: 'critical',
        description: 'innerHTML assignment with potentially untrusted input',
        suggestion: 'Use textContent or sanitize HTML before insertion'
    },
    // innerHTML with concatenation
    {
        pattern: /\.innerHTML\s*=\s*["'`]?[^"'`]*\+/gi,
        severity: 'high',
        description: 'innerHTML with string concatenation',
        suggestion: 'Use textContent or a sanitization library like DOMPurify'
    },
    // outerHTML with user input
    {
        pattern: /\.outerHTML\s*=\s*(?:req\.|request\.|params\.|query\.|body\.|user|input|\$\{)/gi,
        severity: 'critical',
        description: 'outerHTML assignment with potentially untrusted input',
        suggestion: 'Use textContent or sanitize HTML before insertion'
    },
    // document.write with user input
    {
        pattern: /document\.write\s*\(\s*(?:req\.|request\.|params\.|query\.|body\.|user|input|\$\{|`)/gi,
        severity: 'critical',
        description: 'document.write with potentially untrusted input',
        suggestion: 'Avoid document.write; use DOM manipulation methods'
    },
    // document.write with concatenation
    {
        pattern: /document\.write\s*\([^)]*\+/gi,
        severity: 'high',
        description: 'document.write with string concatenation',
        suggestion: 'Avoid document.write; use DOM manipulation methods'
    },
    // eval with user input
    {
        pattern: /eval\s*\(\s*(?:req\.|request\.|params\.|query\.|body\.|user|input|\$\{)/gi,
        severity: 'critical',
        description: 'eval() with potentially untrusted input',
        suggestion: 'Never use eval with user input; use JSON.parse for data'
    },
    // setTimeout/setInterval with string
    {
        pattern: /set(?:Timeout|Interval)\s*\(\s*["'`][^"'`]*(?:req\.|request\.|params\.|query\.|body\.|user|input)/gi,
        severity: 'high',
        description: 'setTimeout/setInterval with string containing user input',
        suggestion: 'Pass a function reference instead of a string'
    },
    // dangerouslySetInnerHTML in React
    {
        pattern: /dangerouslySetInnerHTML\s*=\s*\{\s*\{\s*__html\s*:\s*(?:req\.|request\.|params\.|query\.|body\.|user|input|props\.|state\.)/gi,
        severity: 'critical',
        description: 'React dangerouslySetInnerHTML with potentially untrusted input',
        suggestion: 'Sanitize HTML with DOMPurify before using dangerouslySetInnerHTML'
    },
    // dangerouslySetInnerHTML without sanitization mention
    {
        pattern: /dangerouslySetInnerHTML\s*=\s*\{/gi,
        severity: 'medium',
        description: 'React dangerouslySetInnerHTML usage detected',
        suggestion: 'Ensure HTML is sanitized before using dangerouslySetInnerHTML'
    },
    // v-html in Vue
    {
        pattern: /v-html\s*=\s*["'](?:user|input|query|params|request)/gi,
        severity: 'critical',
        description: 'Vue v-html with potentially untrusted input',
        suggestion: 'Sanitize HTML before using v-html directive'
    },
    // [innerHTML] in Angular
    {
        pattern: /\[innerHTML\]\s*=\s*["'](?:user|input|query|params|request)/gi,
        severity: 'critical',
        description: 'Angular innerHTML binding with potentially untrusted input',
        suggestion: 'Use Angular DomSanitizer or sanitize HTML manually'
    },
    // jQuery html() with user input
    {
        pattern: /\$\([^)]+\)\.html\s*\(\s*(?:req\.|request\.|params\.|query\.|body\.|user|input)/gi,
        severity: 'critical',
        description: 'jQuery .html() with potentially untrusted input',
        suggestion: 'Use .text() or sanitize HTML before using .html()'
    },
    // jQuery append/prepend with user input
    {
        pattern: /\$\([^)]+\)\.(?:append|prepend)\s*\(\s*["'`]?<[^>]*(?:req\.|request\.|params\.|query\.|body\.|user|input)/gi,
        severity: 'high',
        description: 'jQuery append/prepend with HTML containing user input',
        suggestion: 'Create elements programmatically or sanitize HTML'
    },
    // location.href assignment
    {
        pattern: /(?:window\.)?location(?:\.href)?\s*=\s*(?:req\.|request\.|params\.|query\.|body\.|user|input)/gi,
        severity: 'high',
        description: 'Location redirect with potentially untrusted input',
        suggestion: 'Validate and whitelist redirect URLs'
    },
    // URL parameter in script src
    {
        pattern: /<script[^>]*src\s*=\s*["']?\s*(?:\+|`|\$\{)/gi,
        severity: 'critical',
        description: 'Dynamic script src attribute',
        suggestion: 'Never use dynamic URLs for script sources'
    }
];

/**
 * Detect XSS vulnerabilities in code.
 *
 * @param input - Code to analyze
 * @param config - Configuration options
 * @returns Security analysis result
 *
 * @example
 * ```typescript
 * const result = xssDetection({
 *     code: 'element.innerHTML = userInput;'
 * });
 * console.log(result.passed); // false
 * console.log(result.issues[0].type); // 'xss'
 * ```
 */
export function xssDetection(
    input: CodeInput,
    config: CodeSecurityConfig = {}
): CodeSecurityResult {
    const { code, language } = input;
    const issues: SecurityIssue[] = [];

    if (!code || code.trim().length === 0) {
        return createSecurityResult([], config);
    }

    const lines = code.split('\n');

    // Check each pattern
    for (const { pattern, severity, description, suggestion } of XSS_PATTERNS) {
        // Reset regex state
        pattern.lastIndex = 0;

        let match;
        while ((match = pattern.exec(code)) !== null) {
            // Find line number
            const matchIndex = match.index;
            let lineNumber = 1;
            let charCount = 0;
            for (let i = 0; i < lines.length; i++) {
                charCount += lines[i].length + 1;
                if (charCount > matchIndex) {
                    lineNumber = i + 1;
                    break;
                }
            }

            // Check if in comment
            const line = lines[lineNumber - 1] || '';
            const isComment = config.checkComments === false ? false :
                line.trim().startsWith('//') ||
                line.trim().startsWith('#') ||
                line.trim().startsWith('*') ||
                line.trim().startsWith('{/*');

            if (!isComment) {
                issues.push({
                    type: 'xss',
                    severity,
                    line: lineNumber,
                    match: match[0].substring(0, 100),
                    description,
                    suggestion
                });
            }
        }
    }

    // Check custom patterns
    if (config.customPatterns) {
        for (const patternStr of config.customPatterns) {
            try {
                const customPattern = new RegExp(patternStr, 'gi');
                let match;
                while ((match = customPattern.exec(code)) !== null) {
                    issues.push({
                        type: 'xss-custom',
                        severity: 'medium',
                        match: match[0].substring(0, 100),
                        description: 'Custom XSS pattern matched'
                    });
                }
            } catch {
                // Invalid regex, skip
            }
        }
    }

    const result = createSecurityResult(issues, config);
    result.language = language;
    return result;
}

/**
 * Alias for xssDetection
 */
export const noXss = xssDetection;
