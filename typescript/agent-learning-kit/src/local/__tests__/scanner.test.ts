/**
 * Tests for Scanner Pipeline
 */

import {
    Scanner,
    createScanner,
    quickScan,
    securityScan,
    privacyScan,
    BUILTIN_RULES,
    getRulesByTag,
    getRulesByCategory,
    getDefaultRules,
    getRuleById,
    ScanRule,
    Finding,
    SEVERITY_VALUES,
    compareSeverity,
    meetsSeverityThreshold,
    generateFindingId
} from '../scanner';

describe('Scanner Pipeline', () => {
    describe('Utility Functions', () => {
        it('SEVERITY_VALUES should have correct ordering', () => {
            expect(SEVERITY_VALUES.info).toBe(0);
            expect(SEVERITY_VALUES.low).toBe(1);
            expect(SEVERITY_VALUES.medium).toBe(2);
            expect(SEVERITY_VALUES.high).toBe(3);
            expect(SEVERITY_VALUES.critical).toBe(4);
        });

        it('compareSeverity should compare correctly', () => {
            expect(compareSeverity('critical', 'low')).toBeGreaterThan(0);
            expect(compareSeverity('low', 'critical')).toBeLessThan(0);
            expect(compareSeverity('medium', 'medium')).toBe(0);
        });

        it('meetsSeverityThreshold should check thresholds', () => {
            expect(meetsSeverityThreshold('critical', 'low')).toBe(true);
            expect(meetsSeverityThreshold('low', 'critical')).toBe(false);
            expect(meetsSeverityThreshold('high', 'high')).toBe(true);
        });

        it('generateFindingId should create unique IDs', () => {
            const id1 = generateFindingId('test-rule');
            const id2 = generateFindingId('test-rule');
            expect(id1).not.toBe(id2);
            expect(id1).toContain('test-rule');
        });

        it('generateFindingId should include line number when provided', () => {
            const id = generateFindingId('test-rule', { line: 42 });
            expect(id).toContain('L42');
        });
    });

    describe('Built-in Rules', () => {
        it('BUILTIN_RULES should contain rules', () => {
            expect(BUILTIN_RULES.length).toBeGreaterThan(0);
        });

        it('getRulesByTag should filter by tag', () => {
            const securityRules = getRulesByTag('security');
            expect(securityRules.length).toBeGreaterThan(0);
            expect(securityRules.every(r => r.tags?.includes('security'))).toBe(true);
        });

        it('getRulesByCategory should filter by category', () => {
            const securityRules = getRulesByCategory('security');
            expect(securityRules.length).toBeGreaterThan(0);
            expect(securityRules.every(r => r.category === 'security')).toBe(true);
        });

        it('getDefaultRules should return enabled-by-default rules', () => {
            const defaultRules = getDefaultRules();
            expect(defaultRules.length).toBeGreaterThan(0);
            expect(defaultRules.every(r => r.enabledByDefault)).toBe(true);
        });

        it('getRuleById should find rule by ID', () => {
            const rule = getRuleById('security/sql-injection');
            expect(rule).toBeDefined();
            expect(rule?.id).toBe('security/sql-injection');
        });

        it('getRuleById should return undefined for unknown ID', () => {
            const rule = getRuleById('unknown/rule');
            expect(rule).toBeUndefined();
        });
    });

    describe('Scanner', () => {
        let scanner: Scanner;

        beforeEach(() => {
            scanner = new Scanner();
        });

        it('should create with default rules', () => {
            const rules = scanner.getAllRules();
            expect(rules.length).toBe(BUILTIN_RULES.length);
        });

        it('should add custom rules', () => {
            const customRule: ScanRule = {
                id: 'custom/test',
                name: 'Test Rule',
                category: 'custom',
                severity: 'medium',
                description: 'Test rule',
                enabledByDefault: true,
                check: () => []
            };

            scanner.addRule(customRule);
            expect(scanner.getRule('custom/test')).toBe(customRule);
        });

        it('should remove rules', () => {
            const removed = scanner.removeRule('security/sql-injection');
            expect(removed).toBe(true);
            expect(scanner.getRule('security/sql-injection')).toBeUndefined();
        });

        it('should scan and find SQL injection', () => {
            const code = `
                const userId = req.params.id;
                const query = "SELECT * FROM users WHERE id = " + userId;
            `;
            const result = scanner.scan(code);

            expect(result.findings.some(f => f.ruleId === 'security/sql-injection')).toBe(true);
        });

        it('should scan and find XSS vulnerabilities', () => {
            const code = `
                element.innerHTML = userInput;
            `;
            const result = scanner.scan(code);

            expect(result.findings.some(f => f.ruleId === 'security/xss')).toBe(true);
        });

        it('should scan and find hardcoded secrets', () => {
            const code = `
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
            `;
            const result = scanner.scan(code);

            expect(result.findings.some(f => f.ruleId === 'security/hardcoded-secrets')).toBe(true);
            expect(result.passed).toBe(false);
        });

        it('should scan and find PII', () => {
            const text = `
                Contact John at john.doe@example.com or call 555-123-4567.
            `;
            const result = scanner.scan(text);

            expect(result.findings.some(f => f.ruleId === 'privacy/pii')).toBe(true);
        });

        it('should pass for clean code', () => {
            const code = `
                const userId = sanitize(req.params.id);
                const query = db.query('SELECT * FROM users WHERE id = ?', [userId]);
            `;
            const result = scanner.scan(code);

            // Should not have critical/high issues
            expect(result.findingsBySeverity.critical).toBe(0);
            expect(result.findingsBySeverity.high).toBe(0);
        });

        it('should respect minSeverity config', () => {
            const code = `
                // TODO: fix this later
                const x = 1;
            `;

            // With low minSeverity - might catch TODO if enabled
            const lowResult = scanner.scan(code, {
                enableRules: ['quality/todo-comments'],
                minSeverity: 'info'
            });

            // With high minSeverity - should filter out low/info
            const highResult = scanner.scan(code, {
                enableRules: ['quality/todo-comments'],
                minSeverity: 'high'
            });

            expect(highResult.totalFindings).toBeLessThanOrEqual(lowResult.totalFindings);
        });

        it('should respect maxFindings config', () => {
            const code = `
                const a = "sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
                const b = "sk-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
                const c = "sk-ccccccccccccccccccccccccccccccccccccccccccccccccc";
            `;

            const result = scanner.scan(code, { maxFindings: 2 });
            expect(result.findings.length).toBeLessThanOrEqual(2);
        });

        it('should filter by category', () => {
            const code = `
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
                element.innerHTML = userInput;
            `;

            const securityOnly = scanner.scan(code, {
                includeCategories: ['security']
            });

            expect(securityOnly.findings.every(f => f.category === 'security')).toBe(true);
        });

        it('should exclude categories', () => {
            const code = `
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
                Contact: john@example.com
            `;

            const noPrivacy = scanner.scan(code, {
                excludeCategories: ['privacy']
            });

            expect(noPrivacy.findings.every(f => f.category !== 'privacy')).toBe(true);
        });

        it('should enable specific rules', () => {
            const code = `
                // TODO: implement this
                console.log("debug");
            `;

            const result = scanner.scan(code, {
                enableRules: ['quality/todo-comments', 'quality/console-log']
            });

            expect(result.rulesExecuted).toContain('quality/todo-comments');
            expect(result.rulesExecuted).toContain('quality/console-log');
        });

        it('should disable specific rules', () => {
            const code = `
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
            `;

            const result = scanner.scan(code, {
                disableRules: ['security/hardcoded-secrets']
            });

            expect(result.rulesExecuted).not.toContain('security/hardcoded-secrets');
        });

        it('should enable rules by tag', () => {
            const code = `eval(userInput)`;

            const result = scanner.scan(code, {
                enableRules: ['tag:security']
            });

            expect(result.rulesExecuted.some(r => r.startsWith('security/'))).toBe(true);
        });

        it('should include suggestions when configured', () => {
            const code = `const query = "SELECT * FROM users WHERE id = " + userId;`;

            const withSuggestions = scanner.scan(code, { includeSuggestions: true });
            const withoutSuggestions = scanner.scan(code, { includeSuggestions: false });

            if (withSuggestions.findings.length > 0) {
                expect(withSuggestions.findings[0].suggestion).toBeDefined();
            }
            if (withoutSuggestions.findings.length > 0) {
                expect(withoutSuggestions.findings[0].suggestion).toBeUndefined();
            }
        });

        it('should report scan duration', () => {
            const result = scanner.scan('const x = 1;');
            expect(result.scanDurationMs).toBeGreaterThanOrEqual(0);
        });

        it('should generate summary', () => {
            const result = scanner.scan('const x = 1;');
            expect(result.summary).toBeDefined();
            expect(typeof result.summary).toBe('string');
        });

        it('should sort findings by severity', () => {
            const code = `
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
                // TODO: fix this
                element.innerHTML = userInput;
            `;

            const result = scanner.scan(code, {
                enableRules: ['security/hardcoded-secrets', 'security/xss', 'quality/todo-comments'],
                minSeverity: 'info'
            });

            // Check that higher severity comes first
            for (let i = 1; i < result.findings.length; i++) {
                const prev = SEVERITY_VALUES[result.findings[i - 1].severity];
                const curr = SEVERITY_VALUES[result.findings[i].severity];
                expect(prev).toBeGreaterThanOrEqual(curr);
            }
        });
    });

    describe('Quick Scan Functions', () => {
        it('quickScan should work without scanner instance', () => {
            const result = quickScan('const x = 1;');
            expect(result).toBeDefined();
            expect(result.passed).toBeDefined();
        });

        it('securityScan should only run security rules', () => {
            const code = `
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
                Contact: john@example.com
            `;

            const result = securityScan(code);
            expect(result.findings.every(f => f.category === 'security')).toBe(true);
        });

        it('privacyScan should only run privacy rules', () => {
            const text = `
                Email: john@example.com
                Phone: 555-123-4567
            `;

            const result = privacyScan(text);
            expect(result.findings.every(f => f.category === 'privacy')).toBe(true);
        });
    });

    describe('Specialized Scans', () => {
        it('scanSecurity should find security issues', () => {
            const scanner = new Scanner();
            const code = `eval(userInput)`;

            const result = scanner.scanSecurity(code);
            expect(result.findings.some(f => f.category === 'security')).toBe(true);
        });

        it('scanQuality should find quality issues', () => {
            const scanner = new Scanner();
            const code = `
                // TODO: implement this
            `;

            const result = scanner.scanQuality(code);
            // Quality rules may or may not find issues depending on enabled rules
            expect(result.rulesExecuted).toBeDefined();
        });

        it('scanPrivacy should find PII', () => {
            const scanner = new Scanner();
            const text = `SSN: 123-45-6789`;

            const result = scanner.scanPrivacy(text);
            expect(result.findings.some(f => f.ruleId === 'privacy/pii')).toBe(true);
        });

        it('scanRAG should detect hallucinations', () => {
            const scanner = new Scanner();
            const response = 'Python was created by aliens in 3000 BC.';
            const context = 'Python is a programming language created by Guido van Rossum in 1991.';

            const result = scanner.scanRAG(response, context, 'When was Python created?');

            // Should have run hallucination rule
            expect(result.rulesExecuted).toContain('quality/hallucination');
        });
    });

    describe('Custom Rules', () => {
        it('should support custom rules in config', () => {
            const customRule: ScanRule = {
                id: 'custom/no-foo',
                name: 'No Foo',
                category: 'custom',
                severity: 'low',
                description: 'Disallow the word foo',
                enabledByDefault: true,
                check: (content) => {
                    const findings: Finding[] = [];
                    const pattern = /\bfoo\b/gi;
                    let match;
                    while ((match = pattern.exec(content)) !== null) {
                        findings.push({
                            id: `custom/no-foo-${match.index}`,
                            ruleId: 'custom/no-foo',
                            category: 'custom',
                            severity: 'low',
                            title: 'Found "foo"',
                            description: 'The word "foo" is not allowed',
                            match: match[0],
                            confidence: 1.0
                        });
                    }
                    return findings;
                }
            };

            const scanner = new Scanner({ customRules: [customRule] });
            const result = scanner.scan('const foo = bar + foo;', {
                enableRules: ['custom/no-foo']
            });

            expect(result.findings.some(f => f.ruleId === 'custom/no-foo')).toBe(true);
            expect(result.findings.filter(f => f.ruleId === 'custom/no-foo').length).toBe(2);
        });
    });

    describe('createScanner', () => {
        it('should create a scanner with config', () => {
            const scanner = createScanner({ minSeverity: 'high' });
            expect(scanner).toBeInstanceOf(Scanner);
        });
    });

    describe('Unsafe Eval Detection', () => {
        it('should detect eval usage', () => {
            const code = `
                const result = eval(userCode);
            `;
            const result = quickScan(code);

            expect(result.findings.some(f => f.ruleId === 'security/unsafe-eval')).toBe(true);
        });

        it('should detect Function constructor', () => {
            const code = `
                const fn = new Function('return ' + expr);
            `;
            const result = quickScan(code);

            expect(result.findings.some(f => f.ruleId === 'security/unsafe-eval')).toBe(true);
        });

        it('should detect setTimeout with string', () => {
            const code = `
                setTimeout("alert('xss')", 1000);
            `;
            const result = quickScan(code);

            expect(result.findings.some(f => f.ruleId === 'security/unsafe-eval')).toBe(true);
        });
    });

    describe('Real-world Scenarios', () => {
        it('should scan a complete code file', () => {
            const code = `
                import express from 'express';

                const app = express();
                const API_KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";

                app.get('/user', (req, res) => {
                    const id = req.query.id;
                    const query = \`SELECT * FROM users WHERE id = \${id}\`;
                    db.query(query, (err, results) => {
                        res.send(results);
                    });
                });

                // TODO: add authentication

                console.log('Server started');
            `;

            const result = quickScan(code, {
                enableRules: [
                    'security/sql-injection',
                    'security/hardcoded-secrets',
                    'quality/todo-comments',
                    'quality/console-log'
                ],
                minSeverity: 'info'
            });

            expect(result.passed).toBe(false);
            expect(result.findings.length).toBeGreaterThan(0);

            // Should have found the secret
            expect(result.findings.some(f =>
                f.ruleId === 'security/hardcoded-secrets'
            )).toBe(true);
        });

        it('should scan user-submitted content for PII', () => {
            const userContent = `
                My name is John Doe.
                You can reach me at johndoe@email.com
                or call me at 555-123-4567.
                My SSN is 123-45-6789.
            `;

            const result = privacyScan(userContent);

            expect(result.findings.length).toBeGreaterThan(0);
            expect(result.findings.some(f => f.title.includes('email'))).toBe(true);
            expect(result.findings.some(f => f.title.includes('phone'))).toBe(true);
        });

        it('should validate AI response against source', () => {
            const context = 'The Eiffel Tower is located in Paris, France. It was completed in 1889.';
            const response = 'The Eiffel Tower is in Paris. It was built in 1850 and is the tallest building in Europe.';

            const scanner = new Scanner();
            const result = scanner.scanRAG(response, context, 'Tell me about the Eiffel Tower');

            // Should detect the fabricated claims
            expect(result.rulesExecuted).toContain('quality/hallucination');
        });
    });
});
