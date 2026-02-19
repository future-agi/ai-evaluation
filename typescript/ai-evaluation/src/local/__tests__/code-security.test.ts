/**
 * Tests for code security metrics
 */

import {
    sqlInjection,
    noSqlInjection,
    xssDetection,
    noXss,
    secretsDetection,
    noSecrets,
    noHardcodedSecrets,
    allSecurityChecks,
    codeSecurityScan,
    CodeInput,
    CodeSecurityConfig,
    SEVERITY_LEVELS,
    meetsMinSeverity
} from '../metrics/code';

describe('Code Security Metrics', () => {
    describe('SQL Injection Detection', () => {
        it('should detect string concatenation in SQL queries', () => {
            const code = `
                const userId = req.params.id;
                const query = "SELECT * FROM users WHERE id = " + userId;
                db.query(query);
            `;
            const result = sqlInjection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.length).toBeGreaterThan(0);
            expect(result.issues[0].type).toBe('sql-injection');
        });

        it('should detect template literal interpolation in SQL', () => {
            const code = `
                const query = \`SELECT * FROM users WHERE id = \${req.params.id}\`;
            `;
            const result = sqlInjection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.length).toBeGreaterThan(0);
        });

        it('should detect Python f-string interpolation in SQL', () => {
            const code = `
                query = f"SELECT * FROM users WHERE id = {user_id}"
                cursor.execute(query)
            `;
            const result = sqlInjection({ code, language: 'python' });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.description.includes('f-string'))).toBe(true);
        });

        it('should detect Python .format() in SQL', () => {
            const code = `
                query = "DELETE FROM users WHERE id = {}".format(user_id)
            `;
            const result = sqlInjection({ code, language: 'python' });
            expect(result.passed).toBe(false);
        });

        it('should pass for safe parameterized queries', () => {
            const code = `
                // Using parameterized query
                const query = "SELECT * FROM users WHERE id = ?";
                db.query(query, [userId]);
            `;
            const result = sqlInjection({ code });
            expect(result.passed).toBe(true);
            expect(result.issues.length).toBe(0);
        });

        it('should detect ORDER BY with dynamic input', () => {
            const code = `
                const orderBy = "ORDER BY " + req.query.sort;
            `;
            const result = sqlInjection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.description.includes('ORDER BY'))).toBe(true);
        });

        it('should ignore issues in comments when checkComments is true (default)', () => {
            const code = `
// SELECT * FROM users WHERE id = " + userId
const safeQuery = "SELECT * FROM users WHERE id = ?";
            `;
            // Default behavior skips comments
            const result = sqlInjection({ code });
            expect(result.passed).toBe(true);
        });

        it('noSqlInjection should be an alias', () => {
            expect(noSqlInjection).toBe(sqlInjection);
        });
    });

    describe('XSS Detection', () => {
        it('should detect innerHTML with user input', () => {
            const code = `
                element.innerHTML = userInput;
            `;
            const result = xssDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'xss')).toBe(true);
        });

        it('should detect innerHTML with template literal', () => {
            const code = `
                element.innerHTML = \`<div>\${userInput}</div>\`;
            `;
            const result = xssDetection({ code });
            expect(result.passed).toBe(false);
        });

        it('should detect document.write with user input', () => {
            const code = `
                document.write(userInput);
            `;
            const result = xssDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.description.includes('document.write'))).toBe(true);
        });

        it('should detect dangerouslySetInnerHTML in React', () => {
            const code = `
                return <div dangerouslySetInnerHTML={{ __html: userInput }} />;
            `;
            const result = xssDetection({ code });
            expect(result.passed).toBe(false);
        });

        it('should detect v-html in Vue', () => {
            const code = `
                <div v-html="userInput"></div>
            `;
            const result = xssDetection({ code });
            expect(result.passed).toBe(false);
        });

        it('should detect jQuery .html() with user input', () => {
            const code = `
                $("#element").html(userInput);
            `;
            const result = xssDetection({ code });
            expect(result.passed).toBe(false);
        });

        it('should pass for safe textContent usage', () => {
            const code = `
                element.textContent = userInput;
            `;
            const result = xssDetection({ code });
            expect(result.passed).toBe(true);
        });

        it('should detect eval with user input', () => {
            const code = `
                eval(userInput);
            `;
            const result = xssDetection({ code });
            expect(result.passed).toBe(false);
        });

        it('noXss should be an alias', () => {
            expect(noXss).toBe(xssDetection);
        });
    });

    describe('Secrets Detection', () => {
        it('should detect AWS access key', () => {
            const code = `
                const awsKey = "AKIAIOSFODNN7EXAMPLE";
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'aws-access-key')).toBe(true);
            expect(result.severityCounts.critical).toBeGreaterThan(0);
        });

        it('should detect OpenAI API key', () => {
            // OpenAI keys are sk- followed by 48 alphanumeric chars
            const code = `
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'openai-api-key')).toBe(true);
        });

        it('should detect Anthropic API key', () => {
            const code = `
                const key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789AB";
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'anthropic-api-key')).toBe(true);
        });

        it('should detect GitHub token', () => {
            const code = `
                const token = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'github-token')).toBe(true);
        });

        it('should detect Stripe API key', () => {
            // Fake key assembled at runtime so it doesn't trigger push protection
            const fakeStripeKey = ['sk', 'live', 'abcdefghijklmnopqrstuvwx'].join('_');
            const code = `
                const stripeKey = "${fakeStripeKey}";
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'stripe-api-key')).toBe(true);
        });

        it('should detect private key', () => {
            const code = `
                const key = \`-----BEGIN RSA PRIVATE KEY-----
                MIIEpAIBAAKCAQEA...
                -----END RSA PRIVATE KEY-----\`;
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'private-key')).toBe(true);
        });

        it('should detect hardcoded password', () => {
            const code = `
                const password = "mysecretpassword123";
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'hardcoded-password')).toBe(true);
        });

        it('should detect database connection string with credentials', () => {
            const code = `
                const dbUrl = "mongodb://admin:password123@localhost:27017/mydb";
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'database-connection-string')).toBe(true);
        });

        it('should detect JWT token', () => {
            const code = `
                const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'jwt-token')).toBe(true);
        });

        it('should detect SendGrid API key', () => {
            // SendGrid keys are SG.{22 chars}.{43 chars}
            // Fake key assembled at runtime so it doesn't trigger push protection
            const fakeSendGridKey = ['SG', '1234567890abcdefghijkl', '1234567890abcdefghijklmnopqrstuvwxyzABCDEFG'].join('.');
            const code = `
                const sg = "${fakeSendGridKey}";
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'sendgrid-api-key')).toBe(true);
        });

        it('should pass for environment variable references', () => {
            const code = `
                const apiKey = process.env.API_KEY;
                const secret = os.environ.get('SECRET');
            `;
            const result = secretsDetection({ code });
            expect(result.passed).toBe(true);
        });

        it('should mask secrets in output', () => {
            const code = `
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
            `;
            const result = secretsDetection({ code });
            // Check that the match is masked
            const matchedSecret = result.issues[0]?.match || '';
            expect(matchedSecret).toContain('***');
            expect(matchedSecret).not.toBe('sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL');
        });

        it('should support custom patterns', () => {
            const code = `
                const mySecret = "CUSTOM_SECRET_12345";
            `;
            const result = secretsDetection({ code }, {
                customPatterns: ['CUSTOM_SECRET_\\d+']
            });
            expect(result.passed).toBe(false);
            expect(result.issues.some(i => i.type === 'custom-secret')).toBe(true);
        });

        it('noSecrets and noHardcodedSecrets should be aliases', () => {
            expect(noSecrets).toBe(secretsDetection);
            expect(noHardcodedSecrets).toBe(secretsDetection);
        });
    });

    describe('Combined Security Checks', () => {
        it('should detect multiple vulnerability types', () => {
            // Use a valid 48-char OpenAI key pattern
            const code = `
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
                const query = "SELECT * FROM users WHERE id = " + req.params.id;
                element.innerHTML = userInput;
            `;
            const result = allSecurityChecks({ code });
            expect(result.passed).toBe(false);
            expect(result.issues.length).toBeGreaterThan(2);

            // Check for all vulnerability types
            const types = new Set(result.issues.map(i => i.type));
            expect(types.has('openai-api-key')).toBe(true);
            expect(types.has('sql-injection')).toBe(true);
            expect(types.has('xss')).toBe(true);
        });

        it('should pass for clean code', () => {
            const code = `
                const apiKey = process.env.API_KEY;
                const query = "SELECT * FROM users WHERE id = ?";
                db.query(query, [userId]);
                element.textContent = sanitize(userInput);
            `;
            const result = allSecurityChecks({ code });
            expect(result.passed).toBe(true);
            expect(result.issues.length).toBe(0);
        });

        it('codeSecurityScan should be an alias', () => {
            expect(codeSecurityScan).toBe(allSecurityChecks);
        });

        it('should correctly count severities', () => {
            // AWS key is critical, OpenAI is critical, JWT is medium
            const code = `
                const awsKey = "AKIAIOSFODNN7EXAMPLE";
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
                const jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
            `;
            const result = allSecurityChecks({ code });
            expect(result.severityCounts.critical).toBeGreaterThanOrEqual(2);
            expect(result.severityCounts.medium).toBeGreaterThanOrEqual(1);
        });
    });

    describe('Severity Configuration', () => {
        it('should filter by minimum severity', () => {
            const code = `
                const jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
            `;
            // JWT is medium severity
            const resultAll = secretsDetection({ code }, { minSeverity: 'low' });
            const resultHighOnly = secretsDetection({ code }, { minSeverity: 'high' });

            expect(resultAll.issues.length).toBeGreaterThan(0);
            expect(resultHighOnly.issues.length).toBe(0);
        });

        it('SEVERITY_LEVELS should have correct values', () => {
            expect(SEVERITY_LEVELS['low']).toBe(1);
            expect(SEVERITY_LEVELS['medium']).toBe(2);
            expect(SEVERITY_LEVELS['high']).toBe(3);
            expect(SEVERITY_LEVELS['critical']).toBe(4);
        });

        it('meetsMinSeverity should work correctly', () => {
            expect(meetsMinSeverity('critical', 'low')).toBe(true);
            expect(meetsMinSeverity('low', 'critical')).toBe(false);
            expect(meetsMinSeverity('high', 'high')).toBe(true);
        });
    });

    describe('Empty/Invalid Input', () => {
        it('should handle empty code', () => {
            const result = allSecurityChecks({ code: '' });
            expect(result.passed).toBe(true);
            expect(result.issues.length).toBe(0);
        });

        it('should handle whitespace-only code', () => {
            const result = allSecurityChecks({ code: '   \n\t  ' });
            expect(result.passed).toBe(true);
            expect(result.issues.length).toBe(0);
        });
    });

    describe('Score Calculation', () => {
        it('should return score of 1.0 for clean code', () => {
            const code = `const x = 1;`;
            const result = allSecurityChecks({ code });
            expect(result.score).toBe(1.0);
        });

        it('should decrease score with issues', () => {
            const code = `
                const apiKey = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL";
            `;
            const result = secretsDetection({ code });
            expect(result.score).toBeLessThan(1.0);
        });

        it('should weigh critical issues more heavily', () => {
            const codeWithCritical = `const key = "AKIAIOSFODNN7EXAMPLE";`;
            const codeWithMedium = `const jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";`;

            const criticalResult = secretsDetection({ code: codeWithCritical });
            const mediumResult = secretsDetection({ code: codeWithMedium });

            expect(criticalResult.score).toBeLessThan(mediumResult.score);
        });
    });

    describe('Language Support', () => {
        it('should preserve language in result', () => {
            const code = `print("hello")`;
            const result = allSecurityChecks({ code, language: 'python' });
            expect(result.language).toBe('python');
        });

        it('should detect Python-specific patterns', () => {
            const pythonCode = `
import sqlite3
user_id = input("Enter ID: ")
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
            `;
            const result = sqlInjection({ code: pythonCode, language: 'python' });
            expect(result.passed).toBe(false);
        });
    });
});
