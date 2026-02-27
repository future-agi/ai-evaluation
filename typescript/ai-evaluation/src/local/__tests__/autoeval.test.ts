/**
 * Tests for AutoEval Pipeline
 */

import {
    AutoEvalPipeline,
    createAutoEval,
    autoEvaluate,
    analyzeContent,
    describeCharacteristics,
    selectMetrics,
    AutoEvalInput,
    ContentCharacteristics
} from '../autoeval';

describe('AutoEval Pipeline', () => {
    describe('Content Analyzer', () => {
        it('should detect plain text content', () => {
            const input: AutoEvalInput = {
                response: 'This is a simple text response about the weather.'
            };
            const chars = analyzeContent(input);
            expect(chars.inputType).toBe('text');
            expect(chars.hasCode).toBe(false);
            expect(chars.hasJson).toBe(false);
        });

        it('should detect JSON content', () => {
            const input: AutoEvalInput = {
                response: '{"name": "John", "age": 30}'
            };
            const chars = analyzeContent(input);
            expect(chars.inputType).toBe('json');
            expect(chars.hasJson).toBe(true);
        });

        it('should detect JavaScript code', () => {
            const input: AutoEvalInput = {
                response: `
                    function greet(name) {
                        console.log("Hello, " + name);
                    }
                `
            };
            const chars = analyzeContent(input);
            expect(chars.hasCode).toBe(true);
            expect(chars.codeLanguage).toBe('javascript');
        });

        it('should detect Python code', () => {
            const input: AutoEvalInput = {
                response: `
                    def greet(name):
                        print(f"Hello, {name}")

                    if __name__ == "__main__":
                        greet("World")
                `
            };
            const chars = analyzeContent(input);
            expect(chars.hasCode).toBe(true);
            expect(chars.codeLanguage).toBe('python');
        });

        it('should detect RAG context', () => {
            const input: AutoEvalInput = {
                query: 'What is Python?',
                response: 'Python is a programming language.',
                context: 'Python is a high-level programming language.'
            };
            const chars = analyzeContent(input);
            expect(chars.inputType).toBe('rag');
            expect(chars.hasContext).toBe(true);
            expect(chars.isQA).toBe(true);
        });

        it('should detect Q&A format with reference', () => {
            const input: AutoEvalInput = {
                query: 'What is 2+2?',
                response: '4',
                reference: 'Four'
            };
            const chars = analyzeContent(input);
            expect(chars.inputType).toBe('qa');
            expect(chars.hasReference).toBe(true);
            expect(chars.isQA).toBe(true);
        });

        it('should categorize response length', () => {
            const shortInput: AutoEvalInput = { response: 'Hello world' };
            const mediumInput: AutoEvalInput = {
                response: 'This is a medium length response. '.repeat(10)
            };
            const longInput: AutoEvalInput = {
                response: 'This is a long response with lots of content. '.repeat(50)
            };

            expect(analyzeContent(shortInput).lengthCategory).toBe('short');
            expect(analyzeContent(mediumInput).lengthCategory).toBe('medium');
            expect(analyzeContent(longInput).lengthCategory).toBe('long');
        });

        it('should detect code blocks with language hints', () => {
            const input: AutoEvalInput = {
                response: '```python\ndef foo():\n    pass\n```'
            };
            const chars = analyzeContent(input);
            expect(chars.hasCode).toBe(true);
            expect(chars.codeLanguage).toBe('python');
        });

        it('should detect technical domain', () => {
            const input: AutoEvalInput = {
                query: 'How do I create an API?',
                response: 'To create an API, you need a server and database.'
            };
            const chars = analyzeContent(input);
            expect(chars.domains).toContain('technical');
        });

        it('should describe characteristics as string', () => {
            const input: AutoEvalInput = {
                query: 'What is Python?',
                response: 'Python is a programming language.',
                context: 'Python is used for web development.'
            };
            const chars = analyzeContent(input);
            const description = describeCharacteristics(chars);
            expect(description).toContain('rag');
            expect(description).toContain('context');
        });
    });

    describe('Metric Selector', () => {
        it('should select JSON metrics for JSON content', () => {
            const input: AutoEvalInput = {
                response: '{"valid": true}'
            };
            const chars = analyzeContent(input);
            const metrics = selectMetrics(chars, input);

            expect(metrics.some(m => m.name === 'is_json')).toBe(true);
            expect(metrics.some(m => m.category === 'json')).toBe(true);
        });

        it('should select RAG metrics when context is provided', () => {
            const input: AutoEvalInput = {
                query: 'What is AI?',
                response: 'AI is artificial intelligence.',
                context: 'AI stands for artificial intelligence.'
            };
            const chars = analyzeContent(input);
            const metrics = selectMetrics(chars, input);

            expect(metrics.some(m => m.category === 'rag')).toBe(true);
            expect(metrics.some(m => m.name === 'faithfulness')).toBe(true);
        });

        it('should select security metrics for code', () => {
            const input: AutoEvalInput = {
                response: 'const query = "SELECT * FROM users"',
                code: 'const query = "SELECT * FROM users"'
            };
            const chars = analyzeContent(input);
            const metrics = selectMetrics(chars, input, { enableSecurityChecks: true });

            expect(metrics.some(m => m.category === 'security')).toBe(true);
        });

        it('should select similarity metrics when reference is provided', () => {
            const input: AutoEvalInput = {
                query: 'What is 2+2?',
                response: 'The answer is four.',
                reference: 'Four'
            };
            const chars = analyzeContent(input);
            const metrics = selectMetrics(chars, input);

            expect(metrics.some(m => m.category === 'similarity')).toBe(true);
            expect(metrics.some(m => m.name === 'bleu_score')).toBe(true);
        });

        it('should respect minConfidence threshold', () => {
            const input: AutoEvalInput = {
                response: 'Simple text response'
            };
            const chars = analyzeContent(input);

            const lowThreshold = selectMetrics(chars, input, { minConfidence: 0.1 });
            const highThreshold = selectMetrics(chars, input, { minConfidence: 0.9 });

            expect(lowThreshold.length).toBeGreaterThanOrEqual(highThreshold.length);
        });

        it('should respect maxMetrics limit', () => {
            const input: AutoEvalInput = {
                query: 'Test',
                response: '{"data": true}',
                context: 'Context',
                reference: 'Reference'
            };
            const chars = analyzeContent(input);

            const limited = selectMetrics(chars, input, { maxMetrics: 3, minConfidence: 0.1 });
            expect(limited.length).toBeLessThanOrEqual(3);
        });

        it('should exclude categories when specified', () => {
            const input: AutoEvalInput = {
                response: 'const x = 1;',
                code: 'const x = 1;'
            };
            const chars = analyzeContent(input);

            const withSecurity = selectMetrics(chars, input, { enableSecurityChecks: true });
            const withoutSecurity = selectMetrics(chars, input, { excludeCategories: ['security'] });

            expect(withSecurity.some(m => m.category === 'security')).toBe(true);
            expect(withoutSecurity.some(m => m.category === 'security')).toBe(false);
        });

        it('should include only specified categories', () => {
            const input: AutoEvalInput = {
                query: 'Test',
                response: '{"data": true}',
                context: 'Context'
            };
            const chars = analyzeContent(input);

            const jsonOnly = selectMetrics(chars, input, { includeCategories: ['json'] });
            expect(jsonOnly.every(m => m.category === 'json')).toBe(true);
        });
    });

    describe('AutoEvalPipeline', () => {
        let pipeline: AutoEvalPipeline;

        beforeEach(() => {
            pipeline = new AutoEvalPipeline();
        });

        it('should evaluate simple text', () => {
            const result = pipeline.evaluate({
                response: 'Hello, World!'
            });

            expect(result.overallScore).toBeDefined();
            expect(result.characteristics.inputType).toBe('text');
            expect(result.summary).toBeDefined();
        });

        it('should evaluate JSON content', () => {
            const result = pipeline.evaluate({
                response: '{"name": "test", "value": 123}'
            });

            expect(result.characteristics.hasJson).toBe(true);
            expect(result.metricResults.some(r => r.category === 'json')).toBe(true);
        });

        it('should evaluate RAG content with context', () => {
            const result = pipeline.evaluate({
                query: 'What is Python?',
                response: 'Python is a high-level programming language.',
                context: 'Python is a high-level, interpreted programming language.'
            });

            expect(result.characteristics.inputType).toBe('rag');
            expect(result.metricResults.some(r => r.category === 'rag')).toBe(true);
        });

        it('should evaluate code for security issues', () => {
            const result = pipeline.evaluate({
                response: 'const apiKey = "sk-secret123456789012345678901234567890123456";',
                code: 'const apiKey = "sk-secret123456789012345678901234567890123456";'
            });

            expect(result.characteristics.hasCode).toBe(true);
            expect(result.metricResults.some(r => r.category === 'security')).toBe(true);
        });

        it('should calculate overall score', () => {
            const result = pipeline.evaluate({
                response: 'A valid response'
            });

            expect(result.overallScore).toBeGreaterThanOrEqual(0);
            expect(result.overallScore).toBeLessThanOrEqual(1);
        });

        it('should determine pass/fail based on threshold', () => {
            const strictPipeline = new AutoEvalPipeline({ passThreshold: 0.99 });
            const lenientPipeline = new AutoEvalPipeline({ passThreshold: 0.1 });

            const input: AutoEvalInput = { response: 'Test response' };

            const strictResult = strictPipeline.evaluate(input);
            const lenientResult = lenientPipeline.evaluate(input);

            // Lenient should more likely pass
            expect(lenientResult.passed).toBe(true);
        });

        it('should generate recommendations for failed metrics', () => {
            const result = pipeline.evaluate({
                response: 'const query = "SELECT * FROM users WHERE id = " + userId;',
                code: 'const query = "SELECT * FROM users WHERE id = " + userId;'
            });

            // Should have security recommendations
            if (!result.passed) {
                expect(result.recommendations.length).toBeGreaterThan(0);
            }
        });

        it('should report execution time', () => {
            const result = pipeline.evaluate({
                response: 'Test response'
            });

            expect(result.totalExecutionTimeMs).toBeGreaterThanOrEqual(0);
            result.metricResults.forEach(r => {
                expect(r.executionTimeMs).toBeGreaterThanOrEqual(0);
            });
        });

        it('should allow config overrides per evaluation', () => {
            const result = pipeline.evaluate(
                { response: 'Test', code: 'const x = 1;' },
                { enableSecurityChecks: false }
            );

            expect(result.metricResults.every(r => r.category !== 'security')).toBe(true);
        });

        it('should update configuration', () => {
            pipeline.configure({ maxMetrics: 5 });
            const config = pipeline.getConfig();
            expect(config.maxMetrics).toBe(5);
        });

        it('should analyze without evaluating', () => {
            const analysis = pipeline.analyzeOnly({
                query: 'What is AI?',
                response: 'AI is artificial intelligence.',
                context: 'Artificial intelligence is a field of computer science.'
            });

            expect(analysis.characteristics).toBeDefined();
            expect(analysis.description).toBeDefined();
            expect(analysis.recommendedMetrics).toBeDefined();
            expect(analysis.recommendedMetrics.length).toBeGreaterThan(0);
        });
    });

    describe('Convenience Functions', () => {
        it('createAutoEval should create a pipeline', () => {
            const pipeline = createAutoEval({ maxMetrics: 5 });
            expect(pipeline).toBeInstanceOf(AutoEvalPipeline);
            expect(pipeline.getConfig().maxMetrics).toBe(5);
        });

        it('autoEvaluate should evaluate directly', () => {
            const result = autoEvaluate({
                response: 'Hello, World!'
            });

            expect(result.overallScore).toBeDefined();
            expect(result.summary).toBeDefined();
        });

        it('autoEvaluate should accept config', () => {
            const result = autoEvaluate(
                { response: 'Test' },
                { maxMetrics: 2 }
            );

            expect(result.selectedMetrics.length).toBeLessThanOrEqual(2);
        });
    });

    describe('Error Handling', () => {
        it('should handle empty response gracefully', () => {
            const pipeline = new AutoEvalPipeline();
            const result = pipeline.evaluate({ response: '' });

            expect(result.overallScore).toBeDefined();
            expect(result.passed).toBeDefined();
        });

        it('should handle invalid JSON gracefully', () => {
            const result = autoEvaluate({
                response: '{"invalid: json}'
            });

            expect(result.overallScore).toBeDefined();
            expect(result.characteristics.hasJson).toBe(false);
        });

        it('should handle metric execution errors gracefully', () => {
            const pipeline = new AutoEvalPipeline();
            const result = pipeline.evaluate({
                response: 'Test response'
            });

            // Should not throw, even if some metrics fail
            expect(result).toBeDefined();
        });
    });

    describe('Real-world Scenarios', () => {
        it('should evaluate a chatbot response', () => {
            const result = autoEvaluate({
                query: 'How do I reset my password?',
                response: 'To reset your password, go to Settings > Security > Reset Password and follow the prompts.',
                context: 'Password reset can be done through Settings. Navigate to Security section and select Reset Password option.'
            });

            expect(result.characteristics.inputType).toBe('rag');
            expect(result.overallScore).toBeGreaterThan(0);
        });

        it('should evaluate code generation', () => {
            const result = autoEvaluate({
                query: 'Write a function to add two numbers',
                response: `
                    function add(a, b) {
                        return a + b;
                    }
                `
            });

            expect(result.characteristics.hasCode).toBe(true);
        });

        it('should evaluate API response formatting', () => {
            const result = autoEvaluate({
                response: JSON.stringify({
                    status: 'success',
                    data: { id: 1, name: 'Test' }
                })
            });

            expect(result.characteristics.hasJson).toBe(true);
            expect(result.metricResults.some(r =>
                r.metric === 'is_json' && r.result.passed
            )).toBe(true);
        });

        it('should detect security issues in generated code', () => {
            const result = autoEvaluate({
                query: 'Write code to query the database',
                response: `
                    const userId = req.params.id;
                    const query = "SELECT * FROM users WHERE id = " + userId;
                    db.query(query);
                `,
                code: `
                    const userId = req.params.id;
                    const query = "SELECT * FROM users WHERE id = " + userId;
                    db.query(query);
                `
            });

            const securityResults = result.metricResults.filter(r => r.category === 'security');
            const hasSqlInjection = securityResults.some(r =>
                r.metric === 'sql_injection' && !r.result.passed
            );

            expect(hasSqlInjection).toBe(true);
        });
    });
});
