/**
 * Tests for hallucination detection metrics
 */

import {
    hallucinationDetection,
    detectHallucination,
    noHallucination,
    HallucinationInput,
    hallucinationExtractSentences as extractSentences,
    isFactualClaim,
    calculateOverlap,
    hallucinationTokenize as tokenize,
    hallucinationGetNgrams as getNgrams
} from '../metrics/hallucination';

describe('Hallucination Detection', () => {
    describe('hallucinationDetection', () => {
        it('should detect no hallucination when response is fully supported', () => {
            const input: HallucinationInput = {
                query: 'What is the capital of France?',
                response: 'Paris is the capital of France.',
                context: 'Paris is the capital and largest city of France.'
            };
            const result = hallucinationDetection(input);
            expect(result.hallucinationRate).toBeLessThan(0.5);
            expect(result.supportedClaims).toBeGreaterThan(0);
        });

        it('should detect hallucination when response contains unsupported claims', () => {
            const input: HallucinationInput = {
                query: 'What is the capital of France?',
                response: 'Paris is the capital of France. The Eiffel Tower was built in 1889 and is 324 meters tall.',
                context: 'Paris is the capital of France.'
            };
            const result = hallucinationDetection(input);
            // Some claims should be unsupported (Eiffel Tower details)
            expect(result.unsupportedClaims).toBeGreaterThan(0);
        });

        it('should handle completely unsupported responses', () => {
            const input: HallucinationInput = {
                query: 'What is Python?',
                response: 'Python was invented by aliens in the year 3000.',
                context: 'Python is a high-level programming language created by Guido van Rossum.'
            };
            const result = hallucinationDetection(input);
            expect(result.hallucinationRate).toBeGreaterThan(0.5);
            expect(result.passed).toBe(false);
        });

        it('should handle array of context strings', () => {
            const input: HallucinationInput = {
                query: 'Tell me about Python',
                response: 'Python is a programming language. Guido created it.',
                context: [
                    'Python is a high-level programming language.',
                    'Guido van Rossum created Python.'
                ]
            };
            const result = hallucinationDetection(input);
            expect(result.supportedClaims).toBeGreaterThan(0);
            expect(result.hallucinationRate).toBeLessThan(0.5);
        });

        it('should include reference material when provided', () => {
            const input: HallucinationInput = {
                query: 'What year was Python created?',
                response: 'Python was first released in 1991.',
                context: 'Python is a programming language.',
                reference: 'Python was first released in 1991 by Guido van Rossum.'
            };
            const result = hallucinationDetection(input);
            expect(result.supportedClaims).toBeGreaterThan(0);
        });

        it('should handle empty response', () => {
            const input: HallucinationInput = {
                query: 'What is Python?',
                response: '',
                context: 'Python is a programming language.'
            };
            const result = hallucinationDetection(input);
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
            expect(result.hallucinationRate).toBe(0);
        });

        it('should handle empty context', () => {
            const input: HallucinationInput = {
                query: 'What is Python?',
                response: 'Python is a programming language.',
                context: ''
            };
            const result = hallucinationDetection(input);
            expect(result.passed).toBe(false);
            expect(result.hallucinationRate).toBe(1.0);
            expect(result.reason).toContain('No context');
        });

        it('should include claim analysis when enabled', () => {
            const input: HallucinationInput = {
                query: 'What is Python?',
                response: 'Python is a programming language. It was created by Guido.',
                context: 'Python is a high-level programming language created by Guido van Rossum.'
            };
            const result = hallucinationDetection(input, { includeClaimAnalysis: true });
            expect(result.claims).toBeDefined();
            expect(result.claims!.length).toBeGreaterThan(0);
            expect(result.claims![0]).toHaveProperty('text');
            expect(result.claims![0]).toHaveProperty('supported');
            expect(result.claims![0]).toHaveProperty('confidence');
        });

        it('should not include claim analysis when disabled', () => {
            const input: HallucinationInput = {
                query: 'What is Python?',
                response: 'Python is a programming language.',
                context: 'Python is a high-level programming language.'
            };
            const result = hallucinationDetection(input, { includeClaimAnalysis: false });
            expect(result.claims).toBeUndefined();
        });

        it('should respect custom threshold', () => {
            const input: HallucinationInput = {
                query: 'What is Python?',
                response: 'Python is a programming language used for web development.',
                context: 'Python is a programming language.'
            };
            // With high threshold, might fail
            const strictResult = hallucinationDetection(input, { threshold: 0.95 });
            // With low threshold, should pass
            const lenientResult = hallucinationDetection(input, { threshold: 0.3 });

            expect(lenientResult.passed).toBe(true);
        });

        it('should skip opinion statements', () => {
            const input: HallucinationInput = {
                query: 'What do you think about Python?',
                response: 'I think Python is a great language. In my opinion, it is easy to learn.',
                context: 'Python is a programming language.'
            };
            const result = hallucinationDetection(input);
            // Opinion statements should not count as factual claims
            expect(result.reason).toContain('claim');
        });

        it('should find evidence for supported claims', () => {
            const input: HallucinationInput = {
                query: 'What is Paris?',
                response: 'Paris is the capital of France.',
                context: 'Paris is the capital and most populous city of France.'
            };
            const result = hallucinationDetection(input, { includeClaimAnalysis: true });
            const supportedClaim = result.claims?.find(c => c.supported);
            if (supportedClaim) {
                expect(supportedClaim.evidence).toBeDefined();
            }
        });

        it('aliases should work correctly', () => {
            expect(detectHallucination).toBe(hallucinationDetection);
            expect(noHallucination).toBe(hallucinationDetection);
        });
    });

    describe('Helper Functions', () => {
        describe('tokenize', () => {
            it('should tokenize text correctly', () => {
                const tokens = tokenize('Hello, World! This is a test.');
                expect(tokens).toContain('hello');
                expect(tokens).toContain('world');
                expect(tokens).toContain('test');
            });

            it('should handle empty string', () => {
                const tokens = tokenize('');
                expect(tokens).toEqual([]);
            });

            it('should lowercase tokens', () => {
                const tokens = tokenize('UPPERCASE lowercase MixedCase');
                expect(tokens).toEqual(['uppercase', 'lowercase', 'mixedcase']);
            });
        });

        describe('getNgrams', () => {
            it('should get bigrams', () => {
                const tokens = ['hello', 'world', 'test'];
                const bigrams = getNgrams(tokens, 2);
                expect(bigrams.has('hello world')).toBe(true);
                expect(bigrams.has('world test')).toBe(true);
                expect(bigrams.size).toBe(2);
            });

            it('should get trigrams', () => {
                const tokens = ['a', 'b', 'c', 'd'];
                const trigrams = getNgrams(tokens, 3);
                expect(trigrams.has('a b c')).toBe(true);
                expect(trigrams.has('b c d')).toBe(true);
                expect(trigrams.size).toBe(2);
            });

            it('should handle tokens shorter than n', () => {
                const tokens = ['hello'];
                const bigrams = getNgrams(tokens, 2);
                expect(bigrams.size).toBe(0);
            });
        });

        describe('extractSentences', () => {
            it('should extract sentences', () => {
                const text = 'First sentence. Second sentence! Third sentence?';
                const sentences = extractSentences(text);
                expect(sentences.length).toBe(3);
                expect(sentences[0]).toBe('First sentence');
                expect(sentences[1]).toBe('Second sentence');
            });

            it('should filter short sentences', () => {
                const text = 'Hi. This is a longer sentence.';
                const sentences = extractSentences(text);
                // "Hi" is too short (< 10 chars)
                expect(sentences.length).toBe(1);
            });
        });

        describe('isFactualClaim', () => {
            it('should identify factual claims', () => {
                expect(isFactualClaim('Paris is the capital of France')).toBe(true);
                expect(isFactualClaim('The Earth contains mostly water')).toBe(true);
                expect(isFactualClaim('Studies show that exercise is beneficial')).toBe(true);
            });

            it('should reject opinions', () => {
                expect(isFactualClaim('I think this is great')).toBe(false);
                expect(isFactualClaim('In my opinion, it could be better')).toBe(false);
                expect(isFactualClaim('It might be true')).toBe(false);
            });

            it('should reject questions', () => {
                expect(isFactualClaim('Is this true?')).toBe(false);
            });
        });

        describe('calculateOverlap', () => {
            it('should calculate word overlap', () => {
                const overlap = calculateOverlap(
                    'Paris is the capital',
                    'Paris is the capital of France'
                );
                expect(overlap).toBe(1.0); // All words in text1 are in text2
            });

            it('should handle partial overlap', () => {
                const overlap = calculateOverlap(
                    'Paris is beautiful',
                    'Paris is the capital'
                );
                // 'paris' and 'is' overlap, 'beautiful' doesn't
                expect(overlap).toBeCloseTo(0.67, 1);
            });

            it('should handle no overlap', () => {
                const overlap = calculateOverlap(
                    'Hello world',
                    'Goodbye universe'
                );
                expect(overlap).toBe(0);
            });

            it('should handle empty strings', () => {
                expect(calculateOverlap('', 'test')).toBe(0);
                expect(calculateOverlap('test', '')).toBe(0);
            });
        });
    });

    describe('Real-world Scenarios', () => {
        it('should handle technical documentation context', () => {
            const input: HallucinationInput = {
                query: 'How do I create a function in Python?',
                response: 'To create a function in Python, use the def keyword followed by the function name and parentheses. The function body is indented.',
                context: 'In Python, functions are defined using the def keyword. The function definition starts with def, followed by the function name and parentheses. The body of the function is indented.'
            };
            const result = hallucinationDetection(input);
            expect(result.passed).toBe(true);
        });

        it('should detect fabricated statistics', () => {
            const input: HallucinationInput = {
                query: 'What is Python used for?',
                response: 'Python is used for web development. According to a 2024 survey, 95% of developers prefer Python for AI.',
                context: 'Python is a popular programming language commonly used for web development, data science, and machine learning.'
            };
            const result = hallucinationDetection(input);
            // The survey statistic is fabricated
            expect(result.unsupportedClaims).toBeGreaterThan(0);
        });

        it('should handle multi-paragraph responses', () => {
            const input: HallucinationInput = {
                query: 'Tell me about machine learning',
                response: `Machine learning is a subset of artificial intelligence. It enables computers to learn from data.

Deep learning is a type of machine learning using neural networks. It has revolutionized image recognition.`,
                context: `Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.

Deep learning is a subfield of machine learning that uses neural networks with many layers.`
            };
            const result = hallucinationDetection(input);
            expect(result.supportedClaims).toBeGreaterThan(0);
        });
    });
});
