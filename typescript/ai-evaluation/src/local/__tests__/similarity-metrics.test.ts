/**
 * Tests for similarity heuristic metrics
 */

import {
    bleuScore,
    rougeScore,
    recallScore,
    levenshteinSimilarity,
    numericSimilarity,
    semanticListContains
} from '../metrics/similarity';

describe('Similarity Metrics', () => {
    describe('bleuScore', () => {
        it('should return 1.0 for identical text', () => {
            const result = bleuScore('the cat sat on the mat', {
                reference: 'the cat sat on the mat'
            });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should return high score for similar text', () => {
            const result = bleuScore('the cat sat on the mat', {
                reference: 'the cat is on the mat'
            });
            expect(result.score).toBeGreaterThanOrEqual(0.5);
            expect(result.passed).toBe(true);
        });

        it('should return low score for different text', () => {
            const result = bleuScore('hello world', {
                reference: 'the cat sat on the mat'
            });
            expect(result.score).toBeLessThan(0.5);
        });

        it('should return 0 for empty response', () => {
            const result = bleuScore('', { reference: 'the cat sat on the mat' });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should apply brevity penalty for short responses', () => {
            const result = bleuScore('cat', { reference: 'the cat sat on the mat' });
            expect(result.score).toBeLessThan(0.5);
        });
    });

    describe('rougeScore', () => {
        it('should return 1.0 for identical text (ROUGE-L)', () => {
            const result = rougeScore('the quick brown fox', {
                reference: 'the quick brown fox',
                variant: 'rouge-l'
            });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should calculate ROUGE-1', () => {
            const result = rougeScore('the quick brown fox', {
                reference: 'the fast brown fox',
                variant: 'rouge-1'
            });
            expect(result.score).toBeGreaterThan(0.5);
        });

        it('should calculate ROUGE-2', () => {
            const result = rougeScore('the quick brown fox', {
                reference: 'the quick brown dog',
                variant: 'rouge-2'
            });
            // "the quick" and "quick brown" match, "brown fox" vs "brown dog" differ
            expect(result.score).toBeGreaterThan(0);
        });

        it('should return 0 for completely different text', () => {
            const result = rougeScore('hello world', {
                reference: 'abc xyz',
                variant: 'rouge-l'
            });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should handle empty input', () => {
            const result = rougeScore('', { reference: 'some text' });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });

    describe('recallScore', () => {
        it('should return 1.0 when all reference tokens present', () => {
            const result = recallScore('the quick brown fox jumps', {
                reference: 'the quick brown'
            });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should return partial score for partial recall', () => {
            const result = recallScore('the quick dog', {
                reference: 'the quick brown fox'
            });
            expect(result.score).toBe(0.5); // 2/4 tokens
        });

        it('should return 0 for no overlap', () => {
            const result = recallScore('hello world', {
                reference: 'abc xyz def'
            });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should handle empty reference', () => {
            const result = recallScore('some text', { reference: '' });
            expect(result.score).toBe(1.0); // Trivially satisfied
            expect(result.passed).toBe(true);
        });
    });

    describe('levenshteinSimilarity', () => {
        it('should return 1.0 for identical strings', () => {
            const result = levenshteinSimilarity('hello', { reference: 'hello' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should return high score for similar strings', () => {
            const result = levenshteinSimilarity('hello', { reference: 'hallo' });
            expect(result.score).toBe(0.8); // 1 edit out of 5 chars = 80% similarity
            expect(result.passed).toBe(true);
        });

        it('should return low score for different strings', () => {
            const result = levenshteinSimilarity('abc', { reference: 'xyz' });
            expect(result.score).toBe(0.0); // 3 edits out of 3 chars
            expect(result.passed).toBe(false);
        });

        it('should handle empty strings', () => {
            const result = levenshteinSimilarity('', { reference: '' });
            expect(result.score).toBe(1.0);
        });

        it('should handle one empty string', () => {
            const result = levenshteinSimilarity('hello', { reference: '' });
            expect(result.score).toBe(0.0);
        });
    });

    describe('numericSimilarity', () => {
        it('should return 1.0 for exact match', () => {
            const result = numericSimilarity('42', { reference: 42 });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should pass within tolerance', () => {
            const result = numericSimilarity('42.5', { reference: 42, tolerance: 1 });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail outside tolerance', () => {
            const result = numericSimilarity('45', { reference: 42, tolerance: 1 });
            expect(result.passed).toBe(false);
        });

        it('should handle floating point', () => {
            const result = numericSimilarity('3.14159', { reference: 3.14159 });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail for non-numeric response', () => {
            const result = numericSimilarity('not a number', { reference: 42 });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should handle negative numbers', () => {
            const result = numericSimilarity('-10', { reference: -10 });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });
    });

    describe('semanticListContains', () => {
        it('should pass when all items found', () => {
            const result = semanticListContains('I have an apple, banana, and cherry', {
                items: ['apple', 'banana', 'cherry']
            });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should return partial score for partial matches', () => {
            const result = semanticListContains('I have an apple and banana', {
                items: ['apple', 'banana', 'cherry', 'date']
            });
            expect(result.score).toBe(0.5); // 2/4 items
        });

        it('should fail below threshold', () => {
            const result = semanticListContains('just an apple', {
                items: ['apple', 'banana', 'cherry', 'date'],
                threshold: 0.5
            });
            expect(result.passed).toBe(false);
        });

        it('should pass with custom threshold', () => {
            const result = semanticListContains('apple and banana', {
                items: ['apple', 'banana', 'cherry', 'date'],
                threshold: 0.4
            });
            expect(result.passed).toBe(true);
        });

        it('should be case insensitive', () => {
            const result = semanticListContains('APPLE BANANA CHERRY', {
                items: ['apple', 'banana', 'cherry']
            });
            expect(result.score).toBe(1.0);
        });

        it('should return 0 for no matches', () => {
            const result = semanticListContains('xyz abc', {
                items: ['apple', 'banana', 'cherry']
            });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });
});
