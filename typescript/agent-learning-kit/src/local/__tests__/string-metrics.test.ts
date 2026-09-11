/**
 * Tests for string heuristic metrics
 */

import {
    regex,
    contains,
    containsAll,
    containsAny,
    containsNone,
    oneLine,
    equals,
    startsWith,
    endsWith,
    lengthLessThan,
    lengthGreaterThan,
    lengthBetween
} from '../metrics/string';

describe('String Metrics', () => {
    describe('regex', () => {
        it('should match valid regex pattern', () => {
            const result = regex('Hello World 123', { pattern: '\\d+' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail on non-matching pattern', () => {
            const result = regex('Hello World', { pattern: '\\d+' });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should handle case insensitive flag', () => {
            const result = regex('HELLO', { pattern: 'hello', flags: 'i' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should require pattern config', () => {
            expect(() => regex('test', {} as any)).toThrow('Regex pattern is required');
        });
    });

    describe('contains', () => {
        it('should find keyword in text', () => {
            const result = contains('Hello world', { keyword: 'world' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should be case insensitive by default', () => {
            const result = contains('Hello WORLD', { keyword: 'world' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should respect case sensitive option', () => {
            const result = contains('Hello WORLD', { keyword: 'world', caseSensitive: true });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should fail when keyword not found', () => {
            const result = contains('Hello world', { keyword: 'foo' });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });

    describe('containsAll', () => {
        it('should pass when all keywords found', () => {
            const result = containsAll('Hello world foo bar', { keywords: ['hello', 'world', 'foo'] });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail when some keywords missing', () => {
            const result = containsAll('Hello world', { keywords: ['hello', 'world', 'foo'] });
            expect(result.score).toBeLessThan(1.0);
            expect(result.passed).toBe(false);
        });

        it('should return partial score based on matches', () => {
            const result = containsAll('Hello world', { keywords: ['hello', 'world', 'foo', 'bar'] });
            expect(result.score).toBe(0.5); // 2/4 keywords
        });
    });

    describe('containsAny', () => {
        it('should pass when any keyword found', () => {
            const result = containsAny('Hello world', { keywords: ['foo', 'bar', 'world'] });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail when no keywords found', () => {
            const result = containsAny('Hello world', { keywords: ['foo', 'bar', 'baz'] });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });

    describe('containsNone', () => {
        it('should pass when no keywords found', () => {
            const result = containsNone('Hello world', { keywords: ['foo', 'bar', 'baz'] });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail when any keyword found', () => {
            const result = containsNone('Hello world', { keywords: ['foo', 'world', 'baz'] });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });

    describe('oneLine', () => {
        it('should pass for single line text', () => {
            const result = oneLine('Hello world');
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail for multi-line text', () => {
            const result = oneLine('Hello\nworld');
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should handle empty string', () => {
            const result = oneLine('');
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });
    });

    describe('equals', () => {
        it('should pass for exact match', () => {
            const result = equals('Hello world', { expected: 'Hello world' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should be case insensitive by default', () => {
            const result = equals('HELLO WORLD', { expected: 'hello world' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should trim whitespace by default', () => {
            const result = equals('  Hello world  ', { expected: 'Hello world' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail for different text', () => {
            const result = equals('Hello', { expected: 'World' });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });

    describe('startsWith', () => {
        it('should pass when text starts with prefix', () => {
            const result = startsWith('Hello world', { prefix: 'Hello' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail when text does not start with prefix', () => {
            const result = startsWith('Hello world', { prefix: 'World' });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should be case insensitive by default', () => {
            const result = startsWith('HELLO world', { prefix: 'hello' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });
    });

    describe('endsWith', () => {
        it('should pass when text ends with suffix', () => {
            const result = endsWith('Hello world', { suffix: 'world' });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail when text does not end with suffix', () => {
            const result = endsWith('Hello world', { suffix: 'Hello' });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });

    describe('lengthLessThan', () => {
        it('should pass when length is less than max', () => {
            const result = lengthLessThan('Hello', { maxLength: 10 });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail when length is equal to max', () => {
            const result = lengthLessThan('Hello', { maxLength: 5 });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should fail when length is greater than max', () => {
            const result = lengthLessThan('Hello world', { maxLength: 5 });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });

    describe('lengthGreaterThan', () => {
        it('should pass when length is greater than min', () => {
            const result = lengthGreaterThan('Hello world', { minLength: 5 });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail when length is equal to min', () => {
            const result = lengthGreaterThan('Hello', { minLength: 5 });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should fail when length is less than min', () => {
            const result = lengthGreaterThan('Hi', { minLength: 5 });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });

    describe('lengthBetween', () => {
        it('should pass when length is within range', () => {
            const result = lengthBetween('Hello', { minLength: 3, maxLength: 10 });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should pass at minimum boundary', () => {
            const result = lengthBetween('Hi!', { minLength: 3, maxLength: 10 });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should pass at maximum boundary', () => {
            const result = lengthBetween('HelloWorld', { minLength: 3, maxLength: 10 });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail when too short', () => {
            const result = lengthBetween('Hi', { minLength: 3, maxLength: 10 });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should fail when too long', () => {
            const result = lengthBetween('Hello World!', { minLength: 3, maxLength: 10 });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });
});
