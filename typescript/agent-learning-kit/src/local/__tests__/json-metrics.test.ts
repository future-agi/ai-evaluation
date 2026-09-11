/**
 * Tests for JSON heuristic metrics
 */

import { containsJson, isJson, jsonSchema } from '../metrics/json';

describe('JSON Metrics', () => {
    describe('containsJson', () => {
        it('should detect JSON object in text', () => {
            const text = 'Here is the response: {"name": "John", "age": 30}';
            const result = containsJson(text);
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should detect JSON array in text', () => {
            const text = 'The items are: [1, 2, 3, 4, 5]';
            const result = containsJson(text);
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail when no JSON present', () => {
            const text = 'This is just plain text with no JSON.';
            const result = containsJson(text);
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should detect JSON in code block', () => {
            const text = '```json\n{"key": "value"}\n```';
            const result = containsJson(text);
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should detect nested JSON', () => {
            const text = 'Response: {"user": {"name": "John", "address": {"city": "NYC"}}}';
            const result = containsJson(text);
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });
    });

    describe('isJson', () => {
        it('should pass for valid JSON object', () => {
            const result = isJson('{"name": "John", "age": 30}');
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should pass for valid JSON array', () => {
            const result = isJson('[1, 2, 3, "four"]');
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should pass for JSON with whitespace', () => {
            const result = isJson('  {"key": "value"}  ');
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail for invalid JSON', () => {
            const result = isJson('{name: "John"}'); // Missing quotes around key
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should fail for plain text', () => {
            const result = isJson('This is not JSON');
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should handle empty string', () => {
            const result = isJson('');
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });

    describe('jsonSchema', () => {
        const simpleSchema = {
            type: 'object',
            properties: {
                name: { type: 'string' },
                age: { type: 'number' }
            },
            required: ['name']
        };

        it('should pass for valid JSON matching schema', () => {
            const result = jsonSchema('{"name": "John", "age": 30}', { schema: simpleSchema });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should pass with only required fields', () => {
            const result = jsonSchema('{"name": "John"}', { schema: simpleSchema });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail for missing required field', () => {
            const result = jsonSchema('{"age": 30}', { schema: simpleSchema });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should fail for wrong type', () => {
            const result = jsonSchema('{"name": 123}', { schema: simpleSchema });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should fail for invalid JSON', () => {
            const result = jsonSchema('not json', { schema: simpleSchema });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });

        it('should handle array schema', () => {
            const arraySchema = {
                type: 'array',
                items: { type: 'number' }
            };
            const result = jsonSchema('[1, 2, 3]', { schema: arraySchema });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should handle nested object schema', () => {
            const nestedSchema = {
                type: 'object',
                properties: {
                    user: {
                        type: 'object',
                        properties: {
                            name: { type: 'string' }
                        },
                        required: ['name']
                    }
                },
                required: ['user']
            };
            const result = jsonSchema('{"user": {"name": "John"}}', { schema: nestedSchema });
            expect(result.score).toBe(1.0);
            expect(result.passed).toBe(true);
        });

        it('should fail nested validation', () => {
            const nestedSchema = {
                type: 'object',
                properties: {
                    user: {
                        type: 'object',
                        properties: {
                            name: { type: 'string' }
                        },
                        required: ['name']
                    }
                },
                required: ['user']
            };
            const result = jsonSchema('{"user": {}}', { schema: nestedSchema });
            expect(result.score).toBe(0.0);
            expect(result.passed).toBe(false);
        });
    });
});
