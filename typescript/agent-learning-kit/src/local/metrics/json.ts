/**
 * JSON-based heuristic metrics for local evaluation.
 * Aligned with Python SDK's fi.evals.local.metrics.json module.
 */

import { MetricResult, MetricConfig } from './types';

/**
 * Check if the response contains valid JSON
 */
export function containsJson(response: string): MetricResult {
    // Try to find JSON in the response
    const jsonPatterns = [
        /```json\s*([\s\S]*?)\s*```/,  // Markdown code block
        /```\s*([\s\S]*?)\s*```/,       // Generic code block
        /(\{[\s\S]*\})/,                // Object
        /(\[[\s\S]*\])/                 // Array
    ];

    for (const pattern of jsonPatterns) {
        const match = response.match(pattern);
        if (match) {
            try {
                JSON.parse(match[1]);
                return {
                    score: 1.0,
                    passed: true,
                    reason: 'Response contains valid JSON'
                };
            } catch {
                // Continue trying other patterns
            }
        }
    }

    // Try parsing the entire response
    try {
        JSON.parse(response);
        return {
            score: 1.0,
            passed: true,
            reason: 'Response is valid JSON'
        };
    } catch {
        return {
            score: 0.0,
            passed: false,
            reason: 'Response does not contain valid JSON'
        };
    }
}

/**
 * Check if the response is valid JSON
 */
export function isJson(response: string): MetricResult {
    try {
        JSON.parse(response.trim());
        return {
            score: 1.0,
            passed: true,
            reason: 'Response is valid JSON'
        };
    } catch (e) {
        return {
            score: 0.0,
            passed: false,
            reason: `Response is not valid JSON: ${(e as Error).message}`
        };
    }
}

/**
 * Validate JSON against a schema
 */
export function jsonSchema(
    response: string,
    config: MetricConfig & { schema: object }
): MetricResult {
    // Parse the response
    let parsed: any;
    try {
        parsed = JSON.parse(response.trim());
    } catch (e) {
        return {
            score: 0.0,
            passed: false,
            reason: `Response is not valid JSON: ${(e as Error).message}`
        };
    }

    // Simple schema validation (supports basic JSON Schema properties)
    const schema = config.schema as any;
    const errors = validateAgainstSchema(parsed, schema, '');

    if (errors.length === 0) {
        return {
            score: 1.0,
            passed: true,
            reason: 'Response matches JSON schema'
        };
    } else {
        return {
            score: 0.0,
            passed: false,
            reason: `Schema validation errors: ${errors.join('; ')}`
        };
    }
}

/**
 * Simple JSON Schema validator
 */
function validateAgainstSchema(
    value: any,
    schema: any,
    path: string
): string[] {
    const errors: string[] = [];

    if (!schema) return errors;

    // Type validation
    if (schema.type) {
        const actualType = getJsonType(value);
        const expectedTypes = Array.isArray(schema.type) ? schema.type : [schema.type];

        // 'number' type should also accept 'integer' (JSON Schema convention)
        const typeMatches = expectedTypes.some((t: string) => {
            if (t === 'number' && actualType === 'integer') return true;
            return t === actualType;
        });

        if (!typeMatches) {
            errors.push(`${path || 'root'}: expected type ${expectedTypes.join('|')}, got ${actualType}`);
            return errors; // Return early if type doesn't match
        }
    }

    // Object validation
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        // Required properties
        if (schema.required && Array.isArray(schema.required)) {
            for (const prop of schema.required) {
                if (!(prop in value)) {
                    errors.push(`${path || 'root'}: missing required property "${prop}"`);
                }
            }
        }

        // Property validation
        if (schema.properties) {
            for (const [prop, propSchema] of Object.entries(schema.properties)) {
                if (prop in value) {
                    const propPath = path ? `${path}.${prop}` : prop;
                    errors.push(...validateAgainstSchema(value[prop], propSchema, propPath));
                }
            }
        }

        // Additional properties
        if (schema.additionalProperties === false) {
            const allowedProps = new Set(Object.keys(schema.properties || {}));
            for (const prop of Object.keys(value)) {
                if (!allowedProps.has(prop)) {
                    errors.push(`${path || 'root'}: unexpected property "${prop}"`);
                }
            }
        }
    }

    // Array validation
    if (Array.isArray(value)) {
        if (schema.minItems !== undefined && value.length < schema.minItems) {
            errors.push(`${path || 'root'}: array length ${value.length} is less than minItems ${schema.minItems}`);
        }
        if (schema.maxItems !== undefined && value.length > schema.maxItems) {
            errors.push(`${path || 'root'}: array length ${value.length} is greater than maxItems ${schema.maxItems}`);
        }
        if (schema.items) {
            value.forEach((item, index) => {
                const itemPath = `${path || 'root'}[${index}]`;
                errors.push(...validateAgainstSchema(item, schema.items, itemPath));
            });
        }
    }

    // String validation
    if (typeof value === 'string') {
        if (schema.minLength !== undefined && value.length < schema.minLength) {
            errors.push(`${path || 'root'}: string length ${value.length} is less than minLength ${schema.minLength}`);
        }
        if (schema.maxLength !== undefined && value.length > schema.maxLength) {
            errors.push(`${path || 'root'}: string length ${value.length} is greater than maxLength ${schema.maxLength}`);
        }
        if (schema.pattern) {
            const regex = new RegExp(schema.pattern);
            if (!regex.test(value)) {
                errors.push(`${path || 'root'}: string does not match pattern "${schema.pattern}"`);
            }
        }
        if (schema.enum && !schema.enum.includes(value)) {
            errors.push(`${path || 'root'}: value "${value}" is not in enum [${schema.enum.join(', ')}]`);
        }
    }

    // Number validation
    if (typeof value === 'number') {
        if (schema.minimum !== undefined && value < schema.minimum) {
            errors.push(`${path || 'root'}: value ${value} is less than minimum ${schema.minimum}`);
        }
        if (schema.maximum !== undefined && value > schema.maximum) {
            errors.push(`${path || 'root'}: value ${value} is greater than maximum ${schema.maximum}`);
        }
    }

    return errors;
}

/**
 * Get JSON Schema type of a value
 */
function getJsonType(value: any): string {
    if (value === null) return 'null';
    if (Array.isArray(value)) return 'array';
    if (typeof value === 'number') {
        return Number.isInteger(value) ? 'integer' : 'number';
    }
    return typeof value;
}
