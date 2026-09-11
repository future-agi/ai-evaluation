import {
    RequiredKeys,
    EvalTags,
    Comparator,
    EvalResult,
    BatchRunResult,
    EvalResultMetric,
    EvalTemplate,
    ConfigParam,
    ConfigPossibleValues,
    DatapointFieldAnnotation,
} from '../types';

describe('Types', () => {
    describe('RequiredKeys Enum', () => {
        it('should have correct string values for basic keys', () => {
            expect(RequiredKeys.text).toBe('text');
            expect(RequiredKeys.response).toBe('response');
            expect(RequiredKeys.query).toBe('query');
            expect(RequiredKeys.context).toBe('context');
        });

        it('should have correct string values for expected keys', () => {
            expect(RequiredKeys.expected_response).toBe('expected_response');
            expect(RequiredKeys.expected_text).toBe('expected_text');
        });

        it('should have correct string values for I/O keys', () => {
            expect(RequiredKeys.input).toBe('input');
            expect(RequiredKeys.output).toBe('output');
            expect(RequiredKeys.prompt).toBe('prompt');
        });

        it('should have correct string values for image keys', () => {
            expect(RequiredKeys.image_url).toBe('image_url');
            expect(RequiredKeys.input_image_url).toBe('input_image_url');
            expect(RequiredKeys.output_image_url).toBe('output_image_url');
        });

        it('should have correct string values for JSON keys', () => {
            expect(RequiredKeys.actual_json).toBe('actual_json');
            expect(RequiredKeys.expected_json).toBe('expected_json');
        });

        it('should have correct string values for other keys', () => {
            expect(RequiredKeys.document).toBe('document');
            expect(RequiredKeys.messages).toBe('messages');
        });
    });

    describe('EvalTags Enum', () => {
        it('should have all expected evaluation tags', () => {
            expect(EvalTags.CONVERSATION).toBe('CONVERSATION');
            expect(EvalTags.HALLUCINATION).toBe('HALLUCINATION');
            expect(EvalTags.RAG).toBe('RAG');
            expect(EvalTags.FUTURE_EVALS).toBe('FUTURE_EVALS');
            expect(EvalTags.LLMS).toBe('LLMS');
            expect(EvalTags.CUSTOM).toBe('CUSTOM');
            expect(EvalTags.FUNCTION).toBe('FUNCTION');
            expect(EvalTags.IMAGE).toBe('IMAGE');
            expect(EvalTags.SAFETY).toBe('SAFETY');
            expect(EvalTags.TEXT).toBe('TEXT');
        });

        it('should have exactly 10 tags', () => {
            const tagValues = Object.values(EvalTags);
            expect(tagValues.length).toBe(10);
        });
    });

    describe('Comparator Enum', () => {
        it('should have all comparator methods', () => {
            expect(Comparator.COSINE).toBe('CosineSimilarity');
            expect(Comparator.LEVENSHTEIN).toBe('NormalisedLevenshteinSimilarity');
            expect(Comparator.JARO_WINKLER).toBe('JaroWincklerSimilarity');
            expect(Comparator.JACCARD).toBe('JaccardSimilarity');
            expect(Comparator.SORENSEN_DICE).toBe('SorensenDiceSimilarity');
        });

        it('should have exactly 5 comparators', () => {
            const comparatorValues = Object.values(Comparator);
            expect(comparatorValues.length).toBe(5);
        });
    });

    describe('EvalResult Interface', () => {
        it('should create a valid eval result with all fields', () => {
            const result: EvalResult = {
                data: { score: 0.95 },
                failure: false,
                reason: 'Test passed successfully',
                runtime: 1500,
                metadata: { usage: { tokens: 100 } },
                metrics: [{ id: 'metric1', value: 0.95 }]
            };

            expect(result.data).toEqual({ score: 0.95 });
            expect(result.failure).toBe(false);
            expect(result.reason).toBe('Test passed successfully');
            expect(result.runtime).toBe(1500);
        });

        it('should allow optional fields', () => {
            const result: EvalResult = {
                reason: 'Minimal result',
                runtime: 100,
                metrics: []
            };

            expect(result.data).toBeUndefined();
            expect(result.failure).toBeUndefined();
            expect(result.metadata).toBeUndefined();
        });

        it('should allow array data', () => {
            const result: EvalResult = {
                data: ['item1', 'item2'],
                reason: 'Array data test',
                runtime: 50,
                metrics: []
            };

            expect(Array.isArray(result.data)).toBe(true);
        });
    });

    describe('BatchRunResult Interface', () => {
        it('should create a valid batch run result', () => {
            const batchResult: BatchRunResult = {
                eval_results: [
                    { reason: 'Result 1', runtime: 100, metrics: [] },
                    { reason: 'Result 2', runtime: 200, metrics: [] }
                ]
            };

            expect(batchResult.eval_results.length).toBe(2);
        });

        it('should allow null values in eval_results', () => {
            const batchResult: BatchRunResult = {
                eval_results: [
                    { reason: 'Result 1', runtime: 100, metrics: [] },
                    null,
                    { reason: 'Result 3', runtime: 300, metrics: [] }
                ]
            };

            expect(batchResult.eval_results.length).toBe(3);
            expect(batchResult.eval_results[1]).toBeNull();
        });

        it('should allow empty eval_results', () => {
            const batchResult: BatchRunResult = {
                eval_results: []
            };

            expect(batchResult.eval_results.length).toBe(0);
        });
    });

    describe('EvalResultMetric Interface', () => {
        it('should create a metric with string id', () => {
            const metric: EvalResultMetric = {
                id: 'accuracy',
                value: 0.95
            };

            expect(metric.id).toBe('accuracy');
            expect(metric.value).toBe(0.95);
        });

        it('should create a metric with number id', () => {
            const metric: EvalResultMetric = {
                id: 123,
                value: 'high'
            };

            expect(metric.id).toBe(123);
            expect(metric.value).toBe('high');
        });

        it('should allow array value', () => {
            const metric: EvalResultMetric = {
                id: 'multi-value',
                value: [1, 2, 3, 4, 5]
            };

            expect(Array.isArray(metric.value)).toBe(true);
        });
    });

    describe('EvalTemplate Interface', () => {
        it('should create a valid eval template with required fields', () => {
            const template: EvalTemplate = {
                eval_id: '47',
                eval_name: 'groundedness'
            };

            expect(template.eval_id).toBe('47');
            expect(template.eval_name).toBe('groundedness');
        });

        it('should create a template with all fields', () => {
            const template: EvalTemplate = {
                eval_id: '47',
                eval_name: 'groundedness',
                description: 'Check if response is grounded in context',
                eval_tags: ['RAG', 'HALLUCINATION'],
                required_keys: ['context', 'output'],
                output: 'boolean',
                eval_type_id: 'rag-eval',
                config_schema: { threshold: { type: 'number', default: 0.7 } },
                criteria: 'Response must be supported by context',
                choices: ['GROUNDED', 'NOT_GROUNDED'],
                multi_choice: false
            };

            expect(template.description).toBe('Check if response is grounded in context');
            expect(template.eval_tags).toContain('RAG');
            expect(template.required_keys).toContain('context');
            expect(template.multi_choice).toBe(false);
        });
    });

    describe('ConfigParam Interface', () => {
        it('should create a config param with type only', () => {
            const param: ConfigParam = {
                type: 'string'
            };

            expect(param.type).toBe('string');
            expect(param.default).toBeUndefined();
        });

        it('should create a config param with default value', () => {
            const param: ConfigParam = {
                type: 'number',
                default: 0.8
            };

            expect(param.type).toBe('number');
            expect(param.default).toBe(0.8);
        });
    });

    describe('ConfigPossibleValues Interface', () => {
        it('should create config with length constraints', () => {
            const config: ConfigPossibleValues = {
                min_length: 10,
                max_length: 1000
            };

            expect(config.min_length).toBe(10);
            expect(config.max_length).toBe(1000);
        });

        it('should create config with keywords', () => {
            const config: ConfigPossibleValues = {
                keywords: ['test', 'example', 'sample'],
                keyword: 'main',
                case_sensitive: true
            };

            expect(config.keywords).toHaveLength(3);
            expect(config.keyword).toBe('main');
            expect(config.case_sensitive).toBe(true);
        });

        it('should create config with URL settings', () => {
            const config: ConfigPossibleValues = {
                url: 'https://api.example.com',
                headers: { 'Authorization': 'Bearer token' },
                payload: { key: 'value' }
            };

            expect(config.url).toBe('https://api.example.com');
            expect(config.headers).toHaveProperty('Authorization');
        });

        it('should create config with grading settings', () => {
            const config: ConfigPossibleValues = {
                grading_criteria: 'Evaluate accuracy and relevance',
                choices: ['A', 'B', 'C', 'D', 'F'],
                multi_choice: true
            };

            expect(config.grading_criteria).toContain('accuracy');
            expect(config.choices).toHaveLength(5);
            expect(config.multi_choice).toBe(true);
        });

        it('should create config with model settings', () => {
            const config: ConfigPossibleValues = {
                model: 'gpt-4',
                system_prompt: 'You are a helpful assistant.',
                eval_prompt: 'Evaluate the following response:'
            };

            expect(config.model).toBe('gpt-4');
            expect(config.system_prompt).toContain('helpful');
        });

        it('should create config with pattern settings', () => {
            const config: ConfigPossibleValues = {
                pattern: '^[a-z]+$',
                validations: ['required', 'format']
            };

            expect(config.pattern).toBe('^[a-z]+$');
            expect(config.validations).toContain('required');
        });
    });

    describe('DatapointFieldAnnotation Interface', () => {
        it('should create a valid annotation', () => {
            const annotation: DatapointFieldAnnotation = {
                field_name: 'response',
                text: 'This section is important',
                annotation_type: 'highlight',
                annotation_note: 'Key finding'
            };

            expect(annotation.field_name).toBe('response');
            expect(annotation.text).toBe('This section is important');
            expect(annotation.annotation_type).toBe('highlight');
            expect(annotation.annotation_note).toBe('Key finding');
        });
    });
});

describe('Type Safety', () => {
    it('should enforce BatchRunResult structure', () => {
        // This test verifies that TypeScript enforces the interface
        const validResult: BatchRunResult = {
            eval_results: [
                {
                    reason: 'Test',
                    runtime: 100,
                    metrics: [{ id: 'test', value: 1 }]
                }
            ]
        };

        expect(validResult).toBeDefined();
    });

    it('should allow EvalResult with different metric value types', () => {
        const stringMetric: EvalResultMetric = { id: 1, value: 'passed' };
        const numberMetric: EvalResultMetric = { id: 2, value: 0.95 };
        const arrayMetric: EvalResultMetric = { id: 3, value: [1, 2, 3] };

        expect(stringMetric.value).toBe('passed');
        expect(numberMetric.value).toBe(0.95);
        expect(arrayMetric.value).toEqual([1, 2, 3]);
    });

    it('should allow EvalResult metadata with different types', () => {
        const stringMetadata: EvalResult = {
            reason: 'test',
            runtime: 100,
            metrics: [],
            metadata: 'simple string metadata'
        };

        const arrayMetadata: EvalResult = {
            reason: 'test',
            runtime: 100,
            metrics: [],
            metadata: ['item1', 'item2']
        };

        const objectMetadata: EvalResult = {
            reason: 'test',
            runtime: 100,
            metrics: [],
            metadata: { usage: { tokens: 100 }, cost: { usd: 0.01 } }
        };

        expect(stringMetadata.metadata).toBe('simple string metadata');
        expect(arrayMetadata.metadata).toEqual(['item1', 'item2']);
        expect(objectMetadata.metadata).toHaveProperty('usage');
    });
});
