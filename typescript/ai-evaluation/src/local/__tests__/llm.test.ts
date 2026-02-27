/**
 * Tests for LLM providers and factory
 */

import axios from 'axios';
import {
    OllamaLLM,
    OpenAILLM,
    AnthropicLLM,
    LLMFactory,
    LocalLLMFactory,
    BaseLLM
} from '../llm';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('OllamaLLM', () => {
    let llm: OllamaLLM;

    beforeEach(() => {
        llm = new OllamaLLM();
        jest.clearAllMocks();
    });

    describe('constructor', () => {
        it('should use default configuration', () => {
            const defaultLlm = new OllamaLLM();
            expect(defaultLlm).toBeDefined();
            expect(defaultLlm.provider).toBe('ollama');
            expect(defaultLlm.model).toBe('llama3.2');
        });

        it('should accept custom configuration', () => {
            const customLlm = new OllamaLLM({
                model: 'mistral',
                baseUrl: 'http://custom:11434',
                temperature: 0.5,
                maxTokens: 2048,
                timeout: 60
            });
            expect(customLlm).toBeDefined();
            expect(customLlm.model).toBe('mistral');
        });
    });

    describe('isAvailable', () => {
        it('should return true when Ollama is reachable', async () => {
            mockedAxios.get.mockResolvedValueOnce({ status: 200, data: { models: [] } });

            const result = await llm.isAvailable();
            expect(result).toBe(true);
            expect(mockedAxios.get).toHaveBeenCalledWith(
                'http://localhost:11434/api/tags',
                { timeout: 5000 }
            );
        });

        it('should return false when Ollama is not reachable', async () => {
            mockedAxios.get.mockRejectedValueOnce(new Error('Connection refused'));

            const result = await llm.isAvailable();
            expect(result).toBe(false);
        });

        it('should cache availability result', async () => {
            mockedAxios.get.mockResolvedValueOnce({ status: 200, data: { models: [] } });

            await llm.isAvailable();
            await llm.isAvailable();
            await llm.isAvailable();

            expect(mockedAxios.get).toHaveBeenCalledTimes(1);
        });
    });

    describe('listModels', () => {
        it('should return list of models', async () => {
            mockedAxios.get.mockResolvedValueOnce({
                status: 200,
                data: {
                    models: [
                        { name: 'llama3.2' },
                        { name: 'mistral' },
                        { name: 'codellama' }
                    ]
                }
            });

            const models = await llm.listModels();
            expect(models).toEqual(['llama3.2', 'mistral', 'codellama']);
        });

        it('should return empty array on error', async () => {
            mockedAxios.get.mockRejectedValueOnce(new Error('Connection refused'));

            const models = await llm.listModels();
            expect(models).toEqual([]);
        });
    });

    describe('generate', () => {
        it('should generate text completion', async () => {
            mockedAxios.post.mockResolvedValueOnce({
                data: { response: 'Hello, how can I help you?' }
            });

            const result = await llm.generate('Say hello');
            expect(result).toBe('Hello, how can I help you?');
            expect(mockedAxios.post).toHaveBeenCalledWith(
                'http://localhost:11434/api/generate',
                expect.objectContaining({
                    model: 'llama3.2',
                    prompt: 'Say hello',
                    stream: false
                }),
                expect.any(Object)
            );
        });

        it('should include system prompt when provided', async () => {
            mockedAxios.post.mockResolvedValueOnce({
                data: { response: 'Custom response' }
            });

            await llm.generate('Hello', { system: 'You are a helpful assistant' });
            expect(mockedAxios.post).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({
                    system: 'You are a helpful assistant'
                }),
                expect.any(Object)
            );
        });

        it('should throw error on failure', async () => {
            mockedAxios.post.mockRejectedValueOnce({ message: 'Server error' });

            await expect(llm.generate('test')).rejects.toThrow('Ollama generate failed');
        });
    });

    describe('chat', () => {
        it('should handle chat messages', async () => {
            mockedAxios.post.mockResolvedValueOnce({
                data: { message: { content: 'I can help with that!' } }
            });

            const result = await llm.chat([
                { role: 'user', content: 'Can you help me?' }
            ]);

            expect(result).toBe('I can help with that!');
            expect(mockedAxios.post).toHaveBeenCalledWith(
                'http://localhost:11434/api/chat',
                expect.objectContaining({
                    model: 'llama3.2',
                    messages: [{ role: 'user', content: 'Can you help me?' }],
                    stream: false
                }),
                expect.any(Object)
            );
        });

        it('should handle multi-turn conversation', async () => {
            mockedAxios.post.mockResolvedValueOnce({
                data: { message: { content: 'Response to context' } }
            });

            await llm.chat([
                { role: 'system', content: 'You are helpful' },
                { role: 'user', content: 'Hello' },
                { role: 'assistant', content: 'Hi there!' },
                { role: 'user', content: 'Follow up question' }
            ]);

            expect(mockedAxios.post).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({
                    messages: expect.arrayContaining([
                        expect.objectContaining({ role: 'system' }),
                        expect.objectContaining({ role: 'user' }),
                        expect.objectContaining({ role: 'assistant' }),
                        expect.objectContaining({ role: 'user' })
                    ])
                }),
                expect.any(Object)
            );
        });
    });

    describe('judge', () => {
        it('should parse valid JSON response', async () => {
            mockedAxios.post.mockResolvedValueOnce({
                data: {
                    message: {
                        content: '{"score": 0.8, "passed": true, "reason": "Good response"}'
                    }
                }
            });

            const result = await llm.judge(
                'What is AI?',
                'AI is artificial intelligence',
                'Evaluate accuracy'
            );

            expect(result.score).toBe(0.8);
            expect(result.passed).toBe(true);
            expect(result.reason).toBe('Good response');
        });

        it('should parse JSON from code block', async () => {
            mockedAxios.post.mockResolvedValueOnce({
                data: {
                    message: {
                        content: '```json\n{"score": 0.9, "passed": true, "reason": "Excellent"}\n```'
                    }
                }
            });

            const result = await llm.judge('query', 'response', 'criteria');
            expect(result.score).toBe(0.9);
        });

        it('should extract score from unstructured response', async () => {
            mockedAxios.post.mockResolvedValueOnce({
                data: {
                    message: {
                        content: 'The response is good. Score: 7 out of 10.'
                    }
                }
            });

            const result = await llm.judge('query', 'response', 'criteria');
            expect(result.score).toBe(0.7); // Normalized from 7/10
        });

        it('should normalize scores on 0-100 scale', async () => {
            mockedAxios.post.mockResolvedValueOnce({
                data: {
                    message: {
                        content: '{"score": 85, "reason": "Good"}'
                    }
                }
            });

            const result = await llm.judge('query', 'response', 'criteria');
            expect(result.score).toBe(0.85);
        });

        it('should include context in evaluation', async () => {
            mockedAxios.post.mockResolvedValueOnce({
                data: {
                    message: {
                        content: '{"score": 1.0, "passed": true, "reason": "Grounded"}'
                    }
                }
            });

            await llm.judge('query', 'response', 'criteria', 'some context');
            expect(mockedAxios.post).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({
                    messages: expect.arrayContaining([
                        expect.objectContaining({
                            content: expect.stringContaining('Context')
                        })
                    ])
                }),
                expect.any(Object)
            );
        });

        it('should return default score on unparseable response', async () => {
            mockedAxios.post.mockResolvedValueOnce({
                data: {
                    message: {
                        content: 'This is completely unparseable as evaluation.'
                    }
                }
            });

            const result = await llm.judge('query', 'response', 'criteria');
            expect(result.score).toBe(0.5);
            expect(result.rawResponse).toBeDefined();
        });
    });

    describe('batchJudge', () => {
        it('should evaluate multiple items', async () => {
            mockedAxios.post
                .mockResolvedValueOnce({
                    data: { message: { content: '{"score": 0.8, "passed": true, "reason": "Good"}' } }
                })
                .mockResolvedValueOnce({
                    data: { message: { content: '{"score": 0.6, "passed": true, "reason": "OK"}' } }
                });

            const results = await llm.batchJudge([
                { query: 'q1', response: 'r1', criteria: 'c1' },
                { query: 'q2', response: 'r2', criteria: 'c2' }
            ]);

            expect(results).toHaveLength(2);
            expect(results[0].score).toBe(0.8);
            expect(results[1].score).toBe(0.6);
        });

        it('should handle errors in batch', async () => {
            mockedAxios.post
                .mockResolvedValueOnce({
                    data: { message: { content: '{"score": 0.8, "passed": true, "reason": "Good"}' } }
                })
                .mockRejectedValueOnce({ message: 'Error' });

            const results = await llm.batchJudge([
                { query: 'q1', response: 'r1', criteria: 'c1' },
                { query: 'q2', response: 'r2', criteria: 'c2' }
            ]);

            expect(results).toHaveLength(2);
            expect(results[0].score).toBe(0.8);
            expect(results[1].score).toBe(0);
            expect(results[1].passed).toBe(false);
        });
    });
});

describe('OpenAILLM', () => {
    const originalEnv = process.env;

    beforeEach(() => {
        jest.resetModules();
        process.env = { ...originalEnv };
    });

    afterAll(() => {
        process.env = originalEnv;
    });

    describe('constructor', () => {
        it('should use default configuration', () => {
            const llm = new OpenAILLM({ apiKey: 'test-key' });
            expect(llm).toBeDefined();
            expect(llm.provider).toBe('openai');
            expect(llm.model).toBe('gpt-4o-mini');
        });

        it('should accept custom configuration', () => {
            const llm = new OpenAILLM({
                model: 'gpt-4o',
                apiKey: 'test-key',
                temperature: 0.7
            });
            expect(llm.model).toBe('gpt-4o');
        });
    });

    describe('isAvailable', () => {
        it('should return false when no API key', async () => {
            delete process.env.OPENAI_API_KEY;
            const llm = new OpenAILLM();
            const result = await llm.isAvailable();
            expect(result).toBe(false);
        });

        // Note: Tests that require the actual openai package are skipped
        // They would need the package installed as a dev dependency
    });
});

describe('AnthropicLLM', () => {
    const originalEnv = process.env;

    beforeEach(() => {
        jest.resetModules();
        process.env = { ...originalEnv };
    });

    afterAll(() => {
        process.env = originalEnv;
    });

    describe('constructor', () => {
        it('should use default configuration', () => {
            const llm = new AnthropicLLM({ apiKey: 'test-key' });
            expect(llm).toBeDefined();
            expect(llm.provider).toBe('anthropic');
            expect(llm.model).toBe('claude-3-5-sonnet-20241022');
        });

        it('should accept custom configuration', () => {
            const llm = new AnthropicLLM({
                model: 'claude-3-opus-20240229',
                apiKey: 'test-key'
            });
            expect(llm.model).toBe('claude-3-opus-20240229');
        });
    });

    describe('isAvailable', () => {
        it('should return false when no API key', async () => {
            delete process.env.ANTHROPIC_API_KEY;
            const llm = new AnthropicLLM();
            const result = await llm.isAvailable();
            expect(result).toBe(false);
        });

        // Note: Tests that require the actual anthropic package are skipped
        // They would need the package installed as a dev dependency
    });
});

describe('LLMFactory', () => {
    describe('fromString', () => {
        it('should create OllamaLLM from model name', () => {
            const llm = LLMFactory.fromString('llama3.2');
            expect(llm).toBeInstanceOf(OllamaLLM);
            expect(llm.model).toBe('llama3.2');
        });

        it('should create OllamaLLM from ollama/model format', () => {
            const llm = LLMFactory.fromString('ollama/mistral');
            expect(llm).toBeInstanceOf(OllamaLLM);
            expect(llm.model).toBe('mistral');
        });

        it('should create OpenAILLM from openai/model format', () => {
            const llm = LLMFactory.fromString('openai/gpt-4o');
            expect(llm).toBeInstanceOf(OpenAILLM);
            expect(llm.model).toBe('gpt-4o');
        });

        it('should create AnthropicLLM from anthropic/model format', () => {
            const llm = LLMFactory.fromString('anthropic/claude-3-5-sonnet-20241022');
            expect(llm).toBeInstanceOf(AnthropicLLM);
            expect(llm.model).toBe('claude-3-5-sonnet-20241022');
        });

        it('should throw for unsupported provider', () => {
            expect(() => LLMFactory.fromString('unknown/model')).toThrow(
                'Unsupported LLM provider: unknown'
            );
        });
    });

    describe('create', () => {
        it('should create Ollama provider', () => {
            const llm = LLMFactory.create('ollama', { model: 'llama3.2' });
            expect(llm).toBeInstanceOf(OllamaLLM);
        });

        it('should create OpenAI provider', () => {
            const llm = LLMFactory.create('openai', { apiKey: 'test' });
            expect(llm).toBeInstanceOf(OpenAILLM);
        });

        it('should create Anthropic provider', () => {
            const llm = LLMFactory.create('anthropic', { apiKey: 'test' });
            expect(llm).toBeInstanceOf(AnthropicLLM);
        });
    });

    describe('createDefault', () => {
        it('should create default OllamaLLM', () => {
            const llm = LLMFactory.createDefault();
            expect(llm).toBeInstanceOf(OllamaLLM);
        });
    });

    describe('checkAvailability', () => {
        it('should check all providers', async () => {
            mockedAxios.get.mockResolvedValueOnce({ status: 200, data: { models: [] } });

            const availability = await LLMFactory.checkAvailability();
            expect(availability).toHaveProperty('ollama');
            expect(availability).toHaveProperty('openai');
            expect(availability).toHaveProperty('anthropic');
        });
    });

    describe('listAvailable', () => {
        it('should list available providers', async () => {
            mockedAxios.get.mockResolvedValueOnce({ status: 200, data: { models: [] } });

            const available = await LLMFactory.listAvailable();
            expect(Array.isArray(available)).toBe(true);
        });
    });
});

describe('LocalLLMFactory (backwards compatibility)', () => {
    it('should be exported and work as LLMFactory', () => {
        expect(LocalLLMFactory).toBeDefined();
        expect(LocalLLMFactory.fromString).toBeDefined();
        expect(LocalLLMFactory.createDefault).toBeDefined();
    });

    describe('fromString', () => {
        it('should create OllamaLLM from model name', () => {
            const llm = LocalLLMFactory.fromString('llama3.2');
            expect(llm).toBeInstanceOf(OllamaLLM);
        });

        it('should support OpenAI format', () => {
            const llm = LocalLLMFactory.fromString('openai/gpt-4o');
            expect(llm).toBeInstanceOf(OpenAILLM);
        });
    });

    describe('createDefault', () => {
        it('should create default OllamaLLM', () => {
            const llm = LocalLLMFactory.createDefault();
            expect(llm).toBeInstanceOf(OllamaLLM);
        });
    });
});

describe('BaseLLM interface compliance', () => {
    it('OllamaLLM should implement BaseLLM', () => {
        const llm: BaseLLM = new OllamaLLM();
        expect(llm.provider).toBe('ollama');
        expect(llm.model).toBeDefined();
        expect(typeof llm.isAvailable).toBe('function');
        expect(typeof llm.generate).toBe('function');
        expect(typeof llm.chat).toBe('function');
        expect(typeof llm.judge).toBe('function');
        expect(typeof llm.batchJudge).toBe('function');
    });

    it('OpenAILLM should implement BaseLLM', () => {
        const llm: BaseLLM = new OpenAILLM({ apiKey: 'test' });
        expect(llm.provider).toBe('openai');
        expect(llm.model).toBeDefined();
        expect(typeof llm.isAvailable).toBe('function');
        expect(typeof llm.generate).toBe('function');
        expect(typeof llm.chat).toBe('function');
        expect(typeof llm.judge).toBe('function');
        expect(typeof llm.batchJudge).toBe('function');
    });

    it('AnthropicLLM should implement BaseLLM', () => {
        const llm: BaseLLM = new AnthropicLLM({ apiKey: 'test' });
        expect(llm.provider).toBe('anthropic');
        expect(llm.model).toBeDefined();
        expect(typeof llm.isAvailable).toBe('function');
        expect(typeof llm.generate).toBe('function');
        expect(typeof llm.chat).toBe('function');
        expect(typeof llm.judge).toBe('function');
        expect(typeof llm.batchJudge).toBe('function');
    });
});
