/**
 * Ollama LLM provider for local inference.
 * @module local/llm/ollama
 */

import axios, { AxiosError } from 'axios';
import { AbstractLLM, DEFAULT_CONFIG } from './base';
import { OllamaConfig, ChatMessage, GenerateOptions, ChatOptions } from './types';

/**
 * Default Ollama configuration
 */
const OLLAMA_DEFAULTS: Required<OllamaConfig> = {
    ...DEFAULT_CONFIG,
    model: 'llama3.2',
    baseUrl: 'http://localhost:11434'
};

/**
 * OllamaLLM - Local LLM client using Ollama API.
 * Provides generation, chat, and LLM-as-judge capabilities.
 *
 * @example
 * ```typescript
 * const llm = new OllamaLLM({ model: 'llama3.2' });
 *
 * // Check availability
 * if (await llm.isAvailable()) {
 *     const response = await llm.generate('What is AI?');
 *     console.log(response);
 * }
 *
 * // Use as judge
 * const result = await llm.judge(
 *     'What is the capital of France?',
 *     'The capital of France is Paris.',
 *     'Evaluate if the response is factually correct.'
 * );
 * console.log(result.score); // 1.0
 * ```
 */
export class OllamaLLM extends AbstractLLM {
    readonly provider = 'ollama';
    readonly model: string;

    private baseUrl: string;
    private _available: boolean | null = null;

    constructor(config: OllamaConfig = {}) {
        super(config);
        const fullConfig = { ...OLLAMA_DEFAULTS, ...config };
        this.model = fullConfig.model;
        this.baseUrl = fullConfig.baseUrl;
    }

    /**
     * Check if Ollama is available
     */
    async isAvailable(): Promise<boolean> {
        if (this._available !== null) {
            return this._available;
        }

        try {
            const response = await axios.get(`${this.baseUrl}/api/tags`, {
                timeout: 5000
            });
            this._available = response.status === 200;
            return this._available;
        } catch {
            this._available = false;
            return false;
        }
    }

    /**
     * List available models
     */
    async listModels(): Promise<string[]> {
        try {
            const response = await axios.get(`${this.baseUrl}/api/tags`, {
                timeout: 5000
            });
            return response.data.models?.map((m: any) => m.name) || [];
        } catch {
            return [];
        }
    }

    /**
     * Generate text completion
     */
    async generate(prompt: string, options: GenerateOptions = {}): Promise<string> {
        const payload: any = {
            model: this.model,
            prompt,
            stream: false,
            options: {
                temperature: options.temperature ?? this.config.temperature,
                num_predict: options.maxTokens ?? this.config.maxTokens
            }
        };

        if (options.system) {
            payload.system = options.system;
        }

        try {
            const response = await axios.post(
                `${this.baseUrl}/api/generate`,
                payload,
                { timeout: this.config.timeout * 1000 }
            );
            return response.data.response || '';
        } catch (error) {
            const axiosError = error as AxiosError;
            throw new Error(`Ollama generate failed: ${axiosError.message}`);
        }
    }

    /**
     * Chat completion with message history
     */
    async chat(messages: ChatMessage[], options: ChatOptions = {}): Promise<string> {
        const payload = {
            model: this.model,
            messages,
            stream: false,
            options: {
                temperature: options.temperature ?? this.config.temperature,
                num_predict: options.maxTokens ?? this.config.maxTokens
            }
        };

        try {
            const response = await axios.post(
                `${this.baseUrl}/api/chat`,
                payload,
                { timeout: this.config.timeout * 1000 }
            );
            return response.data.message?.content || '';
        } catch (error) {
            const axiosError = error as AxiosError;
            throw new Error(`Ollama chat failed: ${axiosError.message}`);
        }
    }
}
