/**
 * OpenAI LLM provider.
 * @module local/llm/openai
 */

import { AbstractLLM, DEFAULT_CONFIG } from './base';
import { OpenAIConfig, ChatMessage, GenerateOptions, ChatOptions } from './types';

/**
 * Default OpenAI configuration
 */
const OPENAI_DEFAULTS: Required<Omit<OpenAIConfig, 'apiKey' | 'organization'>> & { apiKey?: string; organization?: string } = {
    ...DEFAULT_CONFIG,
    model: 'gpt-4o-mini',
    baseUrl: 'https://api.openai.com/v1',
    apiKey: undefined,
    organization: undefined
};

/**
 * OpenAILLM - LLM client using OpenAI API.
 * Supports GPT-4, GPT-4 Turbo, GPT-4o, and other OpenAI models.
 *
 * @example
 * ```typescript
 * const llm = new OpenAILLM({
 *     apiKey: 'sk-...',  // or use OPENAI_API_KEY env var
 *     model: 'gpt-4o'
 * });
 *
 * const response = await llm.generate('What is AI?');
 * console.log(response);
 *
 * // Use as judge
 * const result = await llm.judge(
 *     'What is the capital of France?',
 *     'The capital of France is Paris.',
 *     'Evaluate if the response is factually correct.'
 * );
 * ```
 */
export class OpenAILLM extends AbstractLLM {
    readonly provider = 'openai';
    readonly model: string;

    private apiKey: string;
    private baseUrl: string;
    private organization?: string;
    private _client: any = null;

    constructor(config: OpenAIConfig = {}) {
        super(config);
        const fullConfig = { ...OPENAI_DEFAULTS, ...config };
        this.model = fullConfig.model!;
        this.baseUrl = fullConfig.baseUrl!;
        this.organization = fullConfig.organization;

        // Get API key from config or environment
        this.apiKey = config.apiKey || process.env.OPENAI_API_KEY || '';
    }

    /**
     * Check if OpenAI is available (has valid API key)
     */
    async isAvailable(): Promise<boolean> {
        if (!this.apiKey) {
            return false;
        }

        try {
            // Try to load the OpenAI package
            await this.getClient();
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Get or create OpenAI client (lazy loading)
     */
    private async getClient(): Promise<any> {
        if (this._client) {
            return this._client;
        }

        try {
            // Dynamic import to avoid compile-time dependency
            // @ts-expect-error - Optional dependency, may not be installed
            const { default: OpenAI } = await import('openai');
            this._client = new OpenAI({
                apiKey: this.apiKey,
                baseURL: this.baseUrl,
                organization: this.organization,
                timeout: this.config.timeout * 1000
            });
            return this._client;
        } catch (error) {
            throw new Error(
                'OpenAI package not installed. Run: npm install openai'
            );
        }
    }

    /**
     * Generate text completion
     */
    async generate(prompt: string, options: GenerateOptions = {}): Promise<string> {
        const messages: ChatMessage[] = [];

        if (options.system) {
            messages.push({ role: 'system', content: options.system });
        }
        messages.push({ role: 'user', content: prompt });

        return this.chat(messages, options);
    }

    /**
     * Chat completion with message history
     */
    async chat(messages: ChatMessage[], options: ChatOptions = {}): Promise<string> {
        const client = await this.getClient();

        try {
            const response = await client.chat.completions.create({
                model: this.model,
                messages: messages.map(m => ({
                    role: m.role,
                    content: m.content
                })),
                temperature: options.temperature ?? this.config.temperature,
                max_tokens: options.maxTokens ?? this.config.maxTokens
            });

            return response.choices[0]?.message?.content || '';
        } catch (error: any) {
            if (error?.status === 401) {
                throw new Error('OpenAI API key is invalid');
            }
            if (error?.status === 429) {
                throw new Error('OpenAI rate limit exceeded');
            }
            throw new Error(`OpenAI chat failed: ${error.message}`);
        }
    }
}
