/**
 * Anthropic LLM provider.
 * @module local/llm/anthropic
 */

import { AbstractLLM, DEFAULT_CONFIG } from './base';
import { AnthropicConfig, ChatMessage, GenerateOptions, ChatOptions } from './types';

/**
 * Default Anthropic configuration
 */
const ANTHROPIC_DEFAULTS: Required<Omit<AnthropicConfig, 'apiKey'>> & { apiKey?: string } = {
    ...DEFAULT_CONFIG,
    model: 'claude-3-5-sonnet-20241022',
    baseUrl: 'https://api.anthropic.com',
    apiKey: undefined
};

/**
 * AnthropicLLM - LLM client using Anthropic API.
 * Supports Claude 3, Claude 3.5, and other Anthropic models.
 *
 * @example
 * ```typescript
 * const llm = new AnthropicLLM({
 *     apiKey: 'sk-ant-...',  // or use ANTHROPIC_API_KEY env var
 *     model: 'claude-3-5-sonnet-20241022'
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
export class AnthropicLLM extends AbstractLLM {
    readonly provider = 'anthropic';
    readonly model: string;

    private apiKey: string;
    private baseUrl: string;
    private _client: any = null;

    constructor(config: AnthropicConfig = {}) {
        super(config);
        const fullConfig = { ...ANTHROPIC_DEFAULTS, ...config };
        this.model = fullConfig.model!;
        this.baseUrl = fullConfig.baseUrl!;

        // Get API key from config or environment
        this.apiKey = config.apiKey || process.env.ANTHROPIC_API_KEY || '';
    }

    /**
     * Check if Anthropic is available (has valid API key)
     */
    async isAvailable(): Promise<boolean> {
        if (!this.apiKey) {
            return false;
        }

        try {
            // Try to load the Anthropic package
            await this.getClient();
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Get or create Anthropic client (lazy loading)
     */
    private async getClient(): Promise<any> {
        if (this._client) {
            return this._client;
        }

        try {
            // Dynamic import to avoid compile-time dependency
            // @ts-expect-error - Optional dependency, may not be installed
            const { default: Anthropic } = await import('@anthropic-ai/sdk');
            this._client = new Anthropic({
                apiKey: this.apiKey,
                baseURL: this.baseUrl,
                timeout: this.config.timeout * 1000
            });
            return this._client;
        } catch (error) {
            throw new Error(
                'Anthropic SDK not installed. Run: npm install @anthropic-ai/sdk'
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

        // Extract system message if present
        let systemMessage: string | undefined;
        const nonSystemMessages = messages.filter(m => {
            if (m.role === 'system') {
                systemMessage = m.content;
                return false;
            }
            return true;
        });

        try {
            const response = await client.messages.create({
                model: this.model,
                system: systemMessage,
                messages: nonSystemMessages.map(m => ({
                    role: m.role as 'user' | 'assistant',
                    content: m.content
                })),
                temperature: options.temperature ?? this.config.temperature,
                max_tokens: options.maxTokens ?? this.config.maxTokens
            });

            // Extract text from content blocks
            const content = response.content;
            if (Array.isArray(content)) {
                return content
                    .filter((block: any) => block.type === 'text')
                    .map((block: any) => block.text)
                    .join('');
            }
            return '';
        } catch (error: any) {
            if (error?.status === 401) {
                throw new Error('Anthropic API key is invalid');
            }
            if (error?.status === 429) {
                throw new Error('Anthropic rate limit exceeded');
            }
            throw new Error(`Anthropic chat failed: ${error.message}`);
        }
    }
}
