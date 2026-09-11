/**
 * Type definitions for LLM providers.
 * @module local/llm/types
 */

/**
 * Configuration common to all LLM providers
 */
export interface BaseLLMConfig {
    /** Sampling temperature (0.0 = deterministic, higher = more random) */
    temperature?: number;
    /** Maximum tokens in generated response */
    maxTokens?: number;
    /** Request timeout in seconds */
    timeout?: number;
}

/**
 * Message format for chat API
 */
export interface ChatMessage {
    role: 'system' | 'user' | 'assistant';
    content: string;
}

/**
 * Options for text generation
 */
export interface GenerateOptions {
    /** System prompt for context */
    system?: string;
    /** Sampling temperature override */
    temperature?: number;
    /** Max tokens override */
    maxTokens?: number;
}

/**
 * Options for chat completion
 */
export interface ChatOptions {
    /** Sampling temperature override */
    temperature?: number;
    /** Max tokens override */
    maxTokens?: number;
}

/**
 * Result from judge evaluation
 */
export interface JudgeResult {
    /** Evaluation score from 0.0 to 1.0 */
    score: number;
    /** Whether the evaluation passed (typically score >= 0.5) */
    passed: boolean;
    /** Human-readable explanation of the judgment */
    reason: string;
    /** Raw LLM response before parsing (for debugging) */
    rawResponse?: string;
}

/**
 * Input for batch judge evaluation
 */
export interface JudgeInput {
    query: string;
    response: string;
    criteria: string;
    context?: string;
}

/**
 * Base interface for all LLM providers.
 * All providers must implement these methods.
 */
export interface BaseLLM {
    /** Provider name for identification */
    readonly provider: string;

    /** Model name being used */
    readonly model: string;

    /**
     * Check if the LLM provider is available and configured
     */
    isAvailable(): Promise<boolean>;

    /**
     * Generate text completion
     */
    generate(prompt: string, options?: GenerateOptions): Promise<string>;

    /**
     * Chat completion with message history
     */
    chat(messages: ChatMessage[], options?: ChatOptions): Promise<string>;

    /**
     * Use LLM as a judge to evaluate a response
     */
    judge(
        query: string,
        response: string,
        criteria: string,
        context?: string
    ): Promise<JudgeResult>;

    /**
     * Batch evaluate multiple items
     */
    batchJudge(evaluations: JudgeInput[]): Promise<JudgeResult[]>;
}

/**
 * Ollama-specific configuration
 */
export interface OllamaConfig extends BaseLLMConfig {
    /** Model name (e.g., 'llama3.2', 'mistral') */
    model?: string;
    /** Ollama server URL */
    baseUrl?: string;
}

/**
 * OpenAI-specific configuration
 */
export interface OpenAIConfig extends BaseLLMConfig {
    /** Model name (e.g., 'gpt-4o', 'gpt-4o-mini') */
    model?: string;
    /** OpenAI API key (or use OPENAI_API_KEY env var) */
    apiKey?: string;
    /** Base URL for API (for Azure or proxies) */
    baseUrl?: string;
    /** Organization ID (optional) */
    organization?: string;
}

/**
 * Anthropic-specific configuration
 */
export interface AnthropicConfig extends BaseLLMConfig {
    /** Model name (e.g., 'claude-3-sonnet-20240229') */
    model?: string;
    /** Anthropic API key (or use ANTHROPIC_API_KEY env var) */
    apiKey?: string;
    /** Base URL for API */
    baseUrl?: string;
}

/**
 * Available LLM provider types
 */
export type LLMProvider = 'ollama' | 'openai' | 'anthropic';

/**
 * Union of all LLM configurations
 */
export type LLMConfig = OllamaConfig | OpenAIConfig | AnthropicConfig;
