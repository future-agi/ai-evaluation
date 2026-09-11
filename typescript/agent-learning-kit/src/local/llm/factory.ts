/**
 * Factory for creating LLM provider instances.
 * @module local/llm/factory
 */

import { BaseLLM, LLMProvider, OllamaConfig, OpenAIConfig, AnthropicConfig } from './types';
import { OllamaLLM } from './ollama';
import { OpenAILLM } from './openai';
import { AnthropicLLM } from './anthropic';

/**
 * Configuration options for LLM factory
 */
export interface FactoryConfig {
    /** Default provider if not specified */
    defaultProvider?: LLMProvider;
    /** Ollama configuration */
    ollama?: OllamaConfig;
    /** OpenAI configuration */
    openai?: OpenAIConfig;
    /** Anthropic configuration */
    anthropic?: AnthropicConfig;
}

/**
 * Factory for creating and managing LLM instances.
 *
 * @example
 * ```typescript
 * // Create from string spec
 * const llm1 = LLMFactory.fromString('openai/gpt-4o');
 * const llm2 = LLMFactory.fromString('anthropic/claude-3-5-sonnet-20241022');
 * const llm3 = LLMFactory.fromString('ollama/llama3.2');
 *
 * // Create with explicit config
 * const llm4 = LLMFactory.create('openai', { model: 'gpt-4o', apiKey: 'sk-...' });
 *
 * // Create best available LLM
 * const llm5 = await LLMFactory.createBestAvailable();
 * ```
 */
export class LLMFactory {
    /**
     * Create an LLM from a string specification.
     * Format: "provider/model-name" or just "model-name" (defaults to ollama)
     *
     * @param spec - String specification (e.g., "openai/gpt-4o", "ollama/llama3.2")
     * @returns LLM instance
     */
    static fromString(spec: string): BaseLLM {
        const parts = spec.split('/');

        if (parts.length === 1) {
            // Just model name, default to ollama
            return new OllamaLLM({ model: spec });
        }

        const [provider, model] = parts;
        return this.create(provider.toLowerCase() as LLMProvider, { model });
    }

    /**
     * Create an LLM with explicit provider and configuration.
     *
     * @param provider - LLM provider type
     * @param config - Provider-specific configuration
     * @returns LLM instance
     */
    static create(
        provider: LLMProvider,
        config: OllamaConfig | OpenAIConfig | AnthropicConfig = {}
    ): BaseLLM {
        switch (provider) {
            case 'ollama':
                return new OllamaLLM(config as OllamaConfig);
            case 'openai':
                return new OpenAILLM(config as OpenAIConfig);
            case 'anthropic':
                return new AnthropicLLM(config as AnthropicConfig);
            default:
                throw new Error(`Unsupported LLM provider: ${provider}. Supported: ollama, openai, anthropic`);
        }
    }

    /**
     * Create default Ollama instance.
     * Convenience method for backwards compatibility.
     */
    static createDefault(): OllamaLLM {
        return new OllamaLLM();
    }

    /**
     * Create the best available LLM based on environment.
     * Checks for available API keys and running services.
     *
     * Priority order:
     * 1. OpenAI (if OPENAI_API_KEY is set)
     * 2. Anthropic (if ANTHROPIC_API_KEY is set)
     * 3. Ollama (if server is running)
     *
     * @param config - Optional factory configuration
     * @returns Best available LLM instance
     * @throws Error if no LLM is available
     */
    static async createBestAvailable(config: FactoryConfig = {}): Promise<BaseLLM> {
        // Try OpenAI first
        if (process.env.OPENAI_API_KEY || config.openai?.apiKey) {
            const llm = new OpenAILLM(config.openai);
            if (await llm.isAvailable()) {
                return llm;
            }
        }

        // Try Anthropic
        if (process.env.ANTHROPIC_API_KEY || config.anthropic?.apiKey) {
            const llm = new AnthropicLLM(config.anthropic);
            if (await llm.isAvailable()) {
                return llm;
            }
        }

        // Try Ollama
        const ollama = new OllamaLLM(config.ollama);
        if (await ollama.isAvailable()) {
            return ollama;
        }

        throw new Error(
            'No LLM available. Either:\n' +
            '1. Set OPENAI_API_KEY environment variable\n' +
            '2. Set ANTHROPIC_API_KEY environment variable\n' +
            '3. Start Ollama server (ollama serve)'
        );
    }

    /**
     * Check which LLM providers are available.
     *
     * @param config - Optional factory configuration
     * @returns Object with availability status for each provider
     */
    static async checkAvailability(config: FactoryConfig = {}): Promise<Record<LLMProvider, boolean>> {
        const [ollamaAvailable, openaiAvailable, anthropicAvailable] = await Promise.all([
            new OllamaLLM(config.ollama).isAvailable(),
            new OpenAILLM(config.openai).isAvailable(),
            new AnthropicLLM(config.anthropic).isAvailable()
        ]);

        return {
            ollama: ollamaAvailable,
            openai: openaiAvailable,
            anthropic: anthropicAvailable
        };
    }

    /**
     * List available providers based on current environment.
     *
     * @param config - Optional factory configuration
     * @returns Array of available provider names
     */
    static async listAvailable(config: FactoryConfig = {}): Promise<LLMProvider[]> {
        const availability = await this.checkAvailability(config);
        return (Object.entries(availability) as [LLMProvider, boolean][])
            .filter(([_, available]) => available)
            .map(([provider]) => provider);
    }
}

// Re-export for convenience (backwards compatibility)
export { LLMFactory as LocalLLMFactory };
