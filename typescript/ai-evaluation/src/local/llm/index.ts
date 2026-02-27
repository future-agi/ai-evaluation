/**
 * LLM providers for local and cloud-based inference.
 * @module local/llm
 *
 * @example
 * ```typescript
 * import {
 *     OllamaLLM,
 *     OpenAILLM,
 *     AnthropicLLM,
 *     LLMFactory
 * } from '@future-agi/ai-evaluation/local';
 *
 * // Use Ollama for local inference
 * const ollama = new OllamaLLM({ model: 'llama3.2' });
 *
 * // Use OpenAI
 * const openai = new OpenAILLM({ model: 'gpt-4o' });
 *
 * // Use Anthropic
 * const anthropic = new AnthropicLLM({ model: 'claude-3-5-sonnet-20241022' });
 *
 * // Use factory for flexible instantiation
 * const llm = LLMFactory.fromString('openai/gpt-4o-mini');
 *
 * // Get best available LLM
 * const best = await LLMFactory.createBestAvailable();
 * ```
 */

// Types
export {
    BaseLLM,
    BaseLLMConfig,
    ChatMessage,
    GenerateOptions,
    ChatOptions,
    JudgeResult,
    JudgeInput,
    OllamaConfig,
    OpenAIConfig,
    AnthropicConfig,
    LLMProvider,
    LLMConfig
} from './types';

// Base class
export { AbstractLLM, DEFAULT_CONFIG } from './base';

// Providers
export { OllamaLLM } from './ollama';
export { OpenAILLM } from './openai';
export { AnthropicLLM } from './anthropic';

// Factory
export { LLMFactory, LocalLLMFactory, FactoryConfig } from './factory';
