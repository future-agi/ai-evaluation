/**
 * LLM providers for local and cloud-based inference.
 * This file re-exports from the llm/ module for backwards compatibility.
 *
 * @module local/llm
 * @deprecated Import from './llm/index' instead for new code
 */

// Re-export everything from the new llm module structure
export {
    // Types
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
    LLMConfig,

    // Base class
    AbstractLLM,
    DEFAULT_CONFIG,

    // Providers
    OllamaLLM,
    OpenAILLM,
    AnthropicLLM,

    // Factory
    LLMFactory,
    LocalLLMFactory,
    FactoryConfig
} from './llm/index';

// Backwards compatibility type alias
export type LocalLLMConfig = import('./llm/types').OllamaConfig;
