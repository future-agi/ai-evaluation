/**
 * Local execution module for running evaluations without API calls.
 * Aligned with Python SDK's fi.evals.local module.
 *
 * @example
 * ```typescript
 * import { LocalEvaluator, ExecutionMode } from '@future-agi/ai-evaluation/local';
 *
 * // Run a metric locally
 * const evaluator = new LocalEvaluator();
 * const result = evaluator.evaluate(
 *     'contains',
 *     [{ response: 'Hello world' }],
 *     { keyword: 'world' }
 * );
 * console.log(result.results.eval_results[0].output); // 1.0
 *
 * // Check if a metric can run locally
 * evaluator.canRunLocally('contains');      // true
 * evaluator.canRunLocally('groundedness');  // false (requires LLM)
 *
 * // Use hybrid mode to automatically route
 * import { HybridEvaluator, OllamaLLM } from '@future-agi/ai-evaluation/local';
 *
 * const llm = new OllamaLLM();
 * const hybrid = new HybridEvaluator({ localLLM: llm });
 *
 * // LLM-based metrics will use local LLM if available
 * const result = await hybrid.evaluate(
 *     'groundedness',
 *     [{ query: 'What is AI?', response: 'AI is artificial intelligence.', context: '...' }]
 * );
 * ```
 */

// Execution mode
export {
    ExecutionMode,
    LOCAL_CAPABLE_METRICS,
    LLM_BASED_METRICS,
    canRunLocally,
    requiresLLM,
    selectExecutionMode
} from './execution-mode';

// Metrics
export {
    MetricResult,
    MetricConfig,
    MetricInput,
    MetricFunction,
    MetricRegistryEntry,
    METRIC_REGISTRY,
    getMetric,
    hasMetric,
    runMetric,
    getAvailableMetrics
} from './metrics';

// String metrics
export {
    regex,
    contains,
    containsAll,
    containsAny,
    containsNone,
    oneLine,
    equals,
    startsWith,
    endsWith,
    lengthLessThan,
    lengthGreaterThan,
    lengthBetween
} from './metrics/string';

// JSON metrics
export {
    containsJson,
    isJson,
    jsonSchema
} from './metrics/json';

// Similarity metrics
export {
    bleuScore,
    rougeScore,
    recallScore,
    levenshteinSimilarity,
    numericSimilarity,
    semanticListContains
} from './metrics/similarity';

// RAG metrics
export {
    // Types
    RAGInput,
    RAGConfig,
    RAGResult,
    tokenize,
    jaccardSimilarity,
    ngramOverlap,
    normalizeContext,
    extractSentences,

    // Metrics
    contextPrecision,
    contextRecall,
    faithfulness,
    groundedness,
    answerRelevance,
    contextRelevance,
    contextUtilization,

    // Registry
    RAG_METRICS,
    RAG_METRIC_NAMES
} from './metrics/rag';

// LLM Providers
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
    LocalLLMConfig,

    // Base class
    AbstractLLM,

    // Providers
    OllamaLLM,
    OpenAILLM,
    AnthropicLLM,

    // Factory
    LLMFactory,
    LocalLLMFactory,
    FactoryConfig
} from './llm';

// Evaluators
export {
    LocalEvaluatorConfig,
    LocalEvaluationResult,
    EvaluationRequest,
    LocalEvaluator,
    HybridEvaluator
} from './evaluator';

// Code Security metrics
export {
    CodeInput,
    CodeSecurityConfig,
    CodeSecurityResult,
    SecurityIssue,
    SEVERITY_LEVELS,
    meetsMinSeverity,
    createSecurityResult,
    sqlInjection,
    noSqlInjection,
    xssDetection,
    noXss,
    secretsDetection,
    noSecrets,
    noHardcodedSecrets,
    allSecurityChecks,
    codeSecurityScan
} from './metrics/code';

// Hallucination Detection
export {
    HallucinationInput,
    HallucinationConfig,
    HallucinationResult,
    Claim,
    hallucinationDetection,
    detectHallucination,
    noHallucination,
    isFactualClaim,
    calculateOverlap
} from './metrics/hallucination';

// AutoEval Pipeline
export {
    // Types
    InputType,
    ContentCharacteristics,
    AutoEvalInput,
    AutoEvalConfig,
    MetricCategory,
    SelectedMetric,
    MetricExecutionResult,
    AutoEvalResult,
    CODE_LANGUAGE_PATTERNS,

    // Analyzer
    analyzeContent,
    describeCharacteristics,

    // Selector
    selectMetrics,
    getRecommendedMetrics,
    buildMetricConfig,

    // Pipeline
    AutoEvalPipeline,
    createAutoEval,
    autoEvaluate
} from './autoeval';

// Streaming Evaluation
export {
    // Types
    StreamingEventType,
    StreamingEvent,
    ChunkEvent,
    MetricUpdateEvent,
    ThresholdAlertEvent,
    CompleteEvent,
    ErrorEvent,
    StreamingEvalEvent,
    StreamingEventHandler,
    StreamingEvalConfig,
    StreamingState,
    STREAMING_CAPABLE_METRICS,
    canStreamMetric,

    // Evaluator
    StreamingEvaluator,
    createStreamingEvaluator,
    evaluateStream,
    streamWithEvaluation
} from './streaming';

// Scanner Pipeline
export {
    // Types
    Severity,
    FindingCategory,
    Finding,
    FindingLocation,
    ScanRule,
    ScanContext,
    ScanConfig,
    ScanResult,
    SEVERITY_VALUES,
    compareSeverity,
    meetsSeverityThreshold,
    generateFindingId,

    // Built-in Rules
    sqlInjectionRule,
    xssRule,
    secretsRule,
    hallucinationRule,
    piiRule,
    todoRule,
    unsafeEvalRule,
    consoleLogRule,
    profanityRule,
    BUILTIN_RULES,
    getRulesByTag,
    getRulesByCategory,
    getDefaultRules,
    getRuleById,

    // Scanner
    Scanner,
    createScanner,
    quickScan,
    securityScan,
    privacyScan
} from './scanner';
