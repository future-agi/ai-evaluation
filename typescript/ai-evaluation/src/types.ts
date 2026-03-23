/**
 * @fileoverview Type definitions for the AI Evaluation SDK.
 * These types are aligned with the Python SDK for API compatibility.
 * @module types
 */

/**
 * Configuration parameter definition for evaluation templates.
 */
interface ConfigParam {
  /** The parameter type (e.g., 'string', 'number', 'boolean') */
  type: string;
  /** Default value for the parameter */
  default?: any;
}

/**
 * Possible configuration values for evaluation templates.
 * These correspond to various metric-specific settings that can be customized.
 */
interface ConfigPossibleValues {
  /** Minimum length constraint for text validation */
  min_length?: number;
  /** List of validation rules to apply */
  validations?: string[];
  /** Custom prompt for LLM-based evaluation */
  eval_prompt?: string;
  /** Substring to search for in text */
  substring?: string;
  /** Model identifier for LLM evaluations */
  model?: string;
  /** Code snippet for code-based evaluations */
  code?: string;
  /** List of keywords to check for */
  keywords?: string[];
  /** Single keyword to check for */
  keyword?: string;
  /** Threshold below which evaluation fails */
  failure_threshold?: number;
  /** HTTP headers for webhook/API evaluations */
  headers?: Record<string, string>;
  /** Whether string comparisons are case-sensitive */
  case_sensitive?: boolean;
  /** Similarity comparison method to use */
  comparator?: string;
  /** Payload data for webhook evaluations */
  payload?: Record<string, any>;
  /** URL for webhook or API evaluations */
  url?: string;
  /** Input text for evaluation */
  input?: string;
  /** Maximum length constraint for text validation */
  max_length?: number;
  /** Whether multiple choices are allowed */
  multi_choice?: boolean;
  /** System prompt for LLM evaluations */
  system_prompt?: string;
  /** Regex pattern for pattern matching */
  pattern?: string;
  /** Criteria description for grading evaluations */
  grading_criteria?: string;
  /** JSON schema for structure validation */
  _schema?: string;
  /** Custom rule prompt for rule-based evaluation */
  rule_prompt?: string;
  /** List of valid choices for selection evaluations */
  choices?: string[];
}

/**
 * Represents an annotation for a specific field in a datapoint.
 * Used for logging and tracking field-level metadata in evaluations.
 */
interface DatapointFieldAnnotation {
  /** Name of the field being annotated */
  field_name: string;
  /** Text content of the annotation */
  text: string;
  /** Type/category of annotation (e.g., 'label', 'correction', 'comment') */
  annotation_type: string;
  /** Additional notes or context for the annotation */
  annotation_note: string;
}

/**
 * Represents a single metric within an evaluation result.
 * Used for detailed breakdown of evaluation scores.
 *
 * @example
 * ```typescript
 * const metric: EvalResultMetric = {
 *   id: 'accuracy',
 *   value: 0.92
 * };
 * ```
 */
interface EvalResultMetric {
  /** Unique identifier or name for this metric */
  id: string | number;
  /** Metric value (can be numeric score, string label, or array of values) */
  value: string | number | any[];
}

/**
 * Represents the result of a single LLM evaluation.
 * Aligned with Python SDK structure for API compatibility.
 *
 * @example
 * ```typescript
 * const result: EvalResult = {
 *   name: 'Factual Accuracy',
 *   output: 0.95,
 *   reason: 'Response is factually correct',
 *   runtime: 1.23,
 *   output_type: 'score'
 * };
 * ```
 */
interface EvalResult {
  /** Name of the evaluation metric */
  name: string;
  /** Evaluation output (typically a score 0-1, boolean, or structured data) */
  output?: any;
  /** Human-readable explanation of the evaluation result */
  reason?: string;
  /** Execution time in seconds */
  runtime: number;
  /** Type of output ('score', 'boolean', 'json', 'text') */
  output_type?: string;
  /** Unique identifier for async evaluations */
  eval_id?: string;
  /** @deprecated Legacy field - additional evaluation data */
  data?: Record<string, any> | any[];
  /** @deprecated Legacy field - whether the evaluation failed */
  failure?: boolean;
  /** Additional metadata (usage stats, cost, explanation) */
  metadata?: string | any[] | Record<string, any>;
  /** Detailed metrics breakdown */
  metrics?: EvalResultMetric[];
}

/**
 * Represents the result of a batch evaluation run.
 * Contains an array of individual evaluation results.
 *
 * @example
 * ```typescript
 * const batchResult: BatchRunResult = {
 *   eval_results: [
 *     { name: 'Accuracy', output: 0.95, runtime: 1.2 },
 *     { name: 'Relevance', output: 0.88, runtime: 0.9 }
 *   ]
 * };
 * ```
 */
interface BatchRunResult {
  /** Array of evaluation results (null entries indicate failed evaluations) */
  eval_results: (EvalResult | null)[];
}

/**
 * Standard input field names required by various evaluation templates.
 * Use these to ensure correct field naming when providing evaluation inputs.
 */
enum RequiredKeys {
  /** Raw text content */
  text = "text",
  /** LLM response/output to evaluate */
  response = "response",
  /** User query/question */
  query = "query",
  /** Context information for grounding */
  context = "context",
  /** Expected/reference response for comparison */
  expected_response = "expected_response",
  /** Expected/reference text for comparison */
  expected_text = "expected_text",
  /** Document content for analysis */
  document = "document",
  /** Generic input field */
  input = "input",
  /** Generic output field */
  output = "output",
  /** Prompt template or content */
  prompt = "prompt",
  /** URL of image for vision evaluations */
  image_url = "image_url",
  /** Input image URL for image comparisons */
  input_image_url = "input_image_url",
  /** Output image URL for image comparisons */
  output_image_url = "output_image_url",
  /** Actual JSON for structure comparison */
  actual_json = "actual_json",
  /** Expected JSON for structure comparison */
  expected_json = "expected_json",
  /** Conversation messages array */
  messages = "messages"
}

/**
 * Tags for categorizing evaluation templates.
 * Used for filtering and organizing available evaluations.
 */
enum EvalTags {
  /** Conversational/dialogue evaluations */
  CONVERSATION = "CONVERSATION",
  /** Hallucination detection evaluations */
  HALLUCINATION = "HALLUCINATION",
  /** Retrieval-Augmented Generation evaluations */
  RAG = "RAG",
  /** Beta/upcoming evaluation features */
  FUTURE_EVALS = "FUTURE_EVALS",
  /** LLM-as-judge evaluations */
  LLMS = "LLMS",
  /** Custom user-defined evaluations */
  CUSTOM = "CUSTOM",
  /** Function calling evaluations */
  FUNCTION = "FUNCTION",
  /** Image/vision evaluations */
  IMAGE = "IMAGE",
  /** Safety and content moderation evaluations */
  SAFETY = "SAFETY",
  /** Text quality evaluations */
  TEXT = "TEXT"
}

/**
 * String similarity comparison methods.
 * Used for text comparison and matching evaluations.
 */
enum Comparator {
  /** Cosine similarity based on vector representations */
  COSINE = "CosineSimilarity",
  /** Normalized Levenshtein (edit distance) similarity */
  LEVENSHTEIN = "NormalisedLevenshteinSimilarity",
  /** Jaro-Winkler similarity (optimized for short strings) */
  JARO_WINKLER = "JaroWincklerSimilarity",
  /** Jaccard index (set-based similarity) */
  JACCARD = "JaccardSimilarity",
  /** Sørensen-Dice coefficient (set overlap) */
  SORENSEN_DICE = "SorensenDiceSimilarity"
}

/**
 * Output types for evaluation results
 */
enum OutputType {
  SCORE = "score",
  BOOLEAN = "boolean",
  JSON = "json",
  TEXT = "text"
}

/**
 * Execution mode for local/cloud evaluation routing
 */
enum ExecutionMode {
  LOCAL = "local",
  CLOUD = "cloud",
  HYBRID = "hybrid"
}

/**
 * Configuration options for the Evaluator class.
 * API keys can be provided here or via environment variables.
 *
 * @example
 * ```typescript
 * const config: EvaluatorConfig = {
 *   fiApiKey: 'your-api-key',
 *   fiSecretKey: 'your-secret-key',
 *   timeout: 300,
 *   maxWorkers: 8
 * };
 * ```
 */
interface EvaluatorConfig {
  /** Future AGI API key (or use FI_API_KEY env var) */
  fiApiKey?: string;
  /** Future AGI secret key (or use FI_SECRET_KEY env var) */
  fiSecretKey?: string;
  /** Base URL for API requests (optional, for custom deployments) */
  fiBaseUrl?: string;
  /** Request timeout in seconds (default: 200) */
  timeout?: number;
  /** Maximum queue size for batch operations */
  maxQueue?: number;
  /** Maximum concurrent workers for parallel execution (default: 8) */
  maxWorkers?: number;
  /** Langfuse secret key for observability integration */
  langfuseSecretKey?: string;
  /** Langfuse public key for observability integration */
  langfusePublicKey?: string;
  /** Langfuse host URL (default: cloud.langfuse.com) */
  langfuseHost?: string;
}

/**
 * Options for individual evaluation calls.
 * These override the Evaluator's default configuration.
 *
 * @example
 * ```typescript
 * const options: EvaluateOptions = {
 *   modelName: 'gpt-4o',
 *   customEvalName: 'my-eval-run',
 *   isAsync: true
 * };
 * ```
 */
/**
 * Explanation detail levels for eval results.
 *
 * Controls how much detail the LLM includes in its scoring explanation:
 * - `quick`:    Single sentence verdict (max 20 words).
 * - `detailed`: Intro sentence + 2-3 bullet points with evidence (default).
 * - `thorough`: Full reasoning chain with evidence, steps, edge cases, and recommendations.
 */
type ExplanationDetail = 'quick' | 'detailed' | 'thorough';

interface EvaluateOptions {
  /** Request timeout in seconds (overrides Evaluator config) */
  timeout?: number;
  /** Model to use for LLM-based evaluations (e.g., 'gpt-4o', 'claude-3-sonnet') */
  modelName?: string;
  /** Custom name for this evaluation run (for tracking/logging) */
  customEvalName?: string;
  /** Enable OpenTelemetry tracing for this evaluation */
  traceEval?: boolean;
  /** Platform for logging results ('langfuse') */
  platform?: string;
  /** Run asynchronously - returns immediately with eval_id, poll for results */
  isAsync?: boolean;
  /** Enable error localization to identify specific failure points */
  errorLocalizer?: boolean;
  /** Eval-specific configuration, e.g. { k: 3 } for retrieval metrics (recall_at_k, precision_at_k, ndcg_at_k) */
  evalConfig?: Record<string, any>;
  /** Explanation detail level: 'quick' (one-line), 'detailed' (default, bullets), 'thorough' (full reasoning chain) */
  explanationDetail?: ExplanationDetail;
}

/**
 * Input data for pipeline evaluation.
 * Flexible structure to accommodate various ML pipeline outputs.
 *
 * @example
 * ```typescript
 * const pipelineData: PipelineEvalData = {
 *   input: 'user query',
 *   output: 'model response',
 *   context: 'retrieved context',
 *   latency: 0.5
 * };
 * ```
 */
interface PipelineEvalData {
  /** Dynamic key-value pairs for pipeline data */
  [key: string]: any;
}

/**
 * Result from pipeline evaluation.
 * Contains aggregated metrics and per-sample results.
 *
 * @example
 * ```typescript
 * const result: PipelineResult = {
 *   overall_score: 0.87,
 *   metrics: { accuracy: 0.92, relevance: 0.85 },
 *   samples: [...]
 * };
 * ```
 */
interface PipelineResult {
  /** Dynamic key-value pairs for pipeline results */
  [key: string]: any;
}

/**
 * Result from local or hybrid evaluation.
 * Tracks which metrics ran locally vs cloud and any errors.
 *
 * @example
 * ```typescript
 * const result: LocalEvaluationResult = {
 *   results: { eval_results: [...] },
 *   executedLocally: ['contains', 'is_json'],
 *   executedCloud: ['groundedness'],
 *   errors: []
 * };
 * ```
 */
interface LocalEvaluationResult {
  /** Evaluation results matching BatchRunResult format */
  results: BatchRunResult;
  /** Names of metrics that executed locally */
  executedLocally: string[];
  /** Names of metrics that executed via cloud API */
  executedCloud: string[];
  /** Any errors encountered during evaluation */
  errors: Array<{ metric: string; error: string }>;
}

/**
 * Configuration for local LLM (Ollama).
 * Used to configure the OllamaLLM client for local inference.
 *
 * @example
 * ```typescript
 * const config: LocalLLMConfig = {
 *   model: 'llama3.2',
 *   baseUrl: 'http://localhost:11434',
 *   temperature: 0.0,
 *   maxTokens: 1024,
 *   timeout: 120
 * };
 * ```
 */
interface LocalLLMConfig {
  /** Model name/identifier (e.g., 'llama3.2', 'mistral', 'phi3') */
  model: string;
  /** Ollama server URL (default: 'http://localhost:11434') */
  baseUrl: string;
  /** Sampling temperature (0.0 = deterministic, higher = more random) */
  temperature: number;
  /** Maximum tokens in generated response */
  maxTokens: number;
  /** Request timeout in seconds */
  timeout: number;
}

/**
 * Result from LLM-as-judge evaluation.
 * Returned by OllamaLLM.judge() for local LLM-based evaluations.
 *
 * @example
 * ```typescript
 * const result: JudgeResult = {
 *   score: 0.85,
 *   passed: true,
 *   reason: 'Response is factually accurate and well-structured',
 *   raw_response: '{"score": 0.85, ...}'
 * };
 * ```
 */
interface JudgeResult {
  /** Evaluation score from 0.0 to 1.0 */
  score: number;
  /** Whether the evaluation passed (typically score >= 0.5) */
  passed: boolean;
  /** Human-readable explanation of the judgment */
  reason: string;
  /** Raw LLM response before parsing (for debugging) */
  raw_response?: string;
}

export {
    ConfigParam,
    ConfigPossibleValues,
    DatapointFieldAnnotation,
    EvalResultMetric,
    EvalResult,
    BatchRunResult,
    RequiredKeys,
    EvalTags,
    Comparator,
    OutputType,
    ExecutionMode,
    EvaluatorConfig,
    ExplanationDetail,
    EvaluateOptions,
    PipelineEvalData,
    PipelineResult,
    LocalEvaluationResult,
    LocalLLMConfig,
    JudgeResult
}