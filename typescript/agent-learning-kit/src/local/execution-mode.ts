/**
 * Execution mode definitions and utilities for local/cloud routing.
 * Aligned with Python SDK's fi.evals.local.execution_mode module.
 */

/**
 * Execution mode for evaluation routing
 */
export enum ExecutionMode {
    LOCAL = "local",
    CLOUD = "cloud",
    HYBRID = "hybrid"
}

/**
 * Metrics that can run locally without API calls.
 * These are heuristic metrics that don't require LLM inference.
 */
export const LOCAL_CAPABLE_METRICS: Set<string> = new Set([
    // String metrics
    "regex",
    "contains",
    "contains_all",
    "contains_any",
    "contains_none",
    "one_line",
    "equals",
    "starts_with",
    "ends_with",
    "length_less_than",
    "length_greater_than",
    "length_between",

    // JSON metrics
    "contains_json",
    "is_json",
    "json_schema",

    // Similarity metrics
    "bleu_score",
    "rouge_score",
    "recall_score",
    "levenshtein_similarity",
    "numeric_similarity",
    "embedding_similarity",
    "semantic_list_contains",

    // RAG metrics (heuristic implementations)
    "context_precision",
    "context_recall",
    "faithfulness",
    "groundedness",
    "answer_relevance",
    "context_relevance",
    "context_utilization",

    // Code security metrics (static analysis)
    "sql_injection",
    "no_sql_injection",
    "xss_detection",
    "no_xss",
    "secrets_detection",
    "no_secrets",
    "no_hardcoded_secrets",
    "code_security_scan",
    "all_security_checks",

    // Hallucination detection (heuristic implementation)
    "hallucination_detection",
    "detect_hallucination",
    "no_hallucination",
]);

/**
 * Metrics that require LLM inference (can use local LLM if available)
 */
export const LLM_BASED_METRICS: Set<string> = new Set([
    "groundedness",
    "hallucination",
    "relevance",
    "coherence",
    "fluency",
    "factual_accuracy",
    "context_adherence",
    "context_relevance",
    "completeness",
    "toxicity",
    "bias_detection",
    "summary_quality",
    "tone",
    "prompt_adherence",
]);

/**
 * Check if a metric can run locally (heuristic only)
 * @param metricName - The name of the metric
 * @returns true if the metric can run locally without any LLM
 */
export function canRunLocally(metricName: string): boolean {
    return LOCAL_CAPABLE_METRICS.has(metricName.toLowerCase());
}

/**
 * Check if a metric requires LLM inference
 * @param metricName - The name of the metric
 * @returns true if the metric requires LLM inference
 */
export function requiresLLM(metricName: string): boolean {
    return LLM_BASED_METRICS.has(metricName.toLowerCase());
}

/**
 * Select the appropriate execution mode for a metric
 * @param metricName - The name of the metric
 * @param hasLocalLLM - Whether a local LLM is available
 * @param preferLocal - Whether to prefer local execution
 * @returns The recommended execution mode
 */
export function selectExecutionMode(
    metricName: string,
    hasLocalLLM: boolean = false,
    preferLocal: boolean = true
): ExecutionMode {
    const lowerName = metricName.toLowerCase();

    // Heuristic metrics always run locally
    if (LOCAL_CAPABLE_METRICS.has(lowerName)) {
        return ExecutionMode.LOCAL;
    }

    // LLM-based metrics can run locally if we have a local LLM
    if (LLM_BASED_METRICS.has(lowerName) && hasLocalLLM && preferLocal) {
        return ExecutionMode.LOCAL;
    }

    // Everything else goes to cloud
    return ExecutionMode.CLOUD;
}
