/**
 * Local and hybrid evaluators for running evaluations without API calls.
 * Aligned with Python SDK's fi.evals.local.evaluator module.
 */

import { BatchRunResult, EvalResult } from '../types';
import { Evaluator } from '../evaluator';
import { ExecutionMode, canRunLocally, requiresLLM, selectExecutionMode } from './execution-mode';
import { runMetric, hasMetric, MetricResult } from './metrics';
import { OllamaLLM, JudgeResult } from './llm';

/**
 * Configuration for LocalEvaluator
 */
export interface LocalEvaluatorConfig {
    strictMode?: boolean;  // Throw on unknown metrics
}

/**
 * Result from local evaluation
 */
export interface LocalEvaluationResult {
    results: BatchRunResult;
    executedLocally: string[];
    executedCloud: string[];
    errors: Array<{ metric: string; error: string }>;
}

/**
 * Evaluation request format
 */
export interface EvaluationRequest {
    metricName: string;
    inputs: Array<Record<string, any>>;
    config?: Record<string, any>;
}

/**
 * LocalEvaluator - Runs heuristic metrics locally without API calls.
 */
export class LocalEvaluator {
    private config: LocalEvaluatorConfig;

    constructor(config: LocalEvaluatorConfig = {}) {
        this.config = {
            strictMode: false,
            ...config
        };
    }

    /**
     * Check if a metric can run locally
     */
    canRunLocally(metricName: string): boolean {
        return canRunLocally(metricName) || hasMetric(metricName);
    }

    /**
     * Evaluate a single metric
     */
    evaluate(
        metricName: string,
        inputs: Array<Record<string, any>>,
        config: Record<string, any> = {}
    ): LocalEvaluationResult {
        const evalResults: EvalResult[] = [];
        const errors: Array<{ metric: string; error: string }> = [];

        if (!this.canRunLocally(metricName)) {
            if (this.config.strictMode) {
                throw new Error(`Metric '${metricName}' cannot run locally`);
            }
            errors.push({
                metric: metricName,
                error: `Metric '${metricName}' cannot run locally`
            });
            return {
                results: { eval_results: [] },
                executedLocally: [],
                executedCloud: [],
                errors
            };
        }

        for (const input of inputs) {
            const response = input.response || input.output || input.text || '';

            try {
                const result = runMetric(metricName, response, { ...config, ...input });
                evalResults.push({
                    name: metricName,
                    output: result.score,
                    reason: result.reason,
                    runtime: 0,
                    output_type: 'score'
                });
            } catch (error) {
                errors.push({
                    metric: metricName,
                    error: (error as Error).message
                });
                evalResults.push({
                    name: metricName,
                    output: null,
                    reason: `Error: ${(error as Error).message}`,
                    runtime: 0
                });
            }
        }

        return {
            results: { eval_results: evalResults },
            executedLocally: [metricName],
            executedCloud: [],
            errors
        };
    }

    /**
     * Evaluate multiple metrics in batch
     */
    evaluateBatch(evaluations: EvaluationRequest[]): LocalEvaluationResult {
        const allResults: (EvalResult | null)[] = [];
        const executedLocally: string[] = [];
        const errors: Array<{ metric: string; error: string }> = [];

        for (const evaluation of evaluations) {
            const result = this.evaluate(
                evaluation.metricName,
                evaluation.inputs,
                evaluation.config || {}
            );

            allResults.push(...result.results.eval_results);
            executedLocally.push(...result.executedLocally);
            errors.push(...result.errors);
        }

        return {
            results: { eval_results: allResults },
            executedLocally: [...new Set(executedLocally)],
            executedCloud: [],
            errors
        };
    }
}

/**
 * HybridEvaluator - Routes evaluations between local and cloud execution.
 */
export class HybridEvaluator {
    private localEvaluator: LocalEvaluator;
    private cloudEvaluator?: Evaluator;
    private localLLM?: OllamaLLM;
    private preferLocal: boolean;
    private fallbackToCloud: boolean;
    private offlineMode: boolean;

    constructor(options: {
        localLLM?: OllamaLLM;
        cloudEvaluator?: Evaluator;
        preferLocal?: boolean;
        fallbackToCloud?: boolean;
        offlineMode?: boolean;
    } = {}) {
        this.localEvaluator = new LocalEvaluator();
        this.localLLM = options.localLLM;
        this.cloudEvaluator = options.cloudEvaluator;
        this.preferLocal = options.preferLocal ?? true;
        this.fallbackToCloud = options.fallbackToCloud ?? true;
        this.offlineMode = options.offlineMode ?? false;
    }

    /**
     * Check if we can use local LLM for a metric
     */
    canUseLocalLLM(metricName: string): boolean {
        return requiresLLM(metricName) && this.localLLM !== undefined;
    }

    /**
     * Determine execution mode for a metric
     */
    routeEvaluation(
        metricName: string,
        forceLocal?: boolean,
        forceCloud?: boolean
    ): ExecutionMode {
        if (forceLocal) return ExecutionMode.LOCAL;
        if (forceCloud && !this.offlineMode) return ExecutionMode.CLOUD;

        // Check if it can run locally (heuristic)
        if (canRunLocally(metricName)) {
            return ExecutionMode.LOCAL;
        }

        // Check if we have local LLM for LLM-based metrics
        if (this.canUseLocalLLM(metricName) && this.preferLocal) {
            return ExecutionMode.LOCAL;
        }

        // Fall back to cloud if available
        if (this.cloudEvaluator && this.fallbackToCloud && !this.offlineMode) {
            return ExecutionMode.CLOUD;
        }

        // No cloud available, try local anyway
        return ExecutionMode.LOCAL;
    }

    /**
     * Evaluate a single template
     */
    async evaluate(
        template: string,
        inputs: Array<Record<string, any>>,
        config: Record<string, any> = {}
    ): Promise<LocalEvaluationResult> {
        const mode = this.routeEvaluation(template);

        if (mode === ExecutionMode.LOCAL) {
            // Try heuristic metrics first
            if (canRunLocally(template)) {
                return this.localEvaluator.evaluate(template, inputs, config);
            }

            // Try local LLM for LLM-based metrics
            if (this.localLLM && requiresLLM(template)) {
                return this.evaluateWithLocalLLM(template, inputs, config);
            }

            // Can't run locally
            return {
                results: { eval_results: inputs.map(() => ({
                    name: template,
                    output: null,
                    reason: 'Cannot run locally and no cloud available',
                    runtime: 0
                }))},
                executedLocally: [],
                executedCloud: [],
                errors: [{ metric: template, error: 'Cannot run locally' }]
            };
        }

        // Cloud execution
        if (this.cloudEvaluator) {
            return this.evaluateCloud(template, inputs, config);
        }

        throw new Error('No cloud evaluator configured');
    }

    /**
     * Evaluate using local LLM
     */
    private async evaluateWithLocalLLM(
        template: string,
        inputs: Array<Record<string, any>>,
        config: Record<string, any>
    ): Promise<LocalEvaluationResult> {
        if (!this.localLLM) {
            throw new Error('No local LLM configured');
        }

        const evalResults: EvalResult[] = [];
        const errors: Array<{ metric: string; error: string }> = [];

        // Build criteria based on template
        const criteria = this.getCriteriaForTemplate(template);

        for (const input of inputs) {
            const query = input.query || input.input || input.prompt || '';
            const response = input.response || input.output || input.text || '';
            const context = input.context || undefined;

            try {
                const result = await this.localLLM.judge(query, response, criteria, context);
                evalResults.push({
                    name: template,
                    output: result.score,
                    reason: result.reason,
                    runtime: 0,
                    output_type: 'score'
                });
            } catch (error) {
                errors.push({
                    metric: template,
                    error: (error as Error).message
                });
                evalResults.push({
                    name: template,
                    output: null,
                    reason: `Error: ${(error as Error).message}`,
                    runtime: 0
                });
            }
        }

        return {
            results: { eval_results: evalResults },
            executedLocally: [template],
            executedCloud: [],
            errors
        };
    }

    /**
     * Evaluate using cloud API
     */
    private async evaluateCloud(
        template: string,
        inputs: Array<Record<string, any>>,
        config: Record<string, any>
    ): Promise<LocalEvaluationResult> {
        if (!this.cloudEvaluator) {
            throw new Error('No cloud evaluator configured');
        }

        // Transform inputs to the format expected by cloud API
        const transformedInputs: Record<string, string[]> = {};

        for (const input of inputs) {
            for (const [key, value] of Object.entries(input)) {
                if (!transformedInputs[key]) {
                    transformedInputs[key] = [];
                }
                transformedInputs[key].push(String(value));
            }
        }

        try {
            const result = await this.cloudEvaluator.evaluate(
                template,
                transformedInputs,
                { modelName: config.model || 'gpt-4o' }
            );

            return {
                results: result,
                executedLocally: [],
                executedCloud: [template],
                errors: []
            };
        } catch (error) {
            return {
                results: { eval_results: [] },
                executedLocally: [],
                executedCloud: [],
                errors: [{ metric: template, error: (error as Error).message }]
            };
        }
    }

    /**
     * Partition evaluations by execution mode
     */
    partitionEvaluations(
        evaluations: EvaluationRequest[]
    ): Record<ExecutionMode, EvaluationRequest[]> {
        const partitions: Record<ExecutionMode, EvaluationRequest[]> = {
            [ExecutionMode.LOCAL]: [],
            [ExecutionMode.CLOUD]: [],
            [ExecutionMode.HYBRID]: []
        };

        for (const evaluation of evaluations) {
            const mode = this.routeEvaluation(evaluation.metricName);
            partitions[mode].push(evaluation);
        }

        return partitions;
    }

    /**
     * Evaluate local partition
     */
    evaluateLocalPartition(
        evaluations: EvaluationRequest[]
    ): LocalEvaluationResult {
        return this.localEvaluator.evaluateBatch(evaluations);
    }

    /**
     * Get evaluation criteria for a template
     */
    private getCriteriaForTemplate(template: string): string {
        const criteriaMap: Record<string, string> = {
            groundedness: 'Evaluate if the response is grounded in the provided context. Score higher if all claims are supported by the context.',
            hallucination: 'Evaluate if the response contains hallucinated (made-up) information not present in the context. Score higher if there are no hallucinations.',
            relevance: 'Evaluate if the response is relevant to the query. Score higher if the response directly addresses the question.',
            coherence: 'Evaluate if the response is coherent and well-structured. Score higher if the response flows logically.',
            fluency: 'Evaluate the fluency and grammatical correctness of the response. Score higher for natural, error-free text.',
            factual_accuracy: 'Evaluate the factual accuracy of the response. Score higher if all facts are correct.',
            context_adherence: 'Evaluate if the response adheres to the provided context without adding external information.',
            context_relevance: 'Evaluate if the retrieved context is relevant to answering the query.',
            completeness: 'Evaluate if the response completely answers the query without missing important information.',
            toxicity: 'Evaluate if the response contains toxic, harmful, or offensive content. Score higher if the response is safe.',
            bias_detection: 'Evaluate if the response contains biased statements. Score higher if the response is neutral and fair.',
            summary_quality: 'Evaluate the quality of the summary. Score higher if it captures key points concisely.',
            tone: 'Evaluate the tone of the response. Score higher if the tone is appropriate for the context.',
            prompt_adherence: 'Evaluate if the response follows the instructions in the prompt.'
        };

        return criteriaMap[template.toLowerCase()] ||
            `Evaluate the quality of the response based on: ${template}`;
    }
}
