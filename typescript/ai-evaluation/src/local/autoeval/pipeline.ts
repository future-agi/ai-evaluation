/**
 * AutoEval Pipeline.
 * Automatically analyzes content and runs appropriate metrics.
 *
 * @module local/autoeval/pipeline
 */

import {
    AutoEvalInput,
    AutoEvalConfig,
    AutoEvalResult,
    SelectedMetric,
    MetricExecutionResult,
    ContentCharacteristics
} from './types';
import { analyzeContent, describeCharacteristics } from './analyzer';
import { selectMetrics, buildMetricConfig } from './selector';
import { runMetric, hasMetric } from '../metrics';

/**
 * AutoEval Pipeline - automatically selects and runs appropriate metrics.
 *
 * @example
 * ```typescript
 * const pipeline = new AutoEvalPipeline();
 *
 * // Simple text evaluation
 * const result = pipeline.evaluate({
 *     response: 'Paris is the capital of France.'
 * });
 *
 * // RAG evaluation
 * const ragResult = pipeline.evaluate({
 *     query: 'What is the capital of France?',
 *     response: 'Paris is the capital of France.',
 *     context: 'Paris is the capital and largest city of France.'
 * });
 *
 * // Code security evaluation
 * const codeResult = pipeline.evaluate({
 *     response: 'const query = "SELECT * FROM users WHERE id = " + userId;',
 *     code: 'const query = "SELECT * FROM users WHERE id = " + userId;'
 * });
 * ```
 */
export class AutoEvalPipeline {
    private config: AutoEvalConfig;

    constructor(config: AutoEvalConfig = {}) {
        this.config = {
            minConfidence: 0.5,
            maxMetrics: 10,
            enableSecurityChecks: true,
            enableHallucinationCheck: true,
            passThreshold: 0.7,
            ...config
        };
    }

    /**
     * Evaluate input using automatically selected metrics.
     *
     * @param input - The input to evaluate
     * @param configOverrides - Optional config overrides for this evaluation
     * @returns Evaluation result with all metric scores
     */
    evaluate(
        input: AutoEvalInput,
        configOverrides?: Partial<AutoEvalConfig>
    ): AutoEvalResult {
        const startTime = Date.now();
        const config = { ...this.config, ...configOverrides };

        // Step 1: Analyze content
        const characteristics = analyzeContent(input);

        // Step 2: Select metrics
        const selectedMetrics = selectMetrics(characteristics, input, config);

        // Step 3: Execute metrics
        const metricResults = this.executeMetrics(selectedMetrics, input);

        // Step 4: Calculate overall score
        const overallScore = this.calculateOverallScore(metricResults, config);

        // Step 5: Generate summary and recommendations
        const summary = this.generateSummary(characteristics, metricResults, overallScore);
        const recommendations = this.generateRecommendations(metricResults, characteristics);

        const totalExecutionTimeMs = Date.now() - startTime;

        return {
            overallScore,
            passed: overallScore >= (config.passThreshold ?? 0.7),
            characteristics,
            selectedMetrics,
            metricResults,
            summary,
            recommendations,
            totalExecutionTimeMs
        };
    }

    /**
     * Execute selected metrics
     */
    private executeMetrics(
        selectedMetrics: SelectedMetric[],
        input: AutoEvalInput
    ): MetricExecutionResult[] {
        const results: MetricExecutionResult[] = [];

        for (const metric of selectedMetrics) {
            const startTime = Date.now();

            try {
                // Build config for this metric
                const metricConfig = buildMetricConfig(metric, input);

                // Check if metric exists
                if (!hasMetric(metric.name)) {
                    results.push({
                        metric: metric.name,
                        category: metric.category,
                        result: {
                            score: 0,
                            passed: false,
                            reason: `Metric '${metric.name}' not found`
                        },
                        executionTimeMs: Date.now() - startTime,
                        error: `Metric '${metric.name}' not found`
                    });
                    continue;
                }

                // Run the metric
                const result = runMetric(metric.name, input.response, metricConfig);

                results.push({
                    metric: metric.name,
                    category: metric.category,
                    result,
                    executionTimeMs: Date.now() - startTime
                });
            } catch (error) {
                results.push({
                    metric: metric.name,
                    category: metric.category,
                    result: {
                        score: 0,
                        passed: false,
                        reason: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
                    },
                    executionTimeMs: Date.now() - startTime,
                    error: error instanceof Error ? error.message : 'Unknown error'
                });
            }
        }

        return results;
    }

    /**
     * Calculate overall weighted score
     */
    private calculateOverallScore(
        results: MetricExecutionResult[],
        config: AutoEvalConfig
    ): number {
        if (results.length === 0) {
            return 1.0; // No metrics to run = pass
        }

        const weights = config.metricWeights || {};
        let totalWeight = 0;
        let weightedSum = 0;

        // Category weights (default)
        const categoryWeights: Record<string, number> = {
            'security': 1.5,      // Security issues are critical
            'hallucination': 1.3, // Hallucination is important
            'rag': 1.2,           // RAG quality matters
            'similarity': 1.0,    // Standard weight
            'json': 1.0,
            'string': 0.8,
            'quality': 0.8
        };

        for (const result of results) {
            // Skip errored metrics
            if (result.error) {
                continue;
            }

            // Get weight for this metric
            let weight = weights[result.metric];
            if (weight === undefined) {
                weight = categoryWeights[result.category] ?? 1.0;
            }

            totalWeight += weight;
            weightedSum += result.result.score * weight;
        }

        if (totalWeight === 0) {
            return 1.0;
        }

        return weightedSum / totalWeight;
    }

    /**
     * Generate a human-readable summary
     */
    private generateSummary(
        characteristics: ContentCharacteristics,
        results: MetricExecutionResult[],
        overallScore: number
    ): string {
        const parts: string[] = [];

        // Content description
        parts.push(`Evaluated ${characteristics.inputType} content (${characteristics.lengthCategory} length).`);

        // Metric summary
        const successCount = results.filter(r => r.result.passed && !r.error).length;
        const failCount = results.filter(r => !r.result.passed && !r.error).length;
        const errorCount = results.filter(r => r.error).length;

        parts.push(`Ran ${results.length} metrics: ${successCount} passed, ${failCount} failed${errorCount > 0 ? `, ${errorCount} errors` : ''}.`);

        // Overall assessment
        if (overallScore >= 0.9) {
            parts.push('Excellent overall quality.');
        } else if (overallScore >= 0.7) {
            parts.push('Good overall quality with minor issues.');
        } else if (overallScore >= 0.5) {
            parts.push('Moderate quality with notable issues.');
        } else {
            parts.push('Significant quality issues detected.');
        }

        return parts.join(' ');
    }

    /**
     * Generate recommendations based on results
     */
    private generateRecommendations(
        results: MetricExecutionResult[],
        characteristics: ContentCharacteristics
    ): string[] {
        const recommendations: string[] = [];

        // Check for failed metrics and generate recommendations
        for (const result of results) {
            if (result.error) {
                continue;
            }

            if (!result.result.passed) {
                switch (result.category) {
                    case 'security':
                        if (result.metric === 'sql_injection') {
                            recommendations.push('Use parameterized queries instead of string concatenation for SQL.');
                        } else if (result.metric === 'xss_detection') {
                            recommendations.push('Sanitize user input before inserting into HTML. Use textContent instead of innerHTML.');
                        } else if (result.metric === 'secrets_detection') {
                            recommendations.push('Move hardcoded secrets to environment variables or a secrets manager.');
                        }
                        break;

                    case 'hallucination':
                        recommendations.push('Review claims for accuracy and ensure they are supported by the provided context.');
                        break;

                    case 'rag':
                        if (result.metric === 'faithfulness') {
                            recommendations.push('Ensure the response stays grounded in the provided context.');
                        } else if (result.metric === 'answer_relevance') {
                            recommendations.push('Make the response more directly address the query.');
                        } else if (result.metric === 'context_utilization') {
                            recommendations.push('Better utilize the available context in the response.');
                        }
                        break;

                    case 'json':
                        recommendations.push('Fix JSON syntax errors to produce valid JSON output.');
                        break;
                }
            }
        }

        // Add general recommendations based on characteristics
        if (characteristics.hasCode && !results.some(r => r.category === 'security')) {
            recommendations.push('Consider running security checks on code output.');
        }

        if (characteristics.hasContext && !results.some(r => r.category === 'hallucination')) {
            recommendations.push('Consider checking for hallucinations when context is provided.');
        }

        // Deduplicate
        return [...new Set(recommendations)];
    }

    /**
     * Get the characteristics of an input without running evaluation
     */
    analyzeOnly(input: AutoEvalInput): {
        characteristics: ContentCharacteristics;
        description: string;
        recommendedMetrics: SelectedMetric[];
    } {
        const characteristics = analyzeContent(input);
        const description = describeCharacteristics(characteristics);
        const recommendedMetrics = selectMetrics(characteristics, input, this.config);

        return {
            characteristics,
            description,
            recommendedMetrics
        };
    }

    /**
     * Update pipeline configuration
     */
    configure(config: Partial<AutoEvalConfig>): void {
        this.config = { ...this.config, ...config };
    }

    /**
     * Get current configuration
     */
    getConfig(): AutoEvalConfig {
        return { ...this.config };
    }
}

/**
 * Create an AutoEval pipeline with default configuration
 */
export function createAutoEval(config?: AutoEvalConfig): AutoEvalPipeline {
    return new AutoEvalPipeline(config);
}

/**
 * Quick evaluation function without creating a pipeline instance
 */
export function autoEvaluate(
    input: AutoEvalInput,
    config?: AutoEvalConfig
): AutoEvalResult {
    const pipeline = new AutoEvalPipeline(config);
    return pipeline.evaluate(input);
}
