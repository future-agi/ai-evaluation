/**
 * AutoEval Pipeline
 *
 * Automatically analyzes content and selects appropriate metrics for evaluation.
 * Provides a simple interface for evaluating AI responses without manual metric selection.
 *
 * @module local/autoeval
 *
 * @example
 * ```typescript
 * import { autoEvaluate, AutoEvalPipeline } from '@anthropic/ai-evaluation/local/autoeval';
 *
 * // Quick evaluation
 * const result = autoEvaluate({
 *     query: 'What is Python?',
 *     response: 'Python is a programming language.',
 *     context: 'Python is a high-level programming language created by Guido.'
 * });
 *
 * console.log(result.overallScore);
 * console.log(result.summary);
 *
 * // Pipeline with configuration
 * const pipeline = new AutoEvalPipeline({
 *     enableSecurityChecks: true,
 *     enableHallucinationCheck: true,
 *     passThreshold: 0.8
 * });
 *
 * const result2 = pipeline.evaluate({
 *     response: 'const key = "sk-secret123";',
 *     code: 'const key = "sk-secret123";'
 * });
 * ```
 */

// Types
export {
    InputType,
    ContentCharacteristics,
    AutoEvalInput,
    AutoEvalConfig,
    MetricCategory,
    SelectedMetric,
    MetricExecutionResult,
    AutoEvalResult,
    CODE_LANGUAGE_PATTERNS
} from './types';

// Analyzer
export {
    analyzeContent,
    describeCharacteristics
} from './analyzer';

// Selector
export {
    selectMetrics,
    getRecommendedMetrics,
    buildMetricConfig
} from './selector';

// Pipeline
export {
    AutoEvalPipeline,
    createAutoEval,
    autoEvaluate
} from './pipeline';
