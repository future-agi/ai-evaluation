import {
    APIKeyAuth,
    ResponseHandler,
    HttpMethod,
    RequestConfig,
    Routes,
    InvalidAuthError,
    SDKException,
    InvalidValueType,
    MissingRequiredKey,
    AUTH_ENVVAR_NAME,
} from './core';
import { mapInputsToBackend } from './core/cloudRegistry';


function coerceToApiInput(value: unknown): unknown {
    if (value === null || value === undefined) return value;
    if (typeof value === "string") return [value];
    if (Array.isArray(value)) {
        if (value.every(v => typeof v === "string")) return value;
        if (value.every(v => Array.isArray(v) && v.every(x => typeof x === "string"))) return value;
        return JSON.stringify(value);
    }
    if (typeof value === "object") return JSON.stringify(value);
    return value;
}
import { AxiosResponse } from 'axios';

import { Execution, normalizeStatus } from './execution';
import { EvalTemplate } from './templates';
import {
    BatchRunResult,
    EvalResult,
    EvalResultMetric,
    EvaluatorConfig,
    EvaluateOptions,
    PipelineEvalData,
    PipelineResult
} from './types';


/**
 * Handles responses for evaluation requests
 */
export class EvalResponseHandler extends ResponseHandler<BatchRunResult, any> {
    public static _parseSuccess(response: AxiosResponse): BatchRunResult {
        // Revamped backend (post 2026-04-12) returns pure snake_case:
        //   { result: [{ evaluations: [{ name, reason, runtime, output,
        //     output_type, eval_id, model?, error_localizer_enabled?,
        //     error_localizer? }] }] }
        // Async / single-row paths sometimes return the eval wrapped in
        // { eval_status, result: <eval> } — handle both.
        const data = response.data || {};
        const evalResults: (EvalResult | null)[] = [];

        const topResult = data.result;
        const outerList = Array.isArray(topResult)
            ? topResult
            : topResult && typeof topResult === "object"
                ? [topResult]
                : [];

        for (const item of outerList) {
            const evaluations = Array.isArray(item?.evaluations)
                ? item.evaluations
                : item && typeof item === "object"
                    ? [item]
                    : [];

            for (const evaluation of evaluations) {
                if (!evaluation || typeof evaluation !== "object") continue;

                // Legacy metadata passthrough (usage / cost / explanation)
                // — still returned by some v1 paths and consumed by
                // protect.ts.
                let legacyMetadata: Record<string, any> | undefined;
                if (evaluation.metadata) {
                    let metadata: any = evaluation.metadata;
                    if (typeof metadata === "string") {
                        try { metadata = JSON.parse(metadata); } catch { /* ignore */ }
                    }
                    if (metadata && typeof metadata === "object") {
                        legacyMetadata = {
                            usage: metadata.usage ?? {},
                            cost: metadata.cost ?? {},
                            explanation: metadata.explanation ?? {},
                        };
                    }
                }

                evalResults.push({
                    name: evaluation.name ?? "",
                    output: evaluation.output ?? evaluation.value,
                    reason: evaluation.reason ?? "",
                    runtime: evaluation.runtime ?? 0,
                    output_type: evaluation.output_type,
                    eval_id: evaluation.eval_id,
                    model: evaluation.model,
                    error_localizer_enabled: evaluation.error_localizer_enabled,
                    error_localizer: evaluation.error_localizer ?? null,
                    // Legacy v1 fields (optional, used by Protect)
                    data: evaluation.data,
                    failure: evaluation.failure,
                    metadata: legacyMetadata,
                    metrics: Array.isArray(evaluation.metrics)
                        ? evaluation.metrics.map((m: any): EvalResultMetric => ({
                              id: m.id,
                              value: m.value,
                          }))
                        : undefined,
                });
            }
        }

        return { eval_results: evalResults } as BatchRunResult;
    }

    public static _handleError(response: AxiosResponse): never {
        if (response.status === 400) {
            throw new Error(
                `Evaluation failed with a 400 Bad Request. Please check your input data and evaluation configuration. Response: ${response.data}`
            );
        } else if (response.status === 403) {
            throw new InvalidAuthError();
        } else {
            throw new Error(
                `Error in evaluation: ${response.status}, response: ${response.data}`
            );
        }
    }
}

/**
 * Handles responses for evaluation info requests
 */
export class EvalInfoResponseHandler extends ResponseHandler<Record<string, any>, any> {
    public static _parseSuccess(response: AxiosResponse): Record<string, any> {
        const data = response.data;
        if (data.result) {
            return data.result;
        } else {
            throw new Error(`Failed to get evaluation info: ${data}`);
        }
    }

    public static _handleError(response: AxiosResponse): never {
        if (response.status === 400) {
            // In TypeScript with axios, it's more common to let the caller handle response.data
            throw new Error(`Bad request: ${response.data}`);
        }
        if (response.status === 403) {
            throw new InvalidAuthError();
        }
        throw new Error(`Failed to get evaluation info: ${response.status}`);
    }
}

/**
 * Client for evaluating LLM test cases
 * Aligned with Python SDK API
 */
export class Evaluator extends APIKeyAuth {
    private readonly maxWorkers: number;
    private evalInfoCache = new Map<string, Record<string, any>>();

    // Platform credentials (Langfuse)
    private readonly langfuseSecretKey?: string;
    private readonly langfusePublicKey?: string;
    private readonly langfuseHost?: string;

    constructor(options: EvaluatorConfig = {}) {
        const fiApiKey = process.env.FI_API_KEY || options.fiApiKey;
        const fiSecretKey = process.env.FI_SECRET_KEY || options.fiSecretKey;
        const fiBaseUrl = process.env.FI_BASE_URL || options.fiBaseUrl;

        super({ ...options, fiApiKey, fiSecretKey, fiBaseUrl });
        this.maxWorkers = options.maxWorkers || 8;

        // Platform credentials
        this.langfuseSecretKey = options.langfuseSecretKey || process.env.LANGFUSE_SECRET_KEY;
        this.langfusePublicKey = options.langfusePublicKey || process.env.LANGFUSE_PUBLIC_KEY;
        this.langfuseHost = options.langfuseHost || process.env.LANGFUSE_HOST;
    }

    public async evaluate(
        evalTemplates: string | EvalTemplate | (string | EvalTemplate)[],
        inputs: Record<string, string | string[]>,
        options: EvaluateOptions = {}
    ): Promise<BatchRunResult> {

        const {
            timeout,
            modelName,
            customEvalName,
            traceEval: initialTraceEval = false,
            platform,
            isAsync = false,
            errorLocalizer = false,
            evalConfig
        } = options;

        // Handle platform configuration (e.g., Langfuse)
        if (platform) {
            if (typeof evalTemplates === 'string' && typeof inputs === 'object' && customEvalName) {
                return this._configureEvaluations(
                    evalTemplates,
                    inputs,
                    platform,
                    customEvalName,
                    modelName
                );
            } else {
                throw new Error("Invalid arguments for platform configuration");
            }
        }

        let traceEval = initialTraceEval;
        let spanId: string | undefined = undefined;

        const extractName = (t: string | EvalTemplate): string | undefined => {
            if (typeof t === 'string') {
                return t;
            }
            if (typeof t === 'object' && t.eval_name) {
                return t.eval_name;
            }
            return undefined;
        };

        const firstTemplate = Array.isArray(evalTemplates) ? evalTemplates[0] : evalTemplates;
        const evalName = extractName(firstTemplate);

        if (!evalName) {
            throw new TypeError("Unsupported eval_templates argument. Expect eval template class/obj or name str.");
        }

        // OpenTelemetry logic
        if (traceEval) {
            if (!customEvalName) {
                traceEval = false;
                console.warn("Failed to trace the evaluation. Please set the customEvalName.");
            } else {
                try {
                    // Dynamically import to avoid making OTEL a hard dependency
                    const otel = await import('@opentelemetry/api');
                    
                    const { checkCustomEvalConfigExists } = await import('@traceai/fi-core');

                    
                    const currentSpan = otel.trace.getSpan(otel.context.active());
                    if (currentSpan && currentSpan.isRecording()) {
                        const spanContext = currentSpan.spanContext();
                        if (otel.isSpanContextValid(spanContext)) {
                            spanId = spanContext.spanId;
                            
                            // Accessing the resource is not part of the public API interface,
                            // but is available on SDK implementations. This mirrors the Python SDK's approach.
                            const tracerProvider = otel.trace.getTracerProvider();
                            // @ts-ignore
                            const resource = tracerProvider.resource || (currentSpan && (currentSpan).resource);
                            
                            let projectName = resource?.attributes['project_name'] as string | undefined;
                            if (!projectName) {
                                // Fallback to standard OTEL service.name if custom attribute is absent
                                projectName = resource?.attributes['service.name'] as string | undefined;
                            }

                             
                            if (projectName) {
                                const evalTags = [{
                                    custom_eval_name: customEvalName,
                                    eval_name: evalName,
                                    mapping: {},
                                    config: {},
                                }];
                                const customEvalExists = await checkCustomEvalConfigExists(projectName, evalTags);

                                if (customEvalExists) {
                                    traceEval = false;
                                    console.warn("Failed to trace the evaluation. Custom eval configuration with the same name already exists for this project");
                                }
                            } else {
                                traceEval = false;
                                console.warn(
                                    "Could not determine project_name from OpenTelemetry context. " +
                                    "Skipping check for existing custom eval configuration."
                                );
                            }
                        }
                    }
                } catch (error) {
                    console.warn(
                        "OpenTelemetry API not found. Please install '@opentelemetry/api' to enable tracing. " +
                        "Skipping trace for this evaluation.",
                        error
                    );
                    traceEval = false;
                }
            }
        }

        if (Array.isArray(inputs)) {
            throw new TypeError("'inputs' must be a dictionary, array-of-dicts is not supported.");
        }

        // Dynamic registry: map user kwargs to exact required_keys before POST.
        // The api rejects supersets (e.g. {output,input,context} for a template
        // that only wants {output}), so this can't be a pass-through. On
        // registry miss, leaves inputs untouched — backend surfaces its own error.
        let mappedInputs: Record<string, any> = inputs;
        try {
            mappedInputs = await mapInputsToBackend(evalName, inputs, {
                baseUrl: this.baseUrl,
                apiKey: this.fiApiKey,
                secretKey: this.fiSecretKey,
            });
        } catch (err) {
            console.debug("Dynamic input mapping skipped:", err);
        }

        // The api accepts strings, list[str], or list[list[str]]. Serialize
        // dict / list-of-dicts / list-of-mixed values (e.g. conversation
        // messages) to JSON so users can pass native objects without
        // manually stringifying.
        const transformedApiInputs: Record<string, any> = {};
        for (const [key, value] of Object.entries(mappedInputs)) {
            transformedApiInputs[key] = coerceToApiInput(value);
        }

        const finalApiPayload: Record<string, any> = {
            eval_name: evalName,
            inputs: transformedApiInputs,
            model: modelName,
            span_id: spanId,
            custom_eval_name: customEvalName,
            trace_eval: traceEval,
            is_async: isAsync,
            error_localizer: errorLocalizer,
        };

        if (evalConfig) {
            finalApiPayload.config = { params: evalConfig };
        }

        const timeoutMs = timeout !== undefined ? timeout * 1000 : this.defaultTimeout * 1000;

        try {
            const response = await this.request(
                {
                    method: HttpMethod.POST,
                    url: `${this.baseUrl}/${Routes.evaluatev2}`,
                    json: finalApiPayload,
                    timeout: timeoutMs,
                },
                EvalResponseHandler
            ) as BatchRunResult;
            return response;
        } catch (error) {
            // Silent-empty fix: return a failed EvalResult with the api's error
            // text so callers can see *why* instead of getting an IndexError
            // downstream from an empty list.
            const message = error instanceof Error ? error.message : String(error);
            console.warn(`Evaluation failed for '${evalName}':`, message);
            return {
                eval_results: [
                    {
                        name: evalName,
                        output: null,
                        reason: message,
                        runtime: 0,
                    } as any,
                ],
            } as BatchRunResult;
        }
    }

    private async _get_eval_info(evalName: string): Promise<Record<string, any>> {
        if (this.evalInfoCache.has(evalName)) {
            return this.evalInfoCache.get(evalName)!;
        }
        
        const response = await this.request(
            {
                method: HttpMethod.GET,
                url: `${this.baseUrl}/${Routes.get_eval_templates}`,
            },
            EvalInfoResponseHandler
        ) as Record<string, any>[];

        const evalInfo = response.find(item => item.name === evalName);

        if (!evalInfo) {
            throw new Error(`Evaluation template with name '${evalName}' not found`);
        }
        
        this.evalInfoCache.set(evalName, evalInfo);
        return evalInfo;
    }

    public async list_evaluations(): Promise<Record<string, any>[]> {
        const config: RequestConfig = {
            method: HttpMethod.GET,
            url: `${this.baseUrl}/${Routes.get_eval_templates}`
        };
        const response = await this.request(config, EvalInfoResponseHandler) as Record<string, any>[];
        return response;
    }

    /**
     * Get the raw evaluation status payload by ID (unparsed).
     * For a higher-level handle that understands the status envelope and
     * can be awaited, use `getExecution` instead.
     */
    public async getEvalResult(evalId: string): Promise<Record<string, any>> {
        const config: RequestConfig = {
            method: HttpMethod.GET,
            url: `${this.baseUrl}/${Routes.get_eval_result}`,
            params: { eval_id: evalId },
            timeout: this.defaultTimeout * 1000,
        };
        const response = await this.request(config);
        return (response as any).data;
    }

    /**
     * Evaluate a pipeline
     */
    public async evaluatePipeline(
        projectName: string,
        version: string,
        evalData: PipelineEvalData[]
    ): Promise<Record<string, any>> {
        const apiPayload = {
            project_name: projectName,
            version: version,
            eval_data: evalData
        };

        const config: RequestConfig = {
            method: HttpMethod.POST,
            url: `${this.baseUrl}/${Routes.evaluate_pipeline}`,
            json: apiPayload,
            timeout: this.defaultTimeout * 1000,
        };

        const response = await this.request(config);
        return (response as any).data;
    }

    /**
     * Get pipeline evaluation results
     */
    public async getPipelineResults(
        projectName: string,
        versions: string[]
    ): Promise<Record<string, any>> {
        if (!Array.isArray(versions) || !versions.every(v => typeof v === 'string')) {
            throw new TypeError("versions must be an array of strings");
        }

        const config: RequestConfig = {
            method: HttpMethod.GET,
            url: `${this.baseUrl}/${Routes.evaluate_pipeline}`,
            params: {
                project_name: projectName,
                versions: versions.join(',')
            },
            timeout: this.defaultTimeout * 1000,
        };

        const response = await this.request(config);
        return (response as any).data;
    }

    /**
     * Configure evaluations on a specified platform (e.g., Langfuse)
     * @private
     */
    private async _configureEvaluations(
        evalTemplates: string,
        inputs: Record<string, any>,
        platform: string,
        customEvalName: string,
        modelName?: string
    ): Promise<BatchRunResult> {
        const kwargs: Record<string, any> = {};

        if (platform === "langfuse") {
            kwargs.langfuse_secret_key = this.langfuseSecretKey;
            kwargs.langfuse_public_key = this.langfusePublicKey;
            kwargs.langfuse_host = this.langfuseHost;
        }

        try {
            const otel = await import('@opentelemetry/api');
            const currentSpan = otel.trace.getSpan(otel.context.active());

            if (currentSpan && currentSpan.isRecording()) {
                const spanContext = currentSpan.spanContext();
                if (otel.isSpanContextValid(spanContext)) {
                    kwargs.span_id = spanContext.spanId;
                    kwargs.trace_id = spanContext.traceId;
                }
            }
        } catch {
            // OpenTelemetry not available
        }

        if (!kwargs.span_id || !kwargs.trace_id) {
            console.warn(
                "span_id and/or trace_id not found. " +
                "Please run this function within a span context."
            );
            return { eval_results: [] } as BatchRunResult;
        }

        const apiPayload = {
            eval_config: {
                eval_templates: evalTemplates,
                inputs: inputs,
                model_name: modelName
            },
            custom_eval_name: customEvalName,
            platform: platform,
            ...kwargs
        };

        const config: RequestConfig = {
            method: HttpMethod.POST,
            url: `${this.baseUrl}/${Routes.configure_evaluations}`,
            json: apiPayload,
            timeout: this.defaultTimeout * 1000,
        };

        try {
            const response = await this.request(config);
            return (response as any).data;
        } catch (error) {
            console.warn(`Error configuring evaluations: ${error}`);
            return { eval_results: [] } as BatchRunResult;
        }
    }

    // ------------------------------------------------------------------
    // Async submission / execution handles
    // ------------------------------------------------------------------

    /**
     * Submit an eval for async execution. Returns an `Execution` handle
     * immediately; call `handle.wait()` or `Evaluator.getExecution(id)`
     * to poll until completion.
     *
     * Internally forces `is_async=true` so the backend records the
     * evaluation and starts a worker without holding the HTTP connection.
     */
    public async submit(
        evalTemplates: string | EvalTemplate | (string | EvalTemplate)[],
        inputs: Record<string, string | string[]>,
        options: {
            timeout?: number;
            modelName?: string;
            customEvalName?: string;
            traceEval?: boolean;
            errorLocalizer?: boolean;
        } = {}
    ): Promise<Execution> {
        const extractName = (t: string | EvalTemplate): string | undefined => {
            if (typeof t === 'string') return t;
            if (typeof t === 'object' && t.eval_name) return t.eval_name;
            return undefined;
        };

        const firstTemplate = Array.isArray(evalTemplates)
            ? evalTemplates[0]
            : evalTemplates;
        const evalName = extractName(firstTemplate);
        if (!evalName) {
            throw new TypeError(
                "Unsupported evalTemplates argument. Expect template class/obj or name string."
            );
        }

        const transformedInputs: Record<string, string[]> = {};
        for (const [key, value] of Object.entries(inputs)) {
            transformedInputs[key] = Array.isArray(value) ? value : [value];
        }

        const payload = {
            eval_name: evalName,
            inputs: transformedInputs,
            model: options.modelName,
            span_id: undefined,
            custom_eval_name: options.customEvalName,
            trace_eval: options.traceEval ?? false,
            is_async: true,
            error_localizer: options.errorLocalizer ?? false,
        };

        const timeoutMs =
            options.timeout !== undefined
                ? options.timeout * 1000
                : (this.defaultTimeout ?? 30) * 1000;

        const raw = (await this.request({
            method: HttpMethod.POST,
            url: `${this.baseUrl}/${Routes.evaluatev2}`,
            json: payload,
            timeout: timeoutMs,
        })) as any;
        const body = raw?.data ?? raw;

        if (body?.status === false) {
            throw new Error(
                `Async submit rejected by backend: ${JSON.stringify(body.result)}`
            );
        }
        const results = body?.result;
        if (!Array.isArray(results) || results.length === 0) {
            throw new Error(
                `Async submit did not return a result list (response: ${JSON.stringify(body)})`
            );
        }
        const evaluations = results[0]?.evaluations ?? [];
        const firstEval = evaluations[0] ?? {};
        const executionId: string | undefined = firstEval.eval_id;
        if (!executionId) {
            throw new Error(
                `Async submit did not return an eval_id (response: ${JSON.stringify(body)})`
            );
        }

        const handle = new Execution({
            id: String(executionId),
            kind: 'eval',
            status: 'pending',
        });
        handle._setRefresher(() => this._refreshEvalExecution(String(executionId)));
        return handle;
    }

    /**
     * Fetch the latest state of an async single-eval execution by id.
     * Returns an `Execution` with a refresher attached — call
     * `handle.wait()` to block until completion.
     */
    public async getExecution(executionId: string): Promise<Execution> {
        const handle = await this._refreshEvalExecution(executionId);
        handle._setRefresher(() => this._refreshEvalExecution(executionId));
        return handle;
    }

    private async _refreshEvalExecution(executionId: string): Promise<Execution> {
        const raw = (await this.request({
            method: HttpMethod.GET,
            url: `${this.baseUrl}/${Routes.get_eval_result}`,
            params: { eval_id: executionId },
            timeout: (this.defaultTimeout ?? 30) * 1000,
        })) as any;
        const body = raw?.data ?? raw;
        // Envelope: { status, result: { eval_status, result: <dict|str>, error_message? } }
        const payload = body?.result ?? {};
        const status = normalizeStatus(payload.eval_status);
        const rawResult = payload.result;
        let parsedResult: EvalResult | null = null;
        let errorLocalizer: Record<string, any> | null = null;
        let errorMessage: string | null = payload.error_message ?? null;

        if (rawResult && typeof rawResult === 'object') {
            parsedResult = {
                name: rawResult.name ?? '',
                output: rawResult.output ?? rawResult.value,
                reason: rawResult.reason ?? '',
                runtime: rawResult.runtime ?? 0,
                output_type: rawResult.output_type,
                eval_id: rawResult.eval_id ?? executionId,
                model: rawResult.model,
                error_localizer_enabled: rawResult.error_localizer_enabled,
                error_localizer: rawResult.error_localizer ?? null,
            };
            errorLocalizer = rawResult.error_localizer ?? null;
            errorMessage = rawResult.error_message ?? errorMessage;
        }
        // else rawResult is a human-readable string like "Evaluation is being processed."

        return new Execution({
            id: executionId,
            kind: 'eval',
            status,
            result: parsedResult,
            errorMessage,
            errorLocalizer,
        });
    }
}

/**
 * Convenience function to run a single or batch of evaluations.
 * @param evalTemplates - Evaluation name string (e.g., "Factual Accuracy") or list of templates.
 * @param inputs - Single test case or list of test cases as dictionaries.
 * @param options - Optional parameters for the evaluation.
 * @returns BatchRunResult containing evaluation results.
 */
export const evaluate = (
    evalTemplates: string | EvalTemplate | (string | EvalTemplate)[],
    inputs: Record<string, string | string[]>,
    options: EvaluatorConfig & EvaluateOptions = {}
): Promise<BatchRunResult> => {
    const { fiApiKey, fiSecretKey, fiBaseUrl, ...evalOptions } = options;
    return new Evaluator({ fiApiKey, fiSecretKey, fiBaseUrl }).evaluate(evalTemplates, inputs, evalOptions);
};

/**
 * Convenience function to fetch information about all available evaluation templates.
 * @returns A list of evaluation template information dictionaries.
 */
export const list_evaluations = (options: EvaluatorConfig = {}): Promise<Record<string, any>[]> => {
    const { fiApiKey, fiSecretKey, fiBaseUrl } = options;
    return new Evaluator({ fiApiKey, fiSecretKey, fiBaseUrl }).list_evaluations();
};

/**
 * Convenience function to get an evaluation result by ID.
 * @param evalId - The evaluation ID.
 * @param options - Optional Evaluator configuration.
 * @returns The evaluation result.
 */
export const get_eval_result = (
    evalId: string,
    options: EvaluatorConfig = {}
): Promise<Record<string, any>> => {
    const { fiApiKey, fiSecretKey, fiBaseUrl } = options;
    return new Evaluator({ fiApiKey, fiSecretKey, fiBaseUrl }).getEvalResult(evalId);
};

/**
 * Convenience function to evaluate a pipeline.
 * @param projectName - The project name.
 * @param version - The version string.
 * @param evalData - The evaluation data.
 * @param options - Optional Evaluator configuration.
 * @returns The evaluation response.
 */
export const evaluate_pipeline = (
    projectName: string,
    version: string,
    evalData: PipelineEvalData[],
    options: EvaluatorConfig = {}
): Promise<Record<string, any>> => {
    const { fiApiKey, fiSecretKey, fiBaseUrl } = options;
    return new Evaluator({ fiApiKey, fiSecretKey, fiBaseUrl }).evaluatePipeline(projectName, version, evalData);
};

/**
 * Convenience function to get pipeline evaluation results.
 * @param projectName - The project name.
 * @param versions - List of versions.
 * @param options - Optional Evaluator configuration.
 * @returns The pipeline results.
 */
export const get_pipeline_results = (
    projectName: string,
    versions: string[],
    options: EvaluatorConfig = {}
): Promise<Record<string, any>> => {
    const { fiApiKey, fiSecretKey, fiBaseUrl } = options;
    return new Evaluator({ fiApiKey, fiSecretKey, fiBaseUrl }).getPipelineResults(projectName, versions);
};



