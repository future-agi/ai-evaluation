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
    AUTH_ENVVAR_NAME
} from '@future-agi/sdk';
import { AxiosResponse } from 'axios';

import { EvalTemplate } from './templates';
import { BatchRunResult, EvalResult, EvalResultMetric } from './types';


/**
 * Handles responses for evaluation requests
 */
export class EvalResponseHandler extends ResponseHandler<BatchRunResult, any> {
    public static _parseSuccess(response: AxiosResponse): BatchRunResult {
        const data = response.data || {};
        const evalResults: (EvalResult | null)[] = [];

        if (Array.isArray(data.result)) {
            for (const result of data.result) {
                if (result && Array.isArray(result.evaluations)) {
                    for (const evaluation of result.evaluations) {
                        const newMetadata: Record<string, any> = {};
                        if (evaluation?.metadata) {
                            let metadata: any = evaluation.metadata;
                            if (typeof metadata === "string") {
                                try {
                                    metadata = JSON.parse(metadata);
                                } catch { /* ignore parse errors */ }
                            }
                            if (metadata && typeof metadata === "object") {
                                newMetadata["usage"] = metadata.usage ?? {};
                                newMetadata["cost"] = metadata.cost ?? {};
                                newMetadata["explanation"] = metadata.explanation ?? {};
                            }
                        }

                        evalResults.push({
                            data: evaluation?.data,
                            failure: evaluation?.failure,
                            reason: evaluation?.reason ?? "",
                            runtime: evaluation?.runtime ?? 0,
                            metadata: newMetadata,
                            metrics: Array.isArray(evaluation?.metrics)
                                ? evaluation.metrics.map((m: any): EvalResultMetric => ({
                                      id: m.id,
                                      value: m.value,
                                  }))
                                : [],
                        });
                    }
                }
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
 */
export class Evaluator extends APIKeyAuth {
    private readonly maxWorkers: number;
    private evalInfoCache = new Map<string, Record<string, any>>();

    constructor(
        options: {
            fiApiKey?: string;
            fiSecretKey?: string;
            fiBaseUrl?: string;
            timeout?: number;
            maxQueue?: number;
            maxWorkers?: number;
        } = {}
    ) {
        const fiApiKey = process.env.FI_API_KEY || options.fiApiKey;
        const fiSecretKey = process.env.FI_SECRET_KEY || options.fiSecretKey;
        const fiBaseUrl = process.env.FI_BASE_URL || options.fiBaseUrl;

        super({ ...options, fiApiKey, fiSecretKey, fiBaseUrl });
        this.maxWorkers = options.maxWorkers || 8;
    }

    public async evaluate(
        evalTemplates: string | EvalTemplate | (string | EvalTemplate)[],
        inputs: Record<string, string | string[]>,
        options: {
            timeout?: number;
            modelName: string; // mandatory
            customEvalName?: string;
            traceEval?: boolean;
        }
    ): Promise<BatchRunResult> {

        const { timeout, modelName, customEvalName } = options;

        if (!modelName || modelName.trim() === "") {
            throw new TypeError("'modelName' is a required option and must be a non-empty string.");
        }

        let traceEval = options.traceEval || false;
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

        const transformedApiInputs: Record<string, string[]> = {};

        if (Array.isArray(inputs)) {
            // Explicitly disallow array-of-dicts per spec
            throw new TypeError("'inputs' must be a dictionary, array-of-dicts is not supported.");
        }

        for (const [key, value] of Object.entries(inputs)) {
            if (Array.isArray(value)) {
                if (!value.every(v => typeof v === "string")) {
                    throw new TypeError(`All values in array for key '${key}' must be strings.`);
                }
                transformedApiInputs[key] = value;
            } else if (typeof value === "string") {
                transformedApiInputs[key] = [value];
            } else {
                throw new TypeError(`Invalid input type for key '${key}'. Expected string or string[].`);
            }
        }

        const finalApiPayload = {
            eval_name: evalName,
            inputs: transformedApiInputs,
            model: modelName,
            span_id: spanId,
            custom_eval_name: customEvalName,
            trace_eval: traceEval,
        };

        // Convert timeout (seconds) to milliseconds for axios. Use a higher default (200s) if not provided.
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
            console.error("Evaluation failed:", error);
            throw error;
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
    options: {
        fiApiKey?: string,
        fiSecretKey?: string,
        fiBaseUrl?: string,
        timeout?: number;
        modelName: string;
        customEvalName?: string;
        traceEval?: boolean;
    }
): Promise<BatchRunResult> => {
    const { fiApiKey, fiSecretKey, fiBaseUrl, ...restOptions } = options;
    return new Evaluator({ fiApiKey, fiSecretKey, fiBaseUrl }).evaluate(evalTemplates, inputs, restOptions);
};

/**
 * Convenience function to fetch information about all available evaluation templates.
 * @returns A list of evaluation template information dictionaries.
 */
export const list_evaluations = (options: {
    fiApiKey?: string,
    fiSecretKey?: string,
    fiBaseUrl?: string,
} = {}): Promise<Record<string, any>[]> => {
    const { fiApiKey, fiSecretKey, fiBaseUrl } = options;
    return new Evaluator({ fiApiKey, fiSecretKey, fiBaseUrl }).list_evaluations();
};



