import {
    APIKeyAuth,
    ResponseHandler,
    HttpMethod,
    RequestConfig,
    Routes,
    InvalidAuthError,
} from '@future-agi/sdk';
import { AxiosResponse } from 'axios';

import { Execution } from './execution';
import {
    BulkDeleteResponse,
    CompositeCreateResponse,
    CompositeDetailResponse,
    CompositeExecutionResponse,
    GroundTruthConfigResponse,
    GroundTruthDataResponse,
    GroundTruthDeleteResponse,
    GroundTruthListResponse,
    GroundTruthRoleMappingResponse,
    GroundTruthSearchResponse,
    GroundTruthStatusResponse,
    GroundTruthUploadResponse,
    GroundTruthVariableMappingResponse,
    PlaygroundRunResponse,
    TemplateChartsResponse,
    TemplateCreateResponse,
    TemplateDetailResponse,
    TemplateDuplicateResponse,
    TemplateFeedbackListResponse,
    TemplateListResponse,
    TemplateUpdateResponse,
    TemplateUsageResponse,
    VersionCreateResponse,
    VersionListResponse,
    VersionRestoreResponse,
    VersionSetDefaultResponse,
} from './manager-types';

// Lightweight UUIDv4 generator so the package doesn't pull in a new dep.
function uuidv4(): string {
    if (typeof globalThis.crypto !== 'undefined' && typeof globalThis.crypto.randomUUID === 'function') {
        return globalThis.crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === 'x' ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

/**
 * Default response handler: returns the `result` payload from the
 * standard `{ status, result }` envelope used by the revamped backend.
 */
export class JsonResultHandler extends ResponseHandler<any, any> {
    public static _parseSuccess(response: AxiosResponse): any {
        const data = response.data ?? {};
        if (data && typeof data === 'object' && 'result' in data) {
            return (data as Record<string, any>).result;
        }
        return data;
    }

    public static _handleError(response: AxiosResponse): never {
        if (response.status === 403) {
            throw new InvalidAuthError();
        }
        if (response.status === 404) {
            throw new Error(`Not found: ${JSON.stringify(response.data)}`);
        }
        throw new Error(
            `Eval template API error ${response.status}: ${JSON.stringify(response.data)}`
        );
    }
}

export type EvalType = 'llm' | 'code' | 'agent';
/**
 * Output shape of an eval template as used by the create/update API.
 * Distinct from `types.OutputType` (which describes the result *value*
 * shape — score/boolean/json/text).
 */
export type EvalTemplateOutputType =
    | 'pass_fail'
    | 'percentage'
    | 'deterministic';
export type AggregationFunction =
    | 'weighted_avg'
    | 'avg'
    | 'min'
    | 'max'
    | 'pass_rate';
export type CompositeChildAxis =
    | ''
    | 'pass_fail'
    | 'percentage'
    | 'choices'
    | 'code';

export interface ListTemplatesOptions {
    page?: number;
    pageSize?: number;
    search?: string;
    ownerFilter?: 'all' | 'user' | 'system';
    filters?: Record<string, any>;
    sortBy?: 'name' | 'updated_at' | 'created_at';
    sortOrder?: 'asc' | 'desc';
}

export interface CreateTemplateOptions {
    name: string;
    instructions?: string;
    evalType?: EvalType;
    model?: string;
    outputType?: EvalTemplateOutputType;
    passThreshold?: number;
    choiceScores?: Record<string, number>;
    description?: string;
    tags?: string[];
    checkInternet?: boolean;
    code?: string;
    codeLanguage?: 'python' | 'javascript';
    messages?: Array<Record<string, any>>;
    fewShotExamples?: Array<Record<string, any>>;
    mode?: 'auto' | 'agent' | 'quick';
    tools?: Record<string, any>;
    knowledgeBases?: string[];
    dataInjection?: Record<string, any>;
    summary?: Record<string, any>;
    isDraft?: boolean;
}

export interface CreateCompositeOptions {
    name: string;
    childTemplateIds: string[];
    description?: string;
    tags?: string[];
    aggregationEnabled?: boolean;
    aggregationFunction?: AggregationFunction;
    childWeights?: Record<string, number>;
    compositeChildAxis?: CompositeChildAxis;
}

export interface ExecuteCompositeOptions {
    mapping: Record<string, any>;
    model?: string;
    config?: Record<string, any>;
    errorLocalizer?: boolean;
    inputDataTypes?: Record<string, string>;
    spanContext?: Record<string, any>;
    traceContext?: Record<string, any>;
    sessionContext?: Record<string, any>;
}

/**
 * Management client for the revamped eval template API
 * (`/model-hub/eval-templates/*`). Covers list / create / detail /
 * update / delete, versioning, and composite evals.
 */
export class EvalTemplateManager extends APIKeyAuth {
    constructor(
        options: {
            fiApiKey?: string;
            fiSecretKey?: string;
            fiBaseUrl?: string;
            timeout?: number;
        } = {}
    ) {
        const fiApiKey = process.env.FI_API_KEY || options.fiApiKey;
        const fiSecretKey = process.env.FI_SECRET_KEY || options.fiSecretKey;
        const fiBaseUrl = process.env.FI_BASE_URL || options.fiBaseUrl;
        super({ ...options, fiApiKey, fiSecretKey, fiBaseUrl });
    }

    private url(routePath: string, params: Record<string, string> = {}): string {
        let out = routePath;
        for (const [k, v] of Object.entries(params)) {
            out = out.replace(`{${k}}`, v);
        }
        return `${this.baseUrl}/${out}`;
    }

    private async call(
        method: HttpMethod,
        urlPath: string,
        body?: Record<string, any>
    ): Promise<any> {
        const config: RequestConfig = {
            method,
            url: urlPath,
            json: body,
            timeout: (this.defaultTimeout ?? 30) * 1000,
        };
        return (await this.request(config, JsonResultHandler)) as any;
    }

    // ------------------------------------------------------------------
    // Templates
    // ------------------------------------------------------------------

    public async listTemplates(opts: ListTemplatesOptions = {}): Promise<TemplateListResponse> {
        const body: Record<string, any> = {
            page: opts.page ?? 0,
            page_size: opts.pageSize ?? 25,
            owner_filter: opts.ownerFilter ?? 'all',
            sort_by: opts.sortBy ?? 'updated_at',
            sort_order: opts.sortOrder ?? 'desc',
        };
        if (opts.search !== undefined) body.search = opts.search;
        if (opts.filters !== undefined) body.filters = opts.filters;
        return this.call(HttpMethod.POST, this.url(Routes.eval_template_list), body);
    }

    public async getTemplate(templateId: string): Promise<TemplateDetailResponse> {
        return this.call(
            HttpMethod.GET,
            this.url(Routes.eval_template_detail, { template_id: templateId })
        );
    }

    public async createTemplate(opts: CreateTemplateOptions): Promise<TemplateCreateResponse> {
        const body: Record<string, any> = {
            name: opts.name,
            is_draft: opts.isDraft ?? false,
            eval_type: opts.evalType ?? 'llm',
            instructions: opts.instructions ?? '',
            model: opts.model ?? 'turing_large',
            output_type: opts.outputType ?? 'pass_fail',
            pass_threshold: opts.passThreshold ?? 0.5,
            tags: opts.tags ?? [],
            check_internet: opts.checkInternet ?? false,
        };
        if (opts.choiceScores !== undefined) body.choice_scores = opts.choiceScores;
        if (opts.description !== undefined) body.description = opts.description;
        if (opts.code !== undefined) body.code = opts.code;
        if (opts.codeLanguage !== undefined) body.code_language = opts.codeLanguage;
        if (opts.messages !== undefined) body.messages = opts.messages;
        if (opts.fewShotExamples !== undefined)
            body.few_shot_examples = opts.fewShotExamples;
        if (opts.mode !== undefined) body.mode = opts.mode;
        if (opts.tools !== undefined) body.tools = opts.tools;
        if (opts.knowledgeBases !== undefined)
            body.knowledge_bases = opts.knowledgeBases;
        if (opts.dataInjection !== undefined)
            body.data_injection = opts.dataInjection;
        if (opts.summary !== undefined) body.summary = opts.summary;

        return this.call(
            HttpMethod.POST,
            this.url(Routes.eval_template_create_v2),
            body
        );
    }

    public async updateTemplate(
        templateId: string,
        fields: Record<string, any>
    ): Promise<TemplateUpdateResponse> {
        const body: Record<string, any> = {};
        for (const [k, v] of Object.entries(fields)) {
            if (v !== undefined) body[k] = v;
        }
        return this.call(
            HttpMethod.PUT,
            this.url(Routes.eval_template_update_v2, { template_id: templateId }),
            body
        );
    }

    public async deleteTemplate(templateId: string): Promise<any> {
        return this.call(
            HttpMethod.POST,
            this.url(Routes.eval_template_delete),
            { eval_template_id: templateId }
        );
    }

    public async bulkDeleteTemplates(templateIds: string[]): Promise<BulkDeleteResponse> {
        return this.call(
            HttpMethod.POST,
            this.url(Routes.eval_template_bulk_delete),
            { template_ids: templateIds }
        );
    }

    // ------------------------------------------------------------------
    // Versions
    // ------------------------------------------------------------------

    public async listVersions(templateId: string): Promise<VersionListResponse> {
        return this.call(
            HttpMethod.GET,
            this.url(Routes.eval_template_version_list, {
                template_id: templateId,
            })
        );
    }

    public async createVersion(
        templateId: string,
        opts: {
            criteria?: string;
            model?: string;
            configSnapshot?: Record<string, any>;
        } = {}
    ): Promise<VersionCreateResponse> {
        const body: Record<string, any> = {};
        if (opts.criteria !== undefined) body.criteria = opts.criteria;
        if (opts.model !== undefined) body.model = opts.model;
        if (opts.configSnapshot !== undefined)
            body.config_snapshot = opts.configSnapshot;
        return this.call(
            HttpMethod.POST,
            this.url(Routes.eval_template_version_create, {
                template_id: templateId,
            }),
            body
        );
    }

    public async setDefaultVersion(
        templateId: string,
        versionId: string
    ): Promise<VersionSetDefaultResponse> {
        return this.call(
            HttpMethod.PUT,
            this.url(Routes.eval_template_version_set_default, {
                template_id: templateId,
                version_id: versionId,
            })
        );
    }

    public async restoreVersion(
        templateId: string,
        versionId: string
    ): Promise<VersionRestoreResponse> {
        return this.call(
            HttpMethod.POST,
            this.url(Routes.eval_template_version_restore, {
                template_id: templateId,
                version_id: versionId,
            })
        );
    }

    // ------------------------------------------------------------------
    // Composite evals
    // ------------------------------------------------------------------

    public async createComposite(opts: CreateCompositeOptions): Promise<CompositeCreateResponse> {
        const body: Record<string, any> = {
            name: opts.name,
            child_template_ids: opts.childTemplateIds,
            aggregation_enabled: opts.aggregationEnabled ?? true,
            aggregation_function: opts.aggregationFunction ?? 'weighted_avg',
            composite_child_axis: opts.compositeChildAxis ?? '',
            tags: opts.tags ?? [],
        };
        if (opts.description !== undefined) body.description = opts.description;
        if (opts.childWeights !== undefined)
            body.child_weights = opts.childWeights;
        return this.call(
            HttpMethod.POST,
            this.url(Routes.composite_eval_create),
            body
        );
    }

    public async getComposite(templateId: string): Promise<CompositeDetailResponse> {
        return this.call(
            HttpMethod.GET,
            this.url(Routes.composite_eval_detail, { template_id: templateId })
        );
    }

    public async updateComposite(
        templateId: string,
        fields: Record<string, any>
    ): Promise<CompositeDetailResponse> {
        const body: Record<string, any> = {};
        for (const [k, v] of Object.entries(fields)) {
            if (v !== undefined) body[k] = v;
        }
        return this.call(
            HttpMethod.PATCH,
            this.url(Routes.composite_eval_detail, { template_id: templateId }),
            body
        );
    }

    /**
     * Submit a composite eval for non-blocking execution. Returns an
     * `Execution` handle immediately; the actual HTTP call runs in a
     * background Promise inside this process.
     *
     * IMPORTANT: The execution id is client-local — if this process
     * dies or the handle reference is dropped, the in-flight work is
     * lost. For cross-process resumable runs, use
     * `Evaluator.submit` against a single eval.
     */
    public submitComposite(
        templateId: string,
        opts: ExecuteCompositeOptions
    ): Execution {
        const handle = new Execution({
            id: uuidv4(),
            kind: 'composite',
            status: 'pending',
        });

        // Local-thread style: the refresher is a no-op because the
        // background Promise mutates the handle directly. We still set
        // one so `Execution.wait()` keeps polling rather than bailing
        // out early on a null refresher.
        handle._setRefresher(async () => handle);

        // Kick off the background run without awaiting.
        (async () => {
            handle.status = 'processing';
            try {
                const result = await this.executeComposite(templateId, opts);
                handle.result = result;
                if (result && typeof result === 'object') {
                    handle.errorLocalizer = result.error_localizer_results ?? null;
                }
                handle.status = 'completed';
            } catch (err: any) {
                handle.errorMessage = err?.message ?? String(err);
                handle.status = 'failed';
            }
        })();

        return handle;
    }

    public async executeComposite(
        templateId: string,
        opts: ExecuteCompositeOptions
    ): Promise<CompositeExecutionResponse> {
        const body: Record<string, any> = {
            mapping: opts.mapping,
            config: opts.config ?? {},
            error_localizer: opts.errorLocalizer ?? false,
            input_data_types: opts.inputDataTypes ?? {},
        };
        if (opts.model !== undefined) body.model = opts.model;
        if (opts.spanContext !== undefined) body.span_context = opts.spanContext;
        if (opts.traceContext !== undefined)
            body.trace_context = opts.traceContext;
        if (opts.sessionContext !== undefined)
            body.session_context = opts.sessionContext;
        return this.call(
            HttpMethod.POST,
            this.url(Routes.composite_eval_execute, { template_id: templateId }),
            body
        );
    }

    // ------------------------------------------------------------------
    // Ground Truth (Phase 9)
    // ------------------------------------------------------------------

    public async listGroundTruth(templateId: string): Promise<GroundTruthListResponse> {
        return this.call(
            HttpMethod.GET,
            this.url(Routes.ground_truth_list, { template_id: templateId })
        );
    }

    public async uploadGroundTruth(
        templateId: string,
        opts: {
            name: string;
            columns: string[];
            data: Array<Record<string, any>>;
            description?: string;
            fileName?: string;
            variableMapping?: Record<string, string>;
            roleMapping?: Record<string, string>;
        }
    ): Promise<GroundTruthUploadResponse> {
        const body: Record<string, any> = {
            name: opts.name,
            description: opts.description ?? '',
            file_name: opts.fileName ?? '',
            columns: opts.columns,
            data: opts.data,
        };
        if (opts.variableMapping !== undefined)
            body.variable_mapping = opts.variableMapping;
        if (opts.roleMapping !== undefined) body.role_mapping = opts.roleMapping;

        return this.call(
            HttpMethod.POST,
            this.url(Routes.ground_truth_upload, { template_id: templateId }),
            body
        );
    }

    public async getGroundTruthConfig(templateId: string): Promise<GroundTruthConfigResponse> {
        return this.call(
            HttpMethod.GET,
            this.url(Routes.ground_truth_config, { template_id: templateId })
        );
    }

    public async setGroundTruthConfig(
        templateId: string,
        opts: {
            enabled?: boolean;
            groundTruthId?: string | null;
            mode?: 'auto' | 'manual' | 'disabled';
            maxExamples?: number;
            similarityThreshold?: number;
            injectionFormat?: 'structured' | 'conversational' | 'xml';
        } = {}
    ): Promise<GroundTruthConfigResponse> {
        const body = {
            enabled: opts.enabled ?? true,
            ground_truth_id: opts.groundTruthId ?? null,
            mode: opts.mode ?? 'auto',
            max_examples: opts.maxExamples ?? 3,
            similarity_threshold: opts.similarityThreshold ?? 0.7,
            injection_format: opts.injectionFormat ?? 'structured',
        };
        return this.call(
            HttpMethod.PUT,
            this.url(Routes.ground_truth_config, { template_id: templateId }),
            body
        );
    }

    public async setGroundTruthVariableMapping(
        groundTruthId: string,
        variableMapping: Record<string, string>
    ): Promise<GroundTruthVariableMappingResponse> {
        return this.call(
            HttpMethod.PUT,
            this.url(Routes.ground_truth_mapping, {
                ground_truth_id: groundTruthId,
            }),
            { variable_mapping: variableMapping }
        );
    }

    public async setGroundTruthRoleMapping(
        groundTruthId: string,
        roleMapping: Record<string, string>
    ): Promise<GroundTruthRoleMappingResponse> {
        return this.call(
            HttpMethod.PUT,
            this.url(Routes.ground_truth_role_mapping, {
                ground_truth_id: groundTruthId,
            }),
            { role_mapping: roleMapping }
        );
    }

    public async getGroundTruthData(
        groundTruthId: string,
        opts: { page?: number; pageSize?: number } = {}
    ): Promise<GroundTruthDataResponse> {
        const config: RequestConfig = {
            method: HttpMethod.GET,
            url: this.url(Routes.ground_truth_data, {
                ground_truth_id: groundTruthId,
            }),
            params: {
                page: opts.page ?? 1,
                page_size: opts.pageSize ?? 50,
            },
            timeout: (this.defaultTimeout ?? 30) * 1000,
        };
        return (await this.request(config, JsonResultHandler)) as any;
    }

    public async getGroundTruthStatus(groundTruthId: string): Promise<GroundTruthStatusResponse> {
        return this.call(
            HttpMethod.GET,
            this.url(Routes.ground_truth_status, {
                ground_truth_id: groundTruthId,
            })
        );
    }

    public async triggerGroundTruthEmbeddings(
        groundTruthId: string
    ): Promise<any> {
        return this.call(
            HttpMethod.POST,
            this.url(Routes.ground_truth_embed, {
                ground_truth_id: groundTruthId,
            })
        );
    }

    public async searchGroundTruth(
        groundTruthId: string,
        opts: { query: string; maxResults?: number }
    ): Promise<GroundTruthSearchResponse> {
        return this.call(
            HttpMethod.POST,
            this.url(Routes.ground_truth_search, {
                ground_truth_id: groundTruthId,
            }),
            { query: opts.query, max_results: opts.maxResults ?? 3 }
        );
    }

    public async deleteGroundTruth(groundTruthId: string): Promise<GroundTruthDeleteResponse> {
        return this.call(
            HttpMethod.DELETE,
            this.url(Routes.ground_truth_delete, {
                ground_truth_id: groundTruthId,
            })
        );
    }

    // ------------------------------------------------------------------
    // Usage, Feedback, and 30-day charts (Phase 10)
    // ------------------------------------------------------------------

    public async getTemplateUsage(
        templateId: string,
        opts: {
            page?: number;
            pageSize?: number;
            period?: '30m' | '6h' | '1d' | '7d' | '30d' | '90d';
            version?: string;
        } = {}
    ): Promise<TemplateUsageResponse> {
        const params: Record<string, any> = {
            page: opts.page ?? 0,
            page_size: opts.pageSize ?? 25,
            period: opts.period ?? '30d',
        };
        if (opts.version !== undefined) params.version = opts.version;
        const config: RequestConfig = {
            method: HttpMethod.GET,
            url: this.url(Routes.eval_template_usage, {
                template_id: templateId,
            }),
            params,
            timeout: (this.defaultTimeout ?? 30) * 1000,
        };
        return (await this.request(config, JsonResultHandler)) as any;
    }

    public async listTemplateFeedback(
        templateId: string,
        opts: { page?: number; pageSize?: number } = {}
    ): Promise<TemplateFeedbackListResponse> {
        const config: RequestConfig = {
            method: HttpMethod.GET,
            url: this.url(Routes.eval_template_feedback_list, {
                template_id: templateId,
            }),
            params: {
                page: opts.page ?? 0,
                page_size: opts.pageSize ?? 25,
            },
            timeout: (this.defaultTimeout ?? 30) * 1000,
        };
        return (await this.request(config, JsonResultHandler)) as any;
    }

    public async getTemplateCharts(templateIds: string[]): Promise<TemplateChartsResponse> {
        return this.call(
            HttpMethod.POST,
            this.url(Routes.eval_template_list_charts),
            { template_ids: templateIds }
        );
    }

    // ------------------------------------------------------------------
    // Duplicate + playground
    // ------------------------------------------------------------------

    /**
     * Clone an existing user-owned eval template under a new name.
     */
    public async duplicateTemplate(
        templateId: string,
        name: string
    ): Promise<TemplateDuplicateResponse> {
        return this.call(
            HttpMethod.POST,
            this.url(Routes.eval_template_duplicate),
            { eval_template_id: templateId, name }
        );
    }

    /**
     * Run a saved eval template via the playground endpoint. Identifies
     * by template_id (not name) and supports auto-context injection
     * from row / span / trace / session / call — either as resolved
     * dicts or as ids the server looks up.
     */
    public async runPlayground(
        templateId: string,
        opts: {
            mapping: Record<string, any>;
            model?: string;
            config?: Record<string, any>;
            inputDataTypes?: Record<string, string>;
            errorLocalizer?: boolean;
            kbId?: string;
            rowContext?: Record<string, any>;
            spanContext?: Record<string, any>;
            traceContext?: Record<string, any>;
            sessionContext?: Record<string, any>;
            callContext?: Record<string, any>;
            spanId?: string;
            traceId?: string;
            sessionId?: string;
            callId?: string;
        }
    ): Promise<PlaygroundRunResponse> {
        const body: Record<string, any> = {
            template_id: templateId,
            mapping: opts.mapping,
            error_localizer: opts.errorLocalizer ?? false,
        };
        if (opts.model !== undefined) body.model = opts.model;
        if (opts.config !== undefined) body.config = opts.config;
        if (opts.inputDataTypes !== undefined)
            body.input_data_types = opts.inputDataTypes;
        if (opts.kbId !== undefined) body.kb_id = opts.kbId;
        if (opts.rowContext !== undefined) body.row_context = opts.rowContext;
        if (opts.spanContext !== undefined)
            body.span_context = opts.spanContext;
        if (opts.traceContext !== undefined)
            body.trace_context = opts.traceContext;
        if (opts.sessionContext !== undefined)
            body.session_context = opts.sessionContext;
        if (opts.callContext !== undefined)
            body.call_context = opts.callContext;
        if (opts.spanId !== undefined) body.span_id = opts.spanId;
        if (opts.traceId !== undefined) body.trace_id = opts.traceId;
        if (opts.sessionId !== undefined) body.session_id = opts.sessionId;
        if (opts.callId !== undefined) body.call_id = opts.callId;

        return this.call(
            HttpMethod.POST,
            this.url(Routes.eval_playground),
            body
        );
    }
}
