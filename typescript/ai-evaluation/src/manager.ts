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
export type OutputType = 'pass_fail' | 'percentage' | 'deterministic';
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
    outputType?: OutputType;
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

    public async listTemplates(opts: ListTemplatesOptions = {}): Promise<any> {
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

    public async getTemplate(templateId: string): Promise<any> {
        return this.call(
            HttpMethod.GET,
            this.url(Routes.eval_template_detail, { template_id: templateId })
        );
    }

    public async createTemplate(opts: CreateTemplateOptions): Promise<any> {
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
    ): Promise<any> {
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

    public async bulkDeleteTemplates(templateIds: string[]): Promise<any> {
        return this.call(
            HttpMethod.POST,
            this.url(Routes.eval_template_bulk_delete),
            { template_ids: templateIds }
        );
    }

    // ------------------------------------------------------------------
    // Versions
    // ------------------------------------------------------------------

    public async listVersions(templateId: string): Promise<any> {
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
    ): Promise<any> {
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
    ): Promise<any> {
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
    ): Promise<any> {
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

    public async createComposite(opts: CreateCompositeOptions): Promise<any> {
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

    public async getComposite(templateId: string): Promise<any> {
        return this.call(
            HttpMethod.GET,
            this.url(Routes.composite_eval_detail, { template_id: templateId })
        );
    }

    public async updateComposite(
        templateId: string,
        fields: Record<string, any>
    ): Promise<any> {
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
    ): Promise<any> {
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
}
