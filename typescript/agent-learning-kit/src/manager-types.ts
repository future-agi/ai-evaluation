/**
 * Typed response interfaces for {@link EvalTemplateManager}.
 *
 * All interfaces use snake_case field names to match the backend JSON
 * wire format. Both `result.id` and `result["id"]` work natively in
 * TypeScript (no special base class needed).
 */

// ---------------------------------------------------------------------------
// Shared sub-types
// ---------------------------------------------------------------------------

export interface TemplateListItem {
    id: string;
    name: string;
    template_type?: string;
    eval_type?: string;
    output_type?: string;
    owner?: string;
    created_by_name?: string;
    version_count?: number;
    current_version?: string;
    last_updated?: string;
    thirty_day_chart?: any[];
    thirty_day_error_rate?: any[];
    thirty_day_run_count?: number;
    tags?: string[];
}

export interface VersionItem {
    id: string;
    version_number: number;
    is_default: boolean;
    criteria?: string;
    model?: string;
    config_snapshot?: Record<string, any>;
    created_by_name?: string;
    created_at?: string;
}

export interface CompositeChildSummary {
    child_id: string;
    child_name: string;
    order: number;
    eval_type?: string;
    pinned_version_id?: string | null;
    pinned_version_number?: number | null;
    weight: number;
    required_keys?: string[];
}

export interface CompositeChildResult {
    child_id: string;
    child_name: string;
    order: number;
    score?: any;
    output?: any;
    reason?: string;
    output_type?: string;
    status: string;
    error?: string | null;
    log_id?: string | null;
    weight: number;
    error_localizer_result?: Record<string, any> | null;
}

export interface GroundTruthItemResponse {
    id: string;
    name: string;
    description?: string;
    file_name?: string;
    columns: string[];
    row_count: number;
    variable_mapping?: Record<string, string> | null;
    role_mapping?: Record<string, string> | null;
    embedding_status: string;
    embedded_row_count?: number;
    storage_type?: string;
    created_at?: string;
}

export interface GroundTruthConfigData {
    enabled: boolean;
    ground_truth_id?: string | null;
    mode?: string;
    max_examples?: number;
    similarity_threshold?: number;
    injection_format?: string;
}

// ---------------------------------------------------------------------------
// Template CRUD
// ---------------------------------------------------------------------------

export interface TemplateCreateResponse {
    id: string;
    name: string;
    version: string;
}

export interface TemplateDetailResponse {
    id: string;
    name: string;
    description?: string;
    template_type?: string;
    eval_type?: string;
    instructions?: string;
    model?: string;
    output_type?: string;
    pass_threshold?: number;
    choice_scores?: Record<string, number> | null;
    choices?: any;
    multi_choice?: boolean;
    code?: string | null;
    code_language?: string | null;
    required_keys?: string[];
    owner?: string;
    created_by_name?: string;
    version_count?: number;
    current_version?: string;
    tags?: string[];
    check_internet?: boolean;
    error_localizer_enabled?: boolean;
    aggregation_enabled?: boolean;
    aggregation_function?: string;
    composite_child_axis?: string;
    config?: Record<string, any> | null;
    created_at?: string;
    updated_at?: string;
}

export interface TemplateUpdateResponse {
    id: string;
    name?: string;
    updated: boolean;
}

export interface TemplateDuplicateResponse {
    message: string;
    eval_template_id: string;
}

export interface TemplateListResponse {
    items: TemplateListItem[];
    total: number;
    page: number;
    page_size: number;
}

export interface BulkDeleteResponse {
    deleted_count: number;
}

// ---------------------------------------------------------------------------
// Versioning
// ---------------------------------------------------------------------------

export interface VersionListResponse {
    template_id: string;
    versions: VersionItem[];
    total: number;
}

export interface VersionCreateResponse {
    id: string;
    version_number: number;
    is_default: boolean;
}

export interface VersionSetDefaultResponse {
    id: string;
    version_number: number;
    is_default: boolean;
}

export interface VersionRestoreResponse {
    id: string;
    version_number: number;
    is_default: boolean;
    restored_from?: number;
}

// ---------------------------------------------------------------------------
// Composite
// ---------------------------------------------------------------------------

export interface CompositeCreateResponse {
    id: string;
    name: string;
    template_type: string;
    aggregation_enabled: boolean;
    aggregation_function?: string;
    composite_child_axis?: string;
    children: CompositeChildSummary[];
}

export interface CompositeDetailResponse extends CompositeCreateResponse {
    description?: string;
    tags?: string[];
    created_at?: string;
    updated_at?: string;
}

export interface CompositeExecutionResponse {
    composite_id: string;
    composite_name?: string;
    aggregation_enabled: boolean;
    aggregation_function?: string;
    aggregate_score?: number | null;
    aggregate_pass?: boolean | null;
    children: CompositeChildResult[];
    summary?: any;
    error_localizer_results?: Record<string, any> | null;
    total_children: number;
    completed_children: number;
    failed_children: number;
}

// ---------------------------------------------------------------------------
// Ground Truth
// ---------------------------------------------------------------------------

export interface GroundTruthListResponse {
    template_id: string;
    items: GroundTruthItemResponse[];
    total: number;
}

export interface GroundTruthUploadResponse {
    id: string;
    name: string;
    row_count: number;
    columns: string[];
    embedding_status: string;
}

export interface GroundTruthDataResponse {
    id: string;
    page: number;
    page_size: number;
    total_rows: number;
    total_pages: number;
    columns: string[];
    rows: Array<Record<string, any>>;
}

export interface GroundTruthStatusResponse {
    id: string;
    embedding_status: string;
    embedded_row_count?: number;
    total_rows: number;
    progress_percent?: number;
}

export interface GroundTruthVariableMappingResponse {
    id: string;
    variable_mapping: Record<string, string>;
}

export interface GroundTruthRoleMappingResponse {
    id: string;
    role_mapping: Record<string, string>;
    embedding_status?: string;
}

export interface GroundTruthConfigResponse {
    ground_truth: GroundTruthConfigData;
}

export interface GroundTruthSearchResponse {
    query: string;
    results: Array<Record<string, any>>;
    total: number;
}

export interface GroundTruthDeleteResponse {
    deleted: boolean;
    id: string;
}

// ---------------------------------------------------------------------------
// Usage / Feedback / Charts
// ---------------------------------------------------------------------------

export interface TemplateUsageResponse {
    template_id: string;
    stats: Record<string, any>;
    chart: any[];
    logs?: any;
}

export interface TemplateFeedbackListResponse {
    template_id: string;
    items: Array<Record<string, any>>;
    total: number;
    page: number;
    page_size: number;
}

export interface TemplateChartsResponse {
    charts: Record<string, any>;
}

// ---------------------------------------------------------------------------
// Playground
// ---------------------------------------------------------------------------

export interface PlaygroundRunResponse {
    output?: any;
    reason?: string;
    model?: string | null;
    metadata?: Record<string, any> | null;
    output_type?: string;
    log_id?: string;
}
